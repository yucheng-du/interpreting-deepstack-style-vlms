"""Training-arm sampling for the v19 compression-aware LoRA pilot (pure Python; no torch, no model).

Arms
  sft      every sample gets C|C (full input)
  generic  k = s*|k|, s = +1/-1 with probability 1/2 from its own hash stream (sign-symmetric)
  role     k = role*|k|   (main-type role=+1: k>0 keeps main finer; later-type role=-1: k<0 keeps later finer)
  anti     k = -role*|k|  (mirror of role)
Per sample index i the three compression arms share the byte budget, |k| and the full-input draw; every draw
is sha256(seed|i|stream), so it depends only on (seed, i). Given k, the config is the offset-axis name
'q<q>|k<k>' with the finest main QP q in q_grid whose bytes on that image's code-length table fit the budget
(levels.config_bytes). Same codec, same budget distribution; only the main/later split differs.

Budget (config 'budget'):
  mode ref_config  budget = bytes of the image at a reference config drawn from train_refs (e.g. N axis 'q4|k0')
  mode raw_frac    budget = frac * raw BF16 bytes of the image's active streams, frac drawn from train_fracs
eval_refs / eval_fracs are recorded only: evaluation runs fixed runner configs on real code lengths.

Rate tables: rate_audit bytes.json ({qps, per_unit: {unit: [{stream, qp, bytes}]}}; qp 'raw' = 2*numel) or the
per-unit JSONL of train/rate_table.py. Budget matching uses whatever table it is given (real or proxy).

CLI
  build  --config C --manifest M --rate-table T --arms sft generic role anti --out DIR   (DIR/<arm>/schedule.jsonl)
  qps    --config C [--offsets O]                                                       (per-key QP sets, JSON)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from cloud_prep.measurement_v19 import levels

ARMS = ('sft', 'generic', 'role', 'anti')
COMPRESSION_ARMS = ('generic', 'role', 'anti')
TRAIN_DIR = Path(__file__).resolve().parent
OFFSETS_DIR = TRAIN_DIR.parent / 'offsets'


# ---- config ------------------------------------------------------------------------------------

def default_config(model):
    return TRAIN_DIR / 'configs' / f'{model}.json'


def set_dotted(cfg, key, value):
    d = cfg
    parts = key.split('.')
    for p in parts[:-1]:
        d = d.setdefault(p, {})
    d[parts[-1]] = value


def parse_set(items):
    """['lr=5e-5', 'budget.train_refs=["q2|k0"]'] -> {key: value}; values are JSON, else plain strings."""
    out = {}
    for it in items or []:
        if '=' not in it:
            raise ValueError(f'--set needs KEY=VALUE, got {it!r}')
        k, v = it.split('=', 1)
        try:
            out[k] = json.loads(v)
        except json.JSONDecodeError:
            out[k] = v
    return out


def load_config(path, overrides=None):
    cfg = json.loads(Path(path).read_text())
    for k, v in (overrides or {}).items():
        set_dotted(cfg, k, v)
    check_config(cfg)
    return cfg


def check_config(cfg):
    if cfg.get('role') not in (1, -1):
        raise ValueError('config role must be +1 (main-type) or -1 (later-type)')
    b = cfg['budget']
    if b['mode'] not in ('ref_config', 'raw_frac'):
        raise ValueError('budget.mode must be ref_config or raw_frac')
    vals = b['train_refs'] if b['mode'] == 'ref_config' else b['train_fracs']
    if not vals:
        raise ValueError('empty training budget set')
    for name in b.get('train_refs') or []:
        levels.parse_config(name)
    for w, v in ((b.get('train_weights'), vals), (cfg.get('k_weights'), cfg['k_abs'])):
        if w is not None and (len(w) != len(v) or min(w) < 0 or sum(w) <= 0):
            raise ValueError('weights must match their values and be non-negative')
    if any(int(k) < 0 for k in cfg['k_abs']):
        raise ValueError('k_abs holds magnitudes (>= 0)')
    if not 0 <= cfg.get('full_mix', 0) <= 1:
        raise ValueError('full_mix must be in [0, 1]')
    q_values(cfg)
    return cfg


def config_sha256(cfg):
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()


def q_values(cfg):
    g = cfg['q_grid']
    qs = list(range(int(g['min']), int(g['max']) + 1, int(g.get('step', 1)))) if isinstance(g, dict) else [int(q) for q in g]
    if not qs:
        raise ValueError('empty q_grid')
    return sorted(set(qs))


def load_offsets(path_or_model):
    p = Path(path_or_model)
    if not p.suffix:
        p = OFFSETS_DIR / f'{path_or_model}.json'
    return json.loads(p.read_text()), p


# ---- deterministic draws -----------------------------------------------------------------------

def uniform(seed, i, stream):
    h = hashlib.sha256(f'{seed}|{i}|{stream}'.encode()).digest()
    return int.from_bytes(h[:8], 'big') / 2.0 ** 64


def pick(values, weights, u):
    w = weights or [1] * len(values)
    t = u * sum(w)
    acc = 0.0
    for v, x in zip(values, w):
        acc += x
        if t < acc:
            return v
    return values[-1]


def draws(seed, i, cfg):
    """Arm-independent draws of sample i: budget reference, |k|, generic sign, full-input flag."""
    b = cfg['budget']
    vals = b['train_refs'] if b['mode'] == 'ref_config' else b['train_fracs']
    return dict(ref=pick(vals, b.get('train_weights'), uniform(seed, i, 'budget')),
                kabs=int(pick(cfg['k_abs'], cfg.get('k_weights'), uniform(seed, i, 'kabs'))),
                sign=1 if uniform(seed, i, 'sign') < 0.5 else -1,
                full=uniform(seed, i, 'full') < cfg.get('full_mix', 0.0))


def arm_k(arm, kabs, sign, role):
    if arm == 'generic':
        return sign * kabs
    if arm == 'role':
        return role * kabs
    if arm == 'anti':
        return -role * kabs
    raise ValueError('k is defined for compression arms only: ' + repr(arm))


# ---- sample order ------------------------------------------------------------------------------

def sample_order(rows, n, seed, mode='manifest'):
    """Indices of the first n training rows: manifest order, or sha256(seed|id) order ('shuffle')."""
    idx = list(range(len(rows)))
    if mode == 'shuffle':
        idx.sort(key=lambda j: hashlib.sha256(f"{seed}|{rows[j]['id']}".encode()).hexdigest())
    elif mode != 'manifest':
        raise ValueError('order must be manifest or shuffle')
    if n is not None and n > len(idx):
        raise ValueError(f'n_samples {n} > {len(idx)} manifest rows')
    return idx[:n] if n is not None else idx


def ids_sha256(ids):
    return hashlib.sha256('\n'.join(str(x) for x in ids).encode()).hexdigest()


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


# ---- rate tables and the solver ----------------------------------------------------------------

def _qp(v):
    return v if v == 'raw' else int(v)


def load_rate_table(path):
    """{kind, path, per_unit: {unit: {(stream, qp|'raw'): bytes}}} from bytes.json or rate_table JSONL."""
    path = Path(path)
    per = {}
    meta = {}
    if path.suffix == '.jsonl':
        with open(path) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    per[r['unit']] = {(x['stream'], _qp(x['qp'])): x['bytes'] for x in r['records']}
                    meta.setdefault('kind', r.get('kind'))
    else:
        bt = json.loads(path.read_text())
        meta = {k: v for k, v in bt.items() if k != 'per_unit'}
        per = {u: {(x['stream'], _qp(x['qp'])): x['bytes'] for x in recs} for u, recs in bt['per_unit'].items()}
    return dict(kind=meta.get('kind') or 'nncodec', path=str(path), meta=meta, per_unit=per)


def unit_keys(unit_table):
    return sorted({k for k, _ in unit_table})


def name_bytes(unit_table, name, keys, main_key, offsets):
    """{'main', 'later', 'total'} or None when the table lacks a QP the config needs."""
    inactive = {k for k in keys if (k, 'raw') not in unit_table}
    if main_key in inactive:
        raise ValueError(f'Rate table entry has no {main_key} stream')
    try:
        b = levels.config_bytes(unit_table, main_key, name, None, keys, inactive=inactive, offsets=offsets)
    except ValueError as exc:
        if 'Byte table lacks' in str(exc):
            return None
        raise
    return dict(main=b['main'], later=b['later'], total=b['main'] + b['later'])


def raw_bytes(unit_table):
    return sum(v for (k, q), v in unit_table.items() if q == 'raw')


def budget_of(unit_table, ref, cfg, keys, main_key, offsets):
    if cfg['budget']['mode'] == 'raw_frac':
        return float(ref) * raw_bytes(unit_table)
    b = name_bytes(unit_table, ref, keys, main_key, offsets)
    if b is None:
        raise ValueError(f'Rate table lacks the QPs of budget reference {ref}')
    return b['total']


def solve_q(unit_table, k, budget, qs, keys, main_key, offsets, strict=True):
    """Finest q (smallest QP) in qs with bytes(q|k) <= budget. Infeasible: the smallest-byte candidate, flagged.
    strict (training schedules): a grid q scanned before the answer whose QPs the table lacks raises (no silent
    move to a coarser q, which would unmatch the arms); code schedule.required_qps for the whole q_grid.
    strict=False skips such q (diagnostics on sparse tables, rate_table validate)."""
    cands, missing = [], []
    for q in sorted(qs):
        name = levels.offset_name(q, k)
        b = name_bytes(unit_table, name, keys, main_key, offsets)
        if b is None:
            missing.append(name)
            continue
        if b['total'] <= budget:
            if missing and strict:
                raise ValueError(f'Rate table lacks QPs of {missing}, finer than the chosen {name}')
            return dict(q=q, name=name, bytes=b, feasible=True)
        cands.append((b['total'], q, name, b))
    if not cands:
        raise ValueError(f'No q in the grid has all QPs in the rate table (k={k})')
    if missing and strict:
        raise ValueError(f'Rate table lacks QPs of {missing}; no q in the grid fits the budget')
    _, q, name, b = min(cands)
    return dict(q=q, name=name, bytes=b, feasible=False)


def plan_sample(i, row, arm, cfg, table, offsets):
    """One schedule record (what the trainer applies and logs) for sample i."""
    keys = [offsets['main_key']] + list(offsets['later_keys'])
    main_key = offsets['main_key']
    dr = draws(cfg['seed'], i, cfg)
    unit = row['unit_id']
    rec = dict(i=i, id=row['id'], unit_id=unit, arm=arm, kabs=dr['kabs'], sign=dr['sign'], full=dr['full'],
               budget_ref=dr['ref'], budget=None, config='C|C', q=None, k=None, feasible=None,
               bytes_main=None, bytes_later=None, bytes_total=None, raw_bytes=None)
    if table is None:
        if arm != 'sft':
            raise ValueError('Compression arms need a rate table')
        return rec
    ut = table['per_unit'].get(unit)
    if ut is None:
        raise ValueError(f'Rate table has no entry for unit {unit}')
    rec['raw_bytes'] = raw_bytes(ut)
    rec['budget'] = budget_of(ut, dr['ref'], cfg, keys, main_key, offsets)
    if arm == 'sft' or dr['full']:
        b = name_bytes(ut, 'C|C', keys, main_key, offsets)
        rec.update(bytes_main=b['main'], bytes_later=b['later'], bytes_total=b['total'])
        return rec
    k = arm_k(arm, dr['kabs'], dr['sign'], cfg['role'])
    s = solve_q(ut, k, rec['budget'], q_values(cfg), keys, main_key, offsets)
    rec.update(config=s['name'], q=s['q'], k=k, feasible=s['feasible'], bytes_main=s['bytes']['main'],
               bytes_later=s['bytes']['later'], bytes_total=s['bytes']['total'])
    return rec


def build_schedule(rows, order, arm, cfg, table, offsets):
    if arm not in ARMS:
        raise ValueError('arm must be one of ' + ', '.join(ARMS))
    missing = sorted({rows[j]['unit_id'] for j in order} - set(table['per_unit'])) if table else []
    if missing:
        raise ValueError(f'Rate table lacks {len(missing)} training units, e.g. {missing[:3]}')
    return [plan_sample(i, rows[j], arm, cfg, table, offsets) for i, j in enumerate(order)]


def required_qps(cfg, keys, main_key, offsets):
    """{key: sorted QPs} any arm can need: budget references, q_grid x {+/-|k|} on the offset axis."""
    names = set(cfg['budget'].get('train_refs') or []) if cfg['budget']['mode'] == 'ref_config' else set()
    ks = {s * int(k) for k in cfg['k_abs'] for s in (1, -1)}
    names |= {levels.offset_name(q, k) for q in q_values(cfg) for k in ks}
    need = {k: set() for k in keys}
    for n in names:
        for key, lv in levels.resolve_config(n, keys, main_key, offsets).items():
            kind, qp = levels.parse_level(lv)
            if kind == 'q':
                need[key].add(qp)
    return {k: sorted(v) for k, v in need.items()}


def summary(records):
    """Compact per-arm description of a schedule (counts, byte means, q/k histograms)."""
    import statistics as st
    comp = [r for r in records if r['k'] is not None]
    hist = lambda key: {str(v): sum(1 for r in comp if r[key] == v) for v in sorted({r[key] for r in comp})}
    tot = [r['bytes_total'] for r in records if r['bytes_total'] is not None]
    return dict(n=len(records), n_compressed=len(comp), n_full=len(records) - len(comp),
                infeasible=sum(1 for r in comp if r['feasible'] is False),
                bytes_total_mean=st.mean(tot) if tot else None,
                budget_mean=st.mean([r['budget'] for r in records if r['budget'] is not None]) if tot else None,
                k_hist=hist('k'), q_hist=hist('q'), ids_sha256=ids_sha256([r['id'] for r in records]))


def write_jsonl(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with open(tmp, 'w') as f:
        for r in records:
            f.write(json.dumps(r) + '\n')
    tmp.replace(path)
    return file_sha256(path)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    b = sub.add_parser('build', help='write DIR/<arm>/schedule.jsonl for each arm (no model)')
    b.add_argument('--config', type=Path, required=True)
    b.add_argument('--set', action='append', default=[], metavar='KEY=JSON')
    b.add_argument('--manifest', type=Path, required=True)
    b.add_argument('--rate-table', type=Path)
    b.add_argument('--offsets', help='offsets JSON path or model key (default: config model)')
    b.add_argument('--arms', nargs='+', default=list(ARMS), choices=ARMS)
    b.add_argument('--out', type=Path, required=True)
    q = sub.add_parser('qps', help='per-key QP sets a config needs (JSON)')
    q.add_argument('--config', type=Path, required=True)
    q.add_argument('--set', action='append', default=[], metavar='KEY=JSON')
    q.add_argument('--offsets')
    a = p.parse_args(argv)
    cfg = load_config(a.config, parse_set(a.set))
    offsets, _ = load_offsets(a.offsets or cfg['model'])
    keys = [offsets['main_key']] + list(offsets['later_keys'])
    if a.cmd == 'qps':
        print(json.dumps(required_qps(cfg, keys, offsets['main_key'], offsets)))
        return 0
    from cloud_prep.measurement_v19 import common
    rows = common.load_rows(a.manifest)
    order = sample_order(rows, cfg['n_samples'], cfg['seed'], cfg['order'])
    table = load_rate_table(a.rate_table) if a.rate_table else None
    for arm in a.arms:
        recs = build_schedule(rows, order, arm, cfg, table, offsets)
        sha = write_jsonl(a.out / arm / 'schedule.jsonl', recs)
        print(json.dumps(dict(arm=arm, schedule_sha256=sha, **summary(recs))))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
