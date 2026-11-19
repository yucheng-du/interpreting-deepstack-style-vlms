"""Code-length tables of training images (bytes per image x stream x QP), used only to draw training configs.

Two routes; evaluation always uses real NNCodec code lengths (runner sweep with rate_audit bytes).
  nncodec      real bytes, nncodec-v19 env, CPU: streams a runner extract dir unit by unit, codes only the QPs
               each key needs (schedule.required_qps), process pool, resumable JSONL, optional
               --delete-features after a unit is coded, --follow SEC to keep pace with a running extract.
               Cost scales with images x sum_k |QPs_k| encodes (calibration audits: ~0.1 s per encode+decode).
  proxy        entropy proxy from a runner extract dir (any env with torch): bytes = ceil(numel*H0/8) + header,
               H0 = empirical entropy (bits) of the integer levels of the exact quantizer replica.
  proxy-model  same proxy computed online from the frozen encoder (model environment); writes
               no features. Cheap, but it is a proxy: check it with `validate` on a subset coded by `nncodec`.
Helpers
  subset-manifest  rows of the first n_samples training rows (absolute image paths) -> runner extract input
  validate         proxy vs real on shared units: byte ratios per stream x QP; with --config, the configs the
                   proxy picks per compression arm, their real bytes, and q agreement with the real-table solve
  merge            JSONL -> bytes.json-format table (what runner/rate_audit consumers read)

JSONL line: {unit, kind: nncodec|proxy, records: [{stream, qp, bytes}], ...}; qp 'raw' = 2*numel (BF16).
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import time

from cloud_prep.measurement_v19 import levels
from cloud_prep.measurement_v19.train import schedule

HEADER_BYTES = 42      # all-zero bitstream size in the v19 calibration audits (41-43 bytes)


def entropy_bytes(lv, header=HEADER_BYTES):
    """ceil(numel * H0 / 8) + header for an integer level tensor (torch, any device) or ndarray."""
    import torch
    t = lv if hasattr(lv, 'detach') else torch.from_numpy(lv)
    t = t.reshape(-1)
    n = t.numel()
    mn, mx = int(t.min()), int(t.max())
    if mx - mn < 50_000_000:
        c = torch.bincount(t - mn)
        c = c[c > 0]
    else:
        c = torch.unique(t, return_counts=True)[1]
    p = c.double() / n
    h = float(-(p * torch.log2(p)).sum())
    return int(math.ceil(n * h / 8)) + int(header)


def load_calibration(path):
    """{stream: [(qp, median real/proxy)]} from a `validate` output (per_stream_qp rows)."""
    if path is None:
        return None
    d = json.loads(Path(path).read_text())
    cal = {}
    for r in d['bytes']['per_stream_qp']:
        cal.setdefault(r['stream'], []).append((r['qp'], r['ratio_median']))
    return {s: sorted(v) for s, v in cal.items()}


def cal_factor(cal, stream, qp):
    """Median real/proxy ratio at (stream, qp): linear in QP between measured QPs, clamped outside."""
    pts = (cal or {}).get(stream)
    if not pts:
        return 1.0
    if qp <= pts[0][0]:
        return pts[0][1]
    if qp >= pts[-1][0]:
        return pts[-1][1]
    for (q0, r0), (q1, r1) in zip(pts, pts[1:]):
        if q0 <= qp <= q1:
            return r0 + (r1 - r0) * (qp - q0) / (q1 - q0)


def proxy_records(vals, qps_per_key, header=HEADER_BYTES, cal=None):
    """vals: {key: tensor or ndarray [tokens, hidden]} -> [{stream, qp, bytes}] incl. raw.
    cal: per-(stream, qp) real/proxy factors from `validate` on a coded subset (None: raw H0 proxy)."""
    import torch
    recs = []
    for k, x in vals.items():
        t = x if hasattr(x, 'detach') else torch.from_numpy(x)
        recs.append(dict(stream=k, qp='raw', bytes=2 * t.numel()))
        for qp in qps_per_key.get(k, []):
            b = entropy_bytes(levels.quantize_levels(t, qp), header)
            recs.append(dict(stream=k, qp=qp, bytes=int(round(b * cal_factor(cal, k, qp))) if cal else b))
    return recs


def done_units(path):
    found = set()
    if Path(path).exists():
        with open(path) as f:
            for n, line in enumerate(f, 1):
                try:
                    found.add(json.loads(line)['unit'])
                except json.JSONDecodeError as exc:
                    raise ValueError(f'{path}:{n} incomplete; remove the partial last line before resuming') from exc
    return found


def feature_units(features):
    """Records of a runner extract dir: features.jsonl plus shard files (runner --shard i/n), first line per key."""
    out, seen = [], set()
    paths = [Path(features) / 'features.jsonl'] + sorted(Path(features).glob('features.shard-*-of-*.jsonl'))
    for p in paths:
        if not p.exists():
            continue
        with open(p) as f:
            for line in f:
                if line.strip():
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        break          # extract is still writing this line
                    if r.get('key', r['unit_id']) not in seen:
                        seen.add(r.get('key', r['unit_id'])); out.append(r)
    return out


def extract_finished(features):
    """Every runner extract meta in the dir has left loading/running (False before the first meta appears)."""
    metas = list(Path(features).glob('run-extract-*.json'))
    return bool(metas) and all(json.loads(p.read_text()).get('status') not in ('loading', 'running') for p in metas)


def load_unit(features, rec):
    import numpy as np
    d = Path(features) / 'features' / rec['unit_id'].replace('/', '_')
    return {k: (lambda x: x.reshape(-1, x.shape[-1]))(np.load(d / f'{k}.npy'))
            for k in rec['streams'] if k not in rec['inactive']}


def qps_from_args(a, keys):
    if a.qps:
        return {k: sorted(set(a.qps)) for k in keys}
    cfg = schedule.load_config(a.config, schedule.parse_set(a.set))
    offsets, _ = schedule.load_offsets(a.offsets or cfg['model'])
    return schedule.required_qps(cfg, [offsets['main_key']] + list(offsets['later_keys']), offsets['main_key'], offsets)


def _nncodec_unit(job):
    """Worker: code one unit at its per-key QPs; optional decode == replica check."""
    import numpy as np
    from cloud_prep.measurement_v19 import codec
    features, rec, qmap, verify = job
    t0 = time.time()
    vals = load_unit(features, rec)
    recs, ok = [], True
    for k, x in vals.items():
        recs.append(dict(stream=k, qp='raw', bytes=2 * x.size))
        for qp in qmap.get(k, []):
            bs = codec.encode(x, qp)
            recs.append(dict(stream=k, qp=qp, bytes=len(bs)))
            if verify:
                import torch
                ok &= bool(np.array_equal(codec.decode(bs), levels.quantize(torch.from_numpy(x), qp).numpy()))
    return dict(unit=rec['unit_id'], kind='nncodec', nncodec=codec.nncodec_version(), records=recs,
                seconds=round(time.time() - t0, 3), replica_checked=bool(verify), replica_equal=ok if verify else None)


def cmd_nncodec(a):
    import multiprocessing as mp
    from cloud_prep.measurement_v19 import common
    if a.delete_features:
        common.require_under_root(a.features)
    done = done_units(a.out)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    keys = None
    qmap = None
    idle_since = time.time()
    n_new = 0
    finishing = False
    ctx = mp.get_context('spawn')
    with ctx.Pool(a.workers) as pool, open(a.out, 'a', buffering=1) as out:
        while True:
            recs = [r for r in feature_units(a.features) if r['unit_id'] not in done]
            if a.max_units is not None:
                recs = recs[:max(0, a.max_units - n_new)]
            if recs:
                if qmap is None:
                    keys = list(recs[0]['streams'])
                    qmap = qps_from_args(a, keys)
                jobs = [(str(a.features), r, qmap, a.verify_every > 0 and (len(done) + j) % a.verify_every == 0)
                        for j, r in enumerate(recs)]
                for res in pool.imap_unordered(_nncodec_unit, jobs):
                    if res['replica_equal'] is False:
                        raise RuntimeError(f"NNCodec decode differs from the replica: {res['unit']}")
                    out.write(json.dumps(res) + '\n')
                    done.add(res['unit']); n_new += 1
                    if a.delete_features:
                        import shutil
                        shutil.rmtree(Path(a.features) / 'features' / res['unit'].replace('/', '_'), ignore_errors=True)
                    print(json.dumps(dict(event='unit', unit=res['unit'], seconds=res['seconds'], done=len(done))), flush=True)
                idle_since = time.time()
            if a.max_units is not None and n_new >= a.max_units:
                break
            if not a.follow or time.time() - idle_since > a.follow:
                break
            if not recs and extract_finished(a.features):
                if finishing:          # a pass after the extract finished found nothing new
                    break
                finishing = True       # one more pass for units written just before the extract finished
                continue
            time.sleep(min(30, a.follow))
    print(json.dumps(dict(event='done', new_units=n_new, total=len(done))))


def cmd_proxy(a):
    done = done_units(a.out)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    qmap = None
    cal = load_calibration(a.calibrate)
    with open(a.out, 'a', buffering=1) as out:
        for n, rec in enumerate(feature_units(a.features)):
            if a.max_units is not None and n >= a.max_units:
                break
            if rec['unit_id'] in done:
                continue
            t0 = time.time()
            vals = load_unit(a.features, rec)
            qmap = qmap or qps_from_args(a, list(rec['streams']))
            out.write(json.dumps(dict(unit=rec['unit_id'], kind='proxy', header_bytes=a.header_bytes,
                                      calibration=str(a.calibrate) if cal else None,
                                      records=proxy_records(vals, qmap, a.header_bytes, cal),
                                      seconds=round(time.time() - t0, 3))) + '\n')


def cmd_proxy_model(a):
    """Frozen encoder -> per-unit proxy bytes on the device; nothing per token is saved."""
    from cloud_prep.measurement_v19 import common
    import torch
    from cloud_prep.measurement_v15.adapter import tensors
    rows = common.load_rows(a.manifest)
    if a.config:
        cfg = schedule.load_config(a.config, schedule.parse_set(a.set))
        rows = [rows[j] for j in schedule.sample_order(rows, cfg['n_samples'], cfg['seed'], cfg['order'])]
    first = {}
    for r in rows:
        first.setdefault(r['unit_id'], r)
    units = list(first.items())[:a.max_units] if a.max_units else list(first.items())
    common.require_research()
    common.require_under_root(a.out)
    done = done_units(a.out)
    cal = load_calibration(a.calibrate)
    adapter, family, _, _ = common.load_model(a.model, a.device)
    qmap = None
    with open(a.out, 'a', buffering=1) as out, torch.inference_mode():
        for unit, row in units:
            if unit in done:
                continue
            t0 = time.time()
            inp, _ = adapter.prepare(row, common.variant(row), 'a')
            vals = tensors(adapter.encode(inp), family)
            if family == 'deepstack':
                raise ValueError('proxy-model covers Qwen and Granite (no inactive groups)')
            qmap = qmap or qps_from_args(a, list(vals))
            out.write(json.dumps(dict(unit=unit, kind='proxy', header_bytes=a.header_bytes, source='proxy-model',
                                      calibration=str(a.calibrate) if cal else None,
                                      records=proxy_records(vals, qmap, a.header_bytes, cal),
                                      seconds=round(time.time() - t0, 3))) + '\n')


def cmd_subset_manifest(a):
    from cloud_prep.measurement_v19 import common
    cfg = schedule.load_config(a.config, schedule.parse_set(a.set))
    rows = common.load_rows(a.manifest)
    order = schedule.sample_order(rows, cfg['n_samples'], cfg['seed'], cfg['order'])
    a.out.parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    with open(a.out, 'w') as f:
        for j in order:
            r = rows[j]
            if a.one_row_per_unit and r['unit_id'] in seen:
                continue
            seen.add(r['unit_id'])
            f.write(json.dumps(r) + '\n')     # load_rows already made source_images absolute
    print(json.dumps(dict(rows=len(order), units=len(seen), out=str(a.out))))


def _q(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))] if xs else None


def shared_units(proxy, real, half=None):
    """Sorted shared unit ids; half=0/1 keeps every other one (calibrate on one half, check on the other)."""
    s = sorted(set(proxy['per_unit']) & set(real['per_unit']))
    return s if half is None else s[half::2]


def compare_tables(proxy, real, half=None):
    """Per (stream, qp) ratio real/proxy over shared units and Pearson r of log bytes."""
    shared = shared_units(proxy, real, half)
    by, lp, lr = {}, [], []
    for u in shared:
        P, R = proxy['per_unit'][u], real['per_unit'][u]
        for key in set(P) & set(R):
            if key[1] == 'raw':
                continue
            by.setdefault(key, []).append(R[key] / P[key])
            lp.append(math.log(P[key])); lr.append(math.log(R[key]))
    rows = []
    for (s, qp), rs in sorted(by.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        rows.append(dict(stream=s, qp=qp, n=len(rs), ratio_median=_q(rs, .5), ratio_p10=_q(rs, .1), ratio_p90=_q(rs, .9)))
    r = None
    if len(lp) > 2:
        mp_, mr = sum(lp) / len(lp), sum(lr) / len(lr)
        cov = sum((x - mp_) * (y - mr) for x, y in zip(lp, lr))
        r = cov / math.sqrt(sum((x - mp_) ** 2 for x in lp) * sum((y - mr) ** 2 for y in lr))
    allr = [x for rs in by.values() for x in rs]
    return dict(shared_units=len(shared), half=half, pearson_log_bytes=r, ratio_median_all=_q(allr, .5),
                ratio_p10_all=_q(allr, .1), ratio_p90_all=_q(allr, .9), per_stream_qp=rows)


def compare_solves(proxy, real, cfg, offsets, half=None):
    """Configs each compression arm draws from the proxy table vs the real table, on shared units."""
    import statistics as st
    keys = [offsets['main_key']] + list(offsets['later_keys'])
    mk = offsets['main_key']
    shared = shared_units(proxy, real, half)
    qs = schedule.q_values(cfg)
    out = {}
    for arm in schedule.COMPRESSION_ARMS:
        res = []
        for i, u in enumerate(shared):
            P, R = proxy['per_unit'][u], real['per_unit'][u]
            dr = schedule.draws(cfg['seed'], i, cfg)
            k = schedule.arm_k(arm, dr['kabs'], dr['sign'], cfg['role'])
            try:
                bp, br = (schedule.budget_of(T, dr['ref'], cfg, keys, mk, offsets) for T in (P, R))
                sp = schedule.solve_q(P, k, bp, qs, keys, mk, offsets, strict=False)
                sr = schedule.solve_q(R, k, br, qs, keys, mk, offsets, strict=False)
            except ValueError:
                continue
            real_of_p = schedule.name_bytes(R, sp['name'], keys, mk, offsets)
            res.append(dict(q_p=sp['q'], q_r=sr['q'], budget_r=br, proxy=sp['bytes']['total'],
                            real=real_of_p['total'] if real_of_p else None))
        ok = [r for r in res if r['real'] is not None]
        out[arm] = dict(n=len(res), n_real=len(ok),
                        q_agree=sum(r['q_p'] == r['q_r'] for r in res) / len(res) if res else None,
                        q_absdiff_mean=st.mean(abs(r['q_p'] - r['q_r']) for r in res) if res else None,
                        proxy_bytes_mean=st.mean(r['proxy'] for r in res) if res else None,
                        real_bytes_mean=st.mean(r['real'] for r in ok) if ok else None,
                        real_over_real_budget_mean=st.mean(r['real'] / r['budget_r'] for r in ok) if ok else None,
                        frac_real_within_real_budget=sum(r['real'] <= r['budget_r'] for r in ok) / len(ok) if ok else None)
    rb = {k: v['real_bytes_mean'] for k, v in out.items()}
    if rb.get('role') and rb.get('anti'):
        out['real_bytes_ratio_role_over_anti'] = rb['role'] / rb['anti']
    if rb.get('generic') and rb.get('anti'):
        out['real_bytes_ratio_generic_over_anti'] = rb['generic'] / rb['anti']
    return out


def cmd_validate(a):
    proxy, real = schedule.load_rate_table(a.proxy), schedule.load_rate_table(a.real)
    res = dict(proxy=str(a.proxy), real=str(a.real), bytes=compare_tables(proxy, real, a.half))
    if a.config:
        cfg = schedule.load_config(a.config, schedule.parse_set(a.set))
        offsets, _ = schedule.load_offsets(a.offsets or cfg['model'])
        res['solves'] = compare_solves(proxy, real, cfg, offsets, a.half)
    text = json.dumps(res, indent=1)
    if a.out:
        Path(a.out).write_text(text + '\n')
    print(text)


def cmd_merge(a):
    t = schedule.load_rate_table(a.jsonl)
    per_key = {}
    for recs in t['per_unit'].values():
        for (s, qp) in recs:
            if qp != 'raw':
                per_key.setdefault(s, set()).add(qp)
    out = dict(kind=t['kind'], source=str(a.jsonl), qps=sorted(set().union(*per_key.values())) if per_key else [],
               per_key_qps={k: sorted(v) for k, v in per_key.items()},
               per_unit={u: [dict(stream=s, qp=qp, bytes=b) for (s, qp), b in recs.items()] for u, recs in t['per_unit'].items()})
    Path(a.out).write_text(json.dumps(out))
    print(json.dumps(dict(units=len(out['per_unit']), qps=len(out['qps']), out=str(a.out))))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)

    def qp_args(s):
        s.add_argument('--config', type=Path, help='train config: QPs per key = schedule.required_qps')
        s.add_argument('--set', action='append', default=[], metavar='KEY=JSON')
        s.add_argument('--offsets', help='offsets JSON or model key (default: config model)')
        s.add_argument('--qps', type=int, nargs='*', help='explicit QPs for every key (instead of --config)')

    n = sub.add_parser('nncodec', help='real NNCodec bytes from a runner extract dir (nncodec-v19 env)')
    n.add_argument('--features', type=Path, required=True)
    n.add_argument('--out', type=Path, required=True, help='JSONL (resumable)')
    n.add_argument('--workers', type=int, default=os.cpu_count())
    n.add_argument('--verify-every', type=int, default=100, help='decode == replica check on every Nth unit (0: off)')
    n.add_argument('--delete-features', action='store_true', help='remove a unit\'s .npy dir once coded')
    n.add_argument('--follow', type=float, default=0, help='keep polling for new units; stop after SEC without any')
    n.add_argument('--max-units', type=int)
    qp_args(n)
    x = sub.add_parser('proxy', help='entropy proxy from a runner extract dir')
    x.add_argument('--features', type=Path, required=True)
    x.add_argument('--out', type=Path, required=True)
    x.add_argument('--header-bytes', type=int, default=HEADER_BYTES)
    x.add_argument('--calibrate', type=Path, help='validate output: scale by median real/proxy per (stream, qp)')
    x.add_argument('--max-units', type=int)
    qp_args(x)
    m = sub.add_parser('proxy-model', help='entropy proxy online from the frozen encoder (model env)')
    m.add_argument('--model', required=True)
    m.add_argument('--manifest', type=Path, required=True)
    m.add_argument('--out', type=Path, required=True)
    m.add_argument('--device', choices=['cuda', 'mps'])
    m.add_argument('--header-bytes', type=int, default=HEADER_BYTES)
    m.add_argument('--calibrate', type=Path, help='validate output: scale by median real/proxy per (stream, qp)')
    m.add_argument('--max-units', type=int)
    qp_args(m)
    s = sub.add_parser('subset-manifest', help='manifest of the first n_samples training rows (absolute paths)')
    s.add_argument('--config', type=Path, required=True)
    s.add_argument('--set', action='append', default=[], metavar='KEY=JSON')
    s.add_argument('--manifest', type=Path, required=True)
    s.add_argument('--out', type=Path, required=True)
    s.add_argument('--one-row-per-unit', action='store_true')
    v = sub.add_parser('validate', help='proxy vs real table on shared units')
    v.add_argument('--proxy', type=Path, required=True)
    v.add_argument('--real', type=Path, required=True)
    v.add_argument('--config', type=Path)
    v.add_argument('--set', action='append', default=[], metavar='KEY=JSON')
    v.add_argument('--offsets')
    v.add_argument('--half', type=int, choices=[0, 1], help='use every other shared unit (0: even, 1: odd index)')
    v.add_argument('--out', type=Path)
    g = sub.add_parser('merge', help='JSONL -> bytes.json-format table')
    g.add_argument('--jsonl', type=Path, required=True)
    g.add_argument('--out', type=Path, required=True)
    a = p.parse_args(argv)
    if a.cmd in ('nncodec', 'proxy', 'proxy-model') and not (a.config or a.qps):
        p.error('give --config (per-key QPs from the train config) or --qps')
    dict(nncodec=cmd_nncodec, proxy=cmd_proxy, validate=cmd_validate, merge=cmd_merge,
         **{'proxy-model': cmd_proxy_model, 'subset-manifest': cmd_subset_manifest})[a.cmd](a)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
