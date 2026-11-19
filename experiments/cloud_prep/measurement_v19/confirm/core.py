"""Shared pieces of the v19 confirmation and training-pilot analysis (CPU only: json + numpy; no torch, no model).

Protocol: one JSON file (protocol_confirm.json is the default). Keys starting with '_' are comments and
are left out of the hash. Observations: runner observations.jsonl rows (one per question x config); model and
LoRA come from the run-*.json files next to each observations.jsonl (meta['model'], meta.get('lora')).

Scores are unit means: the mean over bootstrap units (tasks.<t>.unit_field) of the within-unit question mean,
as in the paper. Bytes (mean input code length per evaluation unit): within-unit mean of the per-question image bytes
(main + later of the image each question uses), then the equal-weight mean over units; DocVQA units are documents
(cluster_id), so a document's page-image bytes enter as the mean over its questions, not as a sum over its pages.
Bootstrap: one multinomial resample of the units per cell (or per training model), shared by every statistic
drawn from it, so all differences are paired; percentile intervals.
"""
from __future__ import annotations

import csv
import datetime
import hashlib
import json
import os
from pathlib import Path
import re

import numpy as np

from cloud_prep.measurement_v19 import levels, offset_axis

HERE = Path(__file__).resolve().parent
DEFAULT_PROTOCOL = HERE / 'protocol_confirm.json'
CC, ZZ = 'C|C', 'Z|Z'
EPS = 1e-9


class LockError(RuntimeError):
    pass


def now_utc():
    return datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=str) + '\n')
    tmp.replace(path)


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = []
    for r in rows:
        cols += [k for k in r if k not in cols]
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols or ['empty'])
        w.writeheader()
        w.writerows(rows)


# ---- protocol, hashing, lock -----------------------------------------------------------------

def load_protocol(path=None):
    path = Path(path or DEFAULT_PROTOCOL).resolve()
    p = json.loads(path.read_text())
    p['_path'] = str(path)
    return p


def strip_comments(o):
    if isinstance(o, dict):
        return {k: strip_comments(v) for k, v in o.items() if not k.startswith('_')}
    if isinstance(o, list):
        return [strip_comments(v) for v in o]
    return o


def canonical(p):
    return json.dumps(strip_comments(p), sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()


def protocol_sha(p):
    return hashlib.sha256(canonical(p)).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def resolve(p, s):
    """Protocol path: $VARS and ~ expanded, relative paths taken from the protocol file's directory."""
    if not s:
        return None
    q = os.path.expanduser(os.path.expandvars(s))
    if '$' in q:
        raise ValueError(f'Unset environment variable in protocol path {s!r}')
    q = Path(q)
    return q if q.is_absolute() else Path(p['_path']).parent / q


def read_units(path, field='unit_id'):
    """Unit ids from .txt (one per line), .json (list or {'units': [...]}) or .jsonl (rows carrying `field`)."""
    path = Path(path)
    if path.suffix == '.jsonl':
        return sorted({json.loads(l)[field] for l in path.open() if l.strip()})
    if path.suffix == '.json':
        o = json.loads(path.read_text())
        return sorted(set(o['units'] if isinstance(o, dict) else o))
    return sorted({l.strip() for l in path.open() if l.strip()})


def unit_spec(p, which, task):
    return ((p.get('units') or {}).get(which) or {}).get(task) or {}


def lock_record(p):
    """(record, problems): protocol hash plus the confirmation unit lists as they are on disk now."""
    field = p['units'].get('match_field', 'unit_id')
    units, problems = {}, []
    confirm = p['units'].get('confirm') or {}
    if not confirm:
        problems.append('units.confirm lists no task')
    for task, spec in sorted(confirm.items()):
        path = resolve(p, (spec or {}).get('path'))
        if path is None or not path.is_file():
            problems.append(f'units.confirm.{task}: unit list missing ({path})')
            continue
        sha = file_sha(path)
        if spec.get('sha256') != sha:
            problems.append(f'units.confirm.{task}: protocol sha256 {spec.get("sha256")} != file sha256 {sha}')
        units[task] = dict(path=str(path), sha256=sha, n_units=len(read_units(path, field)))
    ps = protocol_sha(p)
    both = hashlib.sha256((ps + json.dumps({t: u['sha256'] for t, u in units.items()}, sort_keys=True)).encode())
    return dict(protocol_sha256=ps, confirm_units=units, lock_sha256=both.hexdigest()), problems


def default_lock_path(p):
    q = Path(p['_path'])
    return q.with_name(q.stem + '.lock.json')


def write_lock(p, out=None, allow_pending=False):
    rec, problems = lock_record(p)
    if p.get('pending') and not allow_pending:
        problems.append('pending is not empty: ' + ', '.join(p['pending']))
    if problems:
        raise LockError('Not locked:\n  ' + '\n  '.join(problems))
    rec.update(locked_at_utc=now_utc(), protocol_path=p['_path'], protocol_name=p.get('name'),
               protocol_version=p.get('version'), pending_at_lock=list(p.get('pending') or []))
    out = Path(out or default_lock_path(p))
    write_json(out, rec)
    return rec, out


def check_lock(p, lock_path=None):
    """Lock record if it matches the current protocol and unit lists; LockError otherwise."""
    path = Path(lock_path or default_lock_path(p))
    if not path.is_file():
        raise LockError(f'Confirmation report refused: no lock file {path}. '
                        'Lock the protocol before looking at confirmation results.')
    lk = json.loads(path.read_text())
    rec, problems = lock_record(p)
    if lk.get('protocol_sha256') != rec['protocol_sha256']:
        problems.append(f'protocol changed since the lock ({str(lk.get("protocol_sha256"))[:12]} locked, '
                        f'{rec["protocol_sha256"][:12]} now)')
    locked = {t: u.get('sha256') for t, u in (lk.get('confirm_units') or {}).items()}
    if locked != {t: u['sha256'] for t, u in rec['confirm_units'].items()}:
        problems.append('confirmation unit lists differ from the lock')
    if problems:
        raise LockError('Confirmation report refused:\n  ' + '\n  '.join(problems))
    return {**lk, 'lock_path': str(path)}


# ---- observations ----------------------------------------------------------------------------

def read_jsonl(path):
    lines = Path(path).read_text().splitlines()
    rows = []
    for i, l in enumerate(lines):
        if not l.strip():
            continue
        try:
            rows.append(json.loads(l))
        except json.JSONDecodeError:
            if i == len(lines) - 1:      # unterminated last line of a file still being written
                break
            raise
    return rows


def find_obs(specs):
    """[(observations file, model|None)]. A spec is PATH or MODEL=PATH; a directory is searched recursively for
    observations.jsonl and the runner's unmerged shard files observations.shard-i-of-n.jsonl (rows present in both
    a merged file and its shards count as duplicates)."""
    out = []
    for s in specs:
        s, model = str(s), None
        if '=' in s and not Path(s).exists():
            model, s = s.split('=', 1)
        q = Path(s)
        if q.is_file():
            out.append((q, model))
        elif q.is_dir():
            out += [(f, model) for f in sorted(q.rglob('observations*.jsonl'))
                    if f.name == 'observations.jsonl' or re.fullmatch(r'observations\.shard-\d+-of-\d+\.jsonl', f.name)]
        else:
            raise FileNotFoundError(s)
    return out


def run_meta(obs_file):
    """(model, lora) from the run-*.json next to an observations file; (None, None) without meta."""
    models, loras = set(), set()
    for f in sorted(Path(obs_file).parent.glob('run-*.json')):
        try:
            m = json.loads(f.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if isinstance(m, dict) and m.get('model'):
            models.add(m['model'])
            loras.add(json.dumps(m.get('lora'), sort_keys=True))
    if len(models) > 1 or len(loras) > 1:
        raise ValueError(f'{obs_file}: run-*.json files disagree on model or lora: {sorted(models)} {sorted(loras)}')
    return (models.pop() if models else None), (json.loads(loras.pop()) if loras else None)


def task_alias(p):
    return {a: t for t, spec in p['tasks'].items() for a in [t] + list(spec.get('aliases', []))}


def load_groups(p, specs, lora=False):
    """({(model, lora_key, task): group}, skipped). group: model, lora, task, rows {(id, config): row}, files,
    duplicates, conflicts (duplicate rows whose score differs). The later file / line wins. lora=False skips
    LoRA evaluations (meta lora not null)."""
    alias = task_alias(p)
    groups, skipped = {}, []
    for f, model in find_obs(specs):
        m, lo = run_meta(f)
        model = model or m
        if model is None:
            raise ValueError(f'{f}: no run-*.json with a model; pass MODEL=PATH')
        if lo is not None and not lora:
            skipped.append(dict(file=str(f), reason='LoRA evaluation (not part of this report)'))
            continue
        lk = json.dumps(lo, sort_keys=True) if lo is not None else None
        unknown = 0
        for r in read_jsonl(f):
            t = alias.get(r.get('task'))
            if t is None:
                unknown += 1
                continue
            g = groups.setdefault((model, lk, t), dict(model=model, lora=lo, task=t, rows={}, files=[],
                                                       duplicates=0, conflicts=0))
            k = (r['id'], r['config'])
            if k in g['rows']:
                sf = p['tasks'][t].get('score_field', 'correct')
                g['duplicates'] += 1
                g['conflicts'] += int(g['rows'][k].get(sf) != r.get(sf))
            g['rows'][k] = r
            if str(f) not in g['files']:
                g['files'].append(str(f))
        if unknown:
            skipped.append(dict(file=str(f), reason=f'{unknown} rows with a task not in the protocol'))
    return groups, skipped


def split_rows(p, task, rows, mode):
    """(calibration rows, test rows, info). Unit lists when the protocol gives them, else the rows' split labels
    (units.split_labels). Confirm mode: test = units.confirm[task], required. A unit in both raises."""
    u = p['units']
    field, labels = u.get('match_field', 'unit_id'), u.get('split_labels', {})

    def take(which, label_key):
        spec = unit_spec(p, which, task)
        path = resolve(p, spec.get('path'))
        if path is not None:
            if spec.get('sha256') and file_sha(path) != spec['sha256']:
                raise ValueError(f'units.{which}.{task}: {path} does not match the sha256 in the protocol')
            ids = set(read_units(path, field))
            return [r for r in rows if r.get(field) in ids], dict(source=str(path), n_listed=len(ids))
        if which == 'confirm':
            raise ValueError(f'units.confirm.{task} has no unit list')
        lab = set(labels.get(label_key, []))
        return [r for r in rows if r.get('split') in lab], dict(source='split labels: ' + ','.join(sorted(lab)))

    cal, ci = take('calibration', 'calibration')
    test, ti = take('confirm' if mode == 'confirm' else 'dev_test', 'test')
    overlap = {r[field] for r in cal} & {r[field] for r in test}
    if overlap:
        raise ValueError(f'{task}: {len(overlap)} units in both calibration and test, e.g. {sorted(overlap)[:3]}')
    ci['n_observed'] = len({r[field] for r in cal})
    ti['n_observed'] = len({r[field] for r in test})
    return cal, test, dict(calibration=ci, test=ti)


class View:
    """One split of one cell (or one evaluation). by[config][qid] = (unit, score, bytes, main, later, secondary)."""

    def __init__(self, rows, unit_field='unit_id', score_field='correct', secondary=None):
        self.by = {}
        self.missing_bytes = {}          # config -> rows without real code-length bytes (runner --allow-missing-bytes)
        for r in rows:
            b = r.get('bytes') or {}
            if b.get('main') is None or b.get('later') is None:
                self.missing_bytes[r['config']] = self.missing_bytes.get(r['config'], 0) + 1
                b = {}
            mb, lb = float(b.get('main', np.nan)), float(b.get('later', np.nan))
            s2 = r.get(secondary) if secondary else None
            self.by.setdefault(r['config'], {})[r['id']] = (
                r[unit_field], float(r[score_field]), mb + lb, mb, lb, None if s2 is None else float(s2))
        self.units = sorted({v[0] for d in self.by.values() for v in d.values()})
        self._um = {}

    def __contains__(self, c):
        return c in self.by

    def configs(self):
        return list(self.by)

    def unit_mean(self, c, col=1):
        if (c, col) not in self._um:
            acc = {}
            for v in self.by[c].values():
                if v[col] is not None:
                    acc.setdefault(v[0], []).append(v[col])
            self._um[c, col] = {u: float(np.mean(x)) for u, x in acc.items()}
        return self._um[c, col]

    def score(self, c, col=1):
        if c not in self.by:
            return None
        m = self.unit_mean(c, col)
        return float(np.mean(list(m.values()))) if m else None

    def nbytes(self, c, col=2):
        return self.score(c, col)

    def units_of(self, c):
        return set(self.unit_mean(c)) if c in self.by else set()

    def n(self, c):
        return len(self.by.get(c, {}))


def qvals(series, coefs):
    """{qid: (unit, sum coef*score)} over the questions present in every series (View.by[config] dicts)."""
    common = set(series[0]).intersection(*series[1:])
    return {q: (series[0][q][0], sum(c * s[q][1] for c, s in zip(coefs, series))) for q in common}


def avg_qvals(qvs):
    """Per question mean over several qvals dicts (a question counts where it is present)."""
    acc = {}
    for qv in qvs:
        for q, (u, v) in qv.items():
            acc.setdefault(q, (u, []))[1].append(v)
    return {q: (u, float(np.mean(v))) for q, (u, v) in acc.items()}


def unit_means(qv):
    acc = {}
    for u, v in qv.values():
        acc.setdefault(u, []).append(v)
    return {u: float(np.mean(x)) for u, x in acc.items()}


class Boot:
    """Shared unit bootstrap: multinomial unit counts drawn once; est() reweights any per-unit statistic."""

    def __init__(self, units, n_boot=4000, seed=0, ci=0.95):
        self.units = sorted(units)
        self.idx = {u: i for i, u in enumerate(self.units)}
        n = len(self.units)
        self.ci, self.n_boot, self.seed = ci, n_boot, seed
        rng = np.random.default_rng(seed)
        self.w = rng.multinomial(n, np.full(n, 1.0 / n), size=n_boot).astype(np.float64) if n else np.zeros((n_boot, 0))

    def est(self, um):
        us = [u for u in um if u in self.idx]
        if not us:
            return dict(est=None, lo=None, hi=None, n_units=0)
        v = np.array([um[u] for u in us])
        w = self.w[:, [self.idx[u] for u in us]]
        s = w.sum(1)
        b = (w[s > 0] @ v) / s[s > 0]
        a = (1 - self.ci) / 2
        lo, hi = np.quantile(b, [a, 1 - a]) if len(b) > 1 else (np.nan, np.nan)
        return dict(est=float(v.mean()), lo=float(lo), hi=float(hi), n_units=len(us))


def label(d, predicted='a>b'):
    """'holds' if the interval lies wholly on the predicted side of 0, 'refuted' if wholly on the other side,
    else 'undetermined'. predicted: 'a>b' or 'a<b'."""
    lo, hi = d.get('lo'), d.get('hi')
    if lo is None or hi is None or np.isnan(lo) or np.isnan(hi):
        return None
    if predicted == 'a<b':
        lo, hi = -hi, -lo
    return 'holds' if lo > 0 else 'refuted' if hi < 0 else 'undetermined'


def tol_flags(d, tol):
    if d.get('lo') is None or np.isnan(d['lo']):
        return dict(noninferior=None, equivalent=None)
    return dict(noninferior=bool(d['lo'] > -tol), equivalent=bool(-tol < d['lo'] and d['hi'] < tol))


# ---- families and selection ------------------------------------------------------------------

def family_configs(spec, role=None):
    """Candidate config names of one family spec. Forms: shared (q|q), offset (q|k), main_only (q|Z),
    later_only (Z|q), or an explicit 'configs' list. Offset takes signed 'ks', or 'abs_ks' with sign 'role' or
    'anti' (needs role +1 main-type / -1 later-type)."""
    if 'configs' in spec:
        out = list(spec['configs'])
    else:
        form, qs = spec['form'], [int(q) for q in spec['qs']]
        if form == 'shared':
            out = [f'q{q}|q{q}' for q in qs]
        elif form == 'main_only':
            out = [f'q{q}|Z' for q in qs]
        elif form == 'later_only':
            out = [f'Z|q{q}' for q in qs]
        elif form == 'offset':
            if 'abs_ks' in spec:
                if role not in (1, -1):
                    raise ValueError('a family with abs_ks needs a main- or later-type cell (role +1/-1)')
                sg = role if spec.get('sign', 'role') == 'role' else -role
                ks = [sg * abs(int(k)) for k in spec['abs_ks']]
            else:
                ks = [int(k) for k in spec['ks']]
            out = [levels.offset_name(q, k) for q in qs for k in ks]
        else:
            raise ValueError('Unknown family form ' + repr(form))
    for c in out:
        levels.parse_config(c)
    return out


def cell_spec(p, model, task):
    """Cell type defaults (cell_types[type]) updated by per-cell keys, with the model's family candidates.
    Family specs: family_defaults, updated by models.<model>.families, updated by the cell's optional
    'family_overrides' ({family: spec}, any family form incl. an explicit 'configs' list; v19 cloud, 2026-11-13).
    The same count and mirror checks apply to overridden families; cells without overrides are unchanged."""
    c = p['cells'][model][task]
    c = {'type': c} if isinstance(c, str) else dict(c)
    t = {**p['cell_types'][c['type']], **c}
    fams = {**p.get('family_defaults', {}), **p.get('models', {}).get(model, {}).get('families', {}),
            **(c.get('family_overrides') or {})}
    missing = [f for f in t['families'] if f not in fams]
    if missing:
        raise ValueError(f'{model}/{task}: families {missing} not defined in family_defaults or models.{model}')
    t['candidates'] = {f: family_configs(fams[f], t.get('role')) for f in t['families']}
    mirror_check(t, f'{model}/{task}')
    return t


def mirror_check(t, where=''):
    """R and A (or Kpos and Kneg) must be mirror images on the offset axis: the same (q, |k|) pairs, R with
    sign(k) = role and A with sign(k) = -role (Kpos k > 0, Kneg k < 0). Raises ValueError otherwise."""
    role = t.get('role')
    for a, b, sa, sb in (('R', 'A', role, None if role is None else -role), ('Kpos', 'Kneg', 1, -1)):
        if a not in t['candidates'] or b not in t['candidates']:
            continue
        pairs = {}
        for f, sign in ((a, sa), (b, sb)):
            ps = []
            for c in t['candidates'][f]:
                form, q, k = levels.parse_config(c)
                if form != 'offset' or k == 0 or (sign is not None and (k > 0) != (sign > 0)):
                    raise ValueError(f'{where}: {f} candidate {c} is not an offset config with k of sign {sign}')
                ps.append((q, abs(k)))
            pairs[f] = sorted(ps)
        if pairs[a] != pairs[b]:
            raise ValueError(f'{where}: {a} and {b} are not mirror images (same q and |k|): {pairs[a]} vs {pairs[b]}')


def config_kind(c, role=None):
    try:
        return offset_axis.config_family(c, role)
    except ValueError:
        return dict(family='other', q=None, k=None, side=None, main=None, later=None)


def has_bytes(view, c):
    """Observed with real code-length bytes on every row (configs without them are never selected)."""
    return c in view and not view.missing_bytes.get(c) and bool(np.isfinite(view.nbytes(c)))


def pick_budget(view, cands, budget):
    """Highest score among candidates with mean bytes <= budget; ties: fewer bytes, then list order."""
    best = None
    for i, c in enumerate(cands):
        if not has_bytes(view, c) or budget is None or not np.isfinite(budget):
            continue
        s, b = view.score(c), view.nbytes(c)
        if b <= budget * (1 + EPS):
            key = (round(s, 12), -b, -i)
            if best is None or key > best[0]:
                best = (key, c)
    return best[1] if best else None


def pick_target(view, cands, target):
    """Fewest mean bytes among candidates with score >= target; ties: higher score, then list order."""
    best = None
    for i, c in enumerate(cands):
        if not has_bytes(view, c) or target is None:
            continue
        s, b = view.score(c), view.nbytes(c)
        if s >= target - EPS:
            key = (-b, round(s, 12), -i)
            if best is None or key > best[0]:
                best = (key, c)
    return best[1] if best else None


def pareto(view, cands):
    """Candidates that some budget would select: by bytes ascending, each strictly better than all cheaper ones."""
    out, top = [], -np.inf
    for c in sorted((c for c in cands if has_bytes(view, c)), key=lambda c: (view.nbytes(c), -view.score(c))):
        if view.score(c) > top + EPS:
            out.append(c)
            top = view.score(c)
    return out


def score_target(view, spec):
    """Absolute score from a rule: abs (value), cc (value * score(C|C)), gap (Z|Z + value * (C|C - Z|Z))."""
    t, v = spec['type'], float(spec['value'])
    if t == 'abs':
        return v
    if CC not in view:
        raise KeyError(f'{CC} not observed')
    if t == 'cc':
        return v * view.score(CC)
    if t == 'gap':
        if ZZ not in view:
            raise KeyError(f'{ZZ} not observed')
        return view.score(ZZ) + v * (view.score(CC) - view.score(ZZ))
    raise ValueError('Unknown target type ' + repr(t))


def tier_budget(view, spec, ref_cands):
    """Budget (mean input code length per evaluation unit, see the module note) of one tier, from calibration only. bytes: fixed; raw_fraction: value x
    bytes of C|C; cc/gap/abs: bytes of the cheapest reference-family config reaching that score target."""
    t = spec['type']
    if t == 'bytes':
        return dict(budget=float(spec['value']))
    if t == 'raw_fraction':
        if CC not in view:
            raise KeyError(f'{CC} not observed')
        return dict(budget=float(spec['value']) * view.nbytes(CC))
    target = score_target(view, spec)
    c = pick_target(view, ref_cands, target)
    return dict(budget=view.nbytes(c) if c else None, target=target, reference_config=c,
                reason=None if c else 'no reference-family candidate reaches the target on calibration')


def bd(curve_a, curve_b, min_points=3, min_points_in_overlap=2, min_overlap_score=0.0, min_overlap_log10=0.0):
    """BD-style summaries of two (bytes, score) curves, piecewise linear in log10 bytes, no extrapolation.
    bd_score: mean score_a - score_b over the common log10-bytes range. bd_rate_pct: 100*(10**mean(log10 a -
    log10 b) - 1) over the common score range, only when both curves increase strictly (negative: a needs fewer
    bytes). Each curve needs min_points points, min_points_in_overlap of them inside the common range."""
    out = {}
    curves = []
    for cur in (curve_a, curve_b):
        best = {}
        for b, s in cur:
            if b is not None and s is not None and b > 0:
                x = float(np.log10(b))
                best[x] = max(best.get(x, -np.inf), s)
        curves.append(np.array(sorted(best.items())).reshape(-1, 2))
    (A, B) = curves
    if len(A) < min_points or len(B) < min_points:
        return dict(reason=f'a curve has fewer than {min_points} points ({len(A)}, {len(B)})')
    xa, ya, xb, yb = A[:, 0], A[:, 1], B[:, 0], B[:, 1]
    lo, hi = max(xa[0], xb[0]), min(xa[-1], xb[-1])
    inside = lambda x, l, h: int(((x >= l - EPS) & (x <= h + EPS)).sum())
    if hi - lo > max(min_overlap_log10, EPS) and min(inside(xa, lo, hi), inside(xb, lo, hi)) >= min_points_in_overlap:
        g = np.linspace(lo, hi, 513)
        out.update(bd_score=float(np.mean(np.interp(g, xa, ya) - np.interp(g, xb, yb))), log10_bytes_range=[lo, hi])
    else:
        out['bd_score_reason'] = 'common log10-bytes range too short or too few points inside it'
    if (np.diff(ya) > 0).all() and (np.diff(yb) > 0).all():
        slo, shi = max(ya[0], yb[0]), min(ya[-1], yb[-1])
        if shi - slo > max(min_overlap_score, EPS) and min(inside(ya, slo, shi), inside(yb, slo, shi)) >= min_points_in_overlap:
            g = np.linspace(slo, shi, 513)
            d = np.interp(g, ya, xa) - np.interp(g, yb, xb)
            out.update(bd_rate_pct=float((10 ** d.mean() - 1) * 100), score_range=[slo, shi])
        else:
            out['bd_rate_reason'] = 'common score range too short or too few points inside it'
    else:
        out['bd_rate_reason'] = 'a curve is not strictly increasing in score'
    return out
