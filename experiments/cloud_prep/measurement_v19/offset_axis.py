"""Scale offsets d_i and the offset-axis config families (no model; numpy, torch optional).

d_i = round_half_away(4 * log2(sigma_i / sigma_main)), sigma = population standard deviation
(ddof 0, about the pooled mean) of all elements of the stream pooled over the calibration
units where the stream is active (structurally inactive DeepStack groups excluded). This is
the 'std' of rate_audit.py summary.json, from which the baseline d values were read.

CLI
  offsets  --features DIR (runner extract output) --model M [--main-key K] --out offsets/M.json
  configs  --qs 0 4 8 --ks 2 4 --role 1     print the family config names (one per line)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from cloud_prep.measurement_v19 import levels

OFFSET_SCALE = 4          # QP units per octave of step size at qp_density=2
DEFINITION = ('sigma: population std (ddof 0, about the pooled mean) over all elements of the stream, pooled over '
              'calibration units where the stream is active; float64 n/sum/sumsq. '
              'd = round_half_away_from_zero(4*log2(sigma_key/sigma_main)). Same quantity as rate_audit.py stats.std.')


def round_half_away(x):
    return int(math.copysign(math.floor(abs(x) + 0.5), x))


class StreamStats:
    """Per-key n, sum, sumsq (float64) plus the unit ids seen; mergeable, JSON round-trip."""

    def __init__(self):
        self.acc = {}
        self.units = set()

    def update(self, key, x, unit=None):
        if hasattr(x, 'detach'):
            import torch
            t = x.detach().to(torch.float64)
            n, s, ss = t.numel(), float(t.sum()), float((t * t).sum())
        else:
            import numpy as np
            a = np.asarray(x, dtype=np.float64)
            n, s, ss = a.size, float(a.sum()), float(np.dot(a.ravel(), a.ravel()))
        e = self.acc.setdefault(key, [0, 0.0, 0.0])
        e[0] += int(n); e[1] += s; e[2] += ss
        if unit is not None:
            self.units.add(str(unit))
        return self

    def merge(self, other):
        for k, (n, s, ss) in other.acc.items():
            e = self.acc.setdefault(k, [0, 0.0, 0.0])
            e[0] += n; e[1] += s; e[2] += ss
        self.units |= other.units
        return self

    def mean(self, key):
        n, s, _ = self.acc[key]
        return s / n

    def sigma(self, key):
        n, s, ss = self.acc[key]
        m = s / n
        return math.sqrt(max(ss / n - m * m, 0.0))

    def to_json(self):
        return dict(acc={k: dict(n=v[0], sum=v[1], sumsq=v[2]) for k, v in self.acc.items()}, units=sorted(self.units))

    @classmethod
    def from_json(cls, obj):
        st = cls()
        st.acc = {k: [int(v['n']), float(v['sum']), float(v['sumsq'])] for k, v in obj['acc'].items()}
        st.units = set(obj.get('units', []))
        return st


def units_sha256(units):
    return hashlib.sha256('\n'.join(sorted(str(u) for u in units)).encode()).hexdigest()


def d_value(sigma_key, sigma_main, scale=OFFSET_SCALE):
    return round_half_away(scale * math.log2(sigma_key / sigma_main))


def offsets_table(stats, model, main_key, *, source=None, scale=OFFSET_SCALE):
    """Offsets JSON (small): d for later keys, sigma/n/mean for all keys, unit provenance."""
    lk = levels.later_keys(stats.acc, main_key)
    sm = stats.sigma(main_key)
    streams = {}
    for k in [main_key] + lk:
        s = stats.sigma(k)
        raw = scale * math.log2(s / sm)
        streams[k] = dict(sigma=s, mean=stats.mean(k), n=stats.acc[k][0], raw_d=raw, d=round_half_away(raw))
    return dict(model=model, main_key=main_key, later_keys=lk, d={k: streams[k]['d'] for k in lk},
                streams=streams, scale=scale, definition=DEFINITION, n_units=len(stats.units),
                units_sha256=units_sha256(stats.units), source=source)


def load_offsets(path):
    return json.loads(Path(path).read_text())


def save_offsets(table, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(table, indent=1) + '\n')


def stats_from_features(features_dir, keys=None):
    """Stream over a runner extract dir (features.jsonl and shard files + features/<feature key>/<key>.npy);
    inactive keys skipped. Every coded image counts (DocVQA pages: runner feature_key unit@sha16); n_units counts
    unit ids."""
    import numpy as np
    features_dir = Path(features_dir)
    st = StreamStats()
    files = [p for p in [features_dir / 'features.jsonl'] if p.exists()] + sorted(features_dir.glob('features.shard-*-of-*.jsonl'))
    seen = set()
    for p in files:
        for line in open(p):
            if not line.strip():
                continue
            r = json.loads(line)
            fk = r.get('key', r['unit_id'])
            if fk in seen:
                continue
            seen.add(fk)
            d = features_dir / 'features' / fk.replace('/', '_')
            for k in r['streams']:
                if k in r['inactive'] or (keys is not None and k not in keys):
                    continue
                st.update(k, np.load(d / f'{k}.npy', mmap_mode='r'), unit=r['unit_id'])
    return st


# ---- config families -------------------------------------------------------------------------

CORNERS = ['C|C', 'C|Z', 'Z|C', 'Z|Z']
MEAN_CORNERS = ['C|M', 'M|C', 'M|M']


def config_families(qs, ks, role, *, corners=True, mean_corners=True, single=True):
    """{family: [names]}. role=+1 main-type (R takes k>0), -1 later-type (R takes k<0). ks: |k| values."""
    if role not in (1, -1):
        raise ValueError('role must be +1 (main-type) or -1 (later-type)')
    ks = sorted({abs(int(k)) for k in ks} - {0})
    qs = [int(q) for q in qs]
    fam = dict(S=[f'q{q}|q{q}' for q in qs], N=[levels.offset_name(q, 0) for q in qs],
               R=[levels.offset_name(q, role * k) for q in qs for k in ks],
               A=[levels.offset_name(q, -role * k) for q in qs for k in ks])
    fam['axis'] = [levels.offset_name(q, k) for q in qs for k in sorted({0} | set(ks) | {-k for k in ks})]
    if corners:
        fam['corner'] = CORNERS + (MEAN_CORNERS if mean_corners else [])
    if single:
        fam['single'] = [f'q{q}|Z' for q in qs] + [f'Z|q{q}' for q in qs]
    return fam


def flat_configs(families, order=('corner', 'S', 'axis', 'single')):
    """Ordered, de-duplicated list (axis already contains N, R and A)."""
    out = []
    for f in order:
        for c in families.get(f, []):
            if c not in out:
                out.append(c)
    return out


def required_qps(configs, keys, main_key, offsets=None):
    return levels.config_qps(configs, keys, main_key, offsets)


def config_family(name, role=None):
    """dict(family, q, k, side, main, later). family: S | axis | corner | single_main | single_later |
    explicit | other. side (axis only): N for k0; with role given, R if sign(k)==role else A."""
    form, x, y = levels.parse_config(name)
    out = dict(family='other', q=None, k=None, side=None, main=None, later=None)
    if form == 'offset':
        side = 'N' if y == 0 else (None if role is None else ('R' if (y > 0) == (role > 0) else 'A'))
        return {**out, 'family': 'axis', 'q': x, 'k': y, 'side': side, 'main': f'q{x}'}
    if form == 'explicit':
        return {**out, 'family': 'explicit', 'main': x, 'later': ','.join(y)}
    (mk, mq), (lk, lq) = levels.parse_level(x), levels.parse_level(y)
    out.update(main=x, later=y)
    if mk == 'q' and lk == 'q':
        return {**out, 'family': 'S', 'q': mq} if mq == lq else out
    if mk != 'q' and lk != 'q':
        return {**out, 'family': 'corner'}
    if mk == 'q' and lk == 'Z':
        return {**out, 'family': 'single_main', 'q': mq}
    if mk == 'Z' and lk == 'q':
        return {**out, 'family': 'single_later', 'q': lq}
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    o = sub.add_parser('offsets')
    o.add_argument('--features', type=Path, required=True)
    o.add_argument('--model', required=True)
    o.add_argument('--main-key', help='default: layer0 for ibm*, else main')
    o.add_argument('--out', type=Path, required=True)
    o.add_argument('--stats-out', type=Path, help='also save the raw StreamStats JSON (mergeable)')
    o.add_argument('--source', help='provenance string stored in the table')
    c = sub.add_parser('configs')
    c.add_argument('--qs', type=int, nargs='+', required=True)
    c.add_argument('--ks', type=int, nargs='*', default=[])
    c.add_argument('--role', type=int, choices=[1, -1], required=True)
    c.add_argument('--families', nargs='*', default=['corner', 'S', 'axis', 'single'])
    c.add_argument('--no-mean-corners', action='store_true')
    a = p.parse_args(argv)
    if a.cmd == 'offsets':
        st = stats_from_features(a.features)
        mk = a.main_key or ('layer0' if a.model.startswith('ibm') else 'main')
        t = offsets_table(st, a.model, mk, source=a.source or f'{a.features.name} ({len(st.units)} units)')
        save_offsets(t, a.out)
        if a.stats_out:
            save_offsets(st.to_json(), a.stats_out)
        print(json.dumps(dict(model=a.model, d=t['d'], raw_d={k: round(v['raw_d'], 3) for k, v in t['streams'].items()},
                              n_units=t['n_units'])))
    else:
        fam = config_families(a.qs, a.ks, a.role, mean_corners=not a.no_mean_corners)
        print('\n'.join(flat_configs(fam, a.families)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
