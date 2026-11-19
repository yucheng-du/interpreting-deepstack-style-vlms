"""Paired config differences with paired cluster-bootstrap 95% CIs (CPU only)."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import random
import statistics


def read(paths):
    rows = []
    for p in paths:
        rows += [json.loads(l) for l in open(p) if l.strip()]
    return rows


def paired(rows, a, b, n_boot=4000, seed=20261110):
    A = {r['id']: r for r in rows if r['config'] == a}
    B = {r['id']: r for r in rows if r['config'] == b}
    ids = sorted(set(A) & set(B))
    by = {}
    for i in ids:
        by.setdefault(A[i]['cluster_id'], []).append(A[i]['correct'] - B[i]['correct'])
    means = [statistics.mean(v) for v in by.values()]
    n = len(means)
    rng = random.Random(seed)
    boots = sorted(statistics.mean(rng.choices(means, k=n)) for _ in range(n_boot))
    return dict(a=a, b=b, n_questions=len(ids), n_clusters=n, diff_cluster=round(statistics.mean(means), 4),
                ci95_low=round(boots[int(.025 * n_boot)], 4), ci95_high=round(boots[int(.975 * n_boot) - 1], 4),
                diff_question=round(statistics.mean(d for v in by.values() for d in v), 4),
                discordant_a_right=sum(1 for i in ids if A[i]['correct'] > B[i]['correct']),
                discordant_b_right=sum(1 for i in ids if A[i]['correct'] < B[i]['correct']))


def qwen_pairs():
    out = [(f'q{q}|Z', f'q{q}|q{q}', '(q,Z)-(q,q)') for q in (-4, 0, 2, 4, 6, 8)]
    out += [(f'q{q}|q{q-4}', f'q{q}|q{q}', '(q,q-4)-(q,q)') for q in (0, 2, 4, 6, 8)]
    out += [('C|Z', 'C|C', '(C,Z)-(C,C)')]
    # (12,12) is run in addition to the 29 listed configs: at QP 12 a few later3 coefficients on test
    # images are nonzero, so (12,Z) is not an exact stand-in.
    out += [(f'q{q}|C', f'q{q}|q{q}', '(q,C)-(q,q)') for q in (4, 8, 12)]
    return out


def deepstack_pairs():
    out = [(f'q{q}|Z', f'q{q}|q{q}', '(q,Z)-(q,q)') for q in (-4, -2, -1, 0)]
    out += [('q-2|Z', 'q0|q0', 'main-only vs shared'), ('q-1|Z', 'q0|q0', 'main-only vs shared'),
            ('q-4|Z', 'q0|q0', 'main-only vs shared'), ('q-2|Z', 'q-2|q-2', 'main-only vs shared'),
            ('q-2|Z', 'C|C', 'main-only vs CC'), ('q-1|Z', 'C|C', 'main-only vs CC')]
    out += [(f'Z|q{q}', f'q{q}|q{q}', '(Z,q)-(q,q)') for q in (0, 2, 8)]
    out += [(f'q{q}|C', 'Z|C', '(q,C)-(Z,C)') for q in (0, 2, 8)]
    out += [(f'q{q}|C', f'q{q}|q{q}', '(q,C)-(q,q)') for q in (0, 2, 8)]
    out += [('C|Z', 'C|C', 'corner'), ('Z|C', 'C|C', 'corner'), ('Z|C', 'Z|Z', 'corner'), ('C|Z', 'Z|Z', 'corner')]
    return out


def granite_pairs():
    out = [(f'M|q{q}', f'q{q}|q{q}', '(M,q)-(q,q)') for q in (-12, -8, -4, 0)]
    out += [('M|q-8', 'q-4|q-4', 'mean-main vs shared'), ('M|q-4', 'q0|q0', 'mean-main vs shared'),
            ('M|q-4', 'q4|q4', 'mean-main vs shared'), ('M|q-12', 'q-8|q-8', 'mean-main vs shared'),
            ('M|q0', 'q4|q4', 'mean-main vs shared')]
    out += [(f'M|q{q}', f'C|q{q}', '(M,q)-(C,q)') for q in (-8, -4, 0)]
    out += [('q4|q-8', 'q-4|q-4', 'mixed'), ('q4|q-8', 'M|q-8', 'mixed'), ('q4|q-4', 'M|q-4', 'mixed')]
    out += [(f'q{q}|C', f'q{q}|q{q}', '(q,C)-(q,q)') for q in (0, 4)]
    out += [(f'C|q{q}', f'q{q}|q{q}', '(C,q)-(q,q)') for q in (0, 4)]
    out += [('M|C', 'C|C', 'corner'), ('C|Z', 'C|C', 'corner'), ('C|M', 'C|C', 'corner'), ('Z|C', 'C|C', 'corner'),
            ('M|C', 'M|M', 'corner'), ('C|M', 'M|M', 'corner')]
    return out


def granite_pairs_extra():
    # CPU comparisons added after block A, followed by the block F comparisons.
    return [('q4|q-8', 'q-8|q-8', 'main q4 family'), ('q4|q-8', 'C|C', 'main q4 family'), ('q4|q-4', 'q0|q0', 'main q4 family'),
            ('q4|q-4', 'q-4|q-4', 'main q4 family'), ('q4|C', 'C|C', 'vs CC'), ('q0|C', 'C|C', 'vs CC'),
            ('q-8|q-8', 'C|C', 'vs CC'), ('C|q-8', 'C|C', 'vs CC'),
            ('q4|q-12', 'q-12|q-12', 'block F'), ('q4|q-12', 'q-8|q-8', 'block F'), ('q4|q-6', 'q-4|q-4', 'block F'),
            ('q4|q-6', 'q0|q0', 'block F'), ('q4|q-6', 'q4|q-8', 'block F'), ('q4|q0', 'q0|q0', 'block F'),
            ('q4|q0', 'q4|q4', 'block F')]


def granite_spatial_pairs():
    return [('q4|q-8', 'q-8|q-8', 'block G'), ('q4|q-8', 'q-4|q-4', 'block G'), ('q4|q-4', 'q0|q0', 'block G'),
            ('q4|q-4', 'q-4|q-4', 'block G'), ('M|q-8', 'q-4|q-4', 'block G'), ('M|q-8', 'q4|q-8', 'block G'),
            ('q4|q-8', 'C|C', 'block G'), ('q-8|q-8', 'C|C', 'block G')]


def deepstack_pairs_b():
    return [('Z|q-1', 'q-4|Z', 'block B'), ('Z|q-1', 'q0|q0', 'block B'), ('Z|q-2', 'q-2|q-2', 'block B'),
            ('Z|q-2', 'q-1|q-1', 'block B'), ('Z|q-2', 'C|C', 'block B'), ('q-4|Z', 'q-1|q-1', 'block B'),
            ('q-4|Z', 'C|C', 'block B'), ('q8|q0', 'q-2|Z', 'block B'), ('q8|q0', 'Z|q0', 'block B'),
            ('q8|q4', 'Z|q4', 'block B'), ('q8|q4', 'Z|q8', 'block B')]


def deepstack_spatial_pairs():
    return [('q-2|Z', 'q0|q0', 'spatial'), ('q-4|Z', 'q0|q0', 'spatial'), ('q-4|Z', 'q-1|q-1', 'spatial'),
            ('q-2|Z', 'q-2|q-2', 'spatial'), ('Z|q0', 'q0|q0', 'spatial'), ('Z|q-2', 'q-2|q-2', 'spatial'),
            ('q-4|Z', 'C|C', 'spatial'), ('Z|q-2', 'C|C', 'spatial')]


def write(splits, csv_path, pairs=None):
    rows = []
    for split, paths in splits.items():
        obs = read(paths)
        for a, b, fam in (pairs or qwen_pairs()):
            if not any(o['config'] == a for o in obs) or not any(o['config'] == b for o in obs):
                continue
            rows.append(dict(split=split, family=fam, **paired(obs, a, b)))
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    return rows
