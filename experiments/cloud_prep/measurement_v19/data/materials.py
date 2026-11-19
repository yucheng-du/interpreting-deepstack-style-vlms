#!/usr/bin/env python3
"""Runner materials (manifest.jsonl + images) for the first n units of a v19 pool, built by the executed v15
importer (scripts/import_followup_benchmark_v15.py, recorded-plan mode), so rows, prompts (answer_instruction),
references, image conversion and severity realizations are those of the historical materials.

  python materials.py --pool confirm-gqa-attribute.jsonl --n 200 --annotations val_balanced_questions.json \
      --images-root GQA_IMAGES --out OUT [--max-questions-per-unit 2] [--split evaluation] [--material-set NAME] \
      [--severities candidates|start] [--dataset-version 1.2]
  python materials.py --from-manifest HISTORICAL.jsonl --n 5 ... --verify      # rebuild old units, compare fields

GQA: --images-root holds <imageId>.jpg. DocVQA: --images-root holds documents/<page>.png and --annotations is the
official (or mirror-rebuilt) Task-1 JSON. The plan used is kept as OUT/selection-plan.json; OUT/v19_selection.json
records the pool, n and rule. With --verify every rebuilt row is compared field by field with the historical row
of the same id; the report goes to OUT/verify.json.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from scripts import import_followup_benchmark_v15 as imp  # noqa: E402


def sha(path):
    return imp.sha(path)


def read_jsonl(path):
    return [json.loads(l) for l in Path(path).read_text().split('\n') if l.strip()]


def plan_from_pool(pool_rows, n, max_q):
    rows = pool_rows[:n]
    if len(rows) < n: raise ValueError(f'pool has {len(rows)} units, fewer than n={n}')
    task, bench, seed = rows[0]['task'], rows[0]['benchmark'], rows[0]['seed']
    units = [dict(unit_id=r['unit_id'], question_ids=[q['question_id'] for q in (r['questions'][:max_q] if max_q else r['questions'])],
                  image_files=sorted({q['image_file'] for q in r['questions']})) for r in rows]
    return task, bench, seed, units


def plan_from_manifest(rows, n):
    by = defaultdict(list)
    for r in rows: by[r['unit_id']].append(r)
    units = [dict(unit_id=u, question_ids=[r['benchmark']['question_id'] for r in by[u]]) for u in sorted(by)[:n]]
    r0 = rows[0]
    return r0['task'], r0['benchmark']['name'], r0['source'].get('selection_seed'), units


def compare(new_rows, old_rows):
    """Per-field equality of rebuilt rows against historical rows of the same id (nested source/benchmark keys split out)."""
    old = {r['id']: r for r in old_rows}
    eq, diff, examples = Counter(), Counter(), {}
    def flat(r):
        out = {}
        for k, v in r.items():
            if k in ('source', 'benchmark') and isinstance(v, dict):
                for kk, vv in v.items(): out[f'{k}.{kk}'] = vv
            elif k == 'source_image_metadata':
                for kk, vv in v.items():
                    for k3, v3 in vv.items(): out[f'{k}.{"clean" if kk == "a" else "degraded"}.{k3}'] = out.get(f'{k}.{"clean" if kk == "a" else "degraded"}.{k3}', []) + [v3]
            elif k == 'source_images':
                out[k] = sorted(v)
            else:
                out[k] = v
        return out
    for r in new_rows:
        o = old.get(r['id'])
        if o is None: diff['row_missing_in_history'] += 1; continue
        a, b = flat(r), flat(o)
        for k in sorted(set(a) | set(b)):
            if a.get(k) == b.get(k): eq[k] += 1
            else:
                diff[k] += 1; examples.setdefault(k, (str(a.get(k))[:200], str(b.get(k))[:200]))
    return dict(rows=len(new_rows), equal=dict(eq), differ=dict(diff), examples=examples)


def pixels_equal(new_rows, new_root, old_rows, old_root):
    """Decoded-pixel equality of every source image (PNG bytes may differ across Pillow builds)."""
    import numpy as np
    from PIL import Image
    old = {r['id']: r for r in old_rows}; res = Counter(); seen = set()
    for r in new_rows:
        o = old.get(r['id'])
        if o is None: continue
        for lab, rel in r['source_images'].items():
            if rel in seen: continue
            seen.add(rel)
            op = Path(old_root) / o['source_images'][lab]
            if not op.is_file(): res['historical_image_absent'] += 1; continue
            with Image.open(Path(new_root) / rel) as x, Image.open(op) as y:
                a, b = np.asarray(x), np.asarray(y)
            res['equal' if a.shape == b.shape and (a == b).all() else 'different'] += 1
    return dict(res)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument('--pool'); src.add_argument('--from-manifest')
    p.add_argument('--n', type=int, required=True, help='first n units of the pool (or of the manifest, sorted by unit id)')
    p.add_argument('--annotations', required=True); p.add_argument('--images-root', required=True); p.add_argument('--out', required=True)
    p.add_argument('--max-questions-per-unit', type=int, default=2, help='first k questions in the pool order; 0 keeps all')
    p.add_argument('--split', choices=['calibration', 'evaluation'], help='manifest split field (default evaluation; history for --from-manifest)')
    p.add_argument('--material-set'); p.add_argument('--severities', choices=['candidates', 'start'], default='candidates')
    p.add_argument('--dataset-version'); p.add_argument('--verify', action='store_true')
    p.add_argument('--overwrite', action='store_true')
    a = p.parse_args(argv)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    data_sha = sha(a.annotations)
    if a.pool:
        task, bench, seed, units = plan_from_pool(read_jsonl(a.pool), a.n, a.max_questions_per_unit)
        split = a.split or 'evaluation'
        material_set = a.material_set or f'confirm-v19-{task}'
        hist = None
    else:
        hist = read_jsonl(a.from_manifest)
        task, bench, seed, units = plan_from_manifest(hist, a.n)
        split = a.split or hist[0]['split']; material_set = a.material_set or hist[0]['material_set']
    plan = dict(schema=imp.PLAN_SCHEMA, mode='official', benchmark=bench, task=task, split=split, material_set=material_set,
                seed=seed, calibration_fraction=None, max_units=len(units), max_questions_per_unit=a.max_questions_per_unit,
                rule='v19: first n units of the ordered pool (see v19_selection.json)', input=dict(sha256=data_sha),
                counts=dict(selected_units=len(units)), units=units)
    plan_path = out / 'selection-plan.json'
    plan_path.write_text(json.dumps(plan, indent=1) + '\n')
    argv_imp = ['--benchmark', bench, '--annotations', str(a.annotations), '--images-root', str(a.images_root),
                '--unit-ids-file', str(plan_path), '--severities', a.severities, '--output', str(out / 'materials')]
    if a.dataset_version: argv_imp += ['--dataset-version', a.dataset_version]
    elif bench == 'gqa': argv_imp += ['--dataset-version', '1.2']
    if a.overwrite: argv_imp.append('--overwrite')
    meta = imp.main(argv_imp)
    sel = dict(pool=a.pool or a.from_manifest, pool_sha256=sha(a.pool or a.from_manifest), n=a.n, units=len(units),
               questions=sum(len(u['question_ids']) for u in units), max_questions_per_unit=a.max_questions_per_unit,
               annotations=str(a.annotations), annotations_sha256=data_sha, importer_sha256=sha(imp.__file__), import_result=meta)
    (out / 'v19_selection.json').write_text(json.dumps(sel, indent=1) + '\n')
    if a.verify and hist is not None:
        new_rows = read_jsonl(out / 'materials' / 'manifest.jsonl')
        rep = compare(new_rows, hist)
        rep['pixels'] = pixels_equal(new_rows, out / 'materials', hist, Path(a.from_manifest).parent)
        (out / 'verify.json').write_text(json.dumps(rep, indent=1) + '\n')
        print(json.dumps(rep, indent=1))


if __name__ == '__main__':
    main()
