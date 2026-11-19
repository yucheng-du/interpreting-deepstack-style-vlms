#!/usr/bin/env python3
"""Ordered unit pools for v19: confirmation (validation minus history), training (official train), legacy units.

Eligibility is the executed v14/v15 importer's (scripts/import_followup_benchmark_v15.py, imported read-only):
GQA balanced questions, isBalanced, attribute = types.semantic attr; spatial = rel with a *rel* step whose
argument names one of the 11 spatial relations; DocVQA Task-1 questions with the document (ucsf_document_id)
as unit. An excluded image drops its unit; an excluded document drops all of its pages.

Pools ignore the v15 calibration/evaluation hash split: every eligible non-excluded unit is in the pool.
Units are ordered by ascending sha256('seed|unit|task|unit_id') and questions within a unit by ascending
sha256('seed|question|task|question_id') (the v15 draw with a new seed), so the first n units of a pool are
a fixed-seed random sample and a larger n only extends it. Nothing is truncated here.

  python pools.py --gqa-val VAL.json --gqa-train TRAIN.json --docvqa-val val_annotations.json
                  [--docvqa-train train_annotations.json] --gqa-exclusions G.tsv --docvqa-exclusions D.tsv
                  --legacy ROLE=TASK=MANIFEST ... --out DIR [--seed 20261112] [--evidence material mention]
Writes DIR/confirm-<task>.jsonl, DIR/train-<task>.jsonl (GQA), DIR/legacy-units.jsonl, DIR/pools_summary.json.
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from scripts import import_followup_benchmark_v15 as imp  # noqa: E402  (read-only reuse of the executed rules)

RULE = 'v19-pool-1'
SEED = 20261112
TASKS = {'gqa-attribute': ('gqa', 'gqa_attribute', 'attribute'), 'gqa-spatial': ('gqa', 'gqa_spatial', 'spatial'),
         'docvqa': ('docvqa', 'docvqa_document_qa', None)}


def read_exclusions(path, evidence=('material', 'mention')):
    with open(path) as f:
        return {r['id'] for r in csv.DictReader(f, delimiter='\t') if r.get('evidence', 'material') in evidence}


def eligible(data, benchmark, group=None):
    """Eligible question records of the executed importer (before exclusions)."""
    a = SimpleNamespace(benchmark=benchmark, gqa_group=group, gqa_assume_balanced=False,
                        spatial_relations=imp.SPATIAL_RELATIONS, gqa_image_extension='.jpg')
    if benchmark == 'docvqa' and data.get('dataset_split') not in (None, 'val', 'validation'):
        data = dict(data, dataset_split='val', data=[dict(r, data_split='val') for r in data['data']])  # train: same rules
    return imp.official_records(data, a, {})[0]


def order(records, seed, task, excluded_units=frozenset(), excluded_images=frozenset()):
    kept, dropped = imp.apply_exclusions(records, set(excluded_units), set(excluded_images))
    units = defaultdict(list)
    for r in kept: units[r['unit_id']].append(r)
    ranked = sorted(units, key=lambda u: imp.draw(seed, 'unit', task, u))
    out = []
    for rank, u in enumerate(ranked):
        qs = sorted(units[u], key=lambda r: imp.draw(seed, 'question', task, r['qid']))
        out.append((rank, u, qs))
    return out, dict(dropped)


def pool_row(rank, unit, qs, benchmark, task, split, seed):
    row = dict(rank=rank, unit_id=unit, task=task, benchmark=benchmark, split=split, rule=RULE, seed=seed,
               questions=[dict(question_id=r['qid'], query=r['query'], reference_answers=r['references'], image_id=r['image_id'],
                               image_file=r['image_file']) for r in qs], n_questions=len(qs))
    if benchmark == 'gqa': row.update(image_id=qs[0]['image_id'])
    else: row.update(document_id=unit[len(imp.UNIT_PREFIX['docvqa']):], pages=sorted({r['image_id'] for r in qs}))
    return row


def write_jsonl(path, rows):
    with open(path, 'w') as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + '\n')


def stats(pool, cap):
    return dict(units=len(pool), questions=sum(len(q) for _, _, q in pool), questions_cap=sum(min(len(q), cap) for _, _, q in pool), cap=cap)


def legacy_rows(specs, pools_unit_sets, extra=None):
    """One row per (task, unit) of historical validation manifests, with roles and local availability; plus
    'other_history_eligible' rows for excluded units that are eligible for the task but in none of those manifests."""
    units = {}
    for spec in specs:
        role, task, path = spec.split('=', 2)
        path = Path(path)
        for line in open(path):
            if not line.strip(): continue
            r = json.loads(line)
            key = (task, r['unit_id'])
            u = units.setdefault(key, dict(task=task, unit_id=r['unit_id'], image_ids=set(), question_ids=set(), roles=set(),
                                           local_manifests=set(), clean_images_local=True))
            u['image_ids'].add(str(r['benchmark']['image_id'])); u['question_ids'].add(str(r['benchmark']['question_id']))
            u['roles'].add(role); u['local_manifests'].add(str(path))
            img = r.get('source_images', {}).get('a')
            u.setdefault('_img_hits', []).append(bool(img) and (path.parent / img).is_file())
    out = []
    for (task, unit), u in sorted(units.items()):
        roles = sorted(u['roles'])
        v19 = 'v19_calibration' if 'v19_calibration' in roles else 'v19_test' if 'v19_test' in roles else 'other_history'
        out.append(dict(task=task, unit_id=unit, v19_role=v19, history_roles=roles, image_ids=sorted(u['image_ids']),
                        question_ids=sorted(u['question_ids']), local_manifests=sorted(u['local_manifests']),
                        clean_image_unpacked=any(u['_img_hits']), in_confirm_pool=unit in pools_unit_sets.get(task, set())))
    for task, extra_units in sorted((extra or {}).items()):
        for unit in sorted(extra_units - {k[1] for k in units if k[0] == task}):
            out.append(dict(task=task, unit_id=unit, v19_role='other_history_eligible', history_roles=[], image_ids=[], question_ids=[],
                            local_manifests=[], clean_image_unpacked=False, in_confirm_pool=unit in pools_unit_sets.get(task, set())))
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--gqa-val', required=True); p.add_argument('--gqa-train')
    p.add_argument('--docvqa-val', required=True); p.add_argument('--docvqa-train')
    p.add_argument('--gqa-exclusions', required=True); p.add_argument('--docvqa-exclusions', required=True)
    p.add_argument('--evidence', nargs='+', default=['material', 'mention'], help='exclusion evidence kinds applied')
    p.add_argument('--legacy', nargs='*', default=[], help='ROLE=TASK=MANIFEST of historical validation materials')
    p.add_argument('--legacy-file', help='text file with one ROLE=TASK=MANIFEST per line (# comments)')
    p.add_argument('--seed', type=int, default=SEED); p.add_argument('--cap', type=int, default=2, help='questions/unit reported')
    p.add_argument('--min-units', type=int, default=200, help='warn below this many confirmation units')
    p.add_argument('--out', required=True)
    a = p.parse_args(argv)
    if a.legacy_file:
        a.legacy += [l.split('#')[0].strip() for l in Path(a.legacy_file).read_text().splitlines() if l.split('#')[0].strip()]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    gx, dx = read_exclusions(a.gqa_exclusions, a.evidence), read_exclusions(a.docvqa_exclusions, a.evidence)
    summary = dict(rule=RULE, seed=a.seed, evidence=a.evidence, excluded_gqa_images=len(gx), excluded_docvqa_documents=len(dx),
                   inputs={k: str(v) for k, v in vars(a).items() if k.startswith(('gqa_', 'docvqa_')) and v}, tasks={}, checks={}, warnings=[])
    checks = summary['checks']
    gval = json.load(open(a.gqa_val))
    val_images = {str(r['imageId']) for r in gval.values()}
    confirm_units, hist_eligible = {}, {}
    for name in ('gqa-attribute', 'gqa-spatial'):
        bench, task, group = TASKS[name]
        recs = eligible(gval, bench, group)
        pool, dropped = order(recs, a.seed, task, excluded_images=gx)
        write_jsonl(out / f'confirm-{name}.jsonl', [pool_row(r, u, q, bench, task, 'validation', a.seed) for r, u, q in pool])
        hist_units = {r['unit_id'] for r in recs} - {u for _, u, _ in pool}
        summary['tasks'][name] = dict(eligible_units=len({r['unit_id'] for r in recs}), eligible_questions=len(recs),
                                      historical_eligible_units=len(hist_units), confirm=stats(pool, a.cap), dropped=dropped)
        confirm_units[name] = {u for _, u, _ in pool}; hist_eligible[name] = hist_units
        checks[f'{name}_confirm_disjoint_from_exclusions'] = not ({u.split('-')[-1] for u in confirm_units[name]} & gx)
    del gval
    dval = json.load(open(a.docvqa_val))
    recs = eligible(dval, 'docvqa')
    excl_units = {imp.UNIT_PREFIX['docvqa'] + d for d in dx}
    pool, dropped = order(recs, a.seed, 'docvqa_document_qa', excluded_units=excl_units)
    write_jsonl(out / 'confirm-docvqa.jsonl', [pool_row(r, u, q, 'docvqa', 'docvqa_document_qa', 'validation', a.seed) for r, u, q in pool])
    val_docs = {r['unit_id'] for r in recs}
    summary['tasks']['docvqa'] = dict(eligible_units=len(val_docs), eligible_questions=len(recs),
                                      historical_eligible_units=len(val_docs & excl_units), confirm=stats(pool, a.cap), dropped=dropped,
                                      confirm_pages=len({x for _, _, q in pool for x in (r['image_id'] for r in q)}))
    confirm_units['docvqa'] = {u for _, u, _ in pool}; hist_eligible['docvqa'] = val_docs & excl_units
    checks['docvqa_confirm_disjoint_from_exclusions'] = not (confirm_units['docvqa'] & excl_units)
    for name, units in confirm_units.items():
        if len(units) < a.min_units: summary['warnings'].append(f'{name}: only {len(units)} confirmation units (< {a.min_units})')
    if a.gqa_train:
        gtrain = json.load(open(a.gqa_train))
        train_images = {str(r['imageId']) for r in gtrain.values()}
        checks['gqa_train_val_images_disjoint_in_source'] = not (train_images & val_images)
        for name in ('gqa-attribute', 'gqa-spatial'):
            bench, task, group = TASKS[name]
            recs = eligible(gtrain, bench, group)
            pool, dropped = order(recs, a.seed, task, excluded_images=gx | val_images)
            write_jsonl(out / f'train-{name}.jsonl', [pool_row(r, u, q, bench, task, 'train', a.seed) for r, u, q in pool])
            imgs = {u.split('-')[-1] for _, u, _ in pool}
            summary['tasks'][name].update(train=stats(pool, a.cap), train_eligible_units=len({r['unit_id'] for r in recs}), train_dropped=dropped)
            checks[f'{name}_train_disjoint_from_val_images'] = not (imgs & val_images)
            checks[f'{name}_train_disjoint_from_exclusions'] = not (imgs & gx)
        del gtrain
    if a.docvqa_train:
        dtrain = json.load(open(a.docvqa_train))
        tdocs = {imp.UNIT_PREFIX['docvqa'] + r['ucsf_document_id'] for r in dtrain['data']}
        clean = tdocs - val_docs - excl_units
        summary['tasks']['docvqa'].update(train_documents=len(tdocs), train_documents_shared_with_val=len(tdocs & val_docs),
                                          train_documents_in_history=len(tdocs & excl_units), train_documents_isolated=len(clean),
                                          train_questions_isolated=sum(1 for r in dtrain['data'] if imp.UNIT_PREFIX['docvqa'] + r['ucsf_document_id'] in clean))
        checks['docvqa_isolated_train_docs_disjoint_from_val_and_history'] = not (clean & (val_docs | excl_units))
    legacy = legacy_rows(a.legacy, confirm_units, hist_eligible)
    write_jsonl(out / 'legacy-units.jsonl', legacy)
    checks['legacy_disjoint_from_confirm'] = not any(r['in_confirm_pool'] for r in legacy)
    summary['legacy'] = {t: dict(Counter(r['v19_role'] for r in legacy if r['task'] == t),
                                 clean_image_unpacked=sum(r['clean_image_unpacked'] for r in legacy if r['task'] == t))
                         for t in sorted({r['task'] for r in legacy})}
    summary['checks_passed'] = all(checks.values())
    (out / 'pools_summary.json').write_text(json.dumps(summary, indent=1) + '\n')
    print(json.dumps(summary, indent=1))
    if not summary['checks_passed']: sys.exit(2)


if __name__ == '__main__':
    main()
