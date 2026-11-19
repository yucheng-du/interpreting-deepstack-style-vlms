#!/usr/bin/env python3
"""Prepare measurement-v3 samples without loading models or using model outcomes."""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import os as _path_os
DATA_ROOT = Path(_path_os.environ.get("DS_DATA_ROOT", "work")).expanduser().resolve()
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from deepstack.slot_swap import slot_pair, stable_seed

ROOT = (DATA_ROOT / 'runs/measurement-v3')
FULL = (DATA_ROOT / 'data/blink/val_seed20260926_all.jsonl')
OLD64 = (DATA_ROOT / 'runs/readout-controls-v2/manifests/all64.jsonl')
SOURCE_SUMMARY = (DATA_ROOT / 'runs/qwen4b-full1901-v1/analysis/all1901/summary.json')
SEED = 20261003
TASKS = ('Art_Style', 'Functional_Correspondence', 'Multi-view_Reasoning',
         'Relative_Reflectance', 'Visual_Correspondence', 'Counting', 'IQ_Test',
         'Object_Localization', 'Semantic_Correspondence', 'Visual_Similarity',
         'Forensic_Detection', 'Jigsaw', 'Relative_Depth', 'Spatial_Relation')
SLOT_TASKS = ('Visual_Similarity', 'Forensic_Detection')


def load_rows(path):
    with Path(path).open() as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    ids = [r['id'] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f'duplicate IDs: {path}')
    return rows


def interleave(groups, tasks):
    return [groups[task][i] for i in range(max(map(len, groups.values()), default=0))
            for task in tasks if i < len(groups[task])]


def select_readout(full, old64, seed=SEED):
    """Selection depends on task/source IDs and the prior panel, never outcomes."""
    full_by_id = {r['id']: r for r in full}
    if len(full_by_id) != len(full) or len({r['id'] for r in old64}) != len(old64):
        raise ValueError('duplicate source IDs')
    for row in old64:
        if row['id'] not in full_by_id:
            raise ValueError(f"old panel ID absent from full source: {row['id']}")
        # Old added readout_control metadata is retained, but source content must agree.
        if any(row.get(k) != v for k, v in full_by_id[row['id']].items()):
            raise ValueError(f"old panel source differs: {row['id']}")
    old_ids = {r['id'] for r in old64}
    prior_counts = Counter()
    pilot = []
    for row in old64:
        if prior_counts[row['task']] < 2:
            pilot.append(row)
        prior_counts[row['task']] += 1
    groups = {}
    for task in TASKS:
        eligible = sorted((r for r in full if r['task'] == task and r['id'] not in old_ids),
                          key=lambda r: (r['source']['row_index'], r['id']))
        random.Random(f'{seed}:{task}').shuffle(eligible)
        if len(eligible) < 10 or prior_counts[task] < 2:
            raise ValueError(f'not enough rows for {task}')
        groups[task] = eligible[:10]
    return pilot, list(old64), interleave(groups, TASKS)


def annotate(rows, panel, *, exploratory=True, **metadata):
    result = []
    for index, source in enumerate(rows):
        row = deepcopy(source)
        row['measurement_v3'] = {'panel': panel, 'panel_index': index,
                                 'exploratory': exploratory, **metadata}
        result.append(row)
    return result


def baseline_grids(summary_path, ids):
    """Read recorded processor grids only; no scores or labels influence assignment."""
    summary = json.loads(Path(summary_path).read_text())
    result = {}
    for source in summary['provenance']['sources']:
        with Path(source['path']).open() as handle:
            for line in handle:
                row = json.loads(line)
                if row.get('id') in ids and row.get('mask') == '111':
                    grid = row['image_grid_thw']
                    if row['id'] in result and result[row['id']] != grid:
                        raise ValueError(f"conflicting baseline grid: {row['id']}")
                    result[row['id']] = grid
    if set(result) != set(ids):
        raise ValueError(f'missing grid IDs: {sorted(set(ids)-set(result))}')
    return result


def select_slots(full):
    # Preserve the existing full-manifest order within each task. No score is read.
    groups = {task: [r for r in full if r['task'] == task] for task in SLOT_TASKS}
    return interleave(groups, SLOT_TASKS), {r['id'] for task in SLOT_TASKS for r in groups[task][:2]}


def assign_donors(rows, grids, seed=SEED):
    """Fix one role-preserving, exact-grid donor per item; outputs are not inspected."""
    for row in rows:
        hashes = {m['sha256'] for m in row['image_metadata']}
        eligible = sorted(other['id'] for other in rows
                          if other['task'] == row['task'] and other['id'] != row['id']
                          and grids[other['id']] == grids[row['id']]
                          and not hashes.intersection(m['sha256'] for m in other['image_metadata']))
        if not eligible:
            raise ValueError(f"no disjoint exact-grid donor for {row['id']}")
        row['v3_donor_id'] = random.Random(stable_seed(seed, row['id'], 'unrelated_donor')).choice(eligible)
        row['v3_slot_plan'] = {'image_grid_thw': grids[row['id']],
                               'swap_pair': list(slot_pair(row, seed=seed)), 'seed': seed}
    return rows


def file_info(path, rows=None):
    info = {'path': str(path), 'sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest()}
    if rows is not None:
        info.update(n=len(rows), by_task=dict(Counter(r['task'] for r in rows)), ids=[r['id'] for r in rows])
    return info


def write_once(path, content):
    """An identical rerun is harmless; a changed design requires a new destination."""
    path = Path(path)
    if path.exists():
        if path.read_text() != content:
            raise ValueError(f'refusing to replace differing manifest: {path}')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as handle:
        handle.write(content)


def prepare(root=ROOT, full_path=FULL, old64_path=OLD64, summary_path=SOURCE_SUMMARY):
    full, old64 = load_rows(full_path), load_rows(old64_path)
    if len(full) != 1901 or len(old64) != 64:
        raise ValueError('expected original1901 and readout64')
    pilot, reused, additional = select_readout(full, old64)
    panels = {
        'readout_pilot28': annotate(pilot, 'readout_pilot28', sample_status='reused_readout_exploratory',
                                   selection='first two per task in existing readout64 order'),
        'readout64': annotate(reused, 'readout64', sample_status='reused_readout_exploratory',
                              selection='complete existing readout64, unchanged order'),
        'additional140': annotate(additional, 'additional140', sample_status='new_to_readout_exploratory',
                                  selection='ten per task, excluding readout64; task-local seeded shuffle', seed=SEED),
    }
    slots, smoke_ids = select_slots(full)
    if Counter(r['task'] for r in slots) != {'Visual_Similarity': 135, 'Forensic_Detection': 132}:
        raise ValueError('expected Visual_Similarity135 and Forensic_Detection132')
    grids = baseline_grids(summary_path, {r['id'] for r in slots})
    slots = annotate(slots, 'slot267', sample_status='reused_full1901_exploratory',
                     selection='all two-task source rows, task-interleaved in existing full-manifest order')
    for row in slots:
        row['measurement_v3']['technical_smoke'] = row['id'] in smoke_ids
        row['measurement_v3']['recorded_image_grid_thw'] = grids[row['id']]
    assign_donors(slots, grids)
    panels.update(slot267=slots,
                  slot_smoke4=[r for r in slots if r['id'] in smoke_ids],
                  slot_remaining263=[r for r in slots if r['id'] not in smoke_ids])
    directory = Path(root) / 'manifests'
    plan = {'schema': 'measurement-v3-manifests-v1', 'seed': SEED,
            'phase': 'exploratory; no subset is unseen confirmatory data',
            'sources': {'full1901': file_info(full_path), 'old_readout64': file_info(old64_path),
                        'grid_summary': file_info(summary_path)},
            'slot_donor_assignment': 'v3_donor_id; same task and recorded full grid, distinct ID and disjoint image SHAs; fixed seed per item; corresponding image roles preserved',
            'manifests': {}}
    for name, rows in panels.items():
        path = directory / f'{name}.jsonl'
        write_once(path, ''.join(json.dumps(r, sort_keys=True, ensure_ascii=False) + '\n' for r in rows))
        plan['manifests'][name] = file_info(path, rows)
    write_once(directory / 'plan.json', json.dumps(plan, ensure_ascii=False, sort_keys=True, indent=2) + '\n')
    return plan


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    plan = prepare(args.root)
    print(json.dumps({name: {'n': data['n'], 'path': data['path']}
                      for name, data in plan['manifests'].items()}, indent=2))
