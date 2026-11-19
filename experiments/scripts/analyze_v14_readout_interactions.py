#!/usr/bin/env python3
"""Reuse accepted v14 estimates and pair readout contrasts by template/photo ID.

No model execution, observation rescoring, archive mutation or full-pack audit.
Only the nine accepted Line1/marginal analysis JSON members and the final summary
are read. The original estimates are copied exactly; new estimates subtract
archived per-unit values before calculating descriptive Student-t intervals.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tarfile
import tempfile

import scipy
from scipy.stats import t as student_t


DEFAULT_RUN = (Path(os.environ.get('DS_DATA_ROOT', 'work')) / 'runs/measurement-v14-cloud-20261028')
DEFAULT_OUTPUT = (Path(os.environ.get('DS_DATA_ROOT', 'work')) / 'analysis/v14-integration-20261029')
MODEL_ORDER = ['qwen8b', 'qwen32b', 'ibm4b', 'deepstack7b']
READOUT_ORDER = ['direct', 'identity', 'cyclic']
CONTRASTS = [('cyclic', 'direct'), ('identity', 'direct'), ('cyclic', 'identity')]
UNIT_ARRAY_KEYS = {'unit_values'}


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def strip_units(value):
    if isinstance(value, dict):
        return {k: strip_units(v) for k, v in value.items() if k not in UNIT_ARRAY_KEYS}
    if isinstance(value, list):
        return [strip_units(v) for v in value]
    return value


def read_member(pack, member, provenance):
    """Read a specified small member; do not extract the archive to disk."""
    proc = subprocess.Popen(['zstd', '-dc', str(pack)], stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    raw = None
    try:
        with tarfile.open(fileobj=proc.stdout, mode='r|') as archive:
            for entry in archive:
                if entry.name == member:
                    if not entry.isfile():
                        raise ValueError(f'Not a regular member: {pack}:{member}')
                    raw = archive.extractfile(entry).read()
        stderr = proc.stderr.read()
        if proc.wait() != 0:
            raise RuntimeError(f'zstd failed for {pack}: {stderr.decode(errors="replace")}')
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
    if raw is None:
        raise FileNotFoundError(f'{pack}:{member}')
    manifest_path = pack.with_name(pack.name.removesuffix('.tar.zst') + '.manifest.json')
    manifest = json.loads(manifest_path.read_text())
    expected = next(x for x in manifest['files'] if x['path'] == member)
    actual_hash = digest(raw)
    if actual_hash != expected['sha256'] or len(raw) != expected['bytes']:
        raise ValueError(f'Selected member differs from its accepted manifest: {member}')
    provenance.append({'pack': str(pack), 'member': member,
                       'member_sha256': actual_hash, 'member_bytes': len(raw),
                       'manifest': str(manifest_path),
                       'pack_sha256_from_existing_manifest': manifest.get('pack_sha256'),
                       'verification_scope': 'selected analysis member only; existing pack receipt reused'})
    return json.loads(raw)


def group_key(group):
    return tuple(group[k] for k in ('model', 'task', 'kind', 'response_format', 'mapping'))


def readout_key(group):
    return 'direct' if group['response_format'] == 'direct' else group['mapping']


def t_summary(values):
    """Descriptive t interval on one scalar per distinct inference unit."""
    vals = list(values)
    if not all(math.isfinite(x) for x in vals):
        raise ValueError('Nonfinite per-unit effect')
    n = len(vals)
    mean = statistics.mean(vals) if n else None
    se = statistics.stdev(vals) / math.sqrt(n) if n >= 2 else None
    se_zero = se == 0 if se is not None else False
    q = float(student_t.ppf(.975, n - 1)) if n >= 2 else None
    ci = [mean - q * se, mean + q * se] if n >= 2 and not se_zero else None
    return {'n': n, 'mean': mean, 'se': se, 'ci95_t': ci,
            't_quantile_975': q, 'df': n - 1 if n else None,
            'se_zero': se_zero, 'n_lt_5': n < 5,
            'positive': sum(x > 0 for x in vals), 'zero': sum(x == 0 for x in vals),
            'negative': sum(x < 0 for x in vals)}


def paired_difference(positive, negative):
    """Pair by ID, never row order or subtraction of aggregate intervals."""
    p, n = positive['unit_values'], negative['unit_values']
    if len(p) != positive['n'] or len(n) != negative['n']:
        raise ValueError('Archived n does not match unique per-unit values')
    ids = sorted(set(p) & set(n))
    diffs = {unit: p[unit] - n[unit] for unit in ids}
    result = t_summary(diffs.values())
    result.update(unit_ids=ids, unit_values=diffs,
                  positive_only_unit_ids=sorted(set(p) - set(n)),
                  negative_only_unit_ids=sorted(set(n) - set(p)),
                  positive_source_denominators=positive.get('denominators'),
                  negative_source_denominators=negative.get('denominators'))
    return result


def exactly_one(items, description):
    if len(items) != 1:
        raise ValueError(f'Expected one {description}; got {len(items)}')
    return items[0]


def metrics_for_pairing(analysis, family):
    if family == 'line1':
        output = {}
        for policy in ('accuracy_invalid_as_incorrect', 'accuracy_valid_only'):
            recovery = analysis['recovery_contrasts'][policy]
            for key, name in [('clean_later_benefit_with_degraded_main', 'later_recovery'),
                              ('quality_interaction', 'recovery_quality_interaction')]:
                output[(name, policy)] = exactly_one(recovery[key], key)
        return output
    marginal = exactly_one(analysis['custom_paired_comparisons'], 'Granite marginal')
    assert marginal['name'] == 'M21_given_15_18'
    return {('granite_M21_given_15_18', 'valid_endpoints_only'):
            marginal['target_minus_control']}


def all_planned_queries_used(stat):
    den = stat.get('denominators', {})
    if 'planned_queries' in den:
        return den['planned_queries'] == den['queries_used']
    return den.get('planned_interventions') == den.get('interventions_used')


def build_pair_contrasts(groups):
    families = {}
    for group in groups:
        family = 'granite_marginal' if 'granite-marginal' in group['job'] else 'line1'
        key = (group['model'], group['material'], group['task'], group['kind'], family)
        block = families.setdefault(key, {})
        fmt = readout_key(group)
        if fmt in block:
            raise ValueError(f'Duplicate analysis group: {key}/{fmt}')
        block[fmt] = group
    contrasts = []
    for key, block in sorted(families.items()):
        if set(block) != set(READOUT_ORDER):
            raise ValueError(f'Incomplete readout family: {key}')
        for positive, negative in CONTRASTS:
            p = metrics_for_pairing(block[positive]['analysis'], key[4])
            n = metrics_for_pairing(block[negative]['analysis'], key[4])
            if p.keys() != n.keys():
                raise ValueError('Mismatched cross-readout estimands')
            for (metric, policy), p_stat in p.items():
                n_stat = n[(metric, policy)]
                common_all = all_planned_queries_used(p_stat) and all_planned_queries_used(n_stat)
                record = dict(zip(('model', 'material', 'task', 'kind', 'family'), key))
                record.update(metric=metric, invalid_policy=policy,
                              contrast=f'{positive}-{negative}', positive_readout=positive,
                              negative_readout=negative,
                              positive_source_group=block[positive]['group_id'],
                              negative_source_group=block[negative]['group_id'],
                              all_planned_within_unit_observations_used=common_all,
                              pairing_scope=('all planned within-unit observations; pair on unit ID'
                                             if common_all else
                                             'intersection of unit IDs with archived valid-endpoint means; '
                                             'within-unit query/intervention membership may differ by readout'))
                record.update(paired_difference(p_stat, n_stat))
                contrasts.append(record)
    return contrasts


def estimate_without_units(value):
    return strip_units(value)


def panel_row(group):
    a = group['analysis']
    row = {k: group[k] for k in ('group_id', 'job', 'model', 'material', 'task', 'kind',
                                 'response_format', 'mapping')}
    row['readout'] = readout_key(group)
    row['native'] = estimate_without_units(a['native_clean_reference_accuracy']['accuracy_invalid_as_incorrect'])
    row['recovery_cells'] = {
        x['main_quality'] + '__' + x['later_quality']: estimate_without_units(x)
        for x in a['recovery_clean_reference_accuracy_invalid_as_incorrect']}
    row['R'] = {x['main_quality']: estimate_without_units(x) for x in a['selectivity_R']}
    row['quality_selectivity_interaction'] = [estimate_without_units(x) for x in a['selectivity_interaction']]
    row['target_control_effects'] = estimate_without_units(a['conflict_query_effects'])
    row['recovery'] = estimate_without_units(a['recovery_contrasts']['accuracy_invalid_as_incorrect'])
    row['marginal'] = estimate_without_units(a['custom_paired_comparisons'])
    row['diagnostics'] = copy.deepcopy(a['diagnostics'])
    row['source'] = copy.deepcopy(group['source'])
    return row


def build_panels(groups, contrasts):
    rows = [panel_row(g) for g in groups]
    return {
        'color_recovery': [r for r in rows if r['task'] == 'color' and r['kind'] == 'generation' and 'line1' in r['job']],
        'color_conflict_quality': [r for r in rows if r['task'] == 'color' and 'line1' in r['job']],
        'natural_conflict_and_recovery': [r for r in rows if r['material'] == 'natural'],
        'granite_conditional_marginal': [r for r in rows if 'granite-marginal' in r['job']],
        'paired_color_recovery': [c for c in contrasts if c['task'] == 'color' and c['kind'] == 'generation'
                                  and c['family'] == 'line1' and c['invalid_policy'] == 'accuracy_invalid_as_incorrect'],
        'paired_granite_marginal': [c for c in contrasts if c['family'] == 'granite_marginal'],
    }


def atomic_json(path, value):
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def atomic_text(path, text):
    fd, name = tempfile.mkstemp(prefix='.' + path.name + '.', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            stream.write(text)
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def number(stat):
    if not stat or stat.get('mean') is None:
        return 'NA'
    ci = stat.get('ci95_t')
    if ci is None:
        return f"{stat['mean']:.4f} (SE=0)" if stat.get('se_zero') else f"{stat['mean']:.4f} (CI unavailable)"
    return f"{stat['mean']:.4f} [{ci[0]:.4f}, {ci[1]:.4f}]"


def build_notes(panels, checks):
    lines = [
        '# v14 readout interactions: copied estimates and new paired contrasts', '',
        'Source analyses and final-summary estimates were reused without inference or rescoring. '
        'New intervals subtract archived per-template/photo values first, then use Student-t 95% '
        'intervals (df=n−1); they are exploratory, descriptive and unadjusted. Zero SE is explicit '
        'and does not imply equivalence outside the sample.', '',
        'Recovery primary policy is accuracy_invalid_as_incorrect on all planned queries. '
        'accuracy_valid_only is retained as a named sensitivity. Conflict and Granite marginal '
        'retain their accepted valid-endpoint policy. Paired sensitivity rows with incomplete '
        'endpoints use the intersection of unit IDs, but may average different valid queries '
        'within those units; each such row is explicitly flagged.', '',
        '## Synthetic color: all generation readouts', '',
        '| Model | Readout | CC | CD | DC | DD | Later recovery | Recovery interaction |',
        '|---|---|---:|---:|---:|---:|---|---|',
    ]
    for row in panels['color_recovery']:
        cells = row['recovery_cells']
        vals = [f"{cells[k]['mean']:.4f}" for k in ('clean__clean', 'clean__degraded', 'degraded__clean', 'degraded__degraded')]
        r = row['recovery']
        lines.append('| ' + ' | '.join([row['model'], row['readout'], *vals,
                                      number(r['clean_later_benefit_with_degraded_main'][0]),
                                      number(r['quality_interaction'][0])]) + ' |')
    lines += ['', '## Directly paired differences in color recovery', '',
              '| Model | Readout difference | Metric | n templates | Estimate [95% t CI] |',
              '|---|---|---|---:|---|']
    for c in panels['paired_color_recovery']:
        lines.append(f"| {c['model']} | {c['contrast']} | {c['metric']} | {c['n']} | {number(c)} |")
    lines += ['', '## Granite conditional marginal: all readouts', '',
              '| Task | Kind | Readout | n templates | Target−control conditional marginal |',
              '|---|---|---|---:|---|']
    for row in panels['granite_conditional_marginal']:
        s = row['marginal'][0]['target_minus_control']
        lines.append(f"| {row['task']} | {row['kind']} | {row['readout']} | {s['n']} | {number(s)} |")
    lines += ['', '## Scope and verification', '',
              f"- {checks['copied_groups']} copied groups match the accepted final-summary JSON exactly after removing unit arrays.",
              f"- {checks['paired_contrasts']} new paired contrasts; no CI is formed by subtracting aggregate CI bounds.",
              '- Synthetic inference units are the 32 shared templates; natural units are the 64 photos. Queries, edits and readouts do not increase n.',
              '- Δ_entry R (historical main−later) and Δ_quality R (v14 degraded−normal) are different estimands.',
              '- Three readouts are direct, numeric-label identity and numeric-label cyclic. These are format conditions, not independent replications.',
              '- Empty arrays or n=0 remain unavailable; missing marginal recovery/native cells are not filled with zero.',
              '- summary.json preserves all source fields and unit IDs; paper_numbers.json is a presentation subset, not a newly selected primary analysis.',
              '- Input member SHA checks are bounded to the ten small JSON sources; existing full-pack extraction receipts are reused.', '']
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, default=DEFAULT_RUN)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    sources = []
    final = read_member(args.run_root / 'packs/analysis-final-results-summary.run-1.tar.zst',
                        'run-1/final-results-summary.json', sources)
    selected = [g for g in final['groups'] if 'line1-batched' in g['job'] or 'granite-marginal-batched' in g['job']]
    if len(selected) != 144:
        raise ValueError(f'Expected 144 Line1/marginal groups, got {len(selected)}')
    final_by_job = {}
    for g in selected:
        key = group_key(g)
        if key in final_by_job.setdefault(g['job'], {}):
            raise ValueError('Duplicate final-summary group')
        final_by_job[g['job']][key] = g
    groups = []
    copied = 0
    for job, expected in sorted(final_by_job.items()):
        member = f'run-1/{job}.json'
        analysis = read_member(args.run_root / f'packs/analysis-{job}.run-1.tar.zst', member, sources)
        if len(analysis['results']) != len(expected):
            raise ValueError(f'Incomplete group set: {job}')
        for index, a in enumerate(analysis['results']):
            key = group_key(a)
            if key not in expected or strip_units(a) != expected[key]['analysis']:
                raise ValueError(f'Original and final-summary estimates differ: {job}/{key}')
            g = copy.deepcopy(expected[key])
            g['analysis'] = a
            g['readout'] = readout_key(g)
            g['group_id'] = '/'.join((job, a['task'], a['kind'], g['readout']))
            g['source'] = {'pack': sources[-1]['pack'], 'member': member,
                           'pointer': f'/results/{index}', 'member_sha256': sources[-1]['member_sha256']}
            groups.append(g)
            copied += 1
    groups.sort(key=lambda g: (MODEL_ORDER.index(g['model']), g['material'], g['task'], g['kind'],
                               READOUT_ORDER.index(g['readout']), g['job']))
    contrasts = build_pair_contrasts(groups)
    panels = build_panels(groups, contrasts)
    checks = {'copied_groups': copied, 'source_analysis_packs': len(final_by_job),
              'paired_contrasts': len(contrasts), 'copied_estimates_match_final_summary': True,
              'all_new_contrasts_pair_unit_ids_before_statistics': True,
              'new_model_runs': 0, 'rescored_observations': 0}
    common = {'schema': 'v14-readout-integration-1', 'created_at': datetime.now(timezone.utc).isoformat(),
              'intervals': 'descriptive Student-t 95%, df=n-1; unadjusted; CI null for SE=0 or n<2',
              'inference_unit': 'template for synthetic (32 planned); photo for natural (64 planned)',
              'model_order': MODEL_ORDER, 'readout_order': READOUT_ORDER,
              'primary_recovery_policy': 'accuracy_invalid_as_incorrect',
              'contrast_direction': 'positive_readout minus negative_readout within the same unit_id'}
    summary = {**common, 'groups': [strip_units(g) for g in groups],
               'paired_readout_contrasts': contrasts, 'checks': checks}
    numbers = {**common, **panels,
               'selection_note': 'Presentation subsets; all copied groups and contrasts remain in summary.json.'}
    args.output.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output / 'summary.json', summary)
    atomic_json(args.output / 'paper_numbers.json', numbers)
    atomic_text(args.output / 'notes.md', build_notes(panels, checks))
    provenance = {**common, 'script': str(Path(__file__).resolve()),
                  'script_sha256': digest(Path(__file__).read_bytes()), 'python': sys.version,
                  'scipy_version': scipy.__version__, 'argv': sys.argv,
                  'sources': sources, 'checks': checks,
                  'outputs': {name: digest((args.output / name).read_bytes())
                              for name in ('summary.json', 'paper_numbers.json', 'notes.md')}}
    atomic_json(args.output / 'provenance.json', provenance)
    print(json.dumps({'output': str(args.output), **checks}, ensure_ascii=False))


if __name__ == '__main__':
    main()
