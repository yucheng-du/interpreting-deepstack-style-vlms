#!/usr/bin/env python3
"""Export and describe the accepted v11 dose grid, without models or new inference.

The existing scene-level means, normal intervals and complete-scene denominators
are copied exactly. Curve shapes and differences of cell means are descriptive;
no paired uncertainty is reconstructed from marginal intervals.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import os as _path_os
DATA_ROOT = Path(_path_os.environ.get("DS_DATA_ROOT", "work")).expanduser().resolve()
from statistics import median

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (DATA_ROOT / 'runs/measurement-v11-20261020/readable/analysis/v2-dose-cross-model-final.json')
OUTPUT = (DATA_ROOT / 'paper/v9-v11-integration-20261022/dose')
MODELS = ('ibm4b', 'qwen8b', 'qwen32b')
READOUTS = (('direct', 'none'), ('label', 'identity'), ('label', 'cyclic'))
STRENGTHS = ((1.0, 1.0), (1.0, .08), (.08, 1.0))
SCORE_K = (0, .25, .5, 1, 2, 4)
GEN_K = (0, 1, 4)
CONTEXT = ('changing_layout', 'fixed_layout', 'changing_strength',
           'fixed_strength', 'fixed_other_assignment')
CONDITION = ('query_type', 'query_target', 'response_format', 'mapping', *CONTEXT, 'region')
EPS = 1e-12
SPECS = (
    ('raw_score', 'raw_curves', 'score_scene_summaries', 'score'),
    ('raw_generation', 'raw_curves', 'generation_scene_summaries', 'generation'),
    ('raw_role_score', 'raw_curves', 'role_difference_scene_summaries', 'score'),
    ('normalized_score', 'normalized_cells', 'score_scene_summaries', 'score'),
    ('normalized_generation', 'normalized_cells', 'generation_scene_summaries', 'generation'),
    ('matched_score', 'canonical_native_reference_contrasts', 'score_summaries', 'score'),
    ('matched_generation', 'canonical_native_reference_contrasts', 'generation_summaries', 'generation'),
    ('exchanged_native_score', None, 'exchanged_native_role_differences', 'score'),
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def export(out, name, rows):
    write_json(out / f'{name}.json', rows)
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with (out / f'{name}.csv').open('w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False, separators=(',', ':'))
                             if isinstance(v, (dict, list, tuple)) else v for k, v in row.items()})


def sign(x):
    return 1 if x > EPS else -1 if x < -EPS else 0


def counts(values):
    c = Counter(sign(x) for x in values)
    return {'positive': c[1], 'negative': c[-1], 'zero': c[0]}


def kstr(v):
    return f'{v:g}'


def identity(row, fields):
    return tuple(row[f] for f in fields)


def collect(data, source, source_sha):
    tables = {}
    for name, section, field, measure in SPECS:
        rows = []
        for mi, model in enumerate(data['models']):
            values = model[section][field] if section else model[field]
            for ri, row in enumerate(values):
                pointer = f'/models/{mi}/' + (f'{section}/' if section else '') + f'{field}/{ri}'
                copied = dict(row)
                lo, hi = row['normal_95_interval']
                copied.update(table=name, measurement=measure,
                              model_key=model['model']['key'], model_revision=model['model']['revision'],
                              readout=('direct' if row['response_format'] == 'direct' else 'label_' + row['mapping']),
                              ci_low=lo, ci_high=hi,
                              source_file=str(source), source_sha256=source_sha, source_json_pointer=pointer,
                              archived_model_summary=model['source_summary'])
                if name.startswith('matched_'):
                    copied['changed_entry'] = 'main_minus_later'
                    copied['dose_comparison'] = 'same_canonical_native_target'
                elif name.startswith('normalized_'):
                    copied['matched_to_native_of'] = 'later' if row['changed_entry'] == 'main' else 'main'
                rows.append(copied)
        tables[name] = rows
    return tables


def verify(data, tables):
    """Check the requested export/grid, not raw execution or scientific acceptance."""
    assert [x['model']['key'] for x in data['models']] == list(MODELS)
    context_grid = set(itertools.product(('M0', 'M1'), ('M0',), STRENGTHS, ('a', 'b')))
    expected_contexts = {(a, b, s[0], s[1], c) for a, b, s, c in context_grid}
    report = {'scope': 'summary grid, value copying, denominators and source pointers only',
              'models': {}, 'all_exported_values_finite': True,
              'no_recomputed_intervals': True, 'no_model_or_raw_reads': True}
    expected_counts = dict(zip((x[0] for x in SPECS), (864, 432, 432, 288, 288, 288, 288, 144)))
    for model in MODELS:
        item = {}
        for name, rows0 in tables.items():
            rows = [x for x in rows0 if x['model_key'] == model]
            assert len(rows) == expected_counts[name], (model, name, len(rows))
            assert {identity(r, CONTEXT) for r in rows} == expected_contexts
            assert {(r['response_format'], r['mapping']) for r in rows} == set(READOUTS)
            assert {r['query_target'] for r in rows} == {'changed', 'unchanged'}
            assert {r['query_type'] for r in rows} == {'identity'}
            assert {r['region'] for r in rows} == {'all'}
            assert all(r['planned_unit_n'] == 16 and 0 < r['complete_unit_n'] <= 16 for r in rows)
            assert all(r['unit_key'] == 'scene_id' for r in rows)
            assert all(math.isfinite(r[k]) for r in rows for k in ('mean', 'ci_low', 'ci_high'))
            assert all(r['ci_low'] <= r['mean'] <= r['ci_high'] for r in rows)
            assert len({r['source_json_pointer'] for r in rows}) == len(rows)
            if name.startswith('raw_'):
                ks = SCORE_K if name != 'raw_generation' else GEN_K
                keys = ('changed_entry', *CONDITION)
                groups = defaultdict(list)
                for r in rows:
                    groups[identity(r, keys)].append(r['dose_value'])
                assert all(sorted(v) == list(ks) for v in groups.values())
                assert all(abs(r['mean']) < EPS for r in rows if r['dose_value'] == 0)
            if name.startswith('matched_'):
                assert {r['reference_target'] for r in rows} == {'later_native', 'main_native'}
                assert {r['dose_metric'] for r in rows} == {'rss', 'l1'}
                keys = (*CONDITION, 'reference_target', 'dose_metric')
                assert len({identity(r, keys) for r in rows}) == len(rows)
            item[name] = {'cells': len(rows), 'actual_n_counts': dict(Counter(r['complete_unit_n'] for r in rows))}
        report['models'][model] = item
    report['total_exported_cells'] = sum(len(x) for x in tables.values())
    # Check native-side selection using full-scene cells only. Different
    # incomplete generation subsets do not permit marginal-mean subtraction.
    arithmetic = {}
    for measure in ('score', 'generation'):
        fields = ('model_key', *CONDITION, 'changed_entry')
        raw = {identity(r, fields): r for r in tables['raw_' + measure] if r['dose_value'] == 1}
        norm = {identity(r, fields) + (r['dose_metric'],): r for r in tables['normalized_' + measure]}
        errors, skipped = [], 0
        for r in tables['matched_' + measure]:
            base = tuple(r[f] for f in fields[:-1])
            if r['reference_target'] == 'main_native':
                a, b = raw[base + ('main',)], norm[base + ('later', r['dose_metric'])]
            else:
                a, b = norm[base + ('main', r['dose_metric'])], raw[base + ('later',)]
            if any(x['complete_unit_n'] != 16 for x in (r, a, b)):
                skipped += 1
                continue
            errors.append(abs(r['mean'] - (a['mean'] - b['mean'])))
        assert max(errors, default=0) < 1e-10, (measure, max(errors))
        arithmetic[measure] = {'full_scene_cells_checked': len(errors),
                               'incomplete_subset_cells_skipped': skipped,
                               'max_absolute_mean_difference': max(errors, default=0)}
    report['same_target_native_side_arithmetic'] = arithmetic
    return report


def curve_shapes(tables):
    output, nonlinear = [], []
    fields = ('model_key', 'measurement', 'changed_entry', *CONDITION)
    for name in ('raw_score', 'raw_generation', 'raw_role_score'):
        groups = defaultdict(list)
        for r in tables[name]:
            groups[identity(r, fields)].append(r)
        for key, rows in groups.items():
            rows.sort(key=lambda x: x['dose_value'])
            means = [r['mean'] for r in rows]
            steps = [b-a for a, b in zip(means, means[1:])]
            nonzero_signs = [sign(x) for x in means if sign(x)]
            nonzero_step_signs = [sign(x) for x in steps if sign(x)]
            turning = sum(a != b for a, b in zip(nonzero_step_signs, nonzero_step_signs[1:]))
            crossings = sum(a != b for a, b in zip(nonzero_signs, nonzero_signs[1:]))
            base = next(r for r in rows if r['dose_value'] == 1)
            result = dict(zip(fields, key))
            result.update(table=name, readout=rows[0]['readout'], raw_k=[r['dose_value'] for r in rows],
                          means=means, complete_unit_n=[r['complete_unit_n'] for r in rows],
                          step_differences=steps, mean_nondecreasing=all(x >= -EPS for x in steps),
                          mean_nonincreasing=all(x <= EPS for x in steps),
                          strict_increase_steps=sum(x > EPS for x in steps),
                          strict_decrease_steps=sum(x < -EPS for x in steps),
                          turning_points=turning, nonzero_sign_crossings=crossings,
                          peak_k=[r['dose_value'] for r in rows if abs(r['mean'] - max(means)) < EPS],
                          mean_k4_minus_k1=means[-1] - base['mean'],
                          descriptive_only=True, source_json_pointers=[r['source_json_pointer'] for r in rows],
                          source_sha256=rows[0]['source_sha256'])
            output.append(result)
            for r in rows:
                if r['dose_value'] in (0, 1):
                    continue
                nonlinear.append({**dict(zip(fields, key)), 'table': name, 'readout': r['readout'],
                                  'raw_k': r['dose_value'], 'mean': r['mean'], 'mean_k1': base['mean'],
                                  'linear_extrapolation_k_times_E1': r['dose_value'] * base['mean'],
                                  'nonlinear_residual_Ek_minus_kE1': r['mean'] - r['dose_value'] * base['mean'],
                                  'descriptive_only': True, 'paired_interval_available': False,
                                  'source_json_pointer': r['source_json_pointer'],
                                  'k1_source_json_pointer': base['source_json_pointer'],
                                  'source_sha256': r['source_sha256']})
    return output, nonlinear


def agreement(tables):
    output = []
    for family in ('raw', 'matched'):
        fields = ('model_key', *CONDITION)
        fields += ('changed_entry', 'dose_value') if family == 'raw' else ('reference_target', 'dose_metric')
        score_index = {identity(r, fields): r for r in tables[family + '_score']}
        for gen in tables[family + '_generation']:
            if family == 'raw' and gen['dose_value'] not in (1, 4):
                continue
            score = score_index[identity(gen, fields)]
            ss, gs = sign(score['mean']), sign(gen['mean'])
            direction = 'same_nonzero' if ss == gs and ss else 'opposite_nonzero' if ss * gs == -1 else 'zero_involved'
            output.append({**dict(zip(fields, identity(gen, fields))), 'comparison_family': family,
                           'readout': gen['readout'], 'score_mean': score['mean'], 'generation_mean': gen['mean'],
                           'score_CI': [score['ci_low'], score['ci_high']],
                           'generation_CI': [gen['ci_low'], gen['ci_high']],
                           'score_n': score['complete_unit_n'], 'generation_n': gen['complete_unit_n'],
                           'mean_sign_agreement': direction, 'descriptive_only': True,
                           'score_source_json_pointer': score['source_json_pointer'],
                           'generation_source_json_pointer': gen['source_json_pointer'],
                           'source_sha256': gen['source_sha256']})
    return output


def reversals(tables):
    result = []
    for measure in ('score', 'generation'):
        for comparison, varied, fixed, levels in (
            ('rss_vs_l1', 'dose_metric', 'reference_target', ('rss', 'l1')),
            ('later_vs_main_native', 'reference_target', 'dose_metric', ('later_native', 'main_native')),
        ):
            fields = ('model_key', *CONDITION, fixed)
            groups = defaultdict(dict)
            for r in tables['matched_' + measure]:
                groups[identity(r, fields)][r[varied]] = r
            for key, pair in groups.items():
                a, b = (pair[x] for x in levels)
                ci_reverse = (a['ci_high'] < 0 and b['ci_low'] > 0) or (a['ci_low'] > 0 and b['ci_high'] < 0)
                result.append({**dict(zip(fields, key)), 'measurement': measure, 'readout': a['readout'],
                               'comparison': comparison, 'level_a': levels[0], 'level_b': levels[1],
                               'mean_a': a['mean'], 'mean_b': b['mean'],
                               'CI_a': [a['ci_low'], a['ci_high']], 'CI_b': [b['ci_low'], b['ci_high']],
                               'n_a': a['complete_unit_n'], 'n_b': b['complete_unit_n'],
                               'opposite_nonzero_means': sign(a['mean']) * sign(b['mean']) == -1,
                               'opposite_original_intervals': ci_reverse, 'descriptive_only': True,
                               'source_json_pointers': [a['source_json_pointer'], b['source_json_pointer']],
                               'source_sha256': a['source_sha256']})
    return result


def summarize_shapes(rows):
    fields = ('table', 'model_key', 'query_target', 'readout', 'changed_entry')
    groups = defaultdict(list)
    for r in rows:
        groups[identity(r, fields)].append(r)
    return [{**dict(zip(fields, key)), 'curves': len(rs),
             'mean_nondecreasing': sum(r['mean_nondecreasing'] for r in rs),
             'mean_nonincreasing': sum(r['mean_nonincreasing'] for r in rs),
             'any_mean_decrease': sum(r['strict_decrease_steps'] > 0 for r in rs),
             'any_nonzero_sign_crossing': sum(r['nonzero_sign_crossings'] > 0 for r in rs),
             'any_turning_point': sum(r['turning_points'] > 0 for r in rs),
             'k4_minus_k1_signs': counts(r['mean_k4_minus_k1'] for r in rs),
             'k4_minus_k1_range': [min(r['mean_k4_minus_k1'] for r in rs), max(r['mean_k4_minus_k1'] for r in rs)],
             'descriptive_only': True} for key, rs in groups.items()]


def summarize_agreement(rows):
    fields = ('comparison_family', 'model_key', 'query_target', 'readout')
    groups = defaultdict(list)
    for r in rows:
        groups[identity(r, fields) + (r.get('dose_value'), r.get('reference_target'), r.get('dose_metric'))].append(r)
    return [{**dict(zip((*fields, 'raw_k', 'reference_target', 'dose_metric'), key)), 'cells': len(rs),
             **dict(Counter(r['mean_sign_agreement'] for r in rs)), 'descriptive_only': True}
            for key, rs in groups.items()]


def summarize_reversals(rows):
    fields = ('comparison', 'model_key', 'measurement', 'query_target', 'readout')
    groups = defaultdict(list)
    for r in rows:
        extra_name = 'reference_target' if r['comparison'] == 'rss_vs_l1' else 'dose_metric'
        groups[identity(r, fields) + (extra_name, r[extra_name])].append(r)
    return [{**dict(zip(fields, key[:len(fields)])), key[-2]: key[-1], 'comparisons': len(rs),
             'opposite_nonzero_means': sum(r['opposite_nonzero_means'] for r in rs),
             'opposite_original_intervals': sum(r['opposite_original_intervals'] for r in rs),
             'descriptive_only': True} for key, rs in groups.items()]


def summarize_nonlinearity(rows):
    fields = ('table', 'model_key', 'query_target', 'readout', 'changed_entry', 'raw_k')
    groups = defaultdict(list)
    for r in rows:
        groups[identity(r, fields)].append(r['nonlinear_residual_Ek_minus_kE1'])
    return [{**dict(zip(fields, key)), 'cells': len(v), **counts(v),
             'residual_min': min(v), 'residual_median': median(v), 'residual_max': max(v),
             'descriptive_only': True} for key, v in groups.items()]


def direction_summary(rows, keys):
    groups = defaultdict(list)
    for r in rows:
        groups[identity(r, keys)].append(r)
    return [{**dict(zip(keys, key)), 'cells': len(rs), **counts(r['mean'] for r in rs),
             'original_CI_above_zero': sum(r['ci_low'] > 0 for r in rs),
             'original_CI_below_zero': sum(r['ci_high'] < 0 for r in rs),
             'mean_min': min(r['mean'] for r in rs), 'mean_max': max(r['mean'] for r in rs),
             'descriptive_only': True} for key, rs in groups.items()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to(DATA_ROOT):
        raise SystemExit('Outputs must remain on Research/DeepStack.')
    args.output.mkdir(parents=True, exist_ok=True)
    table_dir = args.output / 'tables'
    table_dir.mkdir(exist_ok=True)
    data = json.loads(args.source.read_text())
    source_sha = sha(args.source)
    tables = collect(data, args.source.resolve(), source_sha)
    checks = verify(data, tables)
    for name, rows in tables.items():
        export(table_dir, name, rows)
    shapes, nonlinear = curve_shapes(tables)
    agree = agreement(tables)
    reverse = reversals(tables)
    derived = {'curve_shapes': shapes, 'nonlinear_residuals': nonlinear,
               'score_generation_agreement': agree, 'matched_reversals': reverse,
               'curve_shape_summary': summarize_shapes(shapes),
               'nonlinear_residual_summary': summarize_nonlinearity(nonlinear),
               'raw_role_sign_summary': direction_summary(tables['raw_role_score'],
                    ('model_key', 'query_target', 'readout', 'dose_value')),
               'matched_direction_summary': direction_summary(tables['matched_score'] + tables['matched_generation'],
                    ('model_key', 'measurement', 'query_target', 'readout', 'reference_target', 'dose_metric')),
               'score_generation_agreement_summary': summarize_agreement(agree),
               'matched_reversal_summary': summarize_reversals(reverse)}
    for name, rows in derived.items():
        export(table_dir, name, rows)
    write_json(args.output / 'coverage_checks.json', checks)
    manifest = {'schema': 'v11-dose-curve-integration-v1', 'source_file': str(args.source.resolve()),
                'source_sha256': source_sha, 'analysis_script': str(Path(__file__).resolve()),
                'analysis_script_sha256': sha(Path(__file__)),
                'source_schema': data['schema'], 'source_created_at': data['created_at'],
                'raw_score_k': SCORE_K, 'raw_generation_k': GEN_K,
                'actual_contexts': [{'changing_layout': m, 'fixed_layout': 'M0', 'changing_strength': c,
                                    'fixed_strength': f, 'fixed_other_assignment': a}
                                   for m, (c, f), a in itertools.product(('M0', 'M1'), STRENGTHS, ('a', 'b'))],
                'reference_condition': {'changing_layout': 'M0', 'fixed_layout': 'M0',
                                        'changing_strength': 1, 'fixed_strength': 1,
                                        'fixed_other_assignment': 'a', 'query_target': 'changed',
                                        'response_format': 'direct', 'mapping': 'none'},
                'scope_notes': [
                    'All formal cells in this accepted summary are exported; no direction filtering.',
                    'Raw k is a multiplier, not a state-normalized dose. Generation has no .25/.5/2 points.',
                    'Canonical same-target contrasts are copied; exchanged-native contrasts are separate.',
                    'No per-scene calibrated k, achieved dose or query-forward dose values are present here.',
                    'The raw-role summary is score only; no paired raw-generation role CI is invented.',
                    'Original scene-level normal 95% intervals are copied, not simultaneous intervals.',
                    'Shape, sign, nonlinear-residual and interval-sign counts are descriptive cell summaries.',
                    'Generation complete-scene subsets can differ across k and from score; aggregate sign disagreement is not a paired per-example dissociation test.',
                    'EPS=1e-12 handles numerical zero only; it is not a scientific effect-size threshold.',
                    'No difference CI is inferred from two marginal CIs; no ratio with a near-zero denominator.',
                    'n is complete source scenes, not logical row count; actual generation n is retained.',
                ],
                'models': [{k: m[k] for k in ('model', 'planned', 'observed', 'coverage',
                                             'raw_source_references', 'canonical_native_reference_contrasts')
                            if k != 'canonical_native_reference_contrasts'} |
                           {'reference_coverage': m['canonical_native_reference_contrasts']['coverage']}
                           for m in data['models']],
                'table_rows': {k: len(v) for k, v in (tables | derived).items()},
                'output_sha256': {str(p.relative_to(args.output)): sha(p) for p in sorted(table_dir.glob('*'))}}
    write_json(args.output / 'manifest.json', manifest)
    print(json.dumps({'output': str(args.output), 'source_sha256': source_sha,
                      'tables': manifest['table_rows'], 'coverage_passed': True}, ensure_ascii=False))


if __name__ == '__main__':
    main()
