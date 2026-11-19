"""Paired one-object regional effects with planned denominators and scene clusters.

The plan is required: absent entire queries remain in the denominator. No
conditioning on native success or on the expected sign is allowed. ROI labels
describe nominal token geometry, not the effective receptive field.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
from statistics import mean, stdev

from cloud_prep.portable_data import _guard_path
from cloud_prep.measurement_v8.mechanism_design import DESIGN_SCHEMA

GROUP = ('attribute', 'response_format', 'main_strength', 'entry_group', 'main_layout', 'region')


def key(row):
    return row['id'], row['response_format'], row['mapping'], row['condition']


def stats(values):
    n = len(values)
    avg = mean(values) if n else None
    se = stdev(values) / math.sqrt(n) if n > 1 else None
    return {'n': n, 'mean': avg, 'se': se,
        'normal_95_interval': [avg - 1.96 * se, avg + 1.96 * se] if se is not None else None,
        'positive': sum(x > 0 for x in values), 'negative': sum(x < 0 for x in values),
        'zero': sum(x == 0 for x in values)}


def finite_score(row):
    value = row.get('b_minus_a_log_odds') if row else None
    return type(value) in (int, float) and math.isfinite(value)


def valid_generation(row):
    return bool(row and row.get('scorable') is True and not row.get('truncated', False)
                and row.get('original_prediction') in ('(A)', '(B)'))


def summarize(rows, fields=GROUP, value='effect'):
    """Only complete expected within-scene repeats contribute to a scene mean.

    This prevents missing label mappings or target queries from silently changing
    the weighting. Planned counts and incomplete clusters remain visible.
    """
    grouped = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[tuple(row[f] for f in fields)][row['scene_id']].append(row)
    result = []
    for group, units in grouped.items():
        complete = [mean(r[value] for r in members) for members in units.values()
                    if all(r.get(value) is not None for r in members)]
        result.append({**dict(zip(fields, group)), 'unit_key': 'scene_id',
            'planned_unit_n': len(units), 'complete_unit_n': len(complete),
            'incomplete_unit_n': len(units) - len(complete),
            'planned_query_repeats': sum(len(m) for m in units.values()),
            'available_query_repeats': sum(r.get(value) is not None for m in units.values() for r in m),
            'statistics': stats(complete)})
    return result


def _validate_records(records, expected, kind):
    indexed = {}
    for row in records:
        k = key(row)
        if k in indexed:
            raise ValueError(f'Duplicate {kind} cell; select one attempt')
        if k not in expected:
            raise ValueError(f'Unplanned {kind} cell: {k}')
        spec = expected[k]
        if row.get('schema') != DESIGN_SCHEMA:
            raise ValueError('Wrong mechanism scientific schema')
        if row.get('scene_id') != spec['query']['scene_id']:
            raise ValueError('Scene cluster identity changed')
        for field in ('main_layout', 'main_strength', 'entry_group', 'region', 'donor_assignment'):
            if row.get(field) != spec['condition'][field]:
                raise ValueError(f'Condition identity mismatch: {field}')
        indexed[k] = row
    return indexed


def analyze(records, generations=(), expected_plan=None):
    if not expected_plan or expected_plan.get('schema') != DESIGN_SCHEMA:
        raise ValueError('A mechanism execution plan is required for fixed denominators')
    plan = expected_plan
    if len({q['id'] for q in plan['queries']}) != len(plan['queries']):
        raise ValueError('Duplicate planned query')
    expected, expected_gen, pair_specs = {}, {}, defaultdict(dict)
    for query in plan['queries']:
        for readout in plan['readouts']:
            for condition in plan['conditions']:
                k = query['id'], readout['response_format'], readout['mapping'], condition['name']
                if k in expected:
                    raise ValueError('Duplicate planned cell')
                entry = {'query': query, 'readout': readout, 'condition': condition}
                expected[k] = entry
                if query['generation_selected'] and condition['generation']:
                    expected_gen[k] = entry
                if condition['role'] == 'single_object_pair':
                    pair = k[:3] + tuple(condition[f] for f in ('main_strength', 'entry_group', 'main_layout', 'region'))
                    pair_specs[pair][condition['donor_assignment']] = k
    if len(expected) != plan['expected_score_rows'] or len(expected_gen) != plan['expected_generation_rows']:
        raise ValueError('Plan row counts disagree with explicit condition product')
    scores = _validate_records(records, expected, 'score')
    gens = _validate_records(generations, expected_gen, 'generation')
    effects, behavior = [], []
    for pair, keys in pair_specs.items():
        if set(keys) != {'a', 'b'}:
            raise ValueError('Scientific plan lacks a complete donor pair')
        spec = expected[keys['a']]
        q, c, r = spec['query'], spec['condition'], spec['readout']
        base = {**{k: q[k] for k in ('id', 'scene_id', 'attribute', 'target_object', 'target_index')},
            **r, **{k: c[k] for k in ('main_strength', 'entry_group', 'main_layout', 'region')}}
        a, b = scores.get(keys['a']), scores.get(keys['b'])
        valid = finite_score(a) and finite_score(b)
        effects.append({**base, 'effect': b['b_minus_a_log_odds'] - a['b_minus_a_log_odds'] if valid else None,
            'a_score': a['b_minus_a_log_odds'] if finite_score(a) else None,
            'b_score': b['b_minus_a_log_odds'] if finite_score(b) else None,
            'missing_score_rows': sum(x is None for x in (a, b)),
            'nonfinite_score_rows': sum(x is not None and not finite_score(x) for x in (a, b))})
        if keys['a'] in expected_gen and keys['b'] in expected_gen:
            ga, gb = gens.get(keys['a']), gens.get(keys['b'])
            gen_valid = valid_generation(ga) and valid_generation(gb)
            donor_b = ('(A)', '(B)')[1 - q['answer_index']]
            effect = (int(gb['original_prediction'] == donor_b) - int(ga['original_prediction'] == donor_b)) if gen_valid else None
            behavior.append({**base, 'effect': effect, 'both_valid': gen_valid,
                'flip': ga['original_prediction'] != gb['original_prediction'] if gen_valid else None,
                'direction': 'toward_b' if effect == 1 else 'toward_a' if effect == -1 else 'unchanged' if effect == 0 else None,
                'missing_generation_rows': sum(x is None for x in (ga, gb)),
                'invalid_generation_rows': sum(x is not None and not valid_generation(x) for x in (ga, gb)),
                'truncated_generation_rows': sum(bool(x and x.get('truncated')) for x in (ga, gb)),
                'score_effect_available': valid})

    # Contrasts pair the same query and mapping before cluster averaging.
    indexed_effects = {(e['id'], e['response_format'], e['mapping'], e['main_strength'],
                        e['entry_group'], e['main_layout'], e['region']): e for e in effects}
    contrasts = []
    for e in effects:
        prefix = e['id'], e['response_format'], e['mapping'], e['main_strength'], e['entry_group']
        if e['region'] == 'all':
            for left, right in (('roi', 'complement'), ('roi', 'all'), ('complement', 'all')):
                a = indexed_effects[prefix + (e['main_layout'], left)]
                b = indexed_effects[prefix + (e['main_layout'], right)]
                contrasts.append({**{k: e[k] for k in e if k not in ('a_score', 'b_score')},
                    'region': f'{left}_minus_{right}',
                    'effect': a['effect'] - b['effect'] if a['effect'] is not None and b['effect'] is not None else None})
        if e['main_layout'] == 'M0':
            m1 = indexed_effects[prefix + ('M1', e['region'])]
            contrasts.append({**{k: e[k] for k in e if k not in ('a_score', 'b_score')},
                'main_layout': 'M1_minus_M0',
                'effect': m1['effect'] - e['effect'] if m1['effect'] is not None and e['effect'] is not None else None})

    behavior_groups = defaultdict(list)
    for b in behavior:
        behavior_groups[tuple(b[k] for k in GROUP) + (b['mapping'],)].append(b)
    behavior_summary = []
    for group, members in behavior_groups.items():
        n = len(members)
        common = sum(m['both_valid'] for m in members)
        flips = sum(m['flip'] is True for m in members)
        behavior_summary.append({**dict(zip(GROUP + ('mapping',), group)), 'planned_pairs': n,
            'common_valid_pairs': common, 'missing_pairs': sum(m['missing_generation_rows'] > 0 for m in members),
            'invalid_pairs': sum(m['invalid_generation_rows'] > 0 for m in members),
            'truncated_pairs': sum(m['truncated_generation_rows'] > 0 for m in members),
            'flips': flips, 'flips_over_planned_lower_bound': flips / n,
            'flip_rate_common_valid': flips / common if common else None,
            'toward_b': sum(m['direction'] == 'toward_b' for m in members),
            'toward_a': sum(m['direction'] == 'toward_a' for m in members)})

    natives = defaultdict(list)
    audits = []
    for k, spec in expected.items():
        q, c, r = spec['query'], spec['condition'], spec['readout']
        score = scores.get(k)
        if c['role'].startswith('native_'):
            # Alpha 0 carries object geometry but no attribute gold.
            gold_index = None if c['main_strength'] == 0 else q['answer_index'] ^ (c['role'] == 'native_donor' and c['donor_assignment'] == 'b')
            gold = ('(A)', '(B)')[gold_index] if gold_index is not None else None
            gen = gens.get(k)
            natives[(r['response_format'], r['mapping'], c['name'])].append({
                'with_gold': gold is not None, 'score_present': finite_score(score),
                'score_correct': finite_score(score) and score.get('original_prediction') == gold if gold is not None else None,
                'gen_planned': k in expected_gen, 'gen_present': gen is not None,
                'gen_valid': valid_generation(gen),
                'gen_correct': valid_generation(gen) and gen['original_prediction'] == gold if gold is not None else None})
        elif c['role'] == 'self_patch':
            native_key = k[:3] + (f'native_{c["main"]}',)
            baseline = scores.get(native_key)
            audits.append({'id': q['id'], 'scene_id': q['scene_id'], **r,
                'condition': c['name'], 'entry_group': c['entry_group'], 'region': c['region'],
                'delta_b_minus_a': score['b_minus_a_log_odds'] - baseline['b_minus_a_log_odds']
                    if finite_score(score) and finite_score(baseline) else None})
    native_summary = []
    for group, members in natives.items():
        native_summary.append({**dict(zip(('response_format', 'mapping', 'condition'), group)),
            'planned_rows': len(members), 'planned_rows_with_gold': sum(m['with_gold'] for m in members),
            'score_rows_present': sum(m['score_present'] for m in members),
            'score_correct': sum(m['score_correct'] is True for m in members),
            'planned_generation_rows': sum(m['gen_planned'] for m in members),
            'planned_generation_rows_with_gold': sum(m['gen_planned'] and m['with_gold'] for m in members),
            'generation_rows_present': sum(m['gen_present'] for m in members),
            'generation_common_valid_with_gold': sum(m['gen_valid'] and m['with_gold'] for m in members),
            'generation_correct': sum(m['gen_correct'] is True for m in members)})

    return {'schema': DESIGN_SCHEMA, 'unit_key': 'scene_id',
        'coverage': {'planned_score_rows': len(expected), 'present_score_rows': len(scores),
            'missing_score_rows': len(expected) - len(scores), 'nonfinite_score_rows': sum(not finite_score(s) for s in scores.values()),
            'planned_generation_rows': len(expected_gen), 'present_generation_rows': len(gens),
            'missing_generation_rows': len(expected_gen) - len(gens),
            'invalid_generation_rows': sum(not valid_generation(g) for g in gens.values()),
            'truncated_generation_rows': sum(bool(g.get('truncated')) for g in gens.values()),
            'planned_units': len({q['scene_id'] for q in plan['queries']}),
            'planned_query_pairs': len(effects), 'complete_query_pairs': sum(e['effect'] is not None for e in effects)},
        'query_effects': effects, 'query_contrasts': contrasts,
        'cell_summaries': summarize(effects), 'per_mapping_cell_summaries': summarize(effects, GROUP + ('mapping',)),
        'contrast_summaries': summarize(contrasts), 'per_mapping_contrast_summaries': summarize(contrasts, GROUP + ('mapping',)),
        'behavior_pairs': behavior, 'behavior_summary': behavior_summary,
        'behavior_effect_summaries': summarize(behavior), 'native_accuracy': native_summary, 'self_patch_audits': audits,
        'interpretation': 'Exploratory paired b-minus-a effects. Scene is the independent unit; target queries and mappings repeat within scene. Direct and label scales remain separate. Missing values are not zero; planned denominators include wholly absent queries. Complete cluster means require all planned repeats for that cell. Native success never filters effects. ROI marks donor-coordinate token support, not effective receptive fields; complement effects can reflect contextualized information. ROI and complement counts differ, and all versus ROI differences are not a normalized per-token causal comparison. Group/depth/source differences preclude naming IBM Layer versus Spatial effects semantic-versus-spatial importance or assuming additivity.'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--scores', type=Path, nargs='+', required=True)
    parser.add_argument('--generations', type=Path, nargs='*', default=[])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    load = lambda paths: [json.loads(line) for path in paths for line in path.read_text().splitlines() if line.strip()]
    plan = json.loads(args.plan.read_text())
    result = analyze(load(args.scores), load(args.generations), plan)
    output = _guard_path(args.output, write=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'output': str(output), **result['coverage']}, indent=2))


if __name__ == '__main__':
    main()
