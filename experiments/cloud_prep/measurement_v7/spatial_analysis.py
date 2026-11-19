"""Unit-level M x D x P analysis: layout interactions, M flips, template components and generation agreement.

E = (b-a log score | DS bbb) - (b-a | DS aaa) per query, mapping, format, main
strength, layout M, pose D and permutation P, with a/b fixed by the target's object
identity. Units default to scene_id (--unit-key, e.g. a source-cluster id stored in
record_fields). Label mappings are averaged within unit; direct stays separate;
label and direct magnitudes are never pooled. Missing cells stay missing.
"""
from __future__ import annotations
import argparse
from collections import defaultdict, Counter
import json
import math
from pathlib import Path
from statistics import mean, stdev

from cloud_prep.portable_data import _guard_path, sha256_file
from cloud_prep.measurement_v7.spatial_design import (DESIGN_SCHEMA, LAYOUTS, CELLS, CELL_NAMES, VECTOR,
    VECTOR_NAMES, HYPOTHESES, TEMPLATES, decompose, classify, layout_interaction, sign_pattern,
    four_cell_prediction)

GROUP = ('attribute', 'response_format', 'main_kind', 'main_strength')
CELL_AXES = ('main_layout', 'donor_pose', 'token_permutation')
IDS = ('id', 'scene_id', 'attribute', 'target_object', 'target_index', 'target_main_slot', 'mapping')
# Shards merged by --runs must share these (offset/limit/output may differ; overlap is caught as duplicate cells).
SAME_RUN_FIELDS = ('schema', 'execution_schema', 'family', 'row_schema', 'panel', 'manifest_sha256', 'design',
                   'mappings', 'readouts', 'generate_tokens', 'parser', 'image_processor')
VALUE = 'all_b_minus_all_a'


def stats(values):
    n = len(values); avg = mean(values); se = stdev(values) / math.sqrt(n) if n > 1 else None
    return {'n': n, 'mean': avg, 'se': se, 'normal_95_interval': [avg - 1.96 * se, avg + 1.96 * se] if se is not None else None,
            'positive': sum(x > 0 for x in values), 'negative': sum(x < 0 for x in values), 'zero': sum(x == 0 for x in values)}


def unit_of(record, unit_key):
    if unit_key in record:
        return record[unit_key]
    if unit_key in (record.get('record_fields') or {}):
        return record['record_fields'][unit_key]
    raise ValueError(f'Records lack unit key {unit_key!r}')


def key(row):
    return row['id'], row['response_format'], row['mapping'], row['condition']


def unit_means(rows, fields, value_field):
    """Average within-unit repeats (queries, mappings, canvases) for each group."""
    grouped = defaultdict(list)
    for r in rows:
        grouped[tuple(r[k] for k in fields) + (r['unit'],)].append(r)
    out = []
    for k, members in grouped.items():
        names = sorted(set().union(*(m[value_field] for m in members)))
        out.append({**dict(zip(fields, k[:-1])), 'unit_key': members[0]['unit_key'], 'unit': k[-1],
                    'repeats': len(members),
                    value_field: {n: mean(m[value_field][n] for m in members if n in m[value_field]) for n in names}})
    return out


def summarize(unit_rows, fields, value_field):
    groups = defaultdict(list)
    for r in unit_rows:
        groups[tuple(r[k] for k in fields)].append(r)
    out = []
    for k, members in groups.items():
        names = sorted(set().union(*(m[value_field] for m in members)))
        out.append({**dict(zip(fields, k)), 'unit_n': len(members),
                    'statistics': {n: stats([m[value_field][n] for m in members if n in m[value_field]]) for n in names}})
    return out


def sign(x):
    return (x > 0) - (x < 0)


def derive_vector(cells):
    v = [cells[n] for n in VECTOR_NAMES]
    d = decompose(v)
    values = {**{n: cells[n] for n in VECTOR_NAMES}, 'interaction_M0': d['interaction_M0'],
              'interaction_M1': d['interaction_M1'], 'three_way_interaction': d['three_way_interaction'],
              'interaction_sum': d['interaction_sum'], **{'m_flip_delta_' + n: x for n, x in d['m_flip_delta'].items()},
              **{'component_' + k: x for k, x in d['components'].items()}}
    if d['total_sum_squares']:
        values['residual_fraction'] = d['residual_sum_squares'] / d['total_sum_squares']
    reversal = {n: v[i] * v[4 + i] < 0 for i, n in enumerate(CELL_NAMES)}
    main_ref = {n: sign(v[i]) == TEMPLATES['main_reference'][i] and sign(v[4 + i]) == TEMPLATES['main_reference'][4 + i]
                for i, n in enumerate(CELL_NAMES)}
    return values, classify(v), reversal, main_ref


def vector_summary(unit_rows, fields):
    groups = defaultdict(list)
    for r in unit_rows:
        groups[tuple(r[k] for k in fields)].append(r)
    out = []
    for k, members in groups.items():
        names = sorted(set().union(*(m['values'] for m in members)))
        out.append({**dict(zip(fields, k)), 'unit_n': len(members),
            'statistics': {n: stats([m['values'][n] for m in members if n in m['values']]) for n in names},
            'exact_template_units': dict(Counter(m['classification']['exact_template'] or 'none' for m in members)),
            'best_template_units': dict(Counter(m['classification']['best_template'] or 'none' for m in members)),
            'sign_reversal_units': {n: sum(m['sign_reversal'][n] for m in members) for n in CELL_NAMES},
            'main_reference_reversal_units': {n: sum(m['main_reference_reversal'][n] for m in members) for n in CELL_NAMES}})
    return out


def layout_summary(unit_rows, fields):
    groups = defaultdict(list)
    for r in unit_rows:
        groups[tuple(r[k] for k in fields)].append(r)
    out = []
    patterns = {'spatial_+--+': '+--+', 'reversed_-++-': '-++-', 'identity_++++': '++++'}
    for k, members in groups.items():
        names = sorted(set().union(*(m['values'] for m in members)))
        out.append({**dict(zip(fields, k)), 'unit_n': len(members),
            'statistics': {n: stats([m['values'][n] for m in members if n in m['values']]) for n in names},
            'four_cell_pattern_units': {name: sum(m['sign_pattern'] == p for m in members) for name, p in patterns.items()}})
    return out


def analyze(records, generations=(), unit_key='scene_id'):
    if not records:
        raise ValueError('No score records')
    seen, groups, native, unit_by_id = {}, defaultdict(dict), defaultdict(list), {}
    for r in records:
        k = key(r)
        if k in seen:
            raise ValueError('Duplicate score cell; select one attempt')
        if r['schema'] != DESIGN_SCHEMA:
            raise ValueError('Different scientific schema')
        if not math.isfinite(r['b_minus_a_log_odds']):
            raise ValueError('Nonfinite endpoint')
        if r['native_answer_label'] is None and r.get('native_correct') is not None:
            raise ValueError('No accuracy for hybrids/absent attributes')
        unit = unit_of(r, unit_key)
        if unit_by_id.setdefault(r['id'], unit) != unit:
            raise ValueError('One query maps to several units')
        seen[k] = r
        if r['native_source']:
            native[(r['attribute'], r['response_format'], r['mapping'], r['condition'])].append(r)
        if r['ds_code']:
            g = (r['id'], r['mapping'], r['response_format'], r['main_kind'], r['main_strength']) + tuple(r[x] for x in CELL_AXES)
            if r['ds_code'] in groups[g]:
                raise ValueError('Duplicate donor code')
            groups[g][r['ds_code']] = r
    if len({(r.get('family'), r.get('row_schema')) for r in records}) != 1:
        raise ValueError('Analyze one row family at a time')
    effects = []
    for members in groups.values():
        if not {'aaa', 'bbb'} <= set(members):
            continue
        a, b = members['aaa'], members['bbb']
        effects.append({**{x: a[x] for x in IDS}, 'unit_key': unit_key, 'unit': unit_by_id[a['id']],
            **{x: a[x] for x in ('response_format', 'main_kind', 'main_strength') + CELL_AXES},
            'effects': {VALUE: b['b_minus_a_log_odds'] - a['b_minus_a_log_odds']},
            'values': {'aaa': a['b_minus_a_log_odds'], 'bbb': b['b_minus_a_log_odds']},
            'score_flip': a['original_prediction'] != b['original_prediction'], 'template_signs': a['template_signs']})
    cells = defaultdict(dict)
    for e in effects:
        cells[(e['id'], e['mapping'], e['response_format'], e['main_kind'], e['main_strength'])][tuple(e[x] for x in CELL_AXES)] = e
    query_vectors, layout_cells, incomplete = [], [], 0
    for members in cells.values():
        first = next(iter(members.values()))
        base = {x: first[x] for x in IDS + ('unit_key', 'unit', 'response_format', 'main_kind', 'main_strength')}
        for layout in LAYOUTS:
            if all((layout, d, p) in members for d, p in CELLS):
                layout_cells.append({**base, 'main_layout': layout,
                                     'cells': {n: members[(layout, d, p)]['effects'][VALUE] for n, (d, p) in zip(CELL_NAMES, CELLS)}})
        if all(c in members for c in VECTOR):
            query_vectors.append({**base, 'cells': {n: members[c]['effects'][VALUE] for n, c in zip(VECTOR_NAMES, VECTOR)}})
        else:
            incomplete += 1
    unit_vectors = {}
    for label, fields in (('within_format_pooled', GROUP), ('per_mapping', GROUP + ('mapping',))):
        rows = []
        for u in unit_means(query_vectors, fields, 'cells'):
            values, classification, reversal, main_ref = derive_vector(u['cells'])
            rows.append({**u, 'mapping_summary': label, 'values': values, 'classification': classification,
                         'sign_reversal': reversal, 'main_reference_reversal': main_ref})
        unit_vectors[label] = rows
    unit_layouts = {}
    for label, fields in (('within_format_pooled', GROUP + ('main_layout',)), ('per_mapping', GROUP + ('main_layout', 'mapping'))):
        rows = []
        for u in unit_means(layout_cells, fields, 'cells'):
            four = [u['cells'][n] for n in CELL_NAMES]
            interaction = layout_interaction(four)
            rows.append({**u, 'mapping_summary': label, 'sign_pattern': sign_pattern(four),
                         'values': {**u['cells'], 'interaction': interaction, 'alignment_advantage': interaction / 2}})
        unit_layouts[label] = rows
    unit_effects = unit_means(effects, GROUP + CELL_AXES, 'effects')
    mapping_cells = summarize(unit_means(effects, GROUP + ('mapping',) + CELL_AXES, 'effects'), GROUP + ('mapping',) + CELL_AXES, 'effects')
    behavior, behavior_summary, gen_summary = generation_analysis(records, generations, groups, seen, unit_by_id, unit_key)
    native_rows = [dict(zip(('attribute', 'response_format', 'mapping', 'condition'), k)) | {
        'native_source': members[0]['native_source'], 'main_layout': members[0]['main_layout'],
        'n_with_gold': sum(r['native_answer_label'] is not None for r in members),
        'correct': sum(r['native_correct'] is True for r in members),
        'n_without_gold': sum(r['native_answer_label'] is None for r in members)} for k, members in native.items()]
    return {'schema': DESIGN_SCHEMA, 'family': records[0].get('family'), 'row_schema': records[0].get('row_schema'),
        'unit_key': unit_key, 'score_rows': len(records), 'generation_rows': len(generations),
        'coverage': {'query_effects': len(effects), 'complete_query_vectors': len(query_vectors),
                     'incomplete_query_vectors_missing_not_zero': incomplete, 'complete_layout_four_cells': len(layout_cells),
                     'units': len(set(unit_by_id.values()))},
        'templates': {'cell_order': list(VECTOR_NAMES), **{k: list(v) for k, v in TEMPLATES.items()},
                      'four_cell_predictions': {h: {m: list(four_cell_prediction(h, m)) for m in LAYOUTS} for h in HYPOTHESES},
                      'projection': 'component_k = (t_k . E)/(t_k . t_k) = (t_k . E)/8; three_way = I_M0 - I_M1 = 8 * component_main_reference; I_M0 + I_M1 = 8 * component_original_coordinate; cell sum = 8 * component_identity'},
        'query_effects': effects, 'query_vectors': query_vectors,
        'unit_vectors': unit_vectors['within_format_pooled'] + unit_vectors['per_mapping'],
        'unit_layout_cells': unit_layouts['within_format_pooled'] + unit_layouts['per_mapping'],
        'cell_summaries': summarize(unit_effects, GROUP + CELL_AXES, 'effects'), 'per_mapping_cell_summaries': mapping_cells,
        'vector_summaries': vector_summary(unit_vectors['within_format_pooled'], GROUP),
        'per_mapping_vector_summaries': vector_summary(unit_vectors['per_mapping'], GROUP + ('mapping',)),
        'layout_summaries': layout_summary(unit_layouts['within_format_pooled'], GROUP + ('main_layout',)),
        'per_mapping_layout_summaries': layout_summary(unit_layouts['per_mapping'], GROUP + ('main_layout', 'mapping')),
        'behavior_pairs': behavior, 'behavior_summary': behavior_summary, 'generation_summary': gen_summary,
        'native_accuracy': native_rows,
        'interpretation': 'Unit summaries (default scene). M1 reuses the same scenes, so M contrasts are paired within unit, not new samples. Label mappings averaged within unit with per-mapping summaries; direct separate. Intervals are descriptive normal approximations over unit means. Templates are simplified read-out predictions; support for main_reference does not exclude shape matching or coordinate read-out and does not prove unique object binding. Flips change behavior, not accuracy. Missing matched cells stay missing.'}


def generation_analysis(records, generations, groups, seen, unit_by_id, unit_key):
    generated = {}
    for g in generations:
        k = key(g)
        if k in generated or k not in seen:
            raise ValueError('Duplicate generation or score cell missing')
        generated[k] = g
    behavior = []
    for members in groups.values():
        if not {'aaa', 'bbb'} <= set(members):
            continue
        a, b = members['aaa'], members['bbb']
        if not (a.get('generation_planned') and b.get('generation_planned')):
            continue
        ga, gb = generated.get(key(a)), generated.get(key(b))
        valid = bool(ga and gb and ga['scorable'] and gb['scorable'])
        flip = ga['original_prediction'] != gb['original_prediction'] if valid else None
        b_choice = a.get('source_b_original_choice')  # like sign(E): bbb output = b value follows the donor identity
        direction = None if not flip or b_choice is None else 'follows_donor' if gb['original_prediction'] == b_choice else 'reversed'
        behavior.append({**{x: a[x] for x in IDS}, 'unit_key': unit_key, 'unit': unit_by_id[a['id']],
            **{x: a[x] for x in ('response_format', 'main_kind', 'main_strength') + CELL_AXES},
            'planned_pair': True, 'joint_valid': valid, 'observed_a': ga is not None, 'observed_b': gb is not None,
            'invalid_or_missing_a': not (ga and ga['scorable']), 'invalid_or_missing_b': not (gb and gb['scorable']),
            'generation_flip': flip, 'flip_direction': direction,
            'template_agreement': {h: None if direction is None else direction == ('follows_donor' if a['template_signs'][h] > 0 else 'reversed') for h in HYPOTHESES},
            'a_content': ga['content_prediction'] if ga else None, 'b_content': gb['content_prediction'] if gb else None,
            'parse_a': ga['parse']['status'] if ga else 'missing_output', 'parse_b': gb['parse']['status'] if gb else 'missing_output',
            'truncated_a': bool(ga and ga.get('truncated')), 'truncated_b': bool(gb and gb.get('truncated'))})
    summary = []
    for label, fields in (('within_format_pooled', GROUP + CELL_AXES), ('per_mapping', GROUP + ('mapping',) + CELL_AXES),
                          ('layout_total', GROUP + ('main_layout',)), ('eight_cell_total', GROUP)):
        grouped = defaultdict(list)
        for r in behavior:
            grouped[tuple(r[x] for x in fields)].append(r)
        for k, members in grouped.items():
            summary.append({**dict(zip(fields, k)), 'mapping_summary': label, 'unit_n': len({r['unit'] for r in members}),
                'planned_pairs': len(members), 'joint_valid_pairs': sum(r['joint_valid'] for r in members),
                'generation_flips': sum(r['generation_flip'] is True for r in members),
                'flips_follow_donor': sum(r['flip_direction'] == 'follows_donor' for r in members),
                'flips_reversed': sum(r['flip_direction'] == 'reversed' for r in members),
                **{'flips_agree_' + h: sum(r['template_agreement'][h] is True for r in members) for h in HYPOTHESES},
                'invalid_or_missing_a': sum(r['invalid_or_missing_a'] for r in members),
                'invalid_or_missing_b': sum(r['invalid_or_missing_b'] for r in members),
                'truncated_pairs': sum(r['truncated_a'] or r['truncated_b'] for r in members)})
    gen_groups = defaultdict(list)
    for r in records:
        if r.get('generation_planned'):
            gen_groups[(r['attribute'], r['response_format'], r['mapping'], r['condition'])].append((r, generated.get(key(r))))
    gen_summary = []
    for k, members in gen_groups.items():
        observed = [g for _, g in members if g]
        gen_summary.append(dict(zip(('attribute', 'response_format', 'mapping', 'condition'), k)) | {
            'main_layout': members[0][0]['main_layout'], 'role': members[0][0]['condition_spec']['role'] if 'condition_spec' in members[0][0] else None,
            'planned': len(members), 'observed': len(observed), 'missing': len(members) - len(observed),
            'scorable': sum(bool(g['scorable']) for g in observed), 'truncated': sum(bool(g.get('truncated')) for g in observed),
            'native_gold_planned': sum(r['native_answer_label'] is not None for r, _ in members),
            'native_correct_of_planned': sum(bool(g and g.get('native_correct') is True) for _, g in members),
            'parse_status': dict(Counter(g['parse']['status'] for g in observed)),
            'content_predictions': dict(Counter(g['content_prediction'] or 'INVALID' for g in observed))})
    return behavior, summary, gen_summary


def compare_v6(effects, reference):
    """Engineering reproducibility of v7 M0 cells against archived v6 query effects (same scenes/images/conditions)."""
    path = Path(reference) / 'query_effects.jsonl'
    old = {}
    for line in path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            k = (r['id'], r['mapping'], r['response_format'], r['main_kind'], round(r['main_strength'] * 100), r['donor_pose'], r['token_permutation'])
            if k in old:
                raise ValueError('Duplicate v6 reference effect')
            old[k] = r['effects'][VALUE]
    pairs = defaultdict(list)
    m0 = [e for e in effects if e['main_layout'] == 'M0']
    for e in m0:
        k = (e['id'], e['mapping'], e['response_format'], e['main_kind'], round(e['main_strength'] * 100), e['donor_pose'], e['token_permutation'])
        if k in old:
            pairs[e['response_format']].append((e['effects'][VALUE], old[k]))

    def agreement(values):
        if not values:
            return {'n_matched': 0}
        diffs = [abs(x - y) for x, y in values]
        out = {'n_matched': len(values), 'max_abs_difference': max(diffs), 'mean_abs_difference': mean(diffs)}
        if len(values) > 1:
            xs, ys = zip(*values); mx, my = mean(xs), mean(ys)
            sx, sy = math.sqrt(sum((x - mx) ** 2 for x in xs)), math.sqrt(sum((y - my) ** 2 for y in ys))
            out['pearson_r'] = sum((x - mx) * (y - my) for x, y in values) / (sx * sy) if sx and sy else None
        return out
    return {'reference': str(path), 'reference_sha256': sha256_file(path), 'v7_m0_effects': len(m0),
            'v6_effects': len(old), 'overall': agreement([p for v in pairs.values() for p in v]),
            'by_format': {f: agreement(v) for f, v in pairs.items()},
            'interpretation': 'Same scenes, images, queries and M0 conditions as v6: engineering reproducibility, not a new sample.'}


def load_runs(runs, include_partial=False):
    records, generations, sources, models, configs, designs = [], [], [], set(), set(), []
    for run in runs:
        meta = json.loads((run / 'metadata.json').read_text())
        models.add((meta['model']['model'], meta['model']['revision']))
        configs.add(json.dumps({k: meta.get(k) for k in SAME_RUN_FIELDS}, sort_keys=True))
        if not meta.get('finished_at') and not include_partial:
            raise ValueError('Completed runs required, or explicitly --include-partial')
        designs.append({'run': str(run), 'design': meta['design'], 'completed': bool(meta.get('finished_at')),
                        'offset': meta.get('arguments', {}).get('offset'), 'limit': meta.get('arguments', {}).get('limit'),
                        'items': meta.get('items')})
        for file, target, kind in (('results.jsonl', records, 'score'), ('generations.jsonl', generations, 'generation')):
            path = run / file
            rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            if not include_partial and len(rows) != meta['expected_' + kind + '_rows']:
                raise ValueError('Output count mismatch')
            target.extend(rows); sources.append({'path': str(path), 'sha256': sha256_file(path)})
    if len(models) != 1:
        raise ValueError('Analyze checkpoints separately')
    if len(configs) != 1:
        raise ValueError('Runs differ in manifest/design/readout/parser config; analyze separately')
    return records, generations, {'sources': sources, 'model': list(next(iter(models))), 'run_designs': designs}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--runs', type=Path, nargs='+', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--include-partial', action='store_true')
    p.add_argument('--unit-key', default='scene_id', help='Independent unit field (top level or record_fields)')
    p.add_argument('--v6-reference', type=Path, help='Archived v6 content analysis directory with query_effects.jsonl')
    a = p.parse_args(argv)
    records, generations, provenance = load_runs(a.runs, a.include_partial)
    result = analyze(records, generations, a.unit_key)
    result.update(provenance, partial_allowed=a.include_partial)
    if a.v6_reference:
        result['v6_reference'] = compare_v6(result['query_effects'], a.v6_reference)
    output = _guard_path(a.output, write=True); output.mkdir(parents=True, exist_ok=False)
    for name in ('query_effects', 'query_vectors', 'unit_vectors', 'unit_layout_cells', 'behavior_pairs'):
        (output / (name + '.jsonl')).write_text(''.join(json.dumps(row) + '\n' for row in result.pop(name)))
    (output / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'output': str(output), 'score_rows': len(records), 'generation_rows': len(generations),
                      'unit_key': a.unit_key, **result['coverage']}))


if __name__ == '__main__':
    main()
