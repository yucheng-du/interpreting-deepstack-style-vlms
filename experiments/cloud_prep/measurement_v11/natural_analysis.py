"""Paired role/layout/strength effects on new runs, clustered by source photo.

The frozen v10 analyzer handles each role's a/b and native diagnostics after a
validated schema projection. Cross-role contrasts use only this v11 execution
plan: archived v10 outputs cannot enter formal paired estimates.
"""
from collections import defaultdict
from itertools import combinations, product
import argparse
import json
from pathlib import Path

from cloud_prep.measurement_v8.mechanism_analysis import (
    finite_score, valid_generation, key, summarize,
)
from cloud_prep.measurement_v10 import natural_analysis as old_analysis
from cloud_prep.measurement_v10 import natural_design as old_design
from cloud_prep.measurement_v11 import natural_design as design
from cloud_prep.measurement_v10.natural_data import validate_query_groups
from cloud_prep.portable_data import _guard_path

GROUP = ('response_format', 'changed_entry', 'main_layout', 'main_strength', 'query_role')
AXES = {
    'role': ('changed_entry', ('later', 'main'), 'main_minus_later'),
    'layout': ('main_layout', ('M0', 'M1'), 'M1_minus_M0'),
    'strength': ('main_strength', (.08, 1.), '1_minus_0.08'),
    'query': ('query_role', ('unchanged', 'changed'), 'changed_minus_unchanged'),
}


def _validate_plan(plan):
    if not plan or plan.get('schema') != design.DESIGN_SCHEMA:
        raise ValueError('Fixed v11 natural execution plan required; archived v10 is reference only')
    validate_query_groups(plan['queries'])
    if len({q['id'] for q in plan['queries']}) != len(plan['queries']):
        raise ValueError('Duplicate planned query')
    wanted = design.conditions(plan['design'], plan['entry_groups'], plan['panel'])
    if plan['conditions'] != wanted:
        raise ValueError('Noncanonical paired role/source condition plan')
    expected = {}; generation = {}
    for q, r, c in product(plan['queries'], plan['readouts'], plan['conditions']):
        k = q['id'], r['response_format'], r['mapping'], c['name']
        if k in expected:
            raise ValueError('Duplicate planned cell')
        expected[k] = q, r, c
        if q['generation_selected'] and c['generation']:
            generation[k] = expected[k]
    if (len(expected), len(generation)) != (plan['expected_score_rows'], plan['expected_generation_rows']):
        raise ValueError('Expected row count differs from full condition matrix')
    return expected, generation


def _index(rows, expected, kind):
    indexed = {}; source_hashes = {}
    for row in rows:
        k = key(row)
        if k in indexed:
            raise ValueError(f'Duplicate {kind} cell')
        if k not in expected:
            raise ValueError(f'Unplanned {kind} cell')
        q, r, c = expected[k]
        if row.get('schema') != design.DESIGN_SCHEMA:
            raise ValueError('Cannot combine archived v10 rows with the v11 paired run')
        if row.get('condition_spec') != c or row.get('main_source') != c['main']:
            raise ValueError('Full condition/source mismatch')
        for f in design.QUERY_FIELDS:
            if row.get(f) != q[f]:
                raise ValueError(f'Query/ROI identity mismatch: {f}')
        for f in ('changed_entry', 'context_source', 'intervention_type',
                  'v10_bridge_condition', 'varying_entry', 'donor_source',
                  'donor_assignment', 'role', 'main_layout', 'main_strength',
                  'region', 'entry_group', 'token_permutation'):
            if row.get(f) != c[f]:
                raise ValueError(f'Condition identity mismatch: {f}')
        if (row.get('source_a_original_choice'), row.get('source_b_original_choice')) != ('(A)', '(B)'):
            raise ValueError('Direction must remain red=a to blue=b for both roles and queries')
        hashes = {f: row[f] for f in ('source_pixel_sha256', 'source_image_sha256') if f in row}
        hk = q['scene_id'], r['response_format'], r['mapping']
        if hk in source_hashes and source_hashes[hk] != hashes:
            raise ValueError('Source images changed across roles/queries/conditions')
        source_hashes[hk] = hashes
        indexed[k] = row
    return indexed


def _project(plan, rows, entry):
    """Lossless scientific projection; original records have already validated."""
    old_cells = {c['name']: c for c in old_design.conditions(plan['design'], plan['entry_groups'], plan['panel'])}
    names = {}
    for c in plan['conditions']:
        if c['changed_entry'] == entry:
            names[c['name']] = c['name'].split('__', 1)[1]
        elif c['role'].startswith('native_'):
            names[c['name']] = c['name']
    projected = []
    for row in rows:
        if row['condition'] not in names:
            continue
        c = old_cells[names[row['condition']]]
        projected.append({**row, 'schema': old_design.DESIGN_SCHEMA,
            'condition': c['name'], 'condition_spec': c,
            **{f: c[f] for f in ('varying_entry', 'role', 'main_layout', 'main_strength',
                'entry_group', 'region', 'donor_source', 'donor_assignment', 'token_permutation')}})
    p = {**plan, 'schema': old_design.DESIGN_SCHEMA, 'conditions': list(old_cells.values()),
        'expected_score_rows': len(plan['queries'])*len(plan['readouts'])*len(old_cells),
        'expected_generation_rows': sum(q['generation_selected'] for q in plan['queries'])*len(plan['readouts'])*sum(c['generation'] for c in old_cells.values())}
    return p, projected


def factorial_contrasts(effects):
    """Differences are formed per photo/readout/mapping before cluster means.

    Every nonempty subset of available two-level factors is reported. Query
    differences use only photos with an explicitly planned off-target query.
    Absent observed cells remain None and cannot change within-photo weighting.
    """
    groups = defaultdict(list)
    for e in effects:
        groups[e['scene_id'], e['response_format'], e['mapping']].append(e)
    result = []
    for entries in groups.values():
        first = entries[0]
        active = [a for a, (f, levels, _) in AXES.items()
                  if set(x[f] for x in entries) == set(levels)]
        for size in range(1, len(active)+1):
            for axes in combinations(active, size):
                varied = [AXES[a][0] for a in axes]
                fixed = [f for f, _, _ in AXES.values() if f not in varied]
                cells = defaultdict(dict)
                for e in entries:
                    fk = tuple(e[f] for f in fixed)
                    vk = tuple(e[f] for f in varied)
                    if vk in cells[fk]:
                        raise ValueError('Duplicate planned factorial cell')
                    cells[fk][vk] = e
                for fk, members in cells.items():
                    values = []; support = []
                    for bits in product((0, 1), repeat=len(axes)):
                        vk = tuple(AXES[a][1][bit] for a, bit in zip(axes, bits))
                        if vk not in members:
                            raise ValueError('Planned factorial support is incomplete')
                        e = members[vk]; sign = (-1)**sum(bit == 0 for bit in bits)
                        values.append(sign*e['effect'] if e['effect'] is not None else None)
                        support.append({'id': e['id'], 'changed_entry': e['changed_entry'],
                            'main_layout': e['main_layout'], 'main_strength': e['main_strength'],
                            'query_role': e['query_role'], 'coefficient': sign})
                    result.append({**{f:first[f] for f in ('scene_id','response_format','mapping','has_unchanged_query')},
                        **dict(zip(fixed, fk)), **{AXES[a][0]: AXES[a][2] for a in axes},
                        'contrast': '_by_'.join(axes) + ('_difference' if size == 1 else ''),
                        'effect': sum(values) if all(v is not None for v in values) else None,
                        'planned_effect_terms': len(values),
                        'available_effect_terms': sum(v is not None for v in values),
                        'paired_support': support})
    return result


def _behavior_coverage(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[f] for f in GROUP)+ (row['mapping'],)].append(row)
    result = []
    for group, members in groups.items():
        valid = sum(x['both_valid'] for x in members)
        flips = sum(x['flip'] is True for x in members)
        result.append({**dict(zip(GROUP+('mapping',), group)),
            'planned_pairs':len(members), 'common_valid_pairs':valid, 'flips':flips,
            'flip_rate_common_valid':flips/valid if valid else None,
            'flips_over_planned_lower_bound':flips/len(members),
            'missing_pairs':sum(x['missing_rows'] > 0 for x in members),
            'invalid_pairs':sum(x['invalid_rows'] > 0 for x in members),
            'truncated_pairs':sum(x['truncated_rows'] > 0 for x in members)})
    return result


def analyze(records, generations=(), expected_plan=None):
    plan = expected_plan
    expected, expected_gen = _validate_plan(plan)
    scores = _index(records, expected, 'score')
    gens = _index(generations, expected_gen, 'generation')
    # CPU fixture plans omit execution metadata. Real runner plans pin both.
    for row in (*scores.values(), *gens.values()):
        if plan.get('execution_schema') and row.get('execution_schema') != plan['execution_schema']:
            raise ValueError('Execution schema differs across the formal paired run')
        if plan.get('model'):
            from cloud_prep.measurement_v8.model_registry import get_model
            if get_model(row.get('model_key'))['key'] != plan['model']['key']:
                raise ValueError('Model differs across the formal paired run')
    reports = {}
    for entry in ('main', 'later'):
        p, s = _project(plan, scores.values(), entry)
        _, g = _project(plan, gens.values(), entry)
        reports[entry] = old_analysis.analyze(s, g, p)
    effects = [{**e, 'changed_entry': entry} for entry, report in reports.items() for e in report['query_effects']]
    behavior = [{**e, 'changed_entry': entry} for entry, report in reports.items() for e in report['behavior_pairs']]
    contrasts = factorial_contrasts(effects)
    gen_contrasts = factorial_contrasts(behavior)
    shared = reports['main']
    native_keys = ('native_accuracy', 'native_donor_effects', 'native_donor_behavior',
        'native_donor_summaries', 'per_mapping_native_donor_summaries',
        'native_donor_behavior_summaries', 'per_mapping_native_donor_behavior_summaries')
    return {'schema':design.DESIGN_SCHEMA, 'unit_key':'scene_id',
        'coverage':{
            'planned_source_photos':len({q['scene_id'] for q in plan['queries']}),
            'paired_off_target_source_photos':len({q['scene_id'] for q in plan['queries'] if q['has_unchanged_query']}),
            'planned_score_rows':len(expected), 'present_score_rows':len(scores),
            'missing_score_rows':len(expected)-len(scores),
            'nonfinite_score_rows':sum(not finite_score(x) for x in scores.values()),
            'planned_generation_rows':len(expected_gen), 'present_generation_rows':len(gens),
            'missing_generation_rows':len(expected_gen)-len(gens),
            'invalid_generation_rows':sum(not valid_generation(x) for x in gens.values()),
            'truncated_generation_rows':sum(bool(x.get('truncated')) for x in gens.values()),
            'planned_pairs':len(effects), 'complete_pairs':sum(e['effect'] is not None for e in effects),
            'planned_generation_pairs':len(behavior), 'valid_generation_pairs':sum(e['both_valid'] for e in behavior),
            'generation_flips':sum(e['flip'] is True for e in behavior)},
        'query_effects':effects, 'query_contrasts':contrasts,
        'behavior_pairs':behavior, 'behavior_contrasts':gen_contrasts,
        'cell_summaries':summarize(effects,GROUP),
        'per_mapping_cell_summaries':summarize(effects,GROUP+('mapping',)),
        'contrast_summaries':summarize(contrasts,('contrast',)+GROUP),
        'per_mapping_contrast_summaries':summarize(contrasts,('contrast',)+GROUP+('mapping',)),
        'behavior_summaries':summarize(behavior,GROUP),
        'per_mapping_behavior_summaries':summarize(behavior,GROUP+('mapping',)),
        'behavior_contrast_summaries':summarize(gen_contrasts,('contrast',)+GROUP),
        'per_mapping_behavior_contrast_summaries':summarize(gen_contrasts,('contrast',)+GROUP+('mapping',)),
        'behavior_coverage':_behavior_coverage(behavior),
        'self_patch_audits':[{**e,'changed_entry':entry} for entry, report in reports.items() for e in report['self_patch_audits']],
        **{k:shared[k] for k in native_keys},
        'definitions':{**shared['definitions'],
            'role_difference':'E(main)-E(later), new same-backend paired runs only',
            'strength_difference':'E(alpha=1)-E(alpha=.08); all interactions are signed differences of these E values',
            'scope':'ROI feature replacement; original-photo bbox/color-region and contextualized receptive-field limitations remain',
            'bridge':'Frozen v10 later results are reference only and never merged into paired v11 role estimates',
            'independence':'Photo-level means require every planned within-photo mapping/repeat; off-target interactions use the six declared photos'}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,required=True)
    parser.add_argument('--generations',type=Path)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args(argv)
    read = lambda p: [json.loads(x) for x in _guard_path(p).read_text().splitlines() if x]
    report = analyze(read(args.results),read(args.generations) if args.generations else [],json.loads(_guard_path(args.plan).read_text()))
    output = _guard_path(args.output,write=True)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2)+'\n')


if __name__ == '__main__':
    main()
