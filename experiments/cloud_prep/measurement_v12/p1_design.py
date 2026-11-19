"""Natural same-intervention selectivity with canonical native-dose matching."""
from itertools import product
from cloud_prep.measurement_v6.content_intervention import readout_specs
from cloud_prep.measurement_v9.runtime import stable_digest, visual_contract
from cloud_prep.measurement_v10.natural_design import QUERY_FIELDS as NATURAL_FIELDS
from cloud_prep.measurement_v12.v11_v2_runtime import evaluation_plan
from cloud_prep.measurement_v12.v11_v2_dose_design import DEFAULT_DESIGN as V11_DEFAULT

DESIGN_SCHEMA = 'natural-selectivity-dose-v12-v1'
QUERY_FIELDS = tuple(dict.fromkeys((*NATURAL_FIELDS, 'query_type', 'query_target')))
DEFAULT_DESIGN = {'response_formats': ['label', 'direct'], 'strengths': [100, 8],
                  'max_calibrated_k': V11_DEFAULT['max_calibrated_k'],
                  'match_tolerance': V11_DEFAULT['match_tolerance']}


def references(role, layout, strength):
    context = f'{layout.lower()}_s{strength:03d}'
    return tuple({'main': context, 'later': context, 'changed_entry': role,
                  'donor_source': f'd_{side}'} for side in ('a', 'b'))


def calibration_id(role, layout, strength):
    return f'{role}__{layout}_{strength:03d}__natural_roi'


def conditions():
    cells = []
    for role, layout, strength in product(('main', 'later'), ('M0', 'M1'), (100, 8)):
        ra, rb = references(role, layout, strength)
        cid = calibration_id(role, layout, strength)
        base = dict(calibration_id=cid, changed_entry=role, changing_layout=layout,
            fixed_layout=layout, changing_strength=strength/100, fixed_strength=strength/100,
            fixed_other_assignment='original_context', main=ra['main'], donor_source='d_a',
            context_source=ra['main'], reference_a=ra, reference_b=rb,
            main_layout=layout, later_layout=layout, main_strength=strength/100, later_strength=strength/100,
            region='roi', entry_group='all', token_permutation='identity', main_kind='whole_scene',
            donor_pose='original_pixels', source_destination='original', technical_audit=False,
            max_calibrated_k=DEFAULT_DESIGN['max_calibrated_k'], match_tolerance=DEFAULT_DESIGN['match_tolerance'])
        settings = [('raw', 'raw', 0., 'fixed'), ('raw', 'raw', 1., 'fixed')]
        if layout == 'M0' and strength == 100:
            settings += [('baseline_state', metric, 1., 'opposite_native') for metric in ('rss', 'l1')]
        for mode, metric, value, policy in settings:
            pid = f'{cid}__{mode}_{metric}_{value:g}'
            for side in ('a', 'b'):
                cells.append({**base, 'name': f'{pid}__{side}', 'pair_id': pid,
                    'role': 'dose_pair', 'dose_mode': mode, 'dose_metric': metric,
                    'dose_value': value, 'target_policy': policy, 'donor_assignment': side, 'generation': True})
        cells.append({**base, 'name': f'self_{cid}', 'role': 'self_patch',
                      'donor_source': ra['main'], 'donor_assignment': None,
                      'intervention_type': 'local_role', 'generation': False, 'technical_audit': True})
    for name in ('m0_s100', 'm0_s008', 'm1_s100', 'm1_s008', 'd_a', 'd_b'):
        cells.append({'name': f'native_{name}', 'role': 'native_source', 'main': name,
                      'donor_source': None, 'context_source': name, 'region': 'none',
                      'entry_group': 'none', 'token_permutation': 'identity',
                      'changed_entry': None, 'generation': True, 'technical_audit': False})
    return cells


def required_sources(cells):
    sources = {c['main'] for c in cells}
    for c in cells:
        for ref in ('reference_a', 'reference_b'):
            if ref in c:
                sources.update(c[ref][key] for key in ('main', 'later', 'donor_source'))
        if c.get('donor_source'): sources.add(c['donor_source'])
    return sources


def canonical_anchors(rows):
    anchors = {}
    for row in rows:
        if row['query_type'] == 'identity' and row['query_target'] == 'changed':
            if row['intervention_id'] in anchors: raise ValueError('Duplicate canonical anchor')
            anchors[row['intervention_id']] = row
    if set(anchors) != {r['intervention_id'] for r in rows}: raise ValueError('Missing anchor query')
    return anchors


def anchor_query_sha256(row):
    return stable_digest({'id': row['id'], 'query': row['query'], 'choices': row['choices'],
                          'option_labels': row['option_labels'], 'visual': visual_contract(row)})


def calibration_key(row, condition):
    return row['intervention_id'], condition['calibration_id']


def context_signature(row, condition):
    return stable_digest({'visual': visual_contract(row), 'context': condition['context_source'],
                          'reference_direction': ['d_a', 'd_b'], 'roi': row['donor_roi_bbox_xyxy']})


def visual_signature(row, condition):
    return stable_digest({'context': context_signature(row, condition),
        'intervention': row['intervention_id'], 'calibration_id': condition['calibration_id'],
        'reference_a': condition['reference_a'], 'reference_b': condition['reference_b']})


def opposite_condition(c):
    role = 'later' if c['changed_entry'] == 'main' else 'main'
    strength = round(c['main_strength'] * 100); layout = c['main_layout']
    ra, rb = references(role, layout, strength)
    return {**c, 'changed_entry': role, 'calibration_id': calibration_id(role, layout, strength),
            'reference_a': ra, 'reference_b': rb}


def generation_selected(row, panel='main', design=None): return True


def execution_plan(rows, panel='main', mappings=('identity', 'cyclic'), generate_tokens=32):
    from cloud_prep.measurement_v12.p1_prepare import validate_query_groups
    validate_query_groups(rows)
    anchors = canonical_anchors(rows)
    if set(mappings) != {'identity', 'cyclic'} or len(mappings) != 2: raise ValueError('Retain both mappings')
    cells = conditions(); readouts = readout_specs(mappings, DEFAULT_DESIGN)
    queries = [{**{k: r[k] for k in QUERY_FIELDS}, 'query_sha256':anchor_query_sha256(r),
                'generation_selected': bool(generate_tokens)} for r in rows]
    plan = {'schema': DESIGN_SCHEMA, 'design': DEFAULT_DESIGN, 'panel': panel, 'conditions': cells,
        'queries': queries, 'readouts': readouts, 'items': len(rows), 'mappings': list(mappings),
        'entry_groups': ['all'], 'generate_tokens': generate_tokens,
        'expected_score_rows': len(rows)*len(cells)*len(readouts),
        'expected_generation_rows': len(rows)*sum(c['generation'] for c in cells)*len(readouts) if generate_tokens else 0,
        'expected_anchor_forwards': len(anchors)*len({c['calibration_id'] for c in cells if c['role']=='dose_pair'}),
        'calibration_anchors': {k: r['id'] for k, r in anchors.items()},
        'independent_source_photos': len({r['scene_id'] for r in rows}),
        'independence_unit': 'scene_id = source photo; two edits and two queries are paired repeats',
        'shared_intervention': 'Fixed source ROI/difference/k across target/control and all three readouts',
        'normalization': 'a-side post-entry state at one canonical changed/direct anchor per edit and role/context',
        'target_scope': 'Native-target RSS/L1 matching only M0 alpha1; other contexts raw k0/k1 diagnostics',
        'estimands': ['E_target','E_control','R_target_minus_control','R_main_minus_later',
                     'same-native-target main_minus_later, separately by RSS/L1'],
        'success_rule': 'Report both role selectivities and their paired difference; no significance or equivalence gate',
        'new_model_results': False}
    plan.update(evaluation_plan(rows, plan))
    return plan


def record_fields(row, variant, condition, prediction):
    original = variant['label_to_original'][prediction] if prediction is not None else None
    return {**{k: row[k] for k in QUERY_FIELDS}, 'schema': DESIGN_SCHEMA,
        'row_schema': row['schema'], 'family': row['family'], 'task': row['task'],
        'query_sha256': anchor_query_sha256(row), 'mapping': variant['mapping'],
        'response_format': variant['response_format'], 'condition': condition['name'],
        'condition_spec': condition, 'main_source': condition['main'],
        **{k: condition.get(k) for k in ('role','pair_id','calibration_id','changed_entry','main_layout',
             'main_strength','changing_layout','fixed_layout','changing_strength','fixed_strength',
             'fixed_other_assignment','dose_mode','dose_value','target_policy','dose_metric','donor_assignment','region','entry_group')},
        'source_a_original_choice': row['option_labels'][row['direction_a_index']],
        'source_b_original_choice': row['option_labels'][row['direction_b_index']],
        'original_prediction': original,
        'content_prediction': row['choices'][row['option_labels'].index(original)] if original else None}
