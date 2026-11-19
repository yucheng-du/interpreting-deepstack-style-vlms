"""Single-object color intervention; CPU-only scientific condition and row contract."""
from __future__ import annotations

from copy import deepcopy
from itertools import product
import json
from pathlib import Path

from cloud_prep.measurement_v6.content_intervention import readout_specs

DESIGN_SCHEMA = 'single-object-mechanism-v8'
LAYOUTS = ('M0', 'M1')
REGIONS = ('roi', 'complement', 'all')
ENTRY_GROUPS = ('all', 'layer', 'spatial')
DEFAULT_DESIGN = {
    'strengths': [0, 8], 'response_formats': ['label', 'direct'],
    'generation_groups': [0, 1, 2, 3], 'generate_native': True,
    'generate_strengths': [0, 8], 'self_patch': True,
}


def main_source(layout, strength):
    if layout not in LAYOUTS or type(strength) is not int or not 0 <= strength <= 100:
        raise ValueError('M0/M1 and integer percentage strength required')
    return f'm_s{strength:03d}' if layout == 'M0' else f'm1_s{strength:03d}'


def read_design(path=None):
    design = deepcopy(DEFAULT_DESIGN)
    if path is not None:
        supplied = json.loads(Path(path).read_text()) if isinstance(path, (str, Path)) else path
        if not isinstance(supplied, dict) or set(supplied) - set(design):
            raise ValueError('Unknown mechanism design fields')
        design.update(supplied)
    for name in ('strengths', 'generate_strengths', 'generation_groups'):
        values = design[name]
        if not isinstance(values, list) or len(values) != len(set(values)) or any(type(x) is not int or x < 0 for x in values):
            raise ValueError('Unique nonnegative integer design values required')
    if not design['strengths'] or any(x > 100 for x in design['strengths']):
        raise ValueError('Main strengths must lie in 0..100')
    if set(design['generate_strengths']) - set(design['strengths']):
        raise ValueError('Generation strengths must be a subset')
    if not design['response_formats'] or len(set(design['response_formats'])) != len(design['response_formats']) or set(design['response_formats']) - {'label', 'direct'}:
        raise ValueError('Unique direct/label formats required')
    if any(type(design[x]) is not bool for x in ('generate_native', 'self_patch')):
        raise ValueError('Boolean design flags required')
    return design


def conditions(design=None, entry_groups=('all',), panel='main'):
    design = read_design(design)
    if panel not in ('main', 'smoke') or not entry_groups or len(set(entry_groups)) != len(entry_groups) or set(entry_groups) - set(ENTRY_GROUPS):
        raise ValueError('Invalid panel/entry groups')
    result = []
    for strength, layout in product(design['strengths'], LAYOUTS):
        main = main_source(layout, strength)
        base = {'main': main, 'main_layout': layout, 'main_strength': strength / 100,
                'main_kind': 'objects', 'donor_pose': 'same', 'token_permutation': 'identity'}
        for group, region, value in product(entry_groups, REGIONS, ('a', 'b')):
            result.append({**base, 'name': f'{main}__{group}__{region}__{value}',
                'role': 'single_object_pair', 'entry_group': group, 'region': region,
                'donor_source': f'd_{value}', 'donor_assignment': value,
                'generation': strength in design['generate_strengths'], 'technical_audit': False})
        if design['self_patch']:
            for group, region in product(entry_groups, REGIONS if panel == 'smoke' else ('all',)):
                result.append({**base, 'name': f'self_{main}__{group}__{region}', 'role': 'self_patch',
                    'entry_group': group, 'region': region, 'donor_source': main,
                    'donor_assignment': None, 'generation': False, 'technical_audit': True})
        result.append({**base, 'name': f'native_{main}', 'role': 'native_main', 'entry_group': 'none',
            'region': 'none', 'donor_source': None, 'donor_assignment': None,
            'generation': design['generate_native'], 'technical_audit': False})
    # Full-strength mains are capability controls, independent of the weak-main grid.
    for layout in LAYOUTS:
        if 100 not in design['strengths']:
            result.append({'name': f'full_{layout}', 'role': 'native_full', 'main': main_source(layout, 100),
                'main_layout': layout, 'main_kind': 'objects', 'main_strength': 1., 'entry_group': 'none',
                'region': 'none', 'donor_source': None, 'donor_assignment': None, 'donor_pose': 'same',
                'token_permutation': 'identity', 'generation': design['generate_native'], 'technical_audit': False})
    for value in ('a', 'b'):
        result.append({'name': f'native_d_{value}', 'role': 'native_donor', 'main': f'd_{value}',
            'main_layout': None, 'main_kind': 'objects', 'main_strength': 1., 'entry_group': 'none',
            'region': 'none', 'donor_source': None, 'donor_assignment': value, 'donor_pose': 'same',
            'token_permutation': 'identity', 'generation': design['generate_native'], 'technical_audit': False})
    return result


def generation_selected(row, panel='main', design=None):
    return panel == 'smoke' or row['generation_group'] in read_design(design)['generation_groups']


def execution_plan(rows, panel='main', mappings=('identity', 'cyclic'), generate_tokens=32,
                   design=None, entry_groups=('all',)):
    design = read_design(design)
    if not rows or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Need nonempty unique query rows')
    if not mappings or len(set(mappings)) != len(mappings) or set(mappings) - {'identity', 'cyclic'}:
        raise ValueError('Invalid mappings')
    cells = conditions(design, entry_groups, panel)
    readouts = readout_specs(mappings, design)
    queries = [{k: r[k] for k in ('id', 'scene_id', 'attribute', 'generation_group', 'target_object',
                'target_index', 'answer_index')} | {'generation_selected': bool(generate_tokens and generation_selected(r, panel, design))}
               for r in rows]
    gen_items = sum(q['generation_selected'] for q in queries)
    return {'schema': DESIGN_SCHEMA, 'panel': panel, 'design': design, 'entry_groups': list(entry_groups),
        'mappings': list(mappings), 'readouts': readouts, 'conditions': cells, 'queries': queries,
        'items': len(rows), 'scene_templates': len({r['scene_id'] for r in rows}),
        'expected_score_rows': len(rows) * len(readouts) * len(cells),
        'expected_generation_rows': gen_items * len(readouts) * sum(c['generation'] for c in cells),
        'generation_items': gen_items, 'generate_tokens': generate_tokens,
        'independence_unit': 'scene_id; both target counterfactuals, M, regions, groups and readouts are paired repeats',
        'roi_coordinate_contract': 'Half-open original donor pixel bbox; independent of M. Adapter reports actual per-entry/view token geometry and selected counts.',
        'correctness_scope': 'Native full/nonzero-strength images and real donors only; gray or hybrid cells have no attribute gold',
        'score_scope': 'Within-format complete candidate sequence log-probability b minus a; do not compare raw scales across models/formats',
        'effect_gate': 'No predicted sign/effect-size gate; release main on technical validity and interpret native capability separately'}


def record_fields(row, variant, condition, prediction):
    original = variant['label_to_original'][prediction] if prediction is not None else None
    native = condition['main'] if condition['role'].startswith('native_') else None
    gold_index = row['source_answer_indices'][native] if native else None
    gold = variant['original_to_label'][row['option_labels'][gold_index]] if gold_index is not None else None
    return {'schema': DESIGN_SCHEMA, 'row_schema': row['schema'], 'family': row['family'],
        **{k: row[k] for k in ('id', 'scene_id', 'task', 'attribute', 'target_object', 'target_index',
            'target_main_slot', 'target_main_slot_by_layout', 'generation_group', 'donor_roi_bbox_xyxy')},
        'record_fields': row.get('record_fields', {}), 'mapping': variant['mapping'],
        'response_format': variant['response_format'], 'condition': condition['name'], 'condition_spec': condition,
        **{k: condition[k] for k in ('role', 'main_layout', 'main_kind', 'main_strength', 'entry_group',
            'region', 'donor_source', 'donor_assignment', 'donor_pose', 'token_permutation', 'technical_audit')},
        'main_source': condition['main'], 'source_answer_indices': row['source_answer_indices'],
        'source_a_original_choice': row['option_labels'][row['answer_index']],
        'source_b_original_choice': row['option_labels'][1 - row['answer_index']],
        'original_prediction': original,
        'content_prediction': row['choices'][row['option_labels'].index(original)] if original else None,
        'native_source': native, 'native_answer_label': gold,
        'native_correct': prediction == gold if gold is not None else None}
