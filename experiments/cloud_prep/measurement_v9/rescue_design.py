"""Query-independent local ROI relocation, with changed-object color direction."""
from __future__ import annotations

from itertools import product

from cloud_prep.measurement_v6.content_intervention import readout_specs
from cloud_prep.measurement_v8.mechanism_design import read_design, main_source, DEFAULT_DESIGN, LAYOUTS
from cloud_prep.measurement_v9.rescue_data import validate_query_groups

DESIGN_SCHEMA = 'local-rescue-v9'
DESTINATIONS = ('original', 'opposite')
ENTRY_GROUPS = ('all',)


def conditions(design=None, entry_groups=('all',), panel='main'):
    design = read_design(design)
    if panel not in ('main','smoke') or tuple(entry_groups) != ENTRY_GROUPS:
        raise ValueError('Rescue uses main/smoke and all later entries only; IBM layer0 remains main')
    result = []
    for strength, layout in product(design['strengths'], LAYOUTS):
        source = main_source(layout,strength)
        base = {'main': source, 'main_layout': layout, 'main_kind': 'objects',
                'main_strength': strength/100, 'donor_pose': 'same', 'entry_group': 'all', 'region': 'roi'}
        for destination, assignment in product(DESTINATIONS, ('a','b')):
            permutation = 'identity' if destination == 'original' else 'swap_halves'
            result.append({**base, 'name': f'{source}__all__roi__{permutation}__{assignment}',
                'role': 'local_rescue_pair', 'source_destination': destination,
                'token_permutation': permutation, 'donor_source': f'd_{assignment}', 'donor_assignment': assignment,
                'generation': strength in design['generate_strengths'], 'technical_audit': False})
        if design['self_patch']:
            result.append({**base, 'name': f'self_{source}__all__roi', 'role': 'self_patch',
                'source_destination': 'original', 'token_permutation': 'identity',
                'donor_source': source, 'donor_assignment': None, 'generation': False, 'technical_audit': True})
        result.append({**base, 'name': f'native_{source}', 'role': 'native_main', 'entry_group': 'none',
            'region': 'none', 'source_destination': None, 'token_permutation': 'identity', 'donor_source': None,
            'donor_assignment': None, 'generation': design['generate_native'], 'technical_audit': False})
    for layout in LAYOUTS:
        if 100 not in design['strengths']:
            result.append({'name': f'full_{layout}', 'role': 'native_full', 'main': main_source(layout,100),
                'main_layout': layout, 'main_kind': 'objects', 'main_strength': 1., 'entry_group': 'none',
                'region': 'none', 'source_destination': None, 'token_permutation': 'identity',
                'donor_source': None, 'donor_assignment': None, 'donor_pose': 'same',
                'generation': design['generate_native'], 'technical_audit': False})
    for assignment in ('a','b'):
        result.append({'name': f'native_d_{assignment}', 'role': 'native_donor', 'main': f'd_{assignment}',
            'main_layout': 'M0', 'main_kind': 'objects', 'main_strength': 1., 'entry_group': 'none',
            'region': 'none', 'source_destination': None, 'token_permutation': 'identity',
            'donor_source': None, 'donor_assignment': assignment, 'donor_pose': 'same',
            'generation': design['generate_native'], 'technical_audit': False})
    return result


def generation_selected(row, panel='main', design=None):
    return panel == 'smoke' or row['generation_group'] in read_design(design)['generation_groups']


QUERY_FIELDS = ('id','scene_id','attribute','generation_group','intervention_id','changed_object','changed_index',
    'queried_object','queried_index','query_role','target_object','target_index','answer_index',
    'direction_a_index','direction_b_index','source_roi_slot','changed_main_slot_by_layout',
    'queried_main_slot_by_layout','donor_roi_bbox_xyxy','visual_source_signature','source_answer_indices')


def execution_plan(rows, panel='main', mappings=('identity','cyclic'), generate_tokens=32,
                   design=None, entry_groups=('all',)):
    design = read_design(design)
    validate_query_groups(rows)
    if not mappings or len(set(mappings)) != len(mappings) or set(mappings)-{'identity','cyclic'}:
        raise ValueError('Unique identity/cyclic mappings required')
    if type(generate_tokens) is not int or generate_tokens < 0:
        raise ValueError('Nonnegative generation token budget required')
    cells = conditions(design,entry_groups,panel)
    readouts = readout_specs(mappings,design)
    queries = [{**{k:r[k] for k in QUERY_FIELDS},
        'generation_selected': bool(generate_tokens and generation_selected(r,panel,design))} for r in rows]
    n_gen = sum(q['generation_selected'] for q in queries)
    return {'schema': DESIGN_SCHEMA, 'panel': panel, 'design': design, 'entry_groups': list(entry_groups),
        'conditions': cells, 'readouts': readouts, 'mappings': list(mappings), 'queries': queries,
        'items': len(rows), 'scene_templates': len({r['scene_id'] for r in rows}),
        'expected_score_rows': len(rows)*len(cells)*len(readouts),
        'expected_generation_rows': n_gen*sum(c['generation'] for c in cells)*len(readouts),
        'generation_items': n_gen, 'generate_tokens': generate_tokens,
        'independence_unit': 'scene_id; changed object, query, layout, destination and mapping are paired repeats',
        'scene_reuse': 'Accepted 64 v8 mechanism scenes; v9 query design is incremental, not new scene replication',
        'direction_contract': 'direction_a_index/direction_b_index belong to changed_object for both queries',
        'roi_coordinate_contract': 'Fixed changed-object donor M0 bbox; swap_halves moves features and mask jointly',
        'main_contract': 'Main entry, IBM layer0, nonselected ROI tokens and nonselected later entries remain main',
        'correctness_scope': 'Native labels follow queried object and source; hybrids and alpha0 have no attribute gold',
        'score_scope': 'Within-readout complete candidate sequence log-probability of changed-b minus changed-a color',
        'effect_gate': 'No sign, effect-size or native-accuracy filtering of the planned scene panel'}


def record_fields(row, variant, condition, prediction):
    original = variant['label_to_original'][prediction] if prediction is not None else None
    native = condition['main'] if condition['role'].startswith('native_') else None
    gold_index = row['source_answer_indices'][native] if native else None
    gold = variant['original_to_label'][row['option_labels'][gold_index]] if gold_index is not None else None
    destination = condition['source_destination']
    slot = row['source_roi_slot'] ^ (destination == 'opposite') if destination else None
    layout = condition['main_layout']
    occupant = next((obj for obj,s in row['scene']['object_to_slot'].items()
        if s ^ (layout == 'M1') == slot), None) if slot is not None else None
    source_box = row['donor_roi_bbox_xyxy']
    dx = (1 if row['source_roi_slot'] == 0 else -1)*row['donor_roi_image_size'][0]//2
    dest_box = [source_box[i]+(dx if destination == 'opposite' and i in (0,2) else 0) for i in range(4)] if destination else None
    return {'schema': DESIGN_SCHEMA, 'row_schema': row['schema'], 'family': row['family'],
        **{k:row[k] for k in QUERY_FIELDS}, 'task': row['task'],
        'target_main_slot': row['target_main_slot'], 'target_main_slot_by_layout': row['target_main_slot_by_layout'],
        'record_fields': row.get('record_fields',{}), 'mapping': variant['mapping'],
        'response_format': variant['response_format'], 'condition': condition['name'], 'condition_spec': condition,
        **{k:condition[k] for k in ('role','main_layout','main_kind','main_strength','entry_group','region',
            'donor_source','donor_assignment','donor_pose','token_permutation','source_destination','technical_audit')},
        'main_source': condition['main'], 'destination_slot': slot, 'destination_object': occupant,
        'destination_roi_bbox_xyxy': dest_box,
        'destination_matches_queried': occupant == row['queried_object'] if occupant else None,
        'destination_matches_changed': occupant == row['changed_object'] if occupant else None,
        'source_a_original_choice': row['option_labels'][row['direction_a_index']],
        'source_b_original_choice': row['option_labels'][row['direction_b_index']],
        'original_prediction': original,
        'content_prediction': row['choices'][row['option_labels'].index(original)] if original else None,
        'native_source': native, 'native_answer_label': gold,
        'native_correct': prediction == gold if gold is not None else None}
