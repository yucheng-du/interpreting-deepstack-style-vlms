"""Three-object cyclic-layout intervention; planning and records are torch-free."""
from __future__ import annotations
from copy import deepcopy
from itertools import product
import json
from pathlib import Path
from cloud_prep.measurement_v6.content_intervention import readout_specs
from cloud_prep.measurement_v8.mechanism_design import read_design as _read_design, main_source

DESIGN_SCHEMA = 'multiobject-generalization-v9'
PERMUTATIONS = ('identity', 'cycle_forward', 'cycle_inverse')
DEFAULT_DESIGN = {'strengths': [0, 8], 'response_formats': ['label', 'direct'],
    'generation_groups': [0, 1, 2, 3], 'generate_native': True,
    'generate_strengths': [0, 8], 'self_patch': True}


def read_design(path=None):
    return _read_design(path)


def tile_permutation(tile_grid, name):
    if list(tile_grid) not in ([1, 3], [2, 2]) or name not in PERMUTATIONS:
        raise ValueError('Expected 1x3/2x2 grid and declared cycle')
    p = {'identity': [0, 1, 2], 'cycle_forward': [2, 0, 1], 'cycle_inverse': [1, 2, 0]}[name]
    return p + ([3] if list(tile_grid) == [2, 2] else [])


def conditions(design=None, entry_groups=('all',), panel='main'):
    d = read_design(design)
    if panel not in ('main', 'smoke') or not entry_groups or set(entry_groups) - {'all', 'layer', 'spatial'} or len(set(entry_groups)) != len(entry_groups):
        raise ValueError('Invalid panel/entry groups')
    cells = []
    for strength, layout in product(d['strengths'], ('M0', 'M1')):
        src = main_source(layout, strength)
        base = dict(main=src, main_layout=layout, main_kind='objects', main_strength=strength / 100,
                    donor_pose='same', region='all', technical_audit=False)
        for group, perm, value in product(entry_groups, PERMUTATIONS, ('a', 'b')):
            alignment = 'correct' if perm == ('identity' if layout == 'M0' else 'cycle_forward') else 'incorrect'
            cells.append(dict(base, name=f'{src}__{group}__{perm}__{value}', role='multiobject_pair',
                entry_group=group, donor_source=f'd_{value}', donor_assignment=value,
                token_permutation=perm, alignment=alignment, generation=strength in d['generate_strengths']))
        if d['self_patch']:
            for group in entry_groups:
                cells.append(dict(base, name=f'self_{src}__{group}', role='self_patch', entry_group=group,
                    donor_source=src, donor_assignment=None, token_permutation='identity', alignment=None,
                    generation=False, technical_audit=True))
        cells.append(dict(base, name=f'native_{src}', role='native_main', entry_group='none', region='none',
            donor_source=None, donor_assignment=None, token_permutation='identity', alignment=None,
            generation=d['generate_native']))
    for layout in ('M0', 'M1'):
        if 100 not in d['strengths']:
            cells.append(dict(name=f'full_{layout}', role='native_full', main=main_source(layout, 100),
                main_layout=layout, main_kind='objects', main_strength=1., donor_pose='same', region='none',
                entry_group='none', donor_source=None, donor_assignment=None, token_permutation='identity',
                alignment=None, generation=d['generate_native'], technical_audit=False))
    for value in ('a', 'b'):
        cells.append(dict(name=f'native_d_{value}', role='native_donor', main=f'd_{value}', main_layout='M0',
            main_kind='objects', main_strength=1., donor_pose='same', region='none', entry_group='none',
            donor_source=None, donor_assignment=value, token_permutation='identity', alignment=None,
            generation=d['generate_native'], technical_audit=False))
    return cells


def generation_selected(row, panel='main', design=None):
    return panel == 'smoke' or row['generation_group'] in read_design(design)['generation_groups']


QUERY_FIELDS = ('id', 'scene_id', 'unit_id', 'source_cluster_id', 'panel_family', 'geometry', 'tile_grid',
    'attribute', 'generation_group', 'target_object', 'target_index', 'answer_index',
    'direction_a_index', 'direction_b_index', 'source_answer_indices')


def execution_plan(rows, panel='main', mappings=('identity', 'cyclic'), generate_tokens=32, design=None, entry_groups=('all',)):
    d = read_design(design)
    if not rows or len({r['id'] for r in rows}) != len(rows) or not mappings or len(set(mappings)) != len(mappings) or set(mappings) - {'identity', 'cyclic'}:
        raise ValueError('Need unique rows and mappings')
    cells, readouts = conditions(d, entry_groups, panel), readout_specs(mappings, d)
    queries = [{**{k: r[k] for k in QUERY_FIELDS},
        'generation_selected': bool(generate_tokens and generation_selected(r, panel, d))} for r in rows]
    gen_items = sum(q['generation_selected'] for q in queries)
    return dict(schema=DESIGN_SCHEMA, panel=panel, design=d, entry_groups=list(entry_groups),
        mappings=list(mappings), readouts=readouts, conditions=cells, queries=queries, items=len(rows),
        scene_templates=len({r['scene_id'] for r in rows}), independent_units=len({r['unit_id'] for r in rows}),
        expected_score_rows=len(rows) * len(readouts) * len(cells),
        expected_generation_rows=gen_items * len(readouts) * sum(c['generation'] for c in cells),
        generation_items=gen_items, generate_tokens=generate_tokens,
        independence_unit='unit_id: synthetic scene; natural source-photo/crop cluster with paired geometries',
        score_scope='Within-format donor-b minus donor-a of log P(b color)-log P(a color); no native-success filtering',
        geometry_scope='Three occupied equal tiles; M1 forward-cycles them; fourth 2x2 tile is fixed blank. Output tile gathers source tile.',
        correctness_scope='Native nonzero-strength mains/full donors only; grayscale and hybrid cells have no gold',
        effect_gate='No directional or magnitude threshold; technical smoke and native capability reported separately')


def record_fields(row, variant, condition, prediction):
    original = variant['label_to_original'].get(prediction) if prediction is not None else None
    native = condition['main'] if condition['role'].startswith('native_') else None
    gold_index = row['source_answer_indices'][native] if native else None
    gold = variant['original_to_label'][row['option_labels'][gold_index]] if gold_index is not None else None
    fields = {k: row[k] for k in QUERY_FIELDS}
    fields.update({k: row[k] for k in ('target_main_slot', 'target_main_slot_by_layout', 'donor_roi_bbox_xyxy',
        'changed_object', 'queried_object', 'task')})
    return dict(fields, schema=DESIGN_SCHEMA, row_schema=row['schema'], family=row['family'],
        record_fields=row['record_fields'], mapping=variant['mapping'], response_format=variant['response_format'],
        condition=condition['name'], condition_spec=condition,
        **{k: condition[k] for k in ('role', 'main_layout', 'main_kind', 'main_strength', 'entry_group', 'region',
            'donor_source', 'donor_assignment', 'donor_pose', 'token_permutation', 'alignment', 'technical_audit')},
        tile_permutation=tile_permutation(row['tile_grid'], condition['token_permutation']), main_source=condition['main'],
        source_a_original_choice=row['option_labels'][row['direction_a_index']],
        source_b_original_choice=row['option_labels'][row['direction_b_index']], original_prediction=original,
        content_prediction=row['choices'][row['option_labels'].index(original)] if original else None,
        native_source=native, native_answer_label=gold, native_correct=prediction == gold if gold is not None else None)
