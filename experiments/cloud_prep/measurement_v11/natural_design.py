"""Matched main/later ROI replacement on the fixed full-photo v10 panel."""
from copy import deepcopy
from cloud_prep.measurement_v10 import natural_design as v10

DESIGN_SCHEMA = 'natural-reciprocal-v11'
DEFAULT_DESIGN = deepcopy(v10.DEFAULT_DESIGN)
QUERY_FIELDS = v10.QUERY_FIELDS
read_design = v10.read_design
generation_selected = v10.generation_selected


def conditions(design=None, entry_groups=('all',), panel='main'):
    result = []
    for base in v10.conditions(design, entry_groups, panel):
        if base['role'] in ('natural_roi_pair', 'self_patch'):
            for entry in ('main', 'later'):
                result.append({**base, 'name': f'{entry}__{base["name"]}',
                    'changed_entry': entry, 'varying_entry': entry,
                    'intervention_type': 'local_role',
                    'context_source': base['main'],
                    'v10_bridge_condition': base['name'] if entry == 'later' else None})
        else:
            result.append({**base, 'changed_entry': None, 'context_source': base['main'],
                'intervention_type': 'native', 'v10_bridge_condition': base['name']})
    return result


def execution_plan(rows, panel='main', mappings=('identity', 'cyclic'),
                   generate_tokens=32, design=None, entry_groups=('all',)):
    plan = v10.execution_plan(rows, panel, mappings, generate_tokens, design, entry_groups)
    cells = conditions(plan['design'], entry_groups, panel)
    plan.update(schema=DESIGN_SCHEMA, conditions=cells,
        expected_score_rows=len(rows)*len(cells)*len(plan['readouts']),
        expected_generation_rows=plan['generation_items']*sum(c['generation'] for c in cells)*len(plan['readouts']),
        changed_entries=['main', 'later'],
        role_contract='Same nominal ROI, unchanged source/context and a/b direction; main versus all later entries',
        pairing_contract='All formal main/later pairs are newly executed in one backend/run specification; v10 is a bridge reference only',
        bridge_contract='v11 later uses local-role dispatch; at k=1 its tensors must equal the frozen v10 later-only composition')
    return plan


def record_fields(row, variant, condition, prediction):
    return {**v10.record_fields(row, variant, condition, prediction),
        'schema': DESIGN_SCHEMA,
        **{f: condition[f] for f in ('changed_entry', 'context_source',
            'intervention_type', 'v10_bridge_condition')}}
