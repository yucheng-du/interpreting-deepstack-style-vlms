"""P2 local, source-cross and native-target dose bridge on paired attributes."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

from cloud_prep.measurement_v6.content_intervention import readout_specs
from cloud_prep.measurement_v10 import local_design, full_design
from cloud_prep.measurement_v11.runtime import evaluation_plan
from cloud_prep.measurement_v12 import p2_data as data
from cloud_prep.measurement_v12 import v11_v2_dose_design as dose

SCHEMA = 'paired-attribute-entry-role-v12-v1'
ARMS = ('local', 'full', 'dose')


def settings(arm):
    if arm == 'local':
        return local_design.read_design(dict(strengths=[8,100], generate_strengths=[8,100],
            generation_groups=list(range(8))))
    if arm == 'full':
        return full_design.read_design(dict(strengths=[100], generate_strengths=[100],
            generation_groups=list(range(8))))
    if arm == 'dose':
        return dose.read_design(dict(scene_limit=32, smoke_scene_limit=2,
            layout_pairs=[['M0','M0']], strength_pairs=[[100,100]],
            k_values=[0,1], generate_k_values=[0,1], target_policy='opposite_native',
            dose_metrics=['rss','l1'], normalized_modes=['baseline_state'],
            target_ratios=[], generate_target_ratios=[]))
    raise ValueError('Unknown P2 arm')


def conditions(arm):
    module = {'local':local_design, 'full':full_design, 'dose':dose}[arm]
    return module.conditions(settings(arm))


def generation_selected(row, panel='main', design=None):
    # The full new panel, including all eight template replicates, is generated.
    return True


def record_fields(row, variant, condition, prediction, arm):
    module = {'local':local_design, 'full':full_design, 'dose':dose}[arm]
    record = module.record_fields(row,variant,condition,prediction)
    return {**record, 'schema':SCHEMA, 'analysis_arm':arm,
            'template_id':row['template_id'], 'geometry_signature':row['geometry_signature'],
            'dose_bridge_selected':row['dose_bridge_selected'],
            'independence_unit':'template_id'}


def execution_plan(rows, arm, panel='main', mappings=('identity','cyclic'), generate_tokens=32):
    data.validate_query_groups(rows)
    if arm not in ARMS: raise ValueError('Unknown P2 arm')
    if len(mappings)!=2 or set(mappings)!={'identity','cyclic'}:
        raise ValueError('P2 retains both frozen label mappings and direct readout')
    if type(generate_tokens) is not int or generate_tokens < 0:
        raise ValueError('Nonnegative generation token budget required')
    d=settings(arm)
    if arm=='dose':
        if not all(r['dose_bridge_selected'] for r in rows):
            raise ValueError('Dose rows must use the frozen template subset')
        plan=dose.execution_plan(rows,panel,mappings,generate_tokens,d)
    else:
        cells=conditions(arm)
        readouts=readout_specs(mappings,d)
        fields=tuple(dict.fromkeys((*local_design.QUERY_FIELDS,*full_design.QUERY_FIELDS)))
        queries=[{**{k:r[k] for k in fields},'generation_selected':bool(generate_tokens),
                  'template_id':r['template_id'],'geometry_signature':r['geometry_signature']} for r in rows]
        plan=dict(schema=SCHEMA,panel=panel,design=d,entry_groups=['all'],conditions=cells,
            queries=queries,readouts=readouts,mappings=list(mappings),items=len(rows),
            generate_tokens=generate_tokens,generation_items=len(rows) if generate_tokens else 0,
            expected_score_rows=len(rows)*len(cells)*len(readouts),
            expected_generation_rows=len(rows)*sum(c['generation'] for c in cells)*len(readouts) if generate_tokens else 0)
    by_id={r['id']:r for r in rows}
    for query in plan['queries']:
        query.update(template_id=by_id[query['id']]['template_id'],
                     geometry_signature=by_id[query['id']]['geometry_signature'])
    plan.update(schema=SCHEMA,analysis_arm=arm,scene_templates=len({r['template_id'] for r in rows}),
        selected_scene_ids=sorted({r['template_id'] for r in rows}),
        attribute_scenes=len({(r['template_id'],r['attribute']) for r in rows}),
        unique_visual_interventions=len({r['intervention_id'] for r in rows}),
        independence_unit='template_id; both attributes, edits, queries, readouts and conditions are paired repeats',
        selection='New seed 2026102202. Dose: fixed 32 of 64, SHA256 rank then round-robin across shape/layout/assignment; no model outcomes',
        direction_contract='b-a always follows edited-object attribute direction, including off-target query; never entry-dependent sign',
        score_scope='Whole candidate-sequence logprob; compare within attribute/readout; do not rank attribute binding using raw cross-attribute amplitudes',
        effect_gate='Retain every planned template, invalid generation and both signs. Native accuracy is descriptive, not a selection filter',
        geometry_contract='Color and stripe share scene geometry and source canvas. Full and dose reuse exactly the local arm pixels; no relation queries or central-reference redesign',
        generation_contract='All selected templates and all three readouts; greedy EOS and strict phrase/label parser inherited unchanged',
        estimands=['E_target','E_control','R=E_target-E_control',
                    'R_main-R_later with paired CI; non-significance is not equivalence',
                    'local layout x destination x query interaction',
                    'full source-role swap and fixed-physical-configuration four-cell J separately',
                    'dose main-native and later-native comparisons for each RSS/L1 scale'],
        model_results=False)
    if arm=='dose':
        plan.update(evaluation_plan(rows,plan))
        plan['native_target_contract']='Raw k1 at each role is its own native target; opposite-role RSS/L1 matching supplies both main-native and later-native comparisons'
        plan['normalization_contract']='Canonical identity/changed/direct a-side post-entry baseline; shared k across both queries and all readouts; actual forward dose observations retained'
        plan['expected_native_self_audit_source_count']=2
    else:
        plan.update(expected_unique_score_evaluations=plan['expected_score_rows'],
                    expected_unique_generation_evaluations=plan['expected_generation_rows'],
                    expected_score_reuse_rows=0,expected_generation_reuse_rows=0)
        plan['evaluation_count_boundary']='v10 executor does not deduplicate native/self or equal compositions. Counts include declared controls; additional audited native/self score calls separate'
    plan['design_sha256']=hashlib.sha256(json.dumps(d,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return plan


class ExecutionDesign:
    """Small explicit facade consumed by the unchanged inference loop."""
    generation_selected=staticmethod(generation_selected)
    canonical_anchors=staticmethod(dose.canonical_anchors)
    required_sources=staticmethod(dose.required_sources)
    calibration_key=staticmethod(dose.calibration_key)
    opposite_condition=staticmethod(dose.opposite_condition)
    context_signature=staticmethod(dose.context_signature)
    anchor_query_sha256=staticmethod(dose.anchor_query_sha256)
    visual_signature=staticmethod(dose.visual_signature)

