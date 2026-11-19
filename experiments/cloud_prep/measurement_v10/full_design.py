"""Full main x later source matrix; reciprocal a/b effects reuse unique forwards."""
from __future__ import annotations
from copy import deepcopy
from itertools import product
import json
from pathlib import Path
from cloud_prep.measurement_v6.content_intervention import readout_specs
from cloud_prep.measurement_v10.full_data import source_name, validate_query_groups

DESIGN_SCHEMA='full-source-control-v10'
DEFAULT_DESIGN={'strengths':[8,100],'generate_strengths':[8,100],'response_formats':['label','direct'],
                'generation_groups':[0,1,2,3],'generate_native':True,'self_patch':True}
CONDITION_FIELDS=('role','changed_entry','main_layout','later_layout','main_strength','later_strength',
                  'main_assignment','later_assignment','entry_group','region','donor_source','donor_assignment',
                  'token_permutation','technical_audit')
QUERY_FIELDS=('id','scene_id','attribute','generation_group','intervention_id','changed_object','changed_index',
              'queried_object','queried_index','query_type','query_target','query_role','query_slot','target_object','target_index',
              'direction_a_index','direction_b_index','answer_index','source_answer_indices','referred_object_by_source',
              'visual_source_signature')


def read_design(path=None):
    design=deepcopy(DEFAULT_DESIGN)
    supplied=json.loads(Path(path).read_text()) if isinstance(path,(str,Path)) else path
    if supplied is not None:
        if not isinstance(supplied,dict) or set(supplied)-set(design):raise ValueError('Unknown full-source design fields')
        design.update(supplied)
    for field in ('strengths','generate_strengths','generation_groups'):
        values=design[field]
        if not isinstance(values,list) or len(values)!=len(set(values)) or any(type(x)is not int or x<0 for x in values):
            raise ValueError('Unique nonnegative integer design values required')
    if not design['strengths'] or set(design['strengths'])-{0,8,100} or set(design['generate_strengths'])-set(design['strengths']):
        raise ValueError('Supported strengths 0/8/100; generation strengths must be selected')
    if not design['response_formats'] or len(design['response_formats'])!=len(set(design['response_formats'])) or set(design['response_formats'])-{'direct','label'}:
        raise ValueError('Unique direct/label formats required')
    if any(type(design[f])is not bool for f in ('self_patch','generate_native')):raise ValueError('Boolean flags required')
    return design


def conditions(design=None,entry_groups=('all',),panel='main'):
    design=read_design(design)
    if panel not in ('main','smoke') or tuple(entry_groups)!=('all',):raise ValueError('main/smoke and all later entries required')
    cells=[]
    for mm,lm,ms,ls,ma,la in product(('M0','M1'),('M0','M1'),design['strengths'],design['strengths'],('a','b'),('a','b')):
        main=source_name(mm,ms,ma); later=source_name(lm,ls,la)
        cells.append({'name':f'{main}__later_{later}','role':'source_cross','changed_entry':'joint_matrix',
            'main':main,'donor_source':later,'main_layout':mm,'later_layout':lm,'main_strength':ms/100,'later_strength':ls/100,
            'main_assignment':ma,'later_assignment':la,'donor_assignment':la,'main_kind':'objects','donor_pose':'same',
            'entry_group':'all','region':'all','token_permutation':'identity','technical_audit':False,
            'generation':ms in design['generate_strengths'] and ls in design['generate_strengths']})
    for layout,strength,assignment in product(('M0','M1'),design['strengths'],('a','b')):
        source=source_name(layout,strength,assignment)
        base={'main':source,'main_layout':layout,'later_layout':layout,'main_strength':strength/100,'later_strength':strength/100,
            'main_assignment':assignment,'later_assignment':assignment,'donor_assignment':assignment,'main_kind':'objects',
            'donor_pose':'same','token_permutation':'identity','changed_entry':None}
        cells.append({**base,'name':f'native_{source}','role':'native_source','donor_source':None,'entry_group':'none',
                      'region':'none','technical_audit':False,'generation':design['generate_native']})
        if design['self_patch']:
            cells.append({**base,'name':f'self_{source}','role':'self_patch','donor_source':source,'entry_group':'all',
                          'region':'all','technical_audit':True,'generation':False})
    return cells


def generation_selected(row,panel='main',design=None):
    return panel=='smoke' or row['generation_group'] in read_design(design)['generation_groups']


def execution_plan(rows,panel='main',mappings=('identity','cyclic'),generate_tokens=32,design=None,entry_groups=('all',)):
    design=read_design(design);validate_query_groups(rows)
    if not mappings or len(mappings)!=len(set(mappings)) or set(mappings)-{'identity','cyclic'}:raise ValueError('Unique identity/cyclic mappings required')
    if type(generate_tokens)is not int or generate_tokens<0:raise ValueError('Nonnegative generation budget required')
    cells=conditions(design,entry_groups,panel);readouts=readout_specs(mappings,design)
    queries=[{**{k:r[k] for k in QUERY_FIELDS},'generation_selected':bool(generate_tokens and generation_selected(r,panel,design))} for r in rows]
    gen_n=sum(q['generation_selected'] for q in queries)
    return {'schema':DESIGN_SCHEMA,'panel':panel,'design':design,'entry_groups':list(entry_groups),'conditions':cells,
        'queries':queries,'query_ids':[r['id'] for r in rows],'readouts':readouts,'mappings':list(mappings),
        'items':len(rows),'scene_templates':len({r['scene_id'] for r in rows}),'generate_tokens':generate_tokens,'generation_items':gen_n,
        'expected_score_rows':len(rows)*len(cells)*len(readouts),
        'expected_generation_rows':gen_n*sum(c['generation'] for c in cells)*len(readouts),
        'independence_unit':'scene_id; v8-derived scene parameters, changed object, query, readout and conditions are paired repeats',
        'direction_contract':'E_main and E_later both equal score(b)-score(a), where score=logP(changed-b color)-logP(changed-a color); no role sign flip',
        'source_contract':'Independent main/later layout x strength x assignment matrix; all main tokens and all later entries selected separately',
        'execution_reuse':'Each physical source combination is scored once; both reciprocal effects are constructed from its complete a/b matrix',
        'query_contract':'Relation left/right of gray cross refers to M0 queried identity only for query pairing; source native truth follows actual source layout',
        'correctness_scope':'Source-specific native truth only; no hybrid gold and alpha0 native truth unavailable',
        'effect_gate':'No native-success, sign, significance or minimum-effect filter'}


def record_fields(row,variant,condition,prediction):
    original=variant['label_to_original'][prediction] if prediction is not None else None
    native=condition['main'] if condition['role'].startswith('native_') else None
    gold_index=row['source_answer_indices'][native] if native else None
    gold=variant['original_to_label'][row['option_labels'][gold_index]] if gold_index is not None else None
    return {'schema':DESIGN_SCHEMA,'row_schema':row['schema'],'family':row['family'],'task':row['task'],
        **{k:row[k] for k in QUERY_FIELDS},'record_fields':row.get('record_fields',{}),
        'mapping':variant['mapping'],'response_format':variant['response_format'],'condition':condition['name'],'condition_spec':condition,
        **{k:condition[k] for k in CONDITION_FIELDS},'main_source':condition['main'],'main_kind':'objects','donor_pose':'same',
        'source_a_original_choice':row['option_labels'][row['direction_a_index']],
        'source_b_original_choice':row['option_labels'][row['direction_b_index']],
        'original_prediction':original,'content_prediction':row['choices'][row['option_labels'].index(original)] if original else None,
        'native_source':native,'native_answer_label':gold,'native_correct':prediction==gold if gold is not None else None}
