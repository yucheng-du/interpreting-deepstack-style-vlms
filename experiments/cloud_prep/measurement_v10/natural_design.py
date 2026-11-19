"""Whole-scene fixed-ROI donor effects at full and weak main contrast."""
from copy import deepcopy
from itertools import product
import json
from pathlib import Path
from cloud_prep.measurement_v6.content_intervention import readout_specs
from cloud_prep.measurement_v10.natural_data import validate_query_groups

DESIGN_SCHEMA='natural-scene-v10'
DEFAULT_DESIGN={'strengths':[100,8],'response_formats':['label','direct'],
                'generation_groups':[0,1,2,3],'generate_native':True,'generate_strengths':[100,8],'self_patch':True}
QUERY_FIELDS=('id','scene_id','unit_id','intervention_id','attribute','generation_group','query_role',
 'changed_object','changed_object_id','changed_index','queried_object','queried_object_id','queried_index',
 'direction_a_index','direction_b_index','has_unchanged_query','visual_source_signature',
 'donor_roi_bbox_xyxy','changed_bbox_by_layout','queried_bbox_by_layout','source_answer_indices')


def read_design(path=None):
    d=deepcopy(DEFAULT_DESIGN)
    if path is not None:
        update=json.loads(Path(path).read_text()) if isinstance(path,(str,Path)) else path
        if not isinstance(update,dict) or set(update)-set(d):raise ValueError('Unknown natural design fields')
        d.update(update)
    for field in ('strengths','generate_strengths','generation_groups'):
        v=d[field]
        if not isinstance(v,list) or len(v)!=len(set(v)) or any(type(x) is not int for x in v):raise ValueError('Unique integer design values required')
    if not d['strengths'] or set(d['strengths'])-{8,100}:raise ValueError('Prepared natural strengths are 8 and 100')
    if set(d['generate_strengths'])-set(d['strengths']) or set(d['generation_groups'])-{0,1,2,3}:raise ValueError('Generation subset mismatch')
    if not d['response_formats'] or set(d['response_formats'])-{'label','direct'} or len(set(d['response_formats']))!=len(d['response_formats']):raise ValueError('Direct/label formats required')
    if any(type(d[k]) is not bool for k in ('generate_native','self_patch')):raise ValueError('Boolean flags required')
    return d


def conditions(design=None,entry_groups=('all',),panel='main'):
    d=read_design(design)
    if panel not in ('main','smoke') or tuple(entry_groups)!=('all',):raise ValueError('Natural panel uses all later entries only')
    result=[]
    for strength,layout in product(d['strengths'],('M0','M1')):
        main=f'{layout.lower()}_s{strength:03d}'
        base={'main':main,'main_layout':layout,'main_strength':strength/100,'main_kind':'whole_scene',
              'donor_pose':'original_pixels','token_permutation':'identity','source_destination':'original',
              'varying_entry':'later','entry_group':'all','region':'roi'}
        for a in ('a','b'):
            result.append({**base,'name':f'{main}__roi__{a}','role':'natural_roi_pair','donor_source':f'd_{a}',
                           'donor_assignment':a,'generation':strength in d['generate_strengths'],'technical_audit':False})
        if d['self_patch']:
            result.append({**base,'name':f'self_{main}__roi','role':'self_patch','donor_source':main,
                           'donor_assignment':None,'generation':False,'technical_audit':True})
        result.append({**base,'name':f'native_{main}','role':'native_main','donor_source':None,
                       'donor_assignment':None,'entry_group':'none','region':'none','generation':d['generate_native'],'technical_audit':False})
    for a in ('a','b'):
        result.append({'name':f'native_d_{a}','role':'native_donor','main':f'd_{a}','main_layout':'M0',
                       'main_strength':1.,'main_kind':'edited_whole_scene','donor_pose':'original_pixels',
                       'token_permutation':'identity','source_destination':None,'varying_entry':None,
                       'entry_group':'none','region':'none','donor_source':None,'donor_assignment':a,
                       'generation':d['generate_native'],'technical_audit':False})
    return result


def generation_selected(row,panel='main',design=None):
    return panel=='smoke' or row['generation_group'] in read_design(design)['generation_groups']


def execution_plan(rows,panel='main',mappings=('identity','cyclic'),generate_tokens=32,design=None,entry_groups=('all',)):
    d=read_design(design);validate_query_groups(rows)
    if not mappings or len(set(mappings))!=len(mappings) or set(mappings)-{'identity','cyclic'}:raise ValueError('Unique supported mappings required')
    if type(generate_tokens) is not int or generate_tokens<0:raise ValueError('Nonnegative generation budget required')
    cells=conditions(d,entry_groups,panel);readouts=readout_specs(mappings,d)
    queries=[{**{k:r[k] for k in QUERY_FIELDS},'generation_selected':bool(generate_tokens and generation_selected(r,panel,d))} for r in rows]
    ng=sum(q['generation_selected'] for q in queries)
    return {'schema':DESIGN_SCHEMA,'panel':panel,'design':d,'entry_groups':list(entry_groups),'conditions':cells,
            'readouts':readouts,'mappings':list(mappings),'queries':queries,'items':len(rows),'generate_tokens':generate_tokens,
            'expected_score_rows':len(rows)*len(cells)*len(readouts),
            'expected_generation_rows':ng*sum(c['generation'] for c in cells)*len(readouts),
            'generation_items':ng,'independent_source_photos':len({r['scene_id'] for r in rows}),
            'off_target_paired_source_photos':len({r['scene_id'] for r in rows if r['has_unchanged_query']}),
            'direction_contract':'Fixed red=a and blue=b for changed and unchanged queries; E=S(b)-S(a)',
            'roi_coordinate_contract':'Fixed donor original-image bbox; M1 mirrors main pixels only; no token permutation',
            'scope':'Whole natural photograph under resized letterbox, contrast, mirror and bbox-color-region editing; not unedited natural-task performance',
            'correctness_scope':'Native source-specific labels only; hybrids have no imposed answer',
            'independence_unit':'source-photo scene_id; paired off-target contrasts restricted to declared subset',
            'effect_gate':'No model performance or intervention sign filters'}


def record_fields(row,variant,condition,prediction):
    original=variant['label_to_original'][prediction] if prediction is not None else None
    native=condition['main'] if condition['role'].startswith('native_') else None
    gold_index=row['source_answer_indices'][native] if native else None
    gold=variant['original_to_label'][row['option_labels'][gold_index]] if gold_index is not None else None
    return {'schema':DESIGN_SCHEMA,'row_schema':row['schema'],'family':row['family'],'task':row['task'],
            **{k:row[k] for k in QUERY_FIELDS},'record_fields':row.get('record_fields',{}),
            'mapping':variant['mapping'],'response_format':variant['response_format'],
            'condition':condition['name'],'condition_spec':condition,**{k:condition[k] for k in
              ('role','main_layout','main_strength','main_kind','entry_group','region','donor_source','donor_assignment',
               'donor_pose','token_permutation','source_destination','varying_entry','technical_audit')},
            'main_source':condition['main'],'source_a_original_choice':'(A)','source_b_original_choice':'(B)',
            'original_prediction':original,'content_prediction':row['choices'][row['option_labels'].index(original)] if original else None,
            'native_source':native,'native_answer_label':gold,'native_correct':prediction==gold if gold is not None else None}
