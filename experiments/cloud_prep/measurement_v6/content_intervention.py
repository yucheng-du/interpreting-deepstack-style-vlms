"""Routing 2x2 and response-encoding design; imports/planning are torch-free."""
from __future__ import annotations
from copy import deepcopy
from itertools import product
import json

from cloud_prep.measurement_v4.content_intervention import cloud_gpu_lock
from cloud_prep.measurement_v4.content_intervention import compose_visual_output as compose_sources

SCHEMA='content-routing-v6'
STRENGTHS=(0,2,8,25,100)
POSES=('same','cross')
PERMUTATIONS=('identity','swap_halves')
DEFAULT_DESIGN={'strengths':list(STRENGTHS),'include_blank':True,
                'response_formats':['label','direct'],'generation_replicates':[0,1],
                'generate_native':True,'generate_strengths':list(STRENGTHS),'generate_blank':True}


def read_design(path=None):
    design=deepcopy(DEFAULT_DESIGN)
    if path is not None:
        supplied=json.loads(path.read_text())
        if set(supplied)-set(design):raise ValueError('Unknown design fields: '+str(sorted(set(supplied)-set(design))))
        design.update(supplied)
    for key in ('strengths','generate_strengths'):
        if not isinstance(design[key],list) or any(type(x) is not int or not 0<=x<=100 for x in design[key]) or len(set(design[key]))!=len(design[key]):
            raise ValueError('Unique strengths in 0..100 required')
    if not design['strengths']:raise ValueError('Need at least one main strength')
    if not design['response_formats'] or len(set(design['response_formats']))!=len(design['response_formats']) or set(design['response_formats'])-{'label','direct'}:
        raise ValueError('Use label/direct response formats')
    if any(type(x) is not int or x<0 for x in design['generation_replicates']):raise ValueError('Nonnegative generation replicates required')
    for key in ('include_blank','generate_native','generate_blank'):
        if type(design[key]) is not bool:raise ValueError('Boolean design flags required')
    return design


def main_source(strength):return f'm_s{strength:03d}'
def donor_source(pose,value):return f'd_{pose}_{value}'


def conditions(design=None):
    design=design or read_design();output=[]
    mains=[(main_source(s),'objects',s/100) for s in design['strengths']]
    if design['include_blank']:mains.append(('blank','uniform_blank',0.))
    for source,kind,strength in mains:
        generate=design['generate_blank'] if kind=='uniform_blank' else round(strength*100) in design['generate_strengths']
        for pose,perm,value in product(POSES,PERMUTATIONS,('a','b')):
            output.append({'name':f'{source}__{pose}__{perm}__{value*3}','role':'routing_2x2',
                'main':source,'main_kind':kind,'main_strength':strength,'donor_pose':pose,'token_permutation':perm,
                'ds_sources':[donor_source(pose,value)]*3,'ds_code':value*3,'generation':generate})
        output.append({'name':f'native_{source}','role':'native_main','main':source,'main_kind':kind,
            'main_strength':strength,'donor_pose':None,'token_permutation':'identity','ds_sources':[source]*3,
            'ds_code':None,'generation':design['generate_native']})
    for pose,value in product(POSES,('a','b')):
        source=donor_source(pose,value)
        output.append({'name':f'real_{source}','role':'real_donor','main':source,'main_kind':'objects',
            'main_strength':1.,'donor_pose':pose,'token_permutation':'identity','ds_sources':[source]*3,
            'ds_code':None,'generation':design['generate_native']})
    return output


def select_panel(rows,panel):
    if panel=='main':return rows
    if panel=='smoke':
        return [next(r for r in rows if r['attribute']==attribute and r['shape_pair']==pair and r['layout']==0
                     and r['assignment']==0 and r['replicate']==0 and r['target_index']==target)
                for attribute,pair,target in (('color',0,0),('stripe_orientation',1,1))]
    raise ValueError('Unknown panel')


def readout_specs(mappings,design):
    out=[]
    if 'label' in design['response_formats']:out.extend({'response_format':'label','mapping':m} for m in mappings)
    if 'direct' in design['response_formats']:out.append({'response_format':'direct','mapping':'none'})
    return out


def generation_selected(row,panel,design=None):
    return panel=='smoke' or row['replicate'] in (design or DEFAULT_DESIGN)['generation_replicates']


def execution_plan(rows,panel='main',mappings=('identity','cyclic'),generate_tokens=32,design=None):
    design=design or read_design()
    if not rows or len(set(mappings))!=len(mappings) or set(mappings)-{'identity','cyclic'}:raise ValueError('Invalid rows/mappings')
    cells=conditions(design);readouts=readout_specs(mappings,design)
    gen_items=sum(generation_selected(r,panel,design) for r in rows) if generate_tokens else 0
    return {'schema':SCHEMA,'design':design,'panel':panel,'items':len(rows),
        'scene_templates':len({r['scene_id'] for r in rows}),'conditions':cells,'mappings':list(mappings),'readouts':readouts,
        'expected_score_rows':len(rows)*len(readouts)*len(cells),'generation_items':gen_items,
        'expected_generation_rows':gen_items*len(readouts)*sum(c['generation'] for c in cells),'generate_tokens':generate_tokens,
        'independence_unit':'scene_id; all queries, mappings, formats, strengths, poses and permutations are repeated conditions',
        'score_scope':'Complete candidate-text sequence logprob; no length normalization. Compare effects within response format, never pool label/direct score magnitudes.',
        'correctness_scope':'Native nonzero-strength and real donor only; gray, blank and hybrid cells have no attribute gold'}


def spatial_token_indices(grids,permutation='swap_halves',merge_size=2):
    """Output-token -> input-token index for the declared merged row-major grid.

    CUDA smoke checks actual processor grouping and visual rotary ordering.
    This function does not infer object locations or consult answers.
    """
    if len(grids)!=1 or len(grids[0])!=3:raise ValueError('One complete image grid required')
    t,h,w=grids[0]
    if t!=1 or any(type(x) is not int or x<1 for x in (h,w)) or h%merge_size or w%(2*merge_size):
        raise ValueError('Need a static image with two equal merged-grid halves')
    hh,ww=h//merge_size,w//merge_size
    if permutation=='identity':return list(range(hh*ww))
    if permutation!='swap_halves':raise ValueError('Unknown token permutation')
    return [r*ww+(c+ww//2)%ww for r in range(hh) for c in range(ww)]


def compose_visual_output(sources,condition,grids):
    import torch
    main,ds=compose_sources(sources,condition,grids)
    indices=spatial_token_indices(grids,condition['token_permutation'])
    if condition['token_permutation']!='identity':
        ids=torch.tensor(indices,device=main.device)
        ds=[x.index_select(0,ids) for x in ds]
    return main,ds


def audit_spatial_bridge(model,source_inputs,sources,grids):
    """Small CUDA smoke: pixel grouping, rotary order and exact inverse only.

    ViT contextualized features need not equal features of the pixel-swapped
    image. No such equivariance requirement is imposed.
    """
    import torch
    indices=spatial_token_indices(grids)
    patch_ids=torch.tensor([j*4+k for j in indices for k in range(4)],device=next(iter(source_inputs.values()))['pixel_values'].device)
    checks=[]
    for value in ('a','b'):
        same,cross=donor_source('same',value),donor_source('cross',value)
        equal=torch.equal(source_inputs[same]['pixel_values'].index_select(0,patch_ids),source_inputs[cross]['pixel_values'])
        if not equal:raise RuntimeError('Actual processor patch order does not implement the declared equal-half permutation')
        for path,feature in enumerate(sources[same][1]):
            ids=torch.tensor(indices,device=feature.device)
            if not torch.equal(feature.index_select(0,ids).index_select(0,ids),feature):
                raise RuntimeError('Permutation inverse failed')
        checks.append({'source':value,'processor_pixel_permutation_equal':equal,'ds_inverse_exact':True})
    visual=model.model.visual
    grid=next(iter(source_inputs.values()))['image_grid_thw']
    actual=visual.rot_pos_emb(grid)
    _,h,w=grids[0]
    # Match the patch16/merge2 grouping; merged tokens remain row-major.
    coords=[(r*2+dr,c*2+dc) for r in range(h//2) for c in range(w//2) for dr in range(2) for dc in range(2)]
    coord_ids=torch.tensor(coords,device=grid.device)
    table=visual.rotary_pos_emb(max(h,w))
    expected=table[coord_ids].flatten(1)
    if not torch.equal(actual,expected):raise RuntimeError('Actual visual rotary positions do not match declared merged-token order')
    return {'pixel_and_inverse_checks':checks,'rotary_order_exact':True,'merged_indices':indices,
            'feature_pixel_equivariance_required':False,
            'interpretation':'Token permutation moves already contextualized position-bearing features; not pure semantic routing.'}
