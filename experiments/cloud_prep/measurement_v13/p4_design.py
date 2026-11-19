"""Repairable P4 entry decomposition; planning imports no tensor libraries."""
from __future__ import annotations
from copy import deepcopy
from itertools import combinations, product
import hashlib
import json
from pathlib import Path

from cloud_prep.measurement_v8.mechanism_design import main_source
from cloud_prep.measurement_v8.model_registry import get_model
from cloud_prep.measurement_v12 import p2_data

SCHEMA = 'entry-decomposition-common-baseline-v13-v1'
MODELS = ('ibm4b', 'qwen8b', 'qwen32b')
DEFAULT = dict(version='p4-exploratory-v1', attributes=['color', 'stripe_orientation'],
    layouts=['M0','M1'], strengths=[8,100], readouts=['direct','identity','cyclic'],
    modes=['raw','common_low_rss','common_low_l1'], target_multiplier=1.,
    match_tolerance=.03, smoke_templates=2)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',',':')).encode()).hexdigest()


def settings(config=None):
    supplied = json.loads(Path(config).read_text()) if isinstance(config,(str,Path)) else (config or {})
    unknown=set(supplied)-set(DEFAULT)-{'groups'}
    if unknown: raise ValueError('Unknown P4 config keys: '+str(sorted(unknown)))
    d={**deepcopy(DEFAULT), **deepcopy(supplied)}
    for key,allowed in [('attributes',{'color','stripe_orientation'}),('layouts',{'M0','M1'}),
            ('strengths',{8,100}),('readouts',{'direct','identity','cyclic'}),
            ('modes',{'raw','common_low_rss','common_low_l1'})]:
        if not d[key] or len(set(d[key]))!=len(d[key]) or set(d[key])-allowed:
            raise ValueError('Invalid '+key)
    if 'direct' not in d['readouts']: raise ValueError('P4 retains direct readout')
    if not isinstance(d['version'],str) or not d['version']: raise ValueError('Execution version required')
    if not 0 < d['target_multiplier'] < float('inf'): raise ValueError('Positive finite target multiplier required')
    if not 0 <= d['match_tolerance'] <= .25: raise ValueError('Invalid numerical matching tolerance')
    if type(d['smoke_templates']) is not int or d['smoke_templates']<1: raise ValueError('Invalid smoke template count')
    return d


def native_entries(model):
    return ['layer0']+[f'layer{i}' for i in (3,6,9,12,15,18,21)] if model=='ibm4b' else ['main','later0','later1','later2']


def groups(model, config=None):
    d=settings(config); entries=native_entries(model); main,later=entries[0],entries[1:]
    out=[dict(name='main',entries=[main])]
    out += [dict(name=x,entries=[x]) for x in later]
    combos=[['layer3','layer6','layer9'],['layer15','layer18','layer21'],['layer3','layer12','layer21']] if model=='ibm4b' else list(map(list,combinations(later,2)))
    out += [dict(name='+'.join(c),entries=c) for c in combos]
    out += [dict(name='all',entries=later)]
    if 'groups' in d: out=deepcopy(d['groups'].get(model,out))
    if len({g['name'] for g in out})!=len(out): raise ValueError('Duplicate group name')
    for g in out:
        if not g['entries'] or len(set(g['entries']))!=len(g['entries']) or set(g['entries'])-set(entries):
            raise ValueError('Group must contain unique native entries')
    required={frozenset([e]) for e in entries}
    if not required.issubset({frozenset(g['entries']) for g in out}):
        raise ValueError('Keep main and every native singleton for common target calibration')
    return out


def contexts(config=None):
    d=settings(config)
    return [dict(name=f'{layout}_s{strength:03d}',layout=layout,strength=strength/100,
        source=main_source(layout,strength)) for layout,strength in product(d['layouts'],d['strengths'])]


def conditions(model, config=None):
    d=settings(config); out=[]
    for ctx in contexts(d):
        out.append(dict(name=ctx['name']+'__B',context=ctx,role='baseline',group='baseline',entries=[],mode='baseline'))
        for g,mode in product(groups(model,d),d['modes']):
            out.append(dict(name=ctx['name']+'__'+g['name']+'__'+mode,
                context=ctx,role='intervention',group=g['name'],entries=g['entries'],mode=mode))
    for name in ('d_a','d_b'):
        out.append(dict(name='native_'+name,role='native',source=name,context=None,group='native',entries=[],mode='native'))
    return out


def select_rows(rows, config=None, panel='main', template_ids=None):
    d=settings(config)
    selected=p2_data.select_rows(rows,'main',True,template_ids)
    if panel=='smoke':
        tids=[]
        # Round-robin shape families so two default templates cover both.
        buckets={shape:sorted({r['template_id'] for r in selected if r['shape_pair']==shape}) for shape in (0,1)}
        for i in range(max(map(len,buckets.values()))):
            for values in buckets.values():
                if i<len(values): tids.append(values[i])
        chosen=set(tids[:d['smoke_templates']]); selected=[r for r in selected if r['template_id'] in chosen]
    elif panel!='main': raise ValueError('Unknown panel')
    selected=[r for r in selected if r['attribute'] in d['attributes']]
    # Material validator requires both attributes; pairing invariant here is local.
    pairs={}
    for r in selected: pairs.setdefault(r['intervention_id'],set()).add(r['query_role'])
    if not selected or any(x!={'changed','unchanged'} for x in pairs.values()):
        raise ValueError('Shard must retain both queries of each intervention')
    return sorted(selected,key=lambda r:(r['template_id'],r['attribute'],r['changed_index'],r['queried_index']))


def execution_plan(rows, model, config=None, panel='main', generate_tokens=32):
    if model not in MODELS: raise ValueError('P4 currently supports original three models')
    d=settings(config); cells=conditions(model,d)
    readouts=[dict(response_format='direct' if name=='direct' else 'label',mapping='none' if name=='direct' else name) for name in d['readouts']]
    fields=('id','template_id','intervention_id','attribute','changed_object','changed_index','queried_object','queried_index','query_role','direction_a_index','direction_b_index')
    queries=[{k:r[k] for k in fields} for r in rows]
    logical=len(rows)*len(cells)*len(readouts)
    # Only singleton RSS/L1 are physically identical, unless a mode is omitted.
    reused=sum(len(c['entries'])==1 and c['mode']=='common_low_l1' and 'common_low_rss' in d['modes'] for c in cells)
    identity=dict(schema=SCHEMA,config=d,model=get_model(model),queries=queries,
        image_sha256={r['id']:{k:v['sha256'] for k,v in r['source_image_metadata'].items()} for r in rows},
        baseline='All native entries share donor-a ROI over the requested native background; selected entries alone receive k*(b-a)',
        normalization='One full B canonical changed/direct forward; t=min single-entry delta-RMS/B-post-entry-RMS over main and all later entries')
    return dict(schema=SCHEMA,execution_version=d['version'],scientific_identity_sha256=digest(identity),
        model=get_model(model),config=d,panel=panel,queries=queries,conditions=cells,readouts=readouts,
        template_ids=sorted({r['template_id'] for r in rows}),independence_unit='template_id',
        template_n=len({r['template_id'] for r in rows}),generate_tokens=generate_tokens,
        expected_score_rows=logical,expected_generation_rows=logical if generate_tokens else 0,
        expected_unique_score_evaluations=len(rows)*len(readouts)*(len(cells)-reused),
        expected_unique_generation_evaluations=len(rows)*len(readouts)*(len(cells)-reused) if generate_tokens else 0,
        expected_anchor_forwards=len({r['intervention_id'] for r in rows})*len(contexts(d)),
        selection='v12 result-independent 32-template dose subset; full templates are the sharding unit',
        baseline_definition=identity['baseline'],normalization_definition=identity['normalization'],
        status='prepared_not_model_validated',
        repair_policy='Operational changes retain provenance; changed observations use a new execution version and rerun affected complete templates only')
