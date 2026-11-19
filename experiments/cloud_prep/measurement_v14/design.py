"""Editable follow-up plans and row identities, without tensor imports.

Hashes identify observations; they do not freeze this directory or a campaign.
Changing one condition leaves other observation keys reusable. A numerical code
repair should use a new execution_version for affected rows, not erase old data.
"""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

SCHEMA = 'native-entry-conditional-use-v14'
DEFAULTS = dict(execution_version='v14-exploration-1', families=['conflict', 'recovery'],
    source_aliases={}, main_sources={'normal':'a', 'degraded':'degraded_a'},
    later_base='a', later_donor='b', region='roi', k=1., entries='all_later',
    recovery_sources={'clean':'a','degraded':'degraded_a'},
    readouts=[{'response_format':'direct','mapping':'none'},
              {'response_format':'label','mapping':'identity'},
              {'response_format':'label','mapping':'cyclic'}],
    generate_tokens=32, max_sequence=4096, seed=20261028, conditions=[], comparisons=[],
    self_patch_tolerance=.001, audit_per_task=True, qwen_pixel_limits=None,
    # Batched execution (cloud_prep/measurement_v14/batched.py). Value-changing at BF16
    # level, so only configs with their own execution_version enable it. Defaults keep the
    # batch-one path; these keys are not part of observation keys.
    batch_conditions=False, cross_query_cache=False, max_batch=8)


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()


def settings(path=None):
    result=deepcopy(DEFAULTS)
    if path:
        supplied=json.loads(Path(path).read_text())
        if not isinstance(supplied,dict): raise ValueError('Config must be a JSON object')
        unknown=set(supplied)-set(result)
        if unknown: raise ValueError('Unknown config keys (possibly typos): '+str(sorted(unknown)))
        result.update(supplied)
    if not result['execution_version']: raise ValueError('An execution version is required')
    if result['generate_tokens']<0: raise ValueError('Negative generation budget')
    for key in ('batch_conditions','cross_query_cache'):
        if not isinstance(result[key],bool): raise ValueError(key+' must be true/false')
    if result['cross_query_cache'] and not result['batch_conditions']:
        raise ValueError('cross_query_cache is defined for batched configs only (batch_conditions true)')
    if type(result['max_batch']) is not int or result['max_batch']<1: raise ValueError('max_batch must be a positive integer')
    return result


def load_rows(path, config, *, unit_ids=None, limit_units=None, check_files=True):
    path=Path(path).resolve(); rows=[]; selected=[]; seen=set(); file_meta={}
    # Physical lines: str.splitlines also splits at U+2028/U+2029/U+0085 inside JSON strings.
    for line in path.read_text().split('\n'):
        if not line.strip(): continue
        row=json.loads(line)
        if row['id'] in seen: raise ValueError('Duplicate query id: '+row['id'])
        seen.add(row['id'])
        row['unit_id']=str(row.get('unit_id',row.get('template_id',row.get('source_cluster',row.get('scene_id',row['id'])))))
        if unit_ids is not None and row['unit_id'] not in unit_ids: continue
        if row['unit_id'] not in selected:
            if limit_units is not None and len(selected)>=limit_units: continue
            selected.append(row['unit_id'])
        row.setdefault('intervention_id',row['id'])
        row.setdefault('task',row.get('attribute','unspecified'))
        row.setdefault('query_role','task')
        row.setdefault('tile_grid',[1,1])
        row.setdefault('source_image_metadata',{})
        for alias,name in config['source_aliases'].items():
            if name in row['source_images']:
                row['source_images'][alias]=row['source_images'][name]
                if name in row['source_image_metadata']: row['source_image_metadata'][alias]=deepcopy(row['source_image_metadata'][name])
        for name,image in list(row['source_images'].items()):
            if isinstance(image,list):
                if len(image)!=1: raise ValueError('One image per native source required')
                image=image[0]
            image=Path(image); image=image if image.is_absolute() else path.parent/image
            row['source_images'][name]=str(image.resolve())
            if check_files:
                if image not in file_meta:
                    from PIL import Image
                    with Image.open(image) as im: width,height=im.size
                    file_meta[image]=dict(width=width,height=height,sha256=hashlib.sha256(image.read_bytes()).hexdigest())
                actual=file_meta[image]
                expected=row['source_image_metadata'].get(name,{})
                if any(expected.get(k,actual[k])!=actual[k] for k in actual):
                    raise ValueError('Image differs from its row metadata: '+str(image))
                row['source_image_metadata'][name]={**expected,**actual,'source_field':'image_1'}
        sizes={(x['width'],x['height']) for x in row['source_image_metadata'].values()}
        if len(sizes)!=1: raise ValueError('Within-row sources must have one canvas: '+row['id'])
        if row.get('choices'):
            if len(set(row['choices']))!=len(row['choices']) or len(row['choices'])<2: raise ValueError('Distinct choices required')
            row.setdefault('option_labels',[f'({chr(65+i)})' for i in range(len(row['choices']))])
            if len(row['option_labels'])!=len(row['choices']) or len(set(row['option_labels']))!=len(row['choices']):
                raise ValueError('Distinct option labels must match choices')
            row.setdefault('has_reference_answer','answer_index' in row)
            row.setdefault('answer_index',0)  # Only a formatting placeholder if no reference was supplied.
            if not isinstance(row['answer_index'],int) or not 0<=row['answer_index']<len(row['choices']):
                raise ValueError('answer_index must index choices')
            row.setdefault('answer_label',row['option_labels'][row['answer_index']])
            row.setdefault('prompt',row['query']+'\n'+''.join(f'{label} {choice}\n' for label,choice in zip(row['option_labels'],row['choices'])))
        rows.append(row)
    if not rows: raise ValueError('No selected material rows')
    # Every within-intervention query shares the actual intervention and sources.
    groups={}
    for r in rows:
        signature=digest({'unit':r['unit_id'],'sources':r['source_image_metadata'],'roi':r.get('donor_roi_bbox_xyxy')})
        previous=groups.setdefault(r['intervention_id'],signature)
        if previous!=signature: raise ValueError('An intervention id crosses sources/ROI/units')
    return sorted(rows,key=lambda r:(r['unit_id'],r['intervention_id'],r['id']))


def readouts(row,config):
    if not row.get('choices'):
        return [{'response_format':'freeform','mapping':'none'}]
    return row.get('readouts',config['readouts'])


def conditions(row,config):
    families=set(config['families'])
    if 'families' in row:
        # A material can narrow standard families; explicit custom plans are
        # independent new interventions on those same images.
        families=(families&set(row['families']))|(families&{'custom'})
    found=[]
    if 'conflict' in families and config['later_donor'] in row['source_images']:
        for quality,source in config['main_sources'].items():
            for endpoint,t in [('a',0.),('b',1.)]:
                found.append(dict(name=f'conflict__{quality}__{endpoint}',family='conflict',
                    main_quality=quality,main_source=source,later_endpoint=endpoint,
                    later_base=config['later_base'],later_donor=config['later_donor'],
                    t=t,k=deepcopy(config['k']),entries=deepcopy(config['entries']),region=config['region'],
                    answer_semantics='source influence; no intrinsic mixed-state ground truth'))
    if 'recovery' in families:
        for mq,main in config['recovery_sources'].items():
            for lq,later in config['recovery_sources'].items():
                found.append(dict(name=f'recovery__{mq}__{lq}',family='recovery',main_quality=mq,
                    later_quality=lq,main_source=main,later_base=later,later_donor=later,
                    t=0.,k=1.,entries='all_later',region='all',
                    answer_semantics='clean-reference recovery; same underlying image'))
    for custom in config['conditions']:
        if custom.get('family','custom') not in families and 'custom' not in families: continue
        found.append(deepcopy(custom))
    names=set()
    for c in found:
        if c['name'] in names: raise ValueError('Duplicate condition name')
        names.add(c['name'])
        for key in ('main_source','later_base','later_donor'):
            if c[key] not in row['source_images']: raise ValueError(f"{row['id']} needs source {c[key]}")
        if c['region'] not in ('roi','all'): raise ValueError('Supported support: roi/all')
        if c['region']=='roi' and not row.get('donor_roi_bbox_xyxy'): raise ValueError('ROI condition without ROI')
        ks=c['k'].values() if isinstance(c['k'],dict) else [c['k']]
        if any(not math.isfinite(k) or k<0 for k in ks) or not math.isfinite(c['t']): raise ValueError('Finite nonnegative k required')
        if c['entries']!='all_later' and not isinstance(c['entries'],list): raise ValueError('entries must be all_later or a list')
    if not found: raise ValueError('No applicable conditions for '+row['id'])
    return found


def semantic_row(row):
    # Paths can move to a cloud mount; input byte identity cannot silently change.
    return {k:v for k,v in row.items() if k not in ('source_images','material_location')}


def observation_key(row,condition,readout,kind,model,config):
    semantics=dict(row=semantic_row(row),condition=condition,readout=readout,kind=kind,
        model=model,execution_version=config['execution_version'],seed=config['seed'],
        qwen_pixel_limits=config['qwen_pixel_limits'],
        generate_tokens=config['generate_tokens'] if kind=='generation' else None)
    return digest(semantics)


def execution_plan(rows,model,config):
    counts={'score':0,'generation':0}; condition_counts={}
    for row in rows:
        cc=conditions(row,config); condition_counts[row['id']]=len(cc)
        for readout in readouts(row,config):
            if readout['response_format']!='freeform': counts['score']+=len(cc)
            if config['generate_tokens']: counts['generation']+=len(cc)
    return dict(schema=SCHEMA,model=model,config=config,query_count=len(rows),
        units=len({r['unit_id'] for r in rows}),interventions=len({r['intervention_id'] for r in rows}),
        expected_logical_rows=counts,conditions_per_query=condition_counts,
        actual_inference='not executed by planning',
        policy='Editable exploration. Preserve accepted rows; version changes that alter numerical/scientific meaning. No repository hash gate.')
