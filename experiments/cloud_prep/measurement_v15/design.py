"""Editable follow-up plans and row identities, without tensor imports.

Hashes identify observations; they do not freeze this directory or a campaign.
Changing one condition leaves other observation keys reusable. A numerical code
repair should use a new execution_version for affected rows, not erase old data.

v15 (restored from the executed v14 c011be4 design): rows may carry several
named severities (``severity_sources``). With ``severity_sources: "row"`` every
selected label L adds the conflict cells L x {a,b} and the recovery cells
(clean,L), (L,clean), (L,L) that share one degraded file; the clean/normal
reference cells are shared by all labels. ``recovery_cells: native_only`` keeps
only the clean and joint-degraded cells (calibration). Observation keys use only
the sources a condition reads, so adding or dropping other severities leaves
existing keys unchanged. Defaults keep the v14 behaviour.
"""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

SCHEMA = 'native-entry-conditional-use-v15'
RECOVERY_CELLS = ('four_cell', 'native_only')
DEFAULTS = dict(execution_version='v14-exploration-1', families=['conflict', 'recovery'],
    source_aliases={}, main_sources={'normal':'a', 'degraded':'degraded_a'},
    later_base='a', later_donor='b', region='roi', k=1., entries='all_later',
    recovery_sources={'clean':'a','degraded':'degraded_a'},
    readouts=[{'response_format':'direct','mapping':'none'},
              {'response_format':'label','mapping':'identity'},
              {'response_format':'label','mapping':'cyclic'}],
    generate_tokens=32, max_sequence=4096, seed=20261028, conditions=[], comparisons=[],
    self_patch_tolerance=.001, audit_per_task=True, qwen_pixel_limits=None,
    # Batched execution (cloud_prep/measurement_v15/batched.py). Value-changing at BF16
    # level, so only configs with their own execution_version enable it. Defaults keep the
    # batch-one path; these keys are not part of observation keys. They enter
    # execution_signature: the runner refuses to resume an output directory across a
    # change, and the analysis reports mixtures (v14 saw near-tie flips with batch shape).
    batch_conditions=False, cross_query_cache=False, max_batch=8,
    # v15 severity curves: None keeps v14 rows; 'row' reads row['severity_sources'].
    severity_sources=None, severity_labels=None, recovery_cells='four_cell')


def execution_signature(config,libraries=None,cuda_device=None):
    """Numerical execution settings outside observation keys (batch shape, libraries, GPU type)."""
    batched=bool(config.get('batch_conditions'))
    return dict(batch_conditions=batched,cross_query_cache=bool(config.get('cross_query_cache')) if batched else None,
        max_batch=config.get('max_batch',8) if batched else None,
        libraries={k:(libraries or {}).get(k) for k in ('torch','transformers','Pillow')},cuda_device=cuda_device)


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
    if result['severity_sources'] not in (None,'row'): raise ValueError("severity_sources must be null or 'row'")
    if result['recovery_cells'] not in RECOVERY_CELLS: raise ValueError('recovery_cells must be one of '+str(RECOVERY_CELLS))
    chosen=result['severity_labels']
    if chosen is not None:
        if result['severity_sources']!='row': raise ValueError("severity_labels needs severity_sources 'row'")
        lists=chosen.values() if isinstance(chosen,dict) else [chosen]
        from cloud_prep.measurement_v15.severity import parse_label
        for labels in lists:
            if not isinstance(labels,list) or len(set(labels))!=len(labels): raise ValueError('severity_labels: distinct label lists required')
            for label in labels: parse_label(label)
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
        if row.get('severity_sources'): check_severities(row)
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


def check_severities(row):
    """A row's severities: canonical labels, ordered mild to severe, each one degraded file with its spec."""
    from cloud_prep.measurement_v15.severity import ordered,parse_label,spec_id
    named=row['severity_sources']
    if not isinstance(named,dict) or not named: raise ValueError('severity_sources must map labels to sources: '+row['id'])
    if list(named)!=ordered(list(named)): raise ValueError('severity_sources must be ordered mild to severe: '+row['id'])
    for label,source in named.items():
        parse_label(label)
        if source in ('a','b') or source not in row['source_images']: raise ValueError(f"{row['id']}: severity {label} needs its own degraded source")
        spec=row.get('degradations',{}).get(label)
        if not isinstance(spec,dict) or spec.get('label')!=label or spec.get('spec_id')!=spec_id(spec):
            raise ValueError(f"{row['id']}: degradation spec of {label} is missing or inconsistent")
    if len(set(named.values()))!=len(named): raise ValueError('Each severity needs a distinct source: '+row['id'])


def severity_labels(row,config):
    """Labels of this row's severities selected by the run config, mild to severe."""
    if config['severity_sources']!='row' or not row.get('severity_sources'): return []
    available=list(row['severity_sources']); chosen=config['severity_labels']
    if chosen is None: return available
    if isinstance(chosen,dict):
        if row['task'] not in chosen: raise ValueError('severity_labels has no entry for task '+row['task'])
        chosen=chosen[row['task']]
        missing=[x for x in chosen if x not in available]
        if missing: raise ValueError(f"{row['id']} lacks selected severities {missing}")
    found=[x for x in available if x in chosen]
    if chosen and not found: raise ValueError(f"No selected severity matches {row['id']} (available {available})")
    return found


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
    found=[]; labels=severity_labels(row,config); graded=config['severity_sources']=='row'
    native_only=config['recovery_cells']=='native_only'
    def tag(c,label):
        if graded: c['severity_label']=label
        return c
    if 'conflict' in families and config['later_donor'] in row['source_images']:
        mains=[(q,s,None) for q,s in config['main_sources'].items()]+[(x,row['severity_sources'][x],x) for x in labels]
        for quality,source,label in mains:
            for endpoint,t in [('a',0.),('b',1.)]:
                found.append(tag(dict(name=f'conflict__{quality}__{endpoint}',family='conflict',
                    main_quality=quality,main_source=source,later_endpoint=endpoint,
                    later_base=config['later_base'],later_donor=config['later_donor'],
                    t=t,k=deepcopy(config['k']),entries=deepcopy(config['entries']),region=config['region'],
                    answer_semantics='source influence; no intrinsic mixed-state ground truth'),label))
    if 'recovery' in families:
        def cell(mq,main,lq,later,label):
            return tag(dict(name=f'recovery__{mq}__{lq}',family='recovery',main_quality=mq,
                later_quality=lq,main_source=main,later_base=later,later_donor=later,
                t=0.,k=1.,entries='all_later',region='all',
                answer_semantics='clean-reference recovery; same underlying image'),label)
        for mq,main in config['recovery_sources'].items():
            for lq,later in config['recovery_sources'].items():
                if not native_only or mq==lq: found.append(cell(mq,main,lq,later,None))
        if labels:
            clean=[q for q,s in config['recovery_sources'].items() if s==config['later_base']]
            if len(clean)!=1: raise ValueError('Severity recovery needs exactly one clean quality whose source is later_base')
            clean,source=clean[0],config['later_base']
            for x in labels:
                degraded=row['severity_sources'][x]
                pairs=[(x,degraded,x,degraded)] if native_only else [(clean,source,x,degraded),(x,degraded,clean,source),(x,degraded,x,degraded)]
                found.extend(cell(mq,main,lq,later,x) for mq,main,lq,later in pairs)
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


PER_SOURCE_FIELDS=('source_image_metadata','degradations','source_answer_indices')


def semantic_row(row,used=None):
    # Paths can move to a cloud mount; input byte identity cannot silently change.
    result={k:v for k,v in row.items() if k not in ('source_images','material_location')}
    if used is not None:
        # Only the sources a condition reads identify it; other severities may come and go.
        for field in PER_SOURCE_FIELDS:
            if isinstance(result.get(field),dict):
                result[field]={k:v for k,v in result[field].items() if k in used or (result.get('severity_sources') or {}).get(k) in used}
        if isinstance(result.get('severity_sources'),dict):
            result['severity_sources']={k:v for k,v in result['severity_sources'].items() if v in used}
    return result


def used_sources(condition):
    return {condition[k] for k in ('main_source','later_base','later_donor')}


def observation_key(row,condition,readout,kind,model,config):
    semantics=dict(row=semantic_row(row,used_sources(condition)),condition=condition,readout=readout,kind=kind,
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
    selected={}
    for row in rows:
        for label in severity_labels(row,config): selected.setdefault(row['task'],[]).append(label)
    chosen=config['severity_labels']
    tasks={r['task'] for r in rows}
    asked=set(x for t,v in chosen.items() if t in tasks for x in v) if isinstance(chosen,dict) else set(chosen or [])
    unused=sorted(asked-{x for v in selected.values() for x in v})
    if unused: raise ValueError('Selected severities absent from every row (typo?): '+str(unused))
    return dict(schema=SCHEMA,model=model,config=config,query_count=len(rows),
        units=len({r['unit_id'] for r in rows}),interventions=len({r['intervention_id'] for r in rows}),
        expected_logical_rows=counts,conditions_per_query=condition_counts,
        severity_labels_by_task={t:list(dict.fromkeys(v)) for t,v in sorted(selected.items())},
        actual_inference='not executed by planning',
        policy='Editable exploration. Preserve accepted rows; version changes that alter numerical/scientific meaning. No repository hash gate.')
