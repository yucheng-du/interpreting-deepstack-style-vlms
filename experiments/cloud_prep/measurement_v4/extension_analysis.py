"""Strict CPU summaries for completed natural Qwen and Gemma v4 attempts.

Requires the versioned manifest so generation coverage is checked against the
planned IDs, not the successfully observed outputs. Imports no model libraries.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys

from cloud_prep.measurement_v4.natural_analysis import average as base_average
from cloud_prep.measurement_v4.content_analysis import factorial_effects

SCHEMAS = {'natural-qwen-ds-content-v4-1':'natural_qwen',
           'gemma-content-reachability-v4-1':'gemma_reachability'}
BITS = ('000','100','010','001','110','101','011','111')
KV_METRICS = tuple(f'{kind}_{group}_{stat}_delta' for kind in ('key','value')
                   for group in ('visual','nonvisual') for stat in ('max_abs','l2'))
GROUP = ('arm','model','revision','task','format','mapping')
DEFINITIONS = {
    'status':'Analysis preparation; only supplied completed attempts become observations. Synthetic test fixtures are never reported as model results.',
    'unit':'Question within checkpoint/task/format/mapping/main; repeated conditions, mappings, KV forwards and checkpoints are not independent samples.',
    'content_direction':'Log score of the swapped-gold candidate label minus log score of the original-gold candidate label, fixed across all conditions.',
    'natural_baseline':'DS bits 000 within the same main. Bits 1 move that single DS candidate-content source; no path is disabled.',
    'natural_interaction':'Möbius finite difference at other DS sources original (0); pair/triple interaction is conditional nonadditivity, not semantic specialization.',
    'average_switch':'Single DS switch effect averaged over the four configurations of the other two sources, within question and main.',
    'main_by_ds':'Difference between main-swapped and main-original baseline-relative DS subset effects, within the same question/mapping.',
    'gemma_baseline':'native; real_swap changes main image and transports scored gold, so fixed original-gold margin is also reported separately.',
    'correct':'Preserves each runner scoring policy: original gold for natural feature conditions and Gemma PLE, transported gold for Gemma real_swap.',
    'prediction_identity':'Undo label_to_original to identify candidate slot; this is not transported image-object identity or a measured binding mechanism.',
    'generation':'Stored parser/stop criteria retained; invalid outputs stay in all planned denominators and transition tables. No answer recovery from prose.',
    'kv_unit':'Maximum across label-continuation forward ordinals within each question/producer/attention type/metric, then summary across questions. Per-forward scalars retained.',
    'kv_reference':'native and real_swap establish their own reference and have no delta measurement. Missing reference deltas are recorded as unmeasured, never zero.',
    'kv_scope':'Teacher-forced actual shared producer K/V; separately key/value and visual/nonvisual. No intervention-generation KV measurement is implied.',
    'uncertainty':'Descriptive question-level SE and normal 95% interval, not confirmation or multiplicity-adjusted discovery; small panels require caution.',
    'reachability':'Null or nonzero effects describe this image-position content intervention and its producer reachability, not global redundancy.',
}


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def read_jsonl(path):
    with path.open() as f:return [json.loads(x) for x in f if x.strip()]


def unique(rows, fields):
    out={}
    for row in rows:
        key=tuple(row[k] for k in fields)
        if key in out:raise ValueError(f'Duplicate cell: {key}')
        out[key]=row
    return out


def average(values):
    values=list(values)
    base=base_average(values)
    return {'n_questions':base['n'],'mean':base['mean'],'se':base['se'],
            'normal_95_interval':base.get('normal_95_interval'),
            'min':min(values) if values else None,'max':max(values) if values else None,
            'positive':sum(v>0 for v in values),'negative':sum(v<0 for v in values),'zero':sum(v==0 for v in values)}


def factorial(values):
    if set(values)!=set(BITS):raise ValueError('Expected eight DS-source cells')
    # Reuse the content arm's established finite-difference implementation.
    effects=factorial_effects({b.replace('0','a').replace('1','b'):v for b,v in values.items()})
    effects.update({f'joint_{bits}_minus_000':values[bits]-values['000'] for bits in BITS if bits!='000'})
    return effects


def cell_key(row):return tuple(row[k] for k in ('id','format','mapping','condition'))


def validate_attempt(meta,scores,generations,manifest,manifest_sha256):
    """Return exact expected matrices, rejecting incomplete/ambiguous attempts."""
    if meta.get('schema') not in SCHEMAS:raise ValueError('Unsupported extension schema')
    if not meta.get('finished_at') or meta.get('failed_at'):raise ValueError('Select completed nonfailed attempts only')
    if meta['manifest_sha256']!=manifest_sha256:raise ValueError('Manifest hash mismatch')
    ids=meta['ids'];mappings=meta['mappings'];conditions=[c['name'] for c in meta['conditions']]
    if len(ids)!=len(set(ids)) or len(mappings)!=len(set(mappings)) or len(conditions)!=len(set(conditions)):
        raise ValueError('Duplicate planned IDs/mappings/conditions')
    if set(ids)-set(manifest):raise ValueError('Planned ID absent from manifest')
    is_qwen=meta['schema']=='natural-qwen-ds-content-v4-1'
    if is_qwen and set(conditions)!={f'm{m}_ds{b}' for m in (0,1) for b in BITS}:
        raise ValueError('Natural Qwen requires the complete 2x8 source factorial')
    if not is_qwen and 'native' not in conditions:raise ValueError('Gemma native reference absent')
    gen_conditions=conditions if is_qwen else meta['generation_conditions']
    if len(set(gen_conditions))!=len(gen_conditions) or set(gen_conditions)-set(conditions):raise ValueError('Invalid generation condition plan')
    gen_ids=[i for i in ids if manifest[i]['natural_v4']['generation_selected']] if meta['generate_tokens'] else []
    expected_scores={(i,meta['format'],m,c) for i,m,c in itertools.product(ids,mappings,conditions)}
    expected_gen={(i,meta['format'],m,c) for i,m,c in itertools.product(gen_ids,mappings,gen_conditions)}
    for kind,rows,expected in (('score',scores,expected_scores),('generation',generations,expected_gen)):
        ix=unique(rows,('id','format','mapping','condition'))
        if set(ix)!=expected:
            missing=sorted(expected-set(ix));unexpected=sorted(set(ix)-expected)
            raise ValueError(f'{kind} matrix mismatch: missing={missing[:3]}, unexpected={unexpected[:3]}')
        if len(rows)!=meta['expected_'+kind+'_rows'] or len(rows)!=meta['completed_'+kind+'_rows']:
            raise ValueError(f'{kind} metadata row count mismatch')
        for r in rows:
            if r['protocol_sha256']!=meta['protocol_sha256']:raise ValueError('Protocol mismatch')
            source=manifest[r['id']]
            if r['task']!=source['task'] or r['original_answer_label']!=source['answer_label']:
                raise ValueError('Question task/original gold identity mismatch')
            if r['image_sha256']!=[x['sha256'] for x in source['image_metadata']]:raise ValueError('Question image identity mismatch')
            mapping=r['label_to_original']
            if len(mapping)!=len(set(mapping.values())):raise ValueError('Answer map is not one-to-one')
            original=r['answer_label'] if is_qwen else r['original_gold_label']
            swapped=r['swapped_answer_label'] if is_qwen else r['swapped_gold_label']
            if mapping.get(original)!=source['answer_label'] or mapping.get(swapped)!=source['natural_v4']['swapped_answer_label']:
                raise ValueError('Fixed content direction disagrees with manifest/mapping')
            expected_gold=swapped if not is_qwen and r['condition']=='real_swap' else original
            if r['answer_label']!=expected_gold:raise ValueError('Scored gold policy mismatch')
            if is_qwen:
                cfg=next(c for c in meta['conditions'] if c['name']==r['condition'])
                if any(r[f]!=cfg[f] for f in ('main','ds_swap_bits','ds_sources')):raise ValueError('Natural condition/source metadata mismatch')
    return {'score_rows':len(scores),'generation_rows':len(generations),'unique_questions':len(ids),
            'planned_generation_ids':gen_ids,'expected_generation_rows':len(expected_gen)}


def normalize_score(row):
    scores=row['scores'];mapping=row['label_to_original']
    if set(scores)!=set(mapping) or any(not math.isfinite(v) for v in scores.values()):raise ValueError('Invalid option scores')
    best=max(scores.values());pred=row['prediction']
    if pred not in mapping or scores[pred]!=best:raise ValueError('Prediction is not a highest-score label')
    original=row.get('original_gold_label',row['answer_label'])
    swapped=row.get('swapped_gold_label',row.get('swapped_answer_label'))
    z=scores[swapped]-scores[original]
    margin=scores[row['answer_label']]-max(v for k,v in scores.items() if k!=row['answer_label'])
    if not math.isclose(z,row['directed_content_score'],abs_tol=1e-6):raise ValueError('Directed content score mismatch')
    if not math.isclose(margin,row['correct_margin'],abs_tol=1e-6):raise ValueError('Scored margin mismatch')
    if row['correct']!=(pred==row['answer_label']):raise ValueError('Score correctness mismatch')
    return {**row,'original_prediction':mapping[pred],'original_gold_correct':pred==original,
        'swapped_gold_selected':pred==swapped,'original_gold_margin':scores[original]-max(v for k,v in scores.items() if k!=original),
        'highest_score_tie':sum(v==best for v in scores.values())>1}


def normalize_generation(row):
    parsed=row['parse'];label=parsed.get('label');mapping=row['label_to_original']
    usable=bool(parsed['status']=='parsed' and label in mapping and row['eos_reached'] and not row['truncated'])
    if usable!=row['scorable'] or row['prediction']!=(label if usable else None):raise ValueError('Stored generation parse/scorable mismatch')
    if row['correct']!=bool(usable and label==row['answer_label']):raise ValueError('Generation correctness mismatch')
    original=row.get('original_gold_label',row['answer_label'])
    swapped=row.get('swapped_gold_label',row.get('swapped_answer_label'))
    state='invalid' if not usable else 'original_gold_slot' if label==original else 'swapped_gold_slot' if label==swapped else 'other_slot'
    return {**row,'original_prediction':mapping[label] if usable else None,
            'original_gold_correct':bool(usable and label==original),'swapped_gold_selected':bool(usable and label==swapped),'state':state}


def summarize(attempts,manifest,manifest_sha256):
    manifest=unique(manifest,('id',));manifest={key[0]:r for key,r in manifest.items()}
    scores=[];generations=[];checks=[];validations=[];protocols={}
    for attempt in attempts:
        meta=attempt['metadata'];validation=validate_attempt(meta,attempt['scores'],attempt['generations'],manifest,manifest_sha256)
        arm=SCHEMAS[meta['schema']];model=meta['checkpoint']['model'];revision=meta['checkpoint']['revision']
        compatibility={f:meta.get(f) for f in ('conditions','format','mappings','parser','max_sequence','generate_tokens','generation_conditions','source_sha256')}
        group=(arm,model,revision)
        if group in protocols and protocols[group]!=compatibility:raise ValueError('Cannot pool incompatible attempt protocols/source versions')
        protocols[group]=compatibility
        shared={'arm':arm,'model':model,'revision':revision,'source_attempt':attempt.get('path','test_fixture')}
        scores.extend({**normalize_score(r),**shared} for r in attempt['scores'])
        generations.extend({**normalize_generation(r),**shared} for r in attempt['generations'])
        checks.extend({**r,**shared} for r in attempt.get('semantic_checks',[]))
        validations.append({**shared,**validation})
    if not scores:raise ValueError('No completed score observations')
    fields=('arm','model','revision','id','format','mapping','condition')
    score_index=unique(scores,fields);gen_index=unique(generations,fields)
    def baseline(r):
        return ('m1_ds000' if r['main']=='swapped' else 'm0_ds000') if r['arm']=='natural_qwen' else 'native'
    def get_base(r,index):return index[tuple(r[f] for f in fields[:-1])+(baseline(r),)]
    item_effects=[];groups=defaultdict(list);kv_items=[];kv_groups=defaultdict(list)
    for r in scores:
        b=get_base(r,score_index);item={k:r[k] for k in (*GROUP,'id','condition')}
        item.update(main=r.get('main','real_swap' if r['condition']=='real_swap' else 'original'),
            directed_content_score=r['directed_content_score'],delta_directed_content_score=r['directed_content_score']-b['directed_content_score'],
            delta_original_gold_margin=r['original_gold_margin']-b['original_gold_margin'],
            original_prediction=r['original_prediction'],baseline_original_prediction=b['original_prediction'],
            prediction_flip=r['original_prediction']!=b['original_prediction'],correct=r['correct'],baseline_correct=b['correct'],
            original_gold_correct=r['original_gold_correct'],swapped_gold_selected=r['swapped_gold_selected'],
            highest_score_tie=r['highest_score_tie'],scores=r['scores'],label_to_original=r['label_to_original'])
        item_effects.append(item);groups[tuple(item[f] for f in (*GROUP,'main','condition'))].append(item)
        if r['arm']=='gemma_reachability':
            traces=r['kv_propagation'];unique(traces,('producer_layer','attention_type','forward_ordinal'))
            per=defaultdict(list)
            reference_only=r['condition'] in ('native','real_swap')
            for t in traces:
                available=set(t)&set(KV_METRICS)
                if available and available!=set(KV_METRICS):raise ValueError('Incomplete KV delta record')
                if not reference_only and available!=set(KV_METRICS):raise ValueError('Patched KV delta record missing')
                for metric in available:
                    value=t[metric]
                    if not math.isfinite(value) or value<0:raise ValueError('Invalid KV delta magnitude')
                    per[(t['producer_layer'],t['attention_type'],metric)].append((t['forward_ordinal'],value))
            if not traces:raise ValueError('Missing Gemma producer trace records')
            if reference_only and not per:
                kv_items.append({**{k:r[k] for k in (*GROUP,'id','condition')},'status':'reference_without_delta',
                                 'forward_records':[{k:t[k] for k in ('producer_layer','attention_type','forward_ordinal')} for t in traces]})
            for (producer,attn,metric),values in sorted(per.items()):
                kr={**{k:r[k] for k in (*GROUP,'id','condition')},'status':'measured','producer_layer':producer,
                    'attention_type':attn,'metric':metric,'forward_count':len(values),'max_across_forwards':max(v for _,v in values),
                    'forward_values':[{'forward_ordinal':i,'value':v} for i,v in sorted(values)]}
                kv_items.append(kr);kv_groups[tuple(kr[f] for f in (*GROUP,'condition','producer_layer','attention_type','metric'))].append(kr['max_across_forwards'])
    condition_summaries=[]
    for key,rows in sorted(groups.items()):
        condition_summaries.append({**dict(zip((*GROUP,'main','condition'),key)),'n_questions':len(rows),
            'delta_directed_content_score':average(r['delta_directed_content_score'] for r in rows),
            'delta_original_gold_margin':average(r['delta_original_gold_margin'] for r in rows),
            **{f:sum(r[f] for r in rows) for f in ('prediction_flip','correct','original_gold_correct','swapped_gold_selected','highest_score_tie')}})
    factorial_items=[];factorial_groups=defaultdict(list)
    qgroups=defaultdict(dict)
    for r in scores:
        if r['arm']=='natural_qwen':qgroups[tuple(r[k] for k in (*GROUP,'id','main'))][r['ds_swap_bits']]=r['directed_content_score']
    for key,values in sorted(qgroups.items()):
        effect={**dict(zip((*GROUP,'id','main'),key)),'values':values,'effects':factorial(values)}
        factorial_items.append(effect)
        for metric,value in effect['effects'].items():factorial_groups[tuple(effect[k] for k in (*GROUP,'main'))+(metric,)].append(value)
    bymain=defaultdict(dict)
    for r in factorial_items:bymain[tuple(r[k] for k in (*GROUP,'id'))][r['main']]=r
    main_interactions=[]
    for key,mains in sorted(bymain.items()):
        if set(mains)!= {'original','swapped'}:raise ValueError('Missing natural main counterpart')
        effect={f'main_by_joint_{b}':mains['swapped']['effects'][f'joint_{b}_minus_000']-mains['original']['effects'][f'joint_{b}_minus_000'] for b in BITS if b!='000'}
        record={**dict(zip((*GROUP,'id'),key)),'main':'swapped_minus_original','effects':effect}
        main_interactions.append(record)
        for metric,value in effect.items():factorial_groups[tuple(record[k] for k in (*GROUP,'main'))+(metric,)].append(value)
    gen_groups=defaultdict(list);pair_groups=defaultdict(list);generation_pairs=[]
    for r in generations:
        key=tuple(r[k] for k in (*GROUP,'condition'));gen_groups[key].append(r)
        b=get_base(r,gen_index);valid=r['scorable'] and b['scorable']
        pair={**{k:r[k] for k in (*GROUP,'id','condition')},'baseline_condition':baseline(r),'both_scorable':valid,
              'main':r.get('main','real_swap' if r['condition']=='real_swap' else 'original'),
              'left_state':b['state'],'right_state':r['state'],'prediction_flip':r['original_prediction']!=b['original_prediction'] if valid else None,
              'baseline_original_prediction':b['original_prediction'],'original_prediction':r['original_prediction']}
        generation_pairs.append(pair);pair_groups[key].append(pair)
    generation_summaries=[]
    for key,rows in sorted(gen_groups.items()):
        pairs=pair_groups[key];valid=[r for r in pairs if r['both_scorable']]
        agreement=sum(r['scorable'] and r['original_prediction']==score_index[tuple(r[f] for f in fields)]['original_prediction'] for r in rows)
        generation_summaries.append({**dict(zip((*GROUP,'condition'),key)),
            'main':rows[0].get('main','real_swap' if rows[0]['condition']=='real_swap' else 'original'),
            'planned_questions':len(rows),'observed':len(rows),
            'scorable':sum(r['scorable'] for r in rows),'invalid':sum(not r['scorable'] for r in rows),
            'correct_all_planned':sum(r['correct'] for r in rows),'original_gold_correct':sum(r['original_gold_correct'] for r in rows),
            'swapped_gold_selected':sum(r['swapped_gold_selected'] for r in rows),
            'parse_status':dict(Counter(r['parse']['status'] for r in rows)),'stop_reason':dict(Counter(r['stop_reason'] for r in rows)),
            'decode_contract':dict(Counter(r.get('decode_contract','cached_native_greedy') for r in rows)),
            'both_scorable_with_baseline':len(valid),'invalid_in_either':len(pairs)-len(valid),
            'prediction_flips_common_valid':sum(r['prediction_flip'] is True for r in pairs),
            'transition_states':dict(Counter(r['left_state']+'->'+r['right_state'] for r in pairs)),
            'score_generation_agreement':{'n':agreement,'denominator':sum(r['scorable'] for r in rows)}})
    # Raw generation text/tokens/parser/stop fields are retained in full, including invalids.
    report={'schema':'v4-extension-analysis-v1','definitions':DEFINITIONS,'attempt_validation':validations,
        'observations':{'score_rows':len(scores),'generation_rows':len(generations),
            'unique_question_ids':len({r['id'] for r in scores}),
            'checkpoint_question_pairs':len({(r['model'],r['revision'],r['id']) for r in scores})},
        'condition_summaries':condition_summaries,
        'natural_factorial_summaries':[{**dict(zip((*GROUP,'main','metric'),key)),**average(values)} for key,values in sorted(factorial_groups.items())],
        'generation_summaries':generation_summaries,
        'kv_summaries':[{**dict(zip((*GROUP,'condition','producer_layer','attention_type','metric'),key)),**average(values)} for key,values in sorted(kv_groups.items())],
        'semantic_checks':checks}
    artifacts={'score_item_effects':item_effects,'natural_factorial_items':factorial_items,'natural_main_interaction_items':main_interactions,
        'generation_rows':generations,'generation_pair_items':generation_pairs,'kv_item_effects':kv_items}
    return report,artifacts


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',nargs='+',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args(argv)
    if not a.output_dir.is_absolute():p.error('Use absolute output-dir')
    if a.output_dir.exists():raise FileExistsError('Use a new versioned output directory')
    attempts=[];sources=[a.manifest,Path(__file__),Path(__file__).with_name("natural_analysis.py"),Path(__file__).with_name("content_analysis.py")]
    for run in a.runs:
        meta=run/'metadata.json';sp=run/'scores.jsonl';gp=run/'generations.jsonl';vp=run/'semantic_checks.jsonl'
        attempts.append({'path':str(run),'metadata':json.loads(meta.read_text()),'scores':read_jsonl(sp),
                         'generations':read_jsonl(gp),'semantic_checks':read_jsonl(vp) if vp.exists() else []})
        sources.extend([meta,sp,gp]);sources.extend([vp] if vp.exists() else [])
    report,artifacts=summarize(attempts,read_jsonl(a.manifest),digest(a.manifest))
    report['provenance']={'files':{str(p):digest(p) for p in sources},'command':sys.argv}
    a.output_dir.mkdir(parents=True,exist_ok=False)
    for name,rows in artifacts.items():
        (a.output_dir/(name+'.jsonl')).write_text(''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in rows))
    (a.output_dir/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'output':str(a.output_dir),'observations':report['observations']}))


if __name__=='__main__':main()
