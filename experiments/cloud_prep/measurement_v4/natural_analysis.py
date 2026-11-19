"""CPU analysis for v4 natural Qwen/Gemma; all cells and invalid outputs retained."""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import json
import math
from pathlib import Path
import statistics

from cloud_prep.common import write_json,sha256


def average(values):
    values=list(values)
    if not values:return {'n':0,'mean':None,'se':None}
    if not all(math.isfinite(v) for v in values):raise ValueError('Nonfinite analysis value')
    mean=statistics.mean(values)
    se=statistics.stdev(values)/math.sqrt(len(values)) if len(values)>1 else None
    return {'n':len(values),'mean':mean,'se':se,'normal_95_interval':[mean-1.96*se,mean+1.96*se] if se is not None else None}


def keyed(rows):
    result={}
    for r in rows:
        k=(r['id'],r['format'],r['mapping'],r['condition'])
        if k in result:raise ValueError(f'Duplicate cell {k}')
        result[k]=r
    return result


def summarize(scores,generations,metadata):
    data=keyed(scores);gen=keyed(generations)
    if len(data)!=metadata['expected_score_rows'] or len(gen)!=metadata['expected_generation_rows']:
        raise ValueError('Incomplete rows; do not turn partial successful subset into full-panel analysis')
    declared={(i,metadata['format'],m,c['name']) for i in metadata['ids'] for m in metadata['mappings'] for c in metadata['conditions']}
    if set(data)!=declared:raise ValueError('Score identities differ from declared matrix')
    if any(r['protocol_sha256']!=metadata['protocol_sha256'] for r in [*scores,*generations]):raise ValueError('Protocol mismatch')
    groups=defaultdict(list);contrasts=[];interactions=[];kv_groups=defaultdict(list)
    is_qwen=metadata['schema'].startswith('natural-qwen')
    for key,r in data.items():
        base_condition=('m1_ds000' if r['main']=='swapped' else 'm0_ds000') if is_qwen else 'native'
        base=data[(r['id'],r['format'],r['mapping'],base_condition)]
        record={'id':r['id'],'task':r['task'],'format':r['format'],'mapping':r['mapping'],'condition':r['condition'],
            'main':r.get('main','real_swap' if r['condition']=='real_swap' else 'original'),
            'directed_content_score':r['directed_content_score'],
            'delta_directed_content_score':r['directed_content_score']-base['directed_content_score'],
            'prediction_changed':r['prediction']!=base['prediction'],
            'original_gold_correct':r['prediction']==(r.get('original_gold_label') or r['answer_label']),
            'scored_gold_correct':r['correct'],
            'highest_score_tie':sum(v==max(r['scores'].values()) for v in r['scores'].values())>1}
        contrasts.append(record);groups[(record['task'],record['format'],record['mapping'],record['main'],record['condition'])].append(record)
        for trace in r.get('kv_propagation',[]):
            for metric,value in trace.items():
                if metric.endswith('_delta'):
                    kv_groups[(r['mapping'],r['condition'],trace['producer_layer'],metric,r['id'])].append(value)
    summaries=[]
    for key,items in sorted(groups.items()):
        summaries.append(dict(zip(('task','format','mapping','main','condition'),key))|{
            'items':len(items),'delta_directed_content_score':average(r['delta_directed_content_score'] for r in items),
            'prediction_flips':sum(r['prediction_changed'] for r in items),
            'original_gold_correct':sum(r['original_gold_correct'] for r in items),
            'scored_gold_correct':sum(r['scored_gold_correct'] for r in items),'highest_score_ties':sum(r['highest_score_tie'] for r in items)})
    if is_qwen:
        for id_,fmt,mapping in sorted({k[:3] for k in data}):
            for main in (0,1):
                values={b:data[(id_,fmt,mapping,f'm{main}_ds{b}')]['directed_content_score'] for b in ('000','100','010','001','110','101','011','111')}
                for name,terms in {'01':{'110':1,'100':-1,'010':-1,'000':1},
                                   '02':{'101':1,'100':-1,'001':-1,'000':1},
                                   '12':{'011':1,'010':-1,'001':-1,'000':1},
                                   '012':{'111':1,'110':-1,'101':-1,'011':-1,'100':1,'010':1,'001':1,'000':-1}}.items():
                    interactions.append({'id':id_,'format':fmt,'mapping':mapping,'main':main,'interaction':name,
                                         'contrast':sum(values[b]*weight for b,weight in terms.items())})
    inter_groups=defaultdict(list)
    for r in interactions:inter_groups[(r['format'],r['mapping'],r['main'],r['interaction'])].append(r['contrast'])
    generations_grouped=defaultdict(list)
    for r in generations:generations_grouped[(r['format'],r['mapping'],r['condition'])].append(r)
    gen_summary=[]
    for key,items in sorted(generations_grouped.items()):
        gen_summary.append(dict(zip(('format','mapping','condition'),key))|{
            'all_rows':len(items),'scorable':sum(r['scorable'] for r in items),'correct_all_rows':sum(r['correct'] for r in items),
            'parse_status':dict(Counter(r['parse']['status'] for r in items)),
            'stop_reason':dict(Counter(r['stop_reason'] for r in items)),
            'correct_conditional_on_scorable':{'numerator':sum(r['correct'] for r in items),'denominator':sum(r['scorable'] for r in items)}})
    # Collapse repeated label-continuation forwards within item before describing KV effects.
    kv_by_group=defaultdict(list)
    for key,values in kv_groups.items():kv_by_group[key[:4]].append(max(values))
    return {'schema':'natural-gemma-analysis-v4-1','protocol_sha256':metadata['protocol_sha256'],
        'row_counts':{'score':len(scores),'generation':len(generations)},'unique_items':len(metadata['ids']),
        'condition_groups':summaries,'item_effects':contrasts,
        'interaction_groups':[dict(zip(('format','mapping','main','interaction'),key))|average(values) for key,values in sorted(inter_groups.items())],
        'item_interactions':interactions,'generation_groups':gen_summary,
        'kv_groups':[dict(zip(('mapping','condition','producer_layer','metric'),key))|average(values) for key,values in sorted(kv_by_group.items())],
        'interpretation':{'unit':'item within each mapping and main; mappings are repeated measures',
            'positive_direction':'towards candidate-swapped image identity relative to original gold',
            'interactions':'finite differences of directed log scores; not semantic specialization or unique mechanism',
            'uncertainty':'descriptive item SE and normal 95% interval; exploratory, no unseen confirmation claim',
            'invalid_generation':'all outputs in denominator; no successful-generation-only primary analysis',
            'post_kv_null':'content cannot reach text via later KV writing in this image-position intervention; not global redundancy'}}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(argv);meta=json.loads((a.run/'metadata.json').read_text())
    if not meta.get('finished_at'):raise ValueError('Analyze a completed v4 attempt')
    read=lambda name:[json.loads(line) for line in (a.run/name).read_text().splitlines() if line.strip()]
    summary=summarize(read('scores.jsonl'),read('generations.jsonl'),meta)
    summary['source_sha256']={name:sha256(a.run/name) for name in ('metadata.json','scores.jsonl','generations.jsonl')}
    if a.output.exists():raise FileExistsError(a.output)
    write_json(a.output,summary)
    print(json.dumps({'output':str(a.output),'row_counts':summary['row_counts']}))

if __name__=='__main__':main()
