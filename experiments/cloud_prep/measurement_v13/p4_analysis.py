"""P4 complete-template paired estimates; no outcome filtering or tensor imports."""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
from itertools import combinations,product
import json
import math
from pathlib import Path
from statistics import mean,stdev
from cloud_prep.measurement_v13 import p4_design as design
from cloud_prep.measurement_v8.runtime import write_json


def key(row): return tuple(row[k] for k in ('id','response_format','mapping','condition'))
def finite(x): return type(x) in (int,float) and math.isfinite(x)
def subtract(a,b): return a-b if a is not None and b is not None else None


def value(row,kind):
    if row is None or row.get('protocol_status')!='executed': return None
    if kind=='score':
        x=row.get('b_minus_a_log_odds')
        return x if finite(x) else None
    if not row.get('scorable') or row.get('truncated') or row.get('original_prediction') not in ('(A)','(B)'): return None
    # Directed answer-change probability, not accuracy and not +/-1 twice-scaling.
    return float(row['original_prediction']==row['source_b_original_choice'])


def summary(rows,fields):
    groups=defaultdict(lambda:defaultdict(list))
    for row in rows: groups[tuple(row[k] for k in fields)][row['template_id']].append(row['value'])
    out=[]
    for group,units in sorted(groups.items(),key=lambda x:str(x[0])):
        values=[mean(v) for v in units.values() if all(x is not None for x in v)]
        avg=mean(values) if values else None
        se=stdev(values)/math.sqrt(len(values)) if len(values)>1 else None
        out.append({**dict(zip(fields,group)),'planned_n':len(units),'complete_n':len(values),
            'mean':avg,'normal_95_interval':[avg-1.96*se,avg+1.96*se] if se is not None else None,
            'unit':'template_id','planned_edit_pairs':sum(map(len,units.values())),
            'complete_edit_pairs':sum(x is not None for v in units.values() for x in v)})
    return out


def analyze(records,plan,kind='score'):
    if plan.get('schema')!=design.SCHEMA: raise ValueError('P4 plan required')
    expected={}
    for q,r,c in product(plan['queries'],plan['readouts'],plan['conditions']):
        k=q['id'],r['response_format'],r['mapping'],c['name']
        if k in expected: raise ValueError('Duplicate planned row')
        expected[k]=(q,r,c)
    indexed={}; signatures={}; status=Counter()
    for row in records:
        k=key(row)
        if k not in expected or k in indexed: raise ValueError('Unplanned/duplicate observation')
        q,readout,c=expected[k]
        if row.get('schema')!=design.SCHEMA or row.get('model_key')!=plan['model']['key'] or row.get('execution_version')!=plan['execution_version']:
            raise ValueError('Model/schema/execution-version mismatch')
        if row.get('kind')!=kind or row.get('condition_spec')!=c: raise ValueError('Condition/kind mismatch')
        if plan['model'].get('revision') and row.get('model_revision')!=plan['model']['revision']:
            raise ValueError('Model revision mismatch')
        if any(row.get(f)!=v for f,v in q.items()): raise ValueError('Query identity changed')
        for side in ('a','b'):
            if row.get('source_'+side+'_original_choice')!=('(A)','(B)')[q['direction_'+side+'_index']]: raise ValueError('Direction changed')
        if c['role']!='native':
            dose=row.get('dose_metadata')
            if not isinstance(dose,dict): raise ValueError('Missing dose evidence')
            binding=(row.get('calibration_id'),row.get('anchor_query_id'),row.get('anchor_prompt_sha256'),dose.get('effective_k'),dose['matching']['status'])
            old=signatures.setdefault((q['intervention_id'],c['name']),binding)
            if old!=binding: raise ValueError('Shared visual intervention changed across query/readout')
            if not all(binding[:3]): raise ValueError('Missing common-B anchor binding')
            if row.get('protocol_status')=='unmatched_dose':
                if value(row,kind) is not None or (kind=='score' and row.get('b_minus_a_log_odds') is not None): raise ValueError('Unmatched observation carries outcome')
            elif c['mode'].startswith('common_low') and dose['matching']['status']!='matched': raise ValueError('Unmatched enters matched analysis')
            elif c['mode']=='baseline' and dose.get('effective_k')!=0: raise ValueError('Common baseline has nonzero k')
            elif c['mode']=='raw' and dose.get('effective_k')!=1: raise ValueError('Raw k1 changed')
        indexed[k]=row
        status[row.get('protocol_status','unknown')]+=1
    effects=[]
    for k,(q,r,c) in expected.items():
        if c['role']!='intervention': continue
        basekey=k[:3]+(c['context']['name']+'__B',)
        row=indexed.get(k); baseline=indexed.get(basekey)
        effects.append({**q,**r,'context':c['context']['name'],'group':c['group'],'entries':c['entries'],
            'mode':c['mode'],'kind':kind,'effect':subtract(value(row,kind),value(baseline,kind)),
            'missing_cells':sum(x is None for x in (row,baseline)),
            'unmatched_cells':sum(x is not None and x.get('protocol_status')=='unmatched_dose' for x in (row,baseline)),
            'invalid_generation_cells':sum(x is not None and x.get('protocol_status')=='executed' and value(x,kind) is None for x in (row,baseline)) if kind=='generation' else 0})
    paired=defaultdict(dict)
    for e in effects:
        pk=tuple(e[f] for f in ('intervention_id','response_format','mapping','context','group','mode'))
        if e['query_role'] in paired[pk]: raise ValueError('Duplicate target/control')
        paired[pk][e['query_role']]=e
    selectivity=[]; measures=[]
    for cells in paired.values():
        if set(cells)!={'changed','unchanged'}: raise ValueError('Plan lacks paired queries')
        target,control=cells['changed'],cells['unchanged']
        row={**{k:target[k] for k in ('template_id','intervention_id','attribute','changed_index','response_format','mapping','context','group','entries','mode','kind')},
            'E_target':target['effect'],'E_control':control['effect'],'R':subtract(target['effect'],control['effect'])}
        selectivity.append(row)
        for measure in ('E_target','E_control','R'): measures.append({**row,'measure':measure,'value':row[measure]})
    by_intervention=defaultdict(dict)
    for r in selectivity:
        pk=tuple(r[f] for f in ('intervention_id','response_format','mapping','context','mode'))
        by_intervention[pk][r['group']]=r
    contrasts=[]
    for available in by_intervention.values():
        example=next(iter(available.values()))
        base={k:example[k] for k in ('template_id','intervention_id','attribute','changed_index','response_format','mapping','context','mode','kind')}
        gs=design.groups(plan['model']['key'],plan['config'])
        main=next(g['name'] for g in gs if g['entries']==[design.native_entries(plan['model']['key'])[0]])
        pairs=[]
        for left,right in combinations(gs,2):
            if len(left['entries'])==len(right['entries']) and main not in (left['name'],right['name']): pairs.append(('same_count',left['name'],right['name']))
        pairs += [('main_minus_group',main,g['name']) for g in gs if g['name']!=main]
        if 'all' in available: pairs += [('all_minus_group','all',g['name']) for g in gs if g['name'] not in (main,'all')]
        for contrast,left,right in pairs:
            contrasts.append({**base,'contrast':contrast,'left':left,'right':right,
                'value':subtract(available[left]['R'],available[right]['R'])})
        # Raw all contains the exact same single-entry k1 deltas. Matched groups
        # have different per-entry k, so that additive contrast is not emitted.
        if base['mode']=='raw' and 'all' in available:
            singles=[available[g['name']]['R'] for g in gs if len(g['entries'])==1 and g['name']!=main]
            contrasts.append({**base,'contrast':'raw_all_minus_sum_singletons','left':'all','right':'sum_native_later_singletons',
                'value':available['all']['R']-sum(singles) if available['all']['R'] is not None and all(v is not None for v in singles) else None})
    fields=['kind','attribute','response_format','mapping','context','mode']
    native=[]
    for c in plan['conditions']:
        if c['role']!='native': continue
        for readout in plan['readouts']:
            for attribute in plan['config']['attributes']:
                obs=[indexed.get((q['id'],readout['response_format'],readout['mapping'],c['name'])) for q in plan['queries'] if q['attribute']==attribute]
                known=[r['native_correct'] for r in obs if r is not None and r.get('native_correct') is not None]
                native.append(dict(condition=c['name'],**readout,attribute=attribute,planned_queries=len(obs),valid_queries=len(known),correct=sum(known),accuracy=mean(known) if known else None))
    return dict(schema='p4-paired-analysis-v1',kind=kind,model=plan['model'],execution_version=plan['execution_version'],
        planned_rows=len(expected),observed_rows=len(indexed),missing_rows=len(expected)-len(indexed),protocol_status=dict(status),
        effects=effects,paired_selectivity=selectivity,paired_contrasts=contrasts,
        summary=summary(measures,fields+['group','measure']),contrast_summary=summary(contrasts,fields+['contrast','left','right']),
        native_capability=native,
        inference='Template-level complete paired edits; descriptive normal 95% CI, no simultaneous inference; zero-containing CI is not equivalence',
        interpretation='R is target-minus-control directed response change. Raw nonadditivity is an output-scale property of native-entry k1 composition, not proof of neural synergy or a unique mechanism')


def slim(row):
    """Drop per-row delivery payloads analyze() never reads (memory only; outputs unchanged).

    Main shards carry full native-site observations per row (~10-20 KB); analyze()
    uses only identity, outcome, protocol status and the dose binding fields.
    """
    row.pop('visual_delivery',None)
    dose=row.get('dose_metadata')
    if isinstance(dose,dict):
        row['dose_metadata']={k:dose[k] for k in ('effective_k','matching','selected_entries','calibration_id') if k in dose}
    return row


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',nargs='+',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(argv)
    plans=[json.loads((root/'metadata.json').read_text()) for root in a.runs]
    first=plans[0]; queries=[]
    for plan in plans:
        for field in ('schema','execution_version','config','conditions','readouts','manifest_sha256','generate_tokens'):
            if plan[field]!=first[field]: raise ValueError('Incompatible shards: '+field)
        if (plan['model']['key'],plan['model']['revision'])!=(first['model']['key'],first['model']['revision']): raise ValueError('Different model checkpoint')
        queries.extend(plan['queries'])
    if len({q['id'] for q in queries})!=len(queries): raise ValueError('Overlapping shard queries; choose one execution per complete template')
    merged={**first,'queries':queries}
    a.output.mkdir(parents=True,exist_ok=False)
    for kind,file in [('score','results.jsonl'),('generation','generations.jsonl')]:
        if kind=='generation' and not first['generate_tokens']: continue
        def records():
            for root in a.runs:
                path=root/file
                if path.exists():
                    with path.open() as f:
                        for line in f:
                            if line.strip(): yield slim(json.loads(line))
        result=analyze(records(),merged,kind)
        write_json(a.output/(kind+'-analysis.json'),result)
        print(json.dumps(dict(kind=kind,planned=result['planned_rows'],observed=result['observed_rows'],missing=result['missing_rows'])))
    return 0


if __name__=='__main__': raise SystemExit(main())
