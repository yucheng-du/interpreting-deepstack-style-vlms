"""Planned P1 E/R and same-target role differences, photo-paired, no inference."""
from collections import defaultdict
from itertools import product
import argparse
import json
import math
from pathlib import Path
from statistics import mean,stdev
from cloud_prep.portable_data import _guard_path
from cloud_prep.measurement_v12 import p1_design as design

BASE=('scene_id','intervention_id','response_format','mapping','main_layout','main_strength')
DOSE=('changed_entry','dose_mode','dose_metric','dose_value')
SUMMARY=('endpoint','response_format','mapping','main_layout','main_strength','query_role',
         'changed_entry','dose_mode','dose_metric','dose_value','target')


def finite(x): return type(x) in (int,float) and math.isfinite(x)


def index_records(records,expected,kind,model_key=None):
    result={}
    for row in records:
        key=(row['id'],row['response_format'],row['mapping'],row['condition'])
        if key not in expected or key in result: raise ValueError('Unplanned/duplicate P1 record')
        if model_key is not None and row.get('model_key')!=model_key:raise ValueError('Wrong model for P1 plan')
        q,r,c=expected[key]
        if row['schema']!=design.DESIGN_SCHEMA or row['condition_spec']!=c: raise ValueError('P1 schema/condition mismatch')
        if any(row[k]!=q[k] for k in design.QUERY_FIELDS): raise ValueError('P1 query/intervention mismatch')
        if (row['source_a_original_choice'],row['source_b_original_choice'])!=('(A)','(B)'): raise ValueError('Direction changed')
        if kind=='score': value=row.get('b_minus_a_log_odds');value=value if finite(value) else None
        else:
            valid=row.get('scorable') is True and not row.get('truncated',False) and row.get('original_prediction') in ('(A)','(B)')
            value=int(row['original_prediction']=='(B)') if valid else None
        unmatched=row.get('protocol_status')=='unmatched_dose'
        if unmatched:value=None
        result[key]={'value':value,'unmatched':unmatched,
                     'truncated':bool(row.get('truncated')), 'native_correct':row.get('native_correct')}
    return result


def paired_outputs(effects):
    """Return both role R values and direct paired role contrasts; no p-value gates."""
    output=list(effects);groups=defaultdict(dict)
    for e in effects:
        k=tuple(e[f] for f in BASE+DOSE)
        if e['query_role'] in groups[k]:raise ValueError('Duplicate target/control effect')
        groups[k][e['query_role']]=e
    for k,g in groups.items():
        if set(g)!={'changed','unchanged'}:raise ValueError('Missing planned dual query')
        a,b=g['changed'],g['unchanged']
        value=a['value']-b['value'] if finite(a['value']) and finite(b['value']) else None
        output.append({**a,'endpoint':'R','query_role':'changed_minus_unchanged','value':value})
    role_groups=defaultdict(dict)
    for e in output:
        key=tuple(e[f] for f in BASE)+ (e['endpoint'],e['query_role'])
        cell=tuple(e[f] for f in DOSE)
        if cell in role_groups[key]:raise ValueError('Duplicate role/dose cell')
        role_groups[key][cell]=e
    extra=[]
    for key,g in role_groups.items():
        raw_main=g[('main','raw','raw',1.)];raw_later=g[('later','raw','raw',1.)]
        comparisons=[('raw','raw',raw_main,raw_later)]
        if raw_main['main_layout']=='M0' and raw_main['main_strength']==1.:
            for metric in ('rss','l1'):
                comparisons.extend([('main_native',metric,raw_main,g[('later','baseline_state',metric,1.)]),
                                    ('later_native',metric,g[('main','baseline_state',metric,1.)],raw_later)])
        for target,metric,a,b in comparisons:
            value=a['value']-b['value'] if finite(a['value']) and finite(b['value']) else None
            extra.append({**a,'endpoint':'R_main_minus_later' if a['endpoint']=='R' else 'E_main_minus_later',
                'changed_entry':'main_minus_later','dose_mode':'same_native_target' if target!='raw' else 'raw',
                'dose_metric':metric,'dose_value':1.,'target':target,'value':value})
    return output+extra


def summarize(rows):
    grouped=defaultdict(lambda:defaultdict(list))
    for r in rows:grouped[tuple(r.get(k) for k in SUMMARY)][r['scene_id']].append(r)
    result=[]
    for key,photos in sorted(grouped.items(),key=lambda x:str(x[0])):
        if any(len(v)!=2 for v in photos.values()):raise ValueError('Each photo needs both edited objects for this endpoint')
        vals=[mean(r['value'] for r in v) for v in photos.values() if all(finite(r['value']) for r in v)]
        m=mean(vals) if vals else None;se=stdev(vals)/math.sqrt(len(vals)) if len(vals)>1 else None
        result.append({**dict(zip(SUMMARY,key)),'planned_photos':len(photos),'complete_photos':len(vals),
            'mean':m,'se':se,'ci_low':m-1.96*se if se is not None else None,
            'ci_high':m+1.96*se if se is not None else None,
            'incomplete_photos':len(photos)-len(vals)})
    return result


def analyze(records,plan,kind='score'):
    if kind not in ('score','generation'):raise ValueError('Unknown readout kind')
    if plan.get('schema')!=design.DESIGN_SCHEMA:raise ValueError('P1 execution plan required')
    if plan['conditions']!=design.conditions():raise ValueError('Unexpected P1 scientific condition version')
    expected={};pairs=defaultdict(dict)
    for q,r,c in product(plan['queries'],plan['readouts'],plan['conditions']):
        if kind=='generation' and not(q['generation_selected'] and c['generation']):continue
        key=q['id'],r['response_format'],r['mapping'],c['name'];expected[key]=q,r,c
        if c['role']=='dose_pair':pairs[key[:3]+(c['pair_id'],)][c['donor_assignment']]=key
    if len(expected)!=plan['expected_'+kind+'_rows']:raise ValueError('Plan count mismatch')
    records=list(records)
    if not plan.get('synthetic_fixture',False):
        from cloud_prep.measurement_v12.p1_dose_validation import validate_dose_records
        validate_dose_records(records,expected,kind,plan)
    indexed=index_records(records,expected,kind,plan.get('model',{}).get('key'));effects=[]
    for pair,keys in pairs.items():
        if set(keys)!={'a','b'}:raise ValueError('Incomplete planned a/b')
        q,r,c=expected[keys['a']]
        a,b=(indexed.get(keys[s],{}).get('value') for s in ('a','b'))
        effect=b-a if finite(a) and finite(b) else None
        effects.append({**{k:q[k] for k in ('scene_id','intervention_id','query_role')},**r,
            **{k:c[k] for k in ('main_layout','main_strength',*DOSE)},
            'endpoint':'E','target':'opposite_native' if c['dose_mode']!='raw' else 'raw',
            'value':effect,'a':a,'b':b})
    paired=paired_outputs(effects)
    native=[]
    for c in plan['conditions']:
        if not c['role'].startswith('native_'):continue
        for r in plan['readouts']:
            subset=[indexed.get((q['id'],r['response_format'],r['mapping'],c['name'])) for q in plan['queries']]
            observed=[x for x in subset if x and x['value'] is not None and x['native_correct'] is not None]
            native.append({'condition':c['name'],**r,'planned_queries':len(subset),
                           'labeled_observed_queries':len(observed),
                           'correct_queries':sum(x['native_correct'] is True for x in observed),
                           'scope':'query diagnostics; photo-level effects are reported separately'})
    return {'schema':design.DESIGN_SCHEMA,'kind':kind,'unit':'source_photo',
        'synthetic_fixture':bool(plan.get('synthetic_fixture',False)),
        'model':plan.get('model'),'manifest_sha256':plan.get('manifest_sha256'),
        'execution_sources':plan.get('p1_source_sha256'),
        'coverage':{'planned_rows':len(expected),'present_rows':len(indexed),'missing_rows':len(expected)-len(indexed),
                    'invalid_or_unmatched_rows':sum(v['value'] is None for v in indexed.values()),
                    'unmatched_rows':sum(v['unmatched'] for v in indexed.values()),
                    'truncated_rows':sum(v['truncated'] for v in indexed.values())},
        'effects':effects,'paired_endpoints':paired,'summaries':summarize(paired),'native_diagnostics':native,
        'interpretation':'E_target/E_control and each role R remain separate; difference CI containing zero is not equivalence; invalid never zero; complete photo requires both edits',
        'interval':'descriptive normal 95% over photo means, not simultaneous'}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--records',type=Path,nargs='+',required=True);p.add_argument('--kind',choices=('score','generation'),required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args(argv)
    def records():
        for path in a.records:
            with path.open() as f:
                for line in f:
                    if line.strip():yield json.loads(line)
    result=analyze(records(),json.loads(a.plan.read_text()),a.kind)
    out=_guard_path(a.output,write=True);out.mkdir(parents=True,exist_ok=True)
    for key in ('effects','paired_endpoints'):
        with (out/(key+'.jsonl')).open('w') as f:
            for row in result.pop(key):f.write(json.dumps(row,allow_nan=False)+'\n')
    (out/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result['coverage']))


if __name__=='__main__':main()
