"""Analyze one model/arm/path group, retaining planned and observed coverage."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from cloud_prep.measurement_v8.runtime import sha256, write_json


def load_runs(paths, include_partial=False, expected_plan=None):
    records, generations, metas, provenance = [], [], [], []
    for run in paths:
        meta = json.loads((run/'metadata.json').read_text())
        if not meta.get('finished_at') and not include_partial:
            raise ValueError('Select completed runs, or explicitly use --include-partial')
        for name,target,kind in (('results.jsonl',records,'score'),('generations.jsonl',generations,'generation')):
            path = run/name
            members = [json.loads(s) for s in path.read_text().splitlines() if s.strip()]
            if not include_partial and len(members) != meta['expected_'+kind+'_rows']:
                raise ValueError('Output row count differs from plan: '+str(run))
            target.extend(members); provenance.append({'path':str(path),'sha256':sha256(path)})
        metas.append(meta)
    identity = [(m['model']['model'],m['model']['revision'],m['arm'],m['entry_groups'],
                 m['manifest_sha256'],m['design'],m['readouts'],m['panel'],m.get('parser'),
                 m.get('image_processor'),m.get('execution_schema')) for m in metas]
    if len({json.dumps(i,sort_keys=True) for i in identity}) != 1:
        raise ValueError('Do not pool checkpoints, path groups, data, protocols, or response formats across runs')
    ids = [x for m in metas for x in m['ids']]
    if len(ids) != len(set(ids)): raise ValueError('Overlapping query shards / multiple attempts selected')
    plan = dict(metas[0]); plan['ids'] = ids; plan['items'] = len(ids)
    if 'queries' in plan:
        plan['queries'] = [q for m in metas for q in m['queries']]
    for kind in ('score','generation'):
        plan['expected_'+kind+'_rows'] = sum(m['expected_'+kind+'_rows'] for m in metas)
    if expected_plan is not None:
        for field in ('arm','entry_groups','manifest_sha256','design','readouts','panel'):
            if expected_plan[field] != plan[field]:
                raise ValueError('Run differs from full campaign plan: '+field)
        if any(expected_plan['model'][k] != plan['model'][k] for k in ('model','revision')):
            raise ValueError('Checkpoint differs from full campaign plan')
        if set(ids)-set(expected_plan['ids']): raise ValueError('Runs contain unplanned query IDs')
        if not include_partial and set(ids)!=set(expected_plan['ids']):
            raise ValueError('Whole shards missing from the full campaign plan')
        for field in ('ids','queries','items','expected_score_rows','expected_generation_rows'):
            if field in expected_plan: plan[field]=expected_plan[field]
        if not include_partial and (len(records)!=plan['expected_score_rows'] or len(generations)!=plan['expected_generation_rows']):
            raise ValueError('Observed rows differ from full campaign plan')
    return records,generations,plan,provenance


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',nargs='+',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--plan',type=Path,required=True,help='Validated full campaign plan; preserves omitted-shard denominators')
    p.add_argument('--key',help='One model-arm-group key from that plan')
    p.add_argument('--include-partial',action='store_true')
    a = p.parse_args(argv)
    source_plan = json.loads(a.plan.read_text())
    if 'jobs' not in source_plan:
        expected = source_plan
    else:
        campaign = source_plan
        selected=[j for j in campaign['jobs'] if j['key']==a.key]
        if len(selected)!=1: raise ValueError('Select exactly one campaign key')
        expected=next(v['plan'] for v in campaign['validated_commands'] if v['job']==selected[0]['id'])
    records,generations,plan,provenance = load_runs(a.runs,a.include_partial,expected)
    if plan['arm'] == 'rescue':
        from cloud_prep.measurement_v9.rescue_analysis import analyze
        result = analyze(records,generations,expected_plan=plan)
    else:
        from cloud_prep.measurement_v9.multiobject_analysis import analyze
        result = analyze(records,generations,expected_plan=plan)
    result.update(model=plan['model'],arm=plan['arm'],entry_groups=plan['entry_groups'],sources=provenance,
        campaign_plan_sha256=sha256(a.plan),campaign_key=a.key,
        planned_score_rows=plan['expected_score_rows'],planned_generation_rows=plan['expected_generation_rows'],
        observed_score_rows=len(records),observed_generation_rows=len(generations),partial_allowed=a.include_partial)
    a.output.mkdir(parents=True,exist_ok=False)
    for name,value in list(result.items()):
        if isinstance(value,list) and value and isinstance(value[0],dict) and name.startswith(('query_','unit_','behavior_')):
            (a.output/(name+'.jsonl')).write_text(''.join(json.dumps(x)+'\n' for x in result.pop(name)))
    write_json(a.output/'summary.json',result)
    print(json.dumps({'output':str(a.output),'score_rows':len(records),'generation_rows':len(generations)}))


if __name__ == '__main__': main()
