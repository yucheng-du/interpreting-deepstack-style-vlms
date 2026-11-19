"""Fixed-plan paired dose effects and scene-level summaries; no tensor imports."""
from collections import Counter, defaultdict
from itertools import product
import math
from statistics import mean, stdev

from cloud_prep.measurement_v11.dose_design import DESIGN_SCHEMA, QUERY_FIELDS


def finite(x): return type(x) in (int,float) and math.isfinite(x)
def key(r): return r['id'],r['response_format'],r['mapping'],r['condition']
def valid_generation(r): return bool(r and r.get('scorable') is True and not r.get('truncated',False) and r.get('original_prediction') in ('(A)','(B)'))


def _index(rows, expected, kind, plan):
    indexed = {}
    shared = {}
    for row in rows:
        k = key(row)
        if k not in expected or k in indexed: raise ValueError('Unplanned/duplicate dose '+kind+' cell')
        q,r,c = expected[k]
        if plan.get('execution_schema') and row.get('execution_schema')!=plan['execution_schema']: raise ValueError('Wrong dose execution version')
        if plan.get('model') and row.get('model_key')!=plan['model']['key']: raise ValueError('Cross-model dose record')
        if row.get('schema')!=DESIGN_SCHEMA or row.get('condition_spec')!=c: raise ValueError('Dose schema/condition mismatch')
        for f in QUERY_FIELDS:
            if row.get(f)!=q[f]: raise ValueError('Dose query metadata differs: '+f)
        for side in ('a','b'):
            if row.get('source_'+side+'_original_choice')!=('(A)','(B)')[q['direction_'+side+'_index']]:
                raise ValueError('a/b direction must follow source color, not entry role')
        dose = row.get('dose_metadata')
        if dose is None: raise ValueError('Missing dose execution metadata')
        if dose.get('condition')!=c['name'] or dose.get('calibration_id')!=c['calibration_id'] or dose.get('donor_assignment')!=c['donor_assignment']:
            raise ValueError('Dose metadata belongs to another condition')
        if not dose.get('anchor_prompt_sha256') or not dose.get('anchor_query_id'): raise ValueError('Missing fixed calibration anchor')
        if row.get('protocol_status')=='unmatched_dose':
            if row.get('b_minus_a_log_odds') is not None or row.get('scores') not in (None,{}): raise ValueError('Unmatched dose must not carry a score')
            if dose.get('effective_k') is not None or dose.get('matching',{}).get('status')!='unmatched': raise ValueError('Invalid unmatched protocol metadata')
            if kind=='generation' and (row.get('scorable') is not False or row.get('stop_reason')!='not_run_unmatched_dose'):
                raise ValueError('Unmatched generation must be explicitly not run')
            indexed[k]=row
            continue
        scale = dose.get('effective_k')
        if not finite(scale) or scale<0: raise ValueError('Nonfinite effective dose')
        if c['donor_assignment']=='a' and scale!=0: raise ValueError('a baseline was scaled')
        if c['dose_mode']=='raw' and c['donor_assignment']=='b' and scale!=c['dose_value']: raise ValueError('Raw k changed')
        if c['dose_mode']!='raw' and dose.get('matching',{}).get('status')!='matched':
            raise ValueError('Unmatched cells cannot enter matched-dose analysis')
        sigkey = (q['intervention_id'],c['name'])
        signature = (dose['anchor_prompt_sha256'],dose['anchor_query_id'],dose.get('visual_signature'),scale)
        if sigkey in shared and shared[sigkey]!=signature: raise ValueError('Paired queries/readouts used different visual dose or anchor')
        shared[sigkey] = signature
        indexed[k] = row
    return indexed


def _summary(rows, fields, value):
    groups = defaultdict(lambda:defaultdict(list))
    for r in rows: groups[tuple(r[f] for f in fields)][r['scene_id']].append(r)
    result = []
    for group,scenes in groups.items():
        vals = [mean(r[value] for r in cells) for cells in scenes.values() if all(r[value] is not None for r in cells)]
        avg = mean(vals) if vals else None
        se = stdev(vals)/math.sqrt(len(vals)) if len(vals)>1 else None
        result.append({**dict(zip(fields,group)),'unit_key':'scene_id','planned_unit_n':len(scenes),'complete_unit_n':len(vals),
            'planned_pairs':sum(map(len,scenes.values())),'available_pairs':sum(r[value] is not None for cells in scenes.values() for r in cells),
            'mean':avg,'normal_95_interval':[avg-1.96*se,avg+1.96*se] if se is not None else None})
    return result


def analyze(records,generations=(),expected_plan=None):
    p = expected_plan
    if not p or p.get('schema')!=DESIGN_SCHEMA: raise ValueError('Dose execution plan required')
    expected,eg,pairs = {},{},defaultdict(dict)
    for q,r,c in product(p['queries'],p['readouts'],p['conditions']):
        k = q['id'],r['response_format'],r['mapping'],c['name']
        if k in expected: raise ValueError('Duplicate planned dose cell')
        expected[k] = q,r,c
        if q['generation_selected'] and c['generation']: eg[k] = expected[k]
        pk = k[:3]+(c['pair_id'],)
        if c['donor_assignment'] in pairs[pk]: raise ValueError('Duplicate side in planned dose pair')
        pairs[pk][c['donor_assignment']] = k
    if len(expected)!=p['expected_score_rows'] or len(eg)!=p['expected_generation_rows']: raise ValueError('Dose plan counts differ')
    scores,gens = _index(records,expected,'score',p),_index(generations,eg,'generation',p)
    for k in scores.keys() & gens.keys():
        if scores[k]['dose_metadata']!=gens[k]['dose_metadata']: raise ValueError('Score/generation used different dose')
    effects,behavior = [],[]
    for sides in pairs.values():
        if set(sides)!={'a','b'}: raise ValueError('Incomplete planned a/b dose pair')
        ka,kb = sides['a'],sides['b']; q,r,c = expected[ka]
        cb = expected[kb][2]
        if any(c[f]!=cb[f] for f in c if f not in ('name','donor_assignment')): raise ValueError('Dose pair changed its fixed context')
        base = {**{f:q[f] for f in ('id','scene_id','intervention_id','changed_object','query_type','query_target')},**r,
            **{f:c[f] for f in ('calibration_id','pair_id','changed_entry','changing_layout','fixed_layout','changing_strength','fixed_strength','fixed_other_assignment','region','dose_mode','dose_value')}}
        a,b = scores.get(ka),scores.get(kb)
        valid = bool(a and b and finite(a.get('b_minus_a_log_odds')) and finite(b.get('b_minus_a_log_odds')))
        effects.append({**base,'effect':b['b_minus_a_log_odds']-a['b_minus_a_log_odds'] if valid else None,
            'missing_rows':sum(x is None for x in (a,b)),
            'unmatched_rows':sum(x is not None and x.get('protocol_status')=='unmatched_dose' for x in (a,b)),
            'nonfinite_rows':sum(x is not None and x.get('protocol_status')!='unmatched_dose' and not finite(x.get('b_minus_a_log_odds')) for x in (a,b)),
            'effective_k':b['dose_metadata']['effective_k'] if b else None})
        if ka in eg:
            a,b = gens.get(ka),gens.get(kb); valid = valid_generation(a) and valid_generation(b)
            bchoice = ('(A)','(B)')[q['direction_b_index']]
            behavior.append({**base,'effect':int(b['original_prediction']==bchoice)-int(a['original_prediction']==bchoice) if valid else None,
                'flip':a['original_prediction']!=b['original_prediction'] if valid else None,
                'missing_rows':sum(x is None for x in (a,b)),
                'invalid_rows':sum(x is not None and x.get('protocol_status')!='unmatched_dose' and not valid_generation(x) for x in (a,b)),
                'unmatched_rows':sum(x is not None and x.get('protocol_status')=='unmatched_dose' for x in (a,b)),
                'truncated_rows':sum(bool(x and x.get('truncated')) for x in (a,b))})
    role_cells = defaultdict(dict)
    for e in effects:
        fields = ('id','response_format','mapping','changing_layout','fixed_layout','changing_strength','fixed_strength','fixed_other_assignment','region','dose_mode','dose_value')
        role_cells[tuple(e[f] for f in fields)][e['changed_entry']] = e
    contrasts = []
    for cells in role_cells.values():
        if set(cells)=={'main','later'}:
            a,b = cells['main'],cells['later']
            contrasts.append({**a,'contrast':'main_minus_later','changed_entry':'main_minus_later',
                'effect':a['effect']-b['effect'] if a['effect'] is not None and b['effect'] is not None else None})
    query_cells=defaultdict(dict)
    for e in effects:
        group=(e['intervention_id'],e['response_format'],e['mapping'],e['pair_id'],e['query_type'])
        query_cells[group][e['query_target']]=e
    query_contrasts=[]
    for cells in query_cells.values():
        if set(cells)=={'changed','unchanged'}:
            a,b=cells['changed'],cells['unchanged']
            query_contrasts.append({**a,'contrast':'changed_minus_unchanged','query_target':'changed_minus_unchanged',
                'effect':a['effect']-b['effect'] if a['effect'] is not None and b['effect'] is not None else None})
    fields = ('changed_entry','query_type','query_target','response_format','mapping','changing_layout','fixed_layout',
              'changing_strength','fixed_strength','fixed_other_assignment','region','dose_mode','dose_value')
    return {'schema':DESIGN_SCHEMA,'coverage':{'planned_score_rows':len(expected),'present_score_rows':len(scores),
        'missing_score_rows':len(expected)-len(scores),'nonfinite_score_rows':sum(s.get('protocol_status')!='unmatched_dose' and not finite(s.get('b_minus_a_log_odds')) for s in scores.values()),
        'unmatched_score_rows':sum(s.get('protocol_status')=='unmatched_dose' for s in scores.values()),
        'planned_generation_rows':len(eg),'present_generation_rows':len(gens),'missing_generation_rows':len(eg)-len(gens),
        'invalid_generation_rows':sum(g.get('protocol_status')!='unmatched_dose' and not valid_generation(g) for g in gens.values()),
        'unmatched_generation_rows':sum(g.get('protocol_status')=='unmatched_dose' for g in gens.values()),
        'truncated_generation_rows':sum(bool(g.get('truncated')) for g in gens.values()),
        'planned_pairs':len(effects),'complete_pairs':sum(r['effect'] is not None for r in effects),
        'planned_generation_pairs':len(behavior),'complete_generation_pairs':sum(r['effect'] is not None for r in behavior),
        'source_scenes':len({q['scene_id'] for q in p['queries']})},
        'unmatched_reasons':dict(Counter(s['dose_metadata']['matching'].get('reason','unspecified') for s in scores.values() if s.get('protocol_status')=='unmatched_dose')),
        'effects':effects,'behavior':behavior,'role_contrasts':contrasts,'query_contrasts':query_contrasts,
        'cell_summaries':_summary(effects,fields,'effect'),'behavior_summaries':_summary(behavior,fields,'effect'),
        'role_contrast_summaries':_summary(contrasts,fields,'effect'),
        'query_contrast_summaries':_summary(query_contrasts,fields,'effect'),
        'interpretation':'Raw and baseline-state-normalized operational effects; dose response does not uniquely identify a scale-only mechanism'}
