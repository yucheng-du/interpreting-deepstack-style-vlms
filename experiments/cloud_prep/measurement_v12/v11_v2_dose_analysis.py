# V12 vendor copy of accepted archived v11 opposite-native-v2; upstream algorithm unchanged.
"""Fixed-plan paired dose effects and scene-level summaries; no tensor imports."""
from collections import Counter, defaultdict
from itertools import product
import math
import json
from statistics import mean, stdev

from cloud_prep.measurement_v12.v11_v2_dose_design import DESIGN_SCHEMA, QUERY_FIELDS, opposite_condition, anchor_query_sha256
from cloud_prep.measurement_v12.v11_v2_dose_adapter import native_dose, calibration_binding, resolve_scale


def finite(x): return type(x) in (int,float) and math.isfinite(x)
def key(r): return r['id'],r['response_format'],r['mapping'],r['condition']
def valid_generation(r): return bool(r and r.get('scorable') is True and not r.get('truncated',False) and r.get('original_prediction') in ('(A)','(B)'))


def _index(rows, expected, kind, plan):
    indexed = {}
    shared = {}
    anchor_bindings = {}
    for row in rows:
        k = key(row)
        if k not in expected or k in indexed: raise ValueError('Unplanned/duplicate dose '+kind+' cell')
        q,r,c = expected[k]
        if plan.get('execution_schema') and row.get('execution_schema')!=plan['execution_schema']: raise ValueError('Wrong dose execution version')
        if plan.get('model') and row.get('model_key')!=plan['model']['key']: raise ValueError('Cross-model dose record')
        if row.get('schema')!=DESIGN_SCHEMA or row.get('condition_spec')!=c: raise ValueError('Dose schema/condition mismatch')
        for f in QUERY_FIELDS:
            if row.get(f)!=q[f]: raise ValueError('Dose query metadata differs: '+f)
        if q.get('query_sha256') and row.get('query_sha256')!=q['query_sha256']:
            raise ValueError('Dose query language hash differs from plan')
        for side in ('a','b'):
            if row.get('source_'+side+'_original_choice')!=('(A)','(B)')[q['direction_'+side+'_index']]:
                raise ValueError('a/b direction must follow source color, not entry role')
        dose = row.get('dose_metadata')
        if dose is None: raise ValueError('Missing dose execution metadata')
        if dose.get('condition')!=c['name'] or dose.get('calibration_id')!=c['calibration_id'] or dose.get('donor_assignment')!=c['donor_assignment']:
            raise ValueError('Dose metadata belongs to another condition')
        if not dose.get('anchor_prompt_sha256') or not dose.get('anchor_query_id'): raise ValueError('Missing fixed calibration anchor')
        if dose.get('target_policy', 'fixed') != c.get('target_policy','fixed') or dose.get('dose_metric','rss') != c.get('dose_metric','rss'):
            raise ValueError('Dose policy/metric differs from plan')
        if c.get('target_policy') == 'opposite_native':
            own = row.get('dose_calibration')
            other = (own or {}).get('opposite_calibration')
            if not own or not other: raise ValueError('Missing paired canonical calibration')
            anchor_id = plan['calibration_anchors'][q['intervention_id']]
            anchor_q = next(x for x in plan['queries'] if x['id']==anchor_id)
            for cal, expected_c, prefix in ((own,c,'source'),(other,opposite_condition(c),'target')):
                if (cal.get('calibration_id'),cal.get('intervention_id'),cal.get('changed_entry'),
                    cal.get('anchor_query_id'),cal.get('anchor_query_sha256')) != (
                        expected_c['calibration_id'],q['intervention_id'],expected_c['changed_entry'],
                        anchor_id,anchor_query_sha256(anchor_q)):
                    raise ValueError('Wrong paired calibration anchor/context binding')
                if not cal.get('anchor_prompt_sha256') or not cal.get('visual_signature') or not cal.get('context_signature'):
                    raise ValueError('Incomplete paired calibration provenance')
                if plan.get('model') and cal.get('model_revision') != plan['model']['revision']:
                    raise ValueError('Wrong paired calibration model revision')
                if any(dose.get(f)!=v for f,v in calibration_binding(cal,prefix).items()):
                    raise ValueError('Dose paired calibration binding differs from canonical anchor')
                scalar_cal = {f:v for f,v in cal.items() if f!='opposite_calibration'}
                encoded = json.dumps(scalar_cal,sort_keys=True,allow_nan=False)
                anchor_key = q['intervention_id'],cal['calibration_id']
                if anchor_key in anchor_bindings and anchor_bindings[anchor_key]!=encoded:
                    raise ValueError('Paired queries/readouts used different calibration anchors')
                anchor_bindings[anchor_key] = encoded
            if own['context_signature']!=other['context_signature'] or own.get('model_revision')!=other.get('model_revision'):
                raise ValueError('Opposite calibration context/revision mismatch')
            if own['anchor_prompt_sha256']!=other['anchor_prompt_sha256']:
                raise ValueError('Opposite canonical prompt mismatch')
            for cal, field in ((own,'native_self_by_metric'),(other,'target_other_native_by_metric')):
                values = native_dose(cal)
                if dose.get(field)!={m:values[m] for m in ('rss','l1')}:
                    raise ValueError('Native dose differs from bound calibration entries')
            resolved = resolve_scale(c,own,other)
            if any(dose.get(f)!=resolved.get(f) for f in ('resolved_target','requested_k','target_provenance')):
                raise ValueError('Resolved opposite target or k differs from paired anchors')
            if any(dose.get('matching',{}).get(f)!=dose.get(f) for f in (
                'target_policy','dose_metric','resolved_target','requested_k','target_provenance')):
                raise ValueError('Matching metadata differs from bound dose policy/target')
            if row.get('protocol_status')!='unmatched_dose':
                expected_k = 0. if c['donor_assignment']=='a' else resolved['k']
                if dose.get('effective_k')!=expected_k: raise ValueError('Effective k differs from paired anchors')
                if c['dose_mode']!='raw':
                    actual = dose.get('realized_baseline_state_by_metric',{}).get(c['dose_metric'])
                    target = 0. if c['donor_assignment']=='a' else resolved['resolved_target']
                    if not finite(actual) or (abs(actual-target)/target if target else abs(actual))>c['match_tolerance']:
                        raise ValueError('Realized selected metric is outside matched tolerance')
        sigkey = (q['intervention_id'],c['name'])
        signature = (dose['anchor_prompt_sha256'],dose['anchor_query_id'],dose.get('visual_signature'),
            dose.get('effective_k'),dose.get('target_anchor_prompt_sha256'),dose.get('target_anchor_query_sha256'),
            dose.get('target_visual_signature'),dose.get('requested_k'),dose.get('resolved_target'))
        if sigkey in shared and shared[sigkey]!=signature: raise ValueError('Paired queries/readouts used different visual dose or anchor')
        shared[sigkey] = signature
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
            **{f:c[f] for f in ('calibration_id','pair_id','changed_entry','changing_layout','fixed_layout','changing_strength','fixed_strength','fixed_other_assignment','region','dose_mode','dose_value','target_policy','dose_metric')}}
        a,b = scores.get(ka),scores.get(kb)
        valid = bool(a and b and finite(a.get('b_minus_a_log_odds')) and finite(b.get('b_minus_a_log_odds')))
        effects.append({**base,'effect':b['b_minus_a_log_odds']-a['b_minus_a_log_odds'] if valid else None,
            'missing_rows':sum(x is None for x in (a,b)),
            'unmatched_rows':sum(x is not None and x.get('protocol_status')=='unmatched_dose' for x in (a,b)),
            'nonfinite_rows':sum(x is not None and x.get('protocol_status')!='unmatched_dose' and not finite(x.get('b_minus_a_log_odds')) for x in (a,b)),
            'effective_k':b['dose_metadata']['effective_k'] if b else None,
            'requested_k':b['dose_metadata'].get('requested_k') if b else None,
            'resolved_target':b['dose_metadata'].get('resolved_target') if b else None})
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
        fields = ('id','response_format','mapping','changing_layout','fixed_layout','changing_strength','fixed_strength','fixed_other_assignment','region','dose_mode','dose_value','target_policy','dose_metric')
        group = role_cells[tuple(e[f] for f in fields)]
        if e['changed_entry'] in group: raise ValueError('Duplicate role in dose policy/metric group')
        group[e['changed_entry']] = e
    contrasts = []
    for cells in role_cells.values():
        if set(cells)=={'main','later'}:
            a,b = cells['main'],cells['later']
            contrasts.append({**a,'contrast':'main_minus_later','changed_entry':'main_minus_later',
                'contrast_label':('exchanged-native-dose role difference'
                    if a['target_policy']=='opposite_native' and a['dose_mode']!='raw' else 'main-minus-later role difference'),
                'dose_comparison':('exchanged_native_doses'
                    if a['target_policy']=='opposite_native' and a['dose_mode']!='raw'
                    else ('shared_raw_k' if a['dose_mode']=='raw' else 'shared_fixed_target')),
                'role_resolved_targets':{'main':a['resolved_target'],'later':b['resolved_target']},
                'role_effective_k':{'main':a['effective_k'],'later':b['effective_k']},
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
              'changing_strength','fixed_strength','fixed_other_assignment','region','dose_mode','dose_value','target_policy','dose_metric')
    from cloud_prep.measurement_v12.v11_v2_reference_contrasts import analyze_reference_contrasts
    return {'schema':DESIGN_SCHEMA,
        'normalized_dose_metrics':sorted({c['dose_metric'] for c in p['conditions'] if c['dose_mode']!='raw'}),
        'raw_metric':'raw',
        'coverage':{'planned_score_rows':len(expected),'present_score_rows':len(scores),
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
        'role_contrast_summaries':_summary(contrasts,fields+('contrast_label','dose_comparison'),'effect'),
        'query_contrast_summaries':_summary(query_contrasts,fields,'effect'),
        'reference_contrasts':analyze_reference_contrasts(scores.values(),gens.values(),p),
        'interpretation':'Raw and baseline-state-normalized operational effects; dose response does not uniquely identify a scale-only mechanism'}
