"""Paired follow-up contrasts, with source photos/templates as inference units.

Conflict: directed later b-a effect at each main state, then their paired
difference. Target/control effects remain separate; R and its interaction use
only complete target/control intervention pairs. Recovery: clean-reference
accuracy, not intrinsic correctness of a conflicting internal state.
Free-form benchmark metrics belong to the corresponding official evaluator.

Analysis v2 only adds keys; old keys keep their meaning. Added: per-query clean
reference alignment, control effects signed toward the control's own clean
reference, R strata by control alignment, per-estimand denominators with
exclusion reasons, Student-t intervals, reference labels from the run config,
native clean/clean accuracy beside the effects, label-content conflict counts.
Changed row sets (listed in the output's behaviour_changes): restoration uses
its two degraded-main cells only; several queries of one role within an
intervention are averaged before R (previously the last one was kept).
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import sys

try:from cloud_prep.measurement_v14 import design
except ModuleNotFoundError:
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]));from cloud_prep.measurement_v14 import design

ANALYSIS_VERSION='v14-paired-followup-analysis-2'
# An excluded query/intervention is counted once, under the first applicable reason in this order.
REASONS=('missing','truncated','invalid','alignment_unknown','reference_not_a_or_b')
CONTROL_ROLES=('unchanged','control')
STRATA=('a_aligned','b_aligned')
RESULT_FIELDS=('b_minus_a_log_odds','answer_b_indicator','clean_reference_correct','scorable','truncated','stop_reason',
    'original_prediction','direction_a_original','direction_b_original','prediction','parse','scores','error','generated_text')
BEHAVIOUR_CHANGES=[
    'recovery_clean_later_benefit_with_degraded_main uses only the two degraded-main cells of a query (previously all four recovery cells were required); recovery_quality_interaction still requires all four.',
    'selectivity_R, selectivity_interaction and custom target_minus_control average several queries of one role within an intervention (previously the last query was kept).',
    'Inputs are read by physical lines (str.splitlines split JSON strings at U+2028/U+2029/U+0085 and crashed); an unterminated final line that does not parse (a live writer) is skipped and reported.',
    'Without --config, custom comparisons come from the run-metadata configs (previously none); a custom-only group without any comparison source raises.']


def _betacf(a,b,x):
    tiny=1e-300;c=1.;d=1-(a+b)*x/(a+1);d=1/(d if abs(d)>tiny else tiny);h=d
    for m in range(1,1000):
        for aa in (m*(b-m)*x/((a+2*m-1)*(a+2*m)),-(a+m)*(a+b+m)*x/((a+2*m)*(a+2*m+1))):
            d=1+aa*d;d=1/(d if abs(d)>tiny else tiny);c=1+aa/c;c=c if abs(c)>tiny else tiny;h*=d*c
        if abs(d*c-1)<1e-15:break
    return h


def _ibeta(a,b,x):
    if x<=0:return 0.
    if x>=1:return 1.
    front=math.exp(math.lgamma(a+b)-math.lgamma(a)-math.lgamma(b)+a*math.log(x)+b*math.log1p(-x))
    return front*_betacf(a,b,x)/a if x<(a+1)/(a+b+2) else 1-front*_betacf(b,a,1-x)/b


@lru_cache(maxsize=None)
def t_quantile(df,p=.975):
    """Student-t quantile without scipy: bisection on the exact two-sided tail I_{df/(df+t^2)}(df/2,1/2)."""
    if df<1 or not .5<p<1:raise ValueError('t quantile needs df >= 1 and .5 < p < 1')
    tail=2*(1-p);tails=lambda t:_ibeta(df/2,.5,df/(df+t*t));lo,hi=0.,2.
    while tails(hi)>tail:lo,hi=hi,2*hi
    for _ in range(200):
        mid=(lo+hi)/2
        if tails(mid)>tail:lo=mid
        else:hi=mid
        if hi-lo<=1e-13*hi:break
    return (lo+hi)/2


def summary(values):
    values=list(values);n=len(values);mean=statistics.mean(values) if n else None
    se=statistics.stdev(values)/math.sqrt(n) if n>1 else None;t=t_quantile(n-1) if n>1 else None
    return dict(n=n,mean=mean,se=se,ci95=[mean-1.96*se,mean+1.96*se] if se is not None else None,
        positive=sum(x>0 for x in values),zero=sum(x==0 for x in values),negative=sum(x<0 for x in values),
        ci95_t=[mean-t*se,mean+t*se] if se else None,t_quantile_975=t,se_zero=se==0,n_lt_5=n<5)


TABLE_KEYS=('measure','main_quality','later_quality','query_role','control_alignment','comparison')
TABLE_SUMMARY=('n','mean','se','ci95_t','se_zero','n_lt_5')


def first(*reasons):
    reasons=[r for r in reasons if r]
    return min(reasons,key=REASONS.index) if reasons else None


def block(items,level='queries'):
    """items: (unit, value, exclusion reason) for every planned query or intervention of one estimand."""
    used=defaultdict(list);excluded=dict.fromkeys(REASONS,0)
    for unit,value,reason in items:
        if reason is None:used[unit].append(value)
        else:excluded[reason]+=1
    vals={u:statistics.mean(v) for u,v in used.items()}
    return {**summary(vals.values()),'unit_values':vals,'denominators':{'planned_'+level:len(items),level+'_used':sum(map(len,used.values())),
        'planned_units':len({u for u,_,_ in items}),'units_used':len(vals),'excluded':excluded}}


def cell_of(c):
    if c.get('family')=='conflict':return ('conflict',c['main_quality'],c['later_endpoint'])
    if c.get('family')=='recovery':return ('recovery',c['main_quality'],c['later_quality'])
    return ('custom',c['name'])


def applicable(row,cfg,kind,fmt,mapping):
    if (kind=='generation' and not cfg.get('generate_tokens')) or (kind=='score' and fmt=='freeform'):return False
    return {'response_format':fmt,'mapping':mapping} in design.readouts(row,cfg)


def reference_labels(configs,fallback,rows,key):
    """Reference main (conflict) and clean (recovery) labels: the quality whose source is the fixed later base."""
    sources=configs or ([fallback] if fallback else [])
    if sources:
        found=set()
        for c in sources:
            pick=lambda m:(lambda k:k[0] if len(k)==1 else None)([q for q,s in m.items() if s==c['later_base']])
            found.add((pick(c['main_sources']),pick(c['recovery_sources'])))
        if len(found)>1:raise ValueError(f'Run configs disagree on reference quality labels for {key}')
        (conflict,recovery),source=found.pop(),'run_config' if configs else 'config_argument'
    else:conflict,recovery,source='normal','clean','default (no run config)'
    mains={r['condition']['main_quality'] for r in rows if r['condition'].get('family')=='conflict'}
    rmain={r['condition']['main_quality'] for r in rows if r['condition'].get('family')=='recovery'}
    rlater={r['condition']['later_quality'] for r in rows if r['condition'].get('family')=='recovery'}
    if not sources and mains and conflict is not None and conflict not in mains:
        raise ValueError(f'{key}: conflict rows lack the reference main quality {conflict!r} ({source}); found {sorted(mains)}')
    if not sources and rmain and recovery is not None and (recovery not in rmain or recovery not in rlater):
        raise ValueError(f'{key}: recovery rows lack the clean quality {recovery!r} ({source}); found {sorted(rmain|rlater)}')
    return dict(conflict_reference_main_quality=conflict,recovery_clean_quality=recovery,source=source)


def qkey(r):
    """A query, keyed like the original analysis: (unit, intervention, id, role)."""
    return (str(r['unit_id']),r['intervention_id'],r['id'],r['query_role'])


def query_alignment(material,rows):
    """Whether each query's clean reference answer is its a or b option (materials, else 2-option score rows)."""
    derived=defaultdict(set);observed={}
    for r in rows:
        k=qkey(r);observed.setdefault(k,r);res=r['result']
        if r['kind']=='score' and len(res.get('scores') or {})==2 and res.get('answer_b_indicator') is not None and res.get('clean_reference_correct') is not None:
            derived[k].add(res['answer_b_indicator']==res['clean_reference_correct'])
    for k,r in observed.items():
        m=material.get(k[2])
        if m is None:continue
        if qkey(m)!=k or m['task']!=r['task']:raise ValueError('Observation unit/intervention/role/task differs from its material row: '+k[2])
        if r.get('reference_answers') is not None and m.get('reference_answers') is not None and r['reference_answers']!=m['reference_answers']:
            raise ValueError('Reference answers differ from the material row: '+k[2])
    table={}
    for k in sorted(set(observed)|{qkey(m) for m in material.values()}):
        m=material.get(k[2]);d=derived.get(k)
        if d is not None and len(d)>1:raise ValueError('Inconsistent clean-reference alignment across observations of '+k[2])
        if m is not None and m.get('has_reference_answer','answer_index' in m) and 'direction_a_index' in m and 'direction_b_index' in m:
            a,b,source=m['answer_index']==m['direction_a_index'],m['answer_index']==m['direction_b_index'],'materials'
            if d and d!={b}:raise ValueError('Material reference alignment disagrees with observations: '+k[2])
        elif m is not None:a=b=None;source='materials (no two-choice reference/directions)'
        elif d:b=next(iter(d));a=not b;source='observations (two-option score rows)'
        else:a=b=None;source='unavailable'
        label='unknown' if a is None else 'a_aligned' if a and not b else 'b_aligned' if b and not a else 'neither' if not a else 'degenerate'
        table[k]=dict(reference_is_direction_a=a,reference_is_direction_b=b,alignment=label,source=source)
    return table


def plan_queries(key,rows,metas,material,units,cache):
    """Planned queries of one group: run plans x materials when both exist, else materials, else observations."""
    model,revision,version,task,kind,fmt,mapping=key;observed=defaultdict(dict)
    for r in rows:observed[qkey(r)][r['condition']['name']]=r['condition']
    group_conditions={n:c for cs in observed.values() for n,c in cs.items()};planned=defaultdict(dict);unattributed=set()
    if material and metas:
        source='run metadata x materials'
        for meta in metas:
            cfg=meta['config']
            for qid in meta.get('conditions_per_query',{}):
                row=material.get(qid)
                if row is None:unattributed.add(qid);continue
                if row['task']!=task or (units is not None and row['unit_id'] not in units) or not applicable(row,cfg,kind,fmt,mapping):continue
                if (meta['run_id'],qid) not in cache:cache[(meta['run_id'],qid)]={c['name']:c for c in design.conditions(row,cfg)}
                planned[qkey(row)].update(cache[(meta['run_id'],qid)])
    elif material:
        source='materials (no run metadata; conditions as observed in the group)'
        for row in material.values():
            if row['task']==task and (units is None or row['unit_id'] in units) and bool(row.get('choices'))==(fmt!='freeform'):planned[qkey(row)].update(group_conditions)
    else:source='observed queries only (conditions as observed in the group)'+('; run metadata without materials' if metas else '')
    not_planned=sum(k not in planned for k in observed) if material else 0
    for k,conds in observed.items():planned[k].update(conds if material else group_conditions)
    planned={k:dict(cells={cell_of(c) for c in conds.values()},names=set(conds)) for k,conds in planned.items()}
    info=dict(source=source,queries=len(planned),units=len({k[0] for k in planned}),interventions=len({k[:2] for k in planned}),
        observed_queries=len(observed),observed_not_planned=not_planned,planned_ids_without_material=len(unattributed),
        run_metadata=sorted(m.get('_path',m['run_id']) for m in metas))
    return planned,info


def label_content(text,label,display):
    """Content after an explicitly parsed label versus the choice displayed for that label."""
    clean=text.strip().replace('**','').replace('__','');symbol=label[1:-1]
    found=re.search(re.escape(label),clean) or re.search(r'(?<!\w)'+re.escape(symbol)+r'(?!\w)',clean)
    if not found:return 'label_not_located'
    after=clean[found.end():].casefold();mentioned=set()
    for choice in sorted(display.values(),key=len,reverse=True):
        pattern=r'(?<!\w)'+re.escape(choice.casefold())+r'(?!\w)'
        if re.search(pattern,after):mentioned.add(choice);after=re.sub(pattern,' ',after)
    own=display[label]
    if not mentioned:return 'label_only'
    return 'consistent' if mentioned=={own} else 'mixed' if own in mentioned else 'conflict'


def group_result(key,rows,ctx):
    model,revision,version,task,kind,fmt,mapping=key;warn=ctx['warn'];alignment=ctx['alignment'];material=ctx['material']
    base=dict(zip(('model','revision','execution_version','task','kind','response_format','mapping'),key))
    metas=[m for m in ctx['metas'] if (m['model'].get('key'),m['model'].get('revision'),m['config'].get('execution_version'))==(model,revision,version)]
    labels=reference_labels([m['config'] for m in metas],ctx['config'],rows,key)
    ref,clean=labels['conflict_reference_main_quality'],labels['recovery_clean_quality']
    planned,plan_info=plan_queries(key,rows,metas,material,ctx['units'],ctx['plan_cache']);ctx['planned_ids'].update(planned)
    metric='b_minus_a_log_odds' if kind=='score' else 'answer_b_indicator'
    diagnostic=defaultdict(lambda:dict(total=0,invalid=0,truncated=0));cells=defaultdict(dict);named=defaultdict(dict);dupes=set()
    for r in rows:
        c=r['condition'];d=diagnostic[c['name']];d['total']+=1
        d['invalid']+=int(r['kind']=='generation' and r['result'].get('scorable') is False)
        d['truncated']+=int(bool(r['result'].get('truncated')))
        cell=cell_of(c)
        if cell[0]!='custom':
            if cell in cells[qkey(r)]:raise ValueError(f'Ambiguous replicate/version for a {cell[0]} query')
            cells[qkey(r)][cell]=r
        if c['name'] in named[qkey(r)]:dupes.add(c['name'])
        named[qkey(r)][c['name']]=r
    def value(row,field):
        if row is None:return None,'missing'
        v=row['result'].get(field)
        return (v,None) if v is not None else (None,'truncated' if row['result'].get('truncated') else 'invalid')
    def toward(q,v,reason):
        """Signed toward the query's own clean reference: b-aligned keeps b-a, a-aligned flips it."""
        if reason:return None,reason
        label=alignment.get(q,{}).get('alignment','unknown')
        if label=='b_aligned':return v,None
        if label=='a_aligned':return -v,None
        return None,'alignment_unknown' if label=='unknown' else 'reference_not_a_or_b'
    def diff(x,y):return (x[0]-y[0],None) if x[1] is None and y[1] is None else (None,first(x[1],y[1]))
    unit=lambda q:q[0];table=[]
    def record(estimand,entry,**fields):
        d=entry['denominators'];level='queries' if 'planned_queries' in d else 'interventions'
        table.append(dict(estimand=estimand,**{k:fields.get(k) for k in TABLE_KEYS},**{k:entry[k] for k in TABLE_SUMMARY},level=level,
            planned=d['planned_'+level],used=d[level+'_used'],planned_units=d['planned_units'],excluded=d['excluded']))
        return entry
    # Conflict: endpoint pairs per query and main quality, then their paired difference.
    conflict=sorted(q for q,e in planned.items() if any(c[0]=='conflict' for c in e['cells']))
    qualities=sorted({c[1] for q in conflict for c in planned[q]['cells'] if c[0]=='conflict'})
    roles=sorted({q[3] for q in conflict});effect={}
    for q in conflict:
        for quality in qualities:
            if not {('conflict',quality,'a'),('conflict',quality,'b')}&planned[q]['cells']:continue
            have=cells.get(q,{});effect[(q,quality)]=diff(value(have.get(('conflict',quality,'b')),metric),value(have.get(('conflict',quality,'a')),metric))
    incomplete=sum(effect[(q,quality)][1] is not None for q,cs in cells.items() for quality in {c[1] for c in cs if c[0]=='conflict'})
    if ref is None and qualities:warn(f'{key}: no reference main quality in the run config; conflict interactions not computed')
    effects=[];interactions=[];toward_effects=[];toward_interactions=[]
    for quality in qualities:
        for role in roles:
            qs=[q for q in conflict if q[3]==role and (q,quality) in effect]
            if not qs:continue
            effects.append(record('conflict_query_effect',dict(main_quality=quality,query_role=role,
                **block([(unit(q),*effect[(q,quality)]) for q in qs])),main_quality=quality,query_role=role))
            if role in CONTROL_ROLES:
                toward_effects.append(record('conflict_control_effect_toward_clean_reference',dict(main_quality=quality,query_role=role,
                    **block([(unit(q),*toward(q,*effect[(q,quality)])) for q in qs])),main_quality=quality,query_role=role))
            if ref is None or quality==ref:continue
            paired=[(q,diff(effect[(q,quality)],effect[(q,ref)])) for q in qs if (q,ref) in effect]
            if not paired:continue
            interactions.append(record('conflict_query_interaction',dict(main_quality=quality,reference_main_quality=ref,query_role=role,
                **block([(unit(q),*v) for q,v in paired])),main_quality=quality,query_role=role))
            if role in CONTROL_ROLES:
                toward_interactions.append(record('conflict_control_toward_reference_interaction',dict(main_quality=quality,reference_main_quality=ref,
                    query_role=role,**block([(unit(q),*toward(q,*v)) for q,v in paired])),main_quality=quality,query_role=role))
    # Selectivity R: target minus control within an intervention; several queries of one role are averaged first.
    interventions=defaultdict(lambda:defaultdict(list))
    for q in conflict:interventions[q[:2]][q[3]].append(q)
    def role_value(qs,quality):
        got=[effect[(q,quality)] for q in qs if (q,quality) in effect];vals=[e for e,reason in got if reason is None]
        return (statistics.mean(vals),None) if vals else (None,first(*[reason for _,reason in got]) or 'missing')
    r_iids=sorted(i for i,rs in interventions.items() if 'changed' in rs and any(r in rs for r in CONTROL_ROLES));parts={}
    for i in r_iids:
        for quality in qualities:
            t=role_value(interventions[i]['changed'],quality);controls=[role_value(interventions[i][r],quality) for r in CONTROL_ROLES if r in interventions[i]]
            c=next((x for x in controls if x[1] is None),(None,first(*[x[1] for x in controls])));parts[(i,quality)]=(diff(t,c),t,c)
    def stratum(i):
        found={alignment.get(q,{}).get('alignment','unknown') for r in CONTROL_ROLES for q in interventions[i].get(r,[])}
        return found.pop() if len(found)==1 else 'mixed'
    strata={i:stratum(i) for i in r_iids};routput=[];rioutput=[];rstrata=[];ristrata=[];counts=Counter(strata.values())
    flag=('no target/control interventions' if not counts else 'all controls b-aligned' if set(counts)=={'b_aligned'} else
        'all controls a-aligned' if set(counts)=={'a_aligned'} else 'control alignment unknown' if set(counts)<={'unknown'} else
        'mixed control alignment: see selectivity_R_by_control_alignment')
    for quality in qualities:
        if not r_iids:break
        entry=record('selectivity_R',dict(main_quality=quality,**block([(i[0],*parts[(i,quality)][0]) for i in r_iids],'interventions'),
            control_alignment_flag=flag),main_quality=quality)
        if entry['n']:routput.append(entry)
        inter=ref is not None and quality!=ref and ref in qualities
        if inter:
            entry=record('selectivity_interaction',dict(main_quality=quality,**block([(i[0],*diff(parts[(i,quality)][0],parts[(i,ref)][0])) for i in r_iids],'interventions'),
                control_alignment_flag=flag),main_quality=quality)
            if entry['n']:rioutput.append(entry)
        for s in sorted(set(STRATA)|set(strata.values())):
            iids=[i for i in r_iids if strata[i]==s]
            sign=1 if s=='b_aligned' else -1 if s=='a_aligned' else None
            def component(i,index,scale=1):
                reason=parts[(i,quality)][0][1]
                if reason:return i[0],None,reason
                if scale is None:return i[0],None,'alignment_unknown'
                return i[0],scale*parts[(i,quality)][index][0],None
            rstrata.append(record('selectivity_R_by_control_alignment',dict(main_quality=quality,control_alignment=s,
                **block([(i[0],*parts[(i,quality)][0]) for i in iids],'interventions'),
                target_component=block([component(i,1) for i in iids],'interventions'),
                control_component=block([component(i,2) for i in iids],'interventions'),
                control_component_toward_clean_reference=block([component(i,2,sign) for i in iids],'interventions')),
                main_quality=quality,control_alignment=s))
            if inter:
                ristrata.append(record('selectivity_interaction_by_control_alignment',dict(main_quality=quality,reference_main_quality=ref,control_alignment=s,
                    **block([(i[0],*diff(parts[(i,quality)][0],parts[(i,ref)][0])) for i in iids],'interventions')),main_quality=quality,control_alignment=s))
    tally=lambda pick:dict(sorted(Counter(alignment.get(q,{}).get('alignment','unknown') for q in conflict if pick(q[3])).items()))
    control_alignment=dict(flag=flag,interventions_by_control_alignment=dict(sorted(counts.items())),
        control_queries_by_alignment=tally(lambda r:r in CONTROL_ROLES),target_queries_by_alignment=tally(lambda r:r=='changed'))
    # Recovery: clean-reference accuracy of the same image's clean/degraded main x later.
    recovery=sorted(q for q,e in planned.items() if any(c[0]=='recovery' for c in e['cells']))
    has_ref=lambda q:(material[q[2]].get('has_reference_answer','answer_index' in material[q[2]]) if q[2] in material
        else any('clean_reference_correct' in r['result'] for r in cells.get(q,{}).values()))
    scored=[q for q in recovery if has_ref(q)]
    rcells=sorted({(c[1],c[2]) for q in recovery for c in planned[q]['cells'] if c[0]=='recovery'})
    if clean is None and rcells:warn(f'{key}: no clean recovery quality in the run config; recovery contrasts not computed')
    degraded=sorted({m for m,_ in rcells}-{clean}) if clean is not None else []
    def measure(name,q,cell):
        v,reason=value(cells.get(q,{}).get(('recovery',*cell)),'b_minus_a_log_odds' if name=='reference_log_odds' else 'clean_reference_correct')
        if name=='accuracy_invalid_as_incorrect' and reason in ('truncated','invalid'):return 0,None
        return toward(q,v,reason) if name=='reference_log_odds' else (v,reason)
    measures=['accuracy_valid_only','accuracy_invalid_as_incorrect']+(['reference_log_odds'] if kind=='score' else [])
    names={'accuracy_valid_only':'recovery_clean_reference_accuracy','accuracy_invalid_as_incorrect':'recovery_clean_reference_accuracy_invalid_as_incorrect',
        'reference_log_odds':'recovery_clean_reference_log_odds'}
    per_cell={name:[] for name in measures};contrasts={name:dict(clean_later_benefit_with_degraded_main=[],quality_interaction=[]) for name in measures}
    for name in measures:
        for cell in rcells:
            qs=[q for q in scored if ('recovery',*cell) in planned[q]['cells']]
            per_cell[name].append(record(names[name],dict(main_quality=cell[0],later_quality=cell[1],**block([(unit(q),*measure(name,q,cell)) for q in qs])),
                measure=name,main_quality=cell[0],later_quality=cell[1]))
        for d in degraded:
            two=[q for q in scored if {('recovery',d,clean),('recovery',d,d)}<=planned[q]['cells']]
            four=[q for q in two if {('recovery',clean,clean),('recovery',clean,d)}<=planned[q]['cells']]
            weak=lambda q:diff(measure(name,q,(d,clean)),measure(name,q,(d,d)))
            contrasts[name]['clean_later_benefit_with_degraded_main'].append(record('recovery_clean_later_benefit_with_degraded_main',
                dict(main_quality=d,later_quality=clean,cells=[[d,clean],[d,d]],**block([(unit(q),*weak(q)) for q in two])),measure=name,main_quality=d,later_quality=clean))
            contrasts[name]['quality_interaction'].append(record('recovery_quality_interaction',dict(main_quality=d,reference_main_quality=clean,
                cells=[[d,clean],[d,d],[clean,clean],[clean,d]],**block([(unit(q),*diff(weak(q),diff(measure(name,q,(clean,clean)),measure(name,q,(clean,d))))) for q in four])),
                measure=name,main_quality=d,later_quality=clean))
    strip=lambda entry:{k:v for k,v in entry.items() if k not in ('main_quality','later_quality','reference_main_quality','cells')}
    empty={**summary([]),'unit_values':{},'denominators':block([])['denominators']}
    one=contrasts['accuracy_valid_only'] if len(degraded)==1 else None
    note={'note':'needs exactly one degraded recovery quality; see recovery_contrasts'} if degraded else {}
    restoration=strip(one['clean_later_benefit_with_degraded_main'][0]) if one else {**empty,**note}
    rinteraction=strip(one['quality_interaction'][0]) if one else {**empty,**note}
    native=dict(cell=[clean,clean],scope='all pre-selected (planned) queries of this model x task x kind x readout; no native-correct selection',
        status='computed' if scored else 'no two-choice clean reference (free-form: use the benchmark scorer)' if recovery else 'no recovery rows planned')
    for name in measures:
        found=[x for x in per_cell[name] if (x['main_quality'],x['later_quality'])==(clean,clean)]
        native[name]=strip(found[0]) if found else dict(empty)
    # Custom comparisons: --config, else the configs recorded in the run metadata.
    comparisons,comparison_source=(ctx['comparisons'],'config_argument') if ctx['comparisons'] is not None else (None,None)
    if comparisons is None and metas:
        defined={}
        for m in metas:
            for comp in m['config'].get('comparisons',[]):
                if defined.setdefault(comp['name'],comp)!=comp:raise ValueError('Run configs define comparison '+comp['name']+' differently')
        comparisons,comparison_source=list(defined.values()),'run_metadata'
    custom=[]
    for comparison in comparisons or []:
        pos,neg=comparison['positive'],comparison['negative']
        if dupes&{pos,neg}:raise ValueError('Ambiguous replicate/version for a custom comparison query')
        qs=sorted(q for q,e in planned.items() if pos in e['names'] and neg in e['names'])
        paired={q:diff(value(named.get(q,{}).get(pos),metric),value(named.get(q,{}).get(neg),metric)) for q in qs};details=[]
        for role in sorted({q[3] for q in qs}):
            entry=record('custom_query_effect',dict(query_role=role,**block([(unit(q),*paired[q]) for q in qs if q[3]==role])),
                query_role=role,comparison=comparison['name'])
            if entry['n']:details.append(entry)
        byiid=defaultdict(lambda:defaultdict(list))
        for q in qs:byiid[q[:2]][q[3]].append(q)
        def averaged(qs_):
            vals=[paired[q][0] for q in qs_ if paired[q][1] is None]
            return (statistics.mean(vals),None) if vals else (None,first(*[paired[q][1] for q in qs_]))
        tm=[]
        for i,rs in sorted(byiid.items()):
            if 'changed' not in rs or not any(r in rs for r in CONTROL_ROLES):continue
            controls=[averaged(rs[r]) for r in CONTROL_ROLES if r in rs]
            tm.append((i[0],*diff(averaged(rs['changed']),next((x for x in controls if x[1] is None),(None,first(*[x[1] for x in controls]))))))
        custom.append({**comparison,'query_effects':details,'target_minus_control':record('custom_target_minus_control',block(tm,'interventions'),comparison=comparison['name'])})
    present={r['condition']['name'] for r in rows}
    if not any(r['condition'].get('family') in ('conflict','recovery') for r in rows) and not any(c['positive'] in present and c['negative'] in present for c in comparisons or []):
        if comparison_source is None:raise ValueError(f'{key}: only custom conditions and no comparison; pass --config or keep the run-*.json metadata next to the observations')
        warn(f'{key}: only custom conditions and no applicable comparison ({comparison_source}); only diagnostics are reported')
    # Label readouts: content after the parsed label versus the choice displayed for that label (counts only).
    content=dict(status='not applicable (label-readout generations only)',by_condition={})
    if kind=='generation' and fmt=='label':
        if not material:content['status']='materials_unavailable'
        else:
            from cloud_prep.measurement_v6.content_readout import make_variant
            tallies=defaultdict(lambda:dict.fromkeys(('rows','parsed','label_only','consistent','conflict','mixed','label_not_located','material_missing'),0));display={}
            for r in rows:
                n=tallies[r['condition']['name']];n['rows']+=1;res=r['result'];parse=res.get('parse') or {}
                if parse.get('status')!='parsed' or not isinstance(res.get('generated_text'),str):continue
                n['parsed']+=1
                if r['id'] not in material:n['material_missing']+=1;continue
                if r['id'] not in display:
                    v=make_variant(material[r['id']],fmt,mapping);display[r['id']]=dict(zip(v['labels'],v['display_choices']))
                n[label_content(res['generated_text'],parse['label'],display[r['id']])]+=1
            content=dict(status='computed',by_condition=dict(sorted(tallies.items())),
                definition='parsed label; displayed choices mentioned after it: none=label_only, only its own=consistent, only others=conflict, both=mixed')
    geometry=Counter();active=Counter();absent=0
    for r in rows:
        g=(r.get('visual_metadata') or {}).get('geometry')
        if g is None:absent+=1;continue
        geometry[json.dumps(g,sort_keys=True)]+=1
        if isinstance(g,dict) and 'active_later_groups' in g:active[g['active_later_groups']]+=1
    visual=dict(active_later_groups=sorted(active,key=str),active_later_groups_rows={str(k):v for k,v in sorted(active.items(),key=lambda x:str(x[0]))},
        rows_with_geometry=sum(geometry.values()),rows_without_geometry=absent,distinct_geometries=len(geometry))
    by_id={r['observation_id']:r for r in rows};sourced=[r for r in rows if r.get('reused_from_observation_id')]
    compared=('b_minus_a_log_odds','answer_b_indicator','clean_reference_correct','scorable','truncated','original_prediction','generated_text')
    mismatch=sum(any(r['result'].get(f)!=by_id[r['reused_from_observation_id']]['result'].get(f) for f in compared) for r in sourced if r['reused_from_observation_id'] in by_id)
    if mismatch:warn(f'{key}: {mismatch} reused rows differ from their source observation')
    run_ids={r.get('run_id') for r in rows}
    execution=dict(rows=len(rows),run_ids=sorted(run_ids,key=str),run_ids_without_metadata=sorted(run_ids-{m['run_id'] for m in metas},key=str),
        evaluation_reused=sum(bool(r.get('evaluation_reused')) for r in rows),reused_from_observation_id=len(sourced),
        reused_source_not_in_group=sum(r['reused_from_observation_id'] not in by_id for r in sourced),reused_value_mismatch=mismatch,
        elapsed_basis=dict(Counter(str(r['elapsed_basis']) for r in rows if 'elapsed_basis' in r)),
        rows_with_batch_fields=sum(any('batch' in k for k in r) or any('batch' in k for k in (r['result'].get('delivery') or {})) for r in rows))
    return {**base,'reference_labels':labels,'planned':plan_info,'native_clean_reference_accuracy':native,
        'conflict_query_effects':[x for x in effects if x['n']],'conflict_query_interactions':[x for x in interactions if x['n']],
        'conflict_control_effect_toward_clean_reference':toward_effects,'conflict_control_toward_reference_interactions':toward_interactions,
        'control_alignment':control_alignment,'selectivity_R':routput,'selectivity_interaction':rioutput,
        'selectivity_R_by_control_alignment':rstrata,'selectivity_interaction_by_control_alignment':ristrata,'incomplete_endpoint_pairs':incomplete,
        'recovery_clean_reference_accuracy':[x for x in per_cell['accuracy_valid_only'] if x['n']],
        'recovery_clean_reference_accuracy_invalid_as_incorrect':per_cell['accuracy_invalid_as_incorrect'],
        'recovery_clean_reference_log_odds':per_cell.get('reference_log_odds',[]),
        'recovery_clean_later_benefit_with_degraded_main':restoration,'recovery_quality_interaction':rinteraction,'recovery_contrasts':contrasts,
        'custom_paired_comparisons':custom,'custom_comparison_source':comparison_source,'diagnostics':dict(diagnostic),
        'label_content_conflicts':content,'visual_geometry':visual,'execution':execution,'estimand_table':table}


def analyze(records,comparisons=None,*,run_metadata=(),materials=(),execution_versions=None,unit_ids=None,config=None):
    """records: observations. comparisons: from --config (None: the run-metadata configs). materials: design.load_rows rows."""
    warnings=[]
    def warn(message):
        if message not in warnings:warnings.append(message);print('WARNING: '+message,file=sys.stderr)
    versions=None if execution_versions is None else set(execution_versions);units=None if unit_ids is None else {str(u) for u in unit_ids}
    groups=defaultdict(list);seen=set()
    for record in records:
        if record['observation_id'] in seen:raise ValueError('Duplicate observation id across inputs')
        seen.add(record['observation_id'])
        if (versions is not None and record['execution_version'] not in versions) or (units is not None and str(record['unit_id']) not in units):continue
        key=(record['model_key'],record['model_revision'],record['execution_version'],record['task'],record['kind'],
             record['readout']['response_format'],record['readout']['mapping'])
        groups[key].append(record)
    material={}
    for row in materials:
        previous=material.setdefault(row['id'],row)
        if previous is not row and design.semantic_row(previous)!=design.semantic_row(row):raise ValueError('Conflicting material rows for query '+row['id'])
    alignment=query_alignment(material,[r for rows in groups.values() for r in rows])
    ctx=dict(warn=warn,material=material,alignment=alignment,metas=list({m['run_id']:m for m in run_metadata}.values()),units=units,
        config=config,comparisons=comparisons,plan_cache={},planned_ids=set())
    results=[group_result(key,rows,ctx) for key,rows in sorted(groups.items())]
    if not results:warn('No observations selected')
    return dict(schema='v14-paired-followup-analysis',inference_unit='source photo/template/document unit_id; queries and edits are within-unit repeats',
        intervals='descriptive normal 95%; exploratory; no multiplicity correction',
        generation_missingness='invalid endpoints omitted only from affected contrasts; per-condition invalid/truncated counts retained',
        freeform='Use task-specific evaluator; no automatic two-choice conversion',analysis_version=ANALYSIS_VERSION,
        intervals_t='ci95_t: Student-t 95% over unit values (df=n-1); null when n<2 or se=0; se_zero and n_lt_5 flag degenerate or small-n intervals',
        denominators=('planned = run plan x materials when available (see each group planned.source); every planned query/intervention is used or excluded '
            'once, under the first of '+' > '.join(REASONS)+'; estimand_table lists every planned estimand, including n=0'),
        alignment='reference_is_direction_a/b: the clean reference answer is the a/b option; toward-reference values are positive toward that reference',
        behaviour_changes=BEHAVIOUR_CHANGES,filters=dict(execution_versions=sorted(versions) if versions is not None else None,unit_ids=sorted(units) if units is not None else None),
        query_alignment=[dict(zip(('unit_id','intervention_id','id','query_role'),q),**alignment[q]) for q in sorted(ctx['planned_ids']) if q in alignment],
        warnings=warnings,results=results)


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()


def read_jsonl(path,files):
    """Physical lines only; an unterminated final line that does not parse (a live writer's tail) is skipped and reported."""
    info=dict(path=str(Path(path).resolve()),rows=0,incomplete_final_line=False,unterminated_final_line=False);h=hashlib.sha256()
    with open(path,'rb') as f:
        for number,raw in enumerate(f,1):
            h.update(raw)
            if not raw.strip():continue
            try:row=json.loads(raw.decode('utf-8'))
            except (UnicodeDecodeError,json.JSONDecodeError) as exc:
                if raw.endswith(b'\n'):raise ValueError(f'{path}:{number} is not complete JSON') from exc
                info.update(incomplete_final_line=True,incomplete_final_line_number=number,incomplete_final_line_bytes=len(raw));continue
            info['rows']+=1;info['unterminated_final_line']=not raw.endswith(b'\n');yield row
    info['sha256_of_bytes_read']=h.hexdigest();files.append(info)


def slim(r):
    """The fields the analysis reads; delivery records and token-level visual metadata are dropped while streaming."""
    res=r.get('result') or {};vm=r.get('visual_metadata') or {}
    out={k:v for k,v in r.items() if k not in ('result','visual_metadata','source_metadata','source_pixel_sha256','material_provenance')}
    result={k:res[k] for k in RESULT_FIELDS if k in res}
    if isinstance(res.get('delivery'),dict):result['delivery']={k:v for k,v in res['delivery'].items() if 'batch' in k}
    out.update(result=result,visual_metadata={'geometry':vm['geometry']} if 'geometry' in vm else {});return out


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('inputs',type=Path,nargs='+');p.add_argument('--output',type=Path,required=True)
    p.add_argument('--config',type=Path,help='Optional config defining custom paired comparisons (default: those in the run metadata)')
    p.add_argument('--materials',type=Path,nargs='+',help='Material manifests (default: each run-metadata manifest whose sha256 matches)')
    p.add_argument('--run-metadata',type=Path,nargs='+',help='run-*.json files (default: those next to each input)')
    p.add_argument('--no-discovery',action='store_true',help='Use only explicitly passed run metadata and materials')
    p.add_argument('--execution-version',nargs='+',help='Analyse only these execution versions')
    p.add_argument('--unit-ids',nargs='+',help='Analyse only these unit_ids')
    a=p.parse_args(argv);files=[];records=[];notes=[]
    for path in a.inputs:records.extend(slim(r) for r in read_jsonl(path,files))
    for f in files:
        if f['incomplete_final_line']:notes.append(f"skipped an incomplete final line of {f['path']} ({f['incomplete_final_line_bytes']} bytes)")
    paths=list(a.run_metadata or [])+([] if a.no_discovery else sorted({x for path in a.inputs for x in Path(path).resolve().parent.glob('run-*.json')}))
    metas=[]
    for path in dict.fromkeys(Path(x).resolve() for x in paths):
        try:meta=json.loads(path.read_text(encoding='utf-8'))
        except (OSError,json.JSONDecodeError) as exc:notes.append(f'unreadable run metadata {path}: {exc!r}');continue
        if not isinstance(meta,dict) or not all(k in meta for k in ('run_id','model','config')):notes.append(f'not run metadata: {path}');continue
        metas.append({**meta,'_path':str(path)})
    config=None
    if a.config:
        try:config=design.settings(a.config)
        except ValueError as exc:notes.append(f'--config is not a runner config (used for comparisons only): {exc}')
    manifests=[(Path(x).resolve(),'argument',None) for x in a.materials or []]
    if not a.no_discovery and not a.materials:manifests+=[(Path(m['manifest']),'run_metadata',m) for m in metas if m.get('manifest')]
    materials=[];material_info=[];loaded=set()
    for path,source,meta in manifests:
        if not path.exists():notes.append(f'manifest not found: {path}');continue
        digest=sha256(path)
        if meta is not None and digest!=meta.get('manifest_sha256'):notes.append(f'manifest sha256 differs from its run metadata, not used: {path}');continue
        if (path,digest) in loaded:continue
        loaded.add((path,digest));rows=design.load_rows(path,meta['config'] if meta else (config or design.settings()),check_files=False)
        materials.extend(rows);material_info.append(dict(path=str(path),sha256=digest,rows=len(rows),source=source))
    for note in notes:print('WARNING: '+note,file=sys.stderr)
    comparisons=json.loads(a.config.read_text()).get('comparisons',[]) if a.config else None
    result=analyze(records,comparisons,run_metadata=metas,materials=materials,execution_versions=a.execution_version,unit_ids=a.unit_ids,config=config)
    result['warnings']=notes+result['warnings']
    result['provenance']=dict(inputs=files,run_metadata=[dict(path=m['_path'],run_id=m['run_id'],model=m['model'].get('key'),
        execution_version=m['config'].get('execution_version'),status=m.get('status')) for m in metas],materials=material_info,
        config=str(a.config.resolve()) if a.config else None,script_sha256=sha256(__file__))
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    print(json.dumps({'output':str(a.output),'analysis_groups':len(result['results'])}));return 0


if __name__=='__main__':raise SystemExit(main())
