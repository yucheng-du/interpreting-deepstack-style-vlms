"""Severity choice from calibration-split native ability only (CPU; JSON in, JSON out).

Per model x task x readout x kind only two cells are read: native clean (clean,clean) and
joint-degraded (L,L). Recovery/conflict contrasts and (clean,L)/(L,clean) cells never enter, so
later-entry recovery cannot steer the choice. Retained ability
    r(L) = (A_LL - chance)/(A_CC - chance),  chance = 1/len(choices) (two-choice; default 1/2,
    --chance TASK=VALUE otherwise) or 0 (free-form metric).
Entries with A_CC - chance < margin*(1 - chance) are flagged not native-capable and not used.
Within a model r is averaged over the primary readout x kind entries (default all; free-form rows
have one readout); across models the median is taken. Shared physical grid per task: one distinct
label per target (targets descending, labels mild->severe) minimizing sum |median r - target|,
ties toward milder. The capability-calibrated comparison (per model x task the label whose r is
nearest the calibrated target, plus a linear interpolation of the physical value) is a
model-specific severity and a different estimand from the shared grid.

Free-form scorer JSON (scripts/score_followup_benchmark_v15.py; tolerant reader, unrecognised parts
are skipped with a warning): {"reports": [{model, model_revision, execution_version, task,
[material_set], primary_metric, primary_policy, per_label: {label: BODY}}]}. Also read: the other
LABEL_CONTAINERS, "severity_label" (BODY = the report), or one report BODY with label-named cells
(e.g. v14 output, label 'degraded', kept but not a v15 setting). BODY = policies[policy].metrics
[metric].estimates, any level optional; estimates = {name: {mean, n|n_clusters, se,
ci95_t|ci95|ci_low+ci_high}} with the names in BENCHMARK_NAMES ({L} = label; the generic
clean__degraded/degraded__* names only inside a per-label BODY). A per-label clean__clean is that
label's own native-clean estimate (its complete clusters); a report-level one is shared.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import itertools
import json
from pathlib import Path
import statistics
import sys

try:from cloud_prep.measurement_v15 import severity
except ModuleNotFoundError:
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]));from cloud_prep.measurement_v15 import severity

SCHEMA='v15-severity-calibration-1'
# Free-form calibration metric per task (decided 2026-10-29, before any v15 run). v14 GQA strict exact is 0
# for Granite/DeepStack because of answer formatting, so GQA severity calibration reads the named
# sensitivity gqa_case_punct_normalized; evaluation keeps gqa_exact primary and every sensitivity.
# Other tasks use the scorer's primary metric. Override with --benchmark-metric TASK=METRIC.
BENCHMARK_CALIBRATION_METRICS={'gqa_attribute':'gqa_case_punct_normalized','gqa_spatial':'gqa_case_punct_normalized'}
BENCHMARK_METRIC_REASON=('GQA strict exact equals 0 for Granite/DeepStack in v14 because of answer formatting, not visual ability; '
    'severity calibration uses the explicitly named case+punct sensitivity, chosen before v15 runs. Strict exact stays the primary '
    'evaluation metric and all sensitivities are retained.')
CALIBRATED_NOTE='model-specific severity; a different estimand from the shared physical grid'
MEASURES={'accuracy_invalid_as_incorrect':'recovery_clean_reference_accuracy_invalid_as_incorrect',
    'accuracy_valid_only':'recovery_clean_reference_accuracy'}
CELL_ESTIMANDS=(*MEASURES.values(),'recovery_clean_reference_log_odds')
# Analysis estimands copied into curves (a paired native-degradation estimand, if added, maps to realized_native_degradation).
RECOVERY={'recovery_clean_later_benefit_with_degraded_main':'later_recovery','recovery_quality_interaction':'recovery_interaction'}
CONFLICT={'conflict_query_effect':'conflict_{role}_effect','conflict_query_interaction':'conflict_{role}_interaction',
    'conflict_control_effect_toward_clean_reference':'conflict_control_effect_toward_reference',
    'conflict_control_toward_reference_interaction':'conflict_control_interaction_toward_reference',
    'selectivity_R':'selectivity_R','selectivity_interaction':'delta_R_vs_normal'}
ROLES={'changed':'target','unchanged':'control','control':'control'}
BENCHMARK_NAMES=dict(native_clean=('clean__clean','native_clean'),joint_degraded=('{L}__{L}','degraded__degraded','joint_degraded'),
    main_degraded_later_clean=('{L}__clean','degraded__clean'),main_clean_later_degraded=('clean__{L}','clean__degraded'),
    realized_native_degradation=('joint_degradation_loss','realized_native_degradation','realized_degradation'),
    later_recovery=('later_recovery_with_main_degraded','later_recovery'),
    recovery_interaction=('main_quality_by_later_quality_interaction','recovery_interaction'))
LABEL_CONTAINERS=('per_label','severity_labels','labels','by_label','severities')
COPIED=('level','planned','used','planned_units','excluded','se_zero','n_lt_5')
GROUP=('source','model','model_revision','execution_version','material_set','task','response_format','mapping','kind','measure')


def copied(e):
    """Mean, n, se, interval and denominators exactly as the input gives them."""
    ci=interval=None
    for key in ('ci95_t','ci95'):
        if e.get(key):ci,interval=list(e[key]),key;break
    if ci is None and e.get('ci_low') is not None and e.get('ci_high') is not None:ci,interval=[e['ci_low'],e['ci_high']],'ci_low/ci_high'
    n=next((e[k] for k in ('n','n_clusters','n_units') if e.get(k) is not None),None)
    return dict(n=n,mean=e.get('mean'),se=e.get('se'),ci95=ci,interval=interval,**{k:e[k] for k in COPIED if k in e})


def analysis_records(doc,path):
    """Long records from analysis.py output (estimand_table); free-form groups belong to the benchmark scorer."""
    if not isinstance(doc,dict) or not isinstance(doc.get('results'),list):raise ValueError(f'{path}: not an analysis JSON (no results list)')
    for g in doc['results']:
        if g.get('response_format')=='freeform':continue
        if not isinstance(g.get('estimand_table'),list):raise ValueError(f"{path}: group {g.get('model')}/{g.get('task')} has no estimand_table")
        refs=g.get('reference_labels') or {};clean=refs.get('recovery_clean_quality') or 'clean';normal=refs.get('conflict_reference_main_quality') or 'normal'
        base=dict(source='analysis',input=str(path),model=g['model'],model_revision=g.get('revision'),execution_version=g.get('execution_version'),
            material_set=str(g.get('material_set',doc.get('material_set','unspecified'))),split=g.get('split',doc.get('split')),task=g['task'],
            response_format=g['response_format'],mapping=g['mapping'],kind=g['kind'],primary_measure=None)
        metric='b_minus_a_log_odds' if g['kind']=='score' else 'answer_b_indicator'
        for e in g['estimand_table']:
            name,m,l=e['estimand'],e.get('main_quality'),e.get('later_quality')
            if name in CELL_ESTIMANDS:
                if m==l==clean:q,label='native_clean',None
                elif m==l:q,label='joint_degraded',m
                elif l==clean:q,label='main_degraded_later_clean',m
                elif m==clean:q,label='main_clean_later_degraded',l
                else:continue
            elif name in RECOVERY:q,label=RECOVERY[name],m
            elif name in CONFLICT:q,label=CONFLICT[name].format(role=ROLES.get(e.get('query_role'),'query')),(None if m==normal else m)
            else:continue
            yield dict(base,measure=e.get('measure') or metric,quantity=q,quality=label or m,severity_label=label,
                query_role=e.get('query_role'),source_estimand=name,**copied(e))


def estimate_sets(body,report):
    """(metric, policy, estimates) of a v14-like body: policies -> metrics -> estimates, any level optional."""
    policies=body['policies'] if isinstance(body.get('policies'),dict) else {None:body}
    for policy,pb in policies.items():
        if not isinstance(pb,dict):continue
        metrics=pb['metrics'] if isinstance(pb.get('metrics'),dict) else {pb.get('metric',report.get('primary_metric','metric')):pb}
        for metric,mb in metrics.items():
            if isinstance(mb,dict) and isinstance(mb.get('estimates'),dict):yield metric,policy,mb['estimates']


def named_estimates(est,label):
    """(quantity, label, name, entry). label None: one body with label-named X__X cells; its unlabelled
    contrasts are attributed only when it holds a single label, and its clean cell is shared."""
    labels=[label] if label is not None else sorted({a for a,_,b in (n.partition('__') for n in est) if a==b and a!='clean'})
    for q,names in BENCHMARK_NAMES.items():
        per_label=any('{L}' in n for n in names)
        targets=[label] if q=='native_clean' else labels if (per_label or label is not None or len(labels)==1) else []
        for L in targets:
            usable=[n.format(L=L) for n in names if label is not None or '{L}' in n or not per_label]
            name=next((n for n in usable if isinstance(est.get(n),dict)),None)
            if name:yield q,L,name,est[name]


def benchmark_records(doc,path,warn):
    reports=doc.get('reports') if isinstance(doc,dict) else None
    if not isinstance(reports,list):warn(f'{path}: no "reports" list; not read as a free-form scorer output');return
    for i,rep in enumerate(reports):
        if not isinstance(rep,dict) or not rep.get('model') or not rep.get('task'):warn(f'{path}: report {i} without model/task skipped');continue
        bodies=[(label,body) for key in LABEL_CONTAINERS if isinstance(rep.get(key),dict) for label,body in rep[key].items() if isinstance(body,dict)]
        bodies.append((rep.get('severity_label') or None,rep))  # the report body: one label, or label-named cells / a shared clean cell
        base=dict(source='benchmark',input=str(path),model=rep['model'],model_revision=rep.get('model_revision',rep.get('revision')),
            execution_version=rep.get('execution_version'),material_set=str(rep.get('material_set',doc.get('material_set','unspecified'))),
            split=rep.get('split',doc.get('split')),task=rep['task'],response_format='freeform',mapping='none',kind='generation',
            primary_policy=rep.get('primary_policy'))
        found=0
        for label,body in bodies:
            for metric,policy,est in estimate_sets(body,rep):
                primary=rep.get('primary_metric') in (None,metric) and (policy is None or rep.get('primary_policy') in (None,policy))
                for q,L,name,e in named_estimates(est,label):
                    found+=1
                    yield dict(base,measure=metric if policy is None else f'{metric}|{policy}',metric=metric,policy=policy,primary_measure=primary,quantity=q,
                        quality=L or 'clean',severity_label=L,query_role=None,source_estimand=name,**copied(e))
        if not found:warn(f"{path}: report {i} ({rep['model']}/{rep['task']}): no recognised estimates")


def read_inputs(analysis=(),benchmark=(),warn=None):
    """Normalised long records plus input provenance (path, sha256)."""
    warn=warn or (lambda m:print('WARNING: '+m,file=sys.stderr));records=[];inputs=[]
    for kind,paths in (('analysis',analysis),('benchmark',benchmark)):
        for path in paths:
            data=Path(path).read_bytes();doc=json.loads(data)
            got=list(analysis_records(doc,path) if kind=='analysis' else benchmark_records(doc,path,warn))
            inputs.append(dict(kind=kind,path=str(Path(path).resolve()),sha256=hashlib.sha256(data).hexdigest(),records=len(got)));records+=got
    return records,inputs


def label_settings(labels):
    """{label: dict(family, reference_kind, value, rank)}; rank 1 = mildest within its family/reference kind; unparsed -> None."""
    parsed={};kinds=defaultdict(list)
    for label in labels:
        try:s=severity.parse_label(label)
        except ValueError:parsed[label]=None;continue
        parsed[label]=s;kinds[(s['family'],s['reference_kind'])].append(label)
    order={label:i+1 for group in kinds.values() for i,label in enumerate(severity.ordered(group))}
    return {label:s and dict(family=s['family'],reference_kind=s['reference_kind'],value=s['value'],rank=order[label]) for label,s in parsed.items()}


def task_settings(records):
    labels=defaultdict(set)
    for r in records:
        if r['severity_label'] is not None:labels[r['task']].add(r['severity_label'])
    return {task:label_settings(found) for task,found in labels.items()}


def task_family(task,settings):
    if task in severity.DEFAULT_GRIDS:spec=severity.DEFAULT_GRIDS[task];return spec['family'],spec['reference_kind']
    kinds={(s['family'],s['reference_kind']) for s in settings.values() if s}
    return kinds.pop() if len(kinds)==1 else None


def readout_set(tokens):
    """FORMAT/MAPPING[:score|generation] tokens -> {(format, mapping, kind or None)}; None or ['all'] = every readout."""
    if not tokens or list(tokens)==['all']:return None
    found=set()
    for token in tokens:
        readout,_,kind=token.partition(':');fmt,sep,mapping=readout.partition('/')
        if not sep or kind not in ('','score','generation'):raise ValueError('Primary readout token FORMAT/MAPPING[:score|generation] expected: '+token)
        found.add((fmt,mapping,kind or None))
    return found


def in_readouts(r,chosen):
    return chosen is None or r['response_format']=='freeform' or any((f,m)==(r['response_format'],r['mapping']) and k in (None,r['kind']) for f,m,k in chosen)


def choose(candidates,targets):
    """candidates [(label, r)] mild->severe, targets descending: the ordered distinct pairing minimizing sum |r - target|.
    combinations() runs milder-first and only a strictly smaller cost replaces, so ties go to the milder labels."""
    k=min(len(candidates),len(targets));best=None
    for ci in itertools.combinations(range(len(candidates)),k):
        for ti in itertools.combinations(range(len(targets)),k):
            cost=sum(abs(candidates[i][1]-targets[j]) for i,j in zip(ci,ti))
            if best is None or cost<best[0]-1e-12:best=(cost,ci,ti)
    return best


def rises(points,tolerance):
    """Adjacent mild->severe increases of r above the tolerance."""
    return [(a,b,rb-ra) for (a,ra),(b,rb) in zip(points,points[1:]) if rb-ra>tolerance]


def crossing(points,target):
    """Linear interpolation of the physical value at the first crossing of target from the mild side; None if not bracketed."""
    for (v0,r0),(v1,r1) in zip(points,points[1:]):
        if r0==target:return v0
        if (r0-target)*(r1-target)<0:return v0+(target-r0)*(v1-v0)/(r1-r0)
    return points[-1][0] if points and points[-1][1]==target else None


def calibrate(records,*,targets=(.75,.5,.25),calibrated_target=.5,primary=None,measure='accuracy_invalid_as_incorrect',chance=None,
              min_native_margin=.2,floor=.1,ceiling=.9,monotone_tolerance=.05,execution_versions=None,
              benchmark_metrics=None,allow_splits=()):
    if measure not in MEASURES:raise ValueError('measure must be one of '+str(sorted(MEASURES)))
    targets=sorted({float(t) for t in targets},reverse=True);chance=dict(chance or {});chosen_readouts=readout_set(primary)
    if not targets:raise ValueError('At least one target required')
    warnings=[]
    def warn(message):
        if message not in warnings:warnings.append(message)
    metrics={**BENCHMARK_CALIBRATION_METRICS,**(benchmark_metrics or {})};allow_splits=set(allow_splits)
    def wanted(r):
        if r['source']=='analysis':return r['measure']==measure
        metric=metrics.get(r['task'])
        if metric is None:return r['primary_measure']
        return r.get('metric')==metric and r.get('policy') in (None,r.get('primary_policy') or 'invalid_as_zero')
    entries={};offered=defaultdict(set);taken=set()
    for r in records:
        if r['quantity'] not in ('native_clean','joint_degraded'):continue  # the only cells calibration reads
        if r['source']=='benchmark':offered[r['task']].add(r.get('metric'))
        if not wanted(r):continue
        taken.add(r['task'])
        if execution_versions is not None and r['execution_version'] not in execution_versions:continue
        split=r.get('split')
        if split is None:warn(f"{r['input']}: no split tag; cannot confirm calibration units")
        elif split!='calibration' and split not in allow_splits:
            raise ValueError(f"{r['input']}: summary tagged split {split!r} ({r['material_set']}), not calibration units; refusing to "
                f"calibrate on it (pass --allow-split {split} to record a deliberate exception)")
        e=entries.setdefault(tuple(r[k] for k in GROUP),dict(record=r,clean={},degraded={}))
        slot=e['clean'] if r['quantity']=='native_clean' else e['degraded']
        if r['severity_label'] in slot:raise ValueError(f"Duplicate {r['quantity']} cell for {tuple(r[k] for k in GROUP)} label {r['severity_label']}")
        slot[r['severity_label']]=r
    for task in sorted(set(offered)-taken):
        if task in metrics:warn(f'{task}: calibration metric {metrics[task]} absent from the free-form inputs (found {sorted(map(str,offered[task]))}); '
            f'not falling back silently; pass --benchmark-metric {task}=METRIC to choose another')
    if not entries:warn(f'no (clean,clean)/(L,L) cells for measure {measure} (or a primary free-form metric) in the inputs')
    seen={}
    for key in entries:
        reduced=(key[1],key[4],key[5],key[6],key[7],key[8])
        if seen.setdefault(reduced,key)!=key:raise ValueError(f'Two calibration inputs for {reduced}: {seen[reduced]} and {key}; pass one revision/execution version')
    labels=defaultdict(set)
    for e in entries.values():labels[e['record']['task']].update(e['degraded'])
    settings={task:label_settings(found) for task,found in labels.items()};families={task:task_family(task,s) for task,s in settings.items()}
    table=[]
    for key,e in sorted(entries.items(),key=lambda kv:tuple(map(str,kv[0]))):
        r0=e['record'];task=r0['task'];p=float(chance.get(task,0. if r0['response_format']=='freeform' else .5))
        for label,ll in e['degraded'].items():
            cc=e['clean'].get(label) or e['clean'].get(None);s=settings[task][label]
            a_cc,a_ll=cc and cc['mean'],ll['mean'];margin=None if a_cc is None else a_cc-p
            capable=margin is not None and margin>=min_native_margin*(1-p)
            r=(a_ll-p)/margin if a_ll is not None and margin is not None and margin>0 else None
            flags=[f for f,on in (('clean_missing',a_cc is None),('degraded_missing',a_ll is None),('not_native_capable',not capable),
                ('native_ci_includes_chance',bool(cc and cc['ci95'] and cc['ci95'][0]<=p)),('ceiling',r is not None and r>=ceiling),
                ('floor',r is not None and r<=floor),('below_chance',r is not None and r<0),('above_clean',r is not None and r>1),
                ('unparsed_label',s is None),('outside_task_family',s is not None and families[task] is not None and (s['family'],s['reference_kind'])!=families[task])) if on]
            is_primary=in_readouts(r0,chosen_readouts)
            table.append(dict({k:r0[k] for k in GROUP},label=label,family=s and s['family'],reference_kind=s and s['reference_kind'],
                value=s and s['value'],rank=s and s['rank'],chance=p,A_CC=a_cc,A_CC_n=cc and cc['n'],A_CC_ci95=cc and cc['ci95'],
                A_LL=a_ll,A_LL_n=ll['n'],A_LL_ci95=ll['ci95'],native_margin=margin,r=r,flags=flags,primary=is_primary,
                eligible=is_primary and capable and r is not None and not {'unparsed_label','outside_task_family'}&set(flags),input=r0['input']))
    table.sort(key=lambda x:(x['task'],x['model'],x['material_set'],x['response_format'],x['mapping'],x['kind'],x['rank'] or 1e9,x['label']))
    series=defaultdict(list)
    for x in table:
        if 'not_native_capable' not in x['flags'] and x['r'] is not None and x['rank'] is not None and 'outside_task_family' not in x['flags']:
            series[tuple(x[k] for k in GROUP)].append((x['label'],x['r']))
    for key,points in series.items():
        for a,b,rise in rises(points,monotone_tolerance):warn(f'non-monotone curve {"/".join(map(str,key))}: r rises {rise:.3f} from {a} to {b}')
    per_model=defaultdict(list);models=defaultdict(set)
    for x in table:
        models[x['task']].add(x['model'])
        if x['eligible']:per_model[(x['task'],x['model'],x['label'])].append(x['r'])
    curves=[];medians=[];grid={};calibrated=[]
    for task in sorted(settings):
        fam=families[task]
        if fam is None:warn(f'{task}: labels of several families and no default grid; no curves or shared grid');continue
        in_family=sorted((L for L,s in settings[task].items() if s and (s['family'],s['reference_kind'])==fam),key=lambda L:settings[task][L]['rank'])
        usable=[]
        for model in sorted(models[task]):
            points=[dict(label=L,value=settings[task][L]['value'],rank=settings[task][L]['rank'],
                r_mean=statistics.mean(per_model[(task,model,L)]) if per_model[(task,model,L)] else None,entries=len(per_model[(task,model,L)])) for L in in_family]
            curves.append(dict(task=task,model=model,points=points))
            defined=[x for x in points if x['r_mean'] is not None]
            if not defined:warn(f'{task}/{model}: no native-capable primary entry; excluded from the median');continue
            usable.append(model)
            for a,b,rise in rises([(x['label'],x['r_mean']) for x in defined],monotone_tolerance):warn(f'non-monotone model curve {task}/{model}: mean r rises {rise:.3f} from {a} to {b}')
            best=min(defined,key=lambda x:(round(abs(x['r_mean']-calibrated_target),12),x['rank']))
            calibrated.append(dict(task=task,model=model,label=best['label'],value=best['value'],rank=best['rank'],r_mean=best['r_mean'],
                target=calibrated_target,interpolated_value=crossing([(x['value'],x['r_mean']) for x in defined],calibrated_target),
                interpolation='linear in the physical value between adjacent tested settings, first crossing from the mild side; an untested setting',
                family=settings[task][best['label']]['family'],reference_kind=settings[task][best['label']]['reference_kind'],note=CALIBRATED_NOTE))
        cands=[]
        for L in in_family:
            values={m:statistics.mean(per_model[(task,m,L)]) for m in usable if per_model[(task,m,L)]}
            if not values:continue
            if len(values)<len(usable):warn(f'{task}/{L}: median over {len(values)} of {len(usable)} models')
            med=statistics.median(values.values());cands.append((L,med))
            medians.append(dict(task=task,label=L,value=settings[task][L]['value'],rank=settings[task][L]['rank'],median_r=med,models=len(values),model_r=values))
        if not cands:warn(f'{task}: no eligible label; no shared grid');continue
        for a,b,rise in rises(cands,monotone_tolerance):warn(f'non-monotone median curve {task}: median r rises {rise:.3f} from {a} to {b}')
        if len(cands)<len(targets):warn(f'{task}: {len(cands)} eligible labels for {len(targets)} targets')
        if cands[0][1]<targets[0]:warn(f'{task}: mildest candidate {cands[0][0]} median r {cands[0][1]:.3f} < mild target {targets[0]}; candidates start too severe')
        if cands[-1][1]>targets[-1]:warn(f'{task}: most severe candidate {cands[-1][0]} median r {cands[-1][1]:.3f} > severe target {targets[-1]}; extend candidates toward more severe settings')
        for side,found in (('ceiling (median r >= %g)'%ceiling,[L for L,m in cands if m>=ceiling]),('floor (median r <= %g)'%floor,[L for L,m in cands if m<=floor])):
            if found:warn(f'{task}: {len(found)} of {len(cands)} candidates at {side}: {", ".join(found)}')
        cost,ci,ti=choose(cands,targets);picked=[cands[i] for i in ci]
        grid[task]=dict(family=fam[0],reference_kind=fam[1],labels=[L for L,_ in picked],values=[settings[task][L]['value'] for L,_ in picked],
            median_r=[m for _,m in picked],targets=[targets[j] for j in ti],abs_error=[abs(m-targets[j]) for (_,m),j in zip(picked,ti)],
            total_abs_error=cost,models=usable,candidates=[L for L,_ in cands])
    rule=dict(cells_used=['native clean (clean,clean)','joint-degraded (L,L)'],ignored='every recovery/conflict contrast and every (clean,L)/(L,clean) cell',
        retained_ability='r(L) = (A_LL - chance)/(A_CC - chance)',measure=measure+' (two-choice analysis); free-form: benchmark_metrics per task, else the scorer primary metric, under the primary policy',
        benchmark_metrics=dict(per_task=metrics,default_for_other_tasks='scorer primary metric',reason=BENCHMARK_METRIC_REASON),
        splits='only calibration-tagged inputs; untagged inputs warn; others refused unless allowed',allowed_other_splits=sorted(allow_splits),
        chance=dict(default_two_choice=.5,free_form=0.,overrides=chance),native_capable=f'A_CC - chance >= {min_native_margin} x (1 - chance)',
        floor=f'r <= {floor}',ceiling=f'r >= {ceiling}',monotone_tolerance=monotone_tolerance,
        primary_readouts='all' if chosen_readouts is None else sorted(map(str,primary)),
        within_model='mean r over eligible primary readout x kind entries (and material sets)',across_models='median over models with a defined mean r',
        shared_grid='per task (its default-grid family): ordered distinct labels, one per target (targets descending, labels mild->severe), minimizing sum |median r - target|; ties toward milder',
        targets=targets,capability_calibrated=f'per model x task: label with mean r nearest {calibrated_target} (ties toward milder); '+CALIBRATED_NOTE,
        rank='order among the task labels present in these inputs (1 = mildest)',
        execution_versions=None if execution_versions is None else sorted(execution_versions))
    return dict(schema=SCHEMA,rule=rule,r_table=table,model_curves=curves,label_medians=medians,shared_grid=grid,
        severity_labels={task:g['labels'] for task,g in grid.items()},
        capability_calibrated=dict(note=CALIBRATED_NOTE,target=calibrated_target,entries=calibrated),warnings=warnings)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--analysis',type=Path,nargs='*',default=[]);p.add_argument('--benchmark',type=Path,nargs='*',default=[])
    p.add_argument('--output',type=Path,required=True);p.add_argument('--targets',type=float,nargs='+',default=[.75,.5,.25])
    p.add_argument('--calibrated-target',type=float,default=.5);p.add_argument('--measure',choices=sorted(MEASURES),default='accuracy_invalid_as_incorrect')
    p.add_argument('--primary-readouts',nargs='+',help='FORMAT/MAPPING[:score|generation] tokens (default: all readouts and kinds)')
    p.add_argument('--chance',nargs='+',default=[],help='TASK=VALUE overrides of the two-choice chance level')
    p.add_argument('--min-native-margin',type=float,default=.2);p.add_argument('--floor',type=float,default=.1);p.add_argument('--ceiling',type=float,default=.9)
    p.add_argument('--monotone-tolerance',type=float,default=.05);p.add_argument('--execution-version',nargs='+')
    p.add_argument('--benchmark-metric',nargs='+',default=[],help='TASK=METRIC free-form calibration metric (default GQA: gqa_case_punct_normalized)')
    p.add_argument('--allow-split',nargs='+',default=[],help='also accept summaries tagged with these splits (recorded; default calibration only)')
    a=p.parse_args(argv)
    if not a.analysis and not a.benchmark:p.error('pass --analysis and/or --benchmark inputs')
    notes=[];records,inputs=read_inputs(a.analysis,a.benchmark,notes.append)
    chance={}
    for item in a.chance:
        task,sep,value=item.partition('=')
        if not sep:p.error('--chance takes TASK=VALUE')
        chance[task]=float(value)
    bench={}
    for item in a.benchmark_metric:
        task,sep,value=item.partition('=')
        if not sep or not value:p.error('--benchmark-metric takes TASK=METRIC')
        bench[task]=value
    result=calibrate(records,targets=a.targets,calibrated_target=a.calibrated_target,primary=a.primary_readouts,measure=a.measure,chance=chance,
        min_native_margin=a.min_native_margin,floor=a.floor,ceiling=a.ceiling,monotone_tolerance=a.monotone_tolerance,
        execution_versions=None if a.execution_version is None else set(a.execution_version),benchmark_metrics=bench,allow_splits=a.allow_split)
    result['warnings']=notes+result['warnings']
    for w in result['warnings']:print('WARNING: '+w,file=sys.stderr)
    result['inputs']=inputs;result['script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(a.output),severity_labels=result['severity_labels'],rows=len(result['r_table']),warnings=len(result['warnings']))));return 0


if __name__=='__main__':raise SystemExit(main())
