#!/usr/bin/env python3
"""Score v14 freeform recovery four-cells against clean-reference answers.

GQA primary: exact equality, as in the author's evaluator. Lower+strip equality
and gqa_case_punct_normalized (lowercase, strip whitespace, trailing . ! ? and
surrounding quotes) are named sensitivities, not official normalizations; per
cell, rows where the latter disagrees with strict equality are counted. DocVQA
primary follows the organizers' public SP/MP framework (lower+strip, internal
spaces retained, max-reference Levenshtein similarity >= .5); the original
strict-paper threshold is reported separately. No substring, semantic or article
matching.

Question-level four-cell contrasts are averaged within image/document first.
Normal intervals concern equally weighted clusters, not the official full-set
question-weighted leaderboard average (reported only as a separate diagnostic).
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

CELLS=('clean__clean','clean__degraded','degraded__clean','degraded__degraded')
METRIC_SOURCES={
 'gqa_exact':'https://raw.githubusercontent.com/dorarad/gqa/9f88c1118598ba459266b9d3a6b74cd3db60bc44/evaluation/eval.py',
 'docvqa_anls_organizer_framework':'https://raw.githubusercontent.com/rubenpt91/MP-DocVQA-Framework/656d4e5a316720c83a1b1679fb39f424265655d7/metrics.py',
 'docvqa_anls_strict_paper_cutoff':'https://arxiv.org/html/1907.00490',
}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def read_jsonl(path,incomplete_tail=None):
    """Physical lines. With a list for incomplete_tail (live observation files), an unterminated
    final line that does not parse is recorded there and skipped; any other bad line fails."""
    with path.open('rb') as f:
        for number,raw in enumerate(f,1):
            if raw.strip():
                try:yield json.loads(raw.decode('utf-8'))
                except (UnicodeDecodeError,json.JSONDecodeError) as exc:
                    if incomplete_tail is not None and not raw.endswith(b'\n'):
                        incomplete_tail.append(dict(path=str(path.resolve()),line=number,bytes=len(raw)));continue
                    raise ValueError(f'{path}:{number} is not complete JSON') from exc


QUOTES={('"','"'),("'","'"),('`','`'),('\u201c','\u201d'),('\u2018','\u2019')}


def gqa_case_punct(text):
    """Sensitivity normalization: lowercase, strip whitespace, trailing . ! ? and surrounding quotes."""
    text=text.lower().strip()
    while True:
        previous=text;text=text.rstrip('.!?').strip()
        if len(text)>=2 and (text[0],text[-1]) in QUOTES:text=text[1:-1].strip()
        if text==previous:return text


def edit_distance(a,b):
    if len(a)<len(b):a,b=b,a
    previous=list(range(len(b)+1))
    for i,ac in enumerate(a,1):
        current=[i]
        for j,bc in enumerate(b,1):
            current.append(min(current[-1]+1,previous[j]+1,previous[j-1]+(ac!=bc)))
        previous=current
    return previous[-1]


def metric_values(benchmark,prediction,references):
    if not isinstance(prediction,str):raise ValueError('Metric expects a string')
    if not references or not all(isinstance(x,str) and x for x in references):raise ValueError('Invalid references')
    if benchmark=='gqa':
        return dict(gqa_exact=float(prediction in references),
                    gqa_lower_strip_sensitivity=float(prediction.lower().strip() in [r.lower().strip() for r in references]),
                    gqa_case_punct_normalized=float(gqa_case_punct(prediction) in [gqa_case_punct(r) for r in references]))
    if benchmark!='docvqa':raise ValueError('Unknown benchmark '+benchmark)
    p=prediction.lower().strip();refs=[r.lower().strip() for r in references]
    # The organizers' answerable-string branch explicitly gives 'none' zero.
    if not p or p=='none':return dict(docvqa_anls_organizer_framework=0.,docvqa_anls_strict_paper_cutoff=0.,docvqa_exact_lower_strip=0.)
    similarities=[1-edit_distance(p,r)/max(len(p),len(r)) for r in refs]
    best=max(similarities)
    return dict(docvqa_anls_organizer_framework=best if best>=.5 else 0.,
                docvqa_anls_strict_paper_cutoff=best if best>.5 else 0.,
                docvqa_exact_lower_strip=float(p in refs))


def stat(values):
    values=list(values);n=len(values);m=statistics.mean(values) if n else None
    se=statistics.stdev(values)/math.sqrt(n) if n>1 else None
    return dict(n_clusters=n,mean=m,se=se,ci_low=m-1.96*se if se is not None else None,
                ci_high=m+1.96*se if se is not None else None)


def contrasts(values):
    cc,cd,dc,dd=(values[key] for key in CELLS)
    return {**values,'later_recovery_with_main_degraded':dc-dd,
            'later_effect_with_main_clean':cc-cd,
            'main_quality_by_later_quality_interaction':(dc-dd)-(cc-cd),
            'main_recovery_with_later_degraded':cd-dd,
            'joint_degradation_loss':cc-dd}


def invalid_reasons(row):
    result=row.get('result',{});text=result.get('generated_text');reasons=[]
    if not isinstance(text,str):reasons.append('missing_string')
    elif not text.strip():reasons.append('empty_string')
    if result.get('truncated') is True:reasons.append('truncated')
    if row.get('error') or result.get('error'):reasons.append('execution_error')
    if row.get('protocol_status') in ('failed','error','unmatched_dose'):reasons.append(str(row['protocol_status']))
    return reasons


def summarize_group(materials, observations):
    """One model/version/task; all cells share planned clean labels and clusters."""
    benchmark=next(iter(materials.values()))['benchmark']['name']
    if any(r['benchmark']['name']!=benchmark for r in materials.values()):raise ValueError('Mixed benchmark group')
    metrics=list(metric_values(benchmark,'',['__nonempty_reference__']))
    records_by_question=defaultdict(dict)
    diagnostics={cell:Counter() for cell in CELLS};unknown=[]
    for row in observations:
        qid=row['id'];condition=row['condition'];cell=condition['main_quality']+'__'+condition['later_quality']
        if cell not in CELLS:raise ValueError('Unexpected recovery quality '+cell)
        if qid not in materials:unknown.append(qid);continue
        if cell in records_by_question[qid]:raise ValueError('Duplicate query/cell; select one execution version or nonoverlapping files: '+qid+'/'+cell)
        target=materials[qid]
        if row.get('reference_answers')!=target['reference_answers'] or row.get('unit_id')!=target['unit_id']:
            raise ValueError('Reference/cluster mismatch with material: '+qid)
        if row.get('benchmark') and row['benchmark']!=target['benchmark']:raise ValueError('Benchmark metadata mismatch: '+qid)
        if row.get('source_metadata'):
            for source,expected in target['source_image_metadata'].items():
                if row['source_metadata'].get(source,{}).get('sha256')!=expected['sha256']:
                    raise ValueError('Image identity differs from manifest: '+qid)
        records_by_question[qid][cell]=row
        diagnostics[cell]['present']+=1
        for reason in invalid_reasons(row):diagnostics[cell]['invalid_'+reason]+=1
        if invalid_reasons(row):diagnostics[cell]['invalid_any']+=1
        text=row.get('result',{}).get('generated_text')
        if isinstance(text,str) and benchmark=='gqa':
            m=metric_values(benchmark,text,target['reference_answers'])
            diagnostics[cell]['gqa_case_punct_normalized_disagrees_strict']+=int(m['gqa_case_punct_normalized']!=m['gqa_exact'])
        if isinstance(text,str) and benchmark=='docvqa':
            m=metric_values(benchmark,text,target['reference_answers'])
            if m['docvqa_anls_organizer_framework']!=m['docvqa_anls_strict_paper_cutoff']:
                diagnostics[cell]['exact_threshold_boundary']+=1
    if unknown:raise ValueError('Observation not present in manifest: '+unknown[0])
    for cell in CELLS:
        diagnostics[cell]['missing']=len(materials)-diagnostics[cell]['present']
        if benchmark=='gqa':diagnostics[cell]['gqa_case_punct_normalized_disagrees_strict']+=0
    unit_queries=defaultdict(list)
    for qid,row in materials.items():unit_queries[row['unit_id']].append(qid)
    present_queries={qid for qid in materials if set(records_by_question[qid])==set(CELLS)}
    complete_units={unit for unit,qq in unit_queries.items() if all(q in present_queries for q in qq)}
    # All planned questions of a retained cluster need four actually observed
    # cells. Invalid answers are not silently removed from the primary estimate.
    selected_queries={q for unit in complete_units for q in unit_queries[unit]}
    valid_queries={qid for qid in selected_queries if all(not invalid_reasons(r) for r in records_by_question[qid].values())}
    policies={}
    for policy in ('invalid_as_zero','score_returned_strings','valid_four_cells_only'):
        selected=valid_queries if policy=='valid_four_cells_only' else selected_queries
        per_question={metric:{} for metric in metrics}
        for qid in sorted(selected):
            target=materials[qid];cell_values={metric:{} for metric in metrics}
            for cell in CELLS:
                row=records_by_question[qid][cell];text=row.get('result',{}).get('generated_text')
                invalid=invalid_reasons(row)
                if not isinstance(text,str) or (policy=='invalid_as_zero' and invalid):scores={metric:0. for metric in metrics}
                else:scores=metric_values(benchmark,text,target['reference_answers'])
                for metric,value in scores.items():cell_values[metric][cell]=value
            for metric,values in cell_values.items():per_question[metric][qid]=contrasts(values)
        metric_reports={}
        for metric,questions in per_question.items():
            unit_values={}
            for unit,qq in unit_queries.items():
                retained=[questions[q] for q in qq if q in questions]
                if retained:
                    unit_values[unit]={name:statistics.mean(v[name] for v in retained) for name in retained[0]}
            names=list(next(iter(questions.values()))) if questions else list(CELLS)+['later_recovery_with_main_degraded','later_effect_with_main_clean','main_quality_by_later_quality_interaction','main_recovery_with_later_degraded','joint_degradation_loss']
            metric_reports[metric]=dict(estimates={name:stat(v[name] for v in unit_values.values()) for name in names},
                unit_values=unit_values,query_values=questions,
                question_weighted_diagnostic={cell:statistics.mean(v[cell] for v in questions.values()) if questions else None for cell in CELLS})
        policies[policy]=dict(queries=len(selected),clusters=len({materials[q]['unit_id'] for q in selected}),metrics=metric_reports)
    primary_metric='gqa_exact' if benchmark=='gqa' else 'docvqa_anls_organizer_framework'
    return dict(benchmark=benchmark,primary_metric=primary_metric,primary_policy='invalid_as_zero',
        planned_queries=len(materials),planned_clusters=len(unit_queries),
        queries_with_all_four_observations=len(present_queries),clusters_with_all_planned_queries_observed=len(complete_units),
        incomplete_cluster_ids=sorted(set(unit_queries)-complete_units),
        valid_four_cell_queries_in_complete_clusters=len(valid_queries),
        per_cell_diagnostics={cell:dict(counts) for cell,counts in diagnostics.items()},policies=policies,
        inference='Each contrast is formed within question, averaged within source image/document, then averaged across clusters with descriptive normal 95% intervals. No multiplicity adjustment.',
        coverage_policy='Primary retains clusters with all selected questions observed in all four cells. Invalid/empty/truncated output gets zero; separately report returned-string and valid-four-cell sensitivities. Missing observations are never claimed executed or silently zero-filled.',
        metric_scope='Clean-reference task scores on a selected validation subset; not intrinsic correctness of conflicting-source states or a full benchmark leaderboard result.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--observations',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--model');p.add_argument('--execution-version')
    args=p.parse_args()
    rows=list(read_jsonl(args.manifest));materials={r['id']:r for r in rows}
    if len(materials)!=len(rows):raise ValueError('Duplicate manifest query')
    groups=defaultdict(list);tails=[]
    for file in args.observations:
        for row in read_jsonl(file,tails):
            if row.get('kind')!='generation' or row.get('condition',{}).get('family')!='recovery':continue
            if row.get('readout',{}).get('response_format')!='freeform':continue
            if args.model and row.get('model_key')!=args.model:continue
            if args.execution_version and row.get('execution_version')!=args.execution_version:continue
            if row.get('id') not in materials or row.get('task')!=materials[row['id']]['task']:
                raise ValueError('Observation query/task is not in the supplied manifest: '+str(row.get('id')))
            group=(row['model_key'],row.get('model_revision'),row['execution_version'],row['task'])
            groups[group].append(row)
    if not groups:raise ValueError('No matching freeform recovery observations')
    reports=[]
    model_versions={group[:3] for group in groups}
    planned_tasks={r['task'] for r in materials.values()}
    for model,revision,version,task in sorted(((*mv,task) for mv in model_versions for task in planned_tasks),key=str):
        observations=groups.get((model,revision,version,task),[])
        selected={qid:r for qid,r in materials.items() if r['task']==task}
        reports.append(dict(model=model,model_revision=revision,execution_version=version,task=task,
                            **summarize_group(selected,observations)))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    result=dict(schema='followup-benchmark-recovery-analysis-v14',metric_sources=METRIC_SOURCES,
        implementation_note='Metric versions are explicit because the original ANLS paper uses a strict cutoff while organizer framework code includes similarity == .5. GQA official script performs exact equality without text normalization.',
        sensitivity_note='gqa_case_punct_normalized is a named sensitivity (lowercase, strip whitespace, trailing . ! ? and surrounding quotes on prediction and references); gqa_exact stays primary. per_cell_diagnostics counts rows where it disagrees with gqa_exact.',
        provenance=dict(manifest=str(args.manifest.resolve()),manifest_sha256=sha(args.manifest),
            observations=[dict(path=str(f.resolve()),sha256=sha(f),incomplete_final_lines=[t for t in tails if t['path']==str(f.resolve())]) for f in args.observations],
            script_sha256=sha(__file__)),reports=reports)
    args.output.write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(args.output),groups=len(reports),clusters=[r['clusters_with_all_planned_queries_observed'] for r in reports])))

if __name__=='__main__':main()
