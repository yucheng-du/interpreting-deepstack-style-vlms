#!/usr/bin/env python3
"""Score v15 freeform recovery severity curves against clean-reference answers.

Cells come from the observation condition fields main_quality/later_quality,
with reference quality 'clean': one shared clean__clean cell and, per severity
label L, (clean,L), (L,clean) and (L,L) in a four-cell run, or (L,L) alone in a
native-only calibration run. Within a label the executed v14 four-cell code runs
on generic cell names (clean__clean, clean__degraded, degraded__clean,
degraded__degraded); a native-only label reports clean, joint-degraded and the
realized native degradation CC-DD without any recovery contrast. Completeness is
judged per label, so adding or removing another severity leaves the estimates of
a label unchanged.

GQA primary: exact equality, as in the author's evaluator. Lower+strip equality
and gqa_case_punct_normalized (lowercase, strip whitespace, trailing . ! ? and
surrounding quotes) are named sensitivities, not official normalizations; per
cell, rows where the latter disagrees with strict equality are counted. DocVQA
primary follows the organizers' public SP/MP framework (lower+strip, internal
spaces retained, max-reference Levenshtein similarity >= .5); the original
strict-paper threshold is reported separately. No substring, semantic or article
matching.

Question-level contrasts are averaged within image/document first. Normal
intervals concern equally weighted clusters, not the official full-set
question-weighted leaderboard average (reported only as a separate diagnostic).
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cloud_prep.measurement_v15 import severity  # noqa: E402  (Pillow-free label parsing)

REFERENCE='clean'
CELLS=('clean__clean','clean__degraded','degraded__clean','degraded__degraded')
NATIVE_CELLS=('clean__clean','degraded__degraded')
POLICIES=('invalid_as_zero','score_returned_strings','valid_cells_only')
CONTRASTS=('later_recovery_with_main_degraded','later_effect_with_main_clean','main_quality_by_later_quality_interaction',
           'main_recovery_with_later_degraded','joint_degradation_loss')
CONTRAST_DEFINITIONS={
 'clean__clean':'CC: main clean, later clean (native clean ability; shared by all labels)',
 'clean__degraded':'CD: main clean, later entries from the label-L image',
 'degraded__clean':'DC: main from the label-L image, later entries clean',
 'degraded__degraded':'DD: main and later from the label-L image (native joint-degraded ability)',
 'later_recovery_with_main_degraded':'DC-DD: later recovery',
 'later_effect_with_main_clean':'CC-CD',
 'main_quality_by_later_quality_interaction':'(DC-DD)-(CC-CD): quality interaction',
 'main_recovery_with_later_degraded':'CD-DD',
 'joint_degradation_loss':'CC-DD: realized native degradation',
}
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


QUOTES={('"','"'),("'","'"),('`','`'),('“','”'),('‘','’')}


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
    if set(values)==set(NATIVE_CELLS):
        return {**values,'joint_degradation_loss':values['clean__clean']-values['degraded__degraded']}
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


def label_cells(label,mode):
    """Generic cell name -> (main_quality, later_quality) for one severity label."""
    pairs={'clean__clean':(REFERENCE,REFERENCE),'clean__degraded':(REFERENCE,label),
           'degraded__clean':(label,REFERENCE),'degraded__degraded':(label,label)}
    return {cell:pairs[cell] for cell in (CELLS if mode=='four_cell' else NATIVE_CELLS)}


def collect(materials,observations):
    """Validate observations against the manifest; return {qid: {(main_quality, later_quality): row}}."""
    records=defaultdict(dict);unknown=[]
    for row in observations:
        qid=row['id'];condition=row['condition'];qualities=(condition['main_quality'],condition['later_quality'])
        off=set(qualities)-{REFERENCE}
        if len(off)>1:raise ValueError('Mixed-severity recovery cell: '+qid+'/'+str(qualities))
        label=next(iter(off)) if off else None
        if 'severity_label' in condition and condition['severity_label']!=label:
            raise ValueError('Condition severity_label disagrees with its qualities: '+qid+'/'+str(condition.get('name')))
        if qid not in materials:unknown.append(qid);continue
        if qualities in records[qid]:raise ValueError('Duplicate query/cell; select one execution version or nonoverlapping files: '+qid+'/'+'__'.join(qualities))
        target=materials[qid]
        if row.get('reference_answers')!=target['reference_answers'] or row.get('unit_id')!=target['unit_id']:
            raise ValueError('Reference/cluster mismatch with material: '+qid)
        if row.get('benchmark') and row['benchmark']!=target['benchmark']:raise ValueError('Benchmark metadata mismatch: '+qid)
        if label is not None and label not in target.get('severity_sources',{}):raise ValueError('Severity label not planned for this material: '+qid+'/'+label)
        expected={q:'a' if q==REFERENCE else target['severity_sources'][q] for q in qualities}
        for key,quality in (('main_source',qualities[0]),('later_base',qualities[1]),('later_donor',qualities[1])):
            if key in condition and condition[key]!=expected[quality]:raise ValueError(f'Condition {key} does not match its quality: {qid}/{condition[key]}')
        if row.get('source_metadata'):
            # Only the sources this cell used; other severities may be added to the manifest later.
            for source in sorted(set(expected.values())):
                if row['source_metadata'].get(source,{}).get('sha256')!=target['source_image_metadata'][source]['sha256']:
                    raise ValueError('Image identity differs from manifest: '+qid+'/'+source)
        records[qid][qualities]=row
    if unknown:raise ValueError('Observation not present in manifest: '+unknown[0])
    return records


def summarize_label(materials,records,label,mode):
    """One label of one model/version/task; the executed v14 four-cell code on generic cell names."""
    benchmark=next(iter(materials.values()))['benchmark']['name']
    metrics=list(metric_values(benchmark,'',['__nonempty_reference__']))
    cells=label_cells(label,mode)
    by_question={qid:{cell:records[qid][pair] for cell,pair in cells.items() if pair in records.get(qid,{})} for qid in materials}
    diagnostics={cell:Counter() for cell in cells}
    for qid,found in by_question.items():
        for cell,row in found.items():
            diagnostics[cell]['present']+=1
            for reason in invalid_reasons(row):diagnostics[cell]['invalid_'+reason]+=1
            if invalid_reasons(row):diagnostics[cell]['invalid_any']+=1
            text=row.get('result',{}).get('generated_text')
            if isinstance(text,str) and benchmark=='gqa':
                m=metric_values(benchmark,text,materials[qid]['reference_answers'])
                diagnostics[cell]['gqa_case_punct_normalized_disagrees_strict']+=int(m['gqa_case_punct_normalized']!=m['gqa_exact'])
            if isinstance(text,str) and benchmark=='docvqa':
                m=metric_values(benchmark,text,materials[qid]['reference_answers'])
                if m['docvqa_anls_organizer_framework']!=m['docvqa_anls_strict_paper_cutoff']:
                    diagnostics[cell]['exact_threshold_boundary']+=1
    for cell in cells:
        diagnostics[cell]['missing']=len(materials)-diagnostics[cell]['present']
        if benchmark=='gqa':diagnostics[cell]['gqa_case_punct_normalized_disagrees_strict']+=0
    unit_queries=defaultdict(list)
    for qid,row in materials.items():unit_queries[row['unit_id']].append(qid)
    present_queries={qid for qid in materials if set(by_question[qid])==set(cells)}
    complete_units={unit for unit,qq in unit_queries.items() if all(q in present_queries for q in qq)}
    # All planned questions of a retained cluster need every cell of this label actually
    # observed. Invalid answers are not silently removed from the primary estimate.
    selected_queries={q for unit in complete_units for q in unit_queries[unit]}
    valid_queries={qid for qid in selected_queries if all(not invalid_reasons(r) for r in by_question[qid].values())}
    names=list(cells)+(list(CONTRASTS) if mode=='four_cell' else ['joint_degradation_loss'])
    policies={}
    for policy in POLICIES:
        selected=valid_queries if policy=='valid_cells_only' else selected_queries
        per_question={metric:{} for metric in metrics}
        for qid in sorted(selected):
            target=materials[qid];cell_values={metric:{} for metric in metrics}
            for cell in cells:
                row=by_question[qid][cell];text=row.get('result',{}).get('generated_text')
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
            metric_reports[metric]=dict(estimates={name:stat(v[name] for v in unit_values.values()) for name in names},
                unit_values=unit_values,query_values=questions,
                question_weighted_diagnostic={cell:statistics.mean(v[cell] for v in questions.values()) if questions else None for cell in cells})
        policies[policy]=dict(queries=len(selected),clusters=len({materials[q]['unit_id'] for q in selected}),metrics=metric_reports)
    return dict(label=label,cell_mode=mode,cell_qualities={cell:list(pair) for cell,pair in cells.items()},
        planned_queries=len(materials),planned_clusters=len(unit_queries),
        queries_with_all_cells_observed=len(present_queries),clusters_with_all_planned_queries_observed=len(complete_units),
        incomplete_cluster_ids=sorted(set(unit_queries)-complete_units),
        valid_cell_queries_in_complete_clusters=len(valid_queries),
        per_cell_diagnostics={cell:dict(counts) for cell,counts in diagnostics.items()},policies=policies)


def summarize_group(materials,observations,labels=None,cells='auto'):
    """One model/version/task across severity labels; all cells share planned clean labels and clusters."""
    benchmark=next(iter(materials.values()))['benchmark']['name']
    if any(r['benchmark']['name']!=benchmark for r in materials.values()):raise ValueError('Mixed benchmark group')
    records=collect(materials,observations)
    planned=[]
    for row in materials.values():planned+=[x for x in row.get('severity_sources',{}) if x not in planned]
    observed={q for found in records.values() for pair in found for q in pair}-{REFERENCE}
    labels=severity.ordered(labels if labels is not None else observed)
    modes={}
    for label in labels:
        off_diagonal=any((REFERENCE,label) in found or (label,REFERENCE) in found for found in records.values())
        modes[label]=cells if cells!='auto' else ('four_cell' if off_diagonal else 'native_only')
    per_label={label:summarize_label(materials,records,label,modes[label]) for label in labels}
    primary_metric='gqa_exact' if benchmark=='gqa' else 'docvqa_anls_organizer_framework'
    curve=[]
    for label in labels:
        estimates=per_label[label]['policies']['invalid_as_zero']['metrics'][primary_metric]['estimates']
        four=modes[label]=='four_cell'
        curve.append(dict(label=label,value=severity.parse_label(label)['value'],cell_mode=modes[label],
            clusters=per_label[label]['clusters_with_all_planned_queries_observed'],
            native_clean=estimates['clean__clean'],joint_degraded=estimates['degraded__degraded'],
            realized_native_degradation=estimates['joint_degradation_loss'],
            later_recovery=estimates['later_recovery_with_main_degraded'] if four else None,
            quality_interaction=estimates['main_quality_by_later_quality_interaction'] if four else None))
    unit_queries=defaultdict(list)
    for qid,row in materials.items():unit_queries[row['unit_id']].append(qid)
    common={unit for unit in unit_queries if all(unit not in per_label[label]['incomplete_cluster_ids'] for label in labels)} if labels else set()
    return dict(benchmark=benchmark,primary_metric=primary_metric,primary_policy='invalid_as_zero',
        planned_queries=len(materials),planned_clusters=len(unit_queries),planned_labels=severity.ordered(planned) if planned else [],
        labels=labels,unobserved_planned_labels=[x for x in planned if x not in observed],cell_modes=modes,
        clusters_complete_for_all_labels=len(common),curve=curve,per_label=per_label,
        inference='Each contrast is formed within question and label, averaged within source image/document, then averaged across clusters with descriptive normal 95% intervals. No multiplicity adjustment.',
        coverage_policy='Per label, the primary retains clusters with all selected questions observed in every cell of that label (CC plus the label cells). Invalid/empty/truncated output gets zero; separately report returned-string and valid-cell sensitivities. Missing observations are never claimed executed or silently zero-filled.',
        metric_scope='Clean-reference task scores on a selected validation subset; not intrinsic correctness of conflicting-source states or a full benchmark leaderboard result.')


def report_tags(materials,observations):
    """split/material_set from the manifest rows; observation tags, when present, must agree."""
    tags={}
    for field in ('split','material_set'):
        values={r.get(field) for r in materials.values()}-{None}
        seen={r.get(field) for r in observations}-{None}
        if seen-values:raise ValueError(f'Observations tagged {field} {sorted(seen-values)} do not match the manifest {sorted(values)}')
        if len(values)>1:tags[field]='mixed';tags[field+'_values']=sorted(values)
        elif values:tags[field]=values.pop()
    return tags


def csv_rows(reports):
    for report in reports:
        for label,entry in report['per_label'].items():
            for policy,block in entry['policies'].items():
                for metric,values in block['metrics'].items():
                    for name,s in values['estimates'].items():
                        yield dict(model=report['model'],model_revision=report['model_revision'],execution_version=report['execution_version'],
                            task=report['task'],split=report.get('split'),material_set=report.get('material_set'),label=label,cell_mode=entry['cell_mode'],policy=policy,metric=metric,estimate=name,**s)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--observations',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--csv',type=Path)
    p.add_argument('--model');p.add_argument('--execution-version')
    p.add_argument('--labels',nargs='+',help='Report these severity labels (default: every observed label)')
    p.add_argument('--cells',choices=['auto','four_cell','native_only'],default='auto',
        help='auto: four_cell when any off-diagonal cell of a label is observed, else native_only')
    args=p.parse_args(argv)
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
                            **report_tags(selected,observations),**summarize_group(selected,observations,args.labels,args.cells)))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    result=dict(schema='followup-benchmark-severity-analysis-v15',metric_sources=METRIC_SOURCES,contrast_definitions=CONTRAST_DEFINITIONS,
        implementation_note='Metric versions are explicit because the original ANLS paper uses a strict cutoff while organizer framework code includes similarity == .5. GQA official script performs exact equality without text normalization.',
        sensitivity_note='gqa_case_punct_normalized is a named sensitivity (lowercase, strip whitespace, trailing . ! ? and surrounding quotes on prediction and references); gqa_exact stays primary. per_cell_diagnostics counts rows where it disagrees with gqa_exact.',
        calibration_note='A native-only label (CC and DD only) estimates clean and joint-degraded ability and CC-DD; it carries no recovery contrast and must not select severities by recovery.',
        provenance=dict(manifest=str(args.manifest.resolve()),manifest_sha256=sha(args.manifest),
            observations=[dict(path=str(f.resolve()),sha256=sha(f),incomplete_final_lines=[t for t in tails if t['path']==str(f.resolve())]) for f in args.observations],
            script_sha256=sha(__file__),labels_argument=args.labels,cells_argument=args.cells),reports=reports)
    args.output.write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    if args.csv:
        fields=['model','model_revision','execution_version','task','split','material_set','label','cell_mode','policy','metric','estimate','n_clusters','mean','se','ci_low','ci_high']
        with args.csv.open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(csv_rows(reports))
    print(json.dumps(dict(output=str(args.output),groups=len(reports),
        clusters={r['task']+'/'+r['model']:{x['label']:x['clusters'] for x in r['curve']} for r in reports})))

if __name__=='__main__':main()
