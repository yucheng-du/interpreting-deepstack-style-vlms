"""Long-format severity curves: model x material x task x readout x kind x severity label.

Rows copy n, mean, se, interval and denominators from the analysis estimand_table and the free-form
scorer estimates (read by calibrate.read_inputs); nothing is re-estimated. realized_native_degradation
is copied when an input reports it; otherwise it is native clean minus joint-degraded of the two
copied means, without an interval (value_source says which). Every severity label is kept, including
floor/ceiling and unfavourable ones, and no row is dropped by effect sign; only estimands outside the
run design (planned = 0) are not listed. Reference rows (clean recovery cell, normal conflict main)
have severity_rank 0. With --calibration, rows say whether their label is in the shared physical
grid, and the capability-calibrated labels are copied into a separate, labelled table: a
model-specific severity, a different estimand from the shared physical grid.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sys

try:from cloud_prep.measurement_v15 import calibrate as C
except ModuleNotFoundError:
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]));from cloud_prep.measurement_v15 import calibrate as C

SCHEMA='v15-severity-curves-1'
FIELDS=('source','model','model_revision','execution_version','material_set','split','task','response_format','mapping','kind','measure',
    'primary_measure','quantity','quality','severity_label','severity_rank','family','reference_kind','setting_value','query_role','source_estimand',
    'n','mean','se','ci_low','ci_high','interval','level','planned','used','planned_units','excluded','se_zero','n_lt_5','value_source',
    'in_shared_grid','input')
CALIBRATED_FIELDS=FIELDS+('calibrated_target','calibration_r_mean','interpolated_value','estimand_note')
DERIVED='derived: native clean minus joint-degraded of the copied means; no interval'
CONFLICT_METRICS=('b_minus_a_log_odds','answer_b_indicator')
KEY=C.GROUP+('quantity','severity_label','query_role')
MD_GROUP=('material_set','task','model','model_revision','execution_version','source','response_format','mapping','kind')
MD_COLUMNS=(('CC','native_clean',False),('LL','joint_degraded',False),('CC-LL','realized_native_degradation',False),('DC-DD','later_recovery',True),
    ('interaction','recovery_interaction',True),('T','conflict_target_effect',False),('C','conflict_control_effect',False),('R','selectivity_R',False),
    ('dR vs normal','delta_R_vs_normal',True))


def make_row(r,settings,grid,value_source):
    label=r['severity_label'];s=settings.get(r['task'],{}).get(label) if label is not None else None;ci=r.get('ci95') or [None,None]
    row={k:r.get(k) for k in FIELDS}
    row.update(severity_rank=0 if label is None else s and s['rank'],family=s and s['family'],reference_kind=s and s['reference_kind'],
        setting_value=s and s['value'],ci_low=ci[0],ci_high=ci[1],value_source=value_source,
        in_shared_grid=None if label is None or r['task'] not in grid else label in grid[r['task']])
    return row


def sort_key(x):
    return tuple(str(x[k]) for k in MD_GROUP+('measure',))+(1e9 if x['severity_rank'] is None else x['severity_rank'],
        str(x['severity_label']),x['quantity'],str(x['query_role']))


def curve_rows(records,calibration=None):
    settings=C.task_settings(records);grid=(calibration or {}).get('severity_labels') or {};keyed={};rows=[]
    for r in records:
        if r.get('planned')==0:continue
        key=tuple(r[k] for k in KEY)
        if key in keyed:raise ValueError('Two inputs give the same curve cell: '+json.dumps(key))
        keyed[key]=r;rows.append(make_row(r,settings,grid,'copied'))
    for key,r in keyed.items():
        if r['quantity']!='joint_degraded':continue
        group,label=key[:len(C.GROUP)],r['severity_label']
        if (*group,'realized_native_degradation',label,None) in keyed:continue
        cc=keyed.get((*group,'native_clean',label,None)) or keyed.get((*group,'native_clean',None,None))
        mean=None if cc is None or cc['mean'] is None or r['mean'] is None else cc['mean']-r['mean']
        rows.append(make_row(dict({k:r[k] for k in C.GROUP+('split','primary_measure','input')},quantity='realized_native_degradation',
            quality=label,severity_label=label,query_role=None,source_estimand=None,n=None,mean=mean,se=None,ci95=None,interval=None),settings,grid,DERIVED))
    return sorted(rows,key=sort_key)


def calibrated_rows(rows,calibration):
    picks={(e['model'],e['task']):e for e in calibration['capability_calibrated']['entries']}
    return [dict(x,calibrated_target=e['target'],calibration_r_mean=e['r_mean'],interpolated_value=e['interpolated_value'],estimand_note=C.CALIBRATED_NOTE)
        for x in rows for e in [picks.get((x['model'],x['task']))] if e and x['severity_label']==e['label']]


def write_csv(path,rows,fields):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for x in rows:w.writerow({k:'' if x.get(k) is None else json.dumps(x[k],sort_keys=True) if isinstance(x[k],(dict,list)) else x[k] for k in fields})


def markdown(rows,calibrated,inputs,measure):
    index=defaultdict(list);present=defaultdict(set)
    for x in rows:
        g=tuple(str(x[k]) for k in MD_GROUP);index[g+(x['quantity'],x['severity_label'])].append(x);present[g].add(x['quantity'])
        if x['severity_label'] is None:present[g].add('reference')
    def pick(g,q,label):
        found=[x for x in index.get(g+(q,label),[]) if (x['measure'] in CONFLICT_METRICS if q.startswith(('conflict','selectivity','delta')) else
            x['measure']==measure or x['primary_measure'])]
        if q=='native_clean' and not found and label is not None:return pick(g,q,None)
        return min(found,key=lambda x:{'unchanged':0,'control':1}.get(x['query_role'],2)) if found else None
    def cell(x,interval):
        if x is None or x['mean'] is None:return '-'
        text=f"{x['mean']:.3f}"+('*' if x['value_source']==DERIVED else '')
        return text+(f" [{x['ci_low']:.3f}, {x['ci_high']:.3f}]" if interval and x['ci_low'] is not None else '')
    def line(g,label,shown):
        n=(pick(g,'native_clean' if label is None else 'joint_degraded',label) or {}).get('n')
        main='reference (clean/normal)' if label is None else label
        return '| '+' | '.join([g[2],g[6]+'/'+g[7],g[8],main,'-' if n is None else str(n)]+[cell(pick(g,q,label),interval)
            if q in present[g] and (label is not None or q in ('native_clean',)+shown) else '' for _,q,interval in MD_COLUMNS])+' |'
    header=['| model | readout | kind | label | n | '+' | '.join(c for c,_,_ in MD_COLUMNS)+' |','|'+'---|'*(5+len(MD_COLUMNS))]
    groups=sorted({k[:len(MD_GROUP)] for k in index});ranks=defaultdict(dict)
    for x in rows:
        if x['severity_label'] is not None:ranks[tuple(str(x[k]) for k in MD_GROUP)][x['severity_label']]=x['severity_rank']
    out=['# v15 severity curves','',f'Values, intervals (t or as given) and denominators are copied from the inputs, never re-estimated. Two-choice recovery '
        f'cells use `{measure}`; free-form rows use the scorer primary metric/policy; T/C/R use b-a log-odds (score) or the b-answer indicator '
        '(generation), vs the normal main for dR. CC-LL marked * is the difference of the copied means (no interval); - = planned but no '
        'estimate; blank = not in the run design. Every label is listed, including floor/ceiling and unfavourable ones. '
        'Full rows (all measures, metrics and policies): severity_curves.csv / severity_curves.json.','']
    out+=[f"- {x['kind']}: `{x['path']}` sha256 {x['sha256'][:16]}" for x in inputs]
    for task_key in sorted({g[:2] for g in groups}):
        out+=['',f'## {task_key[1]} ({task_key[0]})','']+header
        for g in [g for g in groups if g[:2]==task_key]:
            labels=sorted(ranks[g],key=lambda L:(1e9 if ranks[g][L] is None else ranks[g][L],L))
            out+=[line(g,None,('conflict_target_effect','conflict_control_effect','selectivity_R'))]*('reference' in present[g])+[line(g,L,()) for L in labels]
    if calibrated:
        out+=['',f'## Capability-calibrated labels: {C.CALIBRATED_NOTE}','','Same rows as above, restricted per model x task to the calibrated label '
            '(capability_calibrated.csv / .json).','']+header
        seen=sorted({(tuple(str(x[k]) for k in MD_GROUP),x['severity_label']) for x in calibrated})
        out+=[line(g,L,()) for g,L in seen]
    return '\n'.join(out)+'\n'


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--analysis',type=Path,nargs='*',default=[]);p.add_argument('--benchmark',type=Path,nargs='*',default=[])
    p.add_argument('--calibration',type=Path,help='calibrate.py output: marks shared-grid labels and adds the capability-calibrated table')
    p.add_argument('--output',type=Path,required=True,help='output directory')
    p.add_argument('--markdown-measure',choices=sorted(C.MEASURES),default='accuracy_invalid_as_incorrect')
    a=p.parse_args(argv)
    if not a.analysis and not a.benchmark:p.error('pass --analysis and/or --benchmark inputs')
    warnings=[];records,inputs=C.read_inputs(a.analysis,a.benchmark,warnings.append);calibration=None
    if a.calibration:
        data=a.calibration.read_bytes();calibration=json.loads(data)
        if not isinstance(calibration,dict) or calibration.get('schema')!=C.SCHEMA:raise ValueError(f'{a.calibration}: not a calibrate.py output')
        inputs.append(dict(kind='calibration',path=str(a.calibration.resolve()),sha256=hashlib.sha256(data).hexdigest()))
    for task,found in sorted(C.task_settings(records).items()):
        for label in sorted(L for L,s in found.items() if s is None):warnings.append(f'{task}: {label!r} is not a v15 severity label; kept without a physical setting')
    rows=curve_rows(records,calibration);calibrated=calibrated_rows(rows,calibration) if calibration else []
    out=a.output;out.mkdir(parents=True,exist_ok=True)
    notes=dict(copied='n, mean, se, interval and denominators as given by the inputs; never re-estimated',derived=DERIVED,
        kept='every severity label and effect sign; estimands with planned = 0 (outside the run design) are not listed',
        reference_rows='severity_rank 0: clean recovery cell, normal conflict main',
        severity_rank='order among the task labels present in these inputs (1 = mildest); compare across files by family/setting_value',
        capability_calibrated=C.CALIBRATED_NOTE)
    meta=dict(schema=SCHEMA,notes=notes,inputs=inputs,shared_grid=(calibration or {}).get('severity_labels'),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),warnings=warnings)
    (out/'severity_curves.json').write_text(json.dumps(dict(meta,rows=rows),indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    write_csv(out/'severity_curves.csv',rows,FIELDS)
    if calibration:
        (out/'capability_calibrated.json').write_text(json.dumps(dict(meta,note=C.CALIBRATED_NOTE,rows=calibrated),indent=2,ensure_ascii=False,allow_nan=False)+'\n')
        write_csv(out/'capability_calibrated.csv',calibrated,CALIBRATED_FIELDS)
    (out/'severity_curves.md').write_text(markdown(rows,calibrated,inputs,a.markdown_measure))
    for w in warnings:print('WARNING: '+w,file=sys.stderr)
    print(json.dumps(dict(output=str(out),rows=len(rows),capability_calibrated_rows=len(calibrated),warnings=len(warnings))));return 0


if __name__=='__main__':raise SystemExit(main())
