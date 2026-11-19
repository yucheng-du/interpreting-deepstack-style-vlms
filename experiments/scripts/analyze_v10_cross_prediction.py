#!/usr/bin/env python3
"""Small-matrix, grouped exploratory prediction of archived v10 signed effects."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import gzip
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import os as _path_os
DATA_ROOT = Path(_path_os.environ.get("DS_DATA_ROOT", "work")).expanduser().resolve()
import statistics
import sys

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np

MODELS = ('ibm4b', 'qwen8b', 'qwen32b')
READOUTS = (('direct', 'none'), ('label', 'identity'), ('label', 'cyclic'))
BASES = {
    'context_mean': ('one',),
    'identity_only': ('one', 'q'),
    'spatial_gain': ('one', 'z'),
    'spatial_identity': ('one', 'q', 'z'),
    'relocation_extension': ('one', 'q', 'z', 'd', 'qd'),
    'destination_identity': ('one', 'q', 'qd'),
    'condition_saturated': ('one', 'q', 'd', 'm', 'qd', 'qm', 'dm', 'z'),
}
ORIGINAL_METHODS = tuple(list(BASES)[:5])
WORK = (DATA_ROOT / 'runs/measurement-v10-cloud-20261014/remote/work')
V1 = (DATA_ROOT / 'analysis/v10-cross-prediction-20261016')
OUT = (DATA_ROOT / 'analysis/v10-cross-prediction-20261016-v2')
SEED = 'v10-cross-prediction-20261016'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def load_json(path):
    return json.loads(path.read_text())


def input_info(path):
    return dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path))


def folds(scene_ids):
    ordered = sorted(scene_ids, key=lambda s: hashlib.sha256((SEED + '|' + s).encode()).hexdigest())
    assert len(ordered) == 64
    return {s: i // 16 for i, s in enumerate(ordered)}


def basis(row, method, clamp_alpha=False):
    q = 1 if row['query_role'] == 'changed' else -1
    d = 1 if row['destination'] == 'original' else -1
    m = 1 if row['layout'] == 'M0' else -1
    values = dict(one=1., q=q, d=d, m=m, z=q*d*m, qd=q*d, qm=q*m, dm=d*m)
    alpha = .08 if clamp_alpha else row['alpha']
    t = (alpha - .08) / .92
    b = [values[n] for n in BASES[method]]
    active = b + [t*v for v in b]
    zeros = [0.] * len(active)
    return active + zeros if row['entry'] == 'main' else zeros + active


def matrix(rows, method, clamp_alpha=False):
    return np.asarray([basis(r, method, clamp_alpha) for r in rows], dtype=np.float64)


def fit(rows, method):
    x = matrix(rows, method)
    y = np.asarray([r['E'] for r in rows], dtype=np.float64)
    coefficients, _, rank, singular = np.linalg.lstsq(x, y, rcond=None)
    if rank != x.shape[1]:
        raise ValueError(f'Unidentifiable {method}: rank {rank}, columns {x.shape[1]}')
    names = [f'{entry}:{scale}:{term}' for entry in ('main', 'later')
             for scale in ('at_alpha_008', 'slope_t') for term in BASES[method]]
    return coefficients, dict(rank=int(rank), columns=x.shape[1], rows=x.shape[0],
        singular_values=singular.tolist(), condition_number=float(singular[0]/singular[-1]),
        coefficients=dict(zip(names, coefficients.tolist())))


def predict(rows, method, coefficients, clamp_alpha=False):
    return matrix(rows, method, clamp_alpha) @ coefficients


def full_geometry_rows(rows):
    """Whole-image main already carries C's layout; later retains F's main layout.

    This is an explicit post-first-pass design correction, not a fitted choice.
    Alpha remains a cross-arm approximation because local background alpha and
    full fixed-source alpha do not change identical support.
    """
    return [r | {'layout': 'M0' if r['destination']=='original' else 'M1'}
            if r['entry']=='main' else r for r in rows]


def shared_rescue_check(local, rescue):
    fields=('scene','changed_object','readout','mapping','layout','destination','query_role')
    left={tuple(r[k] for k in fields):r['E'] for r in local if r['entry']=='later' and r['alpha']==.08}
    right={tuple(r[k] for k in fields):r['E'] for r in rescue if r['alpha']==.08}
    assert set(left)==set(right)
    delta=[abs(left[k]-right[k]) for k in left]
    return dict(pairs=len(delta),exact_equal=sum(v==0 for v in delta),max_absolute_difference=max(delta),
                mean_absolute_difference=statistics.fmean(delta),interpretation='Repeated conditions, not an independent cross-arm replication')


def load_panel(work, model, panel, provenance):
    directory = work/'analysis'/f'{model}-{panel}-all'
    summary_path, effect_path = directory/'summary.json', directory/'query_effects.jsonl'
    summary = load_json(summary_path)
    provenance.extend((input_info(summary_path), input_info(effect_path)))
    assert summary['coverage']['missing_score_rows'] == 0
    rows, seen = [], set()
    for line in effect_path.open():
        r = json.loads(line)
        if panel == 'full-control' and not (r['query_type'] == 'identity' and r['changing_strength'] == 1 and r['fixed_other_assignment'] == 'a'):
            continue
        assert math.isfinite(r['effect'])
        assert math.isclose(r['effect'], r['b_score']-r['a_score'], abs_tol=1e-10)
        if panel == 'full-control':
            alpha, layout = r['fixed_strength'], r['fixed_layout']
            destination = 'original' if r['changing_layout'] == 'M0' else 'opposite'
            entry, query = r['changed_entry'], r['query_target']
        else:
            alpha, layout, destination = r['main_strength'], r['main_layout'], r['source_destination']
            entry, query = r.get('changed_entry', 'later'), r['query_role']
        row = dict(scene=r['scene_id'], changed_object=r['changed_object'], readout=r['response_format'], mapping=r['mapping'],
                   entry=entry, alpha=float(alpha), layout=layout, destination=destination, query_role=query, E=r['effect'],
                   query_id=r['id'], direction_a=r['direction_a_index'], direction_b=r['direction_b_index'])
        key = tuple(row[k] for k in ('scene','changed_object','readout','mapping','entry','alpha','layout','destination','query_role'))
        assert key not in seen
        seen.add(key)
        rows.append(row)
    assert len(rows) == (6144 if panel == 'rescue' else 12288), (model,panel,len(rows))
    assert len({r['scene'] for r in rows}) == 64
    grouped = defaultdict(list)
    for r in rows:
        grouped[tuple(r[k] for k in ('scene','changed_object','readout','mapping','entry','alpha','layout','destination'))].append(r)
    for pair in grouped.values():
        assert {r['query_role'] for r in pair} == {'changed','unchanged'}
        assert len({(r['direction_a'],r['direction_b']) for r in pair}) == 1
    return rows


def r_pairs(rows, predictions=None):
    pairs = defaultdict(dict)
    if predictions is None:
        predictions = [0.] * len(rows)
    for row, prediction in zip(rows, predictions):
        k = tuple(row[f] for f in ('scene','changed_object','readout','mapping','entry','alpha','layout','destination'))
        assert row['query_role'] not in pairs[k]
        pairs[k][row['query_role']] = (row, float(prediction))
    result = []
    for pair in pairs.values():
        assert set(pair) == {'changed','unchanged'}
        a, b = pair['changed'], pair['unchanged']
        result.append((a[0] | dict(query_role='changed_minus_unchanged', E=a[0]['E']-b[0]['E']), a[1]-b[1]))
    return result


def interval(values):
    mean = statistics.fmean(values)
    se = statistics.stdev(values) / math.sqrt(len(values)) if len(values)>1 else 0.
    return dict(mean=mean, low=mean-1.96*se, high=mean+1.96*se, scene_n=len(values))


def alpha_contrasts(rows, predictions):
    """Pair full R at fixed-source alpha 1 versus .08, keeping C/F and a/b fixed."""
    pairs = defaultdict(dict)
    for row, prediction in r_pairs(rows, predictions):
        key = tuple(row[k] for k in ('scene','changed_object','readout','mapping','entry','layout','destination'))
        assert row['alpha'] not in pairs[key]
        pairs[key][row['alpha']] = (row, prediction)
    result = []
    for pair in pairs.values():
        assert set(pair) == {.08, 1.}
        low, high = pair[.08], pair[1.]
        result.append((high[0] | {'E': high[0]['E']-low[0]['E']}, high[1]-low[1]))
    return result


class Recorder:
    FIELDS = ('model','readout','mapping','fold','task','phase','method','endpoint','scene','changed_object','entry','alpha','layout','destination','query_role','observed','predicted')

    def __init__(self, output):
        self.stream = gzip.open(output/'test-predictions.csv.gz', 'wt', newline='', compresslevel=1)
        self.writer = csv.DictWriter(self.stream, fieldnames=self.FIELDS)
        self.writer.writeheader()
        self.errors = defaultdict(lambda: [0.,0.,0.,0.,0.,0.,0.])
        self.prediction_count = 0

    def emit(self, model, fold, task, phase, method, endpoint, row, prediction):
        truth = row['E']; error = prediction-truth
        if phase == 'test':
            self.writer.writerow(dict(model=model,readout=row['readout'],mapping=row['mapping'],fold=fold,task=task,phase=phase,
                method=method,endpoint=endpoint,scene=row['scene'],changed_object=row['changed_object'],entry=row['entry'],alpha=row['alpha'],
                layout=row['layout'],destination=row['destination'],query_role=row['query_role'],observed=truth,predicted=prediction))
            self.prediction_count += 1
        views = ('direct',) if row['readout']=='direct' else ('label_'+row['mapping'], 'label_pooled')
        strata = ('all', f'entry={row["entry"]};alpha={row["alpha"]}',
                  f'entry={row["entry"]};alpha={row["alpha"]};layout={row["layout"]};destination={row["destination"]}')
        for view, stratum in itertools.product(views,strata):
            key = (model,view,task,phase,method,endpoint,stratum,row['scene'])
            v = self.errors[key]
            v[0] += 1; v[1] += abs(error); v[2] += error*error
            v[3] += truth*truth; v[4] += error
            v[5] += int(truth != 0 and prediction*truth>0); v[6] += int(truth != 0)

    def evaluate(self, model, fold, task, phase, method, rows, predictions, endpoint='both'):
        if endpoint in ('E','both'):
            for row, prediction in zip(rows,predictions):
                self.emit(model,fold,task,phase,method,'E',row,float(prediction))
        if endpoint in ('R','both'):
            for row, prediction in r_pairs(rows,predictions):
                self.emit(model,fold,task,phase,method,'R',row,prediction)

    def finalize(self, output):
        self.stream.close()
        grouped = defaultdict(list)
        for key, v in self.errors.items():
            grouped[key[:-1]].append((key[-1],v))
        metrics = []
        for key, members in sorted(grouped.items()):
            row = dict(zip(('model','readout','task','phase','method','endpoint','stratum'),key))
            n = sum(v[0] for _,v in members)
            mse = sum(v[2] for _,v in members)/n
            target_ms = sum(v[3] for _,v in members)/n
            abs_scene = [v[1]/v[0] for _,v in members]
            ci = interval(abs_scene)
            metrics.append(row | dict(observations=int(n), scenes=len(members), mae=sum(v[1] for _,v in members)/n,
                rmse=math.sqrt(mse), rmse_over_target_rms=math.sqrt(mse/target_ms) if target_ms>0 else None,
                mean_error=sum(v[4] for _,v in members)/n, sign_accuracy=sum(v[5] for _,v in members)/sum(v[6] for _,v in members),
                mae_low=ci['low'],mae_high=ci['high']))
        scene_lookup = {key: {scene:v[1]/v[0] for scene,v in members} for key,members in grouped.items()}
        comparisons = []
        for key, scenes in scene_lookup.items():
            model,readout,task,phase,method,endpoint,stratum = key
            if phase!='test':
                continue
            baselines = ('anchor_constant','anchor_spatial','spatial_identity_clamp008') if task.startswith('full_') else ('context_mean','identity_only','spatial_gain','destination_identity','condition_saturated')
            for baseline in baselines:
                base_key = (model,readout,task,phase,baseline,endpoint,stratum)
                if method==baseline or base_key not in scene_lookup:
                    continue
                base = scene_lookup[base_key]
                assert set(base)==set(scenes)
                differences = [base[s]-scenes[s] for s in sorted(scenes)]
                comparisons.append(dict(model=model,readout=readout,task=task,method=method,baseline=baseline,endpoint=endpoint,stratum=stratum,
                    positive_is='baseline MAE minus method MAE: positive favors method',**interval(differences)))
        save_json(output/'metrics.json', metrics)
        save_json(output/'paired-mae-improvement.json', comparisons)
        with gzip.open(output/'scene-errors.csv.gz','wt',newline='',compresslevel=1) as stream:
            writer=csv.writer(stream);writer.writerow(('model','readout','task','phase','method','endpoint','stratum','scene','n','absolute_error_sum','squared_error_sum','truth_squared_sum','signed_error_sum','correct_sign','nonzero_truth'))
            for key,v in sorted(self.errors.items()):writer.writerow((*key,*v))
        return metrics,comparisons


def subset(rows, scene_ids, predicate=lambda r:True):
    return [r for r in rows if r['scene'] in scene_ids and predicate(r)]


def assert_disjoint(train,test):
    assert not ({r['scene'] for r in train}&{r['scene'] for r in test}), 'Scene leakage'


def execute(work, output):
    if output.resolve() == V1.resolve():
        raise ValueError('v1 is preserved; use the separate v2 output directory')
    output.mkdir(parents=True,exist_ok=True)
    provenance = [input_info(Path(__file__).resolve())]
    protocol = dict(version='v10-cross-prediction-20261016-v2',phase='exploratory after v1 metrics; finite additions recorded before v2 metrics',
        models=BASES,grouping='shared scene_id across arms',fold_rule=SEED+'|scene_id SHA256 sort; 4 consecutive groups of 16',
        estimator='unregularized least squares; full-rank only',strength='affine (alpha-.08)/.92; alpha0 extrapolation separately compared to .08 clamp',
        layouts='leave complete M0 or M1 out of condition training',corners='leave each layout x destination corner out',
        full_bridge='identity, changing_strength1, fixed_assignment a; training-scene-only anchor C=F=M0 fixed_strength .08; one scalar per entry; R only',
        uncertainty='scene paired descriptive normal95 interval of MAE difference; no claim of independent CV training folds',
        additions='destination_identity only local tasks; condition_saturated only local_group; full spatial_identity clamp008 and paired alpha contrast',
        selection='No choice of model/hyperparameter from test outcomes; report all fixed methods')
    save_json(output/'spec-before-metrics.json',protocol)
    save_json(output/'geometry-revision-before-metrics.json',dict(phase='Exploratory revision after initial predictions',
        reason='Compose semantics: local main ROI preserves background; whole full main takes changing source C layout',
        mapping='main layout=C; later main layout=F',parameters='No additional fitted parameters; identical training anchor and scalar',
        limitations='Whole main self-aligns by design, no nontrivial test of spatial rule; background-strength support still differs',
        analysis_script_sha256=sha(__file__)))
    recorder=Recorder(output);coefficients=[];coverage={};split_records=[];all_fold_map=None;alpha_records=[]
    for model in MODELS:
        local=load_panel(work,model,'local-role',provenance)
        rescue=load_panel(work,model,'rescue',provenance)
        full=load_panel(work,model,'full-control',provenance)
        scenes={r['scene'] for r in local}
        assert scenes=={r['scene'] for r in rescue}=={r['scene'] for r in full}
        fm=folds(scenes)
        assert all_fold_map is None or fm==all_fold_map
        all_fold_map=fm
        coverage[model]=dict(local_E=len(local),rescue_E=len(rescue),full_eligible_E=len(full),scenes=len(scenes),shared_rescue=shared_rescue_check(local,rescue))
        for fmt,mapping in READOUTS:
            loc=[r for r in local if (r['readout'],r['mapping'])==(fmt,mapping)]
            res=[r for r in rescue if (r['readout'],r['mapping'])==(fmt,mapping)]
            ful=[r for r in full if (r['readout'],r['mapping'])==(fmt,mapping)]
            for fold in range(4):
                training={s for s in scenes if fm[s]!=fold};testing=scenes-training
                all_train=subset(loc,training);all_test=subset(loc,testing)
                scenarios=[('local_group',all_train,all_test,list(BASES))]
                for layout in ('M0','M1'):
                    scenarios.append(('local_hold_'+layout,subset(loc,training,lambda r:r['layout']!=layout),subset(loc,testing,lambda r:r['layout']==layout),[m for m in BASES if m not in ('relocation_extension','condition_saturated')]))
                for layout,dest in itertools.product(('M0','M1'),('original','opposite')):
                    scenarios.append((f'local_hold_{layout}_{dest}',subset(loc,training,lambda r:(r['layout'],r['destination'])!=(layout,dest)),subset(loc,testing,lambda r:(r['layout'],r['destination'])==(layout,dest)),[m for m in BASES if m!='condition_saturated']))
                group_fits={}
                for task,train,test,methods in scenarios:
                    assert_disjoint(train,test)
                    split_records.append(dict(model=model,readout=fmt,mapping=mapping,fold=fold,task=task,train_scenes=sorted({r['scene'] for r in train}),test_scenes=sorted({r['scene'] for r in test}),train_rows=len(train),test_rows=len(test),methods=methods))
                    for method in methods:
                        beta,fit_info=fit(train,method)
                        coefficients.append(dict(model=model,readout=fmt,mapping=mapping,fold=fold,task=task,method=method,**fit_info))
                        recorder.evaluate(model,fold,task,'train',method,train,predict(train,method,beta))
                        recorder.evaluate(model,fold,task,'test',method,test,predict(test,method,beta))
                        if task=='local_group':group_fits[method]=beta
                # Cross-arm tests never use any test scene in source fitting or target calibration.
                for alpha,clamp,task in ((.08,False,'rescue_alpha008'),(0.,False,'rescue_alpha0_affine'),(0.,True,'rescue_alpha0_clamp008')):
                    test=subset(res,testing,lambda r:r['alpha']==alpha)
                    assert_disjoint(all_train,test)
                    for method,beta in group_fits.items():
                        if method not in ORIGINAL_METHODS:continue
                        recorder.evaluate(model,fold,task,'test',method,test,predict(test,method,beta,clamp))
                anchor=subset(ful,training,lambda r:r['alpha']==.08 and r['layout']=='M0' and r['destination']=='original')
                full_test=subset(ful,testing)
                assert_disjoint(anchor,full_test)
                full_train_other=subset(ful,training,lambda r:not(r['alpha']==.08 and r['layout']=='M0' and r['destination']=='original'))
                for entry in ('main','later'):
                    anchor_entry=[r for r in anchor if r['entry']==entry]
                    actual_anchor=r_pairs(anchor_entry)
                    anchor_mean=statistics.fmean(row['E'] for row,_ in actual_anchor)
                    for phase,raw in (('calibration',anchor_entry),('train_nonanchor',[r for r in full_train_other if r['entry']==entry]),('test',[r for r in full_test if r['entry']==entry])):
                        for row,_ in r_pairs(raw):
                            alignment=(1 if row['layout']=='M0' else -1)*(1 if row['destination']=='original' else -1)
                            recorder.emit(model,fold,'full_calibrated_bridge',phase,'anchor_constant','R',row,anchor_mean)
                            recorder.emit(model,fold,'full_calibrated_bridge',phase,'anchor_spatial','R',row,anchor_mean*alignment)
                            recorder.emit(model,fold,'full_geometry_diagnostic',phase,'anchor_constant','R',row,anchor_mean)
                            recorder.emit(model,fold,'full_geometry_diagnostic',phase,'anchor_spatial','R',row,anchor_mean*(1 if entry=='main' else alignment))
                    for method in ('spatial_gain','spatial_identity','relocation_extension'):
                        beta=group_fits[method]
                        anchor_pairs=r_pairs(anchor_entry,predict(anchor_entry,method,beta))
                        x=np.asarray([prediction for _,prediction in anchor_pairs]);y=np.asarray([row['E'] for row,_ in anchor_pairs])
                        den=float(x@x)
                        if den==0:raise ValueError('Unidentifiable scalar anchor')
                        gamma=float(x@y/den)
                        coefficients.append(dict(model=model,readout=fmt,mapping=mapping,fold=fold,task='full_calibrated_bridge',method=method,entry=entry,
                            scalar=gamma,scalar_rank=1,anchor_scenes=sorted(training),anchor_pairs=len(anchor_pairs),anchor_only=True,
                            local_anchor_R_mean=float(x.mean()),full_anchor_R_mean=float(y.mean())))
                        for phase,raw in (('calibration',anchor_entry),('train_nonanchor',[r for r in full_train_other if r['entry']==entry]),('test',[r for r in full_test if r['entry']==entry])):
                            for row,prediction in r_pairs(raw,predict(raw,method,beta)):
                                recorder.emit(model,fold,'full_calibrated_bridge',phase,method,'R',row,gamma*prediction)
                            for row,prediction in r_pairs(raw,predict(full_geometry_rows(raw),method,beta)):
                                recorder.emit(model,fold,'full_geometry_diagnostic',phase,method,'R',row,gamma*prediction)
                            if method=='spatial_identity':
                                adjusted=full_geometry_rows(raw)
                                frozen=gamma*predict(adjusted,method,beta,clamp_alpha=True)
                                for row,prediction in r_pairs(raw,frozen):
                                    recorder.emit(model,fold,'full_geometry_diagnostic',phase,'spatial_identity_clamp008','R',row,prediction)
                                if phase=='test':
                                    affine=gamma*predict(adjusted,method,beta)
                                    for row,prediction in alpha_contrasts(raw,affine):
                                        alpha_records.append(dict(model=model,fold=fold,**{k:row[k] for k in ('scene','changed_object','readout','mapping','entry','layout','destination')},
                                            observed_delta_R=row['E'],predicted_delta_R=prediction,frozen_delta_R=0.))
        print(f'{model}: grouped/condition/cross-arm predictions finished',flush=True)
    metrics,comparisons=recorder.finalize(output)
    save_json(output/'coefficients.json',coefficients)
    save_json(output/'splits.json',dict(scene_fold=all_fold_map,splits=split_records))
    save_json(output/'full-alpha-contrasts.json',alpha_records)
    save_json(output/'provenance.json',dict(inputs=provenance,numpy=np.__version__,coverage=coverage,test_prediction_rows=recorder.prediction_count,
        all_fits_full_rank=True,all_train_test_scenes_disjoint=True,full_anchor_uses_training_scenes_only=True,
        reproduce=[sys.executable,str(Path(__file__).resolve()),'--work',str(work),'--output',str(output)]))
    if V1.is_dir():
        verify_v1(output, metrics, coefficients)


def verify_v1(output, metrics, coefficients):
    key_fields=('model','readout','task','phase','method','endpoint','stratum')
    lookup={tuple(r[k] for k in key_fields):r for r in metrics}
    prior=load_json(V1/'metrics.json')
    for row in prior:assert lookup[tuple(row[k] for k in key_fields)]==row
    def coef_key(row):return tuple(row.get(k) for k in ('model','readout','mapping','fold','task','method','entry'))
    coef_lookup={coef_key(r):r for r in coefficients}
    old_coef=load_json(V1/'coefficients.json')
    for row in old_coef:assert coef_lookup[coef_key(row)]==row
    result=dict(v1_metrics_identical=len(prior),v1_coefficients_identical=len(old_coef),
        v1_inputs=[input_info(V1/name) for name in ('metrics.json','coefficients.json','splits.json')])
    save_json(output/'v1-preservation.json',result)
    return result






def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work',type=Path,default=WORK);parser.add_argument('--output',type=Path,default=OUT)
    args=parser.parse_args()
    execute(args.work,args.output)


if __name__=='__main__':main()
