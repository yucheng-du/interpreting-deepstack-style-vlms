"""CPU tests of severity calibration (clean/joint-degraded only) and copied severity curves."""
import contextlib
import io
import json
from pathlib import Path
import random
import tempfile
import unittest

from cloud_prep.measurement_v15 import calibrate as C, curves, design, severity
from cloud_prep.measurement_v15.analysis import analyze

ACC={'accuracy_valid_only':'recovery_clean_reference_accuracy','accuracy_invalid_as_incorrect':'recovery_clean_reference_accuracy_invalid_as_incorrect'}
COLOR=[s['label'] for s in severity.grid('color','candidates')]
EXCLUDED=dict(missing=0,truncated=0,invalid=0,alignment_unknown=0,reference_not_a_or_b=0)
CELL={'native_clean':lambda L:('clean','clean'),'joint_degraded':lambda L:(L,L),'main_degraded_later_clean':lambda L:(L,'clean'),
    'main_clean_later_degraded':lambda L:('clean',L),'later_recovery':lambda L:(L,'clean'),'recovery_interaction':lambda L:(L,'clean')}


def entry(estimand,mean,main=None,later=None,measure=None,role=None,n=8):
    return dict(estimand=estimand,measure=measure,main_quality=main,later_quality=later,query_role=role,control_alignment=None,comparison=None,
        n=n,mean=mean,se=.05,ci95_t=[mean-.1,mean+.1],se_zero=False,n_lt_5=False,level='queries',planned=n,used=n,planned_units=n,excluded=dict(EXCLUDED))


def group(model,cc,ll,task='color',dc=None,cd=None,recovery=None,interaction=None,conflict=None,fmt='direct',mapping='none'):
    """An analysis.py group as in its estimand_table: recovery cells per label, contrasts, conflict T/C/R vs normal."""
    table=[]
    for measure,name in ACC.items():
        table.append(entry(name,cc,'clean','clean',measure))
        for L,v in ll.items():
            table+=[entry(name,v,L,L,measure),entry(name,(dc or {}).get(L,v),L,'clean',measure),entry(name,(cd or {}).get(L,cc),'clean',L,measure),
                entry('recovery_clean_later_benefit_with_degraded_main',(recovery or {}).get(L,0.),L,'clean',measure),
                entry('recovery_quality_interaction',(interaction or {}).get(L,0.),L,'clean',measure)]
    for q,(t,c,r) in (conflict or {}).items():
        table+=[entry('conflict_query_effect',t,q,role='changed'),entry('conflict_query_effect',c,q,role='unchanged'),entry('selectivity_R',r,q)]
        if q!='normal':table.append(entry('selectivity_interaction',r-conflict['normal'][2],q))
    return dict(model=model,revision='r',execution_version='v15-test',task=task,kind='score',response_format=fmt,mapping=mapping,
        reference_labels=dict(conflict_reference_main_quality='normal',recovery_clean_quality='clean',source='test'),estimand_table=table)


def records(*groups):
    return list(C.analysis_records(dict(results=list(groups)),'fixture.json'))


# Retained ability per model over the six colour candidates (clean accuracy .9, chance .5).
CURVES={'A':[1.,.9,.7,.5,.3,.1],'B':[.95,.85,.6,.45,.2,0.],'C':[1.,.8,.75,.55,.25,.05],'D':[.9,.7,.5,.3,.1,0.]}


def curve_groups(**extra):
    return [group(m,.9,{L:.5+r*.4 for L,r in zip(COLOR,rs)},**extra) for m,rs in CURVES.items()]


def stat(mean,n=20):
    return dict(n_clusters=n,mean=mean,se=.02,ci_low=mean-.04,ci_high=mean+.04)


class CalibrationTests(unittest.TestCase):
    def test_calibration_ignores_recovery_and_conflict_cells(self):
        plain=C.calibrate(records(*curve_groups()))
        self.assertEqual(plain['severity_labels'],{'color':['desat-c0.2','desat-c0.05','desat-c0.025']})
        loud=C.calibrate(records(*curve_groups(dc={L:1. for L in COLOR},cd={L:0. for L in COLOR},recovery={COLOR[0]:50.,COLOR[-1]:-50.},
            interaction={COLOR[2]:99.},conflict={'normal':(1.,0.,1.),COLOR[0]:(40.,-3.,43.)})))
        for key in ('r_table','model_curves','label_medians','shared_grid','severity_labels','capability_calibrated','warnings'):
            self.assertEqual(plain[key],loud[key],key)

    def test_native_capability_floor_ceiling_and_monotonicity_flags(self):
        weak=group('m',.55,{COLOR[0]:.55});good=group('n',.9,{COLOR[0]:.9,COLOR[1]:.5,COLOR[2]:.45,COLOR[3]:.7})
        result=C.calibrate(records(weak,good));rows={(x['model'],x['label']):x for x in result['r_table']}
        self.assertTrue({'not_native_capable','native_ci_includes_chance'}<=set(rows[('m',COLOR[0])]['flags']))
        self.assertFalse(rows[('m',COLOR[0])]['eligible'])
        self.assertEqual((rows[('n',COLOR[0])]['r'],rows[('n',COLOR[1])]['r']),(1.,0.))
        self.assertIn('ceiling',rows[('n',COLOR[0])]['flags']);self.assertIn('floor',rows[('n',COLOR[1])]['flags'])
        self.assertTrue({'floor','below_chance'}<=set(rows[('n',COLOR[2])]['flags']));self.assertEqual(rows[('n',COLOR[3])]['flags'],[])
        self.assertTrue(all(x['eligible'] for (m,_),x in rows.items() if m=='n'))
        self.assertTrue(any('non-monotone' in w and COLOR[3] in w for w in result['warnings']))
        self.assertTrue(any('m: no native-capable' in w for w in result['warnings']))

    def test_selection_is_deterministic_ordered_and_ties_go_milder(self):
        recs=records(*curve_groups());first=C.calibrate(recs);shuffled=list(recs);random.Random(3).shuffle(shuffled)
        self.assertEqual(json.dumps(first,sort_keys=True),json.dumps(C.calibrate(shuffled),sort_keys=True))
        grid=first['shared_grid']['color'];self.assertEqual(grid['labels'],severity.ordered(grid['labels']))
        self.assertEqual(grid['targets'],[.75,.5,.25]);self.assertAlmostEqual(grid['total_abs_error'],.125)
        picked={e['model']:e for e in first['capability_calibrated']['entries']}
        self.assertEqual((picked['A']['label'],picked['B']['label']),('desat-c0.05','desat-c0.05'))
        self.assertAlmostEqual(picked['A']['interpolated_value'],.05);self.assertAlmostEqual(picked['B']['interpolated_value'],.1-.1/3)
        self.assertIn('different estimand',picked['B']['note'])
        cands=list(zip('abcde',[.8,.7,.5,.3,.2]))
        self.assertEqual([cands[i][0] for i in C.choose(cands,[.75,.5,.25])[1]],['a','c','d'])
        self.assertEqual(C.choose(list(zip('abcd',[.98,.5,.48,.46])),[.75,.5,.25])[1],(0,1,3))
        tie=C.calibrate(records(group('T',.9,{COLOR[0]:.5+.6*.4,COLOR[1]:.5+.4*.4})))
        self.assertEqual(tie['capability_calibrated']['entries'][0]['label'],COLOR[0])
        few=C.calibrate(records(group('T',.9,{COLOR[3]:.7})));self.assertEqual(few['shared_grid']['color']['targets'],[.5])
        self.assertTrue(any('1 eligible labels for 3 targets' in w for w in few['warnings']))

    def test_primary_readouts_chance_override_and_duplicate_inputs(self):
        direct=group('m',.9,{COLOR[0]:.9});cyclic=group('m',.9,{COLOR[0]:.5},fmt='label',mapping='cyclic');recs=records(direct,cyclic)
        mean=lambda result:result['model_curves'][0]['points'][0]['r_mean']
        self.assertEqual(mean(C.calibrate(recs)),.5);self.assertEqual(mean(C.calibrate(recs,primary=['label/cyclic'])),0.)
        self.assertEqual(mean(C.calibrate(recs,primary=['direct/none:score'])),1.)
        self.assertIsNone(mean(C.calibrate(recs,primary=['direct/none:generation'])))
        self.assertAlmostEqual(C.calibrate(recs,chance={'color':.25})['r_table'][1]['r'],.25/.65)
        with self.assertRaises(ValueError):C.calibrate(recs,primary=['direct'])
        other=dict(direct,execution_version='v15-other')
        with self.assertRaisesRegex(ValueError,'Two calibration inputs'):C.calibrate(records(direct,other))
        self.assertEqual(len(C.calibrate(records(direct,other),execution_versions={'v15-test'})['r_table']),1)
        with self.assertRaisesRegex(ValueError,'same curve cell'):curves.curve_rows(records(direct,direct))

    def test_benchmark_reader_shapes_primary_policy_and_free_form_chance(self):
        labels=[s['label'] for s in severity.grid('gqa_attribute','start')]
        def body(L,v,four):
            """score_followup_benchmark_v15 per_label entry: generic cell names; native-only labels lack recovery contrasts."""
            est={'clean__clean':stat(.6),'degraded__degraded':stat(v),'joint_degradation_loss':stat(.6-v)}
            if four:est.update({'clean__degraded':stat(.58),'degraded__clean':stat(v+.05),'later_recovery_with_main_degraded':stat(.05),
                'later_effect_with_main_clean':stat(.02),'main_quality_by_later_quality_interaction':stat(.03)})
            policies={'invalid_as_zero':est,'valid_cells_only':{**est,'degraded__degraded':stat(.59)}}
            return dict(label=L,cell_mode='four_cell' if four else 'native_only',policies={p:{'queries':20,'clusters':20,
                'metrics':{m:{'estimates':e,'unit_values':{}} for m in ('gqa_exact','gqa_case_punct_normalized')}} for p,e in policies.items()})
        doc=dict(schema='followup-benchmark-severity-analysis-v15',reports=[dict(model='m',model_revision='r',execution_version='v15',
            task='gqa_attribute',benchmark='gqa',primary_metric='gqa_exact',primary_policy='invalid_as_zero',labels=labels,curve=[dict(label=L) for L in labels],
            per_label={L:body(L,v,L!=labels[2]) for L,v in zip(labels,[.5,.3,.1])})])
        warnings=[];recs=list(C.benchmark_records(doc,'scorer.json',warnings.append));self.assertEqual(warnings,[])
        result=C.calibrate(recs);rows={x['label']:x for x in result['r_table']}
        # GQA calibration reads the named case+punct sensitivity by default (pre-v15 choice); strict stays selectable.
        self.assertEqual({x['measure'] for x in rows.values()},{'gqa_case_punct_normalized|invalid_as_zero'})
        strict=C.calibrate(recs,benchmark_metrics={'gqa_attribute':'gqa_exact'})
        self.assertEqual({x['measure'] for x in strict['r_table']},{'gqa_exact|invalid_as_zero'})
        self.assertEqual(result['rule']['benchmark_metrics']['per_task']['gqa_attribute'],'gqa_case_punct_normalized')
        self.assertEqual({x['chance'] for x in rows.values()},{0.});self.assertAlmostEqual(rows[labels[1]]['r'],.5)
        self.assertEqual(result['severity_labels'],{'gqa_attribute':labels})
        found=curves.curve_rows(recs);loss=[x for x in found if x['quantity']=='realized_native_degradation']
        self.assertEqual(len(loss),3*2*2);self.assertEqual({x['value_source'] for x in loss},{'copied'})
        self.assertEqual({x['severity_label'] for x in found if x['quantity']=='later_recovery'},set(labels[:2]))
        v14=dict(reports=[dict(model='m',model_revision='r',execution_version='v14',task='gqa_spatial',primary_metric='gqa_exact',primary_policy='invalid_as_zero',
            policies={'invalid_as_zero':{'metrics':{'gqa_exact':{'estimates':{'clean__clean':stat(.6),'degraded__degraded':stat(.59),
                'degraded__clean':stat(.6),'later_recovery_with_main_degraded':stat(.01)}}}}})])
        recs=list(C.benchmark_records(v14,'v14.json',warnings.append))
        self.assertEqual({(r['quantity'],r['severity_label']) for r in recs},{('native_clean',None),('joint_degraded','degraded'),
            ('main_degraded_later_clean','degraded'),('later_recovery','degraded')})
        default=C.calibrate(recs);self.assertEqual(default['r_table'],[])
        self.assertTrue(any('gqa_case_punct_normalized absent' in w for w in default['warnings']))
        row=C.calibrate(recs,benchmark_metrics={'gqa_spatial':'gqa_exact'})['r_table'][0]
        self.assertIn('unparsed_label',row['flags']);self.assertFalse(row['eligible'])
        self.assertIn('degraded',{x['severity_label'] for x in curves.curve_rows(recs)})
        self.assertEqual(list(C.benchmark_records({'something':1},'odd.json',warnings.append)),[])
        self.assertTrue(any('odd.json' in w for w in warnings))


class CurveTests(unittest.TestCase):
    def test_curves_keep_every_label_and_copy_values(self):
        ll={COLOR[0]:.9,COLOR[1]:.5,COLOR[2]:.3}
        g=group('m',.9,ll,recovery={COLOR[0]:-.2,COLOR[1]:.1,COLOR[2]:-.4},interaction={COLOR[2]:-.3},
            conflict={'normal':(1.,.2,.8),COLOR[0]:(.5,.3,.2),COLOR[1]:(-.4,.1,-.5),COLOR[2]:(0.,0.,0.)})
        recs=records(g);rows=curves.curve_rows(recs);copied=[x for x in rows if x['value_source']=='copied']
        self.assertEqual(len(copied),len(recs))
        for L in ll:
            for q in ('joint_degraded','later_recovery','recovery_interaction','realized_native_degradation','conflict_target_effect',
                      'conflict_control_effect','selectivity_R','delta_R_vs_normal'):
                self.assertTrue(any(x['severity_label']==L and x['quantity']==q for x in rows),(L,q))
        source={(e['estimand'],e['measure'],e['main_quality'],e['later_quality'],e['query_role']):e for e in g['estimand_table']}
        for x in copied:
            main,later=CELL[x['quantity']](x['severity_label']) if x['quantity'] in CELL else (x['severity_label'] or 'normal',None)
            e=source[(x['source_estimand'],x['measure'] if x['source_estimand'].startswith('recovery') else None,main,later,x['query_role'])]
            self.assertEqual((x['n'],x['mean'],x['se'],[x['ci_low'],x['ci_high']],x['planned'],x['used'],x['planned_units'],x['excluded']),
                (e['n'],e['mean'],e['se'],e['ci95_t'],e['planned'],e['used'],e['planned_units'],e['excluded']))
        derived=[x for x in rows if x['value_source']==curves.DERIVED]
        self.assertEqual(len(derived),2*len(ll));self.assertTrue(all(x['ci_low'] is None and x['n'] is None for x in derived))
        self.assertAlmostEqual(next(x['mean'] for x in derived if x['severity_label']==COLOR[2]),.6)
        cal=C.calibrate(recs);marked=curves.curve_rows(recs,cal);self.assertEqual(len(marked),len(rows))
        for x in marked:
            self.assertEqual(x['in_shared_grid'],None if x['severity_label'] is None else x['severity_label'] in cal['severity_labels']['color'])
        chosen=curves.calibrated_rows(marked,cal);label=cal['capability_calibrated']['entries'][0]['label']
        self.assertEqual(label,COLOR[0])  # r 1 and 0 tie around .5: the milder label
        self.assertTrue(chosen and all(x['severity_label']==label and x['estimand_note']==C.CALIBRATED_NOTE for x in chosen))


def material(unit,labels,role='changed',split='calibration'):
    sources={'a':f'{unit}/a.png','b':f'{unit}/b.png'};meta={k:dict(width=32,height=32,sha256=unit+k) for k in sources};specs={}
    for L in labels:
        sources[L]=f'{unit}/{L}.png';meta[L]=dict(width=32,height=32,sha256=unit+L)
        spec=dict(label=L,family='desaturate',value=severity.parse_label(L)['value'],reference_kind=None,reference_px=None,realized={},canvas=[32,32])
        spec['spec_id']=severity.spec_id(spec);specs[L]=spec
    return dict(id=f'{unit}-{role}',unit_id=unit,intervention_id=unit+'-e',task='color',query_role=role,query='Color?',choices=['red','blue'],
        option_labels=['(A)','(B)'],answer_index=0,answer_label='(A)',direction_a_index=0,direction_b_index=1,donor_roi_bbox_xyxy=[0,0,8,8],
        source_images=sources,source_image_metadata=meta,severity_sources={L:L for L in labels},degradations=specs,split=split,material_set='e2e')


def run_config(cells,families,labels=None):
    c=design.settings();c.update(execution_version='v15-e2e-'+cells,families=families,severity_sources='row',severity_labels=labels,
        recovery_cells=cells,main_sources={'normal':'a'},recovery_sources={'clean':'a'},readouts=[dict(response_format='direct',mapping='none')],generate_tokens=0)
    return c


def observe(models,rows,cfg,correct):
    """Fake score observations; correct(model, unit index, condition) sets the clean-reference outcome."""
    records=[];metas=[];units=sorted({r['unit_id'] for r in rows},key=lambda u:int(u[1:]))
    for model in models:
        metas.append(dict(run_id=f"{model}-{cfg['execution_version']}",model=dict(key=model,revision='r'),config=cfg,
            conditions_per_query={row['id']:len(design.conditions(row,cfg)) for row in rows}))
        for row in rows:
            for c in design.conditions(row,cfg):
                ok=correct(model,units.index(row['unit_id']),c)
                push=(1. if c['family']=='conflict' and c['later_endpoint']=='b' else 0.)*(2. if row['query_role']=='changed' else .5)
                records.append(dict(observation_id=f"{model}-{row['id']}-{c['name']}",run_id=metas[-1]['run_id'],id=row['id'],unit_id=row['unit_id'],
                    intervention_id=row['intervention_id'],task=row['task'],query_role=row['query_role'],kind='score',
                    readout=dict(response_format='direct',mapping='none'),condition=c,model_key=model,model_revision='r',execution_version=cfg['execution_version'],
                    result=dict(clean_reference_correct=int(ok),b_minus_a_log_odds=(-2. if ok else 2.)+push,scorable=True,truncated=False)))
    return records,metas


class EndToEndTests(unittest.TestCase):
    def test_design_analysis_calibrate_curves(self):
        rank={L:i for i,L in enumerate(COLOR)};counts={'p':[8,8,7,6,4,2],'q':[8,8,8,7,6,3]}
        rows=[material(f'c{i}',COLOR) for i in range(8)]
        cal_cfg=run_config('native_only',['recovery'])
        self.assertEqual({c['name'] for c in design.conditions(rows[0],cal_cfg)},{'recovery__clean__clean',*(f'recovery__{L}__{L}' for L in COLOR)})
        native=lambda model,i,c:c['severity_label'] is None or i<counts[model][rank[c['severity_label']]]
        recs,metas=observe(['p','q'],rows,cal_cfg,native);cal_doc=analyze(recs,materials=rows,run_metadata=metas)
        cal=C.calibrate(list(C.analysis_records(cal_doc,'cal.json')))
        rs={(x['model'],x['label']):x['r'] for x in cal['r_table']}
        self.assertEqual([rs[('p',L)] for L in COLOR],[1.,1.,.75,.5,0.,-.5]);self.assertEqual([rs[('q',L)] for L in COLOR],[1.,1.,1.,.75,.5,-.25])
        self.assertEqual(cal['severity_labels'],{'color':['desat-c0.1','desat-c0.05','desat-c0.025']})
        self.assertEqual({e['model']:e['label'] for e in cal['capability_calibrated']['entries']},{'p':'desat-c0.05','q':'desat-c0.025'})
        evaluation=[material(f'e{i}',COLOR,role,'evaluation') for i in range(6) for role in ('changed','unchanged')]
        eval_cfg=run_config('four_cell',['conflict','recovery'],cal['severity_labels'])
        names={c['name'] for c in design.conditions(evaluation[0],eval_cfg)}
        self.assertEqual(len(names),(1+3)*2+1+3*3);self.assertNotIn(f'recovery__{COLOR[0]}__{COLOR[0]}',names)
        def later_helps(model,i,c):
            L=c['severity_label']
            if L is None or c['main_quality']=='clean':return True
            return i<(5 if c.get('later_quality')=='clean' else 3)-(rank[L]>3)
        recs,metas=observe(['p','q'],evaluation,eval_cfg,later_helps);doc=analyze(recs,materials=evaluation,run_metadata=metas)
        source={(g['model'],e['estimand'],e['measure'],e['main_quality'],e['later_quality'],e['query_role']):e for g in doc['results'] for e in g['estimand_table']}
        records_=list(C.analysis_records(doc,'eval.json'));rows_=curves.curve_rows(records_,cal)
        self.assertEqual({x['severity_label'] for x in rows_}-{None},set(cal['severity_labels']['color']))
        self.assertTrue(all(x['in_shared_grid'] for x in rows_ if x['severity_label']))
        for x in rows_:
            if x['value_source']!='copied':continue
            main,later=CELL[x['quantity']](x['severity_label']) if x['quantity'] in CELL else (x['severity_label'] or 'normal',None)
            e=source[(x['model'],x['source_estimand'],x['measure'] if x['source_estimand'].startswith('recovery') else None,main,later,x['query_role'])]
            self.assertEqual((x['n'],x['mean'],x['planned'],x['used'],x['excluded']),(e['n'],e['mean'],e['planned'],e['used'],e['excluded']))
        benefit={(x['model'],x['severity_label']):x['mean'] for x in rows_ if x['quantity']=='later_recovery' and x['measure']=='accuracy_invalid_as_incorrect'}
        self.assertEqual(benefit[('p','desat-c0.1')],2/6);self.assertEqual(benefit[('p','desat-c0.025')],2/6)
        self.assertTrue(curves.calibrated_rows(rows_,cal))
        with tempfile.TemporaryDirectory() as t,contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
            t=Path(t);(t/'cal_analysis.json').write_text(json.dumps(cal_doc));(t/'eval_analysis.json').write_text(json.dumps(doc))
            self.assertEqual(C.main(['--analysis',str(t/'cal_analysis.json'),'--output',str(t/'calibration.json')]),0)
            written=json.loads((t/'calibration.json').read_text())
            self.assertEqual(written['severity_labels'],cal['severity_labels']);self.assertEqual(len(written['inputs'][0]['sha256']),64)
            self.assertEqual(curves.main(['--analysis',str(t/'eval_analysis.json'),'--calibration',str(t/'calibration.json'),'--output',str(t/'curves')]),0)
            md=(t/'curves'/'severity_curves.md').read_text()
            self.assertIn('## color',md);self.assertIn('Capability-calibrated labels: '+C.CALIBRATED_NOTE,md)
            self.assertEqual(len((t/'curves'/'severity_curves.csv').read_text().splitlines())-1,len(json.loads((t/'curves'/'severity_curves.json').read_text())['rows']))
            self.assertTrue((t/'curves'/'capability_calibrated.csv').exists())


if __name__=='__main__':unittest.main()
