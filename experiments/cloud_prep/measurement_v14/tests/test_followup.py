"""Small CPU tests of causal pairing, fixed values and incremental identities."""
import copy
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

from cloud_prep.measurement_v14 import design
from cloud_prep.measurement_v14.adapter import compose_values
from cloud_prep.measurement_v14.analysis import analyze, summary, t_quantile, read_jsonl
from cloud_prep.measurement_v14.runner import completed_rows


class ListOps:
    def clone(self,a):return copy.deepcopy(a)
    def shape(self,a):return (len(a),len(a[0]))
    def select(self,a,ids):return [a[i] for i in ids]
    def diff(self,b,a):return [[x-y for x,y in zip(xx,yy)] for xx,yy in zip(b,a)]
    def mix(self,a,b,ids,k):
        result=self.clone(a)
        for i in ids:result[i]=[x+k*(y-x) for x,y in zip(a[i],b[i])]
        return result
    def measure(self,a):
        flat=[x for row in a for x in row];norm=math.sqrt(sum(x*x for x in flat))
        return dict(norm=norm,rms=norm/math.sqrt(len(flat)),elements=len(flat))


def row():
    return dict(id='q',unit_id='photo',intervention_id='edit',task='color',query_role='changed',
        query='Color?',choices=['red','blue'],option_labels=['(A)','(B)'],answer_index=0,answer_label='(A)',
        direction_a_index=0,direction_b_index=1,donor_roi_bbox_xyxy=[0,0,8,8],
        source_images={'a':'a.png','b':'b.png','degraded_a':'d.png'},
        source_image_metadata={k:dict(width=32,height=32,sha256=k) for k in ('a','b','degraded_a')})


class FixedVisualTests(unittest.TestCase):
    def test_all_later_including_non_roi_fixed_across_main(self):
        a={'main':[[1.],[2.]],'later0':[[10.],[20.]],'later1':[[30.],[40.]]}
        b={'main':[[3.],[4.]],'later0':[[11.],[27.]],'later1':[[31.],[46.]]}
        weak={k:[[999.],[888.]] for k in a};mask={'later0':[0],'later1':[1]}
        c=dict(entries='all_later',t=1.,k=2.)
        before=copy.deepcopy((a,b,weak))
        normal,meta=compose_values(a,a,b,mask,c,'main',ListOps())
        degraded,_=compose_values(weak,a,b,mask,c,'main',ListOps())
        self.assertEqual(normal['later0'],degraded['later0']);self.assertEqual(normal['later1'],degraded['later1'])
        self.assertEqual(normal['later0'],[[12.],[20.]])
        self.assertNotEqual(normal['main'],degraded['main']);self.assertEqual((a,b,weak),before)

    def test_conditional_marginal_does_not_rescale_remaining_entries(self):
        a={'layer0':[[0.]],'layer15':[[1.]],'layer18':[[2.]],'layer21':[[3.]]}
        b={k:[[v[0][0]+2.]] for k,v in a.items()};m={k:[0] for k in a if k!='layer0'}
        c=dict(entries=['layer15','layer18'],t=1.,k={'layer15':.5,'layer18':2.,'layer21':3.})
        complement,_=compose_values(a,a,b,m,c,'layer0',ListOps())
        joint,_=compose_values(a,a,b,m,{**c,'entries':['layer15','layer18','layer21']},'layer0',ListOps())
        self.assertEqual(complement['layer15'],joint['layer15']);self.assertEqual(complement['layer18'],joint['layer18'])
        self.assertEqual(complement['layer21'],a['layer21']);self.assertEqual(joint['layer21'],[[9.]])

    def test_t0_and_unselected_are_baseline(self):
        a={'main':[[0.]],'later0':[[1.]],'later1':[[2.]]};b={k:[[9.]] for k in a}
        c=dict(entries=['later0'],t=0.,k=4.)
        got,_=compose_values(a,a,b,{'later0':[0],'later1':[0]},c,'main',ListOps())
        self.assertEqual(got,a)

    def test_unknown_entry_rejected(self):
        a={'main':[[0.]],'later0':[[1.]]}
        with self.assertRaises(ValueError):compose_values(a,a,a,{'later0':[0]},dict(entries=['layer21'],t=1,k=1),'main',ListOps())


class DesignTests(unittest.TestCase):
    def test_fixed_contract_and_separate_recovery_factorial(self):
        c=design.settings();cc=design.conditions(row(),c)
        self.assertEqual(len(cc),8)
        conflict=[x for x in cc if x['family']=='conflict']
        self.assertEqual({(x['main_quality'],x['later_endpoint']) for x in conflict},{('normal','a'),('normal','b'),('degraded','a'),('degraded','b')})
        self.assertEqual({x['later_base'] for x in conflict},{'a'})
        self.assertEqual({x['later_donor'] for x in conflict},{'b'})
        recovery=[x for x in cc if x['family']=='recovery']
        self.assertEqual({(x['main_quality'],x['later_quality']) for x in recovery},{('clean','clean'),('clean','degraded'),('degraded','clean'),('degraded','degraded')})

    def test_freeform_recovery_needs_neither_b_nor_roi(self):
        r=row();r.pop('choices');r.pop('donor_roi_bbox_xyxy');r['source_images'].pop('b')
        c=design.settings();cc=design.conditions(r,c)
        self.assertEqual(len(cc),4);self.assertEqual(design.readouts(r,c),[dict(response_format='freeform',mapping='none')])
        plan=design.execution_plan([r],{'key':'x'},c)
        self.assertEqual(plan['expected_logical_rows'],{'score':0,'generation':4})

    def test_incremental_key_does_not_freeze_other_conditions_or_paths(self):
        r=row();c=design.settings();condition=design.conditions(r,c)[0];ro=c['readouts'][0];m={'key':'m','revision':'r'}
        key=design.observation_key(r,condition,ro,'score',m,c)
        moved=copy.deepcopy(r);moved['source_images']={k:'/cloud/'+p for k,p in r['source_images'].items()}
        c['conditions'].append({'name':'not_related'})
        self.assertEqual(key,design.observation_key(moved,condition,ro,'score',m,c))
        c['execution_version']='v2'
        self.assertNotEqual(key,design.observation_key(r,condition,ro,'score',m,c))

    def test_resume_preserves_completed_invalid_observation(self):
        with tempfile.TemporaryDirectory() as t:
            path=Path(t)/'r.jsonl';r={'observation_id':'x','result':{'scorable':False}}
            path.write_text(json.dumps(r)+'\n');self.assertEqual(completed_rows(path),{'x':r})
            path.write_text((json.dumps(r)+'\n')*2)
            with self.assertRaises(ValueError):completed_rows(path)

    def test_resume_partial_tail_not_silently_dropped(self):
        with tempfile.TemporaryDirectory() as t:
            path=Path(t)/'r.jsonl';path.write_text('{"observation_id":')
            with self.assertRaisesRegex(ValueError,'Incomplete JSONL'):completed_rows(path)


def observation(i,unit,iid,role,quality,endpoint,value):
    return dict(observation_id=str(i),id=iid+role,unit_id=unit,intervention_id=iid,task='t',kind='score',
        readout=dict(response_format='direct',mapping='none'),model_key='m',model_revision='r',execution_version='v',query_role=role,
        condition=dict(name=f'{quality}-{endpoint}',family='conflict',main_quality=quality,later_endpoint=endpoint),
        result={'b_minus_a_log_odds':value})


class PairedAnalysisTests(unittest.TestCase):
    def recovery_fixture(self):
        material=row();material.update(id='echanged',unit_id='one',intervention_id='e',task='t')
        cfg=design.settings();cfg.update(execution_version='v',families=['recovery'],
            readouts=[dict(response_format='direct',mapping='none')])
        meta=dict(run_id='run',model=dict(key='m',revision='r'),config=cfg,conditions_per_query={material['id']:4})
        records=[]
        for i,(later,correct) in enumerate([('clean',1),('degraded',0)]):
            r=observation(i,'one','e','changed','degraded','a',0);r.update(kind='generation',run_id='run')
            r['condition']=next(c for c in design.conditions(material,cfg)
                                if c['main_quality']=='degraded' and c['later_quality']==later)
            r['result']=dict(clean_reference_correct=correct,scorable=True,truncated=False);records.append(r)
        return material,cfg,meta,records

    def test_two_cells_restore_without_native_or_interaction_cells(self):
        material,cfg,meta,records=self.recovery_fixture()
        r=analyze(records,materials=[material],run_metadata=[meta])['results'][0]
        restore=r['recovery_clean_later_benefit_with_degraded_main']
        self.assertEqual((restore['n'],restore['mean']),(1,1))
        interaction=r['recovery_quality_interaction']
        self.assertEqual(interaction['n'],0)
        self.assertEqual(interaction['denominators']['planned_queries'],1)
        self.assertEqual(interaction['denominators']['excluded']['missing'],1)
        direct=analyze(records,config=cfg)['results'][0]
        self.assertEqual(direct['recovery_clean_later_benefit_with_degraded_main']['mean'],1)
        with self.assertRaises(ValueError):analyze(records)

    def test_missing_and_invalid_restoration_denominators(self):
        material,cfg,meta,records=self.recovery_fixture()
        records[0]['result'].update(clean_reference_correct=None,scorable=False)
        r=analyze(records,materials=[material],run_metadata=[meta])['results'][0]
        self.assertEqual(r['recovery_clean_later_benefit_with_degraded_main']['denominators']['excluded']['invalid'],1)
        r=analyze(records[1:],materials=[material],run_metadata=[meta])['results'][0]
        self.assertEqual(r['recovery_clean_later_benefit_with_degraded_main']['denominators']['excluded']['missing'],1)

    def test_multiple_queries_per_role_are_averaged_order_independently(self):
        records=[]
        for q,role,effect in [('q1','changed',4),('q2','changed',2),('c','unchanged',1)]:
            for endpoint,val in [('a',0),('b',effect)]:
                r=observation(len(records),'p','e',role,'normal',endpoint,val);r['id']=q;records.append(r)
        a=analyze(records)['results'][0]['selectivity_R']
        b=analyze(list(reversed(records)))['results'][0]['selectivity_R']
        self.assertEqual(a,b);self.assertEqual(a[0]['mean'],2)

    def test_student_t_intervals_and_degenerate_samples(self):
        self.assertAlmostEqual(t_quantile(1),12.706204736,places=7)
        self.assertAlmostEqual(t_quantile(31),2.039513446,places=8)
        constant=summary([1,1]);self.assertTrue(constant['se_zero']);self.assertIsNone(constant['ci95_t'])
        empty=summary([]);self.assertEqual([empty[k] for k in ('positive','zero','negative')],[0,0,0])

    def test_unicode_separators_are_not_jsonl_record_boundaries(self):
        with tempfile.TemporaryDirectory() as t:
            path=Path(t)/'rows.jsonl';text='a\u2028b\u2029c\u0085d'
            path.write_text(json.dumps({'text':text},ensure_ascii=False)+'\n'+ '{"partial":')
            files=[];self.assertEqual(list(read_jsonl(path,files)),[{'text':text}])
            self.assertTrue(files[0]['incomplete_final_line'])
            path.write_text('{broken}\n')
            with self.assertRaises(ValueError):list(read_jsonl(path,[]))

    def test_signed_controls_strata_and_excluded_component_denominators(self):
        records=[];materials=[]
        for unit,reference,control in [('a',0,1.),('b',1,-1.)]:
            for role,effect in [('changed',2.),('unchanged',control)]:
                m=row();answer=reference if role=='unchanged' else 1
                m.update(id=unit+'-'+role,unit_id=unit,intervention_id=unit,task='t',query_role=role,
                         answer_index=answer,answer_label=m['option_labels'][answer]);materials.append(m)
                for quality,scale in [('normal',.5),('degraded',1.)]:
                    for endpoint,value in [('a',0.),('b',effect*scale)]:
                        r=observation(len(records),unit,unit,role,quality,endpoint,value)
                        r['id']=m['id'];records.append(r)
        g=analyze(records,materials=materials)['results'][0]
        signed={x['main_quality']:x for x in g['conflict_control_effect_toward_clean_reference']}
        self.assertEqual(signed['normal']['unit_values'],{'a':-.5,'b':-.5})
        self.assertEqual(signed['degraded']['unit_values'],{'a':-1.,'b':-1.})
        self.assertEqual(g['conflict_control_toward_reference_interactions'][0]['unit_values'],{'a':-.5,'b':-.5})
        strata={(x['main_quality'],x['control_alignment']):x for x in g['selectivity_R_by_control_alignment']}
        for label,expected in [('a_aligned',1.),('b_aligned',3.)]:
            s=strata[('degraded',label)];self.assertEqual(s['mean'],expected)
            self.assertEqual(s['target_component']['mean'],2.)
            self.assertEqual(s['control_component_toward_clean_reference']['mean'],-1.)
        self.assertEqual({x['control_alignment']:x['mean'] for x in g['selectivity_interaction_by_control_alignment']},
                         {'a_aligned':.5,'b_aligned':1.5})
        only_b=analyze(records,materials=materials,unit_ids=['b'])['results'][0]
        self.assertEqual(only_b['control_alignment']['flag'],'all controls b-aligned')
        for r in records:
            if (r['unit_id']=='a' and r['query_role']=='changed' and r['condition']['main_quality']=='degraded'
                    and r['condition']['later_endpoint']=='b'):r['result']['b_minus_a_log_odds']=None
        g=analyze(records,materials=materials)['results'][0]
        s=next(x for x in g['selectivity_R_by_control_alignment']
               if x['main_quality']=='degraded' and x['control_alignment']=='a_aligned')
        for key in ('target_component','control_component','control_component_toward_clean_reference'):
            d=s[key]['denominators'];self.assertEqual(d['planned_interventions'],1)
            self.assertEqual(d['interventions_used'],0);self.assertEqual(d['excluded']['invalid'],1)

    def test_query_repeats_do_not_inflate_unit_count(self):
        records=[]
        for unit in ['one','two']:
            for edit in ['left','right']:
                for role in ['changed','unchanged']:
                    for quality in ['normal','degraded']:
                        for endpoint in ['a','b']:
                            val=0 if endpoint=='a' or role=='unchanged' else (1 if quality=='normal' else 3)
                            records.append(observation(len(records),unit,edit,role,quality,endpoint,val))
        result=analyze(records)['results'][0]
        self.assertEqual(result['selectivity_interaction'][0]['n'],2)
        self.assertEqual(result['selectivity_interaction'][0]['mean'],2)
        self.assertEqual(result['conflict_query_interactions'][0]['n'],2)

    def test_missing_endpoint_is_not_zero(self):
        records=[observation(0,'one','e','changed','normal','a',0),observation(1,'one','e','changed','normal','b',None)]
        result=analyze(records)['results'][0]
        self.assertEqual(result['conflict_query_effects'],[]);self.assertEqual(result['incomplete_endpoint_pairs'],1)

    def test_recovery_repair_cannot_silently_override_same_cell(self):
        a=observation(0,'p','e','changed','normal','a',0)
        a['condition']=dict(name='clean',family='recovery',main_quality='clean',later_quality='clean')
        a['result']={'clean_reference_correct':0};b=copy.deepcopy(a);b['observation_id']='1';b['result']={'clean_reference_correct':1}
        with self.assertRaisesRegex(ValueError,'Ambiguous'):analyze([a,b])

    def test_custom_repair_cannot_silently_override_same_cell(self):
        a=observation(0,'p','e','changed','normal','a',0);a['condition']=dict(name='joint',family='custom')
        b=copy.deepcopy(a);b['observation_id']='1'
        with self.assertRaisesRegex(ValueError,'Ambiguous'):analyze([a,b],[dict(name='M',positive='joint',negative='complement')])

    def test_custom_conditional_marginal_paired_before_unit_summary(self):
        records=[]
        for role in ['changed','unchanged']:
            for name,value in [('joint',3 if role=='changed' else .5),('complement',1 if role=='changed' else .25)]:
                r=observation(len(records),'photo','edit',role,'normal','a',value)
                r['condition']={'name':name,'family':'custom'};records.append(r)
        result=analyze(records,[{'name':'M21','positive':'joint','negative':'complement'}])['results'][0]['custom_paired_comparisons'][0]
        self.assertEqual(result['target_minus_control']['n'],1);self.assertEqual(result['target_minus_control']['mean'],1.75)


if __name__=='__main__':unittest.main()
