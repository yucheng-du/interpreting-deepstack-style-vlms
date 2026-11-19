"""Scalar metadata fixtures for existing canonical-anchor and native-dose checks."""
from copy import deepcopy
from itertools import product
import unittest

from cloud_prep.measurement_v8.model_registry import get_model
from cloud_prep.measurement_v12 import p1_design as design
from cloud_prep.measurement_v12 import v11_v2_dose_analysis as accepted
from cloud_prep.measurement_v12.v11_v2_dose_adapter import native_dose,resolve_scale
from cloud_prep.measurement_v12.p1_dose_validation import validate_dose_records


def fixture(kind='score'):
    queries=[]
    for target in ('changed','unchanged'):
        q={k:None for k in design.QUERY_FIELDS}
        q.update(id='photo-edit-'+target,scene_id='photo',unit_id='photo',intervention_id='photo-edit',
            query_type='identity',query_target=target,query_role=target,direction_a_index=0,direction_b_index=1,
            query_sha256='sha256-query-'+target,generation_selected=True)
        queries.append(q)
    cells=design.conditions(); readout={'response_format':'direct','mapping':'none'}
    model=get_model('qwen8b')
    plan=dict(schema=design.DESIGN_SCHEMA,queries=queries,conditions=cells,model=model,
              calibration_anchors={'photo-edit':queries[0]['id']},synthetic_fixture=True)
    def calibration(c):
        ratios=[1.] if c['changed_entry']=='main' else [1.,2.,2.]
        return dict(calibration_id=c['calibration_id'],intervention_id='photo-edit',changed_entry=c['changed_entry'],
            anchor_query_id=queries[0]['id'],anchor_query_sha256=queries[0]['query_sha256'],
            anchor_prompt_sha256='canonical-direct-prompt',visual_signature='visual-'+c['calibration_id'],
            context_signature='context-'+c['main_layout']+'-'+str(c['main_strength']),model_revision=model['revision'],
            entries=[dict(entry='e'+str(i),source_delta={'rms':x},baseline_state={'rms':1.}) for i,x in enumerate(ratios)])
    expected={}; rows=[]
    for q,c in product(queries,cells):
        if kind=='generation' and not c['generation']:continue
        key=q['id'],readout['response_format'],readout['mapping'],c['name']
        expected[key]=q,readout,c
        row={**q,**readout,'schema':design.DESIGN_SCHEMA,'condition_spec':c,'condition':c['name'],
             'model_key':model['key'],'source_a_original_choice':'(A)','source_b_original_choice':'(B)',
             'kind':kind,'b_minus_a_log_odds':0.,'scores':{'red':0.,'blue':0.},
             'scorable':True,'truncated':False,'original_prediction':'(A)'}
        if c['role']=='dose_pair':
            own=calibration(c); other=calibration(design.opposite_condition(c))
            scale=resolve_scale(c,own,other)
            k=0. if c['donor_assignment']=='a' else scale['k']
            row['dose_calibration']={**own,'opposite_calibration':other}
            row['dose_metadata']={**scale,'condition':c['name'],'calibration_id':c['calibration_id'],
                'donor_assignment':c['donor_assignment'],'effective_k':k,
                'anchor_query_id':own['anchor_query_id'],'anchor_prompt_sha256':own['anchor_prompt_sha256'],
                'visual_signature':own['visual_signature'],
                'matching':{**scale,'status':'raw' if c['dose_mode']=='raw' else 'matched'},
                'realized_baseline_state_by_metric':{m:native_dose(own)[m]*k for m in ('rss','l1')}}
        rows.append(row)
    return rows,expected,plan


class P1DoseValidationTests(unittest.TestCase):
    def test_existing_raw_and_matched_controls_pass_native_self_need_no_dose_metadata(self):
        for kind in ('score','generation'):
            rows,expected,plan=fixture(kind)
            result=validate_dose_records(rows,expected,kind,plan)
            dose_n=sum(spec[2]['role']=='dose_pair' for spec in expected.values())
            self.assertEqual(len(result),dose_n)
            self.assertLess(len(result),len(rows))
            self.assertTrue(all(row['condition_spec']['role']=='dose_pair' for row in result.values()))

    def test_wrong_canonical_anchor_is_rejected_and_globals_restored(self):
        rows,expected,plan=fixture()
        bad=next(r for r in rows if r['condition_spec'].get('dose_mode')=='baseline_state')
        bad['dose_calibration']['anchor_query_id']='other-query'
        fields=('DESIGN_SCHEMA','QUERY_FIELDS','opposite_condition','anchor_query_sha256')
        original=tuple(getattr(accepted,f) for f in fields)
        with self.assertRaisesRegex(ValueError,'Wrong paired calibration anchor'):
            validate_dose_records(rows,expected,'score',plan)
        self.assertEqual(tuple(getattr(accepted,f) for f in fields),original)

    def test_changed_raw_k_or_matched_k_is_rejected(self):
        for mode,pattern in (('raw','Raw k changed'),('baseline_state','Effective k differs')):
            rows,expected,plan=fixture()
            bad=next(r for r in rows if r['condition_spec'].get('dose_mode')==mode and
                     r['condition_spec'].get('donor_assignment')=='b' and r['condition_spec'].get('dose_value')==1.)
            bad['dose_metadata']['effective_k']=9.
            with self.assertRaisesRegex(ValueError,pattern):
                validate_dose_records(rows,expected,'score',plan)

    def test_language_hash_and_target_binding_are_checked(self):
        rows,expected,plan=fixture()
        bad=next(r for r in rows if r['condition_spec'].get('dose_mode')=='baseline_state')
        bad['query_sha256']='wrong-query-text'
        with self.assertRaisesRegex(ValueError,'language hash differs'):
            validate_dose_records(rows,expected,'score',plan)
        rows,expected,plan=fixture()
        bad=next(r for r in rows if r['condition_spec'].get('dose_mode')=='baseline_state')
        bad['dose_metadata']['resolved_target']=999.
        with self.assertRaisesRegex(ValueError,'Resolved opposite target'):
            validate_dose_records(rows,expected,'score',plan)


if __name__=='__main__':unittest.main()
