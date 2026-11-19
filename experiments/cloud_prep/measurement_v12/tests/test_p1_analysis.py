"""Scientific estimand fixtures: same-target pairs, clustering and invalid outputs."""
import copy
from itertools import product
import unittest
from cloud_prep.measurement_v12 import p1_design as d, p1_analysis as a


def fixture(kind='score'):
    queries=[]
    for photo,edit,target in product(('p0','p1'),(0,1),('changed','unchanged')):
        q={k:None for k in d.QUERY_FIELDS}
        q.update(id=f'{photo}-e{edit}-{target}',scene_id=photo,unit_id=photo,
                 intervention_id=f'{photo}-e{edit}',query_role=target,
                 query_type='identity',query_target=target,generation_selected=True)
        queries.append(q)
    cells=d.conditions();readouts=[{'response_format':'direct','mapping':'identity'}]
    plan={'schema':d.DESIGN_SCHEMA,'conditions':cells,'queries':queries,'readouts':readouts,
          'model':{'key':'fixture'},'synthetic_fixture':True,'expected_score_rows':len(queries)*len(cells),
          'expected_generation_rows':len(queries)*sum(c['generation'] for c in cells)}
    rows=[]
    for q,r,c in product(queries,readouts,cells):
        if kind=='generation' and not c['generation']:continue
        row={**q,**r,'schema':d.DESIGN_SCHEMA,'condition_spec':c,'condition':c['name'],
             'model_key':'fixture','source_a_original_choice':'(A)','source_b_original_choice':'(B)',
             'scorable':True,'truncated':False,'native_correct':True,
             'original_prediction':'(A)','b_minus_a_log_odds':0.}
        if c['role']=='dose_pair' and c['donor_assignment']=='b' and c['dose_value']!=0:
            # Main-native: 3 - 1.5. Later-native: 4 - 2. Raw: 3 - 2.
            scale=({'main':3.,'later':2.} if c['dose_mode']=='raw' else {'main':4.,'later':1.5})[c['changed_entry']]
            row['b_minus_a_log_odds']=scale*(2 if q['query_role']=='changed' else 1)
            if q['query_role']=='changed':row['original_prediction']='(B)'
        rows.append(row)
    return rows,plan


def select(result,endpoint,**keys):
    candidates=[s for s in result['summaries'] if s['endpoint']==endpoint and
                s['main_layout']=='M0' and s['main_strength']==1. and
                all(s.get(k)==v for k,v in keys.items())]
    if len(candidates)!=1:raise AssertionError(candidates)
    return candidates[0]


class P1Analysis(unittest.TestCase):
    def test_same_target_not_same_role_matching(self):
        records,plan=fixture();r=a.analyze(records,plan)
        for metric in ('rss','l1'):
            self.assertEqual(select(r,'R_main_minus_later',target='main_native',dose_metric=metric)['mean'],1.5)
            self.assertEqual(select(r,'R_main_minus_later',target='later_native',dose_metric=metric)['mean'],2.)
        raw=select(r,'R_main_minus_later',target='raw')
        self.assertEqual(raw['mean'],1.);self.assertEqual(raw['complete_photos'],2)

    def test_both_roles_selective_with_zero_role_difference_is_retained(self):
        records,plan=fixture('generation');r=a.analyze(records,plan,'generation')
        for role in ('main','later'):
            self.assertEqual(select(r,'R',changed_entry=role,dose_mode='raw',dose_value=1.)['mean'],1.)
        self.assertEqual(select(r,'R_main_minus_later',target='raw')['mean'],0.)

    def test_invalid_edit_excludes_photo_from_that_paired_endpoint_only(self):
        records,plan=fixture('generation')
        broken=next(x for x in records if x['id']=='p0-e0-changed' and
            x['condition_spec'].get('changed_entry')=='later' and
            x['condition_spec'].get('dose_mode')=='baseline_state' and
            x['condition_spec'].get('dose_metric')=='rss' and
            x['condition_spec'].get('donor_assignment')=='b')
        broken['truncated']=True  # A parseable answer is nevertheless invalid.
        r=a.analyze(records,plan,'generation')
        paired=select(r,'R_main_minus_later',target='main_native',dose_metric='rss')
        self.assertEqual((paired['complete_photos'],paired['incomplete_photos']),(1,1))
        self.assertEqual(paired['mean'],0.)
        self.assertEqual(select(r,'R_main_minus_later',target='raw')['complete_photos'],2)

    def test_unmatched_numeric_and_wrong_model_are_not_silently_accepted(self):
        records,plan=fixture()
        broken=next(x for x in records if x['id']=='p0-e0-changed' and
            x['condition_spec'].get('changed_entry')=='later' and
            x['condition_spec'].get('dose_mode')=='baseline_state' and
            x['condition_spec'].get('dose_metric')=='rss' and
            x['condition_spec'].get('donor_assignment')=='b')
        broken['protocol_status']='unmatched_dose'
        r=a.analyze(records,plan)
        self.assertEqual(select(r,'R_main_minus_later',target='main_native',dose_metric='rss')['complete_photos'],1)
        records[0]['model_key']='different'
        with self.assertRaisesRegex(ValueError,'Wrong model'):a.analyze(records,plan)


if __name__=='__main__':unittest.main()
