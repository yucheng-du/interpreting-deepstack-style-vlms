"""Paired layout/permutation contrasts with complete scene/source-cluster denominators."""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import mean
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cloud_prep.portable_data import _guard_path
from cloud_prep.measurement_v8.mechanism_analysis import stats, finite_score, valid_generation, key
from cloud_prep.measurement_v9.multiobject_design import DESIGN_SCHEMA, PERMUTATIONS

BASE_GROUP = ('panel_family', 'geometry', 'response_format', 'main_strength', 'entry_group')
CELL_GROUP = BASE_GROUP + ('main_layout', 'token_permutation')


def summarize(rows, fields=CELL_GROUP):
    grouped=defaultdict(lambda:defaultdict(list))
    for r in rows:grouped[tuple(r[f] for f in fields)][r['unit_id']].append(r)
    result=[]
    for group,units in grouped.items():
        complete=[mean(r['effect'] for r in members) for members in units.values() if all(r['effect'] is not None for r in members)]
        result.append({**dict(zip(fields,group)), 'unit_key':'unit_id', 'planned_unit_n':len(units),
            'complete_unit_n':len(complete),'incomplete_unit_n':len(units)-len(complete),
            'planned_query_repeats':sum(len(m) for m in units.values()),
            'available_query_repeats':sum(r['effect'] is not None for m in units.values() for r in m),
            'statistics':stats(complete)})
    return result


def _records(records,expected,kind):
    out={}
    for row in records:
        k=key(row)
        if k in out:raise ValueError(f'Duplicate {kind} record; select one execution attempt')
        if k not in expected:raise ValueError(f'Unplanned {kind} record: {k}')
        spec=expected[k];q,c=spec['query'],spec['condition']
        if row.get('schema')!=DESIGN_SCHEMA:raise ValueError('Wrong scientific schema')
        for f in ('scene_id','unit_id','panel_family','geometry','target_object','direction_a_index','direction_b_index'):
            if row.get(f)!=q[f]:raise ValueError(f'Query identity differs: {f}')
        for f in ('role','main_layout','main_strength','entry_group','token_permutation','donor_assignment'):
            if row.get(f)!=c[f]:raise ValueError(f'Condition identity differs: {f}')
        out[k]=row
    return out


def _contrasts(effects):
    indexed=defaultdict(dict)
    for e in effects:
        k=tuple(e[f] for f in ('id','response_format','mapping','main_strength','entry_group'))
        indexed[k][(e['main_layout'],e['token_permutation'])]=e
    result=[]
    definitions={
        'M0_identity_minus_forward':[(1,'M0','identity'),(-1,'M0','cycle_forward')],
        'M0_identity_minus_inverse':[(1,'M0','identity'),(-1,'M0','cycle_inverse')],
        'M1_forward_minus_identity':[(1,'M1','cycle_forward'),(-1,'M1','identity')],
        'M1_forward_minus_inverse':[(1,'M1','cycle_forward'),(-1,'M1','cycle_inverse')],
        'layout_interaction_forward_vs_identity':[(1,'M1','cycle_forward'),(-1,'M1','identity'),(-1,'M0','cycle_forward'),(1,'M0','identity')],
        'M0_correct_minus_mean_incorrect':[(1,'M0','identity'),(-.5,'M0','cycle_forward'),(-.5,'M0','cycle_inverse')],
        'M1_correct_minus_mean_incorrect':[(1,'M1','cycle_forward'),(-.5,'M1','identity'),(-.5,'M1','cycle_inverse')]}
    for cells in indexed.values():
        e=next(iter(cells.values()))
        for name,terms in definitions.items():
            if any((m,p) not in cells for _,m,p in terms):raise ValueError('Plan lacks necessary cyclic-layout contrast')
            available=all(cells[(m,p)]['effect'] is not None for _,m,p in terms)
            result.append({**{f:e[f] for f in ('id','scene_id','unit_id','target_object','target_index','mapping')+BASE_GROUP},
                'contrast':name,'effect':sum(w*cells[(m,p)]['effect'] for w,m,p in terms) if available else None})
    return result


def geometry_contrasts(contrasts):
    """Only natural geometries share a source cluster; procedural scenes remain independent."""
    paired=defaultdict(dict)
    for r in contrasts:
        if r['panel_family']!='natural_recomposition':continue
        k=tuple(r[f] for f in ('unit_id','target_object','response_format','mapping','main_strength','entry_group','contrast'))
        if r['geometry'] in paired[k]:raise ValueError('Duplicate natural geometry/query contrast')
        paired[k][r['geometry']]=r
    out=[]
    for k,geos in paired.items():
        if set(geos)!={'three_columns','two_dimensional'}:continue
        a,b=geos['three_columns'],geos['two_dimensional']
        out.append({**a,'geometry':'two_dimensional_minus_three_columns',
            'effect':b['effect']-a['effect'] if a['effect'] is not None and b['effect'] is not None else None})
    return out


def analyze(records,generations=(),expected_plan=None):
    if not expected_plan or expected_plan.get('schema')!=DESIGN_SCHEMA:raise ValueError('Explicit multiobject execution plan required')
    plan=expected_plan;expected={};expected_gen={};pair_specs=defaultdict(dict)
    if len({q['id'] for q in plan['queries']})!=len(plan['queries']):raise ValueError('Duplicate planned query')
    for q in plan['queries']:
        for r in plan['readouts']:
            for c in plan['conditions']:
                k=(q['id'],r['response_format'],r['mapping'],c['name'])
                if k in expected:raise ValueError('Duplicate planned cell')
                expected[k]=dict(query=q,readout=r,condition=c)
                if q['generation_selected'] and c['generation']:expected_gen[k]=expected[k]
                if c['role']=='multiobject_pair':
                    pk=k[:3]+tuple(c[f] for f in ('main_strength','entry_group','main_layout','token_permutation'))
                    pair_specs[pk][c['donor_assignment']]=k
    if len(expected)!=plan['expected_score_rows'] or len(expected_gen)!=plan['expected_generation_rows']:raise ValueError('Plan counts disagree')
    scores=_records(records,expected,'score');gens=_records(generations,expected_gen,'generation')
    effects=[];behavior=[]
    for keys in pair_specs.values():
        if set(keys)!={'a','b'}:raise ValueError('Incomplete planned donor pair')
        spec=expected[keys['a']];q,c,r=spec['query'],spec['condition'],spec['readout']
        base={**{f:q[f] for f in ('id','scene_id','unit_id','source_cluster_id','panel_family','geometry','target_object','target_index')},
            **r,**{f:c[f] for f in ('main_strength','entry_group','main_layout','token_permutation','alignment')}}
        a,b=scores.get(keys['a']),scores.get(keys['b']);valid=finite_score(a) and finite_score(b)
        effects.append({**base,'effect':b['b_minus_a_log_odds']-a['b_minus_a_log_odds'] if valid else None,
            'missing_score_rows':sum(v is None for v in (a,b)),
            'nonfinite_score_rows':sum(v is not None and not finite_score(v) for v in (a,b))})
        if keys['a'] in expected_gen:
            ga,gb=gens.get(keys['a']),gens.get(keys['b']);ok=valid_generation(ga) and valid_generation(gb)
            target=('(A)','(B)')[q['direction_b_index']]
            effect=int(gb['original_prediction']==target)-int(ga['original_prediction']==target) if ok else None
            behavior.append({**base,'effect':effect,'both_valid':ok,
                'flip':ga['original_prediction']!=gb['original_prediction'] if ok else None,
                'toward_b':effect==1,'toward_a':effect==-1,
                'missing_generation_rows':sum(v is None for v in (ga,gb)),
                'invalid_generation_rows':sum(v is not None and not valid_generation(v) for v in (ga,gb)),
                'truncated_generation_rows':sum(bool(v and v.get('truncated')) for v in (ga,gb))})
    contrasts=_contrasts(effects);behavior_contrasts=_contrasts(behavior)
    natives=defaultdict(list);audits=[]
    for k,spec in expected.items():
        q,c,r=spec['query'],spec['condition'],spec['readout'];score=scores.get(k);gen=gens.get(k)
        if c['role'].startswith('native_'):
            gold_i=q['source_answer_indices'][c['main']];gold=('(A)','(B)')[gold_i] if gold_i is not None else None
            natives[(q['panel_family'],q['geometry'],r['response_format'],r['mapping'],c['name'])].append(dict(
                with_gold=gold is not None,score_present=finite_score(score),
                score_correct=finite_score(score) and score.get('original_prediction')==gold if gold else None,
                gen_planned=k in expected_gen,gen_present=gen is not None,gen_valid=valid_generation(gen),
                gen_correct=valid_generation(gen) and gen['original_prediction']==gold if gold else None))
        if c['role']=='self_patch':
            baseline=scores.get(k[:3]+(f'native_{c["main"]}',))
            audits.append(dict(id=q['id'],scene_id=q['scene_id'],unit_id=q['unit_id'],**r,
                condition=c['name'],delta_b_minus_a=score['b_minus_a_log_odds']-baseline['b_minus_a_log_odds']
                if finite_score(score) and finite_score(baseline) else None))
    native_summary=[]
    for group,m in natives.items():
        native_summary.append({**dict(zip(('panel_family','geometry','response_format','mapping','condition'),group)),
            'planned_rows':len(m),'planned_rows_with_gold':sum(x['with_gold'] for x in m),
            'score_rows_present':sum(x['score_present'] for x in m),'score_correct':sum(x['score_correct'] is True for x in m),
            'planned_generation_rows':sum(x['gen_planned'] for x in m),
            'planned_generation_rows_with_gold':sum(x['gen_planned'] and x['with_gold'] for x in m),
            'generation_rows_present':sum(x['gen_present'] for x in m),
            'generation_common_valid_with_gold':sum(x['gen_valid'] and x['with_gold'] for x in m),
            'generation_correct':sum(x['gen_correct'] is True for x in m)})
    grouped=defaultdict(list)
    for b in behavior:grouped[tuple(b[f] for f in CELL_GROUP+('mapping',))].append(b)
    behavioral=[]
    for group,m in grouped.items():
        common=sum(x['both_valid'] for x in m);flips=sum(x['flip'] is True for x in m)
        behavioral.append({**dict(zip(CELL_GROUP+('mapping',),group)),'planned_pairs':len(m),'common_valid_pairs':common,
            'missing_pairs':sum(x['missing_generation_rows']>0 for x in m),'invalid_pairs':sum(x['invalid_generation_rows']>0 for x in m),
            'truncated_pairs':sum(x['truncated_generation_rows']>0 for x in m),'flips':flips,
            'flips_over_planned_lower_bound':flips/len(m),'flip_rate_common_valid':flips/common if common else None,
            'toward_b':sum(x['toward_b'] for x in m),'toward_a':sum(x['toward_a'] for x in m)})
    contrast_fields=BASE_GROUP+('contrast',)
    return dict(schema=DESIGN_SCHEMA,unit_key='unit_id',coverage=dict(planned_score_rows=len(expected),present_score_rows=len(scores),
        missing_score_rows=len(expected)-len(scores),nonfinite_score_rows=sum(not finite_score(s) for s in scores.values()),
        planned_generation_rows=len(expected_gen),present_generation_rows=len(gens),missing_generation_rows=len(expected_gen)-len(gens),
        invalid_generation_rows=sum(not valid_generation(g) for g in gens.values()),
        truncated_generation_rows=sum(bool(g.get('truncated')) for g in gens.values()),
        planned_units=len({q['unit_id'] for q in plan['queries']}),planned_scenes=len({q['scene_id'] for q in plan['queries']}),
        planned_query_pairs=len(effects),complete_query_pairs=sum(e['effect'] is not None for e in effects)),
        query_effects=effects,query_contrasts=contrasts,cell_summaries=summarize(effects),
        per_mapping_cell_summaries=summarize(effects,CELL_GROUP+('mapping',)),
        contrast_summaries=summarize(contrasts,contrast_fields),
        per_mapping_contrast_summaries=summarize(contrasts,contrast_fields+('mapping',)),
        natural_paired_geometry_contrasts=summarize(geometry_contrasts(contrasts),contrast_fields),
        behavior_pairs=behavior,behavior_summary=behavioral,behavior_effect_summaries=summarize(behavior),
        behavior_contrast_summaries=summarize(behavior_contrasts,contrast_fields),
        behavior_natural_paired_geometry_contrasts=summarize(geometry_contrasts(behavior_contrasts),contrast_fields),
        native_accuracy=native_summary,self_patch_audits=audits,
        interpretation='Exploratory contrasts paired within query/mapping before complete unit means. Synthetic scenes are independent; natural geometries share six source photos and are paired by source cluster. Families, geometries, models and direct/label scales stay separate. Wholly absent queries remain planned; missing or invalid values never equal zero. Three-cycle alignment extends beyond two-slot swaps but does not isolate a unique mechanism. Natural b substitutions change instance/context as well as color; only synthetic b is an exact attribute edit. Technical and native capability checks never select the effect sample.')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--scores',type=Path,nargs='+',required=True);p.add_argument('--generations',type=Path,nargs='*',default=[])
    p.add_argument('--output',type=Path,required=True);a=p.parse_args(argv)
    load=lambda paths:[json.loads(s) for path in paths for s in path.read_text().splitlines() if s.strip()]
    result=analyze(load(a.scores),load(a.generations),json.loads(a.plan.read_text()))
    output=_guard_path(a.output,write=True);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(output=str(output),**result['coverage']),indent=2))


if __name__=='__main__':main()
