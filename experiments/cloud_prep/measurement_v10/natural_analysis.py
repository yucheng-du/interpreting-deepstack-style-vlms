"""Fixed-direction natural-scene paired effects, with source-photo clustering.

Unchanged-query contrasts use only the explicitly declared paired subset; absent
controls are never zeros. Missing execution cells remain in planned denominators.
"""
from collections import defaultdict
import argparse
import json
from pathlib import Path
from cloud_prep.measurement_v8.mechanism_analysis import finite_score,valid_generation,key,summarize
from cloud_prep.measurement_v10.natural_design import DESIGN_SCHEMA
from cloud_prep.measurement_v10.natural_data import validate_query_groups
from cloud_prep.portable_data import _guard_path

GROUP=('response_format','main_strength','main_layout','query_role')


def _index(records,expected,kind):
    indexed={};pixels={}
    for row in records:
        k=key(row)
        if k in indexed:raise ValueError(f'Duplicate {kind} cell')
        if k not in expected:raise ValueError(f'Unplanned {kind} cell')
        q,c,_=expected[k]
        if row.get('schema')!=DESIGN_SCHEMA:raise ValueError('Scientific schema mismatch')
        for f in ('scene_id','intervention_id','query_role','changed_object','queried_object','direction_a_index','direction_b_index','visual_source_signature'):
            if row.get(f)!=q[f]:raise ValueError(f'Query identity mismatch: {f}')
        if row.get('main_source')!=c['main'] or row.get('condition_spec')!=c:raise ValueError('Full source/condition mismatch')
        for f in ('role','main_layout','main_strength','entry_group','region','donor_source','donor_assignment','varying_entry','token_permutation'):
            if row.get(f)!=c[f]:raise ValueError(f'Condition identity mismatch: {f}')
        if row.get('source_a_original_choice')!='(A)' or row.get('source_b_original_choice')!='(B)':raise ValueError('Direction must stay red=a to blue=b')
        signature={f:row[f] for f in ('source_pixel_sha256','source_image_sha256') if f in row}
        pk=(q['scene_id'],row['response_format'],row['mapping'],row['condition'])
        if pk in pixels and pixels[pk]!=signature:raise ValueError('Identical visual intervention changed across queries')
        pixels[pk]=signature;indexed[k]=row
    return indexed


def _contrasts(effects):
    groups=defaultdict(dict)
    for e in effects:
        k=e['scene_id'],e['response_format'],e['mapping'],e['main_strength']
        groups[k][(e['main_layout'],e['query_role'])]=e
    result=[]
    def subtract(a,b):return a-b if a is not None and b is not None else None
    for group,cells in groups.items():
        first=next(iter(cells.values()));base={k:first[k] for k in ('scene_id','response_format','mapping','main_strength','has_unchanged_query')}
        roles=('changed','unchanged') if first['has_unchanged_query'] else ('changed',)
        if set(cells)!={(m,q) for m in ('M0','M1') for q in roles}:raise ValueError('Planned layout/query factorial incomplete')
        def v(m,q):return cells[m,q]['effect']
        for role in roles:
            result.append({**base,'contrast':'layout_difference','query_role':role,'main_layout':'M1_minus_M0',
                           'effect':subtract(v('M1',role),v('M0',role))})
        if first['has_unchanged_query']:
            diffs={m:subtract(v(m,'changed'),v(m,'unchanged')) for m in ('M0','M1')}
            for m in ('M0','M1'):
                result.append({**base,'contrast':'changed_minus_unchanged','query_role':'changed_minus_unchanged','main_layout':m,'effect':diffs[m]})
            result.append({**base,'contrast':'layout_by_query','query_role':'changed_minus_unchanged','main_layout':'M1_minus_M0',
                           'effect':subtract(diffs['M1'],diffs['M0'])})
    return result


def analyze(records,generations=(),expected_plan=None):
    p=expected_plan
    if not p or p.get('schema')!=DESIGN_SCHEMA:raise ValueError('Fixed natural execution plan required')
    validate_query_groups(p['queries']);expected={};eg={};pairs=defaultdict(dict)
    for q in p['queries']:
        for r in p['readouts']:
            for c in p['conditions']:
                k=q['id'],r['response_format'],r['mapping'],c['name']
                if k in expected:raise ValueError('Duplicate planned cell')
                expected[k]=(q,c,r)
                if q['generation_selected'] and c['generation']:eg[k]=expected[k]
                if c['role']=='natural_roi_pair':
                    pk=k[:3]+(c['main_strength'],c['main_layout'])
                    if c['donor_assignment'] in pairs[pk]:raise ValueError('Duplicate planned assignment')
                    pairs[pk][c['donor_assignment']]=k
    if (len(expected),len(eg))!=(p['expected_score_rows'],p['expected_generation_rows']):raise ValueError('Expected row counts mismatch')
    scores=_index(records,expected,'score');gens=_index(generations,eg,'generation');effects=[];behavior=[]
    for assignments in pairs.values():
        if set(assignments)!={'a','b'}:raise ValueError('Incomplete planned a/b pair')
        ka,kb=assignments['a'],assignments['b'];q,c,r=expected[ka]
        cb=expected[kb][1]
        for f in ('main','main_layout','main_strength','entry_group','region','token_permutation','varying_entry'):
            if c[f]!=cb[f]:raise ValueError(f'a/b pair changes fixed condition: {f}')
        if (c['donor_source'],cb['donor_source'])!=('d_a','d_b'):raise ValueError('Incorrect a/b donor sources')
        base={**{f:q[f] for f in ('id','scene_id','intervention_id','query_role','changed_object','queried_object','has_unchanged_query')},
              **r,'main_strength':c['main_strength'],'main_layout':c['main_layout']}
        a,b=scores.get(ka),scores.get(kb);valid=finite_score(a) and finite_score(b)
        effects.append({**base,'effect':b['b_minus_a_log_odds']-a['b_minus_a_log_odds'] if valid else None,
                        'missing_rows':sum(x is None for x in (a,b)),
                        'nonfinite_rows':sum(x is not None and not finite_score(x) for x in (a,b))})
        if ka in eg:
            a,b=gens.get(ka),gens.get(kb);valid=valid_generation(a) and valid_generation(b)
            effect=int(b['original_prediction']=='(B)')-int(a['original_prediction']=='(B)') if valid else None
            behavior.append({**base,'effect':effect,'both_valid':valid,
                             'flip':a['original_prediction']!=b['original_prediction'] if valid else None,
                             'missing_rows':sum(x is None for x in (a,b)),
                             'invalid_rows':sum(x is not None and not valid_generation(x) for x in (a,b)),
                             'truncated_rows':sum(bool(x and x.get('truncated')) for x in (a,b))})
    contrasts=_contrasts(effects);gen_contrasts=_contrasts(behavior)
    native_effects=[];native_behavior=[]
    for q in p['queries']:
        for r in p['readouts']:
            ka=(q['id'],r['response_format'],r['mapping'],'native_d_a')
            kb=ka[:3]+('native_d_b',)
            base={**{f:q[f] for f in ('id','scene_id','query_role','has_unchanged_query')},**r}
            a,b=scores.get(ka),scores.get(kb)
            native_effects.append({**base,'effect':b['b_minus_a_log_odds']-a['b_minus_a_log_odds'] if finite_score(a) and finite_score(b) else None,
                'missing_rows':sum(x is None for x in (a,b)),'nonfinite_rows':sum(x is not None and not finite_score(x) for x in (a,b))})
            if ka in eg:
                a,b=gens.get(ka),gens.get(kb);valid=valid_generation(a) and valid_generation(b)
                native_behavior.append({**base,'effect':int(b['original_prediction']=='(B)')-int(a['original_prediction']=='(B)') if valid else None,
                    'both_valid':valid,'flip':a['original_prediction']!=b['original_prediction'] if valid else None,
                    'missing_rows':sum(x is None for x in (a,b)),
                    'invalid_rows':sum(x is not None and not valid_generation(x) for x in (a,b)),
                    'truncated_rows':sum(bool(x and x.get('truncated')) for x in (a,b))})
    natives=defaultdict(list);audits=[]
    for k,(q,c,r) in expected.items():
        s=scores.get(k);g=gens.get(k)
        if c['role'].startswith('native_'):
            gi=q['source_answer_indices'][c['main']];gold=('(A)','(B)')[gi] if gi is not None else None
            natives[(r['response_format'],r['mapping'],c['name'],q['query_role'])].append({
                'scene_id':q['scene_id'],'gold':gold,'score_present':finite_score(s),
                'score_correct':bool(finite_score(s) and s.get('original_prediction')==gold) if gold else None,
                'gen_planned':k in eg,'gen_present':g is not None,'gen_valid':valid_generation(g),
                'gen_correct':bool(valid_generation(g) and g['original_prediction']==gold) if gold else None})
        elif c['role']=='self_patch':
            native=scores.get(k[:3]+(f'native_{c["main"]}',))
            audits.append({'id':q['id'],'scene_id':q['scene_id'],**r,'condition':c['name'],
                           'delta_b_minus_a':s['b_minus_a_log_odds']-native['b_minus_a_log_odds'] if finite_score(s) and finite_score(native) else None})
    native_summary=[]
    for k,members in natives.items():
        native_summary.append({**dict(zip(('response_format','mapping','condition','query_role'),k)),
          'planned_rows':len(members),'planned_rows_with_gold':sum(x['gold'] is not None for x in members),
          'score_rows_present':sum(x['score_present'] for x in members),'score_correct':sum(x['score_correct'] is True for x in members),
          'planned_generation_rows':sum(x['gen_planned'] for x in members),'generation_rows_present':sum(x['gen_present'] for x in members),
          'generation_valid':sum(x['gen_valid'] for x in members),'generation_correct':sum(x['gen_correct'] is True for x in members)})
    return {'schema':DESIGN_SCHEMA,'unit_key':'scene_id','coverage':{
        'planned_source_photos':len({q['scene_id'] for q in p['queries']}),
        'paired_off_target_source_photos':len({q['scene_id'] for q in p['queries'] if q['has_unchanged_query']}),
        'planned_score_rows':len(expected),'present_score_rows':len(scores),'missing_score_rows':len(expected)-len(scores),
        'nonfinite_score_rows':sum(not finite_score(s) for s in scores.values()),
        'planned_generation_rows':len(eg),'present_generation_rows':len(gens),'missing_generation_rows':len(eg)-len(gens),
        'invalid_generation_rows':sum(not valid_generation(g) for g in gens.values()),
        'truncated_generation_rows':sum(bool(g.get('truncated')) for g in gens.values()),
        'planned_pairs':len(effects),'complete_pairs':sum(x['effect'] is not None for x in effects),
        'planned_generation_pairs':len(behavior),'valid_generation_pairs':sum(x['both_valid'] for x in behavior),
        'generation_flips':sum(x['flip'] is True for x in behavior),
        'planned_native_donor_pairs':len(native_effects),'complete_native_donor_pairs':sum(x['effect'] is not None for x in native_effects),
        'planned_native_donor_generation_pairs':len(native_behavior),'valid_native_donor_generation_pairs':sum(x['both_valid'] for x in native_behavior)},
        'query_effects':effects,'query_contrasts':contrasts,'behavior_pairs':behavior,'behavior_contrasts':gen_contrasts,
        'cell_summaries':summarize(effects,GROUP),'per_mapping_cell_summaries':summarize(effects,GROUP+('mapping',)),
        'contrast_summaries':summarize(contrasts,('contrast',)+GROUP),
        'per_mapping_contrast_summaries':summarize(contrasts,('contrast',)+GROUP+('mapping',)),
        'behavior_summaries':summarize(behavior,GROUP),'per_mapping_behavior_summaries':summarize(behavior,GROUP+('mapping',)),
        'behavior_contrast_summaries':summarize(gen_contrasts,('contrast',)+GROUP),
        'native_accuracy':native_summary,'self_patch_audits':audits,
        'native_donor_effects':native_effects,'native_donor_behavior':native_behavior,
        'native_donor_summaries':summarize(native_effects,('response_format','query_role')),
        'per_mapping_native_donor_summaries':summarize(native_effects,('response_format','query_role','mapping')),
        'native_donor_behavior_summaries':summarize(native_behavior,('response_format','query_role')),
        'per_mapping_native_donor_behavior_summaries':summarize(native_behavior,('response_format','query_role','mapping')), 
        'definitions':{'E':'S(blue-region donor)-S(red-region donor), S=log P(blue)-log P(red)',
                       'Delta_native':'S(native edited-b)-S(native edited-a), diagnostic only; never filter scenes by this outcome',
                       'layout_by_query':'[E(changed)-E(unchanged)] at M1 minus M0, declared off-target subset only',
                       'interpretation':'Nominal bbox tokens have contextualized receptive fields; mirror does not isolate an abstract index operation'}}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--results',type=Path,required=True);p.add_argument('--generations',type=Path);p.add_argument('--plan',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args(argv)
    read=lambda path:[json.loads(s) for s in _guard_path(path).read_text().splitlines() if s]
    report=analyze(read(a.results),read(a.generations) if a.generations else [],json.loads(_guard_path(a.plan).read_text()))
    output=_guard_path(a.output,write=True);output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()
