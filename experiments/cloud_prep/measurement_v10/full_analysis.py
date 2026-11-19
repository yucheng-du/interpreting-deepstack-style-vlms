"""Fixed-denominator reciprocal source effects and paired role/strength/query interactions."""
from __future__ import annotations
import argparse
from collections import defaultdict
from itertools import product
import json
from pathlib import Path
from cloud_prep.portable_data import _guard_path
from cloud_prep.measurement_v8.mechanism_analysis import finite_score,valid_generation,key,summarize
from cloud_prep.measurement_v10.full_data import validate_query_groups
from cloud_prep.measurement_v10.full_design import DESIGN_SCHEMA,CONDITION_FIELDS,QUERY_FIELDS

GROUP=('response_format','changed_entry','changing_layout','fixed_layout','changing_strength','fixed_strength',
       'fixed_other_assignment','query_type','query_target')
C_GROUP=('contrast','response_format','changing_layout','fixed_layout','fixed_other_assignment','query_target',
         'changed_entry','changing_strength','fixed_strength','query_type')
AXES=('changed_entry','changing_strength','fixed_strength','query_type')
CONTRAST_AXES=(('changed_entry',),('changing_strength',),('fixed_strength',),('query_type',),
               ('changed_entry','query_type'),('changed_entry','changing_strength'),('changed_entry','fixed_strength'),
               ('changing_strength','query_type'),('fixed_strength','query_type'),
               ('changed_entry','changing_strength','query_type'),('changed_entry','fixed_strength','query_type'),
               ('changing_strength','fixed_strength'),('changed_entry','changing_strength','fixed_strength'),
               ('changed_entry','changing_strength','fixed_strength','query_type'))


def _validate(records,expected,kind):
    indexed={}; signatures={}
    for row in records:
        k=key(row)
        if k in indexed:raise ValueError(f'Duplicate {kind} cell; select one attempt')
        if k not in expected:raise ValueError(f'Unplanned {kind} cell')
        q,r,c=expected[k]
        if row.get('schema')!=DESIGN_SCHEMA:raise ValueError('Wrong full-source scientific schema')
        for field in QUERY_FIELDS:
            if row.get(field)!=q[field]:raise ValueError(f'Query field mismatch: {field}')
        for field in CONDITION_FIELDS:
            if row.get(field)!=c[field]:raise ValueError(f'Condition field mismatch: {field}')
        if row.get('main_source')!=c['main'] or row.get('condition_spec')!=c:raise ValueError('Source/condition contract mismatch')
        for assignment in ('a','b'):
            if row.get(f'source_{assignment}_original_choice')!=('(A)','(B)')[q[f'direction_{assignment}_index']]:
                raise ValueError('Entry-role-dependent direction is forbidden')
        # The same physical intervention across queries must use the same processed pixels.
        signature_key=q['intervention_id'],r['response_format'],r['mapping'],c['name']
        actual={f:row[f] for f in ('source_pixel_sha256','source_image_sha256') if f in row}
        if signature_key in signatures and signatures[signature_key]!=actual:raise ValueError('Query changed actual visual sources')
        signatures[signature_key]=actual;indexed[k]=row
    return indexed


def _contrasts(effects):
    """Exchange physical entry roles while keeping changing/fixed source content identical."""
    groups=defaultdict(dict)
    base_fields=('scene_id','intervention_id','changed_object','changed_index','response_format','mapping',
                 'query_target','changing_layout','fixed_layout','fixed_other_assignment')
    for row in effects:
        group=tuple(row[f] for f in base_fields); cell=tuple(row[f] for f in AXES)
        if cell in groups[group]:raise ValueError('Duplicate reciprocal effect coordinate')
        groups[group][cell]=row
    result=[]
    for base,cells in groups.items():
        levels={axis:sorted({cell[i] for cell in cells},key=str) for i,axis in enumerate(AXES)}
        direction={'changed_entry':('main','later'),'query_type':('relation','identity'),
                   'changing_strength':(1.,.08),'fixed_strength':(1.,.08)}
        for contrast_axes in CONTRAST_AXES:
            if any(not set(direction[axis])<=set(levels[axis]) for axis in contrast_axes):continue
            held_axes=[axis for axis in AXES if axis not in contrast_axes]
            for held_values in product(*(levels[axis] for axis in held_axes)):
                held=dict(zip(held_axes,held_values));terms=[]
                for bits in product((0,1),repeat=len(contrast_axes)):
                    coords=held|{axis:direction[axis][bit] for axis,bit in zip(contrast_axes,bits)}
                    member=cells.get(tuple(coords[axis] for axis in AXES))
                    # A missing planned record is None; an absent planned design cell is an error.
                    if member is None:raise ValueError('Planned source factorial is incomplete')
                    terms.append(((-1)**sum(bits),member['effect']))
                value=sum(sign*effect for sign,effect in terms) if all(effect is not None for _,effect in terms) else None
                labels={axis:f'{direction[axis][0]}_minus_{direction[axis][1]}' for axis in contrast_axes}
                result.append({**dict(zip(base_fields,base)),**held,**labels,
                    'contrast':'_by_'.join(contrast_axes),'effect':value,'paired_terms':len(terms)})
    return result


def analyze(records,generations=(),expected_plan=None):
    plan=expected_plan
    if not plan or plan.get('schema')!=DESIGN_SCHEMA:raise ValueError('Full-source execution plan required for fixed denominators')
    validate_query_groups(plan['queries'])
    expected={};expected_gen={};pairs=defaultdict(dict)
    for q,r,c in product(plan['queries'],plan['readouts'],plan['conditions']):
        k=(q['id'],r['response_format'],r['mapping'],c['name'])
        if k in expected:raise ValueError('Duplicate planned cell')
        expected[k]=(q,r,c)
        if q['generation_selected'] and c['generation']:expected_gen[k]=(q,r,c)
        if c['role']=='source_cross':
            for changed in ('main','later'):
                other='later' if changed=='main' else 'main'
                p=k[:3]+(changed,c[other+'_assignment'],c['main_layout'],c['later_layout'],c['main_strength'],c['later_strength'])
                assignment=c[changed+'_assignment']
                if assignment in pairs[p]:raise ValueError('Duplicate source matrix assignment')
                pairs[p][assignment]=k
    if len(expected)!=plan['expected_score_rows'] or len(expected_gen)!=plan['expected_generation_rows']:raise ValueError('Plan count mismatch')
    scores=_validate(records,expected,'score');gens=_validate(generations,expected_gen,'generation')
    effects=[];behavior=[]
    for pair,keys in pairs.items():
        if set(keys)!=set(('a','b')):raise ValueError('Incomplete planned source assignment pair')
        q,r,c=expected[keys['a']]; changed=pair[3];other='later' if changed=='main' else 'main'
        ca=expected[keys['a']][2];cb=expected[keys['b']][2]
        if any(ca[f]!=cb[f] for f in (other+'_assignment',other+'_layout',other+'_strength',changed+'_layout',changed+'_strength')):
            raise ValueError('a/b comparison altered a held source factor')
        base={**{f:q[f] for f in ('id','scene_id','intervention_id','changed_object','changed_index','attribute','queried_object',
                                  'queried_index','query_type','query_target','query_slot','direction_a_index','direction_b_index')},**r,
              'changed_entry':changed,'fixed_other_assignment':c[other+'_assignment'],
              'main_layout':c['main_layout'],'later_layout':c['later_layout'],'main_strength':c['main_strength'],'later_strength':c['later_strength'],
              'changing_layout':c[changed+'_layout'],'fixed_layout':c[other+'_layout'],
              'changing_strength':c[changed+'_strength'],'fixed_strength':c[other+'_strength'],
              'a_condition':keys['a'][3],'b_condition':keys['b'][3]}
        a,b=scores.get(keys['a']),scores.get(keys['b']);valid=finite_score(a) and finite_score(b)
        effects.append({**base,'effect':b['b_minus_a_log_odds']-a['b_minus_a_log_odds'] if valid else None,
            'a_score':a['b_minus_a_log_odds'] if finite_score(a) else None,'b_score':b['b_minus_a_log_odds'] if finite_score(b) else None,
            'missing_score_rows':sum(x is None for x in (a,b)),'nonfinite_score_rows':sum(x is not None and not finite_score(x) for x in (a,b))})
        if keys['a'] in expected_gen and keys['b'] in expected_gen:
            a,b=gens.get(keys['a']),gens.get(keys['b']);valid=valid_generation(a) and valid_generation(b)
            color_b=('(A)','(B)')[q['direction_b_index']]
            effect=int(b['original_prediction']==color_b)-int(a['original_prediction']==color_b) if valid else None
            behavior.append({**base,'effect':effect,'both_valid':valid,
                'flip':a['original_prediction']!=b['original_prediction'] if valid else None,
                'direction':'toward_b' if effect==1 else 'toward_a' if effect==-1 else 'unchanged' if effect==0 else None,
                'missing_generation_rows':sum(x is None for x in (a,b)),
                'invalid_generation_rows':sum(x is not None and not valid_generation(x) for x in (a,b)),
                'truncated_generation_rows':sum(bool(x and x.get('truncated')) for x in (a,b))})
    contrasts=_contrasts(effects);gen_contrasts=_contrasts(behavior)
    behavior_summary=[];grouped=defaultdict(list)
    for row in behavior:grouped[tuple(row[f] for f in GROUP)+ (row['mapping'],)].append(row)
    for group,members in grouped.items():
        valid=sum(m['both_valid'] for m in members);flips=sum(m['flip'] is True for m in members)
        behavior_summary.append({**dict(zip(GROUP+('mapping',),group)),'planned_pairs':len(members),'common_valid_pairs':valid,
            'missing_pairs':sum(m['missing_generation_rows']>0 for m in members),
            'invalid_pairs':sum(m['invalid_generation_rows']>0 for m in members),
            'truncated_pairs':sum(m['truncated_generation_rows']>0 for m in members),'flips':flips,
            'flip_rate_common_valid':flips/valid if valid else None,'flips_over_planned_lower_bound':flips/len(members)})
    natives=defaultdict(list);audits=[]
    for k,(q,r,c) in expected.items():
        score,gen=scores.get(k),gens.get(k)
        if c['role'].startswith('native_'):
            gi=q['source_answer_indices'][c['main']];gold=('(A)','(B)')[gi] if gi is not None else None
            natives[(r['response_format'],r['mapping'],c['name'],q['query_type'],q['query_target'])].append({
                'gold':gold,'score_present':finite_score(score),'score_correct':bool(gold and finite_score(score) and score.get('original_prediction')==gold),
                'generation_planned':k in expected_gen,'generation_present':gen is not None,'generation_valid':valid_generation(gen),
                'generation_correct':bool(gold and valid_generation(gen) and gen['original_prediction']==gold)})
        elif c['role']=='self_patch':
            native=scores.get(k[:3]+(f'native_{c["main"]}',))
            audits.append({'id':q['id'],'scene_id':q['scene_id'],**r,'condition':c['name'],
                'delta_b_minus_a':score['b_minus_a_log_odds']-native['b_minus_a_log_odds'] if finite_score(score) and finite_score(native) else None})
    native_summary=[]
    for group,members in natives.items():
        native_summary.append({**dict(zip(('response_format','mapping','condition','query_type','query_target'),group)),
            'planned_rows':len(members),'planned_rows_with_gold':sum(m['gold'] is not None for m in members),
            'planned_generation_rows_with_gold':sum(m['generation_planned'] and m['gold'] is not None for m in members),
            'generation_valid_with_gold':sum(m['generation_valid'] and m['gold'] is not None for m in members),
            **{field:sum(m[field] for m in members) for field in ('score_present','score_correct','generation_planned','generation_present','generation_valid','generation_correct')}})
    return {'schema':DESIGN_SCHEMA,'unit_key':'scene_id','coverage':{
        'planned_score_rows':len(expected),'present_score_rows':len(scores),'missing_score_rows':len(expected)-len(scores),
        'nonfinite_score_rows':sum(not finite_score(r) for r in scores.values()),
        'planned_generation_rows':len(expected_gen),'present_generation_rows':len(gens),'missing_generation_rows':len(expected_gen)-len(gens),
        'invalid_generation_rows':sum(not valid_generation(r) for r in gens.values()),'truncated_generation_rows':sum(bool(r.get('truncated')) for r in gens.values()),
        'planned_units':len({q['scene_id'] for q in plan['queries']}),'planned_pairs':len(effects),'complete_pairs':sum(e['effect'] is not None for e in effects),
        'planned_generation_pairs':len(behavior),'common_valid_generation_pairs':sum(b['both_valid'] for b in behavior)},
        'query_effects':effects,'query_contrasts':contrasts,'cell_summaries':summarize(effects,GROUP),
        'per_mapping_cell_summaries':summarize(effects,GROUP+('mapping',)),
        'contrast_summaries':summarize(contrasts,C_GROUP),'per_mapping_contrast_summaries':summarize(contrasts,C_GROUP+('mapping',)),
        'behavior_pairs':behavior,'behavior_summary':behavior_summary,'behavior_contrasts':gen_contrasts,
        'behavior_effect_summaries':summarize(behavior,GROUP),'per_mapping_behavior_effect_summaries':summarize(behavior,GROUP+('mapping',)),
        'behavior_contrast_summaries':summarize(gen_contrasts,C_GROUP),
        'per_mapping_behavior_contrast_summaries':summarize(gen_contrasts,C_GROUP+('mapping',)),
        'native_accuracy':native_summary,'self_patch_audits':audits,
        'contrast_definitions':{'E':'S(entry=b,other=fixed)-S(entry=a,other=fixed); S=logP(changed-b)-logP(changed-a)',
            'role_exchange':'E_main-E_later with the changing and fixed source layout/strength/assignment retained as those sources exchange physical entry',
            'query':'relation-identity, referring to the same M0 target slot; source relation truth may differ after layout swap',
            'strength':'1.00-.08 for changing or fixed source; alpha0 is a separate boundary, never pooled into this difference',
            'interaction':'Direct within-intervention alternating differences, then complete scene means; no comparison of independent significance tests'},
        'interpretation':'Exploratory full-source competition with fixed source pairs. Whole-image role exchange does not match feature scale/depth and cannot prove fixed anatomical division. The center reference is additional context. Direct and label scales remain separate; mappings retained and equal weighted only through complete scene repeats. No native-success filtering. Invalid/truncated generations and missing rows never become zero. A/B direction never flips with entry role.'}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--scores',type=Path,nargs='+',required=True);p.add_argument('--generations',type=Path,nargs='*',default=[])
    p.add_argument('--output',type=Path,required=True);a=p.parse_args(argv)
    load=lambda paths:[json.loads(line) for path in paths for line in path.read_text().splitlines() if line.strip()]
    result=analyze(load(a.scores),load(a.generations),json.loads(a.plan.read_text()))
    output=_guard_path(a.output,write=True);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result['coverage'],indent=2))
if __name__=='__main__':main()
