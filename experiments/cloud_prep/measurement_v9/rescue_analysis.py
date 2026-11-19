"""Paired destination x queried-object x main-layout effects, clustered by scene.

The complete execution plan determines every denominator, including wholly
missing queries. Direct and label scales are separate. No native-success filter.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

from cloud_prep.portable_data import _guard_path
from cloud_prep.measurement_v8.mechanism_analysis import stats, summarize, finite_score, valid_generation, key
from cloud_prep.measurement_v9.rescue_data import validate_query_groups
from cloud_prep.measurement_v9.rescue_design import DESIGN_SCHEMA

GROUP = ('attribute','response_format','main_strength','entry_group','main_layout','source_destination','query_role')
CONTRAST_GROUP = ('contrast',) + GROUP


def _validate_records(records, expected, kind):
    indexed = {}
    signatures = {}
    for row in records:
        k = key(row)
        if k in indexed:
            raise ValueError(f'Duplicate {kind} cell; select one execution attempt')
        if k not in expected:
            raise ValueError(f'Unplanned {kind} cell: {k}')
        q,c = expected[k]['query'], expected[k]['condition']
        if row.get('schema') != DESIGN_SCHEMA:
            raise ValueError('Wrong rescue scientific schema')
        for field in ('scene_id','intervention_id','changed_object','changed_index','queried_object','queried_index',
                      'query_role','direction_a_index','direction_b_index','source_roi_slot','visual_source_signature'):
            if row.get(field) != q[field]:
                raise ValueError(f'Query identity mismatch: {field}')
        for field in ('role','main_layout','main_strength','entry_group','region','donor_assignment',
                      'source_destination','token_permutation'):
            if row.get(field) != c[field]:
                raise ValueError(f'Condition identity mismatch: {field}')
        for assignment in ('a','b'):
            if row.get(f'source_{assignment}_original_choice') != ('(A)','(B)')[q[f'direction_{assignment}_index']]:
                raise ValueError('Score/generation direction must use changed-object colors for both queries')
        # Actual processed pixels are optional for CPU fixtures, but must agree
        # whenever runtime records provide them; metadata alone cannot prove CUDA identity.
        pair = q['intervention_id'], row['response_format'], row['mapping'], row['condition']
        actual = {field:row[field] for field in ('source_pixel_sha256','source_image_sha256') if field in row}
        if pair in signatures and actual != signatures[pair]:
            raise ValueError('Two query readouts changed actual visual inputs')
        signatures[pair] = actual
        indexed[k] = row
    return indexed


def _sub(a,b):
    return a-b if a is not None and b is not None else None


def _contrasts(effects):
    """Form all differences within the same intervention/readout before scene means."""
    units = defaultdict(dict)
    for e in effects:
        unit = e['intervention_id'],e['response_format'],e['mapping'],e['main_strength'],e['entry_group']
        cell = e['main_layout'],e['source_destination'],e['query_role']
        if cell in units[unit]:
            raise ValueError('Duplicate planned effect in paired interaction')
        units[unit][cell] = e
    out = []
    for cells in units.values():
        if set(cells) != {(m,d,q) for m in ('M0','M1') for d in ('original','opposite') for q in ('changed','unchanged')}:
            raise ValueError('Paired interaction requires complete planned M x destination x query factorial')
        first = next(iter(cells.values()))
        base = {k:first[k] for k in ('scene_id','intervention_id','attribute','changed_object','changed_index',
                                    'response_format','mapping','main_strength','entry_group')}
        def emit(name,m,d,q,value):
            out.append({**base, 'contrast':name, 'main_layout':m, 'source_destination':d,
                        'query_role':q, 'effect':value})
        def value(m,d,q):
            return cells[m,d,q]['effect']
        readout_difference = {}
        for m in ('M0','M1'):
            for d in ('original','opposite'):
                v = _sub(value(m,d,'changed'),value(m,d,'unchanged'))
                readout_difference[m,d] = v
                emit('changed_minus_unchanged',m,d,'changed_minus_unchanged',v)
            for q in ('changed','unchanged'):
                emit('destination_difference',m,'opposite_minus_original',q,
                     _sub(value(m,'opposite',q),value(m,'original',q)))
            dest_query = _sub(readout_difference[m,'opposite'],readout_difference[m,'original'])
            emit('destination_by_query',m,'opposite_minus_original','changed_minus_unchanged',dest_query)
            # Positive means changed-vs-unchanged readout advantage moves with
            # the current changed object's slot; no expected-sign gate is used.
            emit('current_object_routing',m,'current_changed_minus_current_unchanged',
                 'changed_minus_unchanged', -dest_query if dest_query is not None and m == 'M0' else dest_query)
        for d in ('original','opposite'):
            for q in ('changed','unchanged'):
                emit('layout_difference','M1_minus_M0',d,q,_sub(value('M1',d,q),value('M0',d,q)))
            emit('layout_by_query','M1_minus_M0',d,'changed_minus_unchanged',
                 _sub(readout_difference['M1',d],readout_difference['M0',d]))
        triple = _sub(_sub(readout_difference['M1','opposite'],readout_difference['M1','original']),
                      _sub(readout_difference['M0','opposite'],readout_difference['M0','original']))
        emit('layout_by_destination_by_query','M1_minus_M0','opposite_minus_original','changed_minus_unchanged',triple)
    return out


def _behavior_summary(behavior):
    groups = defaultdict(list)
    for b in behavior:
        groups[tuple(b[k] for k in GROUP)+ (b['mapping'],)].append(b)
    result = []
    for group,members in groups.items():
        n = len(members)
        valid = sum(m['both_valid'] for m in members)
        flips = sum(m['flip'] is True for m in members)
        result.append({**dict(zip(GROUP+('mapping',),group)), 'planned_pairs':n, 'common_valid_pairs':valid,
            'missing_pairs':sum(m['missing_generation_rows']>0 for m in members),
            'invalid_pairs':sum(m['invalid_generation_rows']>0 for m in members),
            'truncated_pairs':sum(m['truncated_generation_rows']>0 for m in members),
            'flips':flips, 'flips_over_planned_lower_bound':flips/n,
            'flip_rate_common_valid':flips/valid if valid else None,
            'toward_b':sum(m['direction']=='toward_b' for m in members),
            'toward_a':sum(m['direction']=='toward_a' for m in members)})
    return result


def analyze(records, generations=(), expected_plan=None):
    if not expected_plan or expected_plan.get('schema') != DESIGN_SCHEMA:
        raise ValueError('A rescue execution plan is required for fixed denominators')
    plan = expected_plan
    validate_query_groups(plan['queries'])
    expected, expected_gen, pair_specs = {},{},defaultdict(dict)
    for q in plan['queries']:
        for r in plan['readouts']:
            for c in plan['conditions']:
                k = q['id'],r['response_format'],r['mapping'],c['name']
                if k in expected:
                    raise ValueError('Duplicate planned cell')
                expected[k] = {'query':q,'readout':r,'condition':c}
                if q['generation_selected'] and c['generation']:
                    expected_gen[k] = expected[k]
                if c['role'] == 'local_rescue_pair':
                    pair = k[:3]+tuple(c[f] for f in ('main_strength','entry_group','main_layout','source_destination'))
                    if c['donor_assignment'] in pair_specs[pair]:
                        raise ValueError('Duplicate planned donor assignment')
                    pair_specs[pair][c['donor_assignment']] = k
    if len(expected) != plan['expected_score_rows'] or len(expected_gen) != plan['expected_generation_rows']:
        raise ValueError('Plan row counts disagree with explicit condition product')
    scores = _validate_records(records,expected,'score')
    gens = _validate_records(generations,expected_gen,'generation')
    effects,behavior = [],[]
    for keys in pair_specs.values():
        if set(keys) != {'a','b'}:
            raise ValueError('Scientific plan lacks a complete donor pair')
        spec = expected[keys['a']]
        q,c,r = spec['query'],spec['condition'],spec['readout']
        base = {**{k:q[k] for k in ('id','scene_id','attribute','intervention_id','changed_object','changed_index',
            'queried_object','queried_index','query_role','direction_a_index','direction_b_index')}, **r,
            **{k:c[k] for k in ('main_strength','entry_group','main_layout','source_destination','token_permutation')}}
        a,b = scores.get(keys['a']),scores.get(keys['b'])
        valid = finite_score(a) and finite_score(b)
        effects.append({**base,'effect':b['b_minus_a_log_odds']-a['b_minus_a_log_odds'] if valid else None,
            'a_score':a['b_minus_a_log_odds'] if finite_score(a) else None,
            'b_score':b['b_minus_a_log_odds'] if finite_score(b) else None,
            'missing_score_rows':sum(x is None for x in (a,b)),
            'nonfinite_score_rows':sum(x is not None and not finite_score(x) for x in (a,b))})
        if keys['a'] in expected_gen and keys['b'] in expected_gen:
            ga,gb = gens.get(keys['a']),gens.get(keys['b'])
            valid = valid_generation(ga) and valid_generation(gb)
            color_b = ('(A)','(B)')[q['direction_b_index']]
            effect = int(gb['original_prediction']==color_b)-int(ga['original_prediction']==color_b) if valid else None
            behavior.append({**base,'effect':effect,'both_valid':valid,
                'flip':ga['original_prediction']!=gb['original_prediction'] if valid else None,
                'direction':'toward_b' if effect==1 else 'toward_a' if effect==-1 else 'unchanged' if effect==0 else None,
                'missing_generation_rows':sum(x is None for x in (ga,gb)),
                'invalid_generation_rows':sum(x is not None and not valid_generation(x) for x in (ga,gb)),
                'truncated_generation_rows':sum(bool(x and x.get('truncated')) for x in (ga,gb))})
    contrasts = _contrasts(effects)
    gen_contrasts = _contrasts(behavior)
    natives, audits = defaultdict(list),[]
    for k,spec in expected.items():
        q,c,r = spec['query'],spec['condition'],spec['readout']
        score,gen = scores.get(k),gens.get(k)
        if c['role'].startswith('native_'):
            gold_index = q['source_answer_indices'][c['main']]
            gold = ('(A)','(B)')[gold_index] if gold_index is not None else None
            natives[(r['response_format'],r['mapping'],c['name'],q['query_role'])].append({
                'scene_id':q['scene_id'], 'gold':gold, 'score_present':finite_score(score),
                'score_correct':finite_score(score) and score.get('original_prediction')==gold if gold else None,
                'gen_planned':k in expected_gen, 'gen_present':gen is not None,
                'gen_valid':valid_generation(gen),
                'gen_correct':valid_generation(gen) and gen['original_prediction']==gold if gold else None})
        elif c['role'] == 'self_patch':
            native = scores.get(k[:3]+(f'native_{c["main"]}',))
            audits.append({'id':q['id'],'scene_id':q['scene_id'],**r,'condition':c['name'],
                'delta_b_minus_a':score['b_minus_a_log_odds']-native['b_minus_a_log_odds'] if finite_score(score) and finite_score(native) else None})
    native_summary = []
    for group,members in natives.items():
        native_summary.append({**dict(zip(('response_format','mapping','condition','query_role'),group)),
            'planned_units':len({m['scene_id'] for m in members}), 'planned_rows':len(members),
            'planned_rows_with_gold':sum(m['gold'] is not None for m in members),
            'score_rows_present':sum(m['score_present'] for m in members),
            'score_correct':sum(m['score_correct'] is True for m in members),
            'planned_generation_rows':sum(m['gen_planned'] for m in members),
            'planned_generation_rows_with_gold':sum(m['gen_planned'] and m['gold'] is not None for m in members),
            'generation_rows_present':sum(m['gen_present'] for m in members),
            'generation_common_valid_with_gold':sum(m['gen_valid'] and m['gold'] is not None for m in members),
            'generation_correct':sum(m['gen_correct'] is True for m in members)})
    return {'schema':DESIGN_SCHEMA,'unit_key':'scene_id','coverage':{
        'planned_score_rows':len(expected),'present_score_rows':len(scores),'missing_score_rows':len(expected)-len(scores),
        'nonfinite_score_rows':sum(not finite_score(s) for s in scores.values()),
        'planned_generation_rows':len(expected_gen),'present_generation_rows':len(gens),
        'missing_generation_rows':len(expected_gen)-len(gens),'invalid_generation_rows':sum(not valid_generation(g) for g in gens.values()),
        'truncated_generation_rows':sum(bool(g.get('truncated')) for g in gens.values()),
        'planned_units':len({q['scene_id'] for q in plan['queries']}),
        'planned_generation_units':len({q['scene_id'] for q in plan['queries'] if q['generation_selected']}),
        'planned_query_pairs':len(effects),'complete_query_pairs':sum(e['effect'] is not None for e in effects),
        'planned_generation_pairs':len(behavior),'common_valid_generation_pairs':sum(b['both_valid'] for b in behavior)},
        'query_effects':effects,'query_contrasts':contrasts,
        'cell_summaries':summarize(effects,GROUP),
        'per_mapping_cell_summaries':summarize(effects,GROUP+('mapping',)),
        'contrast_summaries':summarize(contrasts,CONTRAST_GROUP),
        'per_mapping_contrast_summaries':summarize(contrasts,CONTRAST_GROUP+('mapping',)),
        'behavior_pairs':behavior,'behavior_summary':_behavior_summary(behavior),
        'behavior_effect_summaries':summarize(behavior,GROUP),
        'per_mapping_behavior_effect_summaries':summarize(behavior,GROUP+('mapping',)),
        'behavior_contrasts':gen_contrasts,
        'behavior_contrast_summaries':summarize(gen_contrasts,CONTRAST_GROUP),
        'per_mapping_behavior_contrast_summaries':summarize(gen_contrasts,CONTRAST_GROUP+('mapping',)),
        'native_accuracy':native_summary,'self_patch_audits':audits,
        'contrast_definitions':{
            'E':'[log P(changed-b color)-log P(changed-a color)] under donor b minus donor a',
            'R':'E(queried=changed)-E(queried=unchanged), same visual intervention',
            'destination_by_query':'R(opposite)-R(original), separately for each M',
            'layout_by_destination_by_query':'[R(M1,opposite)-R(M1,original)]-[R(M0,opposite)-R(M0,original)]',
            'current_object_routing':'R(destination=current changed slot)-R(destination=current unchanged slot)',
            'generation':'Replace score contrast with 1(output=changed-b color); invalid/truncated pairs excluded as missing'},
        'interpretation':'Exploratory within-scene paired interactions on reused v8 scenes. Query identity changes text only, while source images, changed-object ROI, a/b direction and destination are held fixed. Complete scene means require all planned changed-object and mapping repeats, also for generation contrasts; missing values never become zero. Native labels follow queried objects, and native success does not filter effects. Direct and label raw score scales remain separate. ROI is nominal support of contextualized features, not a proof of unique local storage, necessity or additive mechanism.'}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--scores',type=Path,nargs='+',required=True)
    p.add_argument('--generations',type=Path,nargs='*',default=[])
    p.add_argument('--output',type=Path,required=True)
    a = p.parse_args(argv)
    load = lambda paths:[json.loads(line) for path in paths for line in path.read_text().splitlines() if line.strip()]
    result = analyze(load(a.scores),load(a.generations),json.loads(a.plan.read_text()))
    output = _guard_path(a.output,write=True)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'output':str(output),**result['coverage']},indent=2))


if __name__ == '__main__':
    main()
