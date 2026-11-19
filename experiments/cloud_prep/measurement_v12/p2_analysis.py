"""P2 schema projection with attribute-separated template clustering and D/J.

The existing v10 estimators and accepted v11-v2 calibration checks are reused
per attribute. No records, attributes or mappings are averaged before pairing.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import contextmanager
from copy import deepcopy
from itertools import product
import json
from pathlib import Path

from cloud_prep.portable_data import _guard_path
from cloud_prep.measurement_v8.mechanism_analysis import finite_score, valid_generation, summarize
from cloud_prep.measurement_v10 import local_analysis, full_analysis
from cloud_prep.measurement_v12 import p2_design
from cloud_prep.measurement_v12 import v11_v2_dose_analysis as dose_analysis
from cloud_prep.measurement_v12.v11_v2_vendor import verify_vendor

SCHEMA='paired-attribute-analysis-v12-v1'
CONTEXTS={
    'local':('main_strength','entry_group','main_layout','source_destination'),
    'full':('changing_layout','fixed_layout','changing_strength','fixed_strength','fixed_other_assignment','query_type'),
    'dose':('changing_layout','fixed_layout','changing_strength','fixed_strength','fixed_other_assignment',
            'region','dose_mode','dose_value','target_policy','dose_metric'),
    'reference':('changing_layout','fixed_layout','changing_strength','fixed_strength','fixed_other_assignment',
                 'region','reference_target','dose_metric'),
}


def key(r): return r['id'],r['response_format'],r['mapping'],r['condition']
def subtract(a,b): return a-b if a is not None and b is not None else None


def paired_query_contract(queries):
    """P2 has two identity queries per intervention, both edits per template."""
    groups=defaultdict(list); templates=defaultdict(list)
    for q in queries:
        groups[q['intervention_id']].append(q)
        templates[q['scene_id']].append(q)
    for rows in groups.values():
        if len(rows)!=2 or {(r['query_type'],r['query_target']) for r in rows}!={('identity','changed'),('identity','unchanged')}:
            raise ValueError('P2 needs both identity queries per intervention')
        for field in ('scene_id','attribute','changed_object','changed_index','direction_a_index','direction_b_index','visual_source_signature'):
            if rows[0][field]!=rows[1][field]: raise ValueError('Paired intervention changed '+field)
    for rows in templates.values():
        if len(rows)!=4 or {(r['changed_index'],r['queried_index']) for r in rows}!={(0,0),(0,1),(1,0),(1,1)}:
            raise ValueError('An attribute-specific template needs both edits and queries')


@contextmanager
def full_validator_bridge():
    # Only the manifest factorial validator differs. Every original per-record
    # source, condition, direction and planned denominator check remains active.
    previous=full_analysis.validate_query_groups
    full_analysis.validate_query_groups=paired_query_contract
    try: yield
    finally: full_analysis.validate_query_groups=previous


def attribute_plan(plan,attribute,legacy_schema):
    result=deepcopy(plan)
    result['schema']=legacy_schema
    result['queries']=[q for q in result['queries'] if q['attribute']==attribute]
    paired_query_contract(result['queries'])
    result['items']=len(result['queries'])
    result['expected_score_rows']=len(result['queries'])*len(result['readouts'])*len(result['conditions'])
    result['expected_generation_rows']=sum(q['generation_selected'] for q in result['queries'])*len(result['readouts'])*sum(c['generation'] for c in result['conditions'])
    if 'calibration_anchors' in result:
        iids={q['intervention_id'] for q in result['queries']}
        result['calibration_anchors']={i:v for i,v in result['calibration_anchors'].items() if i in iids}
    return result


def validate_outer(records,generations,plan):
    if plan.get('schema')!=p2_design.SCHEMA or plan.get('analysis_arm') not in CONTEXTS:
        raise ValueError('Full P2 execution plan required')
    queries={q['id']:q for q in plan['queries']}
    if len(queries)!=len(plan['queries']): raise ValueError('Duplicate planned query')
    if {q['attribute'] for q in queries.values()}!={'color','stripe_orientation'}:
        raise ValueError('Parent plan must retain both paired attributes')
    panels={a:{q['scene_id'] for q in queries.values() if q['attribute']==a}
            for a in ('color','stripe_orientation')}
    if panels['color']!=panels['stripe_orientation']:
        raise ValueError('Both attributes must retain the same template set')
    conditions={c['name']:c for c in plan['conditions']}
    readouts={(r['response_format'],r['mapping']) for r in plan['readouts']}
    signatures={}
    for kind,rows in (('score',records),('generation',generations)):
        seen=set()
        for row in rows:
            k=key(row)
            if k in seen: raise ValueError('Duplicate execution attempt '+kind)
            seen.add(k)
            q=queries.get(row['id']); c=conditions.get(row['condition'])
            if q is None or c is None or (row['response_format'],row['mapping']) not in readouts:
                raise ValueError('Unplanned P2 record')
            if row.get('schema')!=p2_design.SCHEMA or row.get('kind')!=kind:
                raise ValueError('Wrong P2 record schema/kind')
            if row.get('analysis_arm')!=plan['analysis_arm']:
                raise ValueError('P2 arm mismatch')
            for f in ('scene_id','attribute','intervention_id'):
                if row.get(f)!=q[f]: raise ValueError('P2 identity mismatch: '+f)
            if row.get('template_id')!=q['scene_id']:
                raise ValueError('P2 template cluster mismatch')
            if q.get('geometry_signature') and row.get('geometry_signature')!=q['geometry_signature']:
                raise ValueError('P2 geometry signature mismatch')
            if row.get('condition_spec')!=c:
                raise ValueError('P2 condition metadata mismatch')
            if plan.get('execution_schema') and row.get('execution_schema')!=plan['execution_schema']:
                raise ValueError('P2 execution version mismatch')
            if plan.get('model') and row.get('model_key')!=plan['model']['key']:
                raise ValueError('P2 model mismatch')
            if kind=='generation' and (not q['generation_selected'] or not c['generation']):
                raise ValueError('Unplanned generation')
            signature={f:row[f] for f in ('source_pixel_sha256','source_image_sha256') if f in row}
            # All prompts/readouts share image inputs. Dose-specific k/anchor
            # equality is additionally enforced by accepted-v2 _index below.
            sigkey=q['intervention_id'],row['condition']
            if sigkey in signatures and signatures[sigkey]!=signature:
                raise ValueError('Query/readout changed source pixels')
            signatures[sigkey]=signature


def selectivity(effects,attribute,arm,kind):
    """E_target/E_control, R, and valid paired role-R differences per template."""
    contexts=CONTEXTS[arm]
    fields=('intervention_id','response_format','mapping','changed_entry')+contexts
    pairs=defaultdict(dict); out=[]
    for source in effects:
        row={**source,'attribute':attribute,'kind':kind,'template_id':source['scene_id'],
             'query_target':source.get('query_target',source.get('query_role'))}
        target=row['query_target']
        if target not in ('changed','unchanged'): raise ValueError('Unexpected query role in E')
        row['endpoint']='E_target' if target=='changed' else 'E_control'
        out.append(row)
        k=tuple(row[f] for f in fields)
        if target in pairs[k]: raise ValueError('Duplicate query effect in R')
        pairs[k][target]=row
    relative=[]
    for pair in pairs.values():
        if set(pair)!={'changed','unchanged'}: raise ValueError('Missing planned query in R')
        target,control=pair['changed'],pair['unchanged']
        relative.append({**target,'endpoint':'R','query_target':'changed_minus_unchanged',
                         'effect':subtract(target['effect'],control['effect'])})
    out+=relative
    role_pairs=defaultdict(dict)
    role_fields=('intervention_id','response_format','mapping')+contexts
    for row in relative:
        # Opposite-native normalized roles target DIFFERENT native doses.
        # Their matched-target R difference is made in the reference arm below.
        if arm=='dose' and row['dose_mode']!='raw': continue
        role_pairs[tuple(row[f] for f in role_fields)][row['changed_entry']]=row
    for pair in role_pairs.values():
        if set(pair)!={'main','later'}: raise ValueError('Missing planned entry in R difference')
        a,b=pair['main'],pair['later']
        out.append({**a,'endpoint':'R_main_minus_R_later','changed_entry':'main_minus_later',
                    'effect':subtract(a['effect'],b['effect'])})
    group=('attribute','kind','response_format','mapping','endpoint','changed_entry','query_target')+contexts
    return dict(rows=out,summaries=summarize(out,group),
        independence_unit='template_id (= scene_id); mean both edited objects only when all required pairs exist',
        uncertainty='Two-sided normal 95% interval across complete template means; no equivalence claim or significance stopping')


def source_four_cells(records,generations,plan,attribute):
    indexes={'score':{key(r):r for r in records},'generation':{key(r):r for r in generations}}
    cells={}
    for c in plan['conditions']:
        if c['role']=='source_cross':
            k=tuple(c[f] for f in ('main_layout','later_layout','main_strength','later_strength','main_assignment','later_assignment'))
            cells[k]=c
    physicals=sorted({k[:4] for k in cells})
    rows=[]
    for q,r,physical in product(plan['queries'],plan['readouts'],physicals):
        if q['attribute']!=attribute: continue
        for kind,index in indexes.items():
            if kind=='generation' and not q['generation_selected']: continue
            vals={}; ids={}
            for ma,la in product(('a','b'),repeat=2):
                c=cells[physical+(ma,la)]
                record=index.get((q['id'],r['response_format'],r['mapping'],c['name']))
                ids[ma+la]=c['name']
                if kind=='score':
                    vals[ma+la]=record['b_minus_a_log_odds'] if finite_score(record) else None
                else:
                    bchoice=('(A)','(B)')[q['direction_b_index']]
                    vals[ma+la]=int(record['original_prediction']==bchoice) if valid_generation(record) else None
            base={**{f:q[f] for f in ('id','scene_id','intervention_id','query_target')},**r,
                'attribute':attribute,'kind':kind,'template_id':q['scene_id'],
                **dict(zip(('main_layout','later_layout','main_strength','later_strength'),physical)),
                'physical_cells':ids,'cell_values':vals}
            j=vals['bb']-vals['ba']-vals['ab']+vals['aa'] if all(v is not None for v in vals.values()) else None
            rows.append({**base,'endpoint':'J_fixed_physical','effect':j})
            rows.append({**base,'endpoint':'D_fixed_physical','effect':subtract(vals['ba'],vals['ab'])})
    group=('attribute','kind','response_format','mapping','query_target','main_layout','later_layout','main_strength','later_strength','endpoint')
    return dict(rows=rows,summaries=summarize(rows,group),
        definitions={'J_fixed_physical':'y_bb-y_ba-y_ab+y_aa within fixed main/later layout and strengths',
                     'D_fixed_physical':'y_ba-y_ab within one fixed physical table; D_a=D_b is algebra, not replication',
                     'D_swap':'Retained separately in legacy query_contrasts.changed_entry; changing/fixed source layouts follow role exchange and may span P/P-prime'})


def reference_selectivity(reference,attribute,kind):
    effects=[]
    for c in reference['contrasts']:
        if not c[kind]['planned']: continue
        for role in ('main','later'):
            effects.append({**{f:c[f] for f in ('scene_id','intervention_id','query_target','response_format','mapping')+CONTEXTS['reference']},
                            'changed_entry':role,'effect':c[kind][role+'_effect']})
    return selectivity(effects,attribute,'reference',kind)


def analyze(records,generations,plan):
    records=list(records); generations=list(generations)
    verify_vendor(); validate_outer(records,generations,plan)
    arm=plan['analysis_arm']; outputs={}
    module={'local':local_analysis,'full':full_analysis,'dose':dose_analysis}[arm]
    for attribute in ('color','stripe_orientation'):
        projected=attribute_plan(plan,attribute,module.DESIGN_SCHEMA)
        scores=[{**r,'schema':module.DESIGN_SCHEMA} for r in records if r['attribute']==attribute]
        gens=[{**r,'schema':module.DESIGN_SCHEMA} for r in generations if r['attribute']==attribute]
        if arm=='full':
            with full_validator_bridge(): result=module.analyze(scores,gens,projected)
        else:
            result=module.analyze(scores,gens,projected)
        # Old prose describes old material scope (color/v8 or a central cross).
        # Reuse numerical estimators, not those historical scientific narratives.
        result.pop('interpretation',None)
        result.pop('contrast_definitions',None)
        effect_key='effects' if arm=='dose' else 'query_effects'
        behavior_key='behavior' if arm=='dose' else 'behavior_pairs'
        outputs[attribute]=dict(attribute=attribute,legacy_estimator_output=result,
            score_selectivity=selectivity(result[effect_key],attribute,arm,'score'),
            generation_selectivity=selectivity(result[behavior_key],attribute,arm,'generation'))
        if arm=='full':
            outputs[attribute]['source_four_cells']=source_four_cells(records,generations,plan,attribute)
        if arm=='dose':
            for kind in ('score','generation'):
                outputs[attribute][kind+'_matched_target_selectivity']=reference_selectivity(result['reference_contrasts'],attribute,kind)
    return dict(schema=SCHEMA,arm=arm,model=plan.get('model'),by_attribute=outputs,
        synthetic_fixture=bool(plan.get('synthetic_fixture',False)),
        model_results=bool(records or generations) and not plan.get('synthetic_fixture',False),
        independence_unit='template_id (=scene_id). Attributes are separate strata sharing templates, never independent scene counts',
        direction='score: logP(edited-b)-logP(edited-a); generation: indicator(original_prediction is edited-b)',
        exclusions='None based on native success, sign or significance. Missing/invalid/unmatched pairs remain null and incomplete templates stay in planned denominator',
        interval='normal 95% across complete template means, matching existing exploratory analysis; preserve n and raw pairs',
        projection='Only scientific schema name and per-attribute plan counts are projected; original source/condition/direction and dose provenance validations remain active',
        manuscript_written=False)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--scores',type=Path,nargs='+',required=True)
    p.add_argument('--generations',type=Path,nargs='*',default=[])
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(argv)
    load=lambda paths:[json.loads(line) for path in paths for line in path.read_text().splitlines() if line.strip()]
    result=analyze(load(a.scores),load(a.generations),json.loads(a.plan.read_text()))
    output=_guard_path(a.output,write=True); output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({attr:r['legacy_estimator_output']['coverage'] for attr,r in result['by_attribute'].items()},indent=2))


if __name__=='__main__': main()
