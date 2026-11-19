# V12 vendor copy of accepted archived v11 opposite-native-v2; upstream algorithm unchanged.
"""Versioned a-to-b difference-dose design; planning imports no tensor library."""
from copy import deepcopy
from itertools import product
import hashlib
import json
import math
from pathlib import Path

from cloud_prep.measurement_v10.full_data import source_name, validate_query_groups
from cloud_prep.measurement_v10.full_design import QUERY_FIELDS
from cloud_prep.measurement_v6.content_intervention import readout_specs

DESIGN_SCHEMA = 'relative-difference-dose-v11-opposite-native-v2'
DEFAULT_DESIGN = {
    'scene_limit': 16, 'smoke_scene_limit': 1,
    'query_types': ['identity'], 'query_targets': ['changed', 'unchanged'],
    'layout_pairs': [['M0', 'M0'], ['M1', 'M0']],
    'strength_pairs': [[100, 100], [8, 100], [100, 8]],
    'fixed_assignments': ['a', 'b'], 'entry_roles': ['main', 'later'],
    'regions': ['all'], 'k_values': [0, .25, .5, 1, 2, 4],
    'normalized_modes': ['baseline_state'], 'target_ratios': [.1, .3, 1.],
    'target_policy': 'fixed', 'dose_metric': 'rss', 'dose_metrics': None,
    'max_calibrated_k': 32., 'match_tolerance': .05,
    'response_formats': ['label', 'direct'],
    'generate_k_values': [0, 1, 4], 'generate_target_ratios': [.3],
    'self_patch': True,
}


def read_design(value=None):
    d = deepcopy(DEFAULT_DESIGN)
    if isinstance(value, (str, Path)): value = json.loads(Path(value).read_text())
    if value is not None:
        if not isinstance(value, dict) or set(value) - set(d): raise ValueError('Unknown dose design fields')
        d.update(value)
    for key in ('scene_limit', 'smoke_scene_limit'):
        if type(d[key]) is not int or not 1 <= d[key] <= 64: raise ValueError('Scene limits must be 1..64')
    for key, allowed in (
        ('query_types', {'identity', 'relation'}), ('query_targets', {'changed', 'unchanged'}),
        ('fixed_assignments', {'a', 'b'}), ('entry_roles', {'main', 'later'}),
        ('regions', {'all', 'roi'}), ('normalized_modes', {'baseline_state', 'pre_residual'}),
        ('response_formats', {'direct', 'label'})):
        if not isinstance(d[key], list) or len(d[key]) != len(set(d[key])) or set(d[key]) - allowed:
            raise ValueError('Invalid unique design axis: ' + key)
        if not d[key] and key != 'normalized_modes': raise ValueError('Empty design axis: ' + key)
    if 'identity' not in d['query_types'] or 'changed' not in d['query_targets']:
        raise ValueError('Canonical identity/changed calibration anchor must remain in selected queries')
    if d['target_policy'] not in ('fixed', 'opposite_native') or d['dose_metric'] not in ('rss', 'l1'):
        raise ValueError('Unknown target policy or dose metric')
    if d['dose_metrics'] is not None:
        if (not isinstance(d['dose_metrics'],list) or not d['dose_metrics']
            or len(d['dose_metrics'])!=len(set(d['dose_metrics']))
            or set(d['dose_metrics'])-{'rss','l1'}):
            raise ValueError('dose_metrics must be a nonempty unique RSS/L1 list')
    if d['target_policy'] == 'opposite_native':
        if set(d['entry_roles']) != {'main', 'later'} or set(d['normalized_modes']) - {'baseline_state'}:
            raise ValueError('Opposite-native requires both roles and baseline-state normalization')
    for key, allowed in (('layout_pairs', {'M0', 'M1'}), ('strength_pairs', {8, 100})):
        vals = d[key]
        if not vals or len({tuple(v) for v in vals}) != len(vals) or any(len(v) != 2 or set(v)-allowed for v in vals):
            raise ValueError('Invalid changing/fixed pair axis: ' + key)
    for key in ('k_values', 'target_ratios', 'generate_k_values', 'generate_target_ratios'):
        vals = d[key]
        if not isinstance(vals, list) or len(vals) != len(set(vals)) or any(type(x) not in (int, float) or not math.isfinite(x) or x < 0 for x in vals):
            raise ValueError('Unique finite nonnegative doses required')
    if not {0, 1} <= set(d['k_values']): raise ValueError('Raw k=0 and k=1 references are mandatory')
    if any(x <= 0 for x in d['target_ratios']): raise ValueError('Normalized targets must be positive')
    if set(d['generate_k_values']) - set(d['k_values']) or set(d['generate_target_ratios']) - set(d['target_ratios']):
        raise ValueError('Generation must be a planned dose subset')
    if not math.isfinite(d['max_calibrated_k']) or d['max_calibrated_k'] <= 0: raise ValueError('Positive finite k limit required')
    if not 0 < d['match_tolerance'] < 1 or d['self_patch'] is not True: raise ValueError('Bad matching tolerance or disabled required self-patch reference')
    return d


def select_rows(rows, panel='main', design=None):
    if isinstance(panel, dict): design, panel = panel, 'main'
    d = read_design(design)
    validate_query_groups(rows)
    if panel not in ('main', 'smoke'): raise ValueError('Unknown dose panel')
    by_scene = {r['scene_id']: r for r in rows}
    groups = {}
    for sid, r in by_scene.items():
        groups.setdefault((r['shape_pair'],r['layout'],r['assignment']), []).append(sid)
    ranked = {g:sorted(ids, key=lambda sid: hashlib.sha256(('dose-v11-subset-20261016:'+sid).encode()).hexdigest()) for g,ids in groups.items()}
    balanced = [ranked[g][rank] for rank in range(max(map(len,ranked.values()))) for g in sorted(ranked) if rank<len(ranked[g])]
    scenes = balanced[:d['scene_limit'] if panel == 'main' else d['smoke_scene_limit']]
    result = [r for r in rows if r['scene_id'] in scenes and r['query_type'] in d['query_types'] and r['query_target'] in d['query_targets']]
    return sorted(result, key=lambda r: (r['intervention_id'], r['query_type'], r['query_target'], r['id']))


def canonical_anchors(rows):
    anchors = {}
    for r in rows:
        if r['query_type'] == 'identity' and r['query_target'] == 'changed':
            iid = r['intervention_id']
            if iid in anchors: raise ValueError('Duplicate canonical anchor')
            anchors[iid] = r
    if set(anchors) != {r['intervention_id'] for r in rows}: raise ValueError('Every intervention must retain its canonical anchor')
    return anchors


def conditions(design=None, entry_groups=('all',), panel='main'):
    d = read_design(design)
    if tuple(entry_groups) != ('all',): raise ValueError('Dose design selects main or all later entries')
    cells = []
    for role, (cl, fl), (cs, fs), fixed, region in product(d['entry_roles'], d['layout_pairs'], d['strength_pairs'], d['fixed_assignments'], d['regions']):
        a, b, f = source_name(cl, cs, 'a'), source_name(cl, cs, 'b'), source_name(fl, fs, fixed)
        ref_a = {'main': a, 'later': f} if role == 'main' else {'main': f, 'later': a}
        ref_b = {'main': b, 'later': f} if role == 'main' else {'main': f, 'later': b}
        cid = f'{role}_{cl}_{cs:03d}__fixed_{fl}_{fs:03d}_{fixed}__{region}'
        base = dict(calibration_id=cid, changed_entry=role, changing_layout=cl, fixed_layout=fl,
                    changing_strength=cs/100, fixed_strength=fs/100, fixed_other_assignment=fixed,
                    main=ref_a['main'], donor_source=ref_a['later'], reference_a=ref_a, reference_b=ref_b,
                    main_layout=cl if role=='main' else fl, later_layout=fl if role=='main' else cl,
                    main_strength=(cs if role=='main' else fs)/100, later_strength=(fs if role=='main' else cs)/100,
                    region=region, entry_group='all', token_permutation='identity', main_kind='objects',
                    donor_pose='same', technical_audit=False, max_calibrated_k=d['max_calibrated_k'],
                    match_tolerance=d['match_tolerance'], target_policy=d['target_policy'])
        metrics = d['dose_metrics'] if d['dose_metrics'] is not None else [d['dose_metric']]
        doses = [('raw', k, k in d['generate_k_values'], 'raw') for k in d['k_values']]
        if d['target_policy'] == 'opposite_native':
            doses += [(mode, 'opposite_native', True, metric)
                for mode in d['normalized_modes'] for metric in metrics]
        else:
            doses += [(mode, v, v in d['generate_target_ratios'], metric)
                for mode in d['normalized_modes'] for v in d['target_ratios'] for metric in metrics]
        for mode, value, generate, metric in doses:
            label = value if isinstance(value, str) else f'{value:g}'
            pid = f'{cid}__{mode}_{label}__{d["target_policy"]}_{metric}'
            for side in ('a', 'b'):
                cells.append({**base, 'name':f'{pid}__{side}', 'pair_id':pid, 'role':'dose_pair',
                    'dose_mode':mode, 'dose_value':value, 'dose_metric':metric,
                    'donor_assignment':side, 'generation':generate})
    return cells


def required_sources(conditions):
    return {s for c in conditions for ref in (c['reference_a'], c['reference_b']) for s in ref.values()}


def calibration_key(row, condition):
    return row['intervention_id'], condition['calibration_id']


def opposite_condition(condition):
    """The other role with exactly the same changing/fixed source context."""
    c = deepcopy(condition)
    role = 'later' if c['changed_entry'] == 'main' else 'main'
    old = c['calibration_id']
    c['calibration_id'] = role + old[len(c['changed_entry']):]
    for field in ('pair_id', 'name'):
        c[field] = c[field].replace(old, c['calibration_id'], 1)
    c['changed_entry'] = role
    for ref in ('reference_a', 'reference_b'):
        c[ref] = {'main':condition[ref]['later'], 'later':condition[ref]['main']}
    c['main'], c['donor_source'] = c['reference_a']['main'], c['reference_a']['later']
    c['main_layout'], c['later_layout'] = condition['later_layout'], condition['main_layout']
    c['main_strength'], c['later_strength'] = condition['later_strength'], condition['main_strength']
    return c


def context_signature(row, condition):
    sources = {s for ref in (condition['reference_a'], condition['reference_b']) for s in ref.values()}
    payload = {'intervention_id':row['intervention_id'],
        'context':{k:condition[k] for k in ('changing_layout','fixed_layout','changing_strength',
            'fixed_strength','fixed_other_assignment','region')},
        'sources':{s:row['source_image_metadata'][s]['sha256'] for s in sorted(sources)},
        'direction':[row['direction_a_index'],row['direction_b_index']]}
    return hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()


def anchor_query_sha256(row):
    if 'query_sha256' in row:
        return row['query_sha256']  # Frozen full-plan query digest.
    payload = {'fields':{k:row[k] for k in QUERY_FIELDS},
        'language':{k:row.get(k) for k in ('query','prompt','choices','option_labels')}}
    return hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()


def visual_signature(row, condition):
    sources = {s for ref in (condition['reference_a'], condition['reference_b']) for s in ref.values()}
    payload = {'intervention_id':row['intervention_id'], 'calibration_id':condition['calibration_id'],
               'sources':{s:row['source_image_metadata'][s]['sha256'] for s in sorted(sources)},
               'reference_a':condition['reference_a'], 'reference_b':condition['reference_b'],
               'region':condition['region'], 'direction':[row['direction_a_index'], row['direction_b_index']]}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def execution_plan(rows, panel='main', mappings=('identity','cyclic'), generate_tokens=32, design=None, entry_groups=('all',)):
    d = read_design(design); anchors = canonical_anchors(rows)
    if not rows or len({r['id'] for r in rows}) != len(rows): raise ValueError('Unique nonempty rows required')
    expected_queries = set(product(d['query_types'],d['query_targets']))
    for iid in anchors:
        members = [r for r in rows if r['intervention_id']==iid]
        if len(members)!=len(expected_queries) or {(r['query_type'],r['query_target']) for r in members}!=expected_queries:
            raise ValueError('Each shard must retain every configured paired query for each intervention')
    if set(mappings)-{'identity','cyclic'} or len(mappings)!=len(set(mappings)): raise ValueError('Bad mappings')
    if type(generate_tokens) is not int or generate_tokens<0: raise ValueError('Bad generation budget')
    cells = conditions(d, entry_groups, panel); readouts = readout_specs(mappings,d)
    queries = [{**{k:r[k] for k in QUERY_FIELDS}, 'query_sha256':anchor_query_sha256(r),
                'generation_selected': bool(generate_tokens)} for r in rows]
    return dict(schema=DESIGN_SCHEMA, panel=panel, design=d, entry_groups=['all'], conditions=cells,
        design_sha256=hashlib.sha256(json.dumps(d,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
        normalized_dose_metrics=d['dose_metrics'] if d['dose_metrics'] is not None else [d['dose_metric']],
        raw_metric='raw',
        metric_contract='Raw k curve occurs once per role/context; only normalized conditions expand dose_metrics; canonical anchors are shared across metrics',
        queries=queries, readouts=readouts, mappings=list(mappings), items=len(rows), generate_tokens=generate_tokens,
        expected_score_rows=len(rows)*len(readouts)*len(cells),
        expected_generation_rows=len(rows)*len(readouts)*sum(c['generation'] for c in cells) if generate_tokens else 0,
        expected_anchor_forwards=len(anchors)*len({c['calibration_id'] for c in cells}),
        calibration_anchors={k:r['id'] for k,r in anchors.items()}, scene_templates=len({r['scene_id'] for r in rows}),
        selected_scene_ids=sorted({r['scene_id'] for r in rows}),
        selected_scene_strata={str(g):len({r['scene_id'] for r in rows if (r.get('shape_pair'),r.get('layout'),r.get('assignment'))==g})
            for g in sorted({(r.get('shape_pair'),r.get('layout'),r.get('assignment')) for r in rows})},
        selection='Balanced shape_pair/layout/assignment strata, fixed SHA256 rank of scene_id within each; no outcome filters; shared v10 scenes',
        direction_contract='a/b follows changed-object source colors, never a role-dependent sign or globally fixed red/blue',
        dose_contract='z_b(k)=z_a+k*(z_b-z_a), a fixed; one scalar k per role, all other entries/tokens fixed',
        calibration_contract='One identity/changed/direct anchor per intervention and visual condition, reused across all queries/readouts',
        normalization_contract='baseline_state uses a-side post-entry state; pre_residual is distinct and unavailable for Qwen main / zero for IBM layer0',
        independence_unit='scene_id; changed objects, questions, readouts and dose conditions are paired repeats')


def generation_selected(row, panel='main', design=None):
    return True


def record_fields(row, variant, condition, prediction):
    original = variant['label_to_original'][prediction] if prediction is not None else None
    return {**{k:row[k] for k in QUERY_FIELDS}, 'query_sha256':anchor_query_sha256(row), **{k:condition[k] for k in (
        'pair_id','calibration_id','changed_entry','changing_layout','fixed_layout','changing_strength','fixed_strength',
        'fixed_other_assignment','dose_mode','dose_value','target_policy','dose_metric','donor_assignment','region','entry_group')},
        'schema':DESIGN_SCHEMA, 'row_schema':row['schema'], 'family':row['family'], 'task':row['task'],
        'mapping':variant['mapping'], 'response_format':variant['response_format'],
        'condition':condition['name'], 'condition_spec':condition, 'main_source':condition['main'],
        'source_a_original_choice':row['option_labels'][row['direction_a_index']],
        'source_b_original_choice':row['option_labels'][row['direction_b_index']],
        'original_prediction':original, 'native_source':None, 'native_answer_label':None, 'native_correct':None,
        'content_prediction':row['choices'][row['option_labels'].index(original)] if original else None}
