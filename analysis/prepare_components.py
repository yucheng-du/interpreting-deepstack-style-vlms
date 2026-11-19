"""Direct-generation photo transitions and paired Qwen32 stripe composition."""
from collections import defaultdict
from analysis.paths import INPUT
import json
import statistics
import math

def finite(value):
    return type(value) in (int, float) and math.isfinite(value)
def stats(values):
    values = list(values)
    assert all(finite(v) for v in values)
    n = len(values)
    avg = statistics.mean(values) if n else None
    se = statistics.stdev(values) / math.sqrt(n) if n > 1 else None
    return dict(n=n, mean=avg, se=se,
        ci_low=avg - 1.96 * se if se is not None else None,
        ci_high=avg + 1.96 * se if se is not None else None,
        signs=dict(positive=sum(v > 0 for v in values), zero=sum(v == 0 for v in values),
                   negative=sum(v < 0 for v in values)))

def complete_units(rows, value_key='R'):
    by_unit = defaultdict(dict)
    for row in rows:
        unit, edit = row['template_id'], row['changed_index']
        if edit in by_unit[unit]:
            raise ValueError('Duplicate template/edit')
        by_unit[unit][edit] = row[value_key]
    return {unit: statistics.mean(edits.values()) for unit, edits in by_unit.items()
            if len(edits) == 2 and all(finite(v) for v in edits.values())}


def p1_analysis(path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    rows = [r for r in rows if r['endpoint'] == 'E' and r['main_layout'] == 'M0'
            and r['main_strength'] == 1 and r['response_format'] == 'direct'
            and r['mapping'] == 'none' and r['dose_mode'] == 'raw'
            and r['dose_metric'] == 'raw' and r['dose_value'] == 1]
    components = defaultdict(lambda: defaultdict(dict))
    for row in rows:
        key = row['changed_entry'] + '_' + ('target' if row['query_role'] == 'changed' else 'control')
        if row['intervention_id'] in components[key][row['scene_id']]:
            raise ValueError('Duplicate P1 query pair')
        components[key][row['scene_id']][row['intervention_id']] = row
        if finite(row['value']):
            assert row['value'] == row['b'] - row['a']
    keys = ['main_target', 'main_control', 'later_target', 'later_control']
    assert set(components) == set(keys)
    scenes = set(components[keys[0]])
    assert len(scenes) == 64 and all(set(components[k]) == scenes for k in keys)
    assert all(len(edits) == 2 for key in keys for edits in components[key].values())
    own_sets = {key: {scene for scene in scenes
        if all(finite(r['value']) for r in components[key][scene].values())} for key in keys}
    common = sorted(set.intersection(*(own_sets[k] for k in keys)))
    unit_values = {scene: {key: statistics.mean(r['value'] for r in components[key][scene].values())
                          for key in keys} for scene in common}
    for values in unit_values.values():
        values['R_main'] = values['main_target'] - values['main_control']
        values['R_later'] = values['later_target'] - values['later_control']
        values['delta_R'] = values['R_main'] - values['R_later']
    estimate = {key: stats(v[key] for v in unit_values.values())
                for key in keys + ['R_main', 'R_later', 'delta_R']}

    def flips(key, chosen):
        rr = [r for scene in chosen for r in components[key][scene].values()]
        valid = [r for r in rr if finite(r['value'])]
        return dict(query_pairs=len(rr), valid_query_pairs=len(valid),
            at_risk=sum(r['a'] == 0 for r in valid),
            forward=sum(r['value'] > 0 for r in valid), reverse=sum(r['value'] < 0 for r in valid),
            unchanged=sum(r['value'] == 0 for r in valid),
            photos=len(chosen), photos_any_change=sum(any(r['value'] != 0 for r in components[key][s].values()) for s in chosen))

    common_counts = {key: flips(key, common) for key in keys}
    own_counts = {key: flips(key, sorted(own_sets[key])) for key in keys}
    # Delta R is linear in the 16 binary endpoint indicators per photo. The
    # unknowns are observed invalid endpoint cells, not imputed effect values.
    known_total, unknown = 0.0, []
    invalid_by_arm = {}
    for key in keys:
        sign = {'main_target': 1, 'main_control': -1, 'later_target': -1, 'later_control': 1}[key]
        invalid_count = 0
        for scene in sorted(scenes):
            for intervention, row in components[key][scene].items():
                for side, coefficient in (('a', -sign), ('b', sign)):
                    value = row[side]
                    if finite(value):
                        assert value in (0, 1)
                        known_total += coefficient * value
                    else:
                        invalid_count += 1
                        unknown.append(dict(scene_id=scene, intervention_id=intervention,
                            component=key, endpoint=side, coefficient=coefficient))
        invalid_by_arm[key] = invalid_count
    denominator = 2 * len(scenes)
    lower = (known_total + sum(min(0, u['coefficient']) for u in unknown)) / denominator
    upper = (known_total + sum(max(0, u['coefficient']) for u in unknown)) / denominator
    return dict(condition=dict(layout='M0', alpha=1, readout='direct generation', dose='raw', k=1),
        inference_unit='source photo; two edited objects averaged within each photo',
        planned_photos=len(scenes), common_complete_photos=len(common), estimates=estimate,
        common_complete_counts=common_counts, component_complete_counts=own_counts,
        unit_values=unit_values, excluded_common_photos=sorted(scenes - set(common)),
        invalid_sensitivity=dict(lower=lower, upper=upper, invalid_endpoint_cells=len(unknown),
            binary_assignment_combinations=2 ** len(unknown), all_zero=known_total / denominator,
            all_one=(known_total + sum(u['coefficient'] for u in unknown)) / denominator,
            invalid_by_component=invalid_by_arm, invalid_cells=unknown,
            interpretation='Sharp point-estimate bounds over all binary assignments to invalid endpoints; not a confidence interval'))


def selectivity_index(data):
    groups = defaultdict(list)
    for row in data['paired_selectivity']:
        if row['mode'] == 'raw':
            groups[(row['attribute'], row['response_format'], row['mapping'], row['context'], row['group'])].append(row)
    return {key: complete_units(rows) for key, rows in groups.items()}


def qwen32_analysis(data):
    indexed = selectivity_index(data)
    contexts, units_by_context = {}, {}
    names = ['main', 'later0', 'later1', 'later2', 'all']
    for context in ('M0_s008', 'M1_s008', 'M0_s100', 'M1_s100'):
        values = {name: indexed[('stripe_orientation', 'direct', 'none', context, name)] for name in names}
        units = sorted(set.intersection(*(set(v) for v in values.values())))
        unit_values = {unit: {name: values[name][unit] for name in names} for unit in units}
        for v in unit_values.values():
            v['sum_singletons'] = sum(v[name] for name in ('later0', 'later1', 'later2'))
            v['C'] = v['all'] - v['sum_singletons']
        saved = complete_units([r for r in data['paired_contrasts']
            if r['contrast'] == 'raw_all_minus_sum_singletons' and r['attribute'] == 'stripe_orientation'
            and r['response_format'] == 'direct' and r['mapping'] == 'none'
            and r['mode'] == 'raw' and r['context'] == context], 'value')
        max_error = max(abs(unit_values[u]['C'] - saved[u]) for u in units)
        assert max_error < 1e-10
        contexts[context] = dict(estimates={name: stats(v[name] for v in unit_values.values())
            for name in names + ['sum_singletons', 'C']}, unit_values=unit_values,
            max_saved_C_reconstruction_error=max_error)
        units_by_context[context] = unit_values
    aligned = {}
    for alpha in ('s008', 's100'):
        m0, m1 = (units_by_context[f'M{m}_{alpha}'] for m in (0, 1))
        units = sorted(set(m0) & set(m1))
        contrasts = {u: {name: -m1[u][name] - m0[u][name]
                          for name in names + ['sum_singletons', 'C']} for u in units}
        aligned[alpha] = dict(definition='-value(M1)-value(M0)',
            estimates={name: stats(v[name] for v in contrasts.values()) for name in names + ['sum_singletons', 'C']},
            unit_values=contrasts)
    return dict(contexts=contexts, aligned_layout=aligned,
        interpretation='Output-scale nonadditivity; aligned layout contrasts are estimable without assuming component antisymmetry. Zero-containing intervals are not equivalence.')



def main():
    photos = {}
    for model in ('qwen8b','qwen32b','ibm4b','deepstack7b'):
        path = INPUT/'paired_endpoints'/model/'p1-paired_endpoints.jsonl'
        photos[model] = p1_analysis(path)
    stripes = {}
    for kind in ('score','generation'):
        path = INPUT/'decomposition'/'qwen32b'/f'{kind}.json'
        stripes[kind] = qwen32_analysis(json.loads(path.read_text()))
    (INPUT/'components.json').write_text(json.dumps(dict(p1_components=photos, qwen32_composition=stripes))+'\n')


if __name__ == '__main__':
    main()
