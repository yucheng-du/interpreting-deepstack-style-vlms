"""Main-layout x donor-pose x DS-permutation design (M x D x P); torch-free import and planning.

M0 is the original main image; M1 is its exact left/right pixel half swap and nothing
else. D (same/cross) stays defined by the ORIGINAL scene layout, so donors, query,
candidate mapping and the object-identity a/b direction are identical under M0/M1.

Cell order inside one layout is (same/I, same/P, cross/I, cross/P); the 8-cell
vector is [M0 cells, M1 cells]. Template signs come from one read-out model:
E > 0 iff the DS content read at position r, after permutation P, comes from the
target's slot in donor D, i.e. (r xor P) == (t xor D) with t the target's original
slot. main_reference reads r = t xor M (target's current main slot),
original_coordinate reads r = t, donor_identity reads the target object anywhere.
"""
from __future__ import annotations
from copy import deepcopy
from itertools import product
import json

from cloud_prep.measurement_v6.content_intervention import (POSES, PERMUTATIONS, donor_source,
    readout_specs, spatial_token_indices, audit_spatial_bridge)
# Re-exported for qwen_spatial (CUDA composition and the v3/v4 GPU lock).
from cloud_prep.measurement_v6.content_intervention import compose_visual_output, cloud_gpu_lock  # noqa: F401

DESIGN_SCHEMA = 'main-layout-reference-v7'
LAYOUTS = ('M0', 'M1')
CELLS = (('same', 'identity'), ('same', 'swap_halves'), ('cross', 'identity'), ('cross', 'swap_halves'))
CELL_NAMES = ('sI', 'sP', 'cI', 'cP')
VECTOR = tuple((m, d, p) for m in LAYOUTS for d, p in CELLS)
VECTOR_NAMES = tuple(f'{m}_{c}' for m in LAYOUTS for c in CELL_NAMES)
HYPOTHESES = ('main_reference', 'original_coordinate', 'donor_identity')
# Orthogonal 8-cell templates (each has squared norm 8). The identity template is
# the donor-identity prediction; the other two are the two spatial hypotheses.
COMPONENTS = ('identity', 'original_coordinate', 'main_reference')
COMPONENT_HYPOTHESIS = {'identity': 'donor_identity', 'original_coordinate': 'original_coordinate',
                        'main_reference': 'main_reference'}
DEFAULT_DESIGN = {'strengths': [0, 8], 'include_blank': False, 'response_formats': ['label', 'direct'],
                  'generation_groups': [0, 1], 'generate_native': True, 'generate_strengths': [0, 8],
                  'generate_blank': False}


def main_source(layout, strength):
    if layout not in LAYOUTS or type(strength) is not int or not 0 <= strength <= 100:
        raise ValueError('Main source needs layout M0/M1 and integer strength 0..100')
    return f'm_s{strength:03d}' if layout == 'M0' else f'm1_s{strength:03d}'


def predicted_sign(hypothesis, layout, pose, permutation):
    """Sign of E for one cell under a simplified read-out hypothesis (not a fitted model)."""
    if layout not in LAYOUTS or pose not in POSES or permutation not in PERMUTATIONS:
        raise ValueError('Unknown cell')
    m, d, p = layout == 'M1', pose == 'cross', permutation == 'swap_halves'
    if hypothesis == 'donor_identity':
        return 1
    read = m if hypothesis == 'main_reference' else False if hypothesis == 'original_coordinate' else None
    if read is None:
        raise ValueError('Unknown hypothesis')
    # Slots relative to the target's original slot t: read t^read, DS source t^read^p, target t^d.
    return 1 if (read ^ p) == d else -1


def four_cell_prediction(hypothesis, layout):
    return tuple(predicted_sign(hypothesis, layout, d, p) for d, p in CELLS)


def template_vector(component):
    return tuple(predicted_sign(COMPONENT_HYPOTHESIS[component], *cell) for cell in VECTOR)


TEMPLATES = {name: template_vector(name) for name in COMPONENTS}


def layout_interaction(cells):
    """(cP-cI)-(sP-sI) for one layout's (sI, sP, cI, cP)."""
    si, sp, ci, cp = cells
    return (cp - ci) - (sp - si)


def decompose(vector):
    """Projection of the 8-cell vector on the orthogonal templates.

    component_k = (t_k . E) / (t_k . t_k) = (t_k . E) / 8, the least-squares
    coefficient, so a pure template E = beta * t_k gives component_k = beta.
    Consequently I_M0 - I_M1 = 8 * main_reference, I_M0 + I_M1 = 8 *
    original_coordinate and the cell sum = 8 * identity. The residual spans the
    remaining five factorial contrasts (M, D, P main effects, M x D, M x P).
    """
    vector = [float(x) for x in vector]
    if len(vector) != len(VECTOR):
        raise ValueError('Need the complete 8-cell vector')
    comps = {k: sum(t * x for t, x in zip(T, vector)) / sum(t * t for t in T) for k, T in TEMPLATES.items()}
    fitted = [sum(comps[k] * TEMPLATES[k][i] for k in COMPONENTS) for i in range(len(vector))]
    total = sum(x * x for x in vector)
    residual = sum((x - f) ** 2 for x, f in zip(vector, fitted))
    m0, m1 = layout_interaction(vector[:4]), layout_interaction(vector[4:])
    return {'components': comps, 'residual_sum_squares': residual, 'total_sum_squares': total,
            'explained_fraction': {k: (8 * comps[k] ** 2 / total if total else None) for k in COMPONENTS},
            'interaction_M0': m0, 'interaction_M1': m1, 'three_way_interaction': m0 - m1,
            'interaction_sum': m0 + m1,
            'm_flip_delta': {name: vector[4 + i] - vector[i] for i, name in enumerate(CELL_NAMES)}}


def sign_pattern(vector):
    return ''.join('+' if x > 0 else '-' if x < 0 else '0' for x in vector)


def classify(vector):
    """Exact sign-template matches and the best cosine-correlated template."""
    pattern = sign_pattern(vector)
    exact = [k for k, T in TEMPLATES.items() if pattern == sign_pattern(T)]
    norm = sum(x * x for x in vector) ** .5
    cosine = {k: (sum(t * x for t, x in zip(T, vector)) / (norm * 8 ** .5) if norm else 0.) for k, T in TEMPLATES.items()}
    best = max(COMPONENTS, key=lambda k: cosine[k])
    return {'sign_pattern': pattern, 'exact_template': exact[0] if exact else None,
            'best_template': best if cosine[best] > 0 else None, 'cosine': cosine}


def read_design(path=None):
    design = deepcopy(DEFAULT_DESIGN)
    if path is not None:
        supplied = json.loads(path.read_text()) if hasattr(path, 'read_text') else path
        if not isinstance(supplied, dict):
            raise ValueError('Design config must be a JSON object')
        if set(supplied) - set(design):
            raise ValueError('Unknown design fields: ' + str(sorted(set(supplied) - set(design))))
        design.update(supplied)
    for key in ('strengths', 'generate_strengths'):
        if not isinstance(design[key], list) or any(type(x) is not int or not 0 <= x <= 100 for x in design[key]) \
                or len(set(design[key])) != len(design[key]):
            raise ValueError('Unique integer strengths in 0..100 required')
    if not design['strengths']:
        raise ValueError('Need at least one main strength')
    if set(design['generate_strengths']) - set(design['strengths']):
        raise ValueError('generate_strengths must be a subset of strengths')
    formats = design['response_formats']
    if not isinstance(formats, list) or not formats or len(set(formats)) != len(formats) or set(formats) - {'label', 'direct'}:
        raise ValueError('Use label/direct response formats')
    groups = design['generation_groups']
    if not isinstance(groups, list) or any(type(x) is not int or x < 0 for x in groups) or len(set(groups)) != len(groups):
        raise ValueError('Unique nonnegative generation groups required')
    for key in ('include_blank', 'generate_native', 'generate_blank'):
        if type(design[key]) is not bool:
            raise ValueError('Boolean design flags required')
    if design['generate_blank'] and not design['include_blank']:
        raise ValueError('generate_blank requires include_blank')
    return design


def conditions(design=None):
    """All M x D x P x {aaa,bbb} cells per object strength, natives and real donors.

    Both layouts and the full D x P grid are not configurable: dropping any of the
    eight cells would remove the paired M comparison. The uniform blank is
    layout-invariant (its half swap is pixel-identical), so it is run under M0 only.
    """
    design = design or read_design()
    mains = [(main_source(layout, s), layout, 'objects', s / 100, False) for s in design['strengths'] for layout in LAYOUTS]
    if design['include_blank']:
        mains.append(('blank', 'M0', 'uniform_blank', 0., True))
    output = []
    for source, layout, kind, strength, invariant in mains:
        generate = design['generate_blank'] if kind == 'uniform_blank' else round(strength * 100) in design['generate_strengths']
        base = {'main': source, 'main_layout': layout, 'main_kind': kind, 'main_strength': strength,
                'layout_invariant_main': invariant}
        for pose, perm, value in product(POSES, PERMUTATIONS, ('a', 'b')):
            output.append({'name': f'{source}__{pose}__{perm}__{value * 3}', 'role': 'routing_cell', **base,
                'donor_pose': pose, 'token_permutation': perm, 'ds_sources': [donor_source(pose, value)] * 3,
                'ds_code': value * 3, 'generation': generate,
                'template_signs': {h: predicted_sign(h, layout, pose, perm) for h in HYPOTHESES}})
        output.append({'name': f'native_{source}', 'role': 'native_main', **base, 'donor_pose': None,
            'token_permutation': 'identity', 'ds_sources': [source] * 3, 'ds_code': None,
            'generation': design['generate_native'], 'template_signs': None})
    for pose, value in product(POSES, ('a', 'b')):
        source = donor_source(pose, value)
        output.append({'name': f'real_{source}', 'role': 'real_donor', 'main': source, 'main_layout': None,
            'main_kind': 'objects', 'main_strength': 1., 'layout_invariant_main': False, 'donor_pose': pose,
            'token_permutation': 'identity', 'ds_sources': [source] * 3, 'ds_code': None,
            'generation': design['generate_native'], 'template_signs': None})
    return output


def main_pairs(design):
    """(M0, M1) source pairs whose processor patch rows must be exact half-swap permutations."""
    return [(main_source('M0', s), main_source('M1', s)) for s in design['strengths']]


def generation_selected(row, panel, design=None):
    return panel == 'smoke' or row['generation_group'] in (design or DEFAULT_DESIGN)['generation_groups']


def execution_plan(rows, panel='main', mappings=('identity', 'cyclic'), generate_tokens=32, design=None, family=None):
    design = design or read_design()
    if not rows or len(set(mappings)) != len(mappings) or set(mappings) - {'identity', 'cyclic'}:
        raise ValueError('Invalid rows/mappings')
    cells = conditions(design)
    readouts = readout_specs(mappings, design)
    gen_items = sum(generation_selected(r, panel, design) for r in rows) if generate_tokens else 0
    return {'schema': DESIGN_SCHEMA, 'family': family, 'row_schema': sorted({r['schema'] for r in rows}),
        'design': design, 'panel': panel, 'items': len(rows), 'scene_templates': len({r['scene_id'] for r in rows}),
        'layouts': list(LAYOUTS), 'cells': [dict(zip(('main_layout', 'donor_pose', 'token_permutation'), c)) for c in VECTOR],
        'conditions': cells, 'mappings': list(mappings), 'readouts': readouts,
        'expected_score_rows': len(rows) * len(readouts) * len(cells), 'generation_items': gen_items,
        'expected_generation_rows': gen_items * len(readouts) * sum(c['generation'] for c in cells),
        'generate_tokens': generate_tokens,
        'templates': {'cell_order': list(VECTOR_NAMES), **{k: list(v) for k, v in TEMPLATES.items()}},
        'independence_unit': 'scene_id by default (source cluster for recomposed natural objects); queries, mappings, formats, strengths, M, D and P are repeated conditions. M1 reuses the same scenes; it is not a new sample.',
        'score_scope': 'Complete candidate-text sequence logprob; no length normalization. Compare within response format; never pool label/direct magnitudes.',
        'correctness_scope': 'Native nonzero-strength mains (M0 and M1; object answer unchanged by M) and real donors only; gray, blank and hybrid cells have no attribute gold'}


def audit_main_layout_bridge(model, source_inputs, sources, grids, pairs):
    """CUDA smoke: v6 donor/rotary/inverse checks plus exact M1 processor patch permutation.

    Feature equivariance is not required: the descriptive max difference between
    permuted M0 features and M1 features only records how far contextualized ViT
    outputs are from a pure token permutation.
    """
    import torch
    checks = audit_spatial_bridge(model, source_inputs, sources, grids)
    indices = spatial_token_indices(grids)
    device = next(iter(source_inputs.values()))['pixel_values'].device
    patch_ids = torch.tensor([j * 4 + k for j in indices for k in range(4)], device=device)
    layout = []
    for m0, m1 in pairs:
        equal = torch.equal(source_inputs[m0]['pixel_values'].index_select(0, patch_ids), source_inputs[m1]['pixel_values'])
        if not equal:
            raise RuntimeError('Actual processor patch order does not implement the M1 main half swap')
        ids = torch.tensor(indices, device=sources[m0][0].device)
        diffs = {name: float((a.index_select(0, ids).float() - b.float()).abs().max().item())
                 for name, a, b in zip(('main', 'path0', 'path1', 'path2'), (sources[m0][0], *sources[m0][1]), (sources[m1][0], *sources[m1][1]))}
        layout.append({'m0': m0, 'm1': m1, 'processor_pixel_permutation_equal': True,
                       'feature_permutation_max_abs_difference_descriptive': diffs})
    return {**checks, 'main_layout_checks': layout, 'main_feature_equivariance_required': False}
