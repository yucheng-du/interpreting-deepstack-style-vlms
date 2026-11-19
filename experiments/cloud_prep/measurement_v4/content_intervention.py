"""Versioned source selection at Qwen's three actual post-merger DS entries.

Import and planning are torch-free. Tensor use is cloud-only in qwen_content.
"""
from __future__ import annotations
from itertools import product
from contextlib import contextmanager
from deepstack.slot_swap import validate_output

SCHEMA = 'content-counterfactual-v1'
GENERATION_CONDITIONS = ('native_masked', 'main_masked__ds_aaa', 'main_masked__ds_bbb',
                         'main_a__ds_aaa', 'main_a__ds_bbb', 'native_b')


def conditions():
    rows = [{'name': f'main_{main}__ds_{"".join(ds)}', 'main': main,
             'ds_sources': list(ds), 'role': 'source_factorial'}
            for main in ('masked', 'a') for ds in product(('a', 'b'), repeat=3)]
    return [*rows, {'name': 'native_masked', 'main': 'masked',
                    'ds_sources': ['masked'] * 3, 'role': 'attribute_absent_control'},
            {'name': 'native_b', 'main': 'b', 'ds_sources': ['b'] * 3,
             'role': 'native_capability_control'}]


def compose_visual_output(sources, condition, grids):
    """Keep each path index fixed; select actual encoded content, never zero/gain.

    Main and all three supplemental tensors are fresh clones. No image block,
    token, object location, answer label or DS merger index is permuted here.
    ``sources`` supports arbitrary names, including natural original/swapped.
    """
    names = [condition['main'], *condition['ds_sources']]
    if len(condition['ds_sources']) != 3 or any(name not in sources for name in names):
        raise ValueError('Specify a known main source and exactly three known DS sources')
    for name in set(names):
        validate_output(sources[name], grids)
    reference = sources[condition['main']][0]
    if any(sources[name][0].shape != reference.shape or sources[name][0].dtype != reference.dtype
           or sources[name][0].device != reference.device for name in set(names)):
        raise ValueError('All selected sources must share shape, dtype and device')
    return (reference.clone(),
            [sources[name][1][i].clone() for i, name in enumerate(condition['ds_sources'])])


def select_panel(rows, panel):
    if panel == 'main':
        return rows
    if panel == 'smoke':
        # Both attributes; targets differ. Fixed independently of model outcomes.
        return [next(r for r in rows if r['attribute'] == attribute and r['layout'] == 0
                     and r['nuisance'] == 0 and r['target_object'] == target)
                for attribute, target in (('color', 'circle'), ('stripe_orientation', 'square'))]
    raise ValueError('Unknown content panel')


def generation_selected(row, panel):
    return panel == 'smoke' or row['nuisance'] in (0, 1)


def execution_plan(rows, panel='main', mappings=('identity', 'cyclic'), generate_tokens=32):
    if not rows or len(set(mappings)) != len(mappings) or set(mappings) - {'identity', 'cyclic'}:
        raise ValueError('Nonempty rows and unique supported mappings are required')
    return {'schema': SCHEMA, 'panel': panel, 'items': len(rows),
            'scene_templates': len({r['scene_id'] for r in rows}),
            'conditions': conditions(), 'mappings': list(mappings),
            'expected_score_rows': len(rows) * len(mappings) * len(conditions()),
            'generation_items': sum(generation_selected(r, panel) for r in rows) if generate_tokens else 0,
            'expected_generation_rows': (sum(generation_selected(r, panel) for r in rows)
                                         * len(mappings) * len(GENERATION_CONDITIONS) if generate_tokens else 0),
            'generation_conditions': list(GENERATION_CONDITIONS), 'generate_tokens': generate_tokens,
            'independence_unit': 'scene_id; two queried objects and two mappings are paired repeats',
            'correctness_scope': 'native a and native b only; hybrids have source-direction endpoints'}


@contextmanager
def cloud_gpu_lock(campaign_root, gpu):
    """Match the v3 per-visible-device, nonblocking model lock; raises busy."""
    import fcntl
    from pathlib import Path
    from cloud_prep.measurement_v3.runtime import gpu_lock_path
    locks = Path(campaign_root) / 'locks'
    locks.mkdir(parents=True, exist_ok=True)
    with gpu_lock_path(locks, gpu).open('a') as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
