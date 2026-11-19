"""Complete post-merger image-slot controls for Qwen3-VL (torch-free import).

Tensor helpers import torch only when called. Image blocks retain their original within-image token order.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import math
import random

TASKS = ('Visual_Similarity', 'Forensic_Detection')
SCHEMA = 'qwen-slot-swap-v1'
SEED = 20261003
_MISSING = object()


def conditions():
    return [{'name': f'main_{main}__ds_{ds}', 'main': main, 'deepstack': ds}
            for main in ('original', 'swapped')
            for ds in ('original', 'swapped', 'unrelated')]


def stable_seed(seed, item_id, role):
    return int.from_bytes(hashlib.sha256(f'{seed}|{item_id}|{role}'.encode()).digest()[:8], 'big')


def expected_grid(metadata, *, min_pixels=4096, max_pixels=262144, patch_size=16, merge_size=2):
    """Metadata-only Qwen2VL smart_resize prediction, checked against real grids.

    Mirrors the pinned Transformers 4.57.6 slow processor's smart_resize;
    never treats this prediction as a substitute for runtime grid equality.
    """
    h, w = metadata['height'], metadata['width']
    if min(h, w, min_pixels, patch_size, merge_size) <= 0 or max_pixels < min_pixels:
        raise ValueError('Positive geometry and ordered pixel bounds required')
    if max(h, w) / min(h, w) > 200:
        raise ValueError('Unsupported aspect ratio')
    factor = patch_size * merge_size
    hh, ww = round(h / factor) * factor, round(w / factor) * factor
    if hh * ww > max_pixels:
        beta = math.sqrt(h * w / max_pixels)
        hh = max(factor, math.floor(h / beta / factor) * factor)
        ww = max(factor, math.floor(w / beta / factor) * factor)
    elif hh * ww < min_pixels:
        beta = math.sqrt(min_pixels / (h * w))
        hh, ww = math.ceil(h * beta / factor) * factor, math.ceil(w * beta / factor) * factor
    return [1, hh // patch_size, ww // patch_size]


def image_token_counts(grids, merge_size):
    if not grids or type(merge_size) is not int or merge_size < 1:
        raise ValueError('Need complete positive grid geometry')
    result = []
    for grid in grids:
        if len(grid) != 3 or any(type(v) is not int or v < 1 for v in grid):
            raise ValueError('Every image grid must contain positive integer t,h,w')
        t, h, w = grid
        if h % merge_size or w % merge_size:
            raise ValueError('Image grid does not divide into merged spatial blocks')
        result.append(t * h * w // merge_size**2)
    return result


def slot_pair(row, *, seed=SEED):
    """Fix the pair once, independent of mapping, condition and model outputs."""
    if row['task'] == 'Visual_Similarity':
        if len(row['images']) != 3 or [c.casefold().strip() for c in row['choices']] != ['the second image', 'the third image']:
            raise ValueError('VS must have one reference followed by two candidate images')
        return (1, 2)
    if row['task'] == 'Forensic_Detection':
        expected = ['the first image', 'the second image', 'the third image', 'the fourth image']
        if len(row['images']) != 4 or [c.casefold().strip() for c in row['choices']] != expected:
            raise ValueError('Forensic must have four image-slot choices')
        correct = row['answer_index']
        other = random.Random(stable_seed(seed, row['id'], 'forensic_other_slot')).choice(
            [i for i in range(4) if i != correct])
        return (correct, other)
    raise ValueError('Slot swapping supports VS and Forensic only')


def slot_order(grids, pair):
    a, b = pair
    if a == b or not 0 <= a < len(grids) or not 0 <= b < len(grids):
        raise ValueError('Need two distinct valid image slots')
    if list(grids[a]) != list(grids[b]):
        raise ValueError('Swapped slots require identical full (t,h,w), not just token count')
    order = list(range(len(grids)))
    order[a], order[b] = order[b], order[a]
    return order


def block_indices(grids, order, *, merge_size=2):
    counts = image_token_counts(grids, merge_size)
    if sorted(order) != list(range(len(grids))):
        raise ValueError('Order must be a permutation of all image slots')
    if any(list(grids[target]) != list(grids[source]) for target, source in enumerate(order)):
        raise ValueError('Whole-image swaps require identical full grid at each destination')
    starts = [0]
    for count in counts:
        starts.append(starts[-1] + count)
    return [i for source in order for i in range(starts[source], starts[source + 1])]


def plan_item(row, pool, *, seed=SEED, min_pixels=4096, max_pixels=262144):
    grids = [expected_grid(m, min_pixels=min_pixels, max_pixels=max_pixels) for m in row['image_metadata']]
    pair = slot_pair(row, seed=seed)
    order = slot_order(grids, pair)
    target_hashes = {m['sha256'] for m in row['image_metadata']}
    candidates = []
    for candidate in pool:
        if candidate['task'] != row['task'] or candidate['id'] == row['id']:
            continue
        if target_hashes & {m['sha256'] for m in candidate['image_metadata']}:
            continue
        other = [expected_grid(m, min_pixels=min_pixels, max_pixels=max_pixels)
                 for m in candidate['image_metadata']]
        if other == grids:
            candidates.append(candidate['id'])
    if not candidates:
        raise ValueError('No same-task donor with exact full-grid sequence and disjoint image hashes')
    donor = row.get('v3_donor_id')
    if donor is not None and donor not in candidates:
        raise ValueError('Declared v3_donor_id fails same-task/full-grid/disjoint-image requirements')
    if donor is None:
        donor = random.Random(stable_seed(seed, row['id'], 'unrelated_donor')).choice(sorted(candidates))
    supplied = row.get('v3_slot_plan', {})
    if supplied and (supplied.get('image_grid_thw') != grids or supplied.get('swap_pair') != list(pair)
                     or supplied.get('seed') != seed):
        raise ValueError('Prepared v3_slot_plan disagrees with current full-grid/seed/swap plan')
    choice_slots = [1, 2] if row['task'] == 'Visual_Similarity' else [0, 1, 2, 3]
    correct_slot = choice_slots[row['answer_index']]
    swapped_choice = choice_slots.index(order.index(correct_slot))
    return {'id': row['id'], 'task': row['task'], 'seed': seed, 'swap_pair': list(pair),
            'slot_order': order, 'expected_grid_thw': grids, 'donor_id': donor,
            'original_correct_choice': row['answer_label'],
            'swapped_correct_choice': row['option_labels'][swapped_choice],
            'choice_image_slots': choice_slots,
            'donor_rule': 'same task, complete grid sequence, disjoint image SHA; seed-selected once'}


def validate_output(output, grids, *, merge_size=2):
    main, features = output
    count = sum(image_token_counts(grids, merge_size))
    if len(features) != 3 or main.ndim != 2 or main.shape[0] != count:
        raise ValueError('Expected one main tensor and three DS tensors with all merged image tokens')
    if any(x.shape != main.shape or x.dtype != main.dtype or x.device != main.device for x in features):
        raise ValueError('Main/DS tensors must share [visual_tokens, hidden], dtype and device')


def reorder_tensor(tensor, grids, order, *, merge_size=2):
    import torch
    indices = block_indices(grids, order, merge_size=merge_size)
    if tensor.ndim != 2 or tensor.shape[0] != len(indices):
        raise ValueError('Post-merger tensor and complete image grid disagree')
    return tensor.index_select(0, torch.tensor(indices, device=tensor.device))


def intervention_output(original, donor, grids, pair, condition, *, merge_size=2):
    """All allocations are fresh; no source cache is mutated or resized."""
    if condition not in conditions():
        raise ValueError('Unknown slot condition')
    validate_output(original, grids, merge_size=merge_size)
    validate_output(donor, grids, merge_size=merge_size)
    order = slot_order(grids, pair)
    main, features = original
    main_order = order if condition['main'] == 'swapped' else list(range(len(grids)))
    main = reorder_tensor(main, grids, main_order, merge_size=merge_size)
    if condition['deepstack'] == 'unrelated':
        ds = [x.clone() for x in donor[1]]
    else:
        ds_order = order if condition['deepstack'] == 'swapped' else list(range(len(grids)))
        ds = [reorder_tensor(x, grids, ds_order, merge_size=merge_size) for x in features]
    return main, ds


def feature_norms(output, grids, *, merge_size=2):
    counts = image_token_counts(grids, merge_size)
    return [{'stream': name, 'global_norm': float(x.float().norm().item()),
             'image_block_norms': [float(part.float().norm().item()) for part in x.split(counts)]}
            for name, x in zip(('main', 'path0', 'path1', 'path2'), (output[0], *output[1]))]


def compare_outputs(left, right):
    import torch
    a, b = (left[0], *left[1]), (right[0], *right[1])
    if len(a) != len(b) or any(x.shape != y.shape for x, y in zip(a, b)):
        raise ValueError('Feature layout changed')
    return [{'stream': name, 'equal': bool(torch.equal(x, y)),
             'max_abs_difference': float((x.float() - y.float()).abs().max().item())}
            for name, x, y in zip(('main', 'path0', 'path1', 'path2'), a, b)]


def encode_visual(model, inputs):
    """Actual vision call, outside every replacement/cache context."""
    visual = model.model.visual
    if getattr(visual, '_slot_swap_owner', None) is not None or 'forward' in visual.__dict__:
        raise RuntimeError('Real encoding requires unpatched visual.forward (no stale target cache)')
    return visual(inputs['pixel_values'].to(dtype=visual.dtype), grid_thw=inputs['image_grid_thw'])


@contextmanager
def use_visual_output(model, output, grids, *, merge_size=2):
    """Deliver one cached visual output; language state is always recomputed.

    Does not encode images. In particular donor encode must finish before entry.
    Restoration preserves the instance's original attribute state on exceptions.
    """
    import torch
    visual = model.model.visual
    if getattr(visual, '_slot_swap_owner', None) is not None or 'forward' in visual.__dict__:
        raise RuntimeError('Nested or foreign visual cache is not supported')
    validate_output(output, grids, merge_size=merge_size)
    original = visual.__dict__.get('forward', _MISSING)
    state = {'calls': 0}
    def forward(hidden_states, grid_thw=None, **kwargs):
        if grid_thw is None or grid_thw.detach().cpu().tolist() != grids:
            raise ValueError('Forward grid differs from the cached output full geometry')
        if hidden_states.shape[0] != sum(t*h*w for t,h,w in grids):
            raise ValueError('Pixel patch count differs from the cached full geometry')
        state['calls'] += 1
        return output
    visual._slot_swap_owner = state
    visual.forward = forward
    try:
        yield state
    finally:
        if original is _MISSING:
            delattr(visual, 'forward')
        else:
            visual.forward = original
        delattr(visual, '_slot_swap_owner')
