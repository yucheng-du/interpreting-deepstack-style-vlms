"""Paired image swaps and score checks for visual-similarity comparisons."""
from copy import deepcopy
import json
import math

from deepstack.slot_swap import slot_pair


def image_variant(row, condition):
    result = deepcopy(row)
    if condition == 'original':
        return result
    if condition != 'candidate_swap':
        raise ValueError('Unsupported image treatment')
    pair = slot_pair(row)
    supplied = row.get('v3_slot_plan', {}).get('swap_pair')
    if supplied is not None and list(pair) != supplied:
        raise ValueError('Prepared swap pair changed')
    order = list(range(len(row['images'])))
    a, b = pair
    order[a], order[b] = order[b], order[a]
    slots = [1, 2] if row['task'] == 'Visual_Similarity' else [0, 1, 2, 3]
    original_slot = slots[row['answer_index']]
    answer_index = slots.index(order.index(original_slot))
    result.update(images=[row['images'][i] for i in order],
                  image_metadata=[row['image_metadata'][i] for i in order],
                  answer_index=answer_index, answer_label=row['option_labels'][answer_index],
                  raw_answer=row['option_labels'][answer_index],
                  image_treatment={'name':condition, 'order':order, 'pair':list(pair),
                                   'original_gold':row['answer_label']})
    # All images are really re-encoded. No same-grid restriction is needed.
    return result


def emit(handle, record):
    handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False)+'\n')


def score_check(left, right, atol=0.001):
    if set(left) != set(right):
        raise ValueError('Semantic check changed label set')
    delta = max(abs(left[k]-right[k]) for k in left)
    if not math.isfinite(delta) or delta > atol:
        raise RuntimeError(f'Native/full/self-patch mismatch: {delta} > {atol}; inspect semantics')
    return delta
