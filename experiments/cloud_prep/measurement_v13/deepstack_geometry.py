"""Original DeepStack-L nominal patch supports, using official 2D phase sampling.

No Qwen feature-depth interpretation, tensors, image decoding or model imports.
The source image is resized (not letterboxed) by the native default processor.
Contextual CLIP features do not have strictly local receptive fields.
"""
import math


def validate_config(config):
    required = {'model_type': 'deepstack_llama', 'image_aspect_ratio': 'deepstack',
                'num_crops': 4, 'mm_vision_select_layer': -2,
                'mm_vision_select_feature': 'patch', 'mm_patch_merge_type': 'flat',
                'hidden_size': 4096, 'twoviews_unshare': False}
    if any(config.get(k) != v for k, v in required.items()):
        raise ValueError('Expected native DeepStack-L Vicuna-7B geometry')
    if (config.get('is_anyres_deepstack', False)
            or config.get('deepstack_args_for_ablation')
            or config.get('deepstack_sampling', '2d') != '2d'):
        raise ValueError('Ablation/HD/non-2D preprocessing needs a separate geometry version')
    return dict(main_tokens=576, later_entries=[config.get('deepstack_start_id', 1)
        + i * config.get('deepstack_interval', 1) for i in range(4)],
        configured_later_groups=4, patch_size=14, crop_size=336)


def best_grid(image_size, num_crops=4, crop_size=336):
    """Official resolution objective, order and tie breaking, in plain Python."""
    width, height = image_size
    if min(width, height) <= 0:
        raise ValueError('Positive image width/height required')
    best, max_effective, min_waste = None, 0, math.inf
    for gw in range(1, num_crops + 1):
        for gh in range(1, num_crops // gw + 1):
            if (gw, gh) == (1, 1):
                continue
            w, h = gw * crop_size, gh * crop_size
            scale = min(w / width, h / height)
            effective = min(int(width * scale) * int(height * scale), width * height)
            waste = w * h - effective
            if effective > max_effective or (effective == max_effective and waste < min_waste):
                best, max_effective, min_waste = (gw, gh), effective, waste
    if best is None:
        raise ValueError('No native DeepStack resolution')
    return best


def entry_supports(image_size, entry='main', *, grid=None, patches=24, num_crops=4):
    """Return each slot's xyxy nominal source support, or None for padded groups."""
    width, height = image_size
    gw, gh = grid or best_grid(image_size, num_crops)
    if gw * gh > num_crops or min(gw, gh) < 1:
        raise ValueError('Invalid native crop grid')
    if entry == 'main':
        return [(c * width / patches, r * height / patches,
                 (c + 1) * width / patches, (r + 1) * height / patches)
                for r in range(patches) for c in range(patches)]
    if not isinstance(entry, int) or not 0 <= entry < num_crops:
        raise ValueError('later entry must be its zero-based native phase-group index')
    if entry >= gw * gh:
        return [None] * (patches * patches)
    offset_y, offset_x = divmod(entry, gw)
    return [((c * gw + offset_x) * width / (patches * gw),
             (r * gh + offset_y) * height / (patches * gh),
             (c * gw + offset_x + 1) * width / (patches * gw),
             (r * gh + offset_y + 1) * height / (patches * gh))
            for r in range(patches) for c in range(patches)]


def token_order(permutation='identity', patches=24):
    if permutation == 'identity':
        return list(range(patches * patches))
    if permutation not in ('swap_halves', 'swap') or patches % 2:
        raise ValueError('Only identity/left-right half-swap belongs to this adapter')
    return [r * patches + (c + patches // 2) % patches
            for r in range(patches) for c in range(patches)]


def roi_indices(image_size, bbox, entry='main', *, grid=None, permutation='identity', patches=24):
    width, height = image_size
    x0, y0, x1, y1 = bbox
    if not all(math.isfinite(v) for v in (*bbox, width, height)) or not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError('ROI must lie within its donor canvas')
    supports = entry_supports(image_size, entry, grid=grid, patches=patches)
    selected = {i for i, b in enumerate(supports) if b is not None
                and b[0] < x1 and b[2] > x0 and b[1] < y1 and b[3] > y0}
    return [dest for dest, src in enumerate(token_order(permutation, patches)) if src in selected]


def geometry_record(image_size, bbox=None):
    grid = best_grid(image_size)
    return dict(image_size=list(image_size), native_grid=list(grid), main_tokens=576,
        configured_later_groups=4, active_later_groups=grid[0] * grid[1],
        zero_padded_later_groups=list(range(grid[0] * grid[1], 4)),
        source_support='global resized patch grid / interleaved native high-resolution phase groups',
        roi_selected=None if bbox is None else {str(e): len(roi_indices(image_size, bbox, e, grid=grid))
            for e in ('main', 0, 1, 2, 3)})
