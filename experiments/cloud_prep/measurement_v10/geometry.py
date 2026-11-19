"""Check only the actual planned token operations, including whole natural images."""
from cloud_prep.measurement_v9.geometry import (
    row_tile_grid, qwen_token_indices, ibm_token_indices,
    qwen_region_indices, ibm_region_indices, image_geometry,
)


def preflight_geometry(row, conditions):
    sizes = {(m['height'], m['width']) for m in row['source_image_metadata'].values()}
    if len(sizes) != 1:
        raise ValueError('All source images must share one canvas')
    h, w = next(iter(sizes))
    if h % 32 or w % 32:
        raise ValueError('Canvas dimensions must be multiples of patch16/merge2')
    grid = row_tile_grid(row)
    grids = [[1, h // 16, w // 16]]
    geo = image_geometry((h, w))
    permutations = []
    for c in conditions:
        p = c.get('token_permutation', 'identity')
        if p not in permutations:
            permutations.append(p)
    counts = {}
    for p in permutations:
        qi = qwen_token_indices(grids, grid, p)
        bi = ibm_token_indices(geo, grid, p)
        box = row.get('donor_roi_bbox_xyxy')
        if box is not None and any(c.get('region') in ('roi', 'complement') for c in conditions):
            x0, y0, x1, y1 = box
            if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
                raise ValueError('ROI is outside the shared canvas')
            counts[str(p)] = {
                'qwen': len(qwen_region_indices(grids, grid, 'roi', box, [w, h], p)),
                'ibm': len(ibm_region_indices(geo, grid, 'roi', box, p)),
            }
    if counts and any(c != next(iter(counts.values())) for c in counts.values()):
        raise ValueError('Matched local movement changed nominal ROI support')
    return {'image_hw': [h, w], 'tile_grid': grid, 'qwen_grid_thw': grids,
            'qwen_visual_tokens': len(qi), 'ibm_geometry': geo.summary(),
            'permutations': permutations, 'nominal_roi_token_counts': counts,
            'status': 'CPU coordinate preflight only; actual processor and delivery await CUDA'}
