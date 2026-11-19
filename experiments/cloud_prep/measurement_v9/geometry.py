"""Exact output-to-source tile maps in actual processor coordinates; no tensors.

The same gather order moves feature values and source ROI membership. IBM
newline/padding positions remain fixed; nominal slots are not receptive fields.
"""
from __future__ import annotations

from cloud_prep.measurement_v8.ibm_geometry import roi_token_indices as ibm_roi, image_geometry
from cloud_prep.measurement_v8.qwen_adapter import token_geometry, roi_token_indices as qwen_roi


def tile_order(tile_grid, permutation='identity'):
    if len(tile_grid) != 2 or any(type(x) is not int or x < 1 for x in tile_grid):
        raise ValueError('Positive integer tile_grid=[rows,cols] required')
    count = tile_grid[0] * tile_grid[1]
    if isinstance(permutation, (tuple, list)):
        order = list(permutation)
    elif permutation == 'identity':
        order = list(range(count))
    elif permutation == 'swap_halves' and list(tile_grid) == [1, 2]:
        order = [1, 0]
    elif permutation in ('cycle_forward', 'cycle_inverse') and count in (3, 4):
        order = ([2, 0, 1] if permutation == 'cycle_forward' else [1, 2, 0]) + list(range(3, count))
    else:
        raise ValueError(f'Unsupported tile permutation: {tile_grid}/{permutation}')
    if any(type(x) is not int for x in order) or sorted(order) != list(range(count)):
        raise ValueError('Tile gather order must be a bijection')
    return order


def inverse_order(order):
    if sorted(order) != list(range(len(order))):
        raise ValueError('Cannot invert non-bijection')
    inverse = [0] * len(order)
    for dest, source in enumerate(order):
        inverse[source] = dest
    return inverse


def row_tile_grid(row):
    return list(row.get('tile_grid', [1, 2]))


def raster_indices(height, width, tile_grid, permutation='identity'):
    order = tile_order(tile_grid, permutation)
    rows, cols = tile_grid
    if height % rows or width % cols:
        raise ValueError('Actual raster must divide exactly into declared tiles')
    th, tw = height // rows, width // cols
    result = []
    for r in range(height):
        for c in range(width):
            src = order[(r // th) * cols + c // tw]
            sr, sc = divmod(src, cols)
            result.append((sr * th + r % th) * width + sc * tw + c % tw)
    return result


def qwen_token_indices(grids, tile_grid, permutation='identity'):
    hh, ww = token_geometry(grids)
    return raster_indices(hh, ww, tile_grid, permutation)


def ibm_token_indices(geometry, tile_grid, permutation='identity'):
    """Match translated nominal boxes within each packed base/tiles raster."""
    order = tile_order(tile_grid, permutation)
    rows, cols = tile_grid
    h, w = geometry.image_hw
    th, tw = h / rows, w / cols
    ids = list(range(len(geometry.tokens)))
    key = lambda view, box: (view, *(round(float(x), 7) for x in box))
    lookup = {key(t.view, t.bbox_xyxy): t.index for t in geometry.tokens if t.kind == 'visual'}
    for token in geometry.tokens:
        if token.kind != 'visual':
            continue
        x0, y0, x1, y1 = token.bbox_xyxy
        rr, cc = min(rows - 1, int((y0+y1)/2/th)), min(cols - 1, int((x0+x1)/2/tw))
        dest = rr * cols + cc
        src = order[dest]
        if src == dest:
            continue
        if x0 < cc*tw-1e-7 or x1 > (cc+1)*tw+1e-7 or y0 < rr*th-1e-7 or y1 > (rr+1)*th+1e-7:
            raise ValueError('Actual post-pack slot straddles a moved tile boundary')
        sr, sc = divmod(src, cols)
        dx, dy = (sc-cc)*tw, (sr-rr)*th
        source = lookup.get(key(token.view, (x0+dx, y0+dy, x1+dx, y1+dy)))
        if source is None:
            raise ValueError('Actual packed geometry does not admit exact tile translation')
        ids[token.index] = source
    if sorted(ids) != list(range(len(ids))):
        raise ValueError('Actual packed tile map is not bijective')
    if any(ids[i] != i for i in geometry.special_indices):
        raise AssertionError('Special positions moved')
    return ids


def region_indices(order, region, source_roi=None):
    if region == 'all':
        return list(range(len(order)))
    if region not in ('roi', 'complement') or source_roi is None:
        raise ValueError('ROI/complement require source ROI indices')
    roi = set(source_roi)
    selected = [dest for dest, src in enumerate(order) if (src in roi) == (region == 'roi')]
    if not selected:
        raise ValueError('Requested region is empty')
    return selected


def qwen_region_indices(grids, tile_grid, region, bbox=None, canvas=None, permutation='identity'):
    order = qwen_token_indices(grids, tile_grid, permutation)
    source = qwen_roi(grids, bbox, canvas) if bbox is not None else None
    return region_indices(order, region, source)


def ibm_region_indices(geometry, tile_grid, region, bbox=None, permutation='identity'):
    order = ibm_token_indices(geometry, tile_grid, permutation)
    source = ibm_roi(geometry, bbox) if bbox is not None else None
    return region_indices(order, region, source)


def permutation_record(row, condition):
    grid = row_tile_grid(row)
    order = tile_order(grid, condition.get('token_permutation', 'identity'))
    return {'tile_grid': grid, 'tile_output_to_source': order,
            'tile_source_to_output': inverse_order(order),
            'coordinate_contract': 'destination gathers source; ROI mask follows source features'}


def preflight_geometry(row):
    sizes = {(meta['height'], meta['width']) for meta in row['source_image_metadata'].values()}
    if len(sizes) != 1:
        raise ValueError('All shared sources must use identical original canvas geometry')
    h, w = next(iter(sizes)); tile_grid = row_tile_grid(row)
    if h % 32 or w % 32:
        raise ValueError('v9 canvases must divide into patch16/merge2 without resizing')
    grids = [[1, h // 16, w // 16]]; ibm = image_geometry((h, w))
    names = ('identity', 'swap_halves') if tile_grid == [1, 2] else ('identity', 'cycle_forward', 'cycle_inverse')
    counts = {}
    for name in names:
        qp = qwen_token_indices(grids, tile_grid, name)
        bp = ibm_token_indices(ibm, tile_grid, name)
        box = row.get('donor_roi_bbox_xyxy')
        if box is not None:
            counts[name] = {'qwen': len(qwen_region_indices(grids, tile_grid, 'roi', box, [w, h], name)),
                            'ibm': len(ibm_region_indices(ibm, tile_grid, 'roi', box, name))}
    if any(value != next(iter(counts.values())) for value in counts.values()):
        raise AssertionError('Tile movement changed source ROI token count')
    return {'image_hw': [h, w], 'tile_grid': tile_grid, 'qwen_grid_thw': grids,
            'qwen_visual_tokens': len(qp), 'ibm_geometry': ibm.summary(),
            'permutations': list(names), 'nominal_roi_token_counts': counts,
            'status': 'CPU coordinate preflight only; actual processor checks pending CUDA'}
