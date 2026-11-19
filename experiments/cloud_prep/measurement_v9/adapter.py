"""v9 exact-tile adapters, retaining v8 native delivery/load implementations.

Only prepare/composition/geometry audits are extended. Model loading remains
Linux CUDA only; this module and its CPU sentinel tests import no tensors.
"""
from __future__ import annotations

from cloud_prep.measurement_v8 import qwen_adapter as q8, ibm_adapter as b8
from cloud_prep.measurement_v8.qwen_adapter import (
    QwenVisual, condition_sources, _grids, _canvas, token_geometry,
    validate_visual, ROI_RULE,
)
from cloud_prep.measurement_v8.ibm_adapter import IBMVisual, _roi_bbox
from cloud_prep.measurement_v8.ibm_geometry import ENTRY_ORDER, entry_layers, roi_token_indices
from cloud_prep.measurement_v9.geometry import (
    row_tile_grid, tile_order, inverse_order, permutation_record,
    qwen_token_indices, qwen_region_indices, ibm_token_indices, ibm_region_indices,
)


def _tensor_backend():
    import torch
    return torch


def inherited_audit_inputs(source_inputs, sources, row):
    """Alias all source-indexed inputs, including the canvas metadata lookup."""
    return ({'v9_'+k:v for k,v in source_inputs.items()},
            {'v9_'+k:v for k,v in sources.items()},
            {**row, 'source_image_metadata': {'v9_'+k:v for k,v in row['source_image_metadata'].items()}})


def compose_ibm_visual(sources: dict, condition: dict,
                   row: dict, *, hidden_size: int = 2560) -> IBMVisual:
    """Replace selected post-pack entries, retaining main's layer0 and other entries.

    all replaces every packed token, including newline/padding; complement is
    exactly all minus ROI. P fixes special *positions* but preserves donor
    source semantics there. ROI is selected in donor coordinates before P.
    """
    main = sources[condition["main"]]
    donor_name = condition.get("donor_source", condition.get("donor"))
    if donor_name is None:
        old = condition.get("ds_sources", ())
        if len(set(old)) != 1:
            raise ValueError("IBM needs one donor_source, not three Qwen stream sources")
        donor_name = old[0]
    donor = sources[donor_name]
    b8.validate_visual(main, hidden_size=hidden_size)
    b8.validate_visual(donor, hidden_size=hidden_size)
    if main.geometries != donor.geometries:
        raise ValueError("Main/donor post-pack coordinates differ")
    group = condition.get("entry_group", "all")
    selected_layers = entry_layers(group)
    region = condition.get("region", "all")
    permutation = condition.get("token_permutation", "identity")
    box = _roi_bbox(row, condition)
    masks, permutations, counts = [], [], []
    for geometry in main.geometries:
        ids = ibm_token_indices(geometry, row_tile_grid(row), permutation)
        mask = ibm_region_indices(geometry, row_tile_grid(row), region, box, permutation=permutation)
        masks.append(mask)
        permutations.append(ids)
        roi = roi_token_indices(geometry, box) if box is not None else None
        counts.append({"total": len(geometry.tokens), "selected": len(mask),
                       "roi": len(roi) if roi is not None else None,
                       "complement": len(geometry.tokens) - len(roi) if roi is not None else None,
                       "special": len(geometry.special_indices),
                       "selected_special": len(set(mask) & set(geometry.special_indices)),
                       "selected_base": sum(geometry.tokens[i].view == "base" for i in mask),
                       "selected_tiles": sum(geometry.tokens[i].view == "tiles" for i in mask)})
    donor_by_layer = dict(donor.entries)
    output = []
    for layer, main_features in main.entries:
        if layer not in selected_layers:
            output.append((layer, main_features))
            continue
        replaced = []
        for current, other, mask, ids in zip(main_features, donor_by_layer[layer], masks, permutations):
            if getattr(current, "dtype", None) != getattr(other, "dtype", None) or \
                    getattr(current, "device", None) != getattr(other, "device", None):
                raise ValueError("Main/donor tensor dtype or device differs")
            if region == "all":
                result = other if permutation == "identity" else other[ids]
            else:
                result = current.clone()
                result[mask] = other[[ids[i] for i in mask]]
            replaced.append(result)
        output.append((layer, tuple(replaced)))
    result = IBMVisual(tuple(output), main.geometries, {
        "main_source": condition["main"], "donor_source": donor_name,
        "entry_group": group, "replaced_layers": list(selected_layers),
        "preserved_layers": [layer for layer in ENTRY_ORDER if layer not in selected_layers],
        "layer0_source": condition["main"], "token_permutation": permutation,
        "region": region, "donor_roi_bbox_xyxy": list(box) if box is not None else None,
        "region_counts": counts, **permutation_record(row, condition),
        "roi_semantics": "nominal post-projection slot support, not encoder/QFormer receptive field",
    })
    if dict(result.entries)[0] is not dict(main.entries)[0]:
        raise AssertionError("The main layer0 entry was replaced")
    return result


class QwenAdapter(q8.QwenAdapter):
    def prepare(self, row, variant, source):
        from PIL import Image
        path = row['source_images'][source]
        if isinstance(path, (list,tuple)):
            if len(path) != 1:
                raise ValueError('One static image per source required')
            path = path[0]
        with Image.open(path) as raw:
            image = raw.convert('RGB')
        try:
            messages = [{'role':'user','content':[{'type':'text','text':variant['prompt']},{'type':'image'}]}]
            prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prompt += variant.get('assistant_prefix', '')
            inputs = self.processor(text=[prompt], images=[image], return_tensors='pt').to(self.device)
        finally:
            image.close()
        grids = _grids(inputs)
        width, height = _canvas(row, source)
        if width % 32 or height % 32 or grids != [[1, height // 16, width // 16]]:
            raise ValueError('v9 requires unresized patch16/merge2 data; actual processor geometry differs')
        qwen_token_indices(grids, row_tile_grid(row))
        hh, ww = token_geometry(grids)
        actual = int((inputs['input_ids'] == self.model.config.image_token_id).sum().item())
        if actual != hh*ww:
            raise ValueError('Image token placeholders differ from merged grid count')
        if 'mm_token_type_ids' not in inputs:
            raise ValueError('Transformers 5.8 requires processor mm_token_type_ids')
        if inputs['mm_token_type_ids'].shape != inputs['input_ids'].shape:
            raise ValueError('Multimodal type IDs do not align with input IDs')
        return inputs, prompt

    def compose(self, sources, condition, row, inputs=None):
        torch = _tensor_backend()
        main_name, donor_names = condition_sources(condition)
        base = sources[main_name]
        validate_visual(base)
        if inputs is not None and _grids(inputs) != base.grids:
            raise ValueError('Prepared input and source geometry differ')
        permutation = condition.get('token_permutation', 'identity')
        order = qwen_token_indices(base.grids, row_tile_grid(row), permutation)
        perm_ids = torch.tensor(order, device=base.main.device, dtype=torch.long)
        region = condition.get('region', 'all')
        masks, features = [], []
        for path, name in enumerate(donor_names):
            donor = sources[name]
            validate_visual(donor)
            if donor.grids != base.grids or donor.main.shape != base.main.shape:
                raise ValueError('Source geometry/hidden dimensions changed')
            if donor.main.device != base.main.device or donor.main.dtype != base.main.dtype:
                raise ValueError('Source dtype/device changed')
            moved = donor.deepstack[path].index_select(0, perm_ids)
            if region == 'all':
                selected = list(range(len(order)))
                feature = moved
            else:
                bbox = row.get('donor_roi_bbox_xyxy')
                if bbox is None:
                    raise ValueError('ROI conditions require donor_roi_bbox_xyxy')
                selected = qwen_region_indices(base.grids, row_tile_grid(row), region, bbox, _canvas(row,name), permutation)
                if not selected:
                    raise ValueError('Requested region contains no merged visual tokens')
                feature = base.deepstack[path].clone()
                ids = torch.tensor(selected, device=feature.device, dtype=torch.long)
                feature.index_copy_(0, ids, moved.index_select(0, ids))
            features.append(feature)
            masks.append(selected)
        result = QwenVisual(base.main.clone(), features, [list(g) for g in base.grids], base.output_type,
            {'main_source':main_name, 'ds_sources':list(donor_names), 'entry_group':'all', 'region':region,
             'token_permutation':permutation, 'selected_token_indices':masks, 'roi_rule':ROI_RULE,
             **permutation_record(row, condition), 'main_preserved':True, 'unselected_later_entries_keep_main':True})
        validate_visual(result)
        return result

    def audit_sources(self, source_inputs, sources, row):
        # Keep v8 rotary/raster/inverse checks, suppress only its half-swap pair
        # name heuristic: v9 M1 may be a three-cycle instead of a half swap.
        inherited = super().audit_sources(*inherited_audit_inputs(source_inputs, sources, row))
        return {**inherited, 'v9_tile_audit': audit_tiles(self, source_inputs, sources, row, 'qwen')}


class GraniteAdapter(b8.GraniteAdapter):
    def compose(self, sources, condition, row, *, inputs=None):
        result = compose_ibm_visual(sources, condition, row)
        if inputs is not None and self._geometry(inputs) != result.geometries:
            raise ValueError('Cached feature geometry differs from actual main input')
        return result

    def audit_sources(self, source_inputs, sources, row):
        inherited = super().audit_sources(*inherited_audit_inputs(source_inputs, sources, row))
        return {**inherited, 'v9_tile_audit': audit_tiles(self, source_inputs, sources, row, 'ibm')}


def permute_pixel_raster(pixels, tile_grid, permutation):
    """Move last-two-axis tiles; preserving uniform padding is handled by caller."""
    order = tile_order(tile_grid, permutation)
    rows, cols = tile_grid
    h, w = pixels.shape[-2:]
    if h % rows or w % cols:
        raise ValueError('Processed pixel raster cannot divide into exact tiles')
    th, tw = h // rows, w // cols
    result = pixels.clone()
    for dest, source in enumerate(order):
        dr, dc = divmod(dest, cols); sr, sc = divmod(source, cols)
        result[..., dr*th:(dr+1)*th, dc*tw:(dc+1)*tw] = pixels[..., sr*th:(sr+1)*th, sc*tw:(sc+1)*tw]
    return result


def ibm_pixel_permutation(pixels, geometry, tile_grid, permutation):
    result = pixels.clone(); size = geometry.view_size
    result[0, 0] = permute_pixel_raster(pixels[0, 0], tile_grid, permutation)
    th, tw = geometry.tile_grid_hw
    mosaic = pixels[0, 1:].reshape(th, tw, 3, size, size).permute(2, 0, 3, 1, 4).reshape(3, th*size, tw*size)
    rh, rw = geometry.resized_hw
    py, px = (th*size-rh)//2, (tw*size-rw)//2
    moved = mosaic.clone()
    moved[:, py:py+rh, px:px+rw] = permute_pixel_raster(mosaic[:, py:py+rh, px:px+rw], tile_grid, permutation)
    result[0, 1:] = moved.reshape(3, th, size, tw, size).permute(1, 3, 0, 2, 4).reshape(th*tw, 3, size, size)
    return result


def audit_tiles(adapter, source_inputs, sources, row, family):
    torch = _tensor_backend()
    grid = row_tile_grid(row)
    move = 'swap_halves' if grid == [1, 2] else 'cycle_forward'
    permutations = ('identity', move) if grid == [1, 2] else ('identity', 'cycle_forward', 'cycle_inverse')
    first = next(iter(sources.values()))
    geometry = first.grids if family == 'qwen' else first.geometries[0]
    indexer = qwen_token_indices if family == 'qwen' else ibm_token_indices
    orders = {p:indexer(geometry, grid, p) for p in permutations}
    inverse_checks = []
    for p, order in orders.items():
        inverse = inverse_order(order)
        for name, visual in sources.items():
            features = visual.deepstack if family == 'qwen' else [parts[0] for _,parts in visual.entries]
            for feature in features:
                ids = torch.tensor(order, device=feature.device, dtype=torch.long)
                inv = torch.tensor(inverse, device=feature.device, dtype=torch.long)
                if not torch.equal(feature.index_select(0, ids).index_select(0, inv), feature):
                    raise RuntimeError('Actual feature tile-permutation inverse failed')
            inverse_checks.append({'source':name, 'permutation':p, 'inverse_exact':True})
    pair_checks = []
    for name, inp in source_inputs.items():
        other = name.replace('m_s', 'm1_s', 1) if name.startswith('m_s') else None
        if other not in source_inputs:
            continue
        if family == 'qwen':
            order = orders[move]
            ids = torch.tensor([4*i+k for i in order for k in range(4)], device=inp['pixel_values'].device)
            expected = inp['pixel_values'].index_select(0, ids)
        else:
            expected = ibm_pixel_permutation(inp['pixel_values'], geometry, grid, move)
        if not torch.equal(expected, source_inputs[other]['pixel_values']):
            raise RuntimeError(f'Actual processor violates declared exact tile permutation: {name}/{other}')
        pair_checks.append({'source':name, 'moved_source':other, 'permutation':move, 'processed_pixels_equal':True})
    return {'tile_grid':grid, 'token_gather_orders':orders,
            'feature_inverse_checks':inverse_checks, 'processor_pair_checks':pair_checks,
            'feature_pixel_equivariance_required':False}
