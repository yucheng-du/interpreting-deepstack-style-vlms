"""Per-task native geometry audits used by the v14 runner (no torch at import).

``native_geometry_audit`` is what the runner calls for Granite/DeepStack in every
config. It passes straight through to ``adapter.audit_sources`` (same returned
object, hence a byte-identical audit record) except in one case:

Granite, untiled v14 row (tile_grid [1,1], the v10 identity audit), whose
processed geometry cannot realize the inherited v8 exact half swap. The v8
audit unconditionally builds ``spatial_token_indices(geometry, 'swap_halves')``
and raises ValueError for asymmetric natural sizes (GQA 500x333, document
pages). v14 has no token-permutation conditions, so only that check (and the
checks that consume its permutation) is marked not applicable; the actual
pack/unpad sentinel, QFormer window raster inverse, spatial-offset and
area-query raster checks still run on the real model. The decision is made
by evaluating exactly that permutation first; any other failure keeps raising.

``qwen_geometry_record`` is the cheap per-task Qwen record written in batched
mode only (grid, image-token count, ROI token ids used by the composer).
"""
from __future__ import annotations

HALF_SWAP_NOT_APPLICABLE = 'not_applicable_v14_identity_only'
HALF_SWAP_DEPENDENT_CHECKS = ('permutation_involution_exact', 'special_positions_fixed',
                              'feature_inverse_checks', 'processor_half_swap_checks')


def native_geometry_audit(adapter, family, source_inputs, sources, row):
    if family == 'ibm':
        return granite_audit_sources(adapter, source_inputs, sources, row)
    return adapter.audit_sources(source_inputs, sources, row)


def granite_half_swap_unrealizable(sources, row):
    """Reason string when the v8 half-swap check cannot apply to this v14 row, else None."""
    from cloud_prep.measurement_v8.ibm_geometry import spatial_token_indices
    from cloud_prep.measurement_v9.geometry import row_tile_grid
    if row_tile_grid(row) != [1, 1]:
        return None  # tiled rows keep the inherited v9 audit unchanged
    geometry = next(iter(sources.values())).geometries[0]
    try:
        spatial_token_indices(geometry, 'swap_halves')
    except ValueError as exc:
        return f'{type(exc).__name__}: {exc}'
    return None


def granite_audit_sources(adapter, source_inputs, sources, row):
    reason = granite_half_swap_unrealizable(sources, row)
    if reason is None:
        return adapter.audit_sources(source_inputs, sources, row)
    return granite_identity_only_audit(adapter, source_inputs, sources, row, reason)


def granite_identity_only_audit(adapter, source_inputs, sources, row, reason):
    """The v8 GraniteAdapter.audit_sources checks without the half-swap permutation.

    Mirrors cloud_prep/measurement_v8/ibm_adapter.py audit_sources line by line for the
    retained checks, and adds the v10 identity-audit block (cloud_prep/measurement_v10/
    adapter.py _identity_audit) that the passing path also returns.
    """
    import torch
    from cloud_prep.measurement_v8.ibm_adapter import _plain, _roi_bbox, validate_visual
    from cloud_prep.measurement_v8.ibm_geometry import roi_token_indices
    first = next(iter(sources.values()))
    geometry = first.geometries[0]
    for name, source in sources.items():
        validate_visual(source)
        if source.geometries != first.geometries or adapter._geometry(source_inputs[name]) != first.geometries:
            raise ValueError("Source geometry mismatch during Granite audit")
    base = adapter.base
    first_name = next(iter(sources))
    side = geometry.projected_side
    device = source_inputs[first_name]["pixel_values"].device
    sentinel = torch.arange(geometry.num_views * side * side, device=device, dtype=torch.float32)
    sentinel = sentinel.reshape(geometry.num_views, side * side, 1)
    packed, lens = base.pack_and_unpad_image_features([sentinel], source_inputs[first_name]["image_sizes"], "full",
                                                      image_newline=torch.tensor([-1.], device=device))
    if packed[0].flatten().tolist() != geometry.sentinel_order() or _plain(lens) != [len(geometry.tokens)]:
        raise RuntimeError("Actual Granite packing/unpadding order differs from coordinate mapping")
    p = base.layerwise_projectors[0]
    raster = torch.arange(24 * 24, device=device).reshape(1, 24 * 24, 1)
    if not torch.equal(p._unwin(p._win(raster, 24, 8), 3, 8), raster):
        raise RuntimeError("Actual QFormer window/unwindow raster mapping differs")
    spatial_checks = []
    for projector, (oh, ow), layer in zip(base.spatial_projectors, ((0, 0), (0, 1), (1, 0), (1, 1)), (12, 15, 18, 21)):
        expected = [(2 * r + oh) * 24 + 2 * c + ow for r in range(12) for c in range(12)]
        if projector.downsampler(raster).flatten().tolist() != expected:
            raise RuntimeError("Actual SpatialDeepstack sampled offsets differ from slot coordinates")
        spatial_checks.append({"layer": layer, "offset": [oh, ow], "sampled_raster_exact": True})
    area = p.downsampler(raster.float()).flatten().tolist()
    if area != [2 * r * 24 + 2 * c + 12.5 for r in range(12) for c in range(12)]:
        raise RuntimeError("Actual LayerDeepstack area-query downsampling differs")
    box = _roi_bbox(row)
    roi = roi_token_indices(geometry, box) if box is not None else None
    result = {"architecture": adapter.metadata, "geometry": geometry.summary(),
              "actual_pack_sentinel_exact": True, "qformer_window_raster_inverse_exact": True,
              "layer_area_query_raster_exact": True, "spatial_offset_checks": spatial_checks,
              "half_swap_involution_check": {
                  "half_swap_involution": HALF_SWAP_NOT_APPLICABLE, "reason": reason, "permutation": "swap_halves",
                  "skipped_checks": list(HALF_SWAP_DEPENDENT_CHECKS),
                  "scope": "v14 conditions are identity-only (no token permutation); the inherited v8 half-swap "
                           "involution needs an exact pixel-coordinate half swap of this processed geometry"},
              "roi": {"count": len(roi), "complement_count": len(geometry.tokens) - len(roi),
                      "base_count": sum(geometry.tokens[i].view == "base" for i in roi),
                      "tile_count": sum(geometry.tokens[i].view == "tiles" for i in roi),
                      "bbox_xyxy": list(box)} if roi is not None else None,
              "feature_pixel_equivariance_required": False,
              "native_self_patch_and_generation": "verified_separately_by_common_runner"}
    return {**result, 'v10_tile_audit': {'tile_grid': [1, 1], 'permutations': ['identity'],
        'feature_pixel_equivariance_required': False,
        'pixel_mirror_validation_scope': 'natural dataset validator, separate from this adapter audit'}}


def qwen_geometry_record(adapter, source_inputs, sources, row):
    """Cheap per-task Qwen geometry: actual grid, placeholder count and composer ROI ids."""
    from cloud_prep.measurement_v8.qwen_adapter import ROI_RULE, _grids, roi_token_indices, token_geometry
    grids = next(iter(sources.values())).grids
    hh, ww = token_geometry(grids)
    image_token = int(adapter.model.config.image_token_id)
    per_source = {}
    for name, inp in source_inputs.items():
        count = int((inp['input_ids'] == image_token).sum().item())
        per_source[name] = {'image_grid_thw': _grids(inp), 'image_tokens': count}
        if _grids(inp) != grids or sources[name].grids != grids or count != hh * ww:
            raise ValueError('Qwen processed grid/placeholder count differs across sources: ' + name)
    record = {'grid_thw': grids, 'merged_grid_hw': [hh, ww], 'visual_tokens': hh * ww,
              'image_token_count': hh * ww, 'per_source': per_source,
              'prompt_tokens': int(next(iter(source_inputs.values()))['input_ids'].shape[1])}
    box = row.get('donor_roi_bbox_xyxy')
    if box is not None:
        # Same canvas lookup as adapter.support (the composer's ROI mask).
        meta = next(iter(row['source_image_metadata'].values()))
        roi = roi_token_indices(grids, box, (meta['width'], meta['height']))
        record.update(roi_bbox_xyxy=list(box), canvas_wh=[meta['width'], meta['height']],
                      roi_token_indices=roi, roi_tokens=len(roi), roi_rule=ROI_RULE)
    return record
