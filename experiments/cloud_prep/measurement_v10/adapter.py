"""Matched nominal local interventions at main or later entries.

No model/tensor import at module load. Geometry and delivery inherit v9; the
role changes only the selected tensors, never the attribute direction.
"""
from __future__ import annotations

from cloud_prep.measurement_v9 import adapter as v9
from cloud_prep.measurement_v9.geometry import (
    row_tile_grid, qwen_token_indices, qwen_region_indices,
    ibm_token_indices, ibm_region_indices, permutation_record,
)
from cloud_prep.measurement_v8.qwen_adapter import QwenVisual, validate_visual, _grids, _canvas, ROI_RULE
from cloud_prep.measurement_v8.ibm_adapter import IBMVisual, _roi_bbox
from cloud_prep.measurement_v8.ibm_geometry import ENTRY_ORDER


def _tensor_backend():
    import torch
    return torch


def _local_sources(sources, condition):
    role = condition.get('changed_entry')
    if role not in ('main', 'later'):
        raise ValueError('Local operation requires changed_entry main/later')
    if condition.get('region') != 'roi' or condition.get('entry_group') != 'all':
        raise ValueError('Matched local role operation uses ROI and all entry group')
    main = condition['main']
    context = condition.get('context_source', main)
    donor = condition['donor_source']
    if any(name not in sources for name in (main, context, donor)):
        raise ValueError('Missing main/context/donor visual source')
    return role, main, context, donor


def _tensor_compatible(current, donor):
    if current.shape != donor.shape or current.dtype != donor.dtype or current.device != donor.device:
        raise ValueError('Matched entry tensors must share shape, dtype and device')


def _identity_audit(adapter, source_inputs, sources, row, family):
    """Natural whole-image panels have no tile permutation to audit.

    Retain actual processor/packing/rotary checks; suppress only v9's compulsory
    three-cycle and image-name swap heuristics. Pixel mirror identity belongs
    to the natural data validation, not to a feature-equivariance requirement.
    """
    parent = v9.q8.QwenAdapter if family == 'qwen' else v9.b8.GraniteAdapter
    result = parent.audit_sources(adapter,*v9.inherited_audit_inputs(source_inputs,sources,row))
    return {**result,'v10_tile_audit':{'tile_grid':[1,1],'permutations':['identity'],
        'feature_pixel_equivariance_required':False,
        'pixel_mirror_validation_scope':'natural dataset validator, separate from this adapter audit'}}


def compose_ibm_visual(sources, condition, row, *, hidden_size=2560):
    if condition.get('intervention_type') != 'local_role':
        return v9.compose_ibm_visual(sources, condition, row, hidden_size=hidden_size)
    role, main_name, context_name, donor_name = _local_sources(sources, condition)
    main, context, donor = (sources[n] for n in (main_name, context_name, donor_name))
    for visual in (main, context, donor):
        v9.b8.validate_visual(visual, hidden_size=hidden_size)
    if main.geometries != donor.geometries or main.geometries != context.geometries:
        raise ValueError('Main/context/donor post-pack coordinates differ')
    box = _roi_bbox(row, condition)
    if box is None:
        raise ValueError('Local role operation requires source ROI')
    permutation = condition.get('token_permutation', 'identity')
    maps, masks, counts = [], [], []
    for geometry in main.geometries:
        ids = ibm_token_indices(geometry, row_tile_grid(row), permutation)
        selected = ibm_region_indices(geometry, row_tile_grid(row), 'roi', box, permutation)
        special = set(geometry.special_indices)
        if special.intersection(selected):
            raise AssertionError('Local ROI cannot replace packed special tokens')
        maps.append(ids); masks.append(selected)
        counts.append({'total':len(ids), 'selected':len(selected), 'special':len(special),
                       'selected_special':0,
                       'selected_base':sum(geometry.tokens[i].view == 'base' for i in selected),
                       'selected_tiles':sum(geometry.tokens[i].view == 'tiles' for i in selected)})
    mains, contexts, donors = map(lambda v:dict(v.entries), (main,context,donor))
    for layer in ENTRY_ORDER:
        for current, fixed, other in zip(mains[layer],contexts[layer],donors[layer]):
            _tensor_compatible(current,fixed)
            _tensor_compatible(current,other)
    layers = [0] if role == 'main' else [layer for layer in ENTRY_ORDER if layer != 0]
    output = []
    for layer in ENTRY_ORDER:
        fixed = mains[layer] if layer == 0 else contexts[layer]
        if layer not in layers:
            output.append((layer, fixed))
            continue
        replaced = []
        for current, other, ids, selected in zip(fixed, donors[layer], maps, masks):
            _tensor_compatible(current, other)
            result = current.clone()
            result[selected] = other[[ids[i] for i in selected]]
            replaced.append(result)
        output.append((layer, tuple(replaced)))
    result = IBMVisual(tuple(output), main.geometries, {
        'main_source':main_name, 'context_source':context_name, 'donor_source':donor_name,
        'changed_entry':role, 'entry_group':'all', 'replaced_layers':layers,
        'preserved_layers':[layer for layer in ENTRY_ORDER if layer not in layers],
        'layer0_source':main_name, 'layer0_locally_patched':role == 'main',
        'region':'roi', 'token_permutation':permutation, 'donor_roi_bbox_xyxy':list(box),
        'selected_token_indices':masks, 'region_counts':counts,
        'special_tokens_preserved':True, 'source_cache_immutable':True,
        **permutation_record(row,condition),
        'roi_semantics':'Matched nominal packed slot support; contextualized features and entry depth/scale differ',
    })
    v9.b8.validate_visual(result, hidden_size=hidden_size)
    return result


class QwenAdapter(v9.QwenAdapter):
    def audit_sources(self,source_inputs,sources,row):
        if row_tile_grid(row) == [1,1]:
            return _identity_audit(self,source_inputs,sources,row,'qwen')
        return super().audit_sources(source_inputs,sources,row)

    def compose(self, sources, condition, row, inputs=None):
        if condition.get('intervention_type') != 'local_role':
            return super().compose(sources,condition,row,inputs=inputs)
        role, main_name, context_name, donor_name = _local_sources(sources,condition)
        main, context, donor = (sources[n] for n in (main_name,context_name,donor_name))
        for visual in (main,context,donor):
            validate_visual(visual)
            if visual.grids != main.grids:
                raise ValueError('Main/context/donor visual geometry differs')
            _tensor_compatible(main.main,visual.main)
        if inputs is not None and _grids(inputs) != main.grids:
            raise ValueError('Prepared input and source geometry differ')
        if _canvas(row, main_name) != _canvas(row, donor_name) or _canvas(row, main_name) != _canvas(row, context_name):
            raise ValueError('Main/context/donor canvas differs')
        permutation = condition.get('token_permutation','identity')
        order = qwen_token_indices(main.grids,row_tile_grid(row),permutation)
        selected = qwen_region_indices(main.grids,row_tile_grid(row),'roi',
            row.get('donor_roi_bbox_xyxy'),_canvas(row,donor_name),permutation)
        torch = _tensor_backend()
        dest = torch.tensor(selected,device=main.main.device,dtype=torch.long)
        src = torch.tensor([order[i] for i in selected],device=main.main.device,dtype=torch.long)
        def replace(current, other):
            _tensor_compatible(current,other)
            result = current.clone()
            result.index_copy_(0,dest,other.index_select(0,src))
            return result
        new_main = replace(main.main,donor.main) if role == 'main' else main.main.clone()
        later = [replace(current,other) if role == 'later' else current.clone()
                 for current,other in zip(context.deepstack,donor.deepstack)]
        result = QwenVisual(new_main,later,[list(g) for g in main.grids],main.output_type,{
            'main_source':main_name,'context_source':context_name,'donor_source':donor_name,
            'changed_entry':role,'entry_group':'all','region':'roi','token_permutation':permutation,
            'selected_token_indices':[selected] * (1 if role == 'main' else 3),
            'selected_main_token_indices':selected if role == 'main' else [],
            'selected_later_token_indices':[selected]*3 if role == 'later' else [[],[],[]],
            'main_preserved':role != 'main','later_preserved':role != 'later',
            'special_tokens_preserved':True,'source_cache_immutable':True,'roi_rule':ROI_RULE,
            **permutation_record(row,condition)})
        validate_visual(result)
        return result


class GraniteAdapter(v9.GraniteAdapter):
    def audit_sources(self,source_inputs,sources,row):
        if row_tile_grid(row) == [1,1]:
            return _identity_audit(self,source_inputs,sources,row,'ibm')
        return super().audit_sources(source_inputs,sources,row)

    def compose(self,sources,condition,row,*,inputs=None):
        result = compose_ibm_visual(sources,condition,row)
        if inputs is not None and self._geometry(inputs) != result.geometries:
            raise ValueError('Cached feature geometry differs from actual main input')
        return result
