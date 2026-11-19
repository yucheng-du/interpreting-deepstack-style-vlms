"""Batched Granite Vision 4.1 visual delivery (family 'ibm'): n per-row visuals.

Common batched interface (batched_qwen / batched_ibm / batched_deepstack)::

    stack_inputs(adapter, inputs_list) -> dict
    with use_batch(adapter, visuals, inputs_n, *, audit=False) as stats: ...

Module import needs no torch. The batch-one path (GraniteAdapter.use, ibm_adapter
and ibm_geometry) is reused read-only and is not modified by this module.

Row-order guarantee (IBM revision 37d591f, modeling.py sha256 f1302358...)
--------------------------------------------------------------------------
Granite4VisionModel.forward consumes get_image_features(pixel_values,
image_sizes) == [(decoder_layer, [packed_0, ..., packed_{n-1}])] for the eight
entries in ENTRY_ORDER (9, 6, 3, 0, 12, 15, 18, 21) as follows::

    concat = torch.cat(packed, dim=0).to(embeds)               # [sum_i T_i, 2560]
    # first entry only: [n, L, 2560] boolean mask, placeholders zeroed
    vision_mask = (input_ids == image_token_id)[..., None].expand_as(inputs_embeds)
    inputs_embeds = inputs_embeds.masked_fill(vision_mask, 0.0)
    # before decoder layer l, for every entry whose decoder_layer == l
    hidden = hidden.masked_scatter(vision_mask, (hidden[vision_mask] + concat.flatten()).view(-1))

Boolean-mask indexing and masked_scatter enumerate the True positions of the
[n, L, H] mask in row-major order: all placeholders of batch row 0 in sequence
order, then row 1, and so on; torch.cat puts packed_0 first. Packed image i
therefore lands on batch row i's placeholders, token by token in packed order,
exactly when every row i holds T_i placeholders. That is what this module
enforces:

1. stack_inputs concatenates the batch-one processor rows along dim 0 in list
   order. In the native (unpatched) forward, get_image_features slices
   pixel_values[i][:views_i] per row, runs the tower once, splits by views and
   packs/unpads image by image in that same order (pack_and_unpad loops over
   images), so row i receives image i. The result equals the processor called
   with n prompts and n images (checked on the real processor: no padding
   arises because shapes are identical).
2. use_batch returns, for each of the eight entries, [visuals[0][entry], ...,
   visuals[n-1][entry]] in batch-row order; layer0 (main) and the seven later
   entries of row i all come from visuals[i]. Injection is an elementwise add at
   the placeholder positions before each target decoder layer, so it is per
   row; no other row's state is touched.
3. Geometry is validated per row at entry, like the batch-one use(): packed
   geometry from image_sizes[i] equals visuals[i].geometries, the placeholder
   count of row i equals its packed token count, and the view shape is
   (1, 1 + tiles, 3, 384, 384). In addition, a forward pre-hook on the Granite
   base model re-checks every delivering forward (pixel_values present): the
   batch size is n and every row of the actual input_ids (prompt, or prompt
   plus teacher-forced label prefix) has exactly T_i placeholders. The model's
   own check is only on the total count, so a cross-row shift with a
   coincidental total can never pass silently.

Generation: Granite4VisionForConditionalGeneration.prepare_inputs_for_generation
passes pixel_values/image_sizes only on the first iteration (the prefill);
decode steps have pixel_values None, never call get_image_features and reuse the
per-row KV cache. A batched greedy generate therefore delivers exactly once
(stats['calls'] == 1); later steps are not deliveries and are not counted.

State: like GraniteAdapter.use(), only the instance attribute
base.get_image_features (plus the owner marker) is installed, and the forward
pre-hook is registered; all three are removed on every exit, restoring the exact
previous method/hook state. Nested or foreign replacement is refused. Cached
visual tensors are never mutated (the model concatenates them into a new
tensor before adding).
"""
from __future__ import annotations

from contextlib import contextmanager

from cloud_prep.measurement_v8.ibm_adapter import _MISSING, _OWNER, _plain, _shape, validate_visual
from cloud_prep.measurement_v8.ibm_geometry import ENTRY_ORDER

SCHEMA = 'v14-granite-batched-delivery-1'
INPUT_KEYS = ('input_ids', 'attention_mask', 'pixel_values', 'image_sizes')
ROW_ORDER = ('batch row i <- visuals[i] for all eight entries: torch.cat of per-row packed features, '
             'row-major masked_scatter at image placeholders; per-row placeholder count checked on every delivering forward')


def _token_id(adapter):
    config = adapter.model.config
    token = getattr(config, 'image_token_id', None)
    return config.image_token_index if token is None else token


def _row(inputs, index):
    """Batch-one view of one stacked row, for the batch-one geometry check."""
    return {k: inputs[k][index:index + 1] for k in ('input_ids', 'pixel_values', 'image_sizes')}


def stack_inputs(adapter, inputs_list):
    """Stack n prepared batch-one Granite inputs into one unpadded batch-n dict.

    Equivalent to the native processor with n prompts and n images when all rows
    have identical input_ids and view counts (no padding is ever added). Pixel
    values and image sizes may differ per row (native audit: a, b, degraded_a);
    each row must pass the batch-one geometry check. Row order is list order.
    """
    import torch
    rows = list(inputs_list)
    if not rows:
        raise ValueError('stack_inputs needs at least one prepared input')
    first = rows[0]
    keys = set(first.keys())
    if keys - set(INPUT_KEYS) or {'input_ids', 'pixel_values', 'image_sizes'} - keys:
        raise ValueError(f'Unexpected Granite processor keys: {sorted(keys)}')
    for inp in rows:
        if set(inp.keys()) != keys:
            raise ValueError('Prepared Granite inputs differ in keys')
        adapter._geometry(inp)  # batch=1, one image, placeholder/view counts
        if not torch.equal(inp['input_ids'], first['input_ids']):
            raise ValueError('Batched Granite rows need identical input_ids (no padding)')
        mask = inp.get('attention_mask')
        if mask is not None and (mask.shape != inp['input_ids'].shape or not bool(torch.all(mask == 1).item())):
            raise ValueError('Batched Granite rows must be unpadded (all-ones attention mask)')
        for key in keys:
            a, b = inp[key], first[key]
            if a.shape != b.shape or a.dtype != b.dtype or a.device != b.device:
                raise ValueError(f'Batched Granite {key} shape/dtype/device differs; padding is not supported')
    return {key: torch.cat([inp[key] for inp in rows], dim=0) for key in INPUT_KEYS if key in keys}


@contextmanager
def use_batch(adapter, visuals, inputs_n, *, audit=False):
    """Deliver visuals[i] to batch row i on every forward over inputs_n.

    Yields stats: 'calls' (one per delivering batched forward: each score
    forward, or the single generation prefill), 'batch_size', 'forwards'
    (one small record per delivery) and 'rows', where rows[i] is the complete
    batch-one-compatible delivery record of row i (the GraniteAdapter.use()
    state keys, in that order: calls, entry_layers, audit_requested,
    geometries, the visual's intervention metadata; then batch_size and
    batch_row). rows[i]['calls'] counts deliveries to row i and always equals
    stats['calls'].
    """
    base = adapter.base
    visuals = list(visuals)
    n = len(visuals)
    if n < 1:
        raise ValueError('use_batch needs at least one visual')
    if 'inputs_embeds' in inputs_n or _shape(inputs_n['input_ids'])[0] != n:
        raise ValueError('Batched Granite inputs need input_ids with one row per visual')
    if _shape(inputs_n['pixel_values'])[0] != n or _shape(inputs_n['image_sizes'])[0] != n:
        raise ValueError('Batched Granite pixel_values/image_sizes need one row per visual')
    counts = []
    for i, visual in enumerate(visuals):
        validate_visual(visual)
        if adapter._geometry(_row(inputs_n, i)) != visual.geometries:
            raise ValueError(f'Cached output and actual input geometry differ at batch row {i}')
        counts.append(len(visual.geometries[0].tokens))
    by_row = [dict(visual.entries) for visual in visuals]
    features = {}
    for layer in ENTRY_ORDER:
        parts = [entries[layer][0] for entries in by_row]
        if any(p.dtype != parts[0].dtype or p.device != parts[0].device for p in parts):
            raise ValueError(f'Batched Granite layer {layer} features differ in dtype/device across rows')
        features[layer] = tuple(parts)
    if getattr(base, _OWNER, None) is not None or 'get_image_features' in base.__dict__:
        raise RuntimeError('Nested/foreign Granite feature replacement is not supported')
    token_id = _token_id(adapter)
    expected_sizes, expected_pixels = _plain(inputs_n['image_sizes']), _shape(inputs_n['pixel_values'])
    rows = [{'calls': 0, 'entry_layers': list(ENTRY_ORDER), 'audit_requested': bool(audit),
             'geometries': [g.summary() for g in visual.geometries], **visual.intervention,
             'batch_size': n, 'batch_row': i} for i, visual in enumerate(visuals)]
    state = {'calls': 0, 'batch_size': n, 'schema': SCHEMA, 'entry_layers': list(ENTRY_ORDER),
             'audit_requested': bool(audit), 'row_order': ROW_ORDER, 'forwards': [], 'rows': rows}
    pending = []

    def check_rows(module, args, kwargs):
        pixel_values = kwargs.get('pixel_values', args[1] if len(args) > 1 else None)
        if pixel_values is None or pixel_values.size(0) == 0:
            return None  # text-only / decode step: no visual delivery in this forward
        input_ids = args[0] if args else kwargs.get('input_ids')
        if input_ids is None or input_ids.ndim != 2 or input_ids.shape[0] != n:
            raise ValueError('Batched Granite delivery needs input_ids with one row per visual')
        pending.clear()  # a forward that failed after this hook leaves no stale check behind
        actual = (input_ids == token_id).sum(dim=1).tolist()
        if actual != counts:
            raise ValueError(f'Per-row Granite placeholders {actual} differ from packed tokens {counts}')
        pending.append(int(input_ids.shape[1]))
        return None

    def deliver(pixel_values, image_sizes, vision_feature_layer=None,
                vision_feature_select_strategy=None, **kwargs):
        if _plain(image_sizes) != expected_sizes or _shape(pixel_values) != expected_pixels:
            raise ValueError('Fresh batched Granite visual call changed image sizes/view shape')
        if vision_feature_select_strategy not in (None, 'full'):
            raise ValueError('Fresh batched Granite visual call changed feature selection')
        if len(pending) != 1:
            raise RuntimeError('Batched Granite delivery bypassed the per-row placeholder check')
        state['calls'] += 1
        for record in rows:
            record['calls'] += 1
        state['forwards'].append({'call': state['calls'], 'sequence_length': pending.pop(),
                                  'row_placeholders': list(counts), 'row_placeholder_counts_exact': True})
        return [(layer, list(features[layer])) for layer in ENTRY_ORDER]

    original = base.__dict__.get('get_image_features', _MISSING)
    setattr(base, _OWNER, state)
    base.get_image_features = deliver
    handle = None
    try:
        handle = base.register_forward_pre_hook(check_rows, with_kwargs=True)
        yield state
    finally:
        if handle is not None:
            handle.remove()
        if original is _MISSING:
            delattr(base, 'get_image_features')
        else:
            base.get_image_features = original
        delattr(base, _OWNER)
