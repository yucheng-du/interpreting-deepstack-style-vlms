"""Batched Qwen3-VL visual delivery for v14 batched execution (Qwen3-VL-8B and -32B dense).

One batched forward evaluates the physically distinct conditions of one query/readout. All
rows share identical input_ids, attention mask, multimodal type ids and processed image grid;
only the delivered visual tensors differ. No padding is introduced, so each row is the
batch-one computation up to BF16-level kernel differences (GEMM shapes change with n).
No torch/transformers import happens at module import. The batch-one path (QwenAdapter.use)
is not touched; this module only adds a parallel context with the same discipline.

Row-order guarantee (transformers 5.8.0 models/qwen3_vl/modeling_qwen3_vl.py and
processing_qwen3_vl.py):

1. Inputs. ``stack_inputs`` concatenates batch-one processor outputs along dim 0 in list
   order: input_ids, attention_mask and mm_token_type_ids become [n,S] (identical rows);
   pixel_values becomes [sum_i P_i, F] with image i's patches in rows
   [sum_{j<i} P_j, sum_{j<=i} P_j); image_grid_thw becomes [n,3] with row i = grid of image i.
   This is the layout the processor itself returns for text=[prompt]*n and
   images=[image_0, ..., image_{n-1}]: it assigns images to prompts in order (one image per
   prompt here) and concatenates per-image patches in image order.
2. Main entry. ``use_batch`` patches visual.forward to return
   pooler_output = cat_i visuals[i].main ([sum_i N_i, D]) and
   deepstack_features[j] = cat_i visuals[i].deepstack[j], in the same row order.
   Qwen3VLModel.get_image_features splits pooler_output by image_grid_thw.prod(-1) // merge**2
   (one chunk per grid row, in order); Qwen3VLModel.forward concatenates the chunks back and
   calls inputs_embeds.masked_scatter(image_mask, image_embeds). masked_scatter consumes the
   source in row-major order of the [n,S,D] mask: all image placeholders of batch row 0 in
   sequence order, then row 1, and so on. ``use_batch`` checks that row i holds exactly N_i
   placeholders (= visuals[i].main.shape[0]) as one contiguous block whose mm_token_type_ids are
   exactly the image indicator, so row i receives visuals[i].main and nothing else.
3. Later entries. visual_pos_masks = image_mask[..., 0] ([n,S]). After decoder layer j
   (j = 0, 1, 2), Qwen3VLTextModel._deepstack_process adds deepstack_visual_embeds[j] through
   hidden_states[visual_pos_masks, :]; boolean-mask indexing enumerates the same row-major
   positions, so row i receives visuals[i].deepstack[j] at its own placeholders only. The
   unmodified DeepStackIntervention (mask '111': identity-checked features, injection order
   0,1,2, optional visual-only addition audit) wraps the text forward exactly as in batch one;
   its checks are batch-agnostic ([n,S] boolean mask, [sum_i N_i, D] features).
4. Positions and generation. get_rope_index loops over batch rows and consumes one
   image_grid_thw row per image group of each batch row, so row i uses grid i; identical rows
   give the batch-one M-RoPE positions in every row. In generate(),
   _prepare_position_ids_for_generation builds [4,n,S] positions and rope_deltas [n,1]; decode
   steps extend positions, attention mask and mm_token_type_ids per row, and
   prepare_inputs_for_generation sets pixel_values=None after the first iteration. Visuals are
   therefore delivered once per generate (stats['calls'] == 1, the prefill), never in decode
   steps. Rows that stop early are padded by generate with pad_token_id after their EOS; the
   caller truncates each row after its first EOS to recover the batch-one token sequence.

Vision tower and audits. The Qwen vision tower is not batch-invariant in BF16: encoding n images
in one call changes GEMM shapes, and on 384x384 natural images (grid 24x24) its outputs differed
from batch-one encodes by up to 0.375 (bit-identical only by chance on the 384x256 synthetic
grid). Experimental sources must therefore stay batch-one ``adapter.encode`` outputs, as in
the unbatched runner. For the same reason the batched native/self-patch audit must compare the
batched native forward with ``use_batch`` fed by ``encode_batch`` (the same batched vision call,
split per row). That comparison is exact (0.0), whereas self-patching batch-one encodes
measures the vision tower's batch variance, not delivery (up to 2.8 nats on a tail log-prob).
"""
from __future__ import annotations

from contextlib import contextmanager

SCHEMA = 'qwen-batched-visual-delivery-v14'
_MISSING = object()
# Exactly what the 5.8 Qwen3-VL processor returns for one static image prompt.
_IDENTICAL_KEYS = ('input_ids', 'attention_mask', 'mm_token_type_ids')
_PIXEL_KEYS = ('pixel_values', 'image_grid_thw')
_KEYS = frozenset(_IDENTICAL_KEYS + _PIXEL_KEYS)


def _row_grids(grid_thw):
    if hasattr(grid_thw, 'detach'):
        grid_thw = grid_thw.detach().cpu().tolist()
    return [[int(x) for x in grid] for grid in grid_thw]


def stack_inputs(adapter, inputs_list):
    """Stack n prepared batch-one Qwen processor inputs into one unpadded batch-n dict.

    Each element is ``adapter.prepare(row, variant, source)[0]``. Rows keep list order:
    row i of the result is inputs_list[i] (see the module docstring, point 1). input_ids,
    attention_mask and mm_token_type_ids must be identical in every row, the attention mask
    all ones, and image_grid_thw identical (one static image per prompt); pixel_values may
    differ per row (e.g. sources a, b and degraded_a for the batched native audit).
    Returns a plain dict of new tensors; the inputs are not modified.
    """
    import torch
    from cloud_prep.measurement_v8.qwen_adapter import token_geometry
    items = list(inputs_list)
    if not items:
        raise ValueError('stack_inputs needs at least one prepared input')
    for i, inp in enumerate(items):
        if set(inp.keys()) != _KEYS:
            raise ValueError(f'Row {i}: unexpected Qwen processor keys {sorted(inp.keys())}; expected {sorted(_KEYS)}')
    order = list(items[0].keys())
    first = items[0]
    for i, inp in enumerate(items):
        ids = inp['input_ids']
        if ids.ndim != 2 or ids.shape[0] != 1 or ids.shape[1] < 1:
            raise ValueError(f'Row {i}: expected batch-one input_ids [1, prompt_tokens]')
        if ids.shape != first['input_ids'].shape:
            raise ValueError('Batched rows need identical input_ids (prompt lengths differ)')
        for key in ('attention_mask', 'mm_token_type_ids'):
            if inp[key].shape != ids.shape:
                raise ValueError(f'Row {i}: {key} does not align with input_ids')
        if tuple(inp['image_grid_thw'].shape) != (1, 3):
            raise ValueError(f'Row {i}: one static image per prompt required')
        pixels = inp['pixel_values']
        if pixels.ndim != 2 or pixels.shape[1] != first['pixel_values'].shape[1]:
            raise ValueError(f'Row {i}: pixel_values layout differs from row 0')
        if any(inp[key].dtype != first[key].dtype or inp[key].device != first['input_ids'].device for key in order):
            raise ValueError(f'Row {i}: tensor dtype/device differs from row 0')
    batched = {key: torch.cat([inp[key] for inp in items], dim=0) for key in order}
    grids = _row_grids(batched['image_grid_thw'])
    if any(grid != grids[0] for grid in grids):
        raise ValueError('Batched rows need identical processed image geometry (image_grid_thw)')
    token_geometry([grids[0]], merge_size=int(adapter.merge_size))
    t, h, w = grids[0]
    if any(inp['pixel_values'].shape[0] != t * h * w for inp in items):
        raise ValueError('pixel_values patch count differs from image_grid_thw')
    ids, mask, kinds = batched['input_ids'], batched['attention_mask'], batched['mm_token_type_ids']
    same_ids, same_mask, same_kinds, unpadded = torch.stack([
        (ids == ids[:1]).all(), (mask == mask[:1]).all(), (kinds == kinds[:1]).all(), (mask == 1).all(),
    ]).tolist()
    if not same_ids:
        raise ValueError('Batched rows need identical input_ids')
    if not (same_mask and same_kinds):
        raise ValueError('Batched rows need identical attention_mask and mm_token_type_ids')
    if not unpadded:
        raise ValueError('Batched rows must be unpadded (attention_mask all ones)')
    return batched


@contextmanager
def use_batch(adapter, visuals, inputs_n, *, audit=False):
    """Deliver visuals[i] to batch row i of every model forward on ``inputs_n``.

    ``visuals`` are QwenVisual objects (FixedLaterComposer.compose or adapter.encode), one per
    row of ``inputs_n`` (from ``stack_inputs``). Inside the context every visual call (score
    forwards and the generation prefill) returns the row-ordered concatenation of main and of
    each of the three DeepStack streams (module docstring, points 2-3), validated per row like
    the batch-one ``QwenAdapter.use``. Yields stats: calls (one per batched visual delivery),
    records (DeepStackIntervention, one per batched text forward with visual features),
    schema, batch_size and row_visual_tokens. On exit visual.forward, the text-model
    intervention and Qwen3VLModel.rope_deltas (HF's per-forward M-RoPE cache, recomputed at
    every fresh prefill) are restored exactly, so a [n,1] batched value never leaks out.
    """
    import torch
    from deepstack.intervention import DeepStackIntervention
    from cloud_prep.measurement_v8.qwen_adapter import QwenVisual, materialize_visual_output, validate_visual
    visuals = list(visuals)
    n = len(visuals)
    if n < 1:
        raise ValueError('use_batch needs at least one visual')
    ids = inputs_n['input_ids']
    if ids.ndim != 2 or ids.shape[0] != n:
        raise ValueError(f'inputs_n batch size {tuple(ids.shape)} differs from {n} visuals')
    grids = _row_grids(inputs_n['image_grid_thw'])
    if len(grids) != n:
        raise ValueError('Batched inputs need exactly one image grid per row')
    reference = visuals[0]
    for i, visual in enumerate(visuals):
        validate_visual(visual)
        if visual.grids != [grids[i]]:
            raise ValueError(f'Target row {i} differs from cached source geometry')
        if (visual.main.shape[1], visual.main.dtype, visual.main.device) != (
                reference.main.shape[1], reference.main.dtype, reference.main.device):
            raise ValueError(f'Row {i}: visual hidden size/dtype/device differs from row 0')
        if visual.output_type is not reference.output_type:
            raise ValueError(f'Row {i}: visual output container type differs from row 0')
    tokens = [int(visual.main.shape[0]) for visual in visuals]
    image_token = int(adapter.model.config.image_token_id)
    kinds = inputs_n.get('mm_token_type_ids')
    if kinds is None or kinds.shape != ids.shape:
        raise ValueError('Transformers 5.8 requires mm_token_type_ids aligned with input_ids')
    host_ids, host_kinds = ids.detach().cpu(), kinds.detach().cpu()
    for i in range(n):
        placeholders = host_ids[i] == image_token
        where = placeholders.nonzero().flatten()
        if where.numel() != tokens[i] or int(where[-1] - where[0]) + 1 != tokens[i]:
            raise ValueError(f'Row {i}: image placeholders are not one contiguous block of {tokens[i]} tokens')
        if not torch.equal(host_kinds[i], placeholders.to(host_kinds.dtype)):
            raise ValueError(f'Row {i}: mm_token_type_ids are not exactly the image placeholder indicator')
    batched = QwenVisual(torch.cat([visual.main for visual in visuals], dim=0),
                         [torch.cat([visual.deepstack[j] for visual in visuals], dim=0) for j in range(3)],
                         grids, reference.output_type, {'batch_size': n, 'row_visual_tokens': tokens})
    patches = sum(t * h * w for t, h, w in grids)
    module = adapter.visual
    base = adapter.model.model
    if getattr(module, '_v8_visual_owner', None) is not None or 'forward' in module.__dict__:
        raise RuntimeError('Nested/foreign visual cache is unsupported')
    saved = module.__dict__.get('forward', _MISSING)
    saved_rope = base.__dict__.get('rope_deltas', _MISSING)
    stats = {'calls': 0, 'records': [], 'schema': SCHEMA, 'batch_size': n, 'row_visual_tokens': tokens}

    def forward(hidden_states, grid_thw=None, **kwargs):
        if grid_thw is None or _row_grids(grid_thw) != grids:
            raise ValueError('Visual delivery grid differs from batched source geometry')
        if hidden_states.shape[0] != patches:
            raise ValueError('Visual pixel patch count differs from batched source geometry')
        stats['calls'] += 1
        # Fresh container per call: HF replaces pooler_output with its split tuple in place.
        return materialize_visual_output(batched)

    module._v8_visual_owner = stats
    module.forward = forward
    try:
        with DeepStackIntervention(adapter.model, '111', audit=audit) as intervention:
            stats['records'] = intervention.records
            yield stats
    finally:
        if saved is _MISSING:
            delattr(module, 'forward')
        else:
            module.forward = saved
        delattr(module, '_v8_visual_owner')
        if saved_rope is _MISSING:
            base.__dict__.pop('rope_deltas', None)
        else:
            base.rope_deltas = saved_rope


def encode_batch(adapter, inputs_n):
    """Encode stacked native inputs with ONE real visual call and split it per row.

    Optional helper outside the common interface, for the batched native/self-patch audit
    only (module docstring, "Vision tower and audits"): ``use_batch(adapter,
    encode_batch(adapter, inputs_n), inputs_n)`` must reproduce the unpatched batched forward on
    ``inputs_n`` exactly. Row i is split by image_grid_thw row i, the order the HF model uses.
    Never use it for experimental sources; those stay batch-one ``adapter.encode`` outputs.
    """
    import torch
    from cloud_prep.measurement_v8.qwen_adapter import normalize_visual_output, validate_visual
    module = adapter.visual
    if getattr(module, '_v8_visual_owner', None) is not None or 'forward' in module.__dict__:
        raise RuntimeError('Source encoding requires unpatched visual.forward')
    grids = _row_grids(inputs_n['image_grid_thw'])
    output = module(inputs_n['pixel_values'].to(dtype=module.dtype), grid_thw=inputs_n['image_grid_thw'], return_dict=True)
    merge = int(adapter.merge_size) ** 2
    sizes = [t * h * w // merge for t, h, w in grids]
    mains = torch.split(output.pooler_output, sizes)
    streams = [torch.split(feature, sizes) for feature in output.deepstack_features]
    rows = []
    for i, grid in enumerate(grids):
        part = type(output)(pooler_output=mains[i], deepstack_features=[stream[i] for stream in streams])
        visual = normalize_visual_output(part, [grid])
        validate_visual(visual)
        rows.append(visual)
    return rows
