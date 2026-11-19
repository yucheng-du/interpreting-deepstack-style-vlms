"""Batched DeepStack-L delivery: n composed visuals, one official batch-n forward.

Imports are tensor-free; torch loads lazily inside the CUDA functions only. The
batch-one path (DeepStackCoreAdapter.use, runner.generate, score_options) is not
touched; this module is only used by a batched execution_version.

Scope. The conditions of one query/readout share one prompt and one processed
geometry; only the injected visual tensors differ. stack_inputs() turns n
batch-one prepared inputs (DeepStackCoreAdapter.prepare(row, variant, source)[0])
into the batch-n input the official DeepstackLlamaForCausalLM receives for n
prompts with n images: input_ids/attention_mask [n, L] (no padding), images a
list of the n per-row processed view tensors [1 + gw*gh, 3, 336, 336],
image_sizes/bestgrid_list lists of n tuples. Unpatched, that dict is the batched
NATIVE forward (official process_deepstack encodes the n images itself).
use_batch() replaces the official process_deepstack exactly as the batch-one
use() does, but returns the cached main and four later groups of visuals[i] for
batch row i. Nothing else in the official model is replaced.

Row order (exact and structural, no padding anywhere):
  * main. The replacement returns main as a fresh [n, 576, 4096] tensor whose row
    i is visuals[i].main. DeepstackMetaForCausalLM.prepare_inputs_labels_for_
    multimodal walks the batch rows b = 0..n-1 in order and, for each <image>
    placeholder of row b, inserts image_features[cur_image_idx], advancing
    cur_image_idx by one per placeholder. Exactly one placeholder per row
    (checked), so row b receives main[b] = visuals[b].main at its 576 expanded
    positions [s, s + 576), where s is the placeholder index (the same for all
    rows, because all rows have identical input_ids).
  * later. The replacement returns later as a fresh [4, n, 576, 4096] tensor with
    later[g][i] = visuals[i].later[g] (the container the official
    process_plain_deepstack returns, stacked on dim 1). DeepstackLlamaForCausalLM
    .forward (and .generate for the prefill) sets image_id_pos =
    torch.where(input_ids == IMAGE_TOKEN_INDEX) on the unexpanded ids. torch.where
    (nonzero) returns indices in row-major order, so with one placeholder per row
    the i-th occurrence is (b_id = i, l_id = s). CustomLlamaModel.forward adds
    local_features[g][i] to hidden_states[b_id, l_id:l_id + 576] before decoder
    layer start + g * interval. Row i therefore gets visuals[i].later[g] at its own
    expanded positions. No row reads another row's tensor.
  * zero padding. Groups g >= gw*gh of a row are that row's own official zeros.
    They are checked to be exactly zero for every row on entry, so the residual
    add for those rows is +0, structurally inactive per row, including in a
    mixed-grid batch.
  * generation. The official generate calls process_deepstack once for the whole
    batch, builds the prefill inputs_embeds with prepare_inputs_labels_for_
    multimodal (order as above) and stores image_id_pos/local_features in
    model.kwargs_for_generation. The prefill forward consumes and clears them.
    Decode steps receive images=None, so the replacement is never called again
    and visuals are not re-delivered (stats['calls'] == 1 per generate call).

Numerics. Delivery itself is exact. The batched native forward equals the batched
self-patch bitwise, and the official batch-n CLIP encoding equals the batch-one
cached encodings bitwise. A row's value differs from batch one only where cuBLAS
picks a different GEMM kernel for the batch token count n * (L + 575). This was
measured on A100 with torch 2.1.2+cu121 and transformers 4.39.3, for prompts of
about 650 expanded tokens. n in {1, 2, 5, 6, 7, 8} was bitwise identical to batch
one. n in {3, 4} showed BF16-level differences (max |d log-prob| 0.077), with
generated tokens unchanged. Do not assume bitwise equality for other shapes.

Generation output contract, as in batch one: model.generate(inputs=input_ids,
**the other stacked inputs, generation_config=..., stopping_criteria=...,
**greedy_kwargs(model)) returns only the NEW tokens, a LongTensor [n, T] with
T <= max_new_tokens. A row that emits EOS before the others is right-filled with
generation_config.pad_token_id (0 for this checkpoint). The batch-one sequence of
row i is its prefix up to and including the first EOS id. A row without EOS
keeps all T tokens, and then T == max_new_tokens unless a stopping criterion
fired. Callers must clear model.kwargs_for_generation in a finally clause, as
runner.generate does. use_batch also clears it on exit.
"""
from contextlib import contextmanager

BATCH_SCHEMA = 'deepstack-l-common-core-adapter-v13-v1/process_deepstack_batched_replacement-v1'
INPUT_KEYS = frozenset(('input_ids', 'attention_mask', 'images', 'image_sizes', 'bestgrid_list'))
VIEW_SIZE = (3, 336, 336)


def _single(values, row, name):
    if values is None or len(values) != 1:
        raise ValueError(f'Row {row}: one native image per prepared input required ({name})')
    return values[0]


def stack_inputs(adapter, inputs_list):
    """Stack n batch-one prepared inputs into one unpadded batch-n input dict.

    All rows must have identical input_ids, with exactly one native placeholder,
    and an all-ones attention mask. Pixel views may differ per row; for the
    batched NATIVE audit they are the views of sources a, b and degraded_a. Each
    row's view tensor is passed by reference and never copied or modified.
    """
    import torch
    from cloud_prep.measurement_v14.accepted_deepstack_core import IMAGE_TOKEN_INDEX, active_groups
    from cloud_prep.measurement_v13.deepstack_geometry import best_grid
    inputs_list = list(inputs_list)
    if not inputs_list:
        raise ValueError('At least one prepared input is required')
    reference = inputs_list[0]['input_ids']
    if reference.ndim != 2 or reference.shape[0] != 1 or reference.shape[1] < 2:
        raise ValueError('Batch-one prepared input_ids [1, L] required')
    if int((reference == IMAGE_TOKEN_INDEX).sum().item()) != 1:
        raise ValueError('One native image placeholder required')
    images, sizes, grids = [], [], []
    for row, inp in enumerate(inputs_list):
        if set(inp) != INPUT_KEYS:
            raise ValueError(f'Row {row}: unexpected prepared input keys {sorted(inp)}')
        view = _single(inp['images'], row, 'images')
        if images and (view.dtype != images[0].dtype or view.device != images[0].device):
            raise ValueError(f'Row {row}: processed view dtype/device differs')
        ids, mask = inp['input_ids'], inp['attention_mask']
        if (ids.shape != reference.shape or ids.dtype != reference.dtype or ids.device != reference.device
                or not torch.equal(ids, reference)):
            raise ValueError(f'Row {row}: batched rows need identical input_ids (no padding)')
        if (mask is None or mask.shape != ids.shape or mask.device != ids.device
                or not bool(torch.all(mask == 1).item())):
            raise ValueError(f'Row {row}: unpadded all-ones attention_mask required')
        size = tuple(int(x) for x in _single(inp['image_sizes'], row, 'image_sizes'))
        grid = tuple(int(x) for x in _single(inp['bestgrid_list'], row, 'bestgrid_list'))
        if grid != best_grid(size):
            raise ValueError(f'Row {row}: crop grid is not the native grid of its image size')
        if tuple(view.shape) != (1 + active_groups(grid), *VIEW_SIZE):
            raise ValueError(f'Row {row}: processed views disagree with the native grid')
        images.append(view); sizes.append(size); grids.append(grid)
    return dict(input_ids=torch.cat([inp['input_ids'] for inp in inputs_list], dim=0),
                attention_mask=torch.cat([inp['attention_mask'] for inp in inputs_list], dim=0),
                images=images, image_sizes=sizes, bestgrid_list=grids)


def _check_batch(visuals, inputs_n):
    """Per-row geometry exactly as the batch-one use(), plus the row-order premises."""
    import torch
    from cloud_prep.measurement_v14.accepted_deepstack_adapter import validate_visual
    from cloud_prep.measurement_v14.accepted_deepstack_core import (
        CONFIGURED_LATER, IMAGE_TOKEN_INDEX, active_groups)
    if not visuals:
        raise ValueError('At least one visual is required')
    ids = inputs_n.get('input_ids')
    if ids is None or ids.ndim != 2 or ids.shape[0] != len(visuals):
        raise ValueError('input_ids [n, L] must have one row per visual')
    n = len(visuals)
    for name in ('images', 'image_sizes', 'bestgrid_list'):
        if inputs_n.get(name) is None or len(inputs_n[name]) != n:
            raise ValueError(f'{name} must have one entry per batch row')
    for name in ('past_key_values', 'cache_position', 'position_ids', 'inputs_embeds', 'labels'):
        if inputs_n.get(name) is not None:
            raise ValueError(f'Batched delivery requires fresh processor inputs, not {name!r}')
    if not torch.equal(ids, ids[:1].expand_as(ids)):
        raise ValueError('Batched rows need identical input_ids (no padding)')
    if int((ids[0] == IMAGE_TOKEN_INDEX).sum().item()) != 1:
        raise ValueError('One native image placeholder per row required')
    mask = inputs_n.get('attention_mask')
    if mask is not None and (mask.shape != ids.shape or not bool(torch.all(mask == 1).item())):
        raise ValueError('Batched delivery requires an unpadded all-ones attention_mask')
    first = visuals[0]
    inactive = []
    for row, visual in enumerate(visuals):
        validate_visual(visual)
        if visual.main.dtype != first.main.dtype or visual.main.device != first.main.device:
            raise ValueError(f'Row {row}: visual dtype/device differs within the batch')
        grid, size = tuple(visual.grid), tuple(visual.image_size)
        if tuple(inputs_n['bestgrid_list'][row]) != grid or tuple(inputs_n['image_sizes'][row]) != size:
            raise ValueError(f'Row {row}: target inputs differ from cached source geometry')
        if inputs_n['images'][row].shape[0] != 1 + active_groups(grid):
            raise ValueError(f'Row {row}: processed views disagree with the cached native grid')
        inactive.extend(visual.later[g] for g in range(active_groups(grid), CONFIGURED_LATER))
    if inactive and int(sum(value.count_nonzero() for value in inactive).item()):
        raise ValueError('A zero-padded DeepStack later group is not exactly zero in its row')


@contextmanager
def use_batch(adapter, visuals, inputs_n, *, audit=False):
    """Replace official process_deepstack: batch row i receives visuals[i].

    Every forward on inputs_n inside the context (score forwards, teacher-forced
    continuations of the same rows, and the generation prefill) gets main and
    later stacked in visuals order. Row order is described in the module
    docstring. stats['calls'] counts actual deliveries, one per batched forward
    or generate call. stats['batch_size'] = n. With audit=True one record per
    delivery is kept, as in the batch-one use(). On exit the class method is
    restored and model.kwargs_for_generation is cleared, as in the batch-one
    use().
    """
    import torch
    from cloud_prep.measurement_v14.accepted_deepstack_core import CONFIGURED_LATER, active_groups
    visuals = list(visuals)
    _check_batch(visuals, inputs_n)
    model = adapter.model
    if 'process_deepstack' in model.__dict__:
        raise RuntimeError('Nested/foreign visual delivery is unsupported')
    if getattr(model, 'kwargs_for_generation', None) is not None:
        raise RuntimeError('Stale official generation state before batched delivery')
    n = len(visuals)
    grids = [tuple(v.grid) for v in visuals]
    views = [(int(x.data_ptr()), tuple(x.shape)) for x in inputs_n['images']]
    stats = {'calls': 0, 'records': [], 'schema': BATCH_SCHEMA, 'batch_size': n}

    def deliver(images, bestgrid_list):
        if bestgrid_list is None or len(bestgrid_list) != n or any(
                tuple(g) != grid for g, grid in zip(bestgrid_list, grids)):
            raise ValueError('Injected crop grids differ from the prepared batch rows')
        if images is None or len(images) != n or [
                (int(x.data_ptr()), tuple(x.shape)) for x in images] != views:
            raise ValueError('Delivery received processed views other than the prepared batch')
        stats['calls'] += 1
        main = torch.stack([v.main for v in visuals], dim=0)
        later = torch.stack([torch.stack([v.later[g] for v in visuals], dim=0)
                             for g in range(CONFIGURED_LATER)], dim=0)
        if audit:
            stats['records'].append({'call': stats['calls'], 'completed': True, 'batch_size': n,
                'row_grids': [list(g) for g in grids],
                'paths': [{'entry': 'main', 'dtype': str(main.dtype), 'shape': list(main.shape),
                           'site': 'expanded_image_positions_at_LM_input', 'row_order': 'visuals order'},
                          *({'entry': f'later{g}', 'dtype': str(later.dtype), 'shape': list(later[g].shape),
                             'site': f'residual_before_decoder_{adapter.start + g * adapter.interval}',
                             'inactive_zero_padded_rows': [i for i, grid in enumerate(grids)
                                                           if g >= active_groups(grid)]}
                            for g in range(CONFIGURED_LATER))]})
        return main, later

    model.process_deepstack = deliver
    try:
        yield stats
    finally:
        del model.process_deepstack
        # The official generate stashes local streams for its first forward.
        model.kwargs_for_generation = None
