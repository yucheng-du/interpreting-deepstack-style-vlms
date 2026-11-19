"""Original DeepStack-L geometry/delivery/dose for the common v12 P1/P2 interface.

Imports are tensor-free; torch/llava load lazily inside CUDA methods only.

Native geometry (never Qwen's): `main` is the 24x24 projected CLIP grid of the
336-squash global view, placed by the official prepare_inputs_labels_for_multimodal
at the 576 expanded positions of the single <image> placeholder; `later0..3` are
the four configured 2D phase groups of the high-resolution crop grid, added by
CustomLlamaModel to the residual stream of those same 576 positions before
decoder layers start+i*interval (1..4) of model.model.layers. Groups at or beyond
the native crop count gw*gh are the official zero padding: structurally inactive,
never selected, contributing exactly zero (not a missing observation).

The v12 conditions, designs, records and analyzers are reused unchanged. This
module only supplies the model-specific adapter interface they call:
prepare/encode/audit_sources/compose/use and, for dose arms, calibrate.
"""
from contextlib import contextmanager
from copy import deepcopy
import math
from types import SimpleNamespace

from cloud_prep.measurement_v13.deepstack_geometry import (
    validate_config, best_grid, geometry_record, roi_indices, token_order,
)
from cloud_prep.measurement_v13.model_extensions import DEEPSTACK7B
from cloud_prep.measurement_v13.deepstack_adapter import DeepStackVisual, validate_visual
from cloud_prep.measurement_v12.v11_v2_dose_adapter import (
    UnmatchedDose, DoseVisual, TorchOps, describe_delta_geometry, native_dose, resolve_scale,
)

MODEL_KEY = 'deepstack7b'
IMAGE_TOKEN_INDEX = -200          # llava.constants.IMAGE_TOKEN_INDEX, verified at load
VISUAL_TOKENS = 576
HIDDEN = 4096                     # Vicuna-7B hidden size (checked against the loaded config)
CONFIGURED_LATER = 4
CORE_SCHEMA = 'deepstack-l-common-core-adapter-v13-v1'
ROI_RULE = ('positive-area intersection of each entry nominal patch support (global 24x24 grid or '
            'interleaved native phase group) with the donor bounding box; contextual CLIP features '
            'are not local receptive fields')
SCORE_SEPARATOR = ' '
SCORE_SEPARATOR_RULE = ('Vicuna-v1 native assistant turn is "ASSISTANT: <answer></s>"; the generation prompt '
                        'ends at "ASSISTANT:", so each complete candidate is scored as " " + label')

SPEC = dict(deepcopy(DEEPSTACK7B), checkpoint_parameters=DEEPSTACK7B['total_parameters'],
    bf16_parameter_bytes=2 * DEEPSTACK7B['total_parameters'],
    entry_groups={'all': ['later0', 'later1', 'later2', 'later3']},
    image_processor={'crop_size': {'height': 336, 'width': 336}, 'size': {'shortest_edge': 336}},
    conversation_template='vicuna_v1', image_placeholder='single <image> expanded to 576 positions',
    parameter_count_status='official_Hub_safetensors_metadata; verified as loaded count in CUDA smoke',
    preparation_status='common_P1_P2_bridge_v13_deepstack_core')


@contextmanager
def deepstack_registry():
    """Expose deepstack7b to the v12 planners in this process only.

    The planners only read identity fields; execution never routes through the
    Qwen/Granite factories (deepstack_executor builds this adapter itself).
    """
    from cloud_prep.measurement_v8 import model_registry as registry
    aliases = ('deepstack', 'deepstack-l', SPEC['model'].lower())
    sentinel = object()
    previous = registry.MODELS.get(MODEL_KEY, sentinel)
    old_aliases = {name: registry.ALIASES.get(name, sentinel) for name in aliases}
    if previous is not sentinel and previous != SPEC:
        raise ValueError('deepstack7b already has a different process-local identity')
    registry.MODELS[MODEL_KEY] = deepcopy(SPEC)
    registry.ALIASES.update({name: MODEL_KEY for name in aliases})
    try:
        yield
    finally:
        if previous is sentinel:
            registry.MODELS.pop(MODEL_KEY, None)
        else:
            registry.MODELS[MODEL_KEY] = previous
        for name, value in old_aliases.items():
            if value is sentinel:
                registry.ALIASES.pop(name, None)
            else:
                registry.ALIASES[name] = value


def entry_index(entry):
    """'main' -> 'main'; 'later2' -> 2 (zero-based native phase group)."""
    if entry == 'main':
        return 'main'
    if isinstance(entry, str) and entry.startswith('later') and entry[5:].isdigit():
        return int(entry[5:])
    raise ValueError(f'Unknown DeepStack entry {entry!r}')


def active_groups(grid):
    return grid[0] * grid[1]


def entry_names(role, grid):
    if role == 'main':
        return ['main']
    if role == 'later':
        return [f'later{i}' for i in range(active_groups(grid))]
    raise ValueError('changed_entry must be main/later')


def inactive_groups(grid):
    return [f'later{i}' for i in range(active_groups(grid), CONFIGURED_LATER)]


def entry_specs(role, grid, start=1, interval=1):
    """Native boundaries in model.model.layers; only active groups are entries."""
    if role == 'main':
        return [{'entry': 'main', 'decoder_before': 0, 'decoder_after': None,
                 'kind': 'input_embedding_replacement_at_expanded_image_positions',
                 'pre_residual_reason': 'replacement_has_no_additive_pre_residual'}]
    gw, gh = grid
    return [{'entry': f'later{i}', 'decoder_before': start + i * interval,
             'decoder_after': start + i * interval - 1, 'kind': 'pre_decoder_residual_addition',
             'pre_residual_reason': None, 'native_phase_group': i,
             'phase_offset_xy': [i % gw, i // gw]} for i in range(active_groups(grid))]


def inactive_records(role, grid, start=1, interval=1):
    if role != 'later':
        return []
    return [{'entry': name, 'status': 'inactive_zero_padded_group',
             'decoder_before': start + entry_index(name) * interval, 'contributes': 0.,
             'reason': 'official zero padding beyond the native crop grid; no source support'}
            for name in inactive_groups(grid)]


def entry_tensors(visual):
    return {'main': visual.main, **{f'later{i}': v for i, v in enumerate(visual.later)}}


def canvas(row, source):
    meta = row['source_image_metadata'][source]
    return int(meta['width']), int(meta['height'])


def shared_canvas(row):
    sizes = {canvas(row, name) for name in row['source_image_metadata']}
    if len(sizes) != 1:
        raise ValueError('All source images must share one canvas')
    return next(iter(sizes))


def selected_indices(size, bbox, entry, grid, region='roi', permutation='identity'):
    """Destination slots of one entry; inactive padded groups are never selected."""
    index = entry_index(entry)
    if index != 'main' and index >= active_groups(grid):
        return []
    if region == 'all':
        return list(range(VISUAL_TOKENS))
    if bbox is None:
        raise ValueError('ROI conditions require donor_roi_bbox_xyxy')
    roi = roi_indices(size, bbox, index, grid=grid, permutation=permutation)
    if region == 'roi':
        return roi
    if region == 'complement':
        chosen = set(roi)
        return [i for i in range(VISUAL_TOKENS) if i not in chosen]
    raise ValueError('Unknown injection region')


def preflight_geometry(row, conditions):
    """CPU coordinate preflight with DeepStack's own native grid and phase supports."""
    size = shared_canvas(row)
    grid = best_grid(size)
    permutations = []
    for c in conditions:
        p = c.get('token_permutation', 'identity')
        if p not in permutations:
            permutations.append(p)
    counts = {}
    box = row.get('donor_roi_bbox_xyxy')
    if box is not None and any(c.get('region') in ('roi', 'complement') for c in conditions):
        x0, y0, x1, y1 = box
        if not (0 <= x0 < x1 <= size[0] and 0 <= y0 < y1 <= size[1]):
            raise ValueError('ROI is outside the shared canvas')
        for p in permutations:
            counts[str(p)] = {name: len(selected_indices(size, box, name, grid, 'roi', p))
                              for name in ['main', *entry_names('later', grid)]}
    if counts and any(c != next(iter(counts.values())) for c in counts.values()):
        raise ValueError('Matched local movement changed nominal ROI support')
    return {'image_hw': [size[1], size[0]], 'image_wh': list(size), 'tile_grid': row.get('tile_grid'),
            'deepstack_native_grid': list(grid), 'main_tokens': VISUAL_TOKENS,
            'configured_later_groups': CONFIGURED_LATER, 'active_later_groups': active_groups(grid),
            'zero_padded_later_groups': inactive_groups(grid), 'permutations': permutations,
            'nominal_roi_token_counts': counts,
            'nominal_roi_token_counts_scope': 'active entries only; padded groups are structurally inactive',
            'geometry': geometry_record(size, box),
            'status': 'CPU coordinate preflight only; actual processor and delivery await CUDA'}


def official_grid(size, image_processor, num_crops):
    """The grid DeepstackLlamaForCausalLM.generate computes for itself."""
    from llava.mm_utils import get_possible_resolutions_by_crops, select_best_resolution
    crop = image_processor.crop_size
    best = select_best_resolution(size, get_possible_resolutions_by_crops(num_crops, crop))
    return best[0] // crop['width'], best[1] // crop['height']


class DeepStackCoreAdapter:
    """The adapter surface the v10/v11-v2 inference loop calls, natively for DeepStack-L."""

    def __init__(self, base):
        self.base = base
        self.model, self.tokenizer, self.image_processor = base.model, base.tokenizer, base.image_processor
        self.processor = SimpleNamespace(tokenizer=base.tokenizer, image_processor=base.image_processor)
        self.device = base.device
        config = self.model.config.to_dict()
        geometry = validate_config(config)
        self.entry_layers = geometry['later_entries']
        self.start = config.get('deepstack_start_id', 1)
        self.interval = config.get('deepstack_interval', 1)
        self.max_context = int(config['tokenizer_model_max_length'])
        self.metadata = {**base.metadata, 'core_schema': CORE_SCHEMA, 'later_entry_layers': self.entry_layers,
                         'context_budget_expanded_tokens': self.max_context}

    # ---- inputs -----------------------------------------------------------------
    def prompt(self, variant):
        from llava.constants import DEFAULT_IMAGE_TOKEN
        from llava.conversation import conv_templates
        conv = conv_templates['vicuna_v1'].copy()
        conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + '\n' + variant['prompt'])
        conv.append_message(conv.roles[1], None)
        return conv.get_prompt() + variant.get('assistant_prefix', '')

    def prepare(self, row, variant, source):
        from PIL import Image
        from llava.mm_utils import process_images, tokenizer_image_token
        path = row['source_images'][source]
        if isinstance(path, (list, tuple)):
            if len(path) != 1:
                raise ValueError('One static image per source required')
            path = path[0]
        with Image.open(path) as raw:
            image = raw.convert('RGB')
        try:
            size = image.size
            pixels = process_images([image], self.image_processor, self.model.config)
        finally:
            image.close()
        if tuple(size) != canvas(row, source):
            raise ValueError('Decoded image size differs from manifest metadata')
        grid = best_grid(size)
        if tuple(official_grid(size, self.image_processor, self.model.config.num_crops)) != tuple(grid):
            raise ValueError('CPU native grid differs from the official resolution selection')
        prompt = self.prompt(variant)
        ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX,
                                    return_tensors='pt').unsqueeze(0).to(self.device)
        if int((ids == IMAGE_TOKEN_INDEX).sum().item()) != 1:
            raise ValueError('One image placeholder required')
        images = [x.to(device=self.device, dtype=self.model.dtype) for x in pixels]
        if len(images) != 1 or tuple(images[0].shape) != (1 + active_groups(grid), 3, 336, 336):
            raise ValueError('Native processor views disagree with CPU geometry')
        return dict(input_ids=ids, attention_mask=ids.new_ones(ids.shape), images=images,
                    image_sizes=[tuple(size)], bestgrid_list=[tuple(grid)]), prompt

    @staticmethod
    def expanded_length(inputs):
        return int(inputs['input_ids'].shape[1]) - 1 + VISUAL_TOKENS

    @staticmethod
    def image_start(inputs):
        ids = inputs['input_ids'].detach().cpu().tolist()
        if len(ids) != 1:
            raise ValueError('Batch one required')
        found = [i for i, t in enumerate(ids[0]) if t == IMAGE_TOKEN_INDEX]
        if len(found) != 1:
            raise ValueError('One native image placeholder required')
        return found[0]

    def pixel_digest_inputs(self, inputs):
        """Processed views + actual size + native grid, all hashed (not only tensors)."""
        import torch
        return {'images': inputs['images'],
                'image_sizes': torch.tensor([list(s) for s in inputs['image_sizes']]),
                'bestgrid_list': torch.tensor([list(g) for g in inputs['bestgrid_list']])}

    # ---- source features ------------------------------------------------------------
    def encode(self, inputs):
        import torch
        if 'process_deepstack' in self.model.__dict__:
            raise RuntimeError('Source encoding requires the unpatched official process_deepstack')
        with torch.inference_mode():
            main, later = self.model.process_deepstack(inputs['images'], inputs['bestgrid_list'])
        size, grid = tuple(inputs['image_sizes'][0]), tuple(inputs['bestgrid_list'][0])
        if tuple(main.shape) != (1, VISUAL_TOKENS, HIDDEN) or tuple(later.shape) != (CONFIGURED_LATER, 1, VISUAL_TOKENS, HIDDEN):
            raise ValueError('Official process_deepstack output geometry differs')
        visual = DeepStackVisual(main[0].detach().clone(), [later[i][0].detach().clone() for i in range(CONFIGURED_LATER)],
                                 size, grid, geometry_record(size))
        validate_visual(visual)
        for index in range(active_groups(grid), CONFIGURED_LATER):
            if bool(visual.later[index].count_nonzero().item()):
                raise ValueError('Official padded visual group is not zero')
        return visual

    encode_visual = encode

    def audit_sources(self, source_inputs, sources, row):
        """Actual processor/grid/padding/permutation checks; no feature equivariance demand."""
        import torch
        if not source_inputs or set(source_inputs) != set(sources):
            raise ValueError('Audit needs the complete source input/cache mapping')
        first = next(iter(sources.values()))
        size, grid = tuple(first.image_size), tuple(first.grid)
        for name, visual in sources.items():
            validate_visual(visual)
            inp = source_inputs[name]
            if (tuple(visual.image_size), tuple(visual.grid)) != (size, grid):
                raise ValueError('Source geometry is not matched')
            if tuple(inp['bestgrid_list'][0]) != grid or tuple(inp['image_sizes'][0]) != size:
                raise ValueError('Prepared inputs and cached geometry differ')
        order = token_order('swap_halves')
        ids = torch.tensor(order, device=first.main.device)
        inverse = all(torch.equal(x.index_select(0, ids).index_select(0, ids), x)
                      for v in sources.values() for x in (v.main, *v.later))
        padded = all(not bool(v.later[i].count_nonzero().item())
                     for v in sources.values() for i in range(active_groups(grid), CONFIGURED_LATER))
        active_nonzero = all(bool(v.later[i].count_nonzero().item())
                             for v in sources.values() for i in range(active_groups(grid)))
        grid_exact = tuple(official_grid(size, self.image_processor, self.model.config.num_crops)) == grid
        start = self.image_start(next(iter(source_inputs.values())))
        result = {'schema': CORE_SCHEMA, 'image_size': list(size), 'native_grid': list(grid),
                  'processed_views': [list(next(iter(source_inputs.values()))['images'][0].shape)],
                  'visual_tokens_per_entry': VISUAL_TOKENS, 'expanded_image_positions': [start, start + VISUAL_TOKENS],
                  'configured_later_groups': CONFIGURED_LATER, 'active_later_groups': active_groups(grid),
                  'zero_padded_later_groups': inactive_groups(grid), 'later_entry_layers': self.entry_layers,
                  'half_swap_inverse_exact': inverse, 'padded_groups_zero_exact': padded,
                  'official_grid_selection_exact': grid_exact, 'active_groups_nonzero': active_nonzero,
                  'feature_pixel_equivariance_required': False, 'roi_rule': ROI_RULE}
        box = row.get('donor_roi_bbox_xyxy')
        if box is not None:
            result['roi_tokens_by_entry'] = {name: len(selected_indices(size, box, name, grid))
                                             for name in ['main', *entry_names('later', grid)]}
        if not (inverse and padded and grid_exact and active_nonzero):
            raise RuntimeError(f'DeepStack source audit failed: {result}')
        return result

    # ---- composition ----------------------------------------------------------------
    @staticmethod
    def _check(*visuals):
        first = visuals[0]
        for visual in visuals:
            validate_visual(visual)
            if (visual.image_size, visual.grid) != (first.image_size, first.grid):
                raise ValueError('Main/context/donor visual geometry differs')
            if visual.main.dtype != first.main.dtype or visual.main.device != first.main.device:
                raise ValueError('Source dtype/device changed')

    def compose(self, sources, condition, row, inputs=None):
        import torch
        if condition.get('intervention_type') == 'local_role':
            return self._local_role(sources, condition, row, inputs)
        main_name = condition['main']
        donor_name = condition.get('donor_source', condition.get('donor')) or main_name
        if condition.get('ds_sources') is not None and any(x != donor_name for x in condition['ds_sources']):
            raise ValueError('DeepStack later groups take one donor source in this protocol')
        if condition.get('entry_group', 'all') not in ('all', 'all_later'):
            raise ValueError('DeepStack has only the all-later-groups group')
        base, donor = sources[main_name], sources[donor_name]
        self._check(base, donor)
        if inputs is not None and tuple(inputs['bestgrid_list'][0]) != tuple(base.grid):
            raise ValueError('Prepared input and source geometry differ')
        size, grid = tuple(base.image_size), tuple(base.grid)
        if row.get('source_image_metadata') and (canvas(row, main_name), canvas(row, donor_name)) != (size, size):
            raise ValueError('Main/donor canvas differs')
        region = condition.get('region', 'all')
        permutation = condition.get('token_permutation', 'identity')
        order = token_order(permutation)
        perm = torch.tensor(order, device=base.main.device, dtype=torch.long)
        masks, later = {}, []
        for index in range(CONFIGURED_LATER):
            name = f'later{index}'
            moved = donor.later[index].index_select(0, perm)
            if index >= active_groups(grid):
                masks[name] = []
                later.append(moved.clone())   # official zeros in both sources
                continue
            if region == 'all':
                masks[name] = list(range(VISUAL_TOKENS))
                later.append(moved.clone())
                continue
            chosen = selected_indices(size, row.get('donor_roi_bbox_xyxy'), name, grid, region, permutation)
            if not chosen:
                raise ValueError('Requested region contains no active DeepStack tokens')
            feature = base.later[index].clone()
            ids = torch.tensor(chosen, device=feature.device, dtype=torch.long)
            feature.index_copy_(0, ids, moved.index_select(0, ids))
            masks[name] = chosen
            later.append(feature)
        result = DeepStackVisual(base.main.clone(), later, size, grid, {
            'schema': CORE_SCHEMA, 'main_source': main_name, 'later_source': donor_name,
            'ds_sources': [donor_name] * CONFIGURED_LATER, 'entry_group': 'all', 'region': region,
            'token_permutation': permutation, 'selected_token_indices': masks, 'roi_rule': ROI_RULE,
            'main_preserved': True, 'native_grid': list(grid), 'active_later_groups': active_groups(grid),
            'inactive_zero_padded_groups': inactive_groups(grid), 'source_cache_immutable': True})
        validate_visual(result)
        return result

    def _local_role(self, sources, condition, row, inputs):
        import torch
        role = condition.get('changed_entry')
        if role not in ('main', 'later'):
            raise ValueError('Local operation requires changed_entry main/later')
        if condition.get('region') != 'roi' or condition.get('entry_group') != 'all':
            raise ValueError('Matched local role operation uses ROI and all entry group')
        main_name = condition['main']
        context_name = condition.get('context_source', main_name)
        donor_name = condition['donor_source']
        if any(name not in sources for name in (main_name, context_name, donor_name)):
            raise ValueError('Missing main/context/donor visual source')
        main, context, donor = sources[main_name], sources[context_name], sources[donor_name]
        self._check(main, context, donor)
        if inputs is not None and tuple(inputs['bestgrid_list'][0]) != tuple(main.grid):
            raise ValueError('Prepared input and source geometry differ')
        if len({canvas(row, n) for n in (main_name, context_name, donor_name)}) != 1:
            raise ValueError('Main/context/donor canvas differs')
        size, grid = tuple(main.image_size), tuple(main.grid)
        box = row.get('donor_roi_bbox_xyxy')
        permutation = condition.get('token_permutation', 'identity')
        order = token_order(permutation)
        selected = {}
        def replace(current, other, name):
            chosen = selected_indices(size, box, name, grid, 'roi', permutation)
            selected[name] = chosen
            if entry_index(name) == 'main' or entry_index(name) < active_groups(grid):
                if not chosen:
                    raise ValueError('Local ROI has no support in an active DeepStack entry')
            result = current.clone()
            if chosen:
                dest = torch.tensor(chosen, device=current.device, dtype=torch.long)
                src = torch.tensor([order[i] for i in chosen], device=current.device, dtype=torch.long)
                result.index_copy_(0, dest, other.index_select(0, src))
            return result
        new_main = replace(main.main, donor.main, 'main') if role == 'main' else main.main.clone()
        later = [replace(current, other, f'later{i}') if role == 'later' else current.clone()
                 for i, (current, other) in enumerate(zip(context.later, donor.later))]
        result = DeepStackVisual(new_main, later, size, grid, {
            'schema': CORE_SCHEMA, 'main_source': main_name, 'context_source': context_name,
            'donor_source': donor_name, 'changed_entry': role, 'entry_group': 'all', 'region': 'roi',
            'token_permutation': permutation, 'selected_token_indices': selected,
            'selected_main_token_indices': selected.get('main', []),
            'selected_later_token_indices': {k: v for k, v in selected.items() if k != 'main'},
            'main_preserved': role != 'main', 'later_preserved': role != 'later',
            'native_grid': list(grid), 'active_later_groups': active_groups(grid),
            'inactive_zero_padded_groups': inactive_groups(grid), 'donor_roi_bbox_xyxy': list(box) if box else None,
            'source_destination': condition.get('source_destination'), 'roi_rule': ROI_RULE,
            'source_cache_immutable': True})
        validate_visual(result)
        return result

    # ---- delivery ---------------------------------------------------------------
    @contextmanager
    def use(self, visual, inputs, *, audit=False):
        """Replace official process_deepstack for each fresh forward/generate.

        prepare_inputs_labels_for_multimodal places `main` at the expanded image
        positions and CustomLlamaModel adds each later group before its native
        decoder layer; no other model code is replaced. Counts actual calls.
        """
        import torch
        validate_visual(visual)
        if tuple(inputs['bestgrid_list'][0]) != tuple(visual.grid) or tuple(inputs['image_sizes'][0]) != tuple(visual.image_size):
            raise ValueError('Target inputs differ from cached source geometry')
        model = self.model
        if 'process_deepstack' in model.__dict__:
            raise RuntimeError('Nested/foreign visual delivery is unsupported')
        stats = {'calls': 0, 'records': [], 'schema': CORE_SCHEMA + '/process_deepstack_replacement'}
        grid = tuple(visual.grid)
        def deliver(images, bestgrid_list):
            if bestgrid_list is None or tuple(bestgrid_list[0]) != grid:
                raise ValueError('Injected crop grid differs from prepared input')
            if images is None or len(images) != 1 or images[0].shape[0] != 1 + active_groups(grid):
                raise ValueError('Delivery received unexpected processed views')
            stats['calls'] += 1
            main = visual.main.clone().unsqueeze(0)
            later = torch.stack([value.clone().unsqueeze(0) for value in visual.later], dim=0)
            if audit:
                stats['records'].append({'call': stats['calls'], 'completed': True,
                    'paths': [{'entry': 'main', 'dtype': str(main.dtype), 'shape': list(main.shape),
                               'site': 'expanded_image_positions_at_LM_input'},
                              *({'entry': f'later{i}', 'dtype': str(later.dtype), 'shape': list(later[i].shape),
                                 'site': f'residual_before_decoder_{self.start + i * self.interval}',
                                 'inactive_zero_padded_group': i >= active_groups(grid)}
                                for i in range(CONFIGURED_LATER))]})
            return main, later
        model.process_deepstack = deliver
        try:
            yield stats
        finally:
            del model.process_deepstack
            # The official generate stashes local streams for its first forward.
            model.kwargs_for_generation = None

    use_visual_output = use


class DeepStackSiteProbe:
    """Scalar-only observation at native boundaries of model.model.layers.

    later i: output of decoder start+i*interval-1 is the pre-add residual; the
    input of decoder start+i*interval is the post-add state. main: input of
    decoder 0 at the expanded image positions (no additive pre-residual).
    Decode steps (no visual positions in the step) are skipped.
    """
    def __init__(self, adapter, visual, inputs):
        self.ops = adapter.ops
        self.specs = visual.metadata['entries']
        self.layers = adapter.model.model.layers
        start = DeepStackCoreAdapter.image_start(inputs)
        self.positions = list(range(start, start + VISUAL_TOKENS))
        self.rows, self.handles, self.pending = [], [], {}
        self.decode_steps_skipped = 0

    def _selected(self, raw, spec):
        x = self.ops.hidden(raw)
        shape = self.ops.shape(x)
        if len(shape) != 3 or shape[0] != 1:
            raise ValueError('Unexpected decoder state shape')
        if shape[1] <= max(self.positions):
            self.decode_steps_skipped += 1
            return None
        return self.ops.select_hidden(x, self.positions, spec['selected_token_indices'])

    def _pre_entry(self, spec):
        def hook(module, args, output):
            x = self._selected(output, spec)
            if x is None:
                return
            if spec['entry'] in self.pending:
                raise ValueError('Duplicate pre-entry observation before delivery')
            self.pending[spec['entry']] = self.ops.measure(x)
        return hook

    def _post_entry(self, spec):
        def hook(module, args, kwargs):
            raw = kwargs.get('hidden_states', args[0] if args else None)
            if raw is None:
                raise ValueError('No hidden_states at decoder entry')
            x = self._selected(raw, spec)
            if x is None:
                return
            pre = self.pending.pop(spec['entry'], None)
            if spec['decoder_after'] is not None and pre is None:
                raise ValueError('Missing native pre-entry state observation')
            post = self.ops.measure(x)
            delta_rms = spec['realized_delta']['rms']
            self.rows.append({'entry': spec['entry'], 'decoder_before': spec['decoder_before'],
                'pre_residual': pre, 'pre_residual_missing_reason': spec['pre_residual_reason'],
                'post_entry_state': post, 'selected_token_count': len(spec['selected_token_indices']),
                'realized_delta_to_actual_pre_residual': delta_rms / pre['rms'] if pre and pre['rms'] > 0 else None,
                'realized_delta_to_actual_post_state': delta_rms / post['rms'] if post['rms'] > 0 else None})
        return hook

    def __enter__(self):
        try:
            for spec in self.specs:
                if spec['decoder_after'] is not None:
                    self.handles.append(self.layers[spec['decoder_after']].register_forward_hook(self._pre_entry(spec)))
                self.handles.append(self.layers[spec['decoder_before']].register_forward_pre_hook(
                    self._post_entry(spec), with_kwargs=True))
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *exc):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        if exc[0] is None and self.pending:
            raise ValueError('Unpaired pre-entry observations')


class DeepStackDoseAdapter:
    """v11-v2 difference dose on DeepStack's own entries and boundaries.

    Scaling, target resolution and native-dose arithmetic are the shared
    accepted functions (resolve_scale/native_dose); only reference composition,
    per-entry token selection and the native probe sites are DeepStack-specific.
    Inactive padded groups are excluded from entries (they contribute exactly
    zero to RSS and L1) and are recorded as inactive_zero_padded_group.
    """
    family = 'deepstack'

    def __init__(self, base_adapter, design, *, natural=False, ops=None):
        self.base_adapter, self.design, self.natural = base_adapter, design, natural
        self.ops = ops or TorchOps()
        self.model, self.processor = getattr(base_adapter, 'model', None), getattr(base_adapter, 'processor', None)
        self.cache, self.geometry_cache, self.geometry_intervention = {}, {}, None
        self.model_revision = None

    def __getattr__(self, name):
        return getattr(self.base_adapter, name)

    def _reference(self, sources, reference, row, inputs):
        if self.natural:
            return self.base_adapter.compose(sources, {
                'main': reference['main'], 'context_source': reference['later'],
                'donor_source': reference['donor_source'], 'changed_entry': reference['changed_entry'],
                'intervention_type': 'local_role', 'region': 'roi', 'entry_group': 'all',
                'token_permutation': 'identity', 'source_destination': 'original'}, row, inputs=inputs)
        return self.base_adapter.compose(sources, {'main': reference['main'], 'donor_source': reference['later'],
            'region': 'all', 'entry_group': 'all', 'token_permutation': 'identity'}, row, inputs=inputs)

    def _selected(self, visual, condition, row, entry):
        size, grid = tuple(visual.image_size), tuple(visual.grid)
        if self.natural:
            if condition['region'] != 'roi':
                raise ValueError('Natural dose requires fixed donor ROI')
            donor = condition['reference_a']['donor_source']
            if canvas(row, donor) != size:
                raise ValueError('Donor canvas differs from composed geometry')
            return selected_indices(size, row['donor_roi_bbox_xyxy'], entry, grid, 'roi')
        if condition['region'] == 'all':
            return selected_indices(size, None, entry, grid, 'all')
        from cloud_prep.measurement_v10.full_data import bbox
        source = condition['reference_a'][condition['changed_entry']]
        if canvas(row, source) != size:
            raise ValueError('Changing-source canvas differs from composed geometry')
        box = bbox(row['scene'], row['changed_object'], condition['changing_layout'])
        return selected_indices(size, box, entry, grid, 'roi')

    def compose(self, sources, condition, row, inputs=None, calibration=None, opposite_calibration=None):
        design = self.design
        if condition.get('role') != 'dose_pair':
            return self.base_adapter.compose(sources, condition, row, inputs=inputs)
        signature = design.visual_signature(row, condition)
        if self.geometry_intervention != row['intervention_id']:
            self.geometry_cache.clear()
            self.geometry_intervention = row['intervention_id']
        if calibration is not None and (calibration['visual_signature'], calibration['calibration_id'],
                calibration['intervention_id']) != (signature, condition['calibration_id'], row['intervention_id']):
            raise ValueError('Calibration from a different visual intervention')
        opposite = opposite_calibration or (calibration or {}).get('opposite_calibration')
        if condition.get('target_policy') == 'opposite_native' and opposite is not None:
            other_c = design.opposite_condition(condition)
            for cal, expected in ((calibration, condition), (opposite, other_c)):
                if cal is None:
                    continue
                if (cal.get('visual_signature'), cal.get('calibration_id'), cal.get('intervention_id'),
                        cal.get('context_signature'), cal.get('changed_entry')) != (
                        design.visual_signature(row, expected), expected['calibration_id'], row['intervention_id'],
                        design.context_signature(row, expected), expected['changed_entry']):
                    raise ValueError('Opposite calibration from a different visual intervention/context')
            if calibration is not None:
                for field in ('anchor_query_id', 'anchor_query_sha256', 'anchor_prompt_sha256', 'model_revision'):
                    if calibration.get(field) != opposite.get(field):
                        raise ValueError('Paired calibration anchor/revision mismatch: ' + field)
        scale = resolve_scale(condition, calibration, opposite)
        if scale['status'] == 'unmatched':
            raise UnmatchedDose(scale)
        k = 0. if condition['donor_assignment'] == 'a' else scale['k']
        a = self._reference(sources, condition['reference_a'], row, inputs)
        b = self._reference(sources, condition['reference_b'], row, inputs)
        if (a.image_size, a.grid) != (b.image_size, b.grid):
            raise ValueError('Different a/b native geometry')
        grid = tuple(a.grid)
        aa, bb = entry_tensors(a), entry_tensors(b)
        result = dict(aa)
        metrics = []
        source_key, realized_key = (signature, 'source'), (signature, 'realized', k)
        need_source, need_realized = source_key not in self.geometry_cache, realized_key not in self.geometry_cache
        source_deltas, realized_deltas = {}, {}
        cals = {e['entry']: e for e in calibration['entries']} if calibration else {}
        base_adapter = self.base_adapter
        for spec in entry_specs(condition['changed_entry'], grid, base_adapter.start, base_adapter.interval):
            key = spec['entry']
            av, bv = aa[key], bb[key]
            if self.ops.shape(av) != self.ops.shape(bv):
                raise ValueError('Different a/b feature geometry')
            selected = self._selected(a, condition, row, key)
            if not selected:
                raise ValueError('No selected dose tokens in an active DeepStack entry')
            out = self.ops.mix(av, bv, selected, k)
            result[key] = out
            sa, sb, so = (self.ops.select(t, selected) for t in (av, bv, out))
            source_delta, realized_delta = self.ops.diff(sb, sa), self.ops.diff(so, sa)
            if need_source:
                source_deltas[key] = source_delta
            if need_realized:
                realized_deltas[key] = realized_delta
            metrics.append({**spec, 'selected_token_indices': selected, 'selected_token_count': len(selected),
                'hidden_size': self.ops.shape(av)[1], 'requested_k': k,
                'baseline_feature': self.ops.measure(sa), 'donor_feature': self.ops.measure(sb),
                'source_delta': self.ops.measure(source_delta),
                'applied_feature': self.ops.measure(so), 'realized_delta': self.ops.measure(realized_delta),
                'full_baseline_norm': self.ops.measure(av)['norm'], 'full_donor_norm': self.ops.measure(bv)['norm'],
                'full_source_delta_norm': self.ops.measure(self.ops.diff(bv, av))['norm'],
                'full_applied_feature_norm': self.ops.measure(out)['norm'],
                'full_realized_delta_norm': self.ops.measure(self.ops.diff(out, av))['norm'],
                'baseline_state': cals.get(key, {}).get('baseline_state'),
                'pre_residual': cals.get(key, {}).get('pre_residual')})
            if k == 0 and not self.ops.equal(out, av):
                raise ValueError('k=0 must be exact a baseline')
            if k == 1 and condition['region'] == 'all' and not self.ops.equal(out, bv):
                raise ValueError('k=1 must exactly reproduce the full source replacement')
        selected_keys = {m['entry'] for m in metrics}
        if any(not self.ops.equal(result[e], aa[e]) for e in result if e not in selected_keys):
            raise ValueError('Unselected entry changed')
        for name in inactive_groups(grid):
            if not (self.ops.equal(aa[name], bb[name]) and self.ops.equal(result[name], aa[name])):
                raise ValueError('Inactive padded group differs between a/b/result')
        if need_source:
            self.geometry_cache[source_key] = describe_delta_geometry(self.ops, source_deltas,
                {m['entry']: m['source_delta'] for m in metrics})
        if need_realized:
            self.geometry_cache[realized_key] = describe_delta_geometry(self.ops, realized_deltas,
                {m['entry']: m['realized_delta'] for m in metrics})
        matched = dict(scale)
        for mode in ('baseline_state', 'pre_residual'):
            ratios = []
            for m in metrics:
                den = m.get(mode)
                ratio = m['realized_delta']['rms'] / den['rms'] if den and den['rms'] > 0 else None
                m['realized_to_' + mode] = ratio
                ratios.append(ratio)
            complete = all(r is not None for r in ratios)
            matched['realized_' + mode + '_group_ratio'] = math.sqrt(sum(r * r for r in ratios)) if complete else None
            matched['realized_' + mode + '_entry_ratio_l1'] = sum(ratios) if complete else None
            if any(v is not None and not math.isfinite(v) for v in (
                    matched['realized_' + mode + '_group_ratio'], matched['realized_' + mode + '_entry_ratio_l1'])):
                raise ValueError('Nonfinite realized relative dose')
        if scale['status'] == 'requested_match':
            suffix = '_group_ratio' if scale['dose_metric'] == 'rss' else '_entry_ratio_l1'
            actual = matched['realized_' + condition['dose_mode'] + suffix]
            target = 0. if condition['donor_assignment'] == 'a' else scale['resolved_target']
            if actual is None or not math.isfinite(actual):
                raise ValueError('Nonfinite realized relative dose')
            rel_error = abs(actual - target) / target if target else abs(actual)
            ok = rel_error <= condition['match_tolerance']
            matched.update(actual_target=target, relative_error=rel_error, realized_precision_verified=ok,
                reachability='matched' if ok else 'unmatched_realized_precision',
                status='matched' if ok else 'unmatched_realized_precision')
            if not ok:
                raise UnmatchedDose({**matched, 'reason': 'realized_dose_outside_tolerance'})
        metadata = {'schema': design.DESIGN_SCHEMA, 'family': self.family, 'condition': condition['name'],
            'calibration_id': condition['calibration_id'], 'visual_signature': signature,
            'changed_entry': condition['changed_entry'], 'donor_assignment': condition['donor_assignment'],
            'dose_mode': condition['dose_mode'], 'dose_value': condition['dose_value'], 'effective_k': k,
            'target_policy': condition.get('target_policy', 'fixed'), 'dose_metric': condition.get('dose_metric', 'rss'),
            'context_signature': design.context_signature(row, condition),
            'model_revision': calibration.get('model_revision') if calibration else self.model_revision,
            'anchor_query_sha256': calibration.get('anchor_query_sha256') if calibration else None,
            **{field: matched.get(field) for field in ('source_calibration_id', 'target_calibration_id',
                'source_anchor_query_id', 'target_anchor_query_id', 'source_anchor_query_sha256', 'target_anchor_query_sha256',
                'source_anchor_prompt_sha256', 'target_anchor_prompt_sha256', 'source_visual_signature', 'target_visual_signature',
                'source_context_signature', 'target_context_signature', 'source_model_revision', 'target_model_revision',
                'native_self_by_metric', 'target_other_native_by_metric', 'target_provenance', 'resolved_target',
                'requested_k', 'reachability')},
            'realized_baseline_state_by_metric': {
                'rss': matched['realized_baseline_state_group_ratio'],
                'l1': matched['realized_baseline_state_entry_ratio_l1']},
            'entry_count': len(metrics), 'entries': metrics, 'matching': matched,
            'inactive_entries': inactive_records(condition['changed_entry'], grid, base_adapter.start, base_adapter.interval),
            'native_grid': list(grid), 'configured_later_groups': CONFIGURED_LATER,
            'delta_geometry': {'source': self.geometry_cache[source_key], 'realized': self.geometry_cache[realized_key]},
            'anchor_query_id': calibration.get('anchor_query_id') if calibration else None,
            'anchor_prompt_sha256': calibration.get('anchor_prompt_sha256') if calibration else None,
            'source_cache_immutable': True, 'k0_exact_a': k == 0,
            'k1_exact_v10_full': k == 1 and condition['region'] == 'all'}
        visual = DeepStackVisual(result['main'], [result[f'later{i}'] for i in range(CONFIGURED_LATER)],
                                 a.image_size, a.grid, metadata)
        validate_visual(visual)
        if self.natural:
            if k in (0., 1.):
                reference = a if k == 0 else b
                current, expected = entry_tensors(visual), entry_tensors(reference)
                if set(current) != set(expected) or any(not self.ops.equal(current[e], expected[e]) for e in current):
                    raise ValueError('Natural k0/k1 does not reproduce exact local a/b composition')
            metadata.update(natural_fixed_roi=list(row['donor_roi_bbox_xyxy']),
                k0_exact_natural_a=k == 0., k1_exact_natural_b=k == 1.,
                roi_semantics='fixed donor coordinates across both queries; no natural spatial-rebinding claim')
        return DoseVisual(visual, metadata)

    @contextmanager
    def use(self, visual, inputs, *, audit=False):
        if not isinstance(visual, DoseVisual):
            with self.base_adapter.use(visual, inputs, audit=audit) as state:
                yield state
            return
        probe = DeepStackSiteProbe(self, visual, inputs)
        with probe, self.base_adapter.use(visual.visual, inputs, audit=audit) as state:
            state['dose_observations'] = probe.rows
            state['dose_metadata'] = visual.metadata
            yield state
        counts = {e['entry']: sum(r['entry'] == e['entry'] for r in probe.rows) for e in visual.metadata['entries']}
        if any(n != state['calls'] for n in counts.values()):
            raise ValueError('Each fresh visual delivery must have one scalar observation per selected entry')
        state['dose_decode_hook_steps_skipped'] = probe.decode_steps_skipped

    def calibrate(self, sources, condition, anchor_row, anchor_inputs, forward_once, *, anchor_prompt_sha256):
        design = self.design
        if anchor_row['query_type'] != 'identity' or anchor_row['query_target'] != 'changed':
            raise ValueError('Calibration requires canonical changed identity anchor')
        if not isinstance(anchor_prompt_sha256, str) or len(anchor_prompt_sha256) != 64:
            raise ValueError('Actual canonical direct prompt SHA256 required')
        key = design.calibration_key(anchor_row, condition)
        signature = design.visual_signature(anchor_row, condition)
        if key in self.cache:
            cached = self.cache[key]
            if (cached['visual_signature'] != signature or cached['anchor_prompt_sha256'] != anchor_prompt_sha256
                    or cached['anchor_query_id'] != anchor_row['id']
                    or cached.get('anchor_query_sha256') != design.anchor_query_sha256(anchor_row)):
                raise ValueError('Calibration cache key collision / changed anchor prompt')
            return cached
        base_c = {**condition, 'dose_mode': 'raw', 'dose_value': 0., 'donor_assignment': 'a'}
        base = self.compose(sources, base_c, anchor_row, inputs=anchor_inputs)
        with self.use(base, anchor_inputs, audit=False) as delivery:
            forward_once()
        if delivery['calls'] != 1:
            raise ValueError('Calibration must be exactly one fresh anchor forward')
        observed = {r['entry']: r for r in delivery['dose_observations']}
        if len(observed) != len(base.metadata['entries']):
            raise ValueError('Incomplete calibration entry observations')
        entries = []
        for e in base.metadata['entries']:
            o = observed[e['entry']]
            if not math.isfinite(o['post_entry_state']['rms']):
                raise ValueError('Invalid baseline post-entry state denominator')
            entries.append({**e, 'baseline_state': o['post_entry_state'], 'pre_residual': o['pre_residual'],
                            'pre_residual_missing_reason': o['pre_residual_missing_reason']})
        result = {'schema': 'dose-calibration-v11', 'intervention_id': anchor_row['intervention_id'],
            'calibration_id': condition['calibration_id'], 'visual_signature': signature, 'family': self.family,
            'changed_entry': condition['changed_entry'], 'context_signature': design.context_signature(anchor_row, condition),
            'anchor_query_sha256': design.anchor_query_sha256(anchor_row), 'model_revision': self.model_revision,
            'anchor_query_id': anchor_row['id'], 'anchor_response_format': 'direct', 'anchor_mapping': 'none',
            'anchor_prompt_sha256': anchor_prompt_sha256, 'entries': entries, 'fresh_forwards': 1,
            'inactive_entries': base.metadata['inactive_entries'], 'native_grid': base.metadata['native_grid'],
            'source_delta_geometry': base.metadata['delta_geometry']['source'],
            'baseline_state_definition': 'a-side post-entry residual state on exactly selected visual slots of model.model.layers inputs',
            'pre_residual_definition': 'output of the decoder before the native additive injection; main absent (replacement)',
            'inactive_group_definition': 'official zero-padded groups excluded from entries; contribute exactly zero to RSS/L1',
            'status': 'measured', 'shared_across_queries_and_readouts': True}
        result['native_baseline_state_by_metric'] = {m: v for m, v in native_dose(result).items() if m in ('rss', 'l1')}
        self.cache[key] = result
        return result
