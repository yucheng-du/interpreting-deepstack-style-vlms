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
from cloud_prep.measurement_v14.accepted_deepstack_adapter import DeepStackVisual, validate_visual
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

