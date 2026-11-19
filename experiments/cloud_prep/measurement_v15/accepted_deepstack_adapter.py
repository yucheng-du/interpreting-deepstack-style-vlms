"""Native original DeepStack loader and intervention primitives for cloud smoke.

Independent from Qwen/Granite adapters. The full v12 campaign bridge remains a
separate cloud integration task; these primitives implement real native forwards,
source caching, full/local delivery and canonical native-site dose measurement.
Imports are tensor-free. Only load()/prepare()/encode()/forward work uses CUDA.
"""
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from pathlib import Path
import math
import sys
from cloud_prep.measurement_v13.deepstack_geometry import (
    validate_config, best_grid, geometry_record, roi_indices, token_order,
)
from cloud_prep.measurement_v13.model_extensions import DEEPSTACK7B


@dataclass
class DeepStackVisual:
    main: object                 # [576, hidden]
    later: list                  # four [576, hidden], including native zero padding
    image_size: tuple
    grid: tuple
    metadata: dict = field(default_factory=dict)


def validate_visual(visual):
    if tuple(visual.main.shape) != (576, 4096) or len(visual.later) != 4:
        raise ValueError('Expected 576x4096 main and four native later groups')
    for value in visual.later:
        if value.shape != visual.main.shape or value.dtype != visual.main.dtype or value.device != visual.main.device:
            raise ValueError('DeepStack stream shapes/dtypes/devices differ')
    if tuple(visual.grid) != best_grid(visual.image_size):
        raise ValueError('Cached geometry is not the native crop grid')


def load(snapshot, clip_config_root, *, device='cuda', attention='sdpa'):
    """Load every fine-tuned checkpoint tensor, including its vision tower.

    The upstream generic builder delays CLIP construction until after checkpoint
    loading. This subclass constructs CLIP from its *configuration* first so the
    checkpoint's model.vision_tower weights cannot be discarded and replaced by
    standalone CLIP weights. No extra CLIP weight download is needed.
    """
    if sys.platform != 'linux' or not str(device).startswith('cuda'):
        raise RuntimeError('Original DeepStack execution is Linux CUDA only')
    import torch
    import transformers
    from transformers import AutoTokenizer, CLIPImageProcessor, CLIPVisionModel
    from llava.model.language_model.deepstack_llama import DeepstackConfig, DeepstackLlamaForCausalLM
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable')
    config = DeepstackConfig.from_pretrained(str(snapshot), local_files_only=True)
    geometry = validate_config(config.to_dict())
    config.mm_vision_tower = str(Path(clip_config_root).resolve())

    class CheckpointVisionDeepStack(DeepstackLlamaForCausalLM):
        def __init__(self, model_config):
            super().__init__(model_config)
            tower = self.get_vision_tower()
            if tower.is_loaded:
                raise ValueError('Unexpected eager external vision loading')
            tower.image_processor = CLIPImageProcessor.from_pretrained(config.mm_vision_tower, local_files_only=True)
            tower.vision_tower = CLIPVisionModel(tower.cfg_only)
            tower.is_loaded = True

    model, loading = CheckpointVisionDeepStack.from_pretrained(str(snapshot), config=config,
        torch_dtype=torch.bfloat16, device_map={'': device}, low_cpu_mem_usage=True,
        attn_implementation=attention, local_files_only=True, output_loading_info=True)
    failures = {k: v for k, v in loading.items()
                if k in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs') and v}
    if failures:
        raise RuntimeError(f'DeepStack checkpoint load incomplete: {failures}')
    model.eval().requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), use_fast=False, local_files_only=True)
    processor = model.get_vision_tower().image_processor
    count = sum(p.numel() for p in model.parameters())
    if count != DEEPSTACK7B['total_parameters']:
        raise ValueError(f'Loaded count {count} differs from official checkpoint metadata')
    if (model.get_vision_tower().config.image_size, model.get_vision_tower().config.patch_size) != (336, 14):
        raise ValueError('CLIP patch geometry differs from inspected checkpoint')
    from llava.constants import IMAGE_TOKEN_INDEX
    if IMAGE_TOKEN_INDEX != NATIVE_IMAGE_TOKEN_INDEX:
        raise ValueError('Official image placeholder id differs from the inspected source')
    checks = verify_checkpoint_tensors(model, snapshot)
    adapter = DeepStackAdapter(model, tokenizer, processor, device=device)
    adapter.metadata = dict(schema='original-deepstack-native-adapter-v13',
        model=DEEPSTACK7B, actual_transformers=transformers.__version__, actual_torch=torch.__version__,
        parameter_count=count, dtype=str(model.dtype), attention=attention,
        vision_weights='loaded_from_finetuned_checkpoint_not_standalone_CLIP',
        loading_info=loading, geometry=geometry, checkpoint_tensor_checks=checks,
        clip_config_root=config.mm_vision_tower, CUDA_acceptance='smoke_in_progress')
    return adapter


NATIVE_IMAGE_TOKEN_INDEX = -200
CHECKED_TENSORS = (
    'model.vision_tower.vision_tower.vision_model.embeddings.patch_embedding.weight',
    'model.vision_tower.vision_tower.vision_model.embeddings.position_embedding.weight',
    'model.vision_tower.vision_tower.vision_model.pre_layrnorm.weight',
    'model.vision_tower.vision_tower.vision_model.encoder.layers.0.self_attn.q_proj.weight',
    'model.vision_tower.vision_tower.vision_model.encoder.layers.11.mlp.fc1.weight',
    'model.vision_tower.vision_tower.vision_model.encoder.layers.22.self_attn.out_proj.weight',
    'model.vision_tower.vision_tower.vision_model.encoder.layers.23.mlp.fc2.weight',
    'model.vision_tower.vision_tower.vision_model.post_layernorm.weight',
    'model.mm_projector.0.weight', 'model.mm_projector.2.weight',
    'model.layers.0.self_attn.q_proj.weight', 'lm_head.weight')


def verify_checkpoint_tensors(model, snapshot, names=CHECKED_TENSORS):
    """Bitwise-compare loaded tensors with the pinned checkpoint safetensors.

    Also counts every checkpoint-indexed vision-tower tensor against the loaded
    module, so no standalone CLIP weight can stand in for the fine-tuned tower.
    """
    import hashlib
    import json as _json
    import torch
    from safetensors import safe_open
    snapshot = Path(snapshot)
    index = _json.loads((snapshot / 'model.safetensors.index.json').read_text())['weight_map']
    params = dict(model.named_parameters())
    vision_index = sorted(k for k in index if k.startswith('model.vision_tower.'))
    vision_loaded = sorted(k for k in params if k.startswith('model.vision_tower.'))
    if vision_index != vision_loaded:
        raise RuntimeError('Loaded vision-tower parameter names differ from the checkpoint index')
    def digest(tensor):
        return hashlib.sha256(tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
    rows = []
    for name in names:
        with safe_open(str(snapshot / index[name]), framework='pt', device='cpu') as handle:
            stored = handle.get_tensor(name)
        loaded = params[name].detach().to('cpu')
        equal = stored.dtype == loaded.dtype and tuple(stored.shape) == tuple(loaded.shape) and bool(torch.equal(stored, loaded))
        rows.append(dict(name=name, shard=index[name], dtype=str(stored.dtype), shape=list(stored.shape),
            checkpoint_sha256=digest(stored), loaded_sha256=digest(loaded), bitwise_equal=equal))
        if not equal:
            raise RuntimeError(f'Loaded tensor differs from pinned checkpoint: {name}')
    return dict(checked=rows, vision_tensors_in_checkpoint_index=len(vision_index),
        vision_tensors_loaded=len(vision_loaded),
        vision_parameters_loaded=sum(params[k].numel() for k in vision_loaded),
        projector_parameters_loaded=sum(p.numel() for k, p in params.items() if k.startswith('model.mm_projector.')),
        all_bitwise_equal=all(r['bitwise_equal'] for r in rows),
        scope='Selected tensors compared bitwise; full vision-tower name set equals checkpoint index')


class DeepStackAdapter:
    def __init__(self, model, tokenizer, image_processor, *, device='cuda'):
        self.model, self.tokenizer, self.image_processor = model, tokenizer, image_processor
        self.device = device
        self.entry_layers = validate_config(model.config.to_dict())['later_entries']

    def prepare(self, image_path, question, assistant_prefix=''):
        from PIL import Image
        from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
        from llava.conversation import conv_templates
        from llava.mm_utils import process_images, tokenizer_image_token
        with Image.open(image_path) as raw:
            image = raw.convert('RGB')
        try:
            size = image.size
            pixels = process_images([image], self.image_processor, self.model.config)
        finally:
            image.close()
        conv = conv_templates['vicuna_v1'].copy()
        conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + '\n' + question)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt() + assistant_prefix
        ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX,
                                    return_tensors='pt').unsqueeze(0).to(self.device)
        if int((ids == IMAGE_TOKEN_INDEX).sum().item()) != 1:
            raise ValueError('One image placeholder required')
        grid = best_grid(size)
        images = [x.to(device=self.device, dtype=self.model.dtype) for x in pixels]
        if len(images) != 1 or images[0].shape[0] != 1 + grid[0] * grid[1]:
            raise ValueError('Native processor views disagree with CPU geometry')
        if ids.shape[1] + 575 + 32 > self.model.config.tokenizer_model_max_length:
            raise ValueError('Prompt + visual expansion + generation exceeds checkpoint token budget')
        return dict(input_ids=ids, attention_mask=ids.new_ones(ids.shape), images=images,
                    image_sizes=[size], bestgrid_list=[grid]), prompt

    def encode(self, inputs):
        import torch
        with torch.inference_mode():
            main, later = self.model.process_deepstack(inputs['images'], inputs['bestgrid_list'])
        size, grid = tuple(inputs['image_sizes'][0]), tuple(inputs['bestgrid_list'][0])
        result = DeepStackVisual(main[0].detach().clone(),
            [value[0].detach().clone() for value in later], size, grid, geometry_record(size))
        validate_visual(result)
        for index in range(grid[0] * grid[1], 4):
            if bool(result.later[index].count_nonzero().item()):
                raise ValueError('Official padded visual group is not zero')
        return result

    @contextmanager
    def inject(self, visual):
        """Supply cloned native containers for each actual forward/generate call."""
        import torch
        validate_visual(visual)
        saved = self.model.process_deepstack
        def deliver(images, bestgrid_list):
            if list(bestgrid_list[0]) != list(visual.grid):
                raise ValueError('Injected crop grid differs from prepared input')
            return visual.main.clone().unsqueeze(0), torch.stack(
                [value.clone().unsqueeze(0) for value in visual.later], dim=0)
        self.model.process_deepstack = deliver
        try:
            yield
        finally:
            self.model.process_deepstack = saved
            # The official generate passes local streams to the first forward.
            # Failed generation must not leak that state into another condition.
            self.model.kwargs_for_generation = None

    def forward(self, inputs, visual=None):
        import torch
        with torch.inference_mode(), self.inject(visual) if visual is not None else nullcontext():
            return self.model(**inputs, use_cache=False)

    def generate(self, inputs, visual=None, *, max_new_tokens=32):
        import torch
        args = {k: v for k, v in inputs.items() if k != 'input_ids'}
        try:
            with torch.inference_mode(), self.inject(visual) if visual is not None else nullcontext():
                result = self.model.generate(inputs=inputs['input_ids'], **args,
                    do_sample=False, num_beams=1, use_cache=True, max_new_tokens=max_new_tokens,
                    return_dict_in_generate=False, pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id)
        finally:
            self.model.kwargs_for_generation = None
        # This official model generates from inputs_embeds; returned IDs are
        # generated tokens, not prompt IDs. CUDA smoke records the raw IDs too.
        return result, self.tokenizer.batch_decode(result, skip_special_tokens=True)[0]


def _compatible(left, right):
    validate_visual(left); validate_visual(right)
    if (left.image_size, left.grid) != (right.image_size, right.grid):
        raise ValueError('Source and receiver must have identical native geometry')
    if left.main.dtype != right.main.dtype or left.main.device != right.main.device:
        raise ValueError('Source and receiver dtype/device differ')


def compose(main, context, donor, *, role='later', region='roi', bbox=None, permutation='identity'):
    """Local a/b replacement or full main/later source competition.

    Later phase groups retain their native identities. A nominal horizontal
    half-swap permutes each 24x24 destination grid, never swaps phase groups.
    """
    _compatible(main, context); _compatible(main, donor)
    if role not in ('main', 'later') or region not in ('roi', 'all'):
        raise ValueError('Expected main/later and roi/all')
    order = token_order(permutation)
    selected = {}
    def replace(current, other, entry):
        active = entry == 'main' or entry < donor.grid[0] * donor.grid[1]
        ids = (list(range(576)) if region == 'all' and active else
               roi_indices(donor.image_size, bbox, entry, grid=donor.grid, permutation=permutation)
               if region == 'roi' else [])
        selected[str(entry)] = ids
        result = current.clone()
        if ids:
            result[ids] = other[[order[i] for i in ids]]
        return result
    out_main = replace(main.main, donor.main, 'main') if role == 'main' else main.main.clone()
    later = [replace(a, b, i) if role == 'later' else a.clone()
             for i, (a, b) in enumerate(zip(context.later, donor.later))]
    result = DeepStackVisual(out_main, later, main.image_size, main.grid,
        dict(role=role, region=region, selected=selected, permutation=permutation,
             geometry=geometry_record(main.image_size, bbox), source_cache_immutable=True))
    validate_visual(result)
    return result


def interpolate(a, b, role, k):
    """Scalar difference dose over the selected local/full operation, exact at 0/1."""
    _compatible(a, b)
    if role not in ('main', 'later') or not math.isfinite(k) or k < 0:
        raise ValueError('Finite nonnegative k and main/later role required')
    def mix(left, right):
        if k == 0: return left.clone()
        if k == 1: return right.clone()
        return (left.float() + k * (right.float() - left.float())).to(left.dtype)
    return DeepStackVisual(mix(a.main, b.main) if role == 'main' else a.main.clone(),
        [mix(x, y) if role == 'later' else x.clone() for x, y in zip(a.later, b.later)],
        a.image_size, a.grid, dict(b.metadata, effective_k=k))


def canonical_calibration(adapter, inputs, a, b, role):
    """Measure a-side post-entry residuals at native decoder boundaries.

    Only the actual image prefill is measured. Padded groups have no source
    support and are reported as structurally inactive, never as unmatched data.
    Active zero-delta groups contribute zero; a zero whole-group dose cannot be
    used as a denominator for matching and is explicitly unavailable.
    """
    from llava.constants import IMAGE_TOKEN_INDEX
    _compatible(a, b)
    start = (inputs['input_ids'][0] == IMAGE_TOKEN_INDEX).nonzero().flatten().tolist()
    if len(start) != 1:
        raise ValueError('One native image location required')
    entries = [('main', 0, a.main, b.main)] if role == 'main' else [
        (str(i), adapter.entry_layers[i], a.later[i], b.later[i]) for i in range(4)]
    records, handles = {}, []
    selected = b.metadata['selected']
    for entry, layer, left, right in entries:
        indices = selected.get(entry, [])
        if not indices:
            padded = entry != 'main' and int(entry) >= a.grid[0] * a.grid[1]
            records[entry] = dict(status='inactive_zero_padded_group' if padded else 'unavailable_empty_ROI_support',
                entry=entry, delta_rms=0. if padded else None, baseline_rms=None,
                ratio=0. if padded else None)
            continue
        delta_rms = float((right[indices].float() - left[indices].float()).square().mean().sqrt().item())
        def hook(module, args, *, entry=entry, indices=indices, delta_rms=delta_rms, layer=layer):
            hidden = args[0]
            if hidden.shape[1] < start[0] + 576:
                return
            baseline = float(hidden[0, [start[0] + i for i in indices]].detach().float().square().mean().sqrt().item())
            records[entry] = dict(status='measured' if baseline > 0 else 'unavailable_zero_baseline',
                entry=entry, delta_rms=delta_rms, baseline_rms=baseline,
                ratio=delta_rms / baseline if baseline > 0 else None,
                selected_tokens=len(indices), decoder_before=layer)
        handles.append(adapter.model.model.layers[layer].register_forward_pre_hook(hook))
    try:
        adapter.forward(inputs, a)
    finally:
        for handle in handles:
            handle.remove()
    if len(records) != len(entries):
        raise ValueError('Not every active native entry was observed')
    values = [row['ratio'] for row in records.values()]
    valid = all(value is not None and math.isfinite(value) for value in values)
    rss, l1 = (math.hypot(*values), sum(values)) if valid else (None, None)
    return dict(role=role, entries=list(records.values()), rss=rss, l1=l1,
        status='measured' if valid and rss > 0 else 'unavailable_zero_or_invalid_group_dose',
        anchor='same prepared a-side question/visual baseline; post-entry decoder input',
        geometry=geometry_record(a.image_size))
