"""Qwen dense/MoE and Cosmos3 understanding-tower adapter, v8.

No torch/transformers import at module import or in geometry/planning helpers.
Execution is CUDA only. Sources remain immutable; the modern HF vision return
object is reconstructed per delivery because HF splits its pooler_output in place.
"""
from __future__ import annotations
from contextlib import contextmanager
from dataclasses import dataclass, field
import hashlib
import importlib.metadata
import inspect
import math
from pathlib import Path
import sys
from typing import Any

from cloud_prep.measurement_v8.model_registry import (
    COSMOS_LOADER_SHA256, QWEN_TRANSFORMERS_VERSION, validate_spec,
)

_MISSING = object()
SCHEMA = 'qwen-cosmos-visual-adapter-v8'
ROI_RULE = 'positive-area intersection of nominal merged-token patch support with donor bounding box; contextualized tokens are not local receptive fields'


def token_geometry(grids, *, merge_size=2):
    if len(grids) != 1 or len(grids[0]) != 3:
        raise ValueError('v8 requires one static image')
    t, h, w = grids[0]
    if any(type(x) is not int or x < 1 for x in (t, h, w, merge_size)) or t != 1:
        raise ValueError('Positive integer static image grid required')
    if h % merge_size or w % merge_size:
        raise ValueError('Grid must divide into merged tokens')
    return h // merge_size, w // merge_size


def spatial_token_indices(grids, permutation='identity', *, merge_size=2):
    hh, ww = token_geometry(grids, merge_size=merge_size)
    if permutation == 'identity':
        return list(range(hh * ww))
    if permutation != 'swap_halves' or ww % 2:
        raise ValueError('Half swap requires even merged width')
    return [r * ww + (c + ww // 2) % ww for r in range(hh) for c in range(ww)]


def roi_token_indices(grids, bbox_xyxy, image_size, *, permutation='identity', merge_size=2):
    """Destination indices after P for donor ROI token support; no model import.

    Pixel-space bbox and (width,height) refer to the original donor canvas.
    Qwen's resize covers the entire canvas; normalized support therefore maps
    exactly to resized patch coordinates without a crop/letterbox assumption.
    Half-swap moves the donor ROI mask together with its feature values.
    """
    hh, ww = token_geometry(grids, merge_size=merge_size)
    if len(bbox_xyxy) != 4 or len(image_size) != 2:
        raise ValueError('Need xyxy ROI and (width,height) canvas')
    x0, y0, x1, y1 = map(float, bbox_xyxy)
    width, height = map(float, image_size)
    if not all(math.isfinite(v) for v in (x0,y0,x1,y1,width,height)) or not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError('ROI must have positive area and lie within donor canvas')
    source = {r*ww+c for r in range(hh) for c in range(ww)
              if c*width/ww < x1 and (c+1)*width/ww > x0
              and r*height/hh < y1 and (r+1)*height/hh > y0}
    order = spatial_token_indices(grids, permutation, merge_size=merge_size)
    return [dest for dest, src in enumerate(order) if src in source]


def condition_sources(condition):
    main = condition['main']
    donor = condition.get('donor_source', condition.get('donor'))
    names = condition.get('ds_sources')
    if names is None:
        names = [donor if donor is not None else main] * 3
    if not isinstance(names, (list, tuple)) or len(names) != 3 or any(not isinstance(x, str) for x in names):
        raise ValueError('Qwen requires three ordered DS source names')
    if donor is not None and any(x != donor for x in names):
        raise ValueError('donor_source and ds_sources disagree')
    if condition.get('entry_group', 'all') not in ('all', 'all_later'):
        raise ValueError('Qwen has only the all-three-later-entries group in v8')
    if condition.get('region', 'all') not in ('all', 'roi', 'complement'):
        raise ValueError('Unknown injection region')
    return main, tuple(names)


def _tensor_backend():
    import torch
    return torch


def _grids(inputs):
    grid = inputs['image_grid_thw']
    if hasattr(grid, 'detach'):
        grid = grid.detach().cpu().tolist()
    return [list(g) for g in grid]


def _canvas(row, donor):
    meta = row.get('source_image_metadata', {}).get(donor)
    if meta is None:
        meta = row.get('image_metadata', [None])[0]
    if meta is None or 'width' not in meta or 'height' not in meta:
        raise ValueError('ROI composition requires donor canvas width/height metadata')
    return meta['width'], meta['height']


@dataclass
class QwenVisual:
    main: Any
    deepstack: list
    grids: list
    output_type: Any = None
    metadata: dict = field(default_factory=dict)

    def __getitem__(self, index):
        return (self.main, self.deepstack)[index]

    def __iter__(self):
        return iter((self.main, self.deepstack))


def normalize_visual_output(output, grids):
    """Normalize the old two-tuple and 5.8 ModelOutput without any tensor work."""
    if hasattr(output, 'pooler_output') and hasattr(output, 'deepstack_features'):
        main, ds, output_type = output.pooler_output, output.deepstack_features, type(output)
    elif isinstance(output, (tuple, list)) and len(output) == 2:
        main, ds = output
        output_type = None
    else:
        raise TypeError('Unsupported vision output; expect main + DeepStack features')
    if not isinstance(ds, (list, tuple)) or len(ds) != 3:
        raise ValueError('Exactly three post-merger DS streams required')
    if isinstance(main, (tuple, list)):
        raise ValueError('Encode raw visual output, not already image-split get_image_features output')
    return QwenVisual(main, list(ds), [list(g) for g in grids], output_type)


def validate_visual(visual):
    hh, ww = token_geometry(visual.grids)
    if visual.main.ndim != 2 or visual.main.shape[0] != hh * ww or len(visual.deepstack) != 3:
        raise ValueError('Visual tensors must be [merged tokens,hidden] with three DS streams')
    for feature in visual.deepstack:
        if feature.shape != visual.main.shape or feature.dtype != visual.main.dtype or feature.device != visual.main.device:
            raise ValueError('Main and DS shapes/dtypes/devices differ')


def materialize_visual_output(visual):
    """Fresh container per call; HF may replace pooler_output with split tensors."""
    if visual.output_type is None:
        return visual.main, list(visual.deepstack)
    return visual.output_type(pooler_output=visual.main, deepstack_features=list(visual.deepstack))


def verify_cosmos_loader(package):
    """Validate the exact official source payload, independent of pip metadata."""
    root = Path(package.__file__).resolve().parent
    actual = {}
    for filename, expected in COSMOS_LOADER_SHA256.items():
        got = hashlib.sha256((root / filename).read_bytes()).hexdigest()
        if got != expected:
            raise RuntimeError(f'Cosmos loader {filename} differs from the fixed official commit')
        actual[filename] = got
    return actual


def _validate_config(config, spec):
    spec = validate_spec(spec)
    expected_type = {'qwen_dense': 'qwen3_vl', 'qwen_moe': 'qwen3_vl_moe', 'cosmos3': 'cosmos3_omni'}[spec['backend']]
    if config.model_type != expected_type:
        raise ValueError(f'Expected {expected_type}, found {config.model_type}')
    vision = config.vision_config
    expected = spec['vision_indexes']
    if (vision.patch_size, vision.spatial_merge_size) != (16, 2) or list(vision.deepstack_visual_indexes) != expected:
        raise ValueError(f'Checkpoint changed its pinned patch16/merge2 vision contract: expected indexes {expected}')


def load_backend(spec, snapshot, *, device='cuda', attn_implementation='sdpa', min_pixels=4096, max_pixels=262144):
    """Instantiate a pinned full-BF16 tower on an explicitly CUDA device.

    ``snapshot`` is a complete local pinned snapshot resolved by the runner.
    The CPU CLI can import this file, but cannot accidentally load a model.
    """
    fixed = validate_spec(spec)
    if fixed['family'] != 'qwen':
        raise ValueError('Use GraniteAdapter for IBM')
    if sys.platform != 'linux' or not str(device).startswith('cuda'):
        raise RuntimeError('v8 model/tensor execution is Linux CUDA only')
    if importlib.metadata.version('transformers') != QWEN_TRANSFORMERS_VERSION:
        raise RuntimeError(f'Use transformers=={QWEN_TRANSFORMERS_VERSION}; version changes require a bridge')
    import torch
    import transformers
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable')
    if min_pixels < 1 or max_pixels < min_pixels:
        raise ValueError('Ordered positive image pixel limits required')
    loader_hashes = None
    if fixed['backend'] == 'cosmos3':
        import transformers_cosmos3
        loader_hashes = verify_cosmos_loader(transformers_cosmos3)
        model_cls = transformers_cosmos3.Cosmos3ForConditionalGeneration
    else:
        name = 'Qwen3VLMoeForConditionalGeneration' if fixed['backend'] == 'qwen_moe' else 'Qwen3VLForConditionalGeneration'
        model_cls = getattr(transformers, name)
    model, loading_info = model_cls.from_pretrained(str(snapshot), local_files_only=True,
        dtype=torch.bfloat16, device_map={'': str(device)}, attn_implementation=attn_implementation,
        output_loading_info=True)
    model.eval()
    # Intentional non-understanding Cosmos parameters are filtered by the official
    # class ignore list. Any remaining missing/unexpected/mismatched keys fail.
    failures = {key: value for key, value in loading_info.items()
                if key in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs') and value}
    if failures:
        raise RuntimeError(f'Checkpoint did not load exactly: {failures}')
    _validate_config(model.config, fixed)
    count = sum(p.numel() for p in model.parameters())
    if count != fixed['total_parameters']:
        raise ValueError(f'Loaded tower parameter count {count} != expected {fixed["total_parameters"]}')
    processor = transformers.AutoProcessor.from_pretrained(str(snapshot), local_files_only=True, use_fast=False)
    ip = processor.image_processor
    ip.min_pixels, ip.max_pixels = min_pixels, max_pixels
    ip.size = {'shortest_edge': min_pixels, 'longest_edge': max_pixels}
    if (ip.patch_size, ip.merge_size) != (16,2):
        raise ValueError('Expected patch16/merge2 image processor')
    model_source = Path(inspect.getfile(type(model)))
    metadata = {'schema': SCHEMA, 'model': fixed, 'parameter_count': count,
                'loaded_understanding_tower_only': fixed['backend']=='cosmos3',
                'dtype': str(model.dtype), 'attn_implementation': attn_implementation,
                'libraries': {n: importlib.metadata.version(n) for n in ('torch','transformers','accelerate','Pillow')},
                'model_source_sha256': hashlib.sha256(model_source.read_bytes()).hexdigest(),
                'cosmos_loader_sha256': loader_hashes,
                # transformers 5.8 LoadStateDictInfo.to_dict() holds sets; store JSON lists.
                'loading_info': {k: sorted(v, key=str) if isinstance(v, (set, frozenset)) else v
                                 for k, v in loading_info.items()},
                'image_processor': ip.to_dict(), 'vision_indexes': list(model.model.visual.deepstack_visual_indexes)}
    return model, processor, metadata


class QwenAdapter:
    """Uniform runner interface; main always comes from condition['main']."""
    def __init__(self, model, processor, *, device=None):
        self.model, self.processor = model, processor
        self.device = device if device is not None else model.device
        self.visual = model.model.visual
        self.merge_size = int(self.visual.spatial_merge_size)
        # load_backend validates the config against the pinned registry; here
        # verify that the instantiated tower actually implements that config.
        config = model.config.vision_config
        expected = list(config.deepstack_visual_indexes)
        if ((config.patch_size, config.spatial_merge_size, self.merge_size) != (16, 2, 2)
                or len(expected) != 3 or len(set(expected)) != 3
                or list(self.visual.deepstack_visual_indexes) != expected):
            raise ValueError('Actual Qwen visual geometry/DS paths differ from the validated config')

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
        if grids != [[1, 16, 24]]:
            raise ValueError('The shared 384x256 panel requires the original patch16 grid; version any resizing change explicitly')
        hh, ww = token_geometry(grids)
        actual = int((inputs['input_ids'] == self.model.config.image_token_id).sum().item())
        if actual != hh*ww:
            raise ValueError('Image token placeholders differ from merged grid count')
        if 'mm_token_type_ids' not in inputs:
            raise ValueError('Transformers 5.8 requires processor mm_token_type_ids')
        if inputs['mm_token_type_ids'].shape != inputs['input_ids'].shape:
            raise ValueError('Multimodal type IDs do not align with input IDs')
        return inputs, prompt

    def encode(self, inputs):
        if getattr(self.visual, '_v8_visual_owner', None) is not None or 'forward' in self.visual.__dict__:
            raise RuntimeError('Source encoding requires unpatched visual.forward')
        output = self.visual(inputs['pixel_values'].to(dtype=self.visual.dtype), grid_thw=inputs['image_grid_thw'], return_dict=True)
        result = normalize_visual_output(output, _grids(inputs))
        validate_visual(result)
        return result

    encode_visual = encode

    def compose(self, sources, condition, row, inputs=None):
        torch = _tensor_backend()
        main_name, donor_names = condition_sources(condition)
        base = sources[main_name]
        validate_visual(base)
        if inputs is not None and _grids(inputs) != base.grids:
            raise ValueError('Prepared input and source geometry differ')
        permutation = condition.get('token_permutation', 'identity')
        order = spatial_token_indices(base.grids, permutation)
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
                roi = set(roi_token_indices(base.grids, bbox, _canvas(row,name), permutation=permutation))
                selected = sorted(roi if region=='roi' else set(range(len(order)))-roi)
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
             'main_preserved':True, 'unselected_later_entries_keep_main':True})
        validate_visual(result)
        return result

    @contextmanager
    def use(self, visual, inputs, *, audit=False):
        """Yield visual-call and injection records for score/fresh/generation checks."""
        from deepstack.intervention import DeepStackIntervention
        validate_visual(visual)
        grids = _grids(inputs)
        if grids != visual.grids:
            raise ValueError('Target inputs differ from cached source geometry')
        module = self.visual
        if getattr(module, '_v8_visual_owner', None) is not None or 'forward' in module.__dict__:
            raise RuntimeError('Nested/foreign visual cache is unsupported')
        saved = module.__dict__.get('forward', _MISSING)
        stats = {'calls':0, 'records':[], 'schema':SCHEMA}
        def forward(hidden_states, grid_thw=None, **kwargs):
            if grid_thw is None or _grids({'image_grid_thw':grid_thw}) != grids:
                raise ValueError('Visual delivery grid differs from source geometry')
            if hidden_states.shape[0] != sum(t*h*w for t,h,w in grids):
                raise ValueError('Visual pixel patch count differs from source geometry')
            stats['calls'] += 1
            return materialize_visual_output(visual)
        module._v8_visual_owner = stats
        module.forward = forward
        try:
            with DeepStackIntervention(self.model, '111', audit=audit) as intervention:
                stats['records'] = intervention.records
                yield stats
        finally:
            if saved is _MISSING:
                delattr(module, 'forward')
            else:
                module.forward = saved
            delattr(module, '_v8_visual_owner')

    use_visual_output = use

    def audit_sources(self, source_inputs, sources, row):
        """Actual patch/rotary ordering and inverse checks, no equivariance demand."""
        import torch
        if not source_inputs or set(source_inputs) != set(sources):
            raise ValueError('Audit needs the complete source input/cache mapping')
        grids = next(iter(sources.values())).grids
        if any(_grids(i) != grids for i in source_inputs.values()) or any(s.grids != grids for s in sources.values()):
            raise ValueError('Source geometry is not matched')
        indices = spatial_token_indices(grids, 'swap_halves')
        first = next(iter(source_inputs.values()))
        patch_ids = torch.tensor([j*4+k for j in indices for k in range(4)], device=first['pixel_values'].device)
        pairs = []
        for name in source_inputs:
            other = name.replace('m_s','m1_s',1) if name.startswith('m_s') else name.replace('d_same_','d_cross_',1) if name.startswith('d_same_') else None
            if other in source_inputs:
                equal = torch.equal(source_inputs[name]['pixel_values'].index_select(0,patch_ids), source_inputs[other]['pixel_values'])
                if not equal:
                    raise RuntimeError(f'Processor does not realize the declared exact half-swap: {name}/{other}')
                pairs.append({'left':name,'right':other,'processor_pixel_permutation_equal':True})
        for source in sources.values():
            ids = torch.tensor(indices, device=source.main.device)
            for feature in source.deepstack:
                if not torch.equal(feature.index_select(0,ids).index_select(0,ids), feature):
                    raise RuntimeError('DeepStack token permutation inverse failed')
        _, h, w = grids[0]
        grid = first['image_grid_thw']
        coords = [(r*2+dr,c*2+dc) for r in range(h//2) for c in range(w//2) for dr in range(2) for dc in range(2)]
        coordinate_ids = torch.tensor(coords, device=grid.device)
        table = self.visual.rotary_pos_emb(max(h,w))
        expected = table[coordinate_ids].flatten(1)
        if not torch.equal(self.visual.rot_pos_emb(grid),expected):
            raise RuntimeError('Actual visual rotary positions disagree with merged row-major ordering')
        result = {'grid_thw':grids,'patch_pair_checks':pairs,'rotary_order_exact':True,'ds_inverse_exact':True,
                  'feature_pixel_equivariance_required':False,'roi_rule':ROI_RULE}
        if row.get('donor_roi_bbox_xyxy') is not None:
            donor = next(iter(sources))
            roi = roi_token_indices(grids,row['donor_roi_bbox_xyxy'],_canvas(row,donor))
            result.update(roi_token_indices=roi, roi_tokens=len(roi), visual_tokens=len(indices))
        return result
