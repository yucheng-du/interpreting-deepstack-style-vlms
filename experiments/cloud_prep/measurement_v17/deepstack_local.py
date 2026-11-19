"""Local MPS bridge for the pinned original DeepStack implementation.

No accepted or upstream source is patched. The loader follows the accepted
fine-tuned-CLIP construction and tensor checks, with MPS device placement and
an isolated Transformers 4.39.3 / torch 2.8 stack.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import struct
import sys

from cloud_prep.measurement_v13.model_extensions import DEEPSTACK7B

STORAGE = Path(os.environ.get('DS_DATA_ROOT', 'work')).resolve()
HF_ROOT = Path(os.environ.get('HF_HOME', STORAGE / 'huggingface')).resolve()
SOURCE_ROOT = Path(os.environ.get('DEEPSTACK_LOADER_DIR', STORAGE / 'DeepStack-VL'))
SOURCE_TREE = Path(__file__).with_name('loader_tree.json')


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inspect_deepstack_inputs(snapshot_path, clip_config_root, source_root=SOURCE_ROOT):
    """Validate fixed files using headers and Git blob hashes; no tensor work."""
    snapshot, clip, source = map(lambda p: Path(p).absolute(),
                                 (snapshot_path, clip_config_root, source_root))
    for path, revision in ((snapshot, DEEPSTACK7B['revision']),
                           (clip, DEEPSTACK7B['clip_revision'])):
        if path.name != revision:
            raise ValueError('Use the model and vision encoder snapshots at the pinned revisions')
    tree = json.loads(SOURCE_TREE.read_text())
    if tree['sha'] != DEEPSTACK7B['loader_revision']:
        raise ValueError('DeepStack source-tree revision changed')
    checked = 0
    for row in tree['tree']:
        if row['type'] != 'blob' or not row['path'].startswith('llava/') or not row['path'].endswith('.py'):
            continue
        data = (source / row['path']).read_bytes()
        digest = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
        if digest != row['sha']:
            raise ValueError('DeepStack source differs from pinned Git blob: ' + row['path'])
        checked += 1
    for name in ('config.json', 'preprocessor_config.json'):
        if not (clip / name).is_file():
            raise ValueError('Missing fixed CLIP config: ' + name)
    index = json.loads((snapshot / 'model.safetensors.index.json').read_text())
    actual_map, parameter_count, payload_bytes = {}, 0, 0
    for shard in sorted(set(index['weight_map'].values())):
        with (snapshot / shard).open('rb') as handle:
            size = struct.unpack('<Q', handle.read(8))[0]
            if size > 16 * 1024 * 1024:
                raise ValueError('Unexpectedly large safetensors header')
            header = json.loads(handle.read(size))
        ends = []
        for name, value in header.items():
            if name == '__metadata__':
                continue
            if name in actual_map or value['dtype'] != 'BF16':
                raise ValueError('DeepStack duplicate or non-BF16 checkpoint tensor')
            count = math.prod(value['shape'])
            start, end = value['data_offsets']
            if end - start != 2 * count:
                raise ValueError('DeepStack tensor bytes differ from BF16 shape')
            actual_map[name] = shard
            parameter_count += count
            payload_bytes += end - start
            ends.append(end)
        if (snapshot / shard).stat().st_size != 8 + size + max(ends):
            raise ValueError('DeepStack shard length differs from header')
    if (actual_map != index['weight_map'] or parameter_count != DEEPSTACK7B['total_parameters']
            or payload_bytes != DEEPSTACK7B['checkpoint_weight_bytes']):
        raise ValueError('DeepStack checkpoint differs from fixed parameter contract')
    return dict(snapshot_path=str(snapshot), clip_config_root=str(clip),
        upstream_source_root=str(source), upstream_revision=tree['sha'],
        upstream_python_files_verified=checked, upstream_source_modified=False,
        checkpoint_header_parameters=parameter_count, checkpoint_payload_bytes=payload_bytes,
        all_index_names_and_file_lengths_match=True)


def load_deepstack_mps(snapshot_path, clip_config_root=None, *, dtype='bfloat16', source_root=SOURCE_ROOT):
    """Return ``(DeepStackCoreAdapter, 'deepstack', metadata)`` for local pilot.

    The runner retains native
    self-patch, exact corners and fixed-later checks. No weights are downloaded.
    """
    if dtype != 'bfloat16':
        raise ValueError('The original DeepStack local bridge uses BF16')
    if importlib.metadata.version('transformers') != '4.39.3':
        raise RuntimeError('Use the isolated original DeepStack Transformers 4.39.3 environment')
    if clip_config_root is None:
        clip_config_root = (HF_ROOT / 'hub/models--openai--clip-vit-large-patch14-336/snapshots') / DEEPSTACK7B['clip_revision']
    checked = inspect_deepstack_inputs(snapshot_path, clip_config_root, source_root)
    sys.path.insert(0, checked['upstream_source_root'])
    import torch
    from transformers import AutoTokenizer, CLIPImageProcessor, CLIPVisionModel
    from llava.constants import IMAGE_TOKEN_INDEX
    from llava.model.language_model.deepstack_llama import DeepstackConfig, DeepstackLlamaForCausalLM
    from cloud_prep.measurement_v13.deepstack_geometry import validate_config
    from cloud_prep.measurement_v15 import accepted_deepstack_adapter as accepted
    from cloud_prep.measurement_v15.accepted_deepstack_core import DeepStackCoreAdapter
    if not torch.backends.mps.is_available():
        raise RuntimeError('MPS is unavailable')
    if Path(inspect.getfile(DeepstackLlamaForCausalLM)).resolve() != Path(source_root).resolve() / 'llava/model/language_model/deepstack_llama.py':
        raise ValueError('Another llava implementation is already imported')
    config = DeepstackConfig.from_pretrained(checked['snapshot_path'], local_files_only=True)
    geometry = validate_config(config.to_dict())
    config.mm_vision_tower = checked['clip_config_root']

    class CheckpointVisionDeepStack(DeepstackLlamaForCausalLM):
        def __init__(self, model_config):
            super().__init__(model_config)
            tower = self.get_vision_tower()
            if tower.is_loaded:
                raise ValueError('Unexpected eager standalone CLIP loading')
            tower.image_processor = CLIPImageProcessor.from_pretrained(config.mm_vision_tower, local_files_only=True)
            tower.vision_tower = CLIPVisionModel(tower.cfg_only)
            tower.is_loaded = True

    model, loading = CheckpointVisionDeepStack.from_pretrained(checked['snapshot_path'],
        config=config, torch_dtype=torch.bfloat16, device_map={'': 'mps'},
        low_cpu_mem_usage=True, attn_implementation='sdpa', local_files_only=True,
        output_loading_info=True)
    failures = {key: value for key, value in loading.items()
                if key in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs') and value}
    if failures:
        raise RuntimeError('Incomplete DeepStack checkpoint load: ' + str(failures))
    model.eval().requires_grad_(False)
    count = sum(parameter.numel() for parameter in model.parameters())
    if count != DEEPSTACK7B['total_parameters']:
        raise ValueError('DeepStack loaded parameter count differs from fixed checkpoint')
    if (model.get_vision_tower().config.image_size, model.get_vision_tower().config.patch_size) != (336, 14):
        raise ValueError('CLIP patch geometry changed')
    if IMAGE_TOKEN_INDEX != accepted.NATIVE_IMAGE_TOKEN_INDEX:
        raise ValueError('DeepStack image token changed')
    tensor_checks = accepted.verify_checkpoint_tensors(model, checked['snapshot_path'])
    tokenizer = AutoTokenizer.from_pretrained(checked['snapshot_path'], use_fast=False, local_files_only=True)
    base = accepted.DeepStackAdapter(model, tokenizer, model.get_vision_tower().image_processor, device='mps')
    base.metadata = dict(**checked, model=DEEPSTACK7B, parameter_count=count,
        device='mps', dtype=dtype, attention='sdpa', local_loader_sha256=_sha256(__file__),
        accepted_loader_sha256=_sha256(inspect.getfile(accepted)),
        role='local feasibility and intervention pilot; not pooled with formal CUDA results',
        vision_weights='loaded_from_finetuned_checkpoint_not_standalone_CLIP',
        loading_info=loading, geometry=geometry, checkpoint_tensor_checks=tensor_checks,
        libraries={name: importlib.metadata.version(name) for name in
                   ('torch', 'transformers', 'Pillow', 'accelerate', 'sentencepiece')},
        model_smoke_status='pending_runner_self_patch_and_geometry_audit')
    adapter = DeepStackCoreAdapter(base)
    return adapter, 'deepstack', adapter.metadata
