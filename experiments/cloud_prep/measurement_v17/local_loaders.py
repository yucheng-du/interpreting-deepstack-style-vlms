"""Pinned MPS loaders.

Import and snapshot inspection are tensor-free. The Granite adapter, remote
model source, and visual entry semantics are the accepted v15 implementations;
only device placement and the local execution metadata differ.
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

from cloud_prep.measurement_v8.ibm_adapter import SOURCE_SHA256
from cloud_prep.measurement_v8.model_registry import get_model


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inspect_granite_snapshot(snapshot_path):
    """Check fixed identity, reviewed source and shard presence without torch."""
    spec = get_model('ibm4b')
    path = Path(snapshot_path).absolute()
    if path.name != spec['revision']:
        raise ValueError('Granite snapshot directory must name its fixed revision')
    if not path.is_dir():
        raise ValueError('Granite snapshot is missing')
    actual = {name: _sha256(path / name) for name in SOURCE_SHA256}
    if actual != SOURCE_SHA256:
        raise ValueError('Granite remote code differs from the reviewed fixed revision')
    index = json.loads((path / 'model.safetensors.index.json').read_text())
    shards = sorted(set(index['weight_map'].values()))
    missing = [name for name in shards if not (path / name).is_file()]
    if missing:
        raise ValueError('Incomplete Granite checkpoint: ' + ', '.join(missing))
    if index['metadata']['total_size'] != spec['checkpoint_weight_bytes']:
        raise ValueError('Granite checkpoint byte count differs from fixed registry')
    config = json.loads((path / 'config.json').read_text())
    if config['model_type'] != 'granite4_vision':
        raise ValueError('Wrong Granite architecture')
    indexed = {}
    parameter_count = payload_bytes = 0
    for name in shards:
        with (path / name).open('rb') as handle:
            header_bytes = struct.unpack('<Q', handle.read(8))[0]
            if header_bytes > 16 * 1024 * 1024:
                raise ValueError('Unexpectedly large safetensors header')
            header = json.loads(handle.read(header_bytes))
        ends = []
        for key, value in header.items():
            if key == '__metadata__':
                continue
            if key in indexed or value['dtype'] != 'BF16':
                raise ValueError('Duplicate or non-BF16 Granite checkpoint tensor')
            indexed[key] = name
            count = math.prod(value['shape'])
            start, end = value['data_offsets']
            if end - start != 2 * count:
                raise ValueError('Granite tensor byte length differs from BF16 shape')
            parameter_count += count
            payload_bytes += end - start
            ends.append(end)
        if (path / name).stat().st_size != 8 + header_bytes + max(ends):
            raise ValueError('Granite shard length differs from safetensors header')
    if indexed != index['weight_map'] or parameter_count != spec['total_parameters']:
        raise ValueError('Granite checkpoint headers differ from fixed parameter contract')
    return dict(model=spec['model'], revision=spec['revision'],
        snapshot_path=str(path), remote_code_sha256=actual,
        index_recorded_weight_bytes=index['metadata']['total_size'],
        checkpoint_weight_bytes=payload_bytes,
        expected_parameters=spec['total_parameters'],
        safetensors_header_parameter_count=parameter_count,
        all_index_names_and_file_lengths_match=True,
        shard_files=[dict(name=name, file_bytes=(path / name).stat().st_size) for name in shards])


def _jsonable(value):
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (set, frozenset)):
        return [_jsonable(item) for item in sorted(value, key=str)]
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def load_granite_mps(snapshot_path, *, dtype='bfloat16'):
    """Return ``(GraniteAdapter, 'ibm', metadata)`` for a sequential local pilot.

    It performs no download, preserves the fixed remote code and retains the
    adapter's processor, geometry and native-delivery checks. Runtime self-patch
    and intervention audits remain the runner's responsibility.
    """
    if dtype != 'bfloat16':
        raise ValueError('The Granite local bridge uses explicit BF16 precision')
    checked = inspect_granite_snapshot(snapshot_path)
    if importlib.metadata.version('transformers') != '5.8.0':
        raise RuntimeError('Granite accepted remote code requires transformers==5.8.0')
    # Sequential loading avoids the asynchronous reload peak observed in Qwen8.
    os.environ['HF_DEACTIVATE_ASYNC_LOAD'] = '1'
    from cloud_prep.measurement_v10.adapter import GraniteAdapter
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    if not torch.backends.mps.is_available():
        raise RuntimeError('MPS is unavailable')
    path = checked['snapshot_path']
    processor = AutoProcessor.from_pretrained(path, trust_remote_code=True,
        local_files_only=True, use_fast=False)
    model, loading = AutoModelForImageTextToText.from_pretrained(path,
        trust_remote_code=True, local_files_only=True, dtype=torch.bfloat16,
        device_map={'': 'mps'}, attn_implementation='sdpa',
        output_loading_info=True)
    failures = {key: value for key, value in loading.items()
        if key in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs') and value}
    if failures:
        raise RuntimeError('Incomplete Granite checkpoint load: ' + str(failures))
    if _sha256(inspect.getfile(type(model))) != SOURCE_SHA256['modeling.py']:
        raise ValueError('Loaded Granite model source differs from reviewed implementation')
    if _sha256(inspect.getfile(type(processor))) != SOURCE_SHA256['processing.py']:
        raise ValueError('Loaded Granite processor source differs from reviewed implementation')
    count = sum(parameter.numel() for parameter in model.parameters())
    if count != checked['expected_parameters']:
        raise ValueError(f'Granite parameter count {count} differs from fixed checkpoint')
    model.eval().requires_grad_(False)
    meta = dict(**checked, parameter_count=count, device='mps', dtype=dtype,
        attn_implementation='sdpa', native_forward_preserved=True,
        local_loader_sha256=_sha256(__file__),
        role='local feasibility and intervention pilot; not pooled with formal CUDA results',
        loading_info=_jsonable(loading),
        libraries={name: importlib.metadata.version(name)
                   for name in ('torch', 'transformers', 'Pillow', 'accelerate', 'peft')},
        loaded_model_source=inspect.getfile(type(model)),
        model_smoke_status='pending_runner_self_patch_and_geometry_audit')
    adapter = GraniteAdapter(model, processor, device='mps', metadata=meta)
    return adapter, 'ibm', adapter.metadata
