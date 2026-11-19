"""Qwen visual-similarity CUDA runtime; importing and planning never loads tensors."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys

from cloud_prep.common import sha256

CONFIGS = Path(__file__).with_name('configs')
CHECKPOINTS = {
    'qwen4b': ('Qwen/Qwen3-VL-4B-Instruct', 'ebb281ec70b05090aa6165b016eac8ec08e71b17', 4437815808),
    'qwen8b': ('Qwen/Qwen3-VL-8B-Instruct', '0c351dd01ed87e9c1b53cbc748cba10e6187ff3b', 8767123696),

}
CONFIG_FILES = {'qwen4b': 'qwen3_vl_4b.json', 'qwen8b': 'qwen3_vl_8b.json'}


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def read_spec(path):
    spec = json.loads(Path(path).read_text())
    identity = (spec.get('model'), spec.get('revision'), spec.get('total_parameters'))
    if identity not in CHECKPOINTS.values():
        raise ValueError('Use a pinned Qwen checkpoint from the configuration directory')
    if spec.get('runtime', {}).get('dtype', 'bfloat16') != 'bfloat16':
        raise ValueError('This version is a BF16-body protocol')
    return spec


def spec_path(key):
    return CONFIGS / CONFIG_FILES[key]


def backend_for(spec):
    family = spec['family']
    if family != 'qwen3_vl':
        raise ValueError('This comparison uses the Qwen3-VL backend')
    from cloud_prep.measurement_v3 import qwen_native
    return qwen_native


def guard_paths(campaign_root, cache_root, output=None):
    for path in (campaign_root, cache_root, *([output] if output is not None else [])):
        if not Path(path).is_absolute():
            raise ValueError('Campaign, cache and output paths must be absolute')
    if output is not None:
        root, out = Path(campaign_root).resolve(), Path(output).resolve()
        if out == root or not out.is_relative_to(root):
            raise ValueError('Output must be a child of the campaign root')


def prepare_runtime(campaign_root, cache_root, output=None):
    guard_paths(campaign_root, cache_root, output)
    if sys.platform != 'linux':
        raise RuntimeError('No local model execution: this runner requires Linux CUDA')
    root, cache = Path(campaign_root).resolve(), Path(cache_root).resolve()
    for directory in (root, cache, root/'tmp'):
        directory.mkdir(parents=True, exist_ok=True)
    os.environ.update(HF_HOME=str(cache), HF_HUB_CACHE=str(cache/'hub'),
                      HUGGINGFACE_HUB_CACHE=str(cache/'hub'), TORCH_HOME=str(cache/'torch'),
                      TMPDIR=str(root/'tmp'), TOKENIZERS_PARALLELISM='false')


def resolve_snapshot(spec, cache_root, *, model_path=None, local_files_only=False):
    if sys.platform != 'linux':
        raise RuntimeError('Checkpoint resolution for execution is cloud-only')
    if model_path is None:
        from huggingface_hub import snapshot_download
        path = Path(snapshot_download(repo_id=spec['model'], revision=spec['revision'],
            cache_dir=str(Path(cache_root)/'hub'), local_files_only=local_files_only,
            allow_patterns=['*.json', 'tokenizer*', 'vocab*', 'merges*', '*.jinja', '*.safetensors']))
    else:
        path = Path(model_path)
        if not path.is_absolute() or not spec.get('weight_sha256'):
            raise ValueError('An explicit model path needs absolute path and known weight checksums')
    index = path/'model.safetensors.index.json'
    names = (sorted(set(json.loads(index.read_text())['weight_map'].values())) if index.exists()
             else ['model.safetensors'])
    if not names or any(Path(n).name != n for n in names):
        raise ValueError('Invalid checkpoint shard index')
    weights = {name: sha256(path/name) for name in names}
    expected = spec.get('weight_sha256', {})
    if expected and weights != expected:
        raise ValueError('Checkpoint shard checksums differ from declared weights')
    files = {p.name:sha256(p) for p in path.iterdir()
             if p.is_file() and p.suffix in ('.json','.jinja')}
    return path, {'model':spec['model'], 'revision':spec['revision'],
                  'total_parameters':spec['total_parameters'], 'weight_sha256':weights,
                  'metadata_sha256':files, 'expected_weights_verified':bool(expected),
                  'identity_source':'known weights' if expected else 'pinned Hub revision and recorded checksums'}
