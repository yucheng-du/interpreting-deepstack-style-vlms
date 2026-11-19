"""Explicit cloud-only model execution, with pinned Hub identity and provenance."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import sys


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def plan_runtime(args):
    for name in ('campaign_root', 'cache_root'):
        if not getattr(args, name).is_absolute():
            raise ValueError(f'{name} must be absolute')
    root, cache = args.campaign_root.resolve(), args.cache_root.resolve()
    if args.output is not None and (not args.output.is_absolute() or args.output.resolve() == root
                                   or not args.output.resolve().is_relative_to(root)):
        raise ValueError('Output must be a fresh child of campaign-root')
    return {'campaign_root': str(root), 'cache_root': str(cache), 'device': 'cuda', 'dtype': 'bfloat16',
            'batch_size': 1, 'attention': 'sdpa', 'output_head': 'fp32 arithmetic from stored bf16 weights',
            'execution_policy': 'Linux with one visible CUDA GPU'}


def prepare_runtime(args):
    info = plan_runtime(args)
    if sys.platform != 'linux':
        raise RuntimeError('This model runner requires Linux and CUDA')
    for p in (args.campaign_root, args.cache_root, args.campaign_root / 'tmp'):
        p.mkdir(parents=True, exist_ok=True)
    os.environ.update(HF_HOME=str(args.cache_root), HF_HUB_CACHE=str(args.cache_root / 'hub'),
                      TORCH_HOME=str(args.cache_root / 'torch'), TMPDIR=str(args.campaign_root / 'tmp'),
                      TOKENIZERS_PARALLELISM='false', CUDA_VISIBLE_DEVICES=args.gpu)
    return info


def snapshot_weight_files(path):
    """Validate the fixed snapshot's actual loading shards, including subfolders.

    Cosmos' root index points into transformer/ and vision_encoder/. Other
    safetensors in its repository are not implicitly added to the model index.
    HF cache symlinks into blobs are expected; validate lexical relative paths,
    not resolved-path containment (which would reject legitimate cache links).
    """
    path = Path(path)
    index = path / 'model.safetensors.index.json'
    if index.is_file():
        data = json.loads(index.read_text())
        mapping = data.get('weight_map')
        if not isinstance(mapping, dict) or not mapping:
            raise ValueError('Pinned safetensors index has no nonempty weight_map')
        names = mapping.values()
        if any(not isinstance(name, str) or not name or Path(name).is_absolute()
               or '..' in Path(name).parts or Path(name).suffix != '.safetensors' for name in names):
            raise ValueError('Pinned safetensors index contains an invalid shard path')
        names = sorted(set(names))
    else:
        files = sorted(path.glob('*.safetensors'))
        if len(files) != 1:
            raise ValueError('Pinned snapshot needs a safetensors index or one root safetensors file')
        names = [files[0].name]
    missing = [name for name in names if not (path / name).is_file()]
    if missing:
        raise ValueError('Pinned snapshot is missing indexed safetensors shards: ' + ', '.join(missing))
    return {name: {'bytes': (path / name).stat().st_size} for name in names}


def resolve_snapshot(spec, cache_root, local_files_only=False):
    if sys.platform != 'linux':
        raise RuntimeError('Weight resolution is cloud-only')
    from huggingface_hub import snapshot_download
    path = Path(snapshot_download(repo_id=spec['model'], revision=spec['revision'],
        cache_dir=str(cache_root / 'hub'), local_files_only=local_files_only,
        allow_patterns=['*.json', '*.py', '*.safetensors', '*.model', '*.txt', '*.jinja',
                        'tokenizer*', 'vocab*', 'merges*']))
    weights = snapshot_weight_files(path)
    return {**spec, 'path': str(path), 'weight_identity_source': 'fixed Hub commit',
            'weight_files': weights,
            'config_sha256': sha256(path / 'config.json')}
