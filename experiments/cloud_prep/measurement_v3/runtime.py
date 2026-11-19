"""Small shared CUDA runtime. Planning is standard-library-only.

Planning uses the standard library. Inference uses one visible GPU on Linux with CUDA.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

from cloud_prep.portable_data import load_records

ROOT = Path(__file__).resolve().parents[2]
SPEC = Path(__file__).with_name('snapshot_qwen4b.json')
MODEL = 'Qwen/Qwen3-VL-4B-Instruct'
REVISION = 'ebb281ec70b05090aa6165b016eac8ec08e71b17'
PARAMETERS = 4437815808
SUPPORTED = {MODEL: (REVISION, PARAMETERS),
             'Qwen/Qwen3-VL-8B-Instruct': ('0c351dd01ed87e9c1b53cbc748cba10e6187ff3b', 8767123696)}


def add_runtime_args(parser: argparse.ArgumentParser):
    parser.add_argument('--campaign-root', type=Path, required=True)
    parser.add_argument('--cache-root', type=Path, required=True)
    parser.add_argument('--image-root', type=Path)
    parser.add_argument('--model-path', type=Path)
    parser.add_argument('--snapshot', type=Path, default=SPEC)
    parser.add_argument('--local-files-only', action='store_true')


def read_snapshot(args):
    spec = json.loads(args.snapshot.read_text())
    if (spec.get('revision'), spec.get('total_parameters')) != SUPPORTED.get(spec.get('model')):
        raise ValueError('Use a supported pinned Qwen3-VL checkpoint with its full parameter count')
    if spec['model'] == MODEL and not spec.get('weight_sha256'):
        raise ValueError('Expected checkpoint weight checksums are required')
    return spec


def plan_runtime(args):
    for name in ('campaign_root', 'cache_root'):
        path = getattr(args, name)
        if not path.is_absolute():
            raise ValueError(f'--{name.replace("_", "-")} must be absolute')
    root, cache = args.campaign_root.resolve(), args.cache_root.resolve()
    output = getattr(args, 'output', None)
    if output is not None and (not output.is_absolute() or output.resolve() == root
                               or not output.resolve().is_relative_to(root)):
        raise ValueError('Fresh output must be a child of the explicit campaign root')
    if args.model_path is not None and not args.model_path.is_absolute():
        raise ValueError('--model-path must be absolute')
    read_snapshot(args)
    return {'campaign_root': str(root), 'cache_root': str(cache),
            'device': 'cuda', 'dtype': 'bfloat16', 'output_head': 'float32 from stored bf16 weights',
            'attention': 'sdpa', 'batch_size': 1,
            'execution_policy': 'Linux CUDA only; one visible GPU per process'}


def prepare_runtime(args):
    info = plan_runtime(args)
    if sys.platform != 'linux':
        raise RuntimeError('This model runner requires Linux and CUDA')
    root, cache = Path(info['campaign_root']), Path(info['cache_root'])
    root.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    temporary = root / 'tmp'
    temporary.mkdir(exist_ok=True)
    os.environ.update(HF_HOME=str(cache), HF_HUB_CACHE=str(cache/'hub'),
                      HUGGINGFACE_HUB_CACHE=str(cache/'hub'), TRANSFORMERS_CACHE=str(cache/'hub'),
                      TORCH_HOME=str(cache/'torch'), TMPDIR=str(temporary),
                      TOKENIZERS_PARALLELISM='false')
    return info


def resolve_snapshot(args):
    """Resolve and verify the same stored BF16 weights; call after prepare_runtime."""
    if sys.platform != 'linux':
        raise RuntimeError('Snapshot download/model preparation is cloud-only in this runner')
    spec = read_snapshot(args)
    if args.model_path is not None:
        model_path = args.model_path.resolve()
    else:
        from huggingface_hub import snapshot_download
        model_path = Path(snapshot_download(
            repo_id=spec['model'], revision=spec['revision'], cache_dir=str(args.cache_root/'hub'),
            local_files_only=args.local_files_only,
            allow_patterns=['*.json', 'tokenizer*', 'vocab*', 'merges*', '*.jinja', '*.safetensors']))
    expected_weights = spec.get('weight_sha256', {})
    if not expected_weights and args.model_path is not None:
        raise ValueError('Without known weight checksums, resolve the pinned Hub revision instead of --model-path')
    weights = expected_weights or {p.name: None for p in model_path.glob('model*.safetensors')}
    if not weights:
        raise ValueError('No checkpoint weight shards found')
    actual_weights = {}
    for name, expected in weights.items():
        digest = hashlib.sha256()
        with (model_path/name).open('rb') as handle:
            for block in iter(lambda: handle.read(8*1024*1024), b''):
                digest.update(block)
        actual_weights[name] = digest.hexdigest()
        if expected is not None and actual_weights[name] != expected:
            raise ValueError(f'Checkpoint weight checksum mismatch: {name}')
    return {**spec, 'path': str(model_path), 'weight_sha256': actual_weights,
            'weight_checksums_verified': bool(expected_weights),
            'weight_identity_source': 'known checksums' if expected_weights else 'pinned Hub revision; actual checksums recorded'}


def validate_cuda(torch):
    if sys.platform != 'linux' or not torch.cuda.is_available():
        raise RuntimeError('An available Linux CUDA GPU is required; no CPU/MPS fallback')
    if torch.cuda.device_count() != 1:
        raise RuntimeError('Expose one GPU to this process using CUDA_VISIBLE_DEVICES')
    if not torch.cuda.is_bf16_supported(including_emulation=False):
        raise RuntimeError('This protocol requires native BF16 CUDA support')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    torch.cuda.reset_peak_memory_stats()
    props = torch.cuda.get_device_properties(0)
    return {'device_name': props.name, 'total_memory_bytes': props.total_memory,
            'cuda_version': torch.version.cuda, 'torch_version': torch.__version__,
            'visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
            'tf32_matmul': False, 'tf32_cudnn': False}


def gpu_lock_path(locks, gpu):
    """Exclusive per-process GPU lock; slot 0 keeps the original gpu-N.lock.

    DS_GPU_SLOT=k>0 names a distinct concurrent lane on the same GPU, assigned
    centrally from measured memory. Each lane still admits only one process.
    """
    slot = os.environ.get('DS_GPU_SLOT', '0')
    if not slot.isdigit() or int(slot) > 7:
        raise ValueError('DS_GPU_SLOT must be an integer lane 0..7')
    return Path(locks)/(f'gpu-{gpu}.lock' if int(slot) == 0 else f'gpu-{gpu}-slot{int(slot)}.lock')


def memory(torch, device='cuda'):
    if device != 'cuda':
        raise ValueError('Cloud measurement runtime has no CPU/MPS model fallback')
    return {'cuda_allocated_bytes': torch.cuda.memory_allocated(),
            'cuda_reserved_bytes': torch.cuda.memory_reserved(),
            'cuda_peak_allocated_bytes': torch.cuda.max_memory_allocated(),
            'cuda_peak_reserved_bytes': torch.cuda.max_memory_reserved()}


def swap_used_mb():
    path = Path('/proc/meminfo')
    if not path.exists():
        return None
    values = {}
    for line in path.read_text().splitlines():
        fields = line.split()
        if fields[0] in ('SwapTotal:', 'SwapFree:'):
            values[fields[0]] = float(fields[1])
    if len(values) != 2:
        return None
    return (values['SwapTotal:'] - values['SwapFree:']) / 1024


def check_swap(initial):
    current = swap_used_mb()
    if initial is not None and current is not None and current > initial + 128:
        raise RuntimeError(f'Host swap grew by {current-initial:.1f} MiB; stopping this attempt')
    return current
