"""One CUDA model per process. Plan/processor checks do not load weights."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time

from cloud_prep.common import (REPO, check_local_write, parse_generation,
                               read_config, select_rows, sha256, write_json)
from cloud_prep.generation import prepare_native_generation

METADATA_PATTERNS = ['*.json', '*.jinja', '*.txt', '*.model', '*.tiktoken']


def now():
    return datetime.now(timezone.utc).isoformat()


def set_cache(root, cache):
    root, cache = check_local_write(root), check_local_write(cache, cache=True)
    temporary = root / 'tmp'
    for directory in (root, cache, temporary):
        directory.mkdir(parents=True, exist_ok=True)
    os.environ.update(HF_HOME=str(cache), HF_HUB_CACHE=str(cache / 'hub'),
                      HF_DATASETS_CACHE=str(root / 'datasets'), TORCH_HOME=str(cache / 'torch'),
                      TMPDIR=str(temporary), TEMP=str(temporary), TMP=str(temporary))


def snapshot_for(config, cache, metadata_only, local_only):
    from huggingface_hub import snapshot_download
    patterns = list(METADATA_PATTERNS)
    if not metadata_only:
        patterns.append('*.safetensors')
    return Path(snapshot_download(config['model']['id'], revision=config['model']['revision'],
                cache_dir=str(cache / 'hub'), allow_patterns=patterns,
                local_files_only=local_only, max_workers=2))


def provenance(config, args, rows):
    files = [*sorted((REPO / 'cloud_prep').glob('*.py')), *sorted((REPO / 'deepstack').glob('*.py')),
             REPO / 'scripts/prepare_readout_controls.py']
    return {'state': 'prepared', 'started_at': now(), 'phase': 'exploratory_cuda_pilot',
            'model_execution_validated': False, 'resolved_config': config, 'change_note': args.note,
            'manifest': str(args.manifest), 'manifest_sha256': sha256(args.manifest),
            'image_root': str(args.image_root) if args.image_root else None,
            'selected_ids': [row['id'] for row in rows], 'selected_count': len(rows),
            'source_sha256': {str(p.relative_to(REPO)): sha256(p) for p in files},
            'python': sys.version, 'platform': platform.platform(),
            'score_semantics': 'Complete legal-label conditional log probability; no length normalization; fresh uncached forward per scorer call.',
            'generation_semantics': 'Native model generation, no intervention hooks or external vision cache. Distinct from direct-label scoring.'}


def append(path, value):
    with path.open('a') as f:
        f.write(json.dumps(value, ensure_ascii=False, default=str) + '\n')


def memory(torch, device):
    torch.cuda.synchronize(device)
    return {'allocated_bytes': torch.cuda.memory_allocated(device),
            'reserved_bytes': torch.cuda.memory_reserved(device),
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(device),
            'peak_reserved_bytes': torch.cuda.max_memory_reserved(device)}


def tensor_identity(torch, inputs):
    result = {}
    for key, tensor in inputs.items():
        data = tensor.detach().contiguous().cpu()
        result[key] = {'shape': list(data.shape), 'dtype': str(data.dtype),
                       'sha256': hashlib.sha256(data.view(torch.uint8).numpy().tobytes()).hexdigest()}
    return result


def check_scores(reference, candidate, atol):
    if set(reference) != set(candidate):
        raise RuntimeError('Validation changed the legal-label set')
    delta = max(abs(reference[key] - candidate[key]) for key in reference)
    if delta > atol:
        raise RuntimeError(f'Native/full/self-patch score difference {delta} exceeds {atol}')
    return delta


def run_cuda(config, snapshot, rows, output, meta):
    import torch
    import transformers
    from cloud_prep.adapters import build_adapter
    from cloud_prep.processors import make_processor, prepare_inputs
    from deepstack.scoring import score_options, tokenize_options
    from scripts.prepare_readout_controls import generation_stop

    device = config['runtime']['device']
    if not device.startswith('cuda') or not torch.cuda.is_available():
        raise RuntimeError('This prepared runner requires a CUDA device')
    dtype = getattr(torch, config['runtime']['dtype'])
    torch.manual_seed(config['runtime']['seed'])
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError('Selected GPU lacks BF16 support; choose a supported dtype explicitly')
    torch.cuda.reset_peak_memory_stats(device)
    model_class = getattr(transformers, config['model']['class'])
    meta['libraries'] = {name: importlib.metadata.version(name) for name in
                         ('torch', 'transformers', 'huggingface_hub', 'Pillow', 'accelerate')}
    source_path = Path(inspect.getfile(model_class))
    meta['model_implementation'] = {'file': str(source_path), 'sha256': sha256(source_path)}
    meta['gpu'] = {'name': torch.cuda.get_device_name(device),
                   'total_memory_bytes': torch.cuda.get_device_properties(device).total_memory,
                   'torch_cuda': torch.version.cuda}
    meta['state'] = 'loading'
    write_json(output / 'metadata.json', meta)
    shutil.copy2(source_path, output / 'source' / source_path.name)
    started = time.monotonic()
    model = model_class.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False,
                torch_dtype=dtype, device_map=device,
                attn_implementation=config['runtime']['attn_implementation']).eval()
    processor = make_processor(snapshot, config)
    adapter = build_adapter(model, config)
    meta.update(load_seconds=time.monotonic() - started, state='running',
                actual_parameter_count=sum(p.numel() for p in model.parameters()),
                actual_parameter_dtypes=sorted({str(p.dtype) for p in model.parameters()}),
                actual_parameter_devices=sorted({str(p.device) for p in model.parameters()}),
                architecture=adapter.architecture_summary,
                processor=processor.to_dict(), image_processor=processor.image_processor.to_dict(),
                memory_after_load=memory(torch, device))
    write_json(output / 'metadata.json', meta)
    native_generation = None
    resolved = None
    if config['generation']['enabled']:
        native_generation = prepare_native_generation(model, config['generation'])
        resolved = native_generation.resolved_config
        gen_kwargs = native_generation.explicit_kwargs
        meta['generation_config'] = resolved.to_dict()
        meta['generation_mode'] = native_generation.mode
        meta['generation_explicit_kwargs'] = gen_kwargs
        meta['generation_call_protocol'] = 'explicit_kwargs_only_v2'
        write_json(output / 'metadata.json', meta)
    scores_written = generations_written = 0
    with torch.inference_mode():
        for index, row in enumerate(rows):
            meta['current_item'] = row['id']
            write_json(output / 'metadata.json', meta)
            text, inputs, info = prepare_inputs(processor, row, config, tensor_type='pt')
            identity = tensor_identity(torch, inputs)
            inputs = {key: value.to(device=device, dtype=dtype if value.is_floating_point() else value.dtype)
                      for key, value in inputs.items()}
            info['processor_tensor_identity'] = identity
            info['model_tensor_identity'] = tensor_identity(torch, inputs)
            info['image_files'] = [{'path': p, 'sha256': sha256(p)} for p in row['images']]
            append(output / 'inputs.jsonl', {'id': row['id'], 'source_record': row, **info})
            options = tokenize_options(processor.tokenizer, row['option_labels'], prompt_text=text)
            audit = index < config['validation']['audit_items']
            native = None
            if audit:
                native = score_options(model, inputs, options).scores
                if config['intervention']['kind'] != 'native':
                    with adapter.context('111', inputs, audit=True) as active:
                        full = score_options(model, inputs, options)
                    full_records = active.records
                    with adapter.context('000', inputs, audit=True, self_patch=True) as active:
                        restored = score_options(model, inputs, options)
                    restored_records = active.records
                    for expected, records in ((full.forward_count, full_records), (restored.forward_count, restored_records)):
                        if len(records) != expected or not all(r.get('completed') for r in records):
                            raise RuntimeError('Incomplete full/self-patch intervention audit')
                    append(output / 'semantic_checks.jsonl', {
                        'id': row['id'], 'native_vs_full_max_delta': check_scores(native, full.scores, config['validation']['score_atol']),
                        'native_vs_self_patch_max_delta': check_scores(native, restored.scores, config['validation']['score_atol']),
                        'full_records': full_records, 'self_patch_records': restored_records})
                else:
                    append(output / 'semantic_checks.jsonl', {'id': row['id'], 'kind': 'native',
                        'self_patch': 'not_applicable_no_PLE_or_DeepStack', 'finite_native_scores': native})
            for condition in adapter.conditions:
                started = time.monotonic()
                with adapter.context(condition, inputs, audit=audit) as active:
                    result = score_options(model, inputs, options, correct_label=row['answer_label'])
                records = getattr(active, 'records', [])
                if config['intervention']['kind'] != 'native' and (
                        len(records) != result.forward_count or not all(r.get('completed') for r in records)):
                    raise RuntimeError('Intervention did not cover every scoring forward')
                native_repeat_delta = None
                if config['intervention']['kind'] == 'native' and native is not None:
                    native_repeat_delta = check_scores(native, result.scores, config['validation']['score_atol'])
                append(output / 'scores.jsonl', {'id': row['id'], 'task': row['task'], 'condition': condition,
                    'answer_label': row['answer_label'], 'correct': result.prediction == row['answer_label'],
                    **result.to_dict(), 'intervention_records': records,
                    'native_repeat_max_delta': native_repeat_delta,
                    'elapsed_seconds': time.monotonic() - started, 'memory': memory(torch, device)})
                scores_written += 1
            if resolved is not None:
                started = time.monotonic()
                generated = native_generation.generate(model, inputs)
                ids = generated[0, len(info['input_ids']):].detach().cpu().tolist()
                eos = resolved.eos_token_id
                eos_ids = [eos] if isinstance(eos, int) else list(eos or [])
                stop = generation_stop(ids, eos_ids, gen_kwargs['max_new_tokens'])
                decoded = processor.tokenizer.decode(ids, skip_special_tokens=True)
                parsed = parse_generation(decoded, row['option_labels'], config['parsing']['strategy'])
                scorable = parsed['status'] == 'parsed' and not stop['truncated']
                if config['parsing'].get('require_eos', True):
                    scorable = scorable and stop['eos_reached']
                append(output / 'generations.jsonl', {'id': row['id'], 'task': row['task'],
                    'condition': 'native', 'generated_token_ids': ids, 'text': decoded,
                    'text_with_special_tokens': processor.tokenizer.decode(ids, skip_special_tokens=False),
                    'eos_token_ids': eos_ids, **stop, 'parse': parsed, 'scorable': scorable,
                    'prediction': parsed['label'] if scorable else None,
                    'correct': bool(scorable and parsed['label'] == row['answer_label']),
                    'elapsed_seconds': time.monotonic() - started, 'memory': memory(torch, device)})
                generations_written += 1
                del generated
            print(json.dumps({'event': 'item_complete', 'id': row['id'], 'index': index + 1, 'total': len(rows)}), flush=True)
            del inputs
    expected_scores = len(rows) * len(adapter.conditions)
    expected_generations = len(rows) if resolved is not None else 0
    if (scores_written, generations_written) != (expected_scores, expected_generations):
        raise RuntimeError('Output coverage differs from the explicit selection')
    meta.update(state='complete', finished_at=now(), model_execution_validated=True,
                score_rows=scores_written, generation_rows=generations_written,
                memory_at_end=memory(torch, device))
    write_json(output / 'metadata.json', meta)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--image-root', type=Path)
    parser.add_argument('--root', type=Path, required=True, help='Explicit workspace for caches/temp and run artifacts')
    parser.add_argument('--cache-root', type=Path, required=True, help='Explicit shared Hugging Face cache root')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--mode', choices=('plan', 'processor-check', 'run'), default='plan')
    parser.add_argument('--local-files-only', action='store_true')
    parser.add_argument('--set', action='append', default=[], metavar='DOTTED.KEY=JSON')
    parser.add_argument('--note', default='', help='Exploratory change/fix rationale saved with the actual configuration')
    args = parser.parse_args(argv)
    config = read_config(args.config, args.set)
    from cloud_prep.portable_data import load_records
    rows = select_rows(load_records(args.manifest, image_root=args.image_root), config['selection'])
    meta = provenance(config, args, rows)
    if args.mode == 'plan':
        print(json.dumps(meta, indent=2, ensure_ascii=False))
        return
    if args.mode == 'run' and sys.platform != 'linux':
        raise RuntimeError('Model inference requires Linux and CUDA')
    if not args.output:
        parser.error('--output is required for processor-check/run')
    root, cache = check_local_write(args.root), check_local_write(args.cache_root, cache=True)
    output = check_local_write(args.output)
    if not output.is_relative_to(root):
        parser.error('--output must be below --root')
    set_cache(root, cache)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'source').mkdir()
    for relative in meta['source_sha256']:
        target = output / 'source' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / relative, target)
    write_json(output / 'metadata.json', meta)
    try:
        from cloud_prep.portable_data import validate_manifest
        meta['data_validation'] = validate_manifest(args.manifest, image_root=args.image_root)
        snapshot = snapshot_for(config, cache, args.mode == 'processor-check', args.local_files_only)
        meta['snapshot'] = str(snapshot)
        meta['snapshot_metadata_sha256'] = {p.name: sha256(p) for p in snapshot.iterdir()
                                           if p.is_file() and p.suffix in ('.json', '.jinja')}
        write_json(output / 'metadata.json', meta)
        if args.mode == 'processor-check':
            from cloud_prep.processors import make_processor, prepare_inputs
            processor = make_processor(snapshot, config)
            for row in rows:
                _, _, info = prepare_inputs(processor, row, config, tensor_type='np')
                append(output / 'processor_checks.jsonl', {'id': row['id'], **info})
            meta.update(state='processor_check_complete', finished_at=now(),
                        processor_class=type(processor).__name__, torch_imported='torch' in sys.modules,
                        model_execution_validated=False)
            write_json(output / 'metadata.json', meta)
        else:
            run_cuda(config, snapshot, rows, output, meta)
    except BaseException as error:
        meta.update(state='failed', failed_at=now(), error=repr(error))
        write_json(output / 'metadata.json', meta)
        raise


if __name__ == '__main__':
    main()
