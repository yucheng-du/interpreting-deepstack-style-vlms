#!/usr/bin/env python3
"""Run one batch of Qwen4B readout controls."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import os as _path_os
DATA_ROOT = Path(_path_os.environ.get("DS_DATA_ROOT", "work")).expanduser().resolve()
import platform
import sys
import time
import traceback

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from scripts.prepare_readout_controls import (ROOT, ROOT_V2, SCHEMA, SCHEMA_V2, PIXELS, VARIANTS, PARSER_VERSION, campaign_schema,
                                     canonical_digest, digest, generation_stop,
                                     parse_generated_label, read_controls)
from scripts.run_pilot import cached_vision, make_processor, memory, swap_used_mb

REVISION = 'ebb281ec70b05090aa6165b016eac8ec08e71b17'
PARAMETERS = 4437815808
EXECUTION_FILES = ('scripts/run_readout_controls.py', 'scripts/prepare_readout_controls.py',
                   'scripts/run_pilot.py', 'deepstack/data.py',
                   'deepstack/intervention.py', 'deepstack/scoring.py')
GREEDY_KWARGS = {'do_sample': False, 'num_beams': 1, 'num_return_sequences': 1, 'use_cache': True}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--campaign-root', type=Path, default=ROOT,
                        help='Explicit v1/v2 campaign; generation limits are version-specific')
    parser.add_argument('--snapshot', type=Path, default=(Path(__file__).resolve().parents[1] / 'cloud_prep/measurement_v3/snapshot_qwen4b.json'))
    parser.add_argument('--offset', type=int, default=0)
    parser.add_argument('--limit', type=int, default=8)
    parser.add_argument('--budgets', nargs='+', choices=PIXELS, default=list(PIXELS))
    parser.add_argument('--variants', nargs='+', choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument('--masks', nargs='+', choices=('111', '000'), default=['111', '000'])
    parser.add_argument('--mode', choices=('both', 'score', 'generation'), default='both')
    parser.add_argument('--max-sequence', type=int, default=8192)
    parser.add_argument('--generate-tokens', type=int, default=None)
    parser.add_argument('--dry-run', action='store_true', help='Stdlib-only; do not create output or import tensor libraries')
    args = parser.parse_args(argv)
    if args.offset < 0 or not 1 <= args.limit <= 16:
        parser.error('offset must be nonnegative; use 1–16 items per shared-slot batch')
    try:
        schema = campaign_schema(args.campaign_root)
    except ValueError as error:
        parser.error(str(error))
    generation_cap = 2048 if schema == SCHEMA_V2 else 128
    if args.generate_tokens is None:
        args.generate_tokens = generation_cap
    if not 1 <= args.generate_tokens <= generation_cap or not 2 <= args.max_sequence <= 8192:
        parser.error(f'{schema} permits 1–{generation_cap} generated tokens and at most 8192 total tokens')
    for name in ('budgets', 'variants', 'masks'):
        if len(getattr(args, name)) != len(set(getattr(args, name))):
            parser.error(f'Duplicate {name}')
    if args.mode != 'generation' and '111' not in args.masks:
        parser.error('Scoring must include 111 for the common baseline')
    if not args.output.resolve().is_relative_to(args.campaign_root.resolve()) or args.output.resolve() == args.campaign_root.resolve():
        parser.error('Each fresh attempt must be inside the explicitly selected external campaign root')
    if args.output.resolve().is_relative_to(args.campaign_root.resolve()/'manifests'):
        parser.error('Do not use the preparation manifest directory as a run output')
    return args


def scientific_protocol(args, model):
    # Missing campaign_root is the persisted v1 metadata schema. Keep every v1
    # protocol field identical so historical runs remain verifiable after repair.
    schema = campaign_schema(getattr(args, 'campaign_root', ROOT))
    protocol = {'schema': schema, 'checkpoint': {k: model[k] for k in ('model', 'revision', 'weight_sha256', 'total_parameters')},
            'device': 'mps', 'dtype': 'bfloat16', 'batch_size': 1,
            'budgets': {name: PIXELS[name] for name in args.budgets}, 'min_pixels': 4096,
            'max_sequence_including_continuation': args.max_sequence,
            'variants': args.variants, 'permutation': 'cyclic_left_one_choice; preserve question prefix and all image markers',
            'score_masks': args.masks if args.mode != 'generation' else [],
            'score_protocol': 'complete parenthesized legal-label summed log probability; common prompt boundary checked',
            'generation_masks': ['111'] if args.mode != 'score' else [],
            'generation_protocol': 'native greedy model.generate; fresh KV per item/variant/budget; no adapter or external visual cache',
            'generation_max_new_tokens': args.generate_tokens,
            'generation_parser': PARSER_VERSION,
            'generation_accuracy': 'all planned rows denominator; only explicit unambiguous parsed labels with EOS count as scorable; invalid/truncated rows separately reported',
            'source_label_version': 'pinned HF BLINK a3666eb249237ba3d5eca8db21176cc47967e040',
            'original_to_control_boundary': 'New readout protocol; original1901 rows are not pooled into this matrix'}
    if schema == SCHEMA_V2:
        protocol['effective_generation_contract'] = {'use_model_defaults': False, **GREEDY_KWARGS,
                                                       'required_mode': 'greedy_search'}
        protocol['pixel_provenance'] = 'SHA256 of contiguous actual processor pixel_values bytes, plus dtype and shape'
    return protocol


def check_swap(initial):
    current = swap_used_mb()
    if initial is not None and current is not None and current > initial + 128:
        raise RuntimeError(f'Swap grew from {initial} MB to {current} MB; stop own batch')
    return current


def native_greedy(model, inputs, config, *, stopping_criteria=None):
    """No scorer kwargs or cache objects may leak into a native generation call."""
    forbidden = ('past_key_values', 'cache_position', 'position_ids', 'inputs_embeds', 'labels', 'logits_to_keep')
    if any(inputs.get(key) is not None for key in forbidden):
        raise ValueError('Generation requires fresh processor inputs, without scorer or KV state')
    # Inputs may be reused as immutable processor tensors, never as language KV.
    kwargs = dict(inputs)
    # In transformers>=4.50 a passed config's False can be treated as a global
    # default and refilled from checkpoint do_sample=True. Explicit kwargs and
    # use_model_defaults=False jointly make the intended generation mode binding.
    return model.generate(**kwargs, generation_config=config, stopping_criteria=stopping_criteria,
                          use_model_defaults=False, **GREEDY_KWARGS)


def resolve_greedy_config(model, config):
    """Use the installed generation implementation to record effective settings."""
    resolved, _ = model._prepare_generation_config(config, use_model_defaults=False, **GREEDY_KWARGS)
    if any(getattr(resolved, name) != value for name, value in GREEDY_KWARGS.items()):
        raise RuntimeError('Effective generation settings are not the explicit greedy contract')
    if resolved.get_generation_mode().value != 'greedy_search':
        raise RuntimeError('Effective generation mode is not greedy_search')
    return resolved


def input_record(torch, inputs, processor, model, prompt, source, variant, budget, options, max_sequence, generated, *, pixel_hash=False):
    input_ids = inputs.input_ids[0].detach().cpu().tolist()
    grids = inputs.image_grid_thw.detach().cpu().tolist()
    patch = processor.image_processor.patch_size
    merge = processor.image_processor.merge_size
    pixels = [g[1] * g[2] * patch ** 2 for g in grids]
    if any(value > PIXELS[budget] or value < 4096 for value in pixels):
        raise RuntimeError('Actual grid violates declared image budget')
    visual = sum(token == model.config.image_token_id for token in input_ids)
    if visual != sum(g[0] * g[1] * g[2] // merge ** 2 for g in grids):
        raise RuntimeError('Actual image token/grid counts differ')
    longest_label_prefix = max(len(ids) - 1 for ids in options.token_ids)
    if len(input_ids) + max(longest_label_prefix, generated) > max_sequence:
        raise RuntimeError('Actual prompt plus scored/generated continuation exceeds total sequence budget')
    identities = {'input_ids': input_ids, 'image_grid_thw': grids,
                  'image_sha256': [x['sha256'] for x in source['image_metadata']],
                  'prompt_text': prompt, 'option_tokens': options.to_dict()}
    pixels_provenance = {}
    if pixel_hash:
        pixel_tensor = inputs.pixel_values.detach().cpu().contiguous()
        pixels_provenance = {'pixel_values_sha256': hashlib.sha256(pixel_tensor.view(torch.uint8).numpy().tobytes()).hexdigest(),
                            'pixel_values_shape': list(pixel_tensor.shape), 'pixel_values_dtype': str(pixel_tensor.dtype)}
        identities.update(pixels_provenance)
        del pixel_tensor
    return {'prompt_text': prompt, 'prompt_sha256': options.prompt_sha256, **pixels_provenance,
            'question_prefix_sha256': variant['prefix_sha256'], 'input_ids': input_ids,
            'input_signature_sha256': canonical_digest(identities),
            'sequence_length': len(input_ids), 'image_grid_thw': grids,
            'processed_pixels_per_image': pixels, 'visual_token_count': visual,
            'pixel_budget': PIXELS[budget], 'max_sequence': max_sequence,
            'option_tokens': options.to_dict(),
            'model_input_dtypes': {k: str(v.dtype) for k, v in inputs.items() if hasattr(v, 'dtype')},
            'image_metadata': source['image_metadata'], 'source': source['source'],
            'label_to_original': variant['label_to_original'],
            'original_to_label': variant['original_to_label'],
            'answer_label': variant['answer_label'], 'original_answer_label': source['answer_label'],
            'display_choices': variant['display_choices'], 'original_choices': source['choices']}


def main(argv=None):
    args = parse_args(argv)
    selected = read_controls(args.manifest)[args.offset:args.offset + args.limit]
    if not selected:
        raise ValueError('Empty selected batch')
    spec = json.loads(args.snapshot.read_text())
    if spec['model'] != 'Qwen/Qwen3-VL-4B-Instruct' or spec['revision'] != REVISION or spec['total_parameters'] != PARAMETERS:
        raise ValueError('Expected the verified Qwen3-VL-4B snapshot')
    protocol = scientific_protocol(args, spec)
    expected_score = len(selected) * len(args.budgets) * len(args.variants) * len(protocol['score_masks'])
    expected_generation = len(selected) * len(args.budgets) * len(args.variants) * len(protocol['generation_masks'])
    if args.dry_run:
        print(json.dumps({'ids': [r['id'] for r in selected], 'output': str(args.output),
                          'protocol': protocol, 'expected_score_rows': expected_score,
                          'expected_generation_rows': expected_generation}, indent=2))
        return 0
    args.output.mkdir(parents=True, exist_ok=False)
    meta = {'schema': protocol['schema'], 'started_at': utc_now(), 'finished_at': None,
            'manifest': str(args.manifest.resolve()), 'manifest_sha256': digest(args.manifest),
            'ids': [r['id'] for r in selected], 'arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            'model': spec, 'scientific_protocol': protocol, 'scientific_protocol_sha256': canonical_digest(protocol),
            'execution_source_sha256': {name: digest(REPO/name) for name in EXECUTION_FILES},
            'platform': platform.platform(), 'python': sys.version,
            'expected_score_rows': expected_score, 'expected_generation_rows': expected_generation,
            'completed_score_rows': 0, 'completed_generation_rows': 0,
            'initial_swap_mb': swap_used_mb()}
    write_json(args.output/'metadata.json', meta)
    current = {}
    with (args.output/'scores.jsonl').open('x', buffering=1) as score_file, \
            (args.output/'generations.jsonl').open('x', buffering=1) as generation_file, \
            (args.output/'failures.jsonl').open('x', buffering=1) as failures:
        try:
            import torch
            from PIL import Image
            from transformers import Qwen3VLForConditionalGeneration, StoppingCriteria, StoppingCriteriaList
            from transformers.models.qwen3_vl import modeling_qwen3_vl
            from deepstack.intervention import DeepStackIntervention
            from deepstack.scoring import tokenize_options, score_options
            torch.set_num_threads(4)
            torch.manual_seed(20260928)
            if not torch.backends.mps.is_available():
                raise RuntimeError('MPS is unavailable')
            meta['libraries'] = {name: importlib.metadata.version(name) for name in ('torch', 'transformers', 'Pillow')}
            meta['transformers_source_sha256'] = digest(inspect.getfile(modeling_qwen3_vl))
            began = time.monotonic()
            model = Qwen3VLForConditionalGeneration.from_pretrained(spec.get('path') or str(Path(os.environ.get('HF_HOME', DATA_ROOT / 'huggingface')) / 'hub' / ('models--' + spec['model'].replace('/', '--')) / 'snapshots' / spec['revision']), local_files_only=True,
                        torch_dtype=torch.bfloat16, attn_implementation='sdpa').eval().to('mps')
            meta['load_seconds'] = time.monotonic() - began
            meta['parameter_count'] = sum(x.numel() for x in model.parameters())
            if meta['parameter_count'] != PARAMETERS or list(model.config.vision_config.deepstack_visual_indexes) != [5, 11, 17]:
                raise RuntimeError('Actual model identity/path layout differs')
            meta['actual_parameter_dtypes'] = sorted({str(x.dtype) for x in model.parameters()})
            meta['actual_parameter_devices'] = sorted({str(x.device) for x in model.parameters()})
            meta['memory_after_load'] = memory(torch, 'mps')
            if meta['actual_parameter_dtypes'] != ['torch.bfloat16'] or meta['actual_parameter_devices'] != ['mps:0']:
                raise RuntimeError('Actual precision/backend differs')
            processors = {name: make_processor(spec.get('path') or str(Path(os.environ.get('HF_HOME', DATA_ROOT / 'huggingface')) / 'hub' / ('models--' + spec['model'].replace('/', '--')) / 'snapshots' / spec['revision']), 4096, PIXELS[name]) for name in args.budgets}
            meta['processors'] = {name: {'processor': pr.to_dict(), 'image_processor': pr.image_processor.to_dict()}
                                  for name, pr in processors.items()}
            config = deepcopy(model.generation_config)
            config.do_sample = False; config.num_beams = 1; config.num_return_sequences = 1
            config.use_cache = True; config.max_new_tokens = args.generate_tokens
            config.min_new_tokens = 0; config.return_dict_in_generate = False
            config.output_scores = False; config.output_logits = False
            config.stop_strings = None
            meta['generation_config_requested'] = config.to_dict()
            config = resolve_greedy_config(model, config)
            eos = config.eos_token_id
            eos_ids = [eos] if isinstance(eos, int) else list(eos or [])
            if not eos_ids:
                raise RuntimeError('Checkpoint has no EOS configuration')
            meta['generation_config'] = config.to_dict()
            meta['generation_config_resolution'] = {'use_model_defaults': False, 'explicit_kwargs': GREEDY_KWARGS,
                                                    'effective_mode': config.get_generation_mode().value,
                                                    'implementation': 'installed model._prepare_generation_config followed by identical explicit generate kwargs'}
            write_json(args.output/'metadata.json', meta)

            class SwapGuard(StoppingCriteria):
                def __init__(self):
                    self.last = 0.0
                    self.error = None
                def __call__(self, input_ids, scores, **kwargs):
                    if time.monotonic() - self.last >= 2:
                        self.last = time.monotonic()
                        try:
                            check_swap(meta['initial_swap_mb'])
                        except RuntimeError as error:
                            self.error = str(error)
                    return self.error is not None

            with torch.inference_mode():
                for item_index, source in enumerate(selected):
                    for budget, processor in processors.items():
                        for name in args.variants:
                            control = source['readout_control']['variants'][name]
                            current = {'id': source['id'], 'task': source['task'], 'budget': budget, 'variant': name}
                            images = []
                            try:
                                for file in source['images']:
                                    with Image.open(file) as image:
                                        images.append(image.convert('RGB'))
                                messages = [{'role': 'user', 'content': [{'type': 'text', 'text': control['prompt']},
                                            *[{'type': 'image'} for _ in images]]}]
                                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                                inputs = processor(text=[text], images=images, return_tensors='pt').to('mps')
                            finally:
                                for image in images:
                                    image.close()
                            options = tokenize_options(processor.tokenizer, source['option_labels'], prompt_text=text)
                            shared = input_record(torch, inputs, processor, model, text, source, control, budget,
                                                  options, args.max_sequence,
                                                  args.generate_tokens if protocol['generation_masks'] else 0,
                                                  pixel_hash=protocol['schema'] == SCHEMA_V2)
                            check_swap(meta['initial_swap_mb'])
                            if protocol['score_masks']:
                                original = score_options(model, inputs, options).scores if item_index == 0 else None
                                # Cache is scoped to this exact item, prompt and resolution only.
                                with cached_vision(model):
                                    for mask in args.masks:
                                        current['mask'] = mask
                                        began = time.monotonic()
                                        with DeepStackIntervention(model, mask, audit=item_index == 0) as adapter:
                                            scored = score_options(model, inputs, options, correct_label=control['answer_label'])
                                        mapped = control['label_to_original'][scored.prediction]
                                        row = {**current, **shared, 'kind': 'score', 'scores': scored.scores,
                                               'prediction': scored.prediction, 'original_prediction': mapped,
                                               'correct': scored.prediction == control['answer_label'],
                                               'original_correct': mapped == source['answer_label'],
                                               'margin': scored.correct_margin, 'forward_count': scored.forward_count,
                                               'scoring_method': scored.method, 'elapsed_seconds': time.monotonic() - began,
                                               'memory': memory(torch, 'mps'), 'system_swap_used_mb': check_swap(meta['initial_swap_mb'])}
                                        if original is not None and mask == '111':
                                            difference = max(abs(scored.scores[k]-original[k]) for k in original)
                                            row['original_vs_111_max_score_difference'] = difference
                                            if difference > .001:
                                                raise RuntimeError(f'Original/111 mismatch: {difference}')
                                        if item_index == 0:
                                            row['intervention_audit'] = adapter.records
                                        score_file.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
                                        meta['completed_score_rows'] += 1
                            if protocol['generation_masks']:
                                # Outside both adapter and cached_vision; each call owns fresh native KV.
                                current['mask'] = '111'
                                guard = SwapGuard(); began = time.monotonic()
                                sequences = native_greedy(model, inputs, config,
                                                         stopping_criteria=StoppingCriteriaList([guard]))
                                ids = sequences[0, shared['sequence_length']:].detach().cpu().tolist()
                                raw = processor.tokenizer.decode(ids, skip_special_tokens=False)
                                decoded = processor.tokenizer.decode(ids, skip_special_tokens=True)
                                stop = generation_stop(ids, eos_ids, args.generate_tokens)
                                if guard.error:
                                    stop.update(stop_reason='resource_guard', truncated=False)
                                parsed = parse_generated_label(decoded, source['option_labels'])
                                scorable = parsed['status'] == 'parsed' and stop['eos_reached'] and not guard.error
                                label = parsed['label'] if scorable else None
                                mapped = control['label_to_original'][label] if label else None
                                row = {**current, **shared, 'kind': 'generation', 'generated_token_ids': ids,
                                       'generated_text_raw': raw, 'generated_text': decoded,
                                       'generated_tokens': len(ids), 'eos_token_ids': eos_ids,
                                       'max_new_tokens': args.generate_tokens, **stop,
                                       'parse': parsed, 'scorable': scorable, 'prediction': label,
                                       'original_prediction': mapped,
                                       'correct': label == control['answer_label'] if scorable else None,
                                       'original_correct': mapped == source['answer_label'] if scorable else None,
                                       'elapsed_seconds': time.monotonic()-began, 'memory': memory(torch, 'mps'),
                                       'system_swap_used_mb': swap_used_mb(), 'resource_guard_error': guard.error}
                                generation_file.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
                                meta['completed_generation_rows'] += 1
                                del sequences
                                if guard.error:
                                    raise RuntimeError(guard.error)
                                check_swap(meta['initial_swap_mb'])
                            write_json(args.output/'metadata.json', meta)
                            print(json.dumps({'event': 'condition_complete', **current,
                                              'score_rows': meta['completed_score_rows'],
                                              'generation_rows': meta['completed_generation_rows']}), flush=True)
                            del inputs
                    gc.collect(); torch.mps.empty_cache()
            if meta['completed_score_rows'] != expected_score or meta['completed_generation_rows'] != expected_generation:
                raise RuntimeError('Output counts differ from planned matrix')
            meta['finished_at'] = utc_now()
            meta['memory_at_end'] = memory(torch, 'mps')
            meta['final_swap_mb'] = check_swap(meta['initial_swap_mb'])
            write_json(args.output/'metadata.json', meta)
        except BaseException as error:
            failure = {'time': utc_now(), **current, 'error': repr(error), 'traceback': traceback.format_exc()}
            failures.write(json.dumps(failure, ensure_ascii=False) + '\n')
            meta['failed_at'] = utc_now(); meta['failure'] = failure
            write_json(args.output/'metadata.json', meta)
            raise
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
