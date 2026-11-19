#!/usr/bin/env python3
"""CUDA measurement-v3 readout; local CPU planning only, no MPS fallback."""
from __future__ import annotations

import argparse
from copy import deepcopy
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time
import traceback

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO/'scripts'))
from deepstack.readout_v3 import FORMATS, MAPPINGS, PARSER_VERSION, format_variant, make_inputs, tokenize_variant, fp32_output_head, parse_answer
from cloud_prep.measurement_v3.runtime import (add_runtime_args, plan_runtime, prepare_runtime,
    read_snapshot, resolve_snapshot, load_records, validate_cuda, memory, swap_used_mb, check_swap)
from scripts.prepare_readout_controls import canonical_digest, digest, generation_stop
from scripts.run_pilot import cached_vision, make_processor
from scripts.run_readout_controls import (utc_now, write_json,
                                  native_greedy, resolve_greedy_config)




def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--offset', type=int, default=0)
    p.add_argument('--limit', type=int, default=1)
    p.add_argument('--ids', nargs='+')
    p.add_argument('--formats', nargs='+', choices=FORMATS, default=list(FORMATS))
    p.add_argument('--mappings', nargs='+', choices=MAPPINGS, default=list(MAPPINGS))
    p.add_argument('--masks', nargs='+', choices=('111', '000'), default=['111', '000'])
    p.add_argument('--generation-formats', nargs='*', choices=FORMATS,
                   default=['strict_letters', 'strict_digits', 'prefix_digits'])
    p.add_argument('--generate-tokens', type=int, default=64)
    p.add_argument('--max-pixels', type=int, default=262144)
    p.add_argument('--max-sequence', type=int, default=8192)
    add_runtime_args(p)
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args(argv)
    if not 1 <= a.limit <= 8 or a.offset < 0 or not 0 <= a.generate_tokens <= 256:
        p.error('Use 1–8 items per batch and 0–256 generation tokens')
    if not 4096 <= a.max_pixels <= 1048576 or not 2 <= a.max_sequence <= 8192:
        p.error('Invalid input budget')
    try:
        plan_runtime(a)
    except ValueError as error:
        p.error(str(error))
    if any(len(getattr(a, k)) != len(set(getattr(a, k))) for k in ('formats', 'mappings', 'masks', 'generation_formats')):
        p.error('Duplicate condition')
    return a


def main(argv=None):
    args = parse_args(argv)
    rows = load_records(args.manifest, args.image_root)
    if args.ids:
        by_id = {r['id']: r for r in rows}
        rows = [by_id[i] for i in args.ids]
    rows = rows[args.offset:args.offset + args.limit]
    if not rows:
        raise ValueError('No selected items')
    runtime = plan_runtime(args) if args.dry_run else prepare_runtime(args)
    spec = read_snapshot(args) if args.dry_run else resolve_snapshot(args)
    protocol = {'schema': 'measurement-v3-readout-1', 'phase': 'exploratory',
                'checkpoint': {k: spec[k] for k in ('model', 'revision', 'weight_sha256', 'total_parameters')},
                'body_dtype': 'bfloat16', 'output_projection': 'float32 from stored bf16 weights',
                'backend': 'cuda/sdpa', 'min_pixels': 4096, 'max_pixels': args.max_pixels,
                'max_sequence': args.max_sequence, 'formats': args.formats, 'mappings': args.mappings,
                'masks': args.masks, 'generation_formats': [f for f in args.formats if f in args.generation_formats],
                'generate_tokens': args.generate_tokens, 'parser': PARSER_VERSION,
                'generation': 'greedy, fresh KV per condition; exact-image visual outputs may be reused',
                'score': 'complete label continuation summed logprob; prefix_digits conditions on Answer: (',
                'sample_status': 'all BLINK validation items previously scored; no unseen confirmation claim'}
    expected_scores = len(rows)*len(args.formats)*len(args.mappings)*len(args.masks)
    expected_generations = (len(rows)*len(protocol['generation_formats'])*len(args.mappings)*len(args.masks)
                            if args.generate_tokens else 0)
    if args.dry_run:
        print(json.dumps({'ids': [r['id'] for r in rows], 'protocol': protocol,
                          'expected_scores': expected_scores, 'expected_generations': expected_generations}, indent=2))
        return 0
    args.output.mkdir(parents=True, exist_ok=False)
    meta = {'execution_schema': 'measurement-v3-readout-cuda-1', 'runtime': runtime,
            'started_at': utc_now(), 'finished_at': None, 'ids': [r['id'] for r in rows],
            'manifest': str(args.manifest), 'manifest_sha256': digest(args.manifest),
            'protocol': protocol, 'protocol_sha256': canonical_digest(protocol),
            'arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            'source_sha256': {p: digest(REPO/p) for p in ('cloud_prep/measurement_v3/readout.py', 'cloud_prep/measurement_v3/runtime.py',
                               'deepstack/readout_v3.py', 'deepstack/scoring.py', 'deepstack/intervention.py',
                               'scripts/run_pilot.py', 'scripts/run_readout_controls.py', 'scripts/prepare_readout_controls.py')},
            'expected_score_rows': expected_scores, 'expected_generation_rows': expected_generations,
            'completed_score_rows': 0, 'completed_generation_rows': 0, 'initial_swap_mb': swap_used_mb()}
    write_json(args.output/'metadata.json', meta)
    current = {}
    with (args.output/'scores.jsonl').open('x', buffering=1) as sf, (args.output/'generations.jsonl').open('x', buffering=1) as gf:
        try:
            import torch
            from transformers import Qwen3VLForConditionalGeneration, StoppingCriteria, StoppingCriteriaList
            from deepstack.intervention import DeepStackIntervention
            from deepstack.scoring import score_options
            torch.set_num_threads(4)
            torch.manual_seed(20261003)
            meta['cuda'] = validate_cuda(torch)
            meta['libraries'] = {n: importlib.metadata.version(n) for n in ('torch', 'transformers', 'Pillow')}
            began = time.monotonic()
            print(json.dumps({'event': 'loading', 'output': str(args.output)}), flush=True)
            model = Qwen3VLForConditionalGeneration.from_pretrained(spec['path'], local_files_only=True,
                       torch_dtype=torch.bfloat16, attn_implementation='sdpa').eval().to('cuda')
            meta['load_seconds'] = time.monotonic()-began
            if sum(p.numel() for p in model.parameters()) != spec['total_parameters']:
                raise RuntimeError('Wrong parameter count')
            processor = make_processor(spec['path'], 4096, args.max_pixels)
            config = deepcopy(model.generation_config)
            config.max_new_tokens = max(1, args.generate_tokens)
            config.min_new_tokens = 0
            config.return_dict_in_generate = False
            config.output_scores = False
            config.output_logits = False
            config.stop_strings = None
            config = resolve_greedy_config(model, config)
            eos = config.eos_token_id
            eos_ids = [eos] if isinstance(eos, int) else list(eos or [])
            if not eos_ids:
                raise RuntimeError('No EOS token')
            meta['generation_config'] = config.to_dict()
            meta['processor'] = processor.image_processor.to_dict()

            class SwapGuard(StoppingCriteria):
                error = None
                last = 0.0
                def __call__(self, input_ids, scores, **kwargs):
                    if time.monotonic()-self.last >= 2:
                        self.last = time.monotonic()
                        try:
                            check_swap(meta['initial_swap_mb'])
                        except RuntimeError as error:
                            self.error = str(error)
                    return self.error is not None

            with torch.inference_mode(), fp32_output_head(model):
                meta['memory_after_load_and_fp32_head'] = memory(torch, 'cuda')
                for item_index, source in enumerate(rows):
                    image_identity = None
                    with cached_vision(model):
                        for format_name in args.formats:
                            for mapping in args.mappings:
                                variant = format_variant(source, format_name, mapping)
                                inputs, prompt = make_inputs(processor, source, variant, device='cuda')
                                options = tokenize_variant(processor.tokenizer, variant, prompt)
                                ids = inputs.input_ids[0].detach().cpu().tolist()
                                grids = inputs.image_grid_thw.detach().cpu().tolist()
                                pixel_hash = hashlib.sha256(inputs.pixel_values.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
                                this_identity = (grids, pixel_hash)
                                if image_identity is not None and this_identity != image_identity:
                                    raise RuntimeError('Visual cache input changed within item')
                                image_identity = this_identity
                                if len(ids)+max(args.generate_tokens, max(map(len, options.token_ids))) > args.max_sequence:
                                    raise RuntimeError('Sequence budget exceeded')
                                shared = {'id': source['id'], 'task': source['task'], 'format': format_name,
                                          'mapping': mapping, 'answer_label': variant['answer_label'],
                                          'original_answer_label': source['answer_label'],
                                          'label_to_original': variant['label_to_original'],
                                          'display_choices': variant['display_choices'],
                                          'prompt_text': prompt, 'input_ids': ids, 'sequence_length': len(ids),
                                          'image_grid_thw': grids, 'pixel_values_sha256': pixel_hash,
                                          'option_tokens': options.to_dict(), 'protocol_sha256': meta['protocol_sha256']}
                                baseline = score_options(model, inputs, options).scores if item_index == 0 else None
                                for mask in args.masks:
                                    current = {k: shared[k] for k in ('id', 'task', 'format', 'mapping')}
                                    current['mask'] = mask
                                    began = time.monotonic()
                                    with DeepStackIntervention(model, mask, audit=item_index == 0) as adapter:
                                        scored = score_options(model, inputs, options, correct_label=variant['answer_label'])
                                    row = {**shared, **scored.to_dict(), 'kind': 'score', 'mask': mask,
                                           'correct': scored.prediction == variant['answer_label'],
                                           'original_prediction': variant['label_to_original'][scored.prediction],
                                           'elapsed_seconds': time.monotonic()-began,
                                           'system_swap_used_mb': check_swap(meta['initial_swap_mb'])}
                                    if baseline is not None and mask == '111':
                                        difference = max(abs(scored.scores[k]-baseline[k]) for k in baseline)
                                        row['original_vs_111_max_score_difference'] = difference
                                        if difference > 1e-5:
                                            raise RuntimeError(f'Original/111 mismatch: {difference}')
                                    if item_index == 0:
                                        row['intervention_audit'] = adapter.records
                                    sf.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
                                    meta['completed_score_rows'] += 1
                                    if args.generate_tokens and format_name in args.generation_formats:
                                        guard = SwapGuard()
                                        began = time.monotonic()
                                        with DeepStackIntervention(model, mask):
                                            generated = native_greedy(model, inputs, config, stopping_criteria=StoppingCriteriaList([guard]))
                                        generated_ids = generated[0, len(ids):].detach().cpu().tolist()
                                        decoded = processor.tokenizer.decode(generated_ids, skip_special_tokens=True)
                                        stop = generation_stop(generated_ids, eos_ids, args.generate_tokens)
                                        if guard.error:
                                            raise RuntimeError(guard.error)
                                        parsed = parse_answer(decoded, variant['labels'], variant['assistant_prefix'])
                                        scorable = parsed['status'] == 'parsed' and stop['eos_reached']
                                        pred = parsed['label'] if scorable else None
                                        grow = {**shared, 'kind': 'generation', 'mask': mask, 'text': decoded,
                                                'text_raw': processor.tokenizer.decode(generated_ids, skip_special_tokens=False),
                                                'generated_token_ids': generated_ids, 'generated_tokens': len(generated_ids),
                                                'assistant_prefix': variant['assistant_prefix'], 'parse': parsed, 'stop': stop,
                                                'scorable': scorable, 'prediction': pred,
                                                'original_prediction': variant['label_to_original'][pred] if pred else None,
                                                'correct': pred == variant['answer_label'] if scorable else None,
                                                'elapsed_seconds': time.monotonic()-began,
                                                'system_swap_used_mb': check_swap(meta['initial_swap_mb'])}
                                        gf.write(json.dumps(grow, ensure_ascii=False, allow_nan=False)+'\n')
                                        meta['completed_generation_rows'] += 1
                                        del generated
                                write_json(args.output/'metadata.json', meta)
                                print(json.dumps({'event': 'condition_complete', **current,
                                      'score_rows': meta['completed_score_rows'], 'generation_rows': meta['completed_generation_rows']}), flush=True)
                                del inputs
                    gc.collect()
                    torch.cuda.empty_cache()
            if (meta['completed_score_rows'], meta['completed_generation_rows']) != (expected_scores, expected_generations):
                raise RuntimeError('Incomplete matrix')
            meta['finished_at'] = utc_now()
            meta['final_swap_mb'] = check_swap(meta['initial_swap_mb'])
            meta['memory_at_end'] = memory(torch, 'cuda')
            write_json(args.output/'metadata.json', meta)
        except BaseException as error:
            meta['failed_at'] = utc_now()
            meta['failure'] = {**current, 'error': repr(error), 'traceback': traceback.format_exc()}
            write_json(args.output/'metadata.json', meta)
            raise
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
