"""Pinned Qwen3-VL slot controls for Linux CUDA; local CPU planning only."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import importlib.metadata
import inspect
import json
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from deepstack.slot_swap import (SCHEMA, SEED, TASKS, conditions, plan_item)
from scripts.run_pilot import json_dump, make_processor, sha256
from deepstack.readout_v3 import PARSER_VERSION
from cloud_prep.measurement_v3.runtime import (add_runtime_args, plan_runtime, prepare_runtime,
    read_snapshot, resolve_snapshot, load_records, validate_cuda, memory, swap_used_mb, check_swap)

EXECUTION_SCHEMA = 'measurement-v3-slots-cuda-1'
GENERATION_ALIASES = {'m0d0': 'main_original__ds_original', 'm0d1': 'main_original__ds_swapped',
                      'm1d0': 'main_swapped__ds_original', 'm1d1': 'main_swapped__ds_swapped',
                      'm0du': 'main_original__ds_unrelated', 'm1du': 'main_swapped__ds_unrelated'}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--donor-manifest', type=Path, required=True)
    p.add_argument('--output', type=Path)
    add_runtime_args(p)
    p.add_argument('--tasks', nargs='+', choices=TASKS, default=list(TASKS))
    p.add_argument('--offset', type=int, default=0)
    p.add_argument('--limit', type=int, default=1)
    p.add_argument('--format', default='strict_digits')
    p.add_argument('--mappings', nargs='+', choices=('identity', 'cyclic'), default=['identity', 'cyclic'])
    p.add_argument('--min-pixels', type=int, default=4096)
    p.add_argument('--max-pixels', type=int, default=262144)
    p.add_argument('--max-sequence', type=int, default=8192)
    p.add_argument('--seed', type=int, default=SEED)
    p.add_argument('--audit-tolerance', type=float, default=.001)
    p.add_argument('--generate-tokens', type=int, default=0,
                   help='0 disables generation; bounded pilot generation may use 64')
    p.add_argument('--generation-conditions', nargs='+', default=['m0d0','m0d1','m1d0','m1d1'],
                   choices=[*GENERATION_ALIASES, *GENERATION_ALIASES.values()])
    p.add_argument('--dry-run', action='store_true', help='Validate fixed selection/plans without importing torch')
    args = p.parse_args(argv)
    if args.offset < 0 or not 1 <= args.limit <= 14:
        p.error('Use a nonnegative offset and batches of 1–14 items')
    if args.min_pixels != 4096 or not 4096 <= args.max_pixels <= 262144:
        p.error('This arm keeps min_pixels=4096 and a cap no larger than 262144')
    if not 1 <= args.max_sequence <= 8192 or not 0 <= args.audit_tolerance <= .01:
        p.error('Invalid context or audit tolerance')
    if len(set(args.mappings)) != len(args.mappings) or len(set(args.tasks)) != len(args.tasks):
        p.error('Duplicate tasks/mappings would duplicate conditions')
    args.generation_conditions = [GENERATION_ALIASES.get(c, c) for c in args.generation_conditions]
    if len(set(args.generation_conditions)) != len(args.generation_conditions):
        p.error('Duplicate generation conditions')
    if not 0 <= args.generate_tokens <= 2048:
        p.error('Use 0–2048 generation tokens')
    try:
        plan_runtime(args)
    except ValueError as error:
        p.error(str(error))
    if not args.dry_run and args.output is None:
        p.error('--output is required for a model run')
    if args.output is not None and not args.output.resolve().is_relative_to(args.campaign_root.resolve() / 'slots'):
        p.error('Fresh attempt output must be under the explicit campaign-root/slots/')
    return args


def prepare(args):
    """Pure Python; select only requested tasks, then bind every chosen item."""
    rows = [r for r in load_records(args.manifest, args.image_root) if r['task'] in args.tasks]
    chosen = rows[args.offset:args.offset + args.limit]
    if not chosen:
        raise ValueError('Empty selected slot-control batch')
    pool = load_records(args.donor_manifest, args.image_root)
    plans = [plan_item(r, pool, seed=args.seed, min_pixels=args.min_pixels, max_pixels=args.max_pixels)
             for r in chosen]
    # No outcome-dependent filtering or silent skipping of incompatible images.
    return chosen, {r['id']: r for r in pool}, plans


def score_difference(a, b):
    if set(a) != set(b):
        raise ValueError('Compared option identities differ')
    return max(abs(a[k] - b[k]) for k in a)


def input_metadata(inputs, prompt, row, variant, options, torch):
    pixels = inputs['pixel_values'].detach().contiguous().cpu()
    raw = pixels.view(torch.uint8).numpy().tobytes()
    import hashlib
    return {'prompt_text': prompt, 'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
            'sequence_length': int(inputs['input_ids'].shape[1]),
            'input_ids': inputs['input_ids'].detach().cpu().tolist(),
            'image_grid_thw': inputs['image_grid_thw'].detach().cpu().tolist(),
            'pixel_values_sha256': hashlib.sha256(raw).hexdigest(),
            'pixel_values_shape': list(pixels.shape), 'pixel_values_dtype': str(pixels.dtype),
            'option_tokens': options.to_dict(), 'label_to_original': variant['label_to_original'],
            'original_to_label': variant['original_to_label'], 'display_choices': variant['display_choices'],
            'assistant_prefix': variant['assistant_prefix'], 'original_choices': row['choices'],
            'source': row['source'], 'image_metadata': row['image_metadata']}


SLOT_SOURCES = (
    'cloud_prep/measurement_v3/slots.py', 'cloud_prep/measurement_v3/runtime.py',
    'cloud_prep/portable_data.py', 'scripts/run_slot_swap.py', 'deepstack/slot_swap.py', 'deepstack/readout_v3.py',
    'deepstack/intervention.py', 'deepstack/scoring.py', 'deepstack/data.py', 'scripts/run_pilot.py',
    'scripts/run_readout_controls.py', 'scripts/prepare_readout_controls.py')


def main(argv=None):
    args = parse_args(argv)
    rows, pool, plans = prepare(args)
    from deepstack.readout_v3 import format_variant
    variants = {(row['id'], mapping): format_variant(row, args.format, mapping)
                for row in rows for mapping in args.mappings}
    runtime = plan_runtime(args) if args.dry_run else prepare_runtime(args)
    spec = read_snapshot(args) if args.dry_run else resolve_snapshot(args)
    if args.dry_run:
        if 'torch' in sys.modules:
            raise RuntimeError('Dry-run imported torch')
        print(json.dumps({'schema': SCHEMA, 'execution_schema': EXECUTION_SCHEMA, 'runtime': runtime,
                          'model': spec, 'parser': PARSER_VERSION, 'dry_run': True, 'items': len(rows), 'plans': plans,
                          'conditions': conditions(), 'format': args.format, 'mappings': args.mappings,
                          'expected_score_rows': len(rows) * len(args.mappings) * len(conditions()),
                          'expected_generation_rows': (len(rows) * len(args.mappings) * len(args.generation_conditions)
                                                       if args.generate_tokens else 0),
                          'torch_imported': False}, ensure_ascii=False, indent=2))
        return 0
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'failures.jsonl').touch()
    meta = {'schema': SCHEMA, 'execution_schema': EXECUTION_SCHEMA, 'runtime': runtime,
            'parser': PARSER_VERSION, 'started_at': utc_now(), 'ids': [r['id'] for r in rows],
            'model': spec, 'arguments': {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
            'manifest_sha256': sha256(args.manifest), 'donor_manifest_sha256': sha256(args.donor_manifest),
            'plans': plans, 'conditions': conditions(), 'format': args.format, 'mappings': args.mappings,
            'expected_score_rows': len(rows) * len(args.mappings) * len(conditions()),
            'expected_generation_rows': (len(rows) * len(args.mappings) * len(args.generation_conditions)
                                         if args.generate_tokens else 0),
            'completed_score_rows': 0, 'completed_generation_rows': 0,
            'platform': platform.platform(), 'python': sys.version,
            'input_protocol': {'min_pixels': args.min_pixels, 'max_pixels': args.max_pixels,
                               'max_sequence': args.max_sequence, 'batch_size': 1},
            'precision': {'body': 'bfloat16', 'output_head': 'float32', 'backend': 'cuda', 'attention': 'sdpa'},
            'initial_swap_mb': swap_used_mb(),
            # The historical MPS runner scripts/run_slot_swap.py is not part of the
            # six-model bundle and is never imported here; record absence explicitly.
            'source_sha256': {name: sha256(ROOT/name) for name in SLOT_SOURCES if (ROOT/name).exists()},
            'source_missing': [name for name in SLOT_SOURCES if not (ROOT/name).exists()],
            'limits': 'Directed slot-swap response only; donor is unrelated same-grid content, not a semantic-specificity proof.'}
    json_dump(args.output / 'metadata.json', meta)
    active_id = None
    try:
        import torch
        from transformers import Qwen3VLForConditionalGeneration, StoppingCriteria, StoppingCriteriaList
        from transformers.models.qwen3_vl import modeling_qwen3_vl
        from deepstack.intervention import DeepStackIntervention
        from deepstack.scoring import score_options
        from deepstack.readout_v3 import make_inputs, tokenize_variant, fp32_output_head, parse_answer
        from scripts.run_readout_controls import native_greedy, resolve_greedy_config
        from scripts.prepare_readout_controls import generation_stop
        from deepstack.slot_swap import (encode_visual, use_visual_output, intervention_output,
                                         image_token_counts, feature_norms, compare_outputs, validate_output)
        meta['cuda'] = validate_cuda(torch)
        torch.set_num_threads(4)
        torch.manual_seed(args.seed)
        start = time.monotonic()
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            spec['path'], local_files_only=True, torch_dtype=torch.bfloat16,
            attn_implementation='sdpa').eval().to('cuda')
        processor = make_processor(spec['path'], args.min_pixels, args.max_pixels)
        if processor.image_processor.patch_size != 16 or processor.image_processor.merge_size != 2:
            raise ValueError('Grid planner requires the pinned patch16/merge2 processor')
        meta.update(load_seconds=time.monotonic()-start,
                    parameter_count=sum(x.numel() for x in model.parameters()),
                    libraries={n: importlib.metadata.version(n) for n in ('torch', 'transformers', 'Pillow')},
                    transformers_source_sha256=sha256(inspect.getfile(modeling_qwen3_vl)),
                    image_processor=processor.image_processor.to_dict(),
                    vision_indexes=list(model.model.visual.deepstack_visual_indexes))
        if meta['parameter_count'] != spec['total_parameters']:
            raise ValueError('Actual model parameter count differs')
        generation_config = None
        if args.generate_tokens:
            generation_config = deepcopy(model.generation_config)
            generation_config.max_new_tokens = args.generate_tokens
            generation_config.do_sample = False
            generation_config.num_beams = 1
            generation_config.num_return_sequences = 1
            generation_config.use_cache = True
            generation_config = resolve_greedy_config(model, generation_config)
            eos_ids = generation_config.eos_token_id
            eos_ids = [eos_ids] if isinstance(eos_ids, int) else list(eos_ids)
            meta['effective_generation_config'] = generation_config.to_dict()
            meta['effective_generation_mode'] = 'greedy_search'
        class SwapGuard(StoppingCriteria):
            def __init__(self):
                self.checked = 0.; self.error = None
            def __call__(self, input_ids, scores, **kwargs):
                now = time.monotonic()
                if now - self.checked >= 10:
                    self.checked = now
                    try:
                        check_swap(meta['initial_swap_mb'])
                    except RuntimeError as exc:
                        self.error = str(exc)
                return bool(self.error)
        json_dump(args.output / 'metadata.json', meta)
        with (args.output/'results.jsonl').open('w', buffering=1) as out, \
                (args.output/'generations.jsonl').open('w', buffering=1) as generation_out, \
                torch.inference_mode(), fp32_output_head(model):
            for index, (row, plan) in enumerate(zip(rows, plans)):
                active_id = row['id']
                audit = index == 0
                original_output = donor_output = None
                item_audit = {'id': active_id, 'plan': plan, 'mappings': {}}
                for mapping in args.mappings:
                    variant = variants[(active_id, mapping)]
                    inputs, prompt = make_inputs(processor, row, variant, device='cuda')
                    options = tokenize_variant(processor.tokenizer, variant, prompt)
                    grids = inputs['image_grid_thw'].detach().cpu().tolist()
                    if grids != plan['expected_grid_thw']:
                        raise ValueError('Actual target grid disagrees with metadata-only plan')
                    counts = image_token_counts(grids, 2)
                    visual_count = int((inputs['input_ids'] == model.config.image_token_id).sum().item())
                    if visual_count != sum(counts):
                        raise ValueError('Image tokens do not cover all complete image blocks')
                    if inputs['input_ids'].shape[1] + max(max(map(len, options.token_ids)), args.generate_tokens) > args.max_sequence:
                        raise ValueError('Input plus complete option exceeds context bound')
                    details = input_metadata(inputs, prompt, row, variant, options, torch)
                    if original_output is None:
                        original_output = encode_visual(model, inputs)
                        donor_row = pool[plan['donor_id']]
                        donor_inputs, _ = make_inputs(processor, donor_row, variant, device='cuda')
                        donor_grids = donor_inputs['image_grid_thw'].detach().cpu().tolist()
                        if donor_grids != grids:
                            raise ValueError('Actually processed donor full grid does not match target')
                        donor_output = encode_visual(model, donor_inputs)
                        validate_output(original_output, grids)
                        validate_output(donor_output, grids)
                        original_pixel_sha = details['pixel_values_sha256']
                        donor_details = {'id': donor_row['id'], 'images': donor_row['images'],
                                         'image_metadata': donor_row['image_metadata'],
                                         'image_grid_thw': donor_grids,
                                         'encoding': 'actual separate visual forward outside replacement/cache context'}
                        del donor_inputs
                        if audit:
                            saved_original = (original_output[0].clone(), [x.clone() for x in original_output[1]])
                            saved_donor = (donor_output[0].clone(), [x.clone() for x in donor_output[1]])
                    elif details['pixel_values_sha256'] != original_pixel_sha:
                        raise ValueError('Mapping changed actual processed image tensor')
                    expected_original = plan['original_correct_choice']
                    expected_swapped = plan['swapped_correct_choice']
                    original_label = variant['original_to_label'][expected_original]
                    swapped_label = variant['original_to_label'][expected_swapped]
                    native_scores = score_options(model, inputs, options).scores if audit else None
                    condition_scores = {}
                    for condition in conditions():
                        started = time.monotonic()
                        applied = intervention_output(original_output, donor_output, grids, plan['swap_pair'], condition)
                        actual_gold = swapped_label if condition['main'] == 'swapped' else original_label
                        with use_visual_output(model, applied, grids) as delivered:
                            with DeepStackIntervention(model, '111', audit=audit) as adapter:
                                result = score_options(model, inputs, options, correct_label=actual_gold)
                        if delivered['calls'] != result.forward_count:
                            raise ValueError('Cached visual delivery does not cover every fresh scoring forward')
                        condition_scores[condition['name']] = result.scores
                        baseline_diff = (score_difference(result.scores, native_scores)
                                         if audit and condition['main'] == condition['deepstack'] == 'original' else None)
                        record = {'schema': SCHEMA, 'arm': 'slots', 'kind': 'score', 'id': active_id, 'task': row['task'],
                                  'format': args.format, 'mapping': mapping, 'variant': mapping,
                                  'condition': condition['name'], 'condition_spec': condition,
                                  **result.to_dict(), 'margin': result.correct_margin,
                                  'answer_label': actual_gold, 'correct': result.prediction == actual_gold,
                                  'original_prediction': variant['label_to_original'][result.prediction],
                                  'original_answer_label': expected_original,
                                  'main_correct_original_choice': expected_swapped if condition['main'] == 'swapped' else expected_original,
                                  'swap_correct_original_choice': expected_swapped,
                                  'score_original_correct_choice': result.scores[original_label],
                                  'score_swapped_correct_choice': result.scores[swapped_label],
                                  'swap_direction_log_odds': result.scores[swapped_label] - result.scores[original_label],
                                  'slot_plan': plan, 'unrelated_donor': donor_details,
                                  'source_feature_norms': feature_norms(original_output, grids),
                                  'applied_feature_norms': feature_norms(applied, grids),
                                  'main_visual_output_replaced': True,
                                  'selfidentity_max_score_difference': baseline_diff,
                                  'elapsed_seconds': time.monotonic()-started,
                                  'memory': memory(torch, 'cuda'), 'system_swap_used_mb': swap_used_mb(),
                                  'output_head_precision': 'float32', **details}
                        if audit:
                            record['deepstack_injection_audit'] = adapter.records
                        out.write(json.dumps(record, ensure_ascii=False) + '\n')
                        meta['completed_score_rows'] += 1
                        if baseline_diff is not None and baseline_diff > args.audit_tolerance:
                            raise RuntimeError('Selfidentity/native scoring comparison failed')
                        if (meta['initial_swap_mb'] is not None and record['system_swap_used_mb'] is not None
                                and record['system_swap_used_mb'] > meta['initial_swap_mb'] + 128):
                            raise RuntimeError('Swap increased >128 MB; stopping own run')
                        print(json.dumps({k: record[k] for k in ('id','mapping','condition','prediction','swap_direction_log_odds','elapsed_seconds')}), flush=True)
                        if args.generate_tokens and condition['name'] in args.generation_conditions:
                            guard = SwapGuard()
                            began = time.monotonic()
                            with use_visual_output(model, applied, grids) as native_delivery:
                                with DeepStackIntervention(model, '111', audit=audit) as gen_adapter:
                                    sequences = native_greedy(model, inputs, generation_config,
                                        stopping_criteria=StoppingCriteriaList([guard]))
                            ids = sequences[0, details['sequence_length']:].detach().cpu().tolist()
                            if native_delivery['calls'] != 1 or len(gen_adapter.records) != 1:
                                raise ValueError('Generation must consume replaced visual features once, at fresh prefill')
                            decoded = processor.tokenizer.decode(ids, skip_special_tokens=True)
                            parsed = parse_answer(decoded, variant['labels'], variant['assistant_prefix'])
                            stop = generation_stop(ids, eos_ids, args.generate_tokens)
                            scorable = parsed['status'] == 'parsed' and stop['eos_reached'] and not guard.error
                            prediction = parsed['label'] if scorable else None
                            generated = {key: record[key] for key in ('schema','arm','id','task','format','mapping','variant',
                                'condition','condition_spec','answer_label','original_answer_label','main_correct_original_choice',
                                'swap_correct_original_choice','slot_plan','unrelated_donor','source_feature_norms',
                                'applied_feature_norms','main_visual_output_replaced','output_head_precision')}
                            generated.update(**details, kind='generation', generated_token_ids=ids,
                                generated_text_raw=processor.tokenizer.decode(ids, skip_special_tokens=False),
                                generated_text=decoded, generated_tokens=len(ids), eos_token_ids=eos_ids,
                                max_new_tokens=args.generate_tokens, **stop, parse=parsed, scorable=scorable,
                                prediction=prediction,
                                original_prediction=variant['label_to_original'][prediction] if scorable else None,
                                correct=prediction == actual_gold if scorable else None,
                                resource_guard_error=guard.error, elapsed_seconds=time.monotonic()-began,
                                memory=memory(torch, 'cuda'), system_swap_used_mb=swap_used_mb())
                            if audit:
                                generated['deepstack_injection_audit'] = gen_adapter.records
                            generation_out.write(json.dumps(generated, ensure_ascii=False) + '\n')
                            meta['completed_generation_rows'] += 1
                            del sequences
                            if guard.error:
                                raise RuntimeError(guard.error)
                            check_swap(meta['initial_swap_mb'])
                        del applied
                    if audit:
                        swapped_row = dict(row)
                        swapped_row['images'] = [row['images'][i] for i in plan['slot_order']]
                        swapped_inputs, swapped_prompt = make_inputs(processor, swapped_row, variant, device='cuda')
                        if swapped_prompt != prompt or swapped_inputs['image_grid_thw'].detach().cpu().tolist() != grids:
                            raise ValueError('Real image reorder changed prompt/full grid')
                        real_swapped = encode_visual(model, swapped_inputs)
                        block_swapped = intervention_output(original_output, donor_output, grids, plan['swap_pair'],
                            next(c for c in conditions() if c['main'] == c['deepstack'] == 'swapped'))
                        comparison = compare_outputs(real_swapped, block_swapped)
                        real_scores = score_options(model, swapped_inputs, options).scores
                        diff = score_difference(real_scores, condition_scores['main_swapped__ds_swapped'])
                        item_audit['mappings'][mapping] = {'native_original_scores': native_scores,
                            'selfidentity_max_score_difference': score_difference(native_scores, condition_scores['main_original__ds_original']),
                            'reencode_vs_blockswap_features': comparison, 'real_reencode_scores': real_scores,
                            'reencode_vs_blockswap_max_score_difference': diff,
                            'original_source_unchanged': all(x['equal'] for x in compare_outputs(original_output, saved_original)),
                            'donor_source_unchanged': all(x['equal'] for x in compare_outputs(donor_output, saved_donor))}
                        json_dump(args.output/'first_item_audit.json', item_audit)
                        if diff > args.audit_tolerance or any(x['max_abs_difference'] > args.audit_tolerance for x in comparison):
                            raise RuntimeError('Real swapped reencode differs from post-merger block swap')
                        if not all(item_audit['mappings'][mapping][k] for k in ('original_source_unchanged','donor_source_unchanged')):
                            raise RuntimeError('Cached source tensors were mutated')
                        del swapped_inputs, real_swapped, block_swapped
                    del inputs
                del original_output, donor_output
                if audit:
                    del saved_original, saved_donor
                torch.cuda.empty_cache()
                json_dump(args.output/'metadata.json', meta)
        if meta['completed_score_rows'] != meta['expected_score_rows']:
            raise RuntimeError('Final score matrix is incomplete')
        if meta['completed_generation_rows'] != meta['expected_generation_rows']:
            raise RuntimeError('Final generation matrix is incomplete')
        meta.update(finished_at=utc_now(), memory_at_end=memory(torch, 'cuda'))
        json_dump(args.output/'metadata.json', meta)
        print(json.dumps({'event': 'complete', 'output': str(args.output), 'rows': meta['completed_score_rows']}), flush=True)
        return 0
    except BaseException as exc:
        meta.update(failed_at=utc_now(), error=repr(exc))
        json_dump(args.output/'metadata.json', meta)
        with (args.output/'failures.jsonl').open('a') as failed:
            failed.write(json.dumps({'id': active_id, 'error': repr(exc), 'at': utc_now()}) + '\n')
        raise


if __name__ == '__main__':
    raise SystemExit(main())
