"""Main layout M x donor pose D x DS permutation P; CPU dry-run plan, Linux CUDA execution.

Keeps every v6 qwen_content CUDA check (grid guard, processor patch order, rotary
order, permutation inverse, self-patch tolerance, unchanged source cache, fresh
forward delivery counts, single visual delivery per generation, swap guard, GPU
lock exit 75, failures.jsonl, source_sha256) and adds the exact M1 processor
patch-permutation check. Rows come from the spatial_data family registry.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import importlib.metadata
import inspect
import json
from pathlib import Path
import platform
import os
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from cloud_prep.measurement_v3.runtime import (add_runtime_args, plan_runtime, prepare_runtime,
    read_snapshot, resolve_snapshot, validate_cuda, memory, swap_used_mb, check_swap)
from cloud_prep.measurement_v3.slots import input_metadata, score_difference
from cloud_prep.measurement_v6.content_data import model_row
from cloud_prep.measurement_v6.content_readout import make_variant, parse_response
from cloud_prep.measurement_v7.spatial_data import load_rows, select_panel, FAMILIES
from cloud_prep.measurement_v7.spatial_design import (DESIGN_SCHEMA, read_design, execution_plan,
    generation_selected, compose_visual_output, spatial_token_indices, audit_main_layout_bridge,
    main_pairs, cloud_gpu_lock)
from deepstack.readout_v3 import PARSER_VERSION
from scripts.run_pilot import json_dump, make_processor, sha256

EXECUTION_SCHEMA = 'qwen-main-layout-reference-cuda-v7'
SOURCES = ('cloud_prep/measurement_v7/qwen_spatial.py', 'cloud_prep/measurement_v7/spatial_design.py',
           'cloud_prep/measurement_v7/spatial_data.py',
           'cloud_prep/measurement_v6/content_readout.py', 'cloud_prep/measurement_v6/content_data.py',
           'cloud_prep/measurement_v6/content_intervention.py', 'cloud_prep/measurement_v4/content_intervention.py',
           'cloud_prep/measurement_v3/runtime.py', 'cloud_prep/measurement_v3/slots.py', 'cloud_prep/portable_data.py',
           'deepstack/slot_swap.py', 'deepstack/readout_v3.py', 'deepstack/scoring.py',
           'deepstack/intervention.py', 'scripts/run_readout_controls.py', 'scripts/prepare_readout_controls.py',
           'scripts/run_pilot.py')


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--family', help='Registered row family; default: infer from the row schema')
    p.add_argument('--family-module', action='append', default=[], help='Import cloud_prep.measurement_v7.<module> to register a family')
    p.add_argument('--design-config', type=Path, help='Exploratory strengths/formats/generation overrides')
    p.add_argument('--panel', choices=('smoke', 'main'), default='main')
    p.add_argument('--output', type=Path)
    add_runtime_args(p)
    p.add_argument('--offset', type=int, default=0)
    p.add_argument('--limit', type=int, default=None)
    p.add_argument('--mappings', nargs='+', choices=('identity', 'cyclic'), default=['identity', 'cyclic'])
    p.add_argument('--min-pixels', type=int, default=4096)
    p.add_argument('--max-pixels', type=int, default=262144)
    p.add_argument('--max-sequence', type=int, default=4096)
    p.add_argument('--generate-tokens', type=int, default=32)
    p.add_argument('--audit-tolerance', type=float, default=.001)
    p.add_argument('--seed', type=int, default=20261005)
    p.add_argument('--gpu', default=os.environ.get('CUDA_VISIBLE_DEVICES', '0'))
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args(argv)
    if not args.gpu or any(c not in '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ-_' for c in args.gpu):
        p.error('Expose one GPU index/UUID, not a comma-separated list')
    if args.offset < 0 or (args.limit is not None and args.limit < 1):
        p.error('Invalid offset/limit')
    if len(set(args.mappings)) != len(args.mappings):
        p.error('Duplicate mappings')
    if args.min_pixels < 1 or args.max_pixels < args.min_pixels:
        p.error('Positive ordered pixel bounds required')
    if args.max_sequence < 1 or args.generate_tokens < 0:
        p.error('Invalid sequence/generation bound')
    if not 0 <= args.audit_tolerance <= .01:
        p.error('Invalid audit tolerance')
    try:
        plan_runtime(args)
    except ValueError as error:
        p.error(str(error))
    if not args.dry_run and args.output is None:
        p.error('--output is required for execution')
    if args.output is not None and not args.output.resolve().is_relative_to(args.campaign_root.resolve() / 'spatial'):
        p.error('Use a fresh campaign-root/spatial attempt directory')
    return args


def prepare(args):
    args.family, rows = load_rows(args.manifest, args.image_root, family=args.family, modules=args.family_module)
    rows = select_panel(rows, args.panel, args.family)
    rows = rows[args.offset:None if args.limit is None else args.offset + args.limit]
    if not rows:
        raise ValueError('Empty selected panel')
    args.design = read_design(args.design_config)
    plan = execution_plan(rows, args.panel, args.mappings, args.generate_tokens, args.design, args.family)
    required = {c['main'] for c in plan['conditions']} | {name for c in plan['conditions'] for name in c['ds_sources']}
    for row in rows:
        missing = required - set(row['source_images'])
        if missing:
            raise ValueError('Requested sources absent; rebuild the dataset with these strengths: ' + str(sorted(missing)))
    return rows, plan


def content_fields(row, variant, condition, prediction):
    """Exact serialized identities shared by score and generation records."""
    original = variant['label_to_original'][prediction] if prediction is not None else None
    return {'schema': DESIGN_SCHEMA, 'row_schema': row['schema'], 'family': row['family'], 'id': row['id'],
        'scene_id': row['scene_id'], 'task': row['task'],
        **{k: row[k] for k in ('attribute', 'target_object', 'target_index', 'target_main_slot',
                               'target_main_slot_by_layout', 'generation_group')},
        'record_fields': row['record_fields'],
        'mapping': variant['mapping'], 'response_format': variant['response_format'], 'condition': condition['name'],
        'condition_spec': condition, 'source_answer_indices': row['source_answer_indices'],
        'source_original_slot_answer_indices': row['source_original_slot_answer_indices'],
        'donor_current_slot_answer_indices': row['donor_current_slot_answer_indices'],
        'main_source': condition['main'], 'main_layout': condition['main_layout'], 'ds_sources': condition['ds_sources'],
        'ds_code': condition['ds_code'], 'main_kind': condition['main_kind'], 'main_strength': condition['main_strength'],
        'donor_pose': condition['donor_pose'], 'token_permutation': condition['token_permutation'],
        'template_signs': condition['template_signs'],
        'original_prediction': original,
        'content_prediction': row['choices'][row['option_labels'].index(original)] if original else None}


def main(argv=None):
    args = parse_args(argv)
    rows, plan = prepare(args)
    if args.dry_run:
        if 'torch' in sys.modules:
            raise RuntimeError('CPU planning imported torch')
        print(json.dumps({**plan, 'execution_schema': EXECUTION_SCHEMA, 'runtime': plan_runtime(args),
            'model': read_snapshot(args), 'manifest_sha256': sha256(args.manifest),
            'family_module': FAMILIES[args.family]['module'], 'ids': [r['id'] for r in rows],
            'dry_run': True, 'torch_imported': False}, indent=2))
        return 0
    runtime = prepare_runtime(args)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    try:
        with cloud_gpu_lock(args.campaign_root, args.gpu):
            return execute(args, rows, plan, runtime)
    except BlockingIOError:
        print(json.dumps({'event': 'gpu_busy', 'gpu': args.gpu}), flush=True)
        return 75


def execute(args, rows, plan, runtime):
    spec = resolve_snapshot(args)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'failures.jsonl').touch()
    meta = {**plan, 'execution_schema': EXECUTION_SCHEMA, 'runtime': runtime, 'model': spec,
            'started_at': utc_now(), 'parser': {'label': PARSER_VERSION, 'direct': 'strict_content_v6'},
            'formats': args.design['response_formats'], 'family_module': FAMILIES[args.family]['module'],
            'arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            'manifest_sha256': sha256(args.manifest), 'ids': [r['id'] for r in rows],
            'source_sha256': {name: sha256(ROOT / name) for name in SOURCES},
            'family_source_sha256': sha256(sys.modules[FAMILIES[args.family]['load'].__module__].__file__),
            'platform': platform.platform(), 'python': sys.version,
            'completed_score_rows': 0, 'completed_generation_rows': 0,
            'initial_swap_mb': swap_used_mb(), 'gpu_lock_owner': 'qwen_spatial process; campaign-root/locks/gpu-N.lock',
            'interpretation': 'Hybrid source direction is not accuracy. M1 changes only the main image; D is relative to the original layout. Direct scores sum candidate token log probabilities and are not pooled with label scores. Permutation moves contextualized, position-bearing features, not pure semantic identity.'}
    json_dump(args.output / 'metadata.json', meta)
    active_id = None
    try:
        import torch
        from transformers import Qwen3VLForConditionalGeneration, StoppingCriteria, StoppingCriteriaList
        from transformers.models.qwen3_vl import modeling_qwen3_vl
        from deepstack.intervention import DeepStackIntervention
        from deepstack.scoring import score_options
        from deepstack.readout_v3 import make_inputs, tokenize_variant, fp32_output_head
        from contextlib import nullcontext
        from deepstack.slot_swap import (encode_visual, use_visual_output, image_token_counts, expected_grid,
                                        validate_output, compare_outputs)
        from scripts.run_readout_controls import native_greedy, resolve_greedy_config
        from scripts.prepare_readout_controls import generation_stop
        meta['cuda'] = validate_cuda(torch)
        torch.set_num_threads(4)
        torch.manual_seed(args.seed)
        began = time.monotonic()
        model = Qwen3VLForConditionalGeneration.from_pretrained(spec['path'], local_files_only=True,
            torch_dtype=torch.bfloat16, attn_implementation='sdpa').eval().to('cuda')
        processor = make_processor(spec['path'], args.min_pixels, args.max_pixels)
        if processor.image_processor.patch_size != 16 or processor.image_processor.merge_size != 2:
            raise ValueError('Expected pinned patch16/merge2 processor')
        meta.update(load_seconds=time.monotonic() - began,
            parameter_count=sum(x.numel() for x in model.parameters()),
            libraries={n: importlib.metadata.version(n) for n in ('torch', 'transformers', 'Pillow')},
            transformers_source_sha256=sha256(inspect.getfile(modeling_qwen3_vl)),
            image_processor=processor.image_processor.to_dict(),
            vision_indexes=list(model.model.visual.deepstack_visual_indexes))
        if meta['parameter_count'] != spec['total_parameters']:
            raise ValueError('Checkpoint parameter count differs')
        generation_config, eos_ids = None, []
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

        class SwapGuard(StoppingCriteria):
            def __init__(self):
                self.checked = 0.; self.error = None

            def __call__(self, input_ids, scores, **kwargs):
                if time.monotonic() - self.checked >= 10:
                    self.checked = time.monotonic()
                    try:
                        check_swap(meta['initial_swap_mb'])
                    except RuntimeError as exc:
                        self.error = str(exc)
                return bool(self.error)
        pairs = main_pairs(args.design)
        audited_attributes = set()
        json_dump(args.output / 'metadata.json', meta)
        with (args.output / 'results.jsonl').open('w', buffering=1) as out, \
             (args.output / 'generations.jsonl').open('w', buffering=1) as gen_out, \
             (args.output / 'audits.jsonl').open('w', buffering=1) as audit_out, \
             torch.inference_mode(), fp32_output_head(model):
            requested = sorted({c['main'] for c in plan['conditions']} | {s for c in plan['conditions'] for s in c['ds_sources']})
            for row in rows:
                active_id = row['id']
                audit = row['attribute'] not in audited_attributes
                sources, saved, source_pixels = {}, {}, {}
                for readout in plan['readouts']:
                    mapping = readout['mapping']
                    variant = make_variant(row, readout['response_format'], mapping)
                    base_row = model_row(row, requested[0])
                    inputs, prompt = make_inputs(processor, base_row, variant, device='cuda')
                    options = tokenize_variant(processor.tokenizer, variant, prompt)
                    grids = inputs['image_grid_thw'].detach().cpu().tolist()
                    expected = [expected_grid(base_row['image_metadata'][0], min_pixels=args.min_pixels, max_pixels=args.max_pixels)]
                    if grids != expected or int((inputs['input_ids'] == model.config.image_token_id).sum().item()) != sum(image_token_counts(grids, 2)):
                        raise ValueError('Actual input grid/visual token count differs')
                    if inputs['input_ids'].shape[1] + max(max(map(len, options.token_ids)), args.generate_tokens) > args.max_sequence:
                        raise ValueError('Input+continuation exceeds selected context bound')
                    spatial_token_indices(grids)
                    if grids != [[1, 16, 24]]:
                        raise ValueError('v7 pixel-aligned inputs require the original 384x256 patch16 grid; revise data explicitly before resizing')
                    details = input_metadata(inputs, prompt, base_row, variant, options, torch)
                    details.update(response_format=variant['response_format'],
                        candidate_token_lengths={label: len(ids) for label, ids in zip(options.labels, options.token_ids)},
                        score_normalization='sum_sequence_logprob_no_length_normalization')
                    source_inputs = {requested[0]: inputs}
                    for name in requested[1:]:
                        source_inputs[name], source_prompt = make_inputs(processor, model_row(row, name), variant, device='cuda')
                        if source_prompt != prompt or not torch.equal(source_inputs[name]['input_ids'], inputs['input_ids']):
                            raise ValueError('Source image changed prompt/tokens')
                        if source_inputs[name]['image_grid_thw'].detach().cpu().tolist() != grids:
                            raise ValueError('Source image geometry changed')
                    source_details = {}
                    for name in requested:
                        source_details[name] = input_metadata(source_inputs[name], prompt, model_row(row, name), variant, options, torch)
                        pixel_sha = source_details[name]['pixel_values_sha256']
                        if name not in sources:
                            sources[name] = encode_visual(model, source_inputs[name]); validate_output(sources[name], grids)
                            source_pixels[name] = pixel_sha
                            if audit:
                                saved[name] = (sources[name][0].clone(), [f.clone() for f in sources[name][1]])
                        elif source_pixels[name] != pixel_sha:
                            raise ValueError('Mapping changed processed source pixels')
                    spatial_audit = audit_main_layout_bridge(model, source_inputs, sources, grids, pairs) if audit else None
                    a_original = row['option_labels'][row['answer_index']]
                    b_original = row['option_labels'][1 - row['answer_index']]
                    a_label, b_label = variant['original_to_label'][a_original], variant['original_to_label'][b_original]
                    native_scores = {s: score_options(model, inp, options).scores for s, inp in source_inputs.items()} if audit else {}
                    for condition in plan['conditions']:
                        began = time.monotonic()
                        applied = compose_visual_output(sources, condition, grids)
                        is_native = condition['role'] in ('native_main', 'real_donor')
                        real = condition['role'] == 'real_donor'
                        native_source = condition['main'] if is_native else None
                        native_index = row['source_answer_indices'][native_source] if native_source else None
                        native_gold = variant['original_to_label'][row['option_labels'][native_index]] if native_index is not None else None
                        current_inputs = source_inputs[condition['main']] if real else inputs
                        context = nullcontext(None) if real else use_visual_output(model, applied, grids)
                        with context as delivered, DeepStackIntervention(model, '111', audit=audit) as adapter:
                            result = score_options(model, current_inputs, options, correct_label=native_gold)
                        if delivered is not None and delivered['calls'] != result.forward_count:
                            raise ValueError('Replacement missed a fresh score forward')
                        check_swap(meta['initial_swap_mb'])
                        difference = None
                        if audit and native_source:
                            comparison = result.scores
                            if real:
                                with use_visual_output(model, applied, grids), DeepStackIntervention(model, '111', audit=False):
                                    comparison = score_options(model, inputs, options).scores
                            difference = score_difference(comparison, native_scores[native_source])
                        planned = bool(args.generate_tokens and generation_selected(row, args.panel, args.design) and condition['generation'])
                        record = {**content_fields(row, variant, condition, result.prediction),
                            'execution_schema': EXECUTION_SCHEMA, 'kind': 'score', 'arm': 'spatial', **result.to_dict(),
                            'source_a_label': a_label, 'source_b_label': b_label,
                            'source_a_original_choice': a_original, 'source_b_original_choice': b_original,
                            'source_a_value': row['choices'][row['answer_index']],
                            'source_b_value': row['choices'][1 - row['answer_index']],
                            'b_minus_a_log_odds': result.scores[b_label] - result.scores[a_label],
                            'native_source': native_source, 'native_answer_label': native_gold,
                            'native_correct': result.prediction == native_gold if native_gold else None,
                            'generation_planned': planned, 'actual_image_forward': real,
                            'source_images': row['source_images'], 'source_specs': row['source_specs'],
                            'source_image_metadata': row['source_image_metadata'],
                            'source_pixel_sha256': source_pixels, 'selfidentity_max_score_difference': difference,
                            'elapsed_seconds': time.monotonic() - began, 'memory': memory(torch),
                            'actual_pixel_input_source': condition['main'] if real else requested[0],
                            'visual_output_replaced': not real, **(source_details[condition['main']] if real else details),
                            'candidate_token_lengths': details['candidate_token_lengths'],
                            'score_normalization': details['score_normalization'],
                            'response_format': variant['response_format']}
                        if audit:
                            record['deepstack_injection_audit'] = adapter.records
                        out.write(json.dumps(record) + '\n'); meta['completed_score_rows'] += 1
                        if difference is not None and difference > args.audit_tolerance:
                            raise RuntimeError('Native/real-image and source self-patch scores disagree')
                        if planned:
                            guard = SwapGuard(); began = time.monotonic()
                            context = nullcontext(None) if real else use_visual_output(model, applied, grids)
                            with context as delivered, DeepStackIntervention(model, '111', audit=audit) as adapter:
                                sequences = native_greedy(model, current_inputs, generation_config,
                                    stopping_criteria=StoppingCriteriaList([guard]))
                            if (delivered is not None and delivered['calls'] != 1) or len(adapter.records) != 1:
                                raise ValueError('Fresh generation must consume visual input once')
                            ids = sequences[0, details['sequence_length']:].detach().cpu().tolist()
                            decoded = processor.tokenizer.decode(ids, skip_special_tokens=True)
                            parsed = parse_response(decoded, variant)
                            stop = generation_stop(ids, eos_ids, args.generate_tokens)
                            scorable = parsed['status'] == 'parsed' and stop['eos_reached'] and not guard.error
                            prediction = parsed['label'] if scorable else None
                            generated = {k: v for k, v in record.items() if k not in result.to_dict() and k not in ('deepstack_injection_audit', 'selfidentity_max_score_difference')}
                            generated.update(**content_fields(row, variant, condition, prediction), kind='generation',
                                generated_token_ids=ids, generated_text_raw=processor.tokenizer.decode(ids, skip_special_tokens=False),
                                generated_text=decoded, generated_tokens=len(ids), max_new_tokens=args.generate_tokens,
                                eos_token_ids=eos_ids, parse=parsed, scorable=scorable, prediction=prediction,
                                native_correct=prediction == native_gold if scorable and native_gold else None,
                                **stop, resource_guard_error=guard.error, elapsed_seconds=time.monotonic() - began, memory=memory(torch))
                            if audit:
                                generated['deepstack_injection_audit'] = adapter.records
                            gen_out.write(json.dumps(generated) + '\n'); meta['completed_generation_rows'] += 1
                            del sequences
                            if guard.error:
                                raise RuntimeError(guard.error)
                            check_swap(meta['initial_swap_mb'])
                        del applied
                    if audit:
                        checks = {name: compare_outputs(sources[name], saved[name]) for name in sources}
                        audit_out.write(json.dumps({'id': row['id'], 'mapping': mapping, 'native_scores': native_scores,
                            'source_unchanged': checks, 'grids': grids, 'response_format': variant['response_format'],
                            'spatial_bridge': spatial_audit}) + '\n')
                        if any(not item['equal'] for values in checks.values() for item in values):
                            raise RuntimeError('Source cache mutated')
                    del inputs, source_inputs
                audited_attributes.add(row['attribute']); del sources, saved
                torch.cuda.empty_cache(); json_dump(args.output / 'metadata.json', meta)
                print(json.dumps({'event': 'item_complete', 'id': active_id, 'score_rows': meta['completed_score_rows'],
                    'generation_rows': meta['completed_generation_rows']}), flush=True)
        if any(meta['completed_' + kind + '_rows'] != meta['expected_' + kind + '_rows'] for kind in ('score', 'generation')):
            raise RuntimeError('Output matrix incomplete')
        meta.update(finished_at=utc_now(), memory_at_end=memory(torch))
        json_dump(args.output / 'metadata.json', meta)
        return 0
    except BaseException as exc:
        meta.update(failed_at=utc_now(), error=repr(exc))
        json_dump(args.output / 'metadata.json', meta)
        with (args.output / 'failures.jsonl').open('a') as failure:
            failure.write(json.dumps({'id': active_id, 'error': repr(exc), 'at': utc_now()}) + '\n')
        raise


if __name__ == '__main__':
    raise SystemExit(main())
