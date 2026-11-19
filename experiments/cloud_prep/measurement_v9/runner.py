"""v9 local rescue and multiobject runner; tensor-free planning, CUDA execution."""
from __future__ import annotations
import argparse
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import fields, is_dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import time

from cloud_prep.measurement_v9.runtime import (plan_runtime, prepare_runtime, resolve_snapshot, sha256, write_json,
    direction_readout, cache_identity, visual_signature, validate_shared_interventions)
from cloud_prep.measurement_v9.geometry import preflight_geometry
from cloud_prep.measurement_v6.content_readout import make_variant, parse_response
from cloud_prep.measurement_v3.runtime import validate_cuda, memory, check_swap, swap_used_mb
from cloud_prep.measurement_v4.content_intervention import cloud_gpu_lock
from deepstack.readout_v3 import PARSER_VERSION

EXECUTION_SCHEMA = 'local-rescue-multiobject-cuda-v9'
ROOT = Path(__file__).resolve().parents[2]
REUSED_SOURCES = ('deepstack/scoring.py', 'deepstack/readout_v3.py', 'deepstack/intervention.py',
    'cloud_prep/measurement_v7/spatial_data.py', 'cloud_prep/measurement_v7/spatial_design.py',
    'cloud_prep/measurement_v7/qwen_spatial.py', 'cloud_prep/measurement_v7/natobj_family.py',
    'cloud_prep/measurement_v7/natobj_data.py', 'cloud_prep/measurement_v6/content_data.py',
    'cloud_prep/measurement_v6/content_readout.py', 'cloud_prep/measurement_v6/content_intervention.py',
    'cloud_prep/measurement_v4/content_intervention.py', 'cloud_prep/measurement_v3/runtime.py',
    'cloud_prep/portable_data.py', 'scripts/prepare_readout_controls.py')


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True)
    p.add_argument('--arm', choices=('rescue', 'multiobject'), required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--panel', choices=('smoke', 'main'), default='main')
    p.add_argument('--entry-groups', nargs='+', choices=('all', 'layer', 'spatial'), default=['all'])
    p.add_argument('--design-config', type=Path)
    p.add_argument('--campaign-root', type=Path, required=True)
    p.add_argument('--cache-root', type=Path, required=True)
    p.add_argument('--output', type=Path)
    p.add_argument('--offset', type=int, default=0)
    p.add_argument('--limit', type=int)
    p.add_argument('--mappings', nargs='+', choices=('identity', 'cyclic'), default=['identity', 'cyclic'])
    p.add_argument('--generate-tokens', type=int, default=32)
    p.add_argument('--max-sequence', type=int, default=4096)
    p.add_argument('--audit-tolerance', type=float, default=.001)
    p.add_argument('--seed', type=int, default=20261013)
    p.add_argument('--gpu', default=os.environ.get('CUDA_VISIBLE_DEVICES', '0'))
    p.add_argument('--local-files-only', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args(argv)
    if not a.gpu or any(c not in '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ-_' for c in a.gpu):
        p.error('Expose one GPU index/UUID per process')
    if a.offset < 0 or (a.limit is not None and a.limit < 1):
        p.error('Invalid offset/limit')
    if a.generate_tokens < 0 or a.max_sequence < 1 or not 0 <= a.audit_tolerance <= .01:
        p.error('Invalid generation/context/audit bound')
    if len(set(a.entry_groups)) != len(a.entry_groups) or len(set(a.mappings)) != len(a.mappings):
        p.error('Duplicate entry group/mapping')
    if a.entry_groups != ['all']:
        p.error('v9 uses the all-later-entry group; IBM layer0 remains main')
    if not a.dry_run and a.output is None:
        p.error('--output required for execution')
    plan_runtime(a)
    return a


def prepare(a):
    from cloud_prep.measurement_v8.model_registry import get_model
    spec = get_model(a.model)
    if not spec['model'].startswith('ibm-granite/') and a.entry_groups != ['all']:
        raise ValueError('Only IBM has layer/spatial path groups')
    if a.arm == 'rescue':
        from cloud_prep.measurement_v9 import rescue_data as data, rescue_design as design
    else:
        from cloud_prep.measurement_v9 import multiobject_data as data, multiobject_design as design
    rows = data.select_panel(data.load_manifest(a.manifest), a.panel)
    rows = rows[a.offset:None if a.limit is None else a.offset + a.limit]
    settings = design.read_design(a.design_config)
    shared = validate_shared_interventions(rows)
    plan = design.execution_plan(rows, panel=a.panel, mappings=a.mappings,
        generate_tokens=a.generate_tokens, design=settings, entry_groups=a.entry_groups)
    plan.update(shared)
    if not rows:
        raise ValueError('Empty selected panel')
    required = {c['main'] for c in plan['conditions']} | {c['donor_source'] for c in plan['conditions'] if c['donor_source'] is not None}
    for row in rows:
        if required - row['source_images'].keys():
            raise ValueError(f"Missing source images for {row['id']}")
    geometry_preflights = {}
    for row in rows:
        checked = preflight_geometry(row)
        key = str(checked['image_hw'])
        # Every row's ROI is checked; retain compact representative geometry.
        geometry_preflights.setdefault(key, checked)
    plan.update(execution_schema=EXECUTION_SCHEMA, arm=a.arm, model=spec, entry_groups=a.entry_groups,
                ids=[r['id'] for r in rows], manifest_sha256=sha256(a.manifest), runtime=plan_runtime(a),
                geometry_preflight=list(geometry_preflights.values()))
    return rows, plan, design


def record_fields(row, variant, condition, prediction, arm):
    if arm == 'rescue':
        from cloud_prep.measurement_v9.rescue_design import record_fields as arm_fields
    else:
        from cloud_prep.measurement_v9.multiobject_design import record_fields as arm_fields
    return arm_fields(row, variant, condition, prediction)


def tensor_digest(value, torch):
    """Hash cached tensors, independent of adapter container type; audit only."""
    h = hashlib.sha256()
    def visit(x):
        if isinstance(x, torch.Tensor):
            h.update(str((tuple(x.shape), str(x.dtype))).encode())
            h.update(x.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())
        elif is_dataclass(x):
            for f in fields(x):
                visit(getattr(x, f.name))
        elif isinstance(x, dict):
            for k in sorted(x):
                visit(x[k])
        elif isinstance(x, (tuple, list)):
            for y in x:
                visit(y)
    visit(value)
    return h.hexdigest()


def make_adapter(spec, snapshot):
    if spec['model'].startswith('ibm-granite/'):
        from cloud_prep.measurement_v9.adapter import GraniteAdapter
        adapter = GraniteAdapter.load(snapshot['path'], device='cuda', dtype='bfloat16', local_files_only=True)
        return adapter, getattr(adapter, 'metadata', {})
    from cloud_prep.measurement_v9.adapter import QwenAdapter
    from cloud_prep.measurement_v8.qwen_adapter import load_backend
    model, processor, meta = load_backend(spec, snapshot['path'], device='cuda', attn_implementation='sdpa')
    return QwenAdapter(model, processor, device='cuda'), meta


def is_native(condition):
    return condition['role'] in ('native_main', 'real_donor', 'native_full', 'native_donor')


def execute(a, rows, plan, design):
    runtime = prepare_runtime(a)
    a.output.mkdir(parents=True, exist_ok=False)
    meta = {**plan, 'runtime': runtime, 'started_at': utc_now(),
            'arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
            'parser': {'label': PARSER_VERSION, 'direct': 'strict_content_v6'},
            'source_sha256': {str(p.relative_to(ROOT)): sha256(p) for p in
                sorted({*(ROOT / name for name in REUSED_SOURCES), *(ROOT / 'cloud_prep/measurement_v8').glob('*.py'), *(ROOT / 'cloud_prep/measurement_v9').glob('*.py')})},
            'platform': platform.platform(), 'python': sys.version, 'completed_score_rows': 0,
            'completed_generation_rows': 0, 'initial_swap_mb': swap_used_mb()}
    write_json(a.output / 'metadata.json', meta)
    (a.output / 'failures.jsonl').touch()
    active = None
    try:
        import torch
        from transformers import StoppingCriteria, StoppingCriteriaList
        from deepstack.scoring import score_options
        from deepstack.readout_v3 import tokenize_variant, fp32_output_head
        from cloud_prep.measurement_v8.generation import native_greedy, resolve_greedy_config
        from scripts.prepare_readout_controls import generation_stop
        meta['cuda'] = validate_cuda(torch)
        torch.set_num_threads(4); torch.manual_seed(a.seed)
        snapshot = resolve_snapshot(plan['model'], a.cache_root, a.local_files_only)
        began = time.monotonic()
        adapter, adapter_meta = make_adapter(plan['model'], snapshot)
        model, processor = adapter.model, adapter.processor
        meta.update(model=snapshot, adapter=adapter_meta, load_seconds=time.monotonic()-began,
            parameter_count=sum(p.numel() for p in model.parameters()),
            libraries={n: importlib.metadata.version(n) for n in ('torch', 'transformers', 'Pillow')},
            image_processor=processor.image_processor.to_dict())
        config, eos_ids = None, []
        if a.generate_tokens:
            config = deepcopy(model.generation_config)
            config.max_new_tokens = a.generate_tokens
            config.do_sample = False; config.num_beams = 1; config.num_return_sequences = 1; config.use_cache = True
            config = resolve_greedy_config(model, config)
            eos_ids = config.eos_token_id
            eos_ids = [eos_ids] if isinstance(eos_ids, int) else list(eos_ids or [])
            if not eos_ids:
                raise ValueError('Explicit EOS token identity required')
            meta['effective_generation_config'] = config.to_dict()

        class SwapGuard(StoppingCriteria):
            def __init__(self): self.checked = 0.; self.error = None
            def __call__(self, input_ids, scores, **kwargs):
                if time.monotonic() - self.checked >= 10:
                    self.checked = time.monotonic()
                    try: check_swap(meta['initial_swap_mb'])
                    except RuntimeError as e: self.error = str(e)
                return bool(self.error)

        audit_interventions = {}; visual_digests = {}
        cached_identity = None; sources, pixel_hashes = {}, {}
        write_json(a.output / 'metadata.json', meta)
        with (a.output / 'results.jsonl').open('w', buffering=1) as out, \
             (a.output / 'generations.jsonl').open('w', buffering=1) as gout, \
             (a.output / 'audits.jsonl').open('w', buffering=1) as audits, \
             torch.inference_mode(), fp32_output_head(model):
            requested = sorted({c['main'] for c in plan['conditions']} | {c['donor_source'] for c in plan['conditions'] if c['donor_source'] is not None})
            for row in rows:
                active = row['id']
                identity, source_contract = cache_identity(row)
                audit_group = (row['attribute'], str(row.get('geometry')), row.get('panel_family'))
                audit_interventions.setdefault(audit_group, identity)
                audit = a.panel == 'smoke' or audit_interventions[audit_group] == identity
                reused_for_query = identity == cached_identity
                if not reused_for_query:
                    sources, pixel_hashes = {}, {}
                    cached_identity = identity
                for readout in plan['readouts']:
                    variant = make_variant(row, readout['response_format'], readout['mapping'])
                    inputs_by_source, prompt = {}, None
                    for name in requested:
                        inp, text = adapter.prepare(row, variant, name)
                        if prompt is not None and (text != prompt or not torch.equal(inp['input_ids'], next(iter(inputs_by_source.values()))['input_ids'])):
                            raise ValueError('Images changed prompt or token geometry')
                        prompt = text; inputs_by_source[name] = inp
                        current_hash = tensor_digest({k: v for k, v in inp.items() if k in ('pixel_values', 'image_sizes', 'image_grid_thw')}, torch)
                        if name not in sources:
                            sources[name] = adapter.encode(inp); pixel_hashes[name] = current_hash
                        elif pixel_hashes[name] != current_hash:
                            raise ValueError('Readout changed processed source pixels')
                    options = tokenize_variant(processor.tokenizer, variant, prompt)
                    seq_len = next(iter(inputs_by_source.values()))['input_ids'].shape[1]
                    input_ids_hash = hashlib.sha256(json.dumps(next(iter(inputs_by_source.values()))['input_ids'][0].detach().cpu().tolist(), separators=(',', ':')).encode()).hexdigest()
                    if seq_len + max(max(map(len, options.token_ids)), a.generate_tokens) > a.max_sequence:
                        raise ValueError('Input plus continuation exceeds context bound')
                    source_before = tensor_digest(sources, torch) if audit else None
                    audit_data = adapter.audit_sources(inputs_by_source, sources, row) if audit else {}
                    self_diffs, native_scores = {}, {}
                    if audit:
                        # Each unique native image must survive the adapter path before scientific comparisons.
                        for name, inp in inputs_by_source.items():
                            native = score_options(model, inp, options)
                            native_scores[name] = native.scores
                            with adapter.use(sources[name], inp) as delivery:
                                patched = score_options(model, inp, options)
                            if delivery['calls'] != patched.forward_count:
                                raise ValueError('Self-patch missed a fresh score forward')
                            diff = max(abs(native.scores[k]-patched.scores[k]) for k in native.scores)
                            self_diffs[name] = diff
                            if diff > a.audit_tolerance:
                                raise ValueError(f'Native/self-patch score mismatch: {name}: {diff}')
                    for c in plan['conditions']:
                        began = time.monotonic(); inp = inputs_by_source[c['main']]
                        native_source = c['main'] if is_native(c) else None
                        native_index = row['source_answer_indices'].get(native_source) if native_source else None
                        gold = variant['original_to_label'][row['option_labels'][native_index]] if native_index is not None else None
                        visual = sources[c['main']] if native_source else adapter.compose(sources, c, row, inputs=inp)
                        signature = visual_signature(row, c, pixel_hashes)
                        visual_sha = tensor_digest(visual, torch) if audit else None
                        if audit:
                            digest_key = (identity, c['name'])
                            prior = visual_digests.setdefault(digest_key, (signature, visual_sha))
                            if prior != (signature, visual_sha):
                                raise ValueError('Shared intervention changed across query/readout')
                        with (nullcontext(None) if native_source else adapter.use(visual, inp, audit=audit)) as delivered:
                            result = score_options(model, inp, options, correct_label=gold)
                        if delivered is not None and delivered['calls'] != result.forward_count:
                            raise ValueError('Replacement missed a fresh teacher-forced forward')
                        composition_self_difference = None
                        if c['role'] == 'self_patch':
                            if c['main'] not in native_scores:
                                native_scores[c['main']] = score_options(model, inp, options).scores
                            composition_self_difference = max(abs(result.scores[k]-native_scores[c['main']][k]) for k in result.scores)
                            if composition_self_difference > a.audit_tolerance:
                                raise ValueError('ROI/group compose self-patch differs from native scores')
                        planned = bool(a.generate_tokens and design.generation_selected(row, a.panel, plan['design']) and c['generation'])
                        record = {**record_fields(row, variant, c, result.prediction, a.arm),
                            'execution_schema': EXECUTION_SCHEMA, 'kind': 'score', 'arm': a.arm, **result.to_dict(),
                            'model_key': a.model, **direction_readout(row, variant, result.scores),
                            'visual_intervention_id': identity, 'visual_intervention_signature': signature,
                            'visual_tensor_sha256': visual_sha, 'source_contract_sha256': source_contract,
                            'shared_visual_cache_reused_for_query': reused_for_query,
                            'native_source': native_source, 'native_answer_label': gold,
                            'native_correct': result.prediction == gold if gold else None,
                            'generation_planned': planned, 'sequence_length': seq_len,
                            'candidate_token_lengths': {k: len(v) for k,v in zip(options.labels, options.token_ids)},
                            'score_normalization': 'sum_sequence_logprob_no_length_normalization',
                            'source_images': row['source_images'], 'source_pixel_sha256': pixel_hashes,
                            'source_pixel_sha256_scheme': 'v8_tensor_digest_pixels_and_geometry',
                            'source_image_sha256': {k:v['sha256'] for k,v in row['source_image_metadata'].items()},
                            'prompt_sha256': options.prompt_sha256, 'selfidentity_max_score_difference': self_diffs.get(c['main']),
                            'input_ids_sha256': input_ids_hash,
                            'composition_self_max_score_difference': composition_self_difference,
                            'actual_image_forward': bool(native_source), 'visual_output_replaced': not bool(native_source),
                            'actual_pixel_input_source': c['main'], 'visual_delivery': delivered,
                            'visual_intervention': getattr(visual, 'metadata', getattr(visual, 'intervention', {})),
                            'elapsed_seconds': time.monotonic()-began, 'memory': memory(torch)}
                        out.write(json.dumps(record)+'\n'); meta['completed_score_rows'] += 1
                        if planned:
                            guard = SwapGuard()
                            with (nullcontext(None) if native_source else adapter.use(visual, inp, audit=audit)) as delivery:
                                sequence = native_greedy(model, inp, config, stopping_criteria=StoppingCriteriaList([guard]))
                            if delivery is not None and delivery['calls'] != 1:
                                raise ValueError('Fresh cached generation must consume visual features exactly once')
                            ids = sequence[0,seq_len:].detach().cpu().tolist()
                            decoded = processor.tokenizer.decode(ids, skip_special_tokens=True)
                            parsed = parse_response(decoded, variant)
                            stop = generation_stop(ids, eos_ids, a.generate_tokens)
                            scorable = parsed['status'] == 'parsed' and stop['eos_reached'] and not guard.error
                            prediction = parsed['label'] if scorable else None
                            generated = {k:v for k,v in record.items() if k not in result.to_dict()}
                            # Explicit generation fields win over arm record fields (as in the score record literal);
                            # mechanism record_fields also carries native_correct.
                            generated.update(record_fields(row, variant, c, prediction, a.arm))
                            generated.update(kind='generation',
                                generated_token_ids=ids, generated_text=decoded,
                                generated_text_raw=processor.tokenizer.decode(ids, skip_special_tokens=False),
                                generated_tokens=len(ids), max_new_tokens=a.generate_tokens, eos_token_ids=eos_ids,
                                parse=parsed, scorable=scorable, prediction=prediction, **stop,
                                native_correct=prediction == gold if scorable and gold else None,
                                resource_guard_error=guard.error, visual_delivery=delivery, memory=memory(torch))
                            gout.write(json.dumps(generated)+'\n'); meta['completed_generation_rows'] += 1
                            del sequence
                            if guard.error: raise RuntimeError(guard.error)
                        check_swap(meta['initial_swap_mb']); del visual
                    if audit:
                        unchanged = source_before == tensor_digest(sources, torch)
                        audits.write(json.dumps({'id': row['id'], **readout, 'adapter': audit_data,
                            'native_self_max_differences': self_diffs, 'source_cache_unchanged': unchanged,
                            'visual_intervention_id': identity, 'shared_visual_cache_reused_for_query': reused_for_query,
                            'shared_intervention_tensor_checks': True})+'\n')
                        if not unchanged: raise ValueError('Source feature cache mutated')
                    del inputs_by_source
                # Retain only the current intervention cache for its paired query.
                # The next different intervention drops it before source encoding.
                torch.cuda.empty_cache(); write_json(a.output / 'metadata.json', meta)
                print(json.dumps({'event':'item_complete','id':active,'score_rows':meta['completed_score_rows'],
                                  'generation_rows':meta['completed_generation_rows']}), flush=True)
        if any(meta['completed_'+k+'_rows'] != meta['expected_'+k+'_rows'] for k in ('score','generation')):
            raise ValueError('Output matrix incomplete')
        meta.update(finished_at=utc_now(), memory_at_end=memory(torch)); write_json(a.output / 'metadata.json', meta)
        return 0
    except BaseException as e:
        # Record the failure first so a metadata write error cannot erase the evidence.
        with (a.output / 'failures.jsonl').open('a') as f:
            f.write(json.dumps({'id':active,'error':repr(e),'at':utc_now()})+'\n')
        meta.update(failed_at=utc_now(), error=repr(e)); write_json(a.output / 'metadata.json', meta)
        raise


def main(argv=None):
    a = parse_args(argv); rows, plan, design = prepare(a)
    if a.dry_run:
        if 'torch' in sys.modules: raise RuntimeError('CPU planning imported torch')
        print(json.dumps({**plan, 'torch_imported':False,'dry_run':True}, indent=2)); return 0
    prepare_runtime(a)
    try:
        with cloud_gpu_lock(a.campaign_root, a.gpu):
            return execute(a, rows, plan, design)
    except BlockingIOError:
        print(json.dumps({'event':'gpu_busy','gpu':a.gpu}), flush=True); return 75


if __name__ == '__main__':
    raise SystemExit(main())
