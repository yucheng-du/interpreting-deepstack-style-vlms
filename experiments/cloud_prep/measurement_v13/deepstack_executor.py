"""DeepStack-L execution of the unchanged v12 common P1/P2 plans (CUDA only).

A port of measurement_v12.v11_v2_runner.execute (P1 natural, P2 dose) and of
measurement_v10.runner.execute (P2 local/full): same loop order, audits,
calibration/reuse/unmatched handling and record fields. Only these seams are
DeepStack-specific (each marked DEEPSTACK below):
  * adapter factory: official DeepStack-VL loader with the checkpoint's own
    fine-tuned vision tower (deepstack_adapter.load) + deepstack_core adapters;
  * CUDA/BF16 validation compatible with the pinned torch 2.1.2;
  * processed-pixel digest over DeepStack's views, image size and native grid;
  * complete candidates scored in the native Vicuna answer format (" " + label);
  * context bound on the expanded sequence (576 image positions) vs the
    checkpoint's tokenizer_model_max_length, so official truncation never acts;
  * generation through the official inputs_embeds generate, which returns only
    new tokens (verified per call; no prompt slicing).
"""
from __future__ import annotations
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

from cloud_prep.measurement_v12.v11_v2_runtime import (prepare_runtime, resolve_snapshot, sha256, write_json,
    direction_readout, cache_identity, visual_signature as dose_visual_signature, required_sources,
    baseline_evaluation_key, ScalarEvaluationCache)
from cloud_prep.measurement_v10.runtime import visual_signature as v10_visual_signature
from cloud_prep.measurement_v6.content_readout import make_variant, parse_response
from cloud_prep.measurement_v3.runtime import memory, check_swap, swap_used_mb
from deepstack.readout_v3 import PARSER_VERSION

ROOT = Path(__file__).resolve().parents[2]
EXECUTOR_SCHEMA = 'deepstack-l-common-core-executor-v13-v1'
MODES = {
    # mode: (execution_schema used by the common core for this arm, dose loop, v10 record form)
    'p1': ('natural-selectivity-cuda-v12-v1', True, False),
    'local': ('paired-attribute-cuda-v12-v1', False, True),
    'full': ('paired-attribute-cuda-v12-v1', False, True),
    'dose': ('paired-attribute-cuda-v12-v1', True, False),
}
V11_ONLY_SCORE_FIELDS = ('reference_forward_count', 'evaluation_reuse', 'dose_calibration', 'dose_metadata',
                         'protocol_status', 'v10_later_bridge')
REUSED_SOURCES = ('cloud_prep/measurement_v12/v11_v2_runner.py', 'cloud_prep/measurement_v10/runner.py',
    'cloud_prep/measurement_v12/v11_v2_runtime.py', 'cloud_prep/measurement_v12/v11_v2_dose_adapter.py',
    'cloud_prep/measurement_v12/v11_v2_dose_design.py', 'cloud_prep/measurement_v12/p1_design.py',
    'cloud_prep/measurement_v12/p2_design.py', 'cloud_prep/measurement_v10/local_design.py',
    'cloud_prep/measurement_v10/full_design.py', 'cloud_prep/measurement_v10/runtime.py',
    'cloud_prep/measurement_v9/runtime.py', 'cloud_prep/measurement_v8/runtime.py',
    'cloud_prep/measurement_v8/generation.py', 'cloud_prep/measurement_v6/content_readout.py',
    'cloud_prep/measurement_v6/content_intervention.py', 'cloud_prep/measurement_v3/runtime.py',
    'deepstack/scoring.py', 'deepstack/readout_v3.py', 'scripts/prepare_readout_controls.py',
    'cloud_prep/measurement_v13/deepstack_executor.py', 'cloud_prep/measurement_v13/deepstack_core.py',
    'cloud_prep/measurement_v13/deepstack_adapter.py', 'cloud_prep/measurement_v13/deepstack_geometry.py',
    'cloud_prep/measurement_v13/deepstack_runner.py', 'cloud_prep/measurement_v13/model_extensions.py',
    'cloud_prep/measurement_v13/runtime_provenance.py')
OFFICIAL_SOURCES = ('model/language_model/deepstack_llama.py', 'model/language_model/custom_llama.py',
    'model/deepstack_arch.py', 'model/llava_arch.py', 'model/multimodal_encoder/clip_encoder.py',
    'model/multimodal_encoder/builder.py', 'model/multimodal_projector/builder.py', 'mm_utils.py',
    'conversation.py', 'constants.py')


def utc_now():
    return datetime.now(timezone.utc).isoformat()


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


def is_native(condition):
    return condition['role'].startswith('native_') or condition['role'] == 'real_donor'


def validate_cuda(torch):
    """measurement_v3.runtime.validate_cuda semantics for torch 2.1 (no including_emulation)."""  # DEEPSTACK
    if sys.platform != 'linux' or not torch.cuda.is_available():
        raise RuntimeError('An available Linux CUDA GPU is required; no CPU/MPS fallback')
    if torch.cuda.device_count() != 1:
        raise RuntimeError('Expose one GPU to this process using CUDA_VISIBLE_DEVICES')
    capability = torch.cuda.get_device_capability(0)
    if capability < (8, 0) or not torch.cuda.is_bf16_supported():
        raise RuntimeError('This protocol requires native BF16 CUDA support')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    torch.cuda.reset_peak_memory_stats()
    props = torch.cuda.get_device_properties(0)
    return {'device_name': props.name, 'total_memory_bytes': props.total_memory,
            'cuda_version': torch.version.cuda, 'torch_version': torch.__version__,
            'compute_capability': list(capability), 'native_bf16': True,
            'visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
            'tf32_matmul': False, 'tf32_cudnn': False}


def resolve_clip_config(cache_root, local_files_only):
    """Only CLIP config/preprocessor files; the fine-tuned weights come from the checkpoint."""  # DEEPSTACK
    from huggingface_hub import snapshot_download
    from cloud_prep.measurement_v13.model_extensions import DEEPSTACK7B
    path = Path(snapshot_download(repo_id=DEEPSTACK7B['clip_repository'], revision=DEEPSTACK7B['clip_revision'],
        cache_dir=str(Path(cache_root) / 'hub'), local_files_only=local_files_only,
        allow_patterns=['config.json', 'preprocessor_config.json']))
    if any(path.glob('*.safetensors')) or any(path.glob('*.bin')):
        raise RuntimeError('Standalone CLIP weights are present in the config root; refuse to mix them in')
    return {'repository': DEEPSTACK7B['clip_repository'], 'revision': DEEPSTACK7B['clip_revision'],
            'path': str(path), 'files': {n: sha256(path / n) for n in ('config.json', 'preprocessor_config.json')},
            'weights_used': False}


def make_adapter(snapshot, clip, mode, design_module):  # DEEPSTACK
    from cloud_prep.measurement_v13.deepstack_adapter import load
    from cloud_prep.measurement_v13.deepstack_core import DeepStackCoreAdapter, DeepStackDoseAdapter
    base = DeepStackCoreAdapter(load(snapshot['path'], clip['path'], device='cuda', attention='sdpa'))
    adapter = base
    if MODES[mode][1]:
        adapter = DeepStackDoseAdapter(base, design_module, natural=mode == 'p1')
        adapter.model_revision = snapshot['revision']
    return adapter, base.metadata


def tokenize_native(tokenizer, variant, prompt):
    """Complete candidates in the native Vicuna answer format (" " + label)."""  # DEEPSTACK
    from deepstack.scoring import TokenizedOptions, tokenize_options
    from cloud_prep.measurement_v13.deepstack_core import SCORE_SEPARATOR
    encoded = tokenize_options(tokenizer, [SCORE_SEPARATOR + c for c in variant['continuations']], prompt_text=prompt)
    return TokenizedOptions(tuple(variant['labels']), encoded.token_ids, encoded.prompt_sha256)


def native_generate(model, inputs, config, budget, *, stopping_criteria=None):
    """Official DeepstackLlamaForCausalLM.generate (inputs_embeds); returns new tokens only."""  # DEEPSTACK
    from cloud_prep.measurement_v8.generation import greedy_kwargs
    forbidden = ('past_key_values', 'cache_position', 'position_ids', 'inputs_embeds', 'labels')
    if any(inputs.get(key) is not None for key in forbidden):
        raise ValueError('Generation requires fresh processor inputs, without scorer or KV state')
    kwargs = {key: value for key, value in inputs.items() if key != 'input_ids'}
    try:
        sequence = model.generate(inputs=inputs['input_ids'], **kwargs, generation_config=config,
                                  stopping_criteria=stopping_criteria, **greedy_kwargs(model))
    finally:
        model.kwargs_for_generation = None
    if sequence.ndim != 2 or sequence.shape[0] != 1 or sequence.shape[1] > budget:
        raise ValueError('Official DeepStack generate returned prompt tokens or exceeded the new-token budget')
    return sequence


def calibrate_condition(adapter, design, sources, row, anchor_rows, condition, max_sequence, cache):
    """One fixed changed/identity/direct prompt per visual intervention and context (v11-v2)."""
    anchor = anchor_rows[str(row.get('intervention_id', row['id']))]
    def one(c):
        key = design.calibration_key(row, c)
        if key in cache:
            found = cache[key]
            for field, expected in (('visual_signature', lambda: design.visual_signature(anchor, c)),
                    ('context_signature', lambda: design.context_signature(anchor, c)),
                    ('anchor_query_sha256', lambda: design.anchor_query_sha256(anchor))):
                if field in found and found[field] != expected():
                    raise ValueError('Calibration cache context/anchor binding changed: ' + field)
            return found
        variant = make_variant(anchor, 'direct', 'none')
        inputs, prompt = adapter.prepare(anchor, variant, c['main'])
        if adapter.expanded_length(inputs) > max_sequence:  # DEEPSTACK: expanded positions
            raise ValueError('Calibration anchor exceeds context bound')
        def forward_once():
            output = adapter.model(**inputs, use_cache=False)
            del output
        found = adapter.calibrate(sources, c, anchor, inputs, forward_once,
            anchor_prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest())
        json.dumps(found, allow_nan=False)
        cache[key] = found
        return found
    calibration = one(condition)
    if condition.get('target_policy') == 'opposite_native':
        return {**calibration, 'opposite_calibration': one(design.opposite_condition(condition))}
    return calibration


def unmatched_dose_records(row, variant, condition, calibration, report, *, model_key, planned, record_fields,
                           execution_schema, provenance=None):
    """A protocol outcome, never a numerical observation or a clipped matched dose (v11-v2)."""
    dose = {'condition': condition['name'], 'calibration_id': condition['calibration_id'],
        'donor_assignment': condition['donor_assignment'], 'changed_entry': condition['changed_entry'],
        'dose_mode': condition['dose_mode'], 'dose_value': condition['dose_value'],
        'target_policy': condition.get('target_policy', 'fixed'), 'dose_metric': condition.get('dose_metric', 'rss'),
        'effective_k': None, 'matching': report,
        'anchor_query_id': calibration['anchor_query_id'],
        'anchor_prompt_sha256': calibration['anchor_prompt_sha256'],
        'visual_signature': calibration['visual_signature'],
        'context_signature': calibration.get('context_signature'),
        'anchor_query_sha256': calibration.get('anchor_query_sha256'),
        'model_revision': calibration.get('model_revision'),
        **{field: report.get(field) for field in ('source_calibration_id', 'target_calibration_id',
            'source_anchor_query_id', 'target_anchor_query_id', 'source_anchor_query_sha256', 'target_anchor_query_sha256',
            'source_anchor_prompt_sha256', 'target_anchor_prompt_sha256', 'source_visual_signature', 'target_visual_signature',
            'source_context_signature', 'target_context_signature', 'source_model_revision', 'target_model_revision',
            'native_self_by_metric', 'target_other_native_by_metric', 'target_provenance', 'resolved_target',
            'requested_k', 'reachability')},
        'realized_baseline_state_by_metric': {
            'rss': report.get('realized_baseline_state_group_ratio'),
            'l1': report.get('realized_baseline_state_entry_ratio_l1')}}
    record = {**record_fields(row, variant, condition, None),
        **direction_readout(row, variant), **(provenance or {}),
        'execution_schema': execution_schema, 'arm': 'dose', 'kind': 'score',
        'model_key': model_key, 'protocol_status': 'unmatched_dose',
        'scores': {}, 'prediction': None, 'b_minus_a_log_odds': None,
        'forward_count': 0, 'generation_planned': planned, 'dose_metadata': dose,
        'dose_calibration': calibration, 'visual_intervention': dose,
        'visual_delivery': None, 'actual_image_forward': False, 'visual_output_replaced': False}
    generated = None
    if planned:
        generated = {**record, 'kind': 'generation', 'scorable': False, 'truncated': False,
            'generated_token_ids': [], 'generated_text': '', 'generated_text_raw': '',
            'generated_tokens': 0, 'parse': {'status': 'not_run', 'label': None},
            'eos_reached': False, 'stop_reason': 'not_run_unmatched_dose'}
    return record, generated


def compare_cached_score(cached, fresh, tolerance):
    if cached['visual_sha256'] != fresh['visual_sha256']:
        raise ValueError('Reused baseline visual tensors differ')
    a, b = cached['result'], fresh['result']
    if a['token_ids'] != b['token_ids'] or a['prediction'] != b['prediction'] or any(
            abs(a['scores'][key] - b['scores'][key]) > tolerance for key in a['scores']):
        raise ValueError('Reused baseline score differs from fresh evaluation')
    return {'visual_tensors_equal': True, 'prediction_equal': True,
        'max_abs_score_difference': max(abs(a['scores'][key] - b['scores'][key]) for key in a['scores'])}


def compare_cached_generation(cached, fresh):
    if cached['visual_sha256'] != fresh['visual_sha256'] or cached['generated_token_ids'] != fresh['generated_token_ids']:
        raise ValueError('Reused baseline generation differs from fresh evaluation')
    return {'visual_tensors_equal': True, 'generated_token_ids_equal': True}


def official_source_identity():  # DEEPSTACK
    import llava
    root = Path(llava.__file__).resolve().parent
    files = {name: sha256(root / name) for name in OFFICIAL_SOURCES}
    head = None
    git = root.parent / '.git'
    if (git / 'HEAD').is_file():
        ref = (git / 'HEAD').read_text().strip()
        head = (git / ref[5:]).read_text().strip() if ref.startswith('ref: ') and (git / ref[5:]).is_file() else ref
    return {'root': str(root), 'git_head': head, 'sha256': files}


def execute(a, rows, plan, design, *, mode, record_fields, dose_design=None):
    """design: the planner facade (p1_design / p2_design.ExecutionDesign); dose_design: module for the adapter."""
    execution_schema, dose_loop, v10_records = MODES[mode]
    arm_label = 'dose' if dose_loop else a.arm
    runtime = prepare_runtime(a)
    a.output.mkdir(parents=True, exist_ok=False)
    meta = {**plan, 'runtime': runtime, 'started_at': utc_now(),
            'arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
            'parser': {'label': PARSER_VERSION, 'direct': 'strict_content_v6'},
            'executor_schema': EXECUTOR_SCHEMA, 'executor_mode': mode,
            'source_sha256': {name: sha256(ROOT / name) for name in REUSED_SOURCES},
            'platform': platform.platform(), 'python': sys.version, 'completed_score_rows': 0,
            'completed_generation_rows': 0, 'completed_anchor_forwards': 0, 'unmatched_dose_score_rows': 0,
            'executed_score_evaluations': 0, 'executed_score_forward_calls': 0,
            'executed_generation_evaluations': 0, 'reused_score_rows': 0, 'reused_generation_rows': 0,
            'dedup_score_audit_evaluations': 0, 'dedup_generation_audit_evaluations': 0,
            'unmatched_dose_generation_rows': 0, 'initial_swap_mb': swap_used_mb()}
    write_json(a.output / 'metadata.json', meta)
    (a.output / 'failures.jsonl').touch()
    active = None
    try:
        import torch
        from transformers import StoppingCriteria, StoppingCriteriaList
        from deepstack.scoring import score_options, OptionScores
        from deepstack.readout_v3 import fp32_output_head
        from cloud_prep.measurement_v8.generation import resolve_greedy_config
        from scripts.prepare_readout_controls import generation_stop
        from cloud_prep.measurement_v13.deepstack_core import SCORE_SEPARATOR, SCORE_SEPARATOR_RULE, VISUAL_TOKENS
        meta['cuda'] = validate_cuda(torch)
        torch.set_num_threads(4); torch.manual_seed(a.seed)
        snapshot = resolve_snapshot(plan['model'], a.cache_root, a.local_files_only)
        clip = resolve_clip_config(a.cache_root, a.local_files_only)
        began = time.monotonic()
        adapter, adapter_meta = make_adapter(snapshot, clip, mode, dose_design)
        model, processor = adapter.model, adapter.processor
        context_bound = min(a.max_sequence, adapter.max_context)
        meta.update(model=snapshot, clip_config=clip, adapter=adapter_meta, load_seconds=time.monotonic() - began,
            parameter_count=sum(p.numel() for p in model.parameters()),
            libraries={n: importlib.metadata.version(n) for n in ('torch', 'transformers', 'tokenizers', 'Pillow',
                                                                  'accelerate', 'safetensors', 'llava')},
            official_loader_source=official_source_identity(),
            image_processor=processor.image_processor.to_dict(),
            deepstack_contract={'score_continuation_separator': SCORE_SEPARATOR, 'score_separator_rule': SCORE_SEPARATOR_RULE,
                'context_bound_expanded_tokens': context_bound, 'visual_tokens_per_entry': VISUAL_TOKENS,
                'generation_return': 'official inputs_embeds generate returns only new tokens; verified per call',
                'conversation_template': 'vicuna_v1 (official default); image placeholder first, then the question'})
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
        calibration_cache = {}
        write_json(a.output / 'metadata.json', meta)
        with (a.output / 'results.jsonl').open('w', buffering=1) as out, \
             (a.output / 'generations.jsonl').open('w', buffering=1) as gout, \
             (a.output / 'audits.jsonl').open('w', buffering=1) as audits, \
             torch.inference_mode(), fp32_output_head(model):
            requested = sorted(design.required_sources(plan['conditions']) if dose_loop else required_sources(plan['conditions']))
            anchor_rows = design.canonical_anchors(rows) if dose_loop else {}
            for row in rows:
                active = row['id']
                identity, source_contract = cache_identity(row)
                audit_group = (row['attribute'], str(row.get('geometry')), row.get('panel_family'))
                audit_interventions.setdefault(audit_group, identity)
                audit = a.panel == 'smoke' or audit_interventions[audit_group] == identity
                reused_for_query = identity == cached_identity
                if not reused_for_query:
                    sources, pixel_hashes = {}, {}
                    calibration_cache = {}
                    cached_identity = identity
                for readout in plan['readouts']:
                    evaluation_cache = ScalarEvaluationCache()
                    variant = make_variant(row, readout['response_format'], readout['mapping'])
                    inputs_by_source, prompt = {}, None
                    for name in requested:
                        inp, text = adapter.prepare(row, variant, name)
                        if prompt is not None and (text != prompt or not torch.equal(inp['input_ids'], next(iter(inputs_by_source.values()))['input_ids'])):
                            raise ValueError('Images changed prompt or token geometry')
                        prompt = text; inputs_by_source[name] = inp
                        current_hash = tensor_digest(adapter.pixel_digest_inputs(inp), torch)  # DEEPSTACK
                        if name not in sources:
                            sources[name] = adapter.encode(inp); pixel_hashes[name] = current_hash
                        elif pixel_hashes[name] != current_hash:
                            raise ValueError('Readout changed processed source pixels')
                    options = tokenize_native(processor.tokenizer, variant, prompt)  # DEEPSTACK
                    first_input = next(iter(inputs_by_source.values()))
                    seq_len = first_input['input_ids'].shape[1]
                    expanded_len = adapter.expanded_length(first_input)  # DEEPSTACK
                    input_ids_hash = hashlib.sha256(json.dumps(first_input['input_ids'][0].detach().cpu().tolist(), separators=(',', ':')).encode()).hexdigest()
                    if expanded_len + max(max(map(len, options.token_ids)), a.generate_tokens) > context_bound:
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
                            diff = max(abs(native.scores[k] - patched.scores[k]) for k in native.scores)
                            self_diffs[name] = diff
                            if diff > a.audit_tolerance:
                                raise ValueError(f'Native/self-patch score mismatch: {name}: {diff}')
                    for c in plan['conditions']:
                        began = time.monotonic(); inp = inputs_by_source[c['main']]
                        native_source = c['main'] if is_native(c) else None
                        native_index = row['source_answer_indices'].get(native_source) if native_source else None
                        gold = variant['original_to_label'][row['option_labels'][native_index]] if native_index is not None else None
                        calibration = (calibrate_condition(adapter, design, sources, row, anchor_rows,
                            c, context_bound, calibration_cache) if dose_loop and c['role'] == 'dose_pair' else None)
                        planned = bool(a.generate_tokens and design.generation_selected(row, a.panel, plan['design']) and c['generation'])
                        if dose_loop:
                            from cloud_prep.measurement_v12.v11_v2_dose_adapter import UnmatchedDose
                            try:
                                visual = sources[c['main']] if native_source else adapter.compose(
                                    sources, c, row, inputs=inp, calibration=calibration)
                            except UnmatchedDose as exc:
                                record, generated = unmatched_dose_records(row, variant, c, calibration, exc.report,
                                    model_key=a.model, planned=planned, record_fields=record_fields,
                                    execution_schema=execution_schema, provenance={
                                        'visual_intervention_id': identity,
                                        'visual_intervention_signature': dose_visual_signature(row, c, pixel_hashes, calibration=calibration),
                                        'source_contract_sha256': source_contract, 'source_images': row['source_images'],
                                        'source_pixel_sha256': pixel_hashes,
                                        'source_image_sha256': {k: v['sha256'] for k, v in row['source_image_metadata'].items()},
                                        'sequence_length': seq_len, 'expanded_sequence_length': expanded_len,
                                        'prompt_sha256': options.prompt_sha256,
                                        'input_ids_sha256': input_ids_hash, 'elapsed_seconds': time.monotonic() - began})
                                out.write(json.dumps(record) + '\n')
                                meta['completed_score_rows'] += 1; meta['unmatched_dose_score_rows'] += 1
                                if generated is not None:
                                    gout.write(json.dumps(generated) + '\n')
                                    meta['completed_generation_rows'] += 1; meta['unmatched_dose_generation_rows'] += 1
                                check_swap(meta['initial_swap_mb'])
                                continue
                        else:
                            visual = sources[c['main']] if native_source else adapter.compose(sources, c, row, inputs=inp)
                        signature = (dose_visual_signature(row, c, pixel_hashes, calibration=calibration) if dose_loop
                                     else v10_visual_signature(row, c, pixel_hashes))
                        visual_sha = tensor_digest(visual, torch) if audit else None
                        if audit:
                            digest_key = (identity, c['name'])
                            prior = visual_digests.setdefault(digest_key, (signature, visual_sha))
                            if prior != (signature, visual_sha):
                                raise ValueError('Shared intervention changed across query/readout')
                        reuse_key = (baseline_evaluation_key(c, calibration=calibration,
                            input_ids_sha256=input_ids_hash, options=options.to_dict()) if dose_loop else None)
                        def score_once():
                            with (nullcontext(None) if native_source else adapter.use(visual, inp, audit=audit)) as delivery:
                                scored = score_options(model, inp, options, correct_label=gold)
                            if delivery is not None and delivery['calls'] != scored.forward_count:
                                raise ValueError('Replacement missed a fresh teacher-forced forward')
                            meta['executed_score_evaluations'] += 1
                            meta['executed_score_forward_calls'] += scored.forward_count
                            return {'result': scored.to_dict(), 'delivery': delivery, 'visual_sha256': visual_sha}
                        score_value, score_reuse = evaluation_cache.evaluate('score', reuse_key, c['name'], score_once,
                            (lambda old, new: compare_cached_score(old, new, a.audit_tolerance)) if audit else None)
                        result = OptionScores(**score_value['result'])
                        delivered = None if score_reuse['reused'] else score_value['delivery']
                        meta['reused_score_rows'] += int(score_reuse['reused'])
                        meta['dedup_score_audit_evaluations'] += int(score_reuse['fresh_audit'])
                        composition_self_difference = None
                        if c['role'] == 'self_patch':
                            if c['main'] not in native_scores:
                                native_scores[c['main']] = score_options(model, inp, options).scores
                            composition_self_difference = max(abs(result.scores[k] - native_scores[c['main']][k]) for k in result.scores)
                            if composition_self_difference > a.audit_tolerance:
                                raise ValueError('ROI/group compose self-patch differs from native scores')
                        record = {**record_fields(row, variant, c, result.prediction),
                            'execution_schema': execution_schema, 'kind': 'score', 'arm': arm_label, **result.to_dict(),
                            'forward_count': 0 if score_reuse['reused'] else result.forward_count,
                            'reference_forward_count': result.forward_count, 'evaluation_reuse': score_reuse,
                            'model_key': a.model, **direction_readout(row, variant, result.scores),
                            'visual_intervention_id': identity, 'visual_intervention_signature': signature,
                            'dose_calibration': calibration,
                            'dose_metadata': getattr(visual, 'metadata', None) if dose_loop else None,
                            'protocol_status': 'executed', 'v10_later_bridge': None,
                            'visual_tensor_sha256': visual_sha, 'source_contract_sha256': source_contract,
                            'shared_visual_cache_reused_for_query': reused_for_query,
                            'native_source': native_source, 'native_answer_label': gold,
                            'native_correct': result.prediction == gold if gold else None,
                            'generation_planned': planned, 'sequence_length': seq_len,
                            'expanded_sequence_length': expanded_len,
                            'candidate_token_lengths': {k: len(v) for k, v in zip(options.labels, options.token_ids)},
                            'score_normalization': 'sum_sequence_logprob_no_length_normalization',
                            'score_continuation_separator': SCORE_SEPARATOR,
                            'source_images': row['source_images'], 'source_pixel_sha256': pixel_hashes,
                            'source_pixel_sha256_scheme': 'deepstack_tensor_digest_views_size_native_grid',
                            'source_image_sha256': {k: v['sha256'] for k, v in row['source_image_metadata'].items()},
                            'prompt_sha256': options.prompt_sha256, 'selfidentity_max_score_difference': self_diffs.get(c['main']),
                            'input_ids_sha256': input_ids_hash,
                            'composition_self_max_score_difference': composition_self_difference,
                            'actual_image_forward': bool(native_source) and not score_reuse['reused'],
                            'visual_output_replaced': not bool(native_source) and not score_reuse['reused'],
                            'actual_pixel_input_source': c['main'], 'visual_delivery': delivered,
                            'visual_intervention': getattr(visual, 'metadata', getattr(visual, 'intervention', {})),
                            'elapsed_seconds': time.monotonic() - began, 'memory': memory(torch)}
                        if v10_records:
                            # v10 executor record form for P2 local/full (no reuse/dose fields).
                            for key in V11_ONLY_SCORE_FIELDS:
                                record.pop(key)
                        out.write(json.dumps(record) + '\n'); meta['completed_score_rows'] += 1
                        if planned:
                            def generate_once():
                                guard = SwapGuard()
                                with (nullcontext(None) if native_source else adapter.use(visual, inp, audit=audit)) as delivery:
                                    sequence = native_generate(model, inp, config, a.generate_tokens,
                                                               stopping_criteria=StoppingCriteriaList([guard]))  # DEEPSTACK
                                if delivery is not None and delivery['calls'] != 1:
                                    raise ValueError('Fresh cached generation must consume visual features exactly once')
                                ids = sequence[0].detach().cpu().tolist()  # DEEPSTACK: only new tokens are returned
                                del sequence
                                decoded = processor.tokenizer.decode(ids, skip_special_tokens=True)
                                parsed = parse_response(decoded, variant)
                                stop = generation_stop(ids, eos_ids, a.generate_tokens)
                                scorable = parsed['status'] == 'parsed' and stop['eos_reached'] and not guard.error
                                if guard.error: raise RuntimeError(guard.error)
                                meta['executed_generation_evaluations'] += 1
                                return {'generated_token_ids': ids, 'generated_text': decoded,
                                    'generated_text_raw': processor.tokenizer.decode(ids, skip_special_tokens=False),
                                    'generated_tokens': len(ids), 'parse': parsed, 'scorable': scorable,
                                    'prediction': parsed['label'] if scorable else None, **stop,
                                    'resource_guard_error': guard.error, 'delivery': delivery, 'visual_sha256': visual_sha}
                            gen_value, gen_reuse = evaluation_cache.evaluate('generation', reuse_key, c['name'], generate_once,
                                compare_cached_generation if audit else None)
                            prediction, scorable = gen_value['prediction'], gen_value['scorable']
                            delivery = None if gen_reuse['reused'] else gen_value['delivery']
                            meta['reused_generation_rows'] += int(gen_reuse['reused'])
                            meta['dedup_generation_audit_evaluations'] += int(gen_reuse['fresh_audit'])
                            generated = {k: v for k, v in record.items() if k not in result.to_dict() and k != 'reference_forward_count'}
                            generated.update(record_fields(row, variant, c, prediction))
                            generated.update({k: v for k, v in gen_value.items() if k not in ('delivery', 'visual_sha256')})
                            generated.update(kind='generation',
                                max_new_tokens=a.generate_tokens, eos_token_ids=eos_ids,
                                evaluation_reuse=gen_reuse,
                                actual_image_forward=bool(native_source) and not gen_reuse['reused'],
                                visual_output_replaced=not bool(native_source) and not gen_reuse['reused'],
                                native_correct=prediction == gold if scorable and gold else None,
                                generation_return_contract='official_inputs_embeds_generate_new_tokens_only',
                                visual_delivery=delivery, memory=memory(torch))
                            if v10_records:
                                generated.pop('evaluation_reuse')
                            gout.write(json.dumps(generated) + '\n'); meta['completed_generation_rows'] += 1
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
                if dose_loop:
                    meta['completed_anchor_forwards'] = sum(v['fresh_forwards'] for v in adapter.cache.values())
                torch.cuda.empty_cache(); write_json(a.output / 'metadata.json', meta)
                print(json.dumps({'event': 'item_complete', 'id': active, 'score_rows': meta['completed_score_rows'],
                                  'generation_rows': meta['completed_generation_rows']}), flush=True)
        if any(meta['completed_' + k + '_rows'] != meta['expected_' + k + '_rows'] for k in ('score', 'generation')):
            raise ValueError('Output matrix incomplete')
        if meta['completed_anchor_forwards'] != plan.get('expected_anchor_forwards', 0):
            raise ValueError('Fixed-anchor calibration matrix incomplete')
        meta.update(finished_at=utc_now(), memory_at_end=memory(torch)); write_json(a.output / 'metadata.json', meta)
        return 0
    except BaseException as e:
        # Record the failure first so a metadata write error cannot erase the evidence.
        with (a.output / 'failures.jsonl').open('a') as f:
            f.write(json.dumps({'id': active, 'error': repr(e), 'at': utc_now()}) + '\n')
        meta.update(failed_at=utc_now(), error=repr(e)); write_json(a.output / 'metadata.json', meta)
        raise
