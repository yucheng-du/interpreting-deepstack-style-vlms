"""Batched evaluation of the physically distinct conditions of one query/readout.

Configs with ``batch_conditions: true`` use a separate execution_version because
batching can change values at BF16 precision. Other configs use the batch-one runner.

Contract
--------
All conditions of one row/readout share identical input_ids, prompt and processed
geometry (the runner enforces this before calling in); only the injected visual tensors
differ. After deduplication by full visual hash, the remaining distinct conditions are
evaluated as ONE batched score (one batched forward when all labels are single tokens,
otherwise one batched teacher-forced forward per label) and ONE batched greedy
generation, chunked at ``max_batch`` rows. No padding is ever introduced.

Row order: batch row i is the i-th entry of the list given to the family module
(``stack_inputs(adapter, [inputs...])`` and ``use_batch(adapter, [visuals...], inputs_n)``
receive the same list order). The family modules (batched_qwen / batched_ibm /
batched_deepstack) document and enforce how row i receives visuals[i] (row-major
masked_scatter of image placeholders, per-row later-entry injection). This module reads
logits row i / generated sequence row i back to that same list position and never
reorders rows; results are zipped back to the condition entries that built the list.

Per-row semantics equal the batch-one deepstack.scoring.score_options /
runner.generate: sum of label-token log-probabilities from the fp32 output head (the
runner's fp32_output_head context), no EOS, ties to the first legal label; greedy
generation with the same generation config, cut after the row's first EOS (inclusive)
with the generate() pad tokens after it dropped; rows without EOS keep every token.

Records: one observation per logical condition with the batch-one schema plus
``elapsed_basis`` and ``batch_elapsed_seconds``. An evaluated row's
``result.delivery`` is the batch delivery stats (with only this row's entry of a
family's per-row ``stats['rows']``) plus batch_id/batch_index/batch_members; a reused
row gets a pointer stub instead of a copy. With
``cross_query_cache`` raw model outputs are also reused within one unit across
interventions when (kind, prompt, readout, label mapping, input ids, geometry, full
visual hash) are identical; row-dependent outcome fields and the parse are recomputed.

Audits (first block of each task): one batched NATIVE forward (or greedy generation)
over the stacked per-source inputs versus one batched SELF-PATCH forward through
use_batch, per-row max |difference| <= self_patch_tolerance (generation: equal token
lists). The self-patch visuals are the family's optional ``encode_batch`` output (the
same batched vision call the native forward makes) when the family module offers it,
because a vision tower need not be batch-invariant in BF16; the batch-one source cache
that the experimental rows use is then compared with it and the difference recorded.
Without ``encode_batch`` the sources' own batch-one encodes are self-patched.

No torch import at module import.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib
import json
import math
import time

from cloud_prep.measurement_v15 import design

BATCH_SCHEMA = 'v14-batched-conditions-1'
FAMILIES = ('qwen', 'ibm', 'deepstack')
ELAPSED_BATCH = 'batch_apportioned'
ELAPSED_REUSED = 'reused_not_evaluated'
REUSE_SAME_QUERY = 'same_query_readout'
REUSE_CROSS_QUERY = 'cross_query_within_unit'
AUDIT_BATCHED_ENCODE = 'encode_batch_same_batched_vision_call'
AUDIT_SOURCE_CACHE = 'batch_one_source_cache'
_FORBIDDEN = ('past_key_values', 'cache_position', 'position_ids', 'inputs_embeds', 'labels', 'logits_to_keep')
_STOP_FIELDS = ('eos_reached', 'truncated', 'stop_reason')


def family_module(family):
    """Lazily import cloud_prep.measurement_v15.batched_<family> (common interface)."""
    if family not in FAMILIES:
        raise ValueError(f'Unknown batched family {family!r}')
    module = importlib.import_module(f'cloud_prep.measurement_v15.batched_{family}')
    for name in ('stack_inputs', 'use_batch'):
        if not callable(getattr(module, name, None)):
            raise TypeError(f'batched_{family} lacks the common interface function {name}')
    return module


def chunks(items, size):
    items = list(items)
    if type(size) is not int or size < 1:
        raise ValueError('Positive integer batch size required')
    return [items[i:i + size] for i in range(0, len(items), size)]


def apportion(elapsed, rows):
    """Equal share of one batch's wall time per row evaluated in that batch."""
    if rows < 1 or not math.isfinite(elapsed) or elapsed < 0:
        raise ValueError('Finite nonnegative batch time over at least one row required')
    return elapsed / rows


def eos_ids(genconfig):
    eos = genconfig.eos_token_id
    return [eos] if isinstance(eos, int) else list(eos or [])


def effective_pad(genconfig):
    """The pad id generate() writes after a finished row (HF falls back to eos[0])."""
    if genconfig.pad_token_id is not None:
        return int(genconfig.pad_token_id)
    eos = eos_ids(genconfig)
    return int(eos[0]) if eos else None


def cut_at_first_eos(tokens, eos, pad):
    """Batch-one token list of one batched row: keep through the first EOS, drop pads.

    generate() replaces every token after a row's first EOS by ``pad``; anything else
    after it would mean the row kept decoding and is an error, not silently dropped.
    """
    tokens = [int(t) for t in tokens]
    eos = {int(t) for t in eos}
    for index, token in enumerate(tokens):
        if token in eos:
            if any(t != pad for t in tokens[index + 1:]):
                raise ValueError('Batched generation continued after EOS without pad tokens')
            return tokens[:index + 1]
    return tokens


def _check_rows(model, inputs_n):
    """Batch-one score checks, per batch: eval mode, fresh unpadded identical rows."""
    import torch
    if getattr(model, 'training', False):
        raise ValueError('Call model.eval() before deterministic option scoring')
    ids = inputs_n.get('input_ids')
    if ids is None or ids.ndim != 2 or ids.shape[0] < 1 or ids.shape[1] < 1:
        raise ValueError('Expected nonempty input_ids of shape [rows, prompt_tokens]')
    for name in _FORBIDDEN:
        if inputs_n.get(name) is not None:
            raise ValueError(f'Unsupported batched input {name!r}; supply uncached processor prompts')
    if not torch.equal(ids, ids[:1].expand_as(ids)):
        raise ValueError('Batched conditions must share identical input_ids (no padding)')
    mask = inputs_n.get('attention_mask')
    if mask is not None and (mask.shape != ids.shape or not bool(torch.all(mask == 1).item())):
        raise ValueError('Batched evaluation requires an unpadded all-ones attention mask')
    return int(ids.shape[0])


def _batch_logits(output, rows):
    logits = output.logits if hasattr(output, 'logits') else output['logits']
    if logits.ndim != 3 or logits.shape[0] != rows:
        raise ValueError('Expected batched causal logits [rows, sequence, vocabulary]')
    return logits


def _continuation_batch(inputs_n, prefix_tokens):
    """deepstack.scoring._continuation_inputs for identical rows: same tokens in every row."""
    import torch
    result = dict(inputs_n)
    if not prefix_tokens:
        return result
    ids = inputs_n['input_ids']
    rows, extra = ids.shape[0], len(prefix_tokens)
    result['input_ids'] = torch.cat((ids, ids.new_tensor([list(prefix_tokens)]).expand(rows, extra)), dim=-1)
    mask = inputs_n.get('attention_mask')
    if mask is not None:
        result['attention_mask'] = torch.cat((mask, mask.new_ones((rows, extra))), dim=-1)
    for name in ('token_type_ids', 'mm_token_type_ids'):
        if name in inputs_n:
            values = inputs_n[name]
            result[name] = torch.cat((values, values.new_zeros((rows, extra))), dim=-1)
    return result


def score_options_batch(model, inputs_n, options):
    """deepstack.scoring.score_options for every row of one unpadded batch.

    One batched forward for all-single-token labels, otherwise one batched teacher-forced
    forward of prompt + label[:-1] per label. Each row's reduction is the batch-one
    expression applied to that row's logits slice (fp32 log-softmax, label-token sum), so
    identical logits give identical scores. forward_count is the number of batched
    forwards, which is also the number of visual deliveries use_batch must count.
    Returns one OptionScores per row, in row order.
    """
    import torch
    from deepstack.scoring import OptionScores, _forward_kwargs
    rows = _check_rows(model, inputs_n)
    scores = [{} for _ in range(rows)]
    with torch.inference_mode():
        if options.all_single_token:
            logits_all = _batch_logits(model(**_forward_kwargs(model, inputs_n, 1)), rows)
            targets = torch.tensor([ids[0] for ids in options.token_ids], device=logits_all.device)
            values = []
            for i in range(rows):
                logits = logits_all[i, -1].float()
                values.append(logits[targets] - torch.logsumexp(logits, dim=-1))
            for i, row in enumerate(torch.stack(values).detach().cpu().tolist()):
                scores[i].update(zip(options.labels, row))
            forward_count, method = 1, 'single_token_shared_forward'
        else:
            for label, ids in zip(options.labels, options.token_ids):
                label_inputs = _continuation_batch(inputs_n, ids[:-1])
                logits_all = _batch_logits(model(**_forward_kwargs(model, label_inputs, len(ids))), rows)
                targets = torch.tensor(ids, device=logits_all.device)
                values = []
                for i in range(rows):
                    logits = logits_all[i, -len(ids):].float()
                    if logits.shape[0] != len(ids):
                        raise ValueError('Model did not return enough continuation logits')
                    selected = logits.gather(-1, targets[:, None]).squeeze(-1)
                    values.append((selected - torch.logsumexp(logits, dim=-1)).sum())
                for i, value in enumerate(torch.stack(values).detach().cpu().tolist()):
                    scores[i][label] = float(value)
            forward_count, method = len(options.labels), 'complete_label_teacher_forcing'
    token_ids = {label: list(ids) for label, ids in zip(options.labels, options.token_ids)}
    results = []
    for row in scores:
        if not all(math.isfinite(value) for value in row.values()):
            raise FloatingPointError('Non-finite option score')
        # Ties use the first option in the declared legal-option order.
        prediction = max(options.labels, key=row.__getitem__)
        results.append(OptionScores(row, prediction, None, forward_count, method, deepcopy(token_ids)))
    return results


def generate_batch(model, inputs_n, genconfig, family, guard, *, stats=None):
    """Batched greedy decoding; returns per-row new-token id lists in row order.

    Same model.generate call as runner.generate (native_greedy for Qwen/Granite; the
    official DeepStack generate with kwargs_for_generation cleanup), same generation
    config, max_new_tokens and EOS ids. Each row is cut after its first EOS (inclusive);
    rows without EOS keep every token (generation_stop then reports max_new_tokens
    truncation). With ``stats`` (the use_batch delivery stats) exactly one visual
    prefill per batch is required.
    """
    import torch
    rows = _check_rows(model, inputs_n)
    if family not in FAMILIES:
        raise ValueError(f'Unknown family {family!r}')
    if family != 'deepstack':
        from cloud_prep.measurement_v8.generation import native_greedy
        sequence = native_greedy(model, inputs_n, genconfig, stopping_criteria=guard)
        prompt = inputs_n['input_ids']
        if (sequence.ndim != 2 or sequence.shape[0] != rows or sequence.shape[1] < prompt.shape[1]
                or not torch.equal(sequence[:, :prompt.shape[1]], prompt)):
            raise ValueError('Batched generation must return each prompt followed by its new tokens')
        new = sequence[:, prompt.shape[1]:]
    else:
        from cloud_prep.measurement_v8.generation import greedy_kwargs
        if any(inputs_n.get(k) is not None for k in ('past_key_values', 'cache_position', 'position_ids', 'inputs_embeds', 'labels')):
            raise ValueError('DeepStack generation requires fresh processor inputs')
        try:
            new = model.generate(inputs=inputs_n['input_ids'], **{k: v for k, v in inputs_n.items() if k != 'input_ids'},
                                 generation_config=genconfig, stopping_criteria=guard, **greedy_kwargs(model))
        finally:
            model.kwargs_for_generation = None
        if new.ndim != 2 or new.shape[0] != rows:
            raise ValueError('DeepStack must return new tokens only')
    if new.shape[1] > genconfig.max_new_tokens:
        raise ValueError('Generation exceeded max_new_tokens')
    if stats is not None and stats['calls'] != 1:
        raise ValueError('Generation needs exactly one fresh visual prefill per batch')
    eos, pad = eos_ids(genconfig), effective_pad(genconfig)
    return [cut_at_first_eos(tokens, eos, pad) for tokens in new.detach().cpu().tolist()]


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()


def visual_max_abs_difference(left, right, family):
    """Largest elementwise |difference| over all entry tensors of two family visuals."""
    from cloud_prep.measurement_v15.adapter import tensors
    a, b = tensors(left, family), tensors(right, family)
    if set(a) != set(b) or any(tuple(a[k].shape) != tuple(b[k].shape) for k in a):
        raise ValueError('Visual entries/shapes differ')
    return max(float((a[k].float() - b[k].float()).abs().max().item()) for k in a)


def label_mapping(variant, options):
    """Everything about the labels that the model or the parser sees for this readout."""
    return {name: variant.get(name) for name in ('response_format', 'mapping', 'labels', 'continuations',
                                                  'label_to_original', 'assistant_prefix')} | {
        'option_token_ids': [list(ids) for ids in options.token_ids] if options is not None else None}


def reuse_key(kind, prompt_sha256, readout, mapping_sha256, input_ids_sha256, geometry, visual_sha256):
    """Identity of one raw model evaluation. Within one row/readout everything but
    (kind, visual hash) is constant, which reduces to the batch-one reuse key."""
    return (kind, prompt_sha256, _digest(readout), mapping_sha256, input_ids_sha256, _digest(geometry), visual_sha256)


def row_delivery(stats, index, rows):
    """One batch row's delivery record: the shared batch stats, plus only this row's
    entry of a family's per-row ``stats['rows']`` list (if the family keeps one)."""
    per_row = stats.get('rows')
    if per_row is not None and (not isinstance(per_row, list) or len(per_row) != rows):
        raise ValueError('Delivery stats rows must hold one record per batch row')
    delivery = deepcopy({k: v for k, v in stats.items() if k != 'rows'})
    if per_row is not None:
        delivery['row'] = deepcopy(per_row[index])
    return delivery


def reuse_stub(source, row, readout):
    """Delivery pointer for a reused row (never a copy of the source row's delivery)."""
    same = source['id'] == row['id'] and source['readout'] == readout
    return {'reused_from_observation_id': source['observation_id'], 'reused_from_condition': source['condition'],
            'reused_from_id': source['id'], 'reused_from_intervention_id': source['intervention_id'],
            'reuse_scope': REUSE_SAME_QUERY if same else REUSE_CROSS_QUERY, 'calls': 0}


class ConditionBatcher:
    """Per-run batched evaluator, called by runner.execute_locked per row/readout.

    Dependencies are injected so CPU tests can use fake models/adapters/family modules.
    ``write_observation(record)`` must append the record and mark it done;
    ``write_audit(record)`` appends one audit record; ``make_guard()`` returns
    (guard_with_error_attribute_or_None, stopping_criteria_or_None).
    """

    def __init__(self, *, adapter, family, config, plan, model_key, run_id, meta, done, genconfig, eos,
                 tokenizer, write_observation, write_audit, make_guard, parse_response, generation_stop,
                 outcome_fields, family_ops=None, scorer=score_options_batch, generator=generate_batch,
                 clock=time.monotonic, visual_difference=None):
        if not config['batch_conditions']:
            raise ValueError('ConditionBatcher requires batch_conditions true')
        self.adapter, self.model, self.family = adapter, adapter.model, family
        self.config, self.plan, self.model_key, self.run_id = config, plan, model_key, run_id
        self.meta, self.done, self.genconfig, self.eos, self.tokenizer = meta, done, genconfig, list(eos), tokenizer
        self.write_observation, self.write_audit, self.make_guard = write_observation, write_audit, make_guard
        self.parse_response, self.generation_stop, self.outcome_fields = parse_response, generation_stop, outcome_fields
        self.family_ops = family_ops if family_ops is not None else family_module(family)
        self.scorer, self.generator, self.clock = scorer, generator, clock
        self.visual_difference = visual_difference or (lambda a, b: visual_max_abs_difference(a, b, family))
        self.max_batch = int(config['max_batch'])
        self.cross_query = bool(config['cross_query_cache'])
        self.unit_id, self.unit_cache, self.batches = None, {}, 0
        meta['batched_execution'] = {'schema': BATCH_SCHEMA, 'max_batch': self.max_batch,
            'cross_query_cache': self.cross_query, 'score_batches': 0, 'generation_batches': 0,
            'score_rows_evaluated': 0, 'generation_rows_evaluated': 0,
            'reused_same_query_readout': 0, 'reused_cross_query_within_unit': 0,
            'elapsed_basis': ELAPSED_BATCH, 'row_order': 'batch row i = i-th condition entry passed to the family module'}

    # ---- unit scope -----------------------------------------------------------------
    def start_unit(self, unit_id):
        """Cross-query reuse never crosses a unit (rows arrive sorted by unit)."""
        if unit_id != self.unit_id:
            self.unit_id, self.unit_cache = unit_id, {}

    # ---- one row/readout ------------------------------------------------------------
    def readout(self, *, row, readout, variant, prepared, prompt, options, visuals, pixels, sources,
                audit, geometry, fixed_later):
        if audit and options is not None:
            self.write_audit(self.score_audit(row, readout, prepared, options, sources, geometry, fixed_later))
        elif audit and self.config['generate_tokens']:
            self.write_audit(self.generation_audit(row, readout, prepared, sources, geometry, fixed_later))
        input_ids = next(iter(prepared.values()))['input_ids']
        context = dict(row=row, readout=readout, variant=variant, prepared=prepared, options=options,
            visuals=visuals, pixels=pixels, prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
            mapping_sha256=_digest(label_mapping(variant, options)),
            input_ids_sha256=_digest(input_ids.detach().cpu().tolist()), geometry=geometry)
        for kind in (['generation'] if options is None else ['score', 'generation']):
            if kind == 'generation' and not self.config['generate_tokens']:
                continue
            self.evaluate(kind, context)

    def plan_kind(self, kind, context):
        """Pending logical conditions in condition order, each evaluated or pointing to a source."""
        row, readout = context['row'], context['readout']
        scope = self.unit_cache if self.cross_query else {}
        fresh, entries, scheduled = {}, [], []
        for condition in design.conditions(row, self.config):
            key = design.observation_key(row, condition, readout, kind, self.plan['model'], self.config)
            if key in self.done:
                continue
            visual, vm = context['visuals'][condition['name']]
            rkey = reuse_key(kind, context['prompt_sha256'], readout, context['mapping_sha256'],
                             context['input_ids_sha256'], context['geometry'], vm['visual_sha256'])
            entry = dict(condition=condition, key=key, visual=visual, vm=vm,
                         inputs=context['prepared'][condition['main_source']])
            source = scope.get(rkey) or fresh.get(rkey)
            if source is None:
                source = dict(observation_id=key, condition=condition['name'], id=row['id'],
                              intervention_id=row['intervention_id'], readout=readout)
                fresh[rkey] = source
                entry['evaluated'] = True
                scheduled.append(entry)
            else:
                entry['evaluated'] = False
            entry['source'] = source
            entries.append(entry)
        return entries, scheduled, fresh

    def evaluate(self, kind, context):
        entries, scheduled, fresh = self.plan_kind(kind, context)
        for chunk in chunks(scheduled, self.max_batch):
            self.run_chunk(kind, chunk, context)
        for entry in entries:
            record = self.record(kind, entry, context)
            self.write_observation(record)
        if self.cross_query:
            self.unit_cache.update(fresh)
        return entries

    def run_chunk(self, kind, chunk, context):
        began = self.clock()
        inputs_n = self.family_ops.stack_inputs(self.adapter, [e['inputs'] for e in chunk])
        visuals = [e['visual'] for e in chunk]
        if kind == 'score':
            with self.family_ops.use_batch(self.adapter, visuals, inputs_n) as stats:
                scored = self.scorer(self.model, inputs_n, context['options'])
            if len(scored) != len(chunk):
                raise ValueError('Batched score returned a different number of rows')
            if any(stats['calls'] != s.forward_count for s in scored):
                raise ValueError('Score missed native visual delivery')
            raws = [s.to_dict() for s in scored]
        else:
            guard, criteria = self.make_guard()
            with self.family_ops.use_batch(self.adapter, visuals, inputs_n) as stats:
                tokens = self.generator(self.model, inputs_n, self.genconfig, self.family, criteria, stats=stats)
            if guard is not None and guard.error:
                raise RuntimeError(guard.error)
            if stats['calls'] != 1:
                raise ValueError('Generation needs exactly one fresh visual prefill')
            if len(tokens) != len(chunk):
                raise ValueError('Batched generation returned a different number of rows')
            raws = []
            for ids in tokens:
                raws.append(dict(generated_text=self.tokenizer.decode(ids, skip_special_tokens=True),
                    generated_text_raw=self.tokenizer.decode(ids, skip_special_tokens=False),
                    generated_token_ids=list(ids), **self.generation_stop(ids, self.eos, self.config['generate_tokens'])))
        if stats.get('batch_size') != len(chunk):
            raise ValueError('Visual delivery batch size differs from the evaluated batch')
        elapsed = self.clock() - began
        share = apportion(elapsed, len(chunk))
        self.batches += 1
        batch_id = f'{self.run_id}/{kind}/{self.batches:06d}'
        members = [e['condition']['name'] for e in chunk]
        for index, (entry, raw) in enumerate(zip(chunk, raws)):
            entry['source']['raw'] = raw
            entry.update(delivery={**row_delivery(stats, index, len(chunk)), 'batch_id': batch_id,
                                   'batch_index': index, 'batch_members': members},
                         elapsed=share, batch_elapsed=elapsed)
        stats_key = 'executed_score' if kind == 'score' else 'executed_generation'
        self.meta[stats_key] += len(chunk)
        self.meta['batched_execution'][kind + '_batches'] += 1
        self.meta['batched_execution'][kind + '_rows_evaluated'] += len(chunk)

    def result(self, kind, entry, context):
        row, variant, options = context['row'], context['variant'], context['options']
        raw = deepcopy(entry['source']['raw'])
        if entry['evaluated']:
            delivery = entry['delivery']
        else:
            delivery = reuse_stub(entry['source'], row, context['readout'])
            self.meta['batched_execution']['reused_' + delivery['reuse_scope']] += 1
        if kind == 'score':
            return {**raw, **self.outcome_fields(row, variant, raw['prediction'], raw['scores']),
                    'delivery': delivery, 'score_event': 'specified string continuation; no EOS',
                    'score_separator': ' ' if self.family == 'deepstack' else ''}
        stop = {name: raw[name] for name in _STOP_FIELDS}
        parsed = (self.parse_response(raw['generated_text'], variant) if options is not None
                  else {'status': 'freeform_retained', 'label': None})
        scorable = parsed['status'] == 'parsed' and stop['eos_reached']
        prediction = parsed['label'] if scorable else None
        return dict(generated_text=raw['generated_text'], generated_text_raw=raw['generated_text_raw'],
            generated_token_ids=raw['generated_token_ids'], parse=parsed,
            scorable=scorable if options is not None else None, prediction=prediction, **stop,
            **self.outcome_fields(row, variant, prediction), delivery=delivery,
            freeform_metric_status='official metric pending' if options is None else None)

    def record(self, kind, entry, context):
        row, c = context['row'], entry['condition']
        evaluated = entry['evaluated']
        return dict(schema=design.SCHEMA, observation_id=entry['key'], run_id=self.run_id,
            execution_version=self.config['execution_version'],
            id=row['id'], unit_id=row['unit_id'], intervention_id=row['intervention_id'], task=row['task'],
            query_role=row['query_role'], kind=kind, readout=context['readout'], condition=c, model_key=self.model_key,
            model_revision=self.plan['model']['revision'], query=row['query'], reference_answers=row.get('reference_answers'),
            material_provenance=row.get('source', {}), benchmark=row.get('benchmark'),
            source_metadata=row['source_image_metadata'], source_pixel_sha256=context['pixels'],
            prompt_sha256=context['prompt_sha256'], visual_metadata=entry['vm'],
            split=row.get('split'), material_set=row.get('material_set'),
            severity_spec=row.get('degradations', {}).get(c.get('severity_label')),
            result=self.result(kind, entry, context), evaluation_reused=not evaluated,
            elapsed_seconds=entry['elapsed'] if evaluated else 0.,
            elapsed_basis=ELAPSED_BATCH if evaluated else ELAPSED_REUSED,
            batch_elapsed_seconds=entry['batch_elapsed'] if evaluated else None)

    # ---- batched native/self-patch audits ---------------------------------------------
    def audit_visuals(self, names, inputs_n, sources, cache_differences):
        """Self-patch visuals for one batched audit chunk.

        A family vision tower need not be batch-invariant in BF16 (Qwen3-VL on 384x384
        natural images differs by up to 0.375 between batch-one and batched encodes). When
        the family module offers the optional ``encode_batch(adapter, inputs_n)`` (the same
        batched vision call the native forward makes, split per row), the self-patch uses
        it, so the exact native/self-patch comparison isolates per-row delivery. The
        batch-one cached sources (which the experimental rows use) are then compared with
        those batched encodes and the difference is recorded, not gated. Without
        ``encode_batch`` the sources' own batch-one encodes are self-patched, as in the
        batch-one audit, and must pass the same tolerance.
        """
        encode = getattr(self.family_ops, 'encode_batch', None)
        if encode is None:
            return [sources[n] for n in names], AUDIT_SOURCE_CACHE
        visuals = list(encode(self.adapter, inputs_n))
        if len(visuals) != len(names):
            raise ValueError('encode_batch returned a different number of rows')
        for name, visual in zip(names, visuals):
            cache_differences[name] = self.visual_difference(sources[name], visual)
        return visuals, AUDIT_BATCHED_ENCODE

    def audit_record(self, row, readout, field, values, geometry, fixed_later, sizes, mode, cache_differences):
        record = {'run_id': self.run_id, 'id': row['id'], **readout, field: values,
                  'geometry': geometry, 'fixed_later_hashes': fixed_later,
                  'batch_size': sizes[0] if len(sizes) == 1 else sizes, 'self_patch_visuals': mode}
        if mode == AUDIT_BATCHED_ENCODE:
            record['source_cache_vs_batched_encode_max_abs'] = cache_differences
        return record

    def score_audit(self, row, readout, prepared, options, sources, geometry, fixed_later):
        """One batched NATIVE forward over the sources vs one batched SELF-PATCH forward."""
        differences, sizes, modes, cache_differences = {}, [], set(), {}
        for names in chunks(prepared, self.max_batch):
            inputs_n = self.family_ops.stack_inputs(self.adapter, [prepared[n] for n in names])
            native = self.scorer(self.model, inputs_n, options)
            visuals, mode = self.audit_visuals(names, inputs_n, sources, cache_differences)
            modes.add(mode)
            with self.family_ops.use_batch(self.adapter, visuals, inputs_n) as d:
                patched = self.scorer(self.model, inputs_n, options)
            if len(native) != len(names) or len(patched) != len(names):
                raise ValueError('Batched audit returned a different number of rows')
            for name, a, b in zip(names, native, patched):
                difference = max(abs(a.scores[k] - b.scores[k]) for k in a.scores)
                if (d['calls'] != b.forward_count or d.get('batch_size') != len(names)
                        or difference > self.config['self_patch_tolerance']):
                    raise ValueError(f'Native/self-patch failed {name}: {difference} ({mode})')
                differences[name] = difference
            sizes.append(len(names))
        (mode,) = modes
        return self.audit_record(row, readout, 'native_self_differences', differences, geometry, fixed_later,
                                 sizes, mode, cache_differences)

    def generation_audit(self, row, readout, prepared, sources, geometry, fixed_later):
        """Batched native vs batched self-patch greedy generation; equal tokens per row."""
        checks, sizes, modes, cache_differences = {}, [], set(), {}
        for names in chunks(prepared, self.max_batch):
            inputs_n = self.family_ops.stack_inputs(self.adapter, [prepared[n] for n in names])
            native = self.generator(self.model, inputs_n, self.genconfig, self.family, None)
            visuals, mode = self.audit_visuals(names, inputs_n, sources, cache_differences)
            modes.add(mode)
            with self.family_ops.use_batch(self.adapter, visuals, inputs_n) as d:
                patched = self.generator(self.model, inputs_n, self.genconfig, self.family, None, stats=d)
            if len(native) != len(names) or len(patched) != len(names):
                raise ValueError('Batched audit returned a different number of rows')
            for name, a, b in zip(names, native, patched):
                if a != b or d['calls'] != 1 or d.get('batch_size') != len(names):
                    raise ValueError(f'Freeform native/self-patch generation differs: {name} ({mode})')
                checks[name] = dict(equal=True, new_tokens=len(a))
            sizes.append(len(names))
        (mode,) = modes
        return self.audit_record(row, readout, 'native_self_generation', checks, geometry, fixed_later,
                                 sizes, mode, cache_differences)
