# V12 vendor copy of accepted archived v11 opposite-native-v2; upstream algorithm unchanged.
"""Query-independent v11 visual and calibration identity; no tensor imports."""
from copy import deepcopy
import json
from cloud_prep.measurement_v10.runtime import (
    plan_runtime, prepare_runtime, resolve_snapshot, sha256, write_json,
    direction_readout, cache_identity, validate_shared_interventions,
    required_sources,
)
from cloud_prep.measurement_v9.runtime import stable_digest, visual_contract


def visual_signature(row, condition, pixel_hashes, *, calibration=None):
    return stable_digest({"source_contract": visual_contract(row),
        "processed_sources": pixel_hashes, "condition": condition,
        "calibration": calibration})


def baseline_context(condition):
    """Only the fixed a baseline and exact raw b(k=0) are interchangeable.

    Do not reuse arbitrary doses, distinct entry roles, or normalized cells which
    failed matching. Call this only after successful composition during execution.
    """
    if condition.get('role') != 'dose_pair':
        return None
    if condition['donor_assignment'] == 'a' or (
            condition['dose_mode'] == 'raw' and condition['dose_value'] == 0):
        return condition['calibration_id']
    return None


def baseline_evaluation_key(condition, *, calibration, input_ids_sha256, options):
    context = baseline_context(condition)
    if context is None:
        return None
    return stable_digest({'context': context, 'visual_signature': calibration['visual_signature'],
        'input_ids_sha256': input_ids_sha256, 'options': options})


class ScalarEvaluationCache:
    """Reuse deterministic output scalars within a single query/readout only.

    A first duplicate in an audited unit is independently evaluated and checked.
    Its physical work is reported separately; logical rows retain the original
    value and a lineage pointer. No KV, activation, or model object is retained.
    """
    def __init__(self):
        self.values = {}

    def evaluate(self, kind, key, condition, compute, compare=None):
        cell = (kind, key)
        if key is None or cell not in self.values:
            value = compute()
            json.dumps(value, allow_nan=False)
            if key is not None:
                self.values[cell] = {'value': deepcopy(value), 'condition': condition, 'checked': False}
            return value, {'reused': False, 'source_condition': condition, 'fresh_audit': False}
        saved = self.values[cell]
        checked = False
        audit_detail = {}
        if compare is not None and not saved['checked']:
            fresh = compute()
            diagnostics = compare(saved['value'], fresh)
            audit_detail = {'audit_diagnostics': diagnostics, 'fresh_audit_delivery': fresh.get('delivery')}
            saved['checked'] = True
            checked = True
        return deepcopy(saved['value']), {'reused': True, 'source_condition': saved['condition'], 'fresh_audit': checked, **audit_detail}


def evaluation_plan(rows, plan):
    """Counts evaluations separately from logical rows; unmatched cells may lower them."""
    conditions = plan['conditions']
    def counts(selected):
        seen = set(); unique = 0; duplicate_contexts = set()
        for c in selected:
            key = baseline_context(c)
            if key is None or key not in seen:
                unique += 1
            else:
                duplicate_contexts.add(key)
            if key is not None:
                seen.add(key)
        return unique, len(duplicate_contexts)
    score, score_audit = counts(conditions)
    generation, generation_audit = counts([c for c in conditions if c['generation']])
    groups = {}; audited = 0
    for row in rows:
        group = (row['attribute'], str(row.get('geometry')), row.get('panel_family'))
        identity = row['intervention_id']
        groups.setdefault(group, identity)
        if plan['panel'] == 'smoke' or groups[group] == identity:
            audited += 1
    readouts = len(plan['readouts'])
    has_generation = bool(plan['generate_tokens'])
    score_n = len(rows)*readouts*score
    gen_n = len(rows)*readouts*generation if has_generation else 0
    return {'expected_unique_score_evaluations': score_n,
        'expected_unique_generation_evaluations': gen_n,
        'expected_score_reuse_rows': plan['expected_score_rows']-score_n,
        'expected_generation_reuse_rows': plan['expected_generation_rows']-gen_n,
        'expected_dedup_score_audit_evaluations': audited*readouts*score_audit,
        'expected_dedup_generation_audit_evaluations': audited*readouts*generation_audit if has_generation else 0,
        'evaluation_count_boundary': 'All matching succeeds upper bound. Native/self/encoding overhead separate. Audits restart in each shard; sum actual shard plans for scheduling.'}
