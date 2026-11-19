# V12 vendor copy of accepted archived v11 opposite-native-v2; upstream algorithm unchanged.
"""Within-run canonical-anchor reference contrasts; no model/tensor imports.

Build the denominator from the full parent execution plan, then bind four
actual a/b rows. Raw k=1 is shared across normalized metrics. This does not
assert equal state-normalized doses at every query/readout forward.
"""
from collections import Counter
from itertools import product
import math

from cloud_prep.measurement_v12.v11_v2_dose_design import DESIGN_SCHEMA


CONTEXT = ('changing_layout', 'fixed_layout', 'changing_strength',
           'fixed_strength', 'fixed_other_assignment', 'region')
ANCHOR = ('calibration_id', 'anchor_query_id', 'anchor_query_sha256',
          'anchor_prompt_sha256', 'visual_signature', 'context_signature',
          'model_revision')
SCHEMA = 'dose-reference-contrasts-v2'


def _anchor(cal):
    return {f: cal.get(f) for f in ANCHOR}


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _status(row, planned, kind):
    if not planned:
        return ['not_planned']
    if row is None:
        return ['missing']
    if row.get('protocol_status') == 'unmatched_dose':
        return ['unmatched']
    if kind == 'score':
        return [] if _finite(row.get('b_minus_a_log_odds')) else ['nonfinite']
    flags = []
    if row.get('scorable') is not True or row.get('original_prediction') not in ('(A)', '(B)'):
        flags.append('invalid')
    if row.get('truncated', False):
        flags.append('truncated')
    return flags


def _delivery(row):
    """Actual-forward evidence, separate from canonical-anchor matching."""
    delivery = (row or {}).get('visual_delivery') or {}
    if not isinstance(delivery, dict):
        return None
    entries = delivery.get('dose_observations', [])
    return [{f: entry.get(f) for f in (
        'entry', 'realized_delta_to_actual_pre_residual',
        'realized_delta_to_actual_post_state')} for entry in entries] or None


def _evidence(row, metric):
    if row is None:
        return None
    d = row['dose_metadata']
    return {
        'kind': row.get('kind'),
        'b_minus_a_log_odds': (row.get('b_minus_a_log_odds')
                              if row.get('kind') == 'score' and _finite(row.get('b_minus_a_log_odds')) else None),
        'original_prediction': row.get('original_prediction'),
        'scorable': row.get('scorable'), 'truncated': row.get('truncated'),
        'query_sha256': row.get('query_sha256'),
        'prompt_sha256': row.get('prompt_sha256'),
        'model_key': row.get('model_key'),
        'source_anchor': _anchor(row['dose_calibration']),
        'target_anchor': _anchor(row['dose_calibration']['opposite_calibration']),
        'native_self_at_metric': d['native_self_by_metric'].get(metric),
        'target_other_native_at_metric': d['target_other_native_by_metric'].get(metric),
        'resolved_target': d.get('resolved_target'),
        'target_provenance': d.get('target_provenance'),
        'requested_k': d.get('requested_k'), 'effective_k': d.get('effective_k'),
        'realized_anchor_baseline_state_at_metric':
            d.get('realized_baseline_state_by_metric', {}).get(metric),
        'unmatched_reason': d.get('matching', {}).get('reason'),
        'actual_forward_entries': _delivery(row),
    }


def _pair_binding(matched_rows, raw_rows, metric):
    """Verify actual normalized source/target anchors against raw reference."""
    matched = next((row for row in matched_rows if row is not None), None)
    raw = next((row for row in raw_rows if row is not None), None)
    if matched is None or raw is None:
        return {'status': 'incomplete_observations'}
    mc, rc = matched['dose_calibration'], raw['dose_calibration']
    # _index has already bound each calibration to plan/query/context/revision.
    # Here close the cross-condition link, in both directions, without relying
    # on equal summary means or the legacy RSS-only "*_rho" aliases.
    for left, right in ((mc['opposite_calibration'], rc),
                        (mc, rc['opposite_calibration'])):
        if {k: v for k, v in left.items() if k != 'opposite_calibration'} != {
                k: v for k, v in right.items() if k != 'opposite_calibration'}:
            raise ValueError('Reference contrast canonical anchors differ')
    md, rd = matched['dose_metadata'], raw['dose_metadata']
    target = rd['native_self_by_metric'][metric]
    if md['target_other_native_by_metric'][metric] != target:
        raise ValueError('Reference target differs from raw native metric')
    if md.get('resolved_target') != target:
        raise ValueError('Reference resolved target differs from raw native metric')
    if md.get('target_provenance') != 'paired_opposite_role_canonical_anchor':
        raise ValueError('Reference target lacks opposite canonical provenance')
    return {'status': 'bound', 'target_at_metric': target,
            'normalized_source_anchor': _anchor(mc),
            'reference_native_anchor': _anchor(rc)}


def _measurement(keys, expected, observed, kind, bchoice):
    rows = [observed.get(key) for key in keys]
    flags = [_status(row, key in expected, kind) for key, row in zip(keys, rows)]
    counts = Counter(flag for row_flags in flags for flag in row_flags)
    effects = {}
    for role, indices in (('main', (0, 1)), ('later', (2, 3))):
        a, b = (rows[i] for i in indices)
        valid = not any(flags[i] for i in indices)
        if not valid:
            effects[role] = None
        elif kind == 'score':
            effect = b['b_minus_a_log_odds'] - a['b_minus_a_log_odds']
            effects[role] = effect if _finite(effect) else None
        else:
            effects[role] = int(b['original_prediction'] == bchoice) - int(a['original_prediction'] == bchoice)
    valid = all(value is not None for value in effects.values())
    effect = effects['main'] - effects['later'] if valid else None
    arithmetic_nonfinite = bool(valid and not _finite(effect)) or any(
        effects[role] is None and not any(flags[i] for i in indices)
        for role, indices in (('main', (0, 1)), ('later', (2, 3))))
    if arithmetic_nonfinite:
        effect = None
    return {
        'planned': all(key in expected for key in keys),
        'planned_row_slots': sum(key in expected for key in keys),
        'present_row_slots': sum(row is not None for row in rows),
        'not_planned_rows': counts['not_planned'], 'missing_rows': counts['missing'],
        'unmatched_rows': counts['unmatched'], 'nonfinite_rows': counts['nonfinite'],
        'invalid_rows': counts['invalid'], 'truncated_rows': counts['truncated'],
        'arithmetic_nonfinite': arithmetic_nonfinite,
        'main_effect': effects['main'], 'later_effect': effects['later'],
        'effect': effect, 'row_statuses': [f or ['valid'] for f in flags],
    }


def analyze_reference_contrasts(records, generations=(), expected_plan=None):
    """Return plan-denominated contrasts from explicit compatible v2 records.

    Caller supplies the complete parent execution plan, even for partial runs.
    Existing dose_analysis validation is reused; invalid identity/provenance is
    a technical error. Missing and protocol-unmatched observations remain rows.
    """
    # Lazy import permits dose_analysis.analyze to include this independent view.
    from cloud_prep.measurement_v12.v11_v2_dose_analysis import _index, _summary

    p = expected_plan
    if not p or p.get('schema') != DESIGN_SCHEMA:
        raise ValueError('Full v2 dose execution plan required')
    normalized = {}
    raw = {}
    contexts = {}
    for c in p['conditions']:
        context = tuple(c[f] for f in CONTEXT)
        if c.get('target_policy') != 'opposite_native':
            continue
        if c['dose_mode'] == 'baseline_state':
            if c['dose_value'] != 'opposite_native' or c['dose_metric'] not in ('rss', 'l1'):
                raise ValueError('Invalid reference target policy/metric')
            group = context + (c['dose_metric'],)
            contexts[group] = dict(zip(CONTEXT, context))
            k = group + (c['changed_entry'], c['donor_assignment'])
            if k in normalized:
                raise ValueError('Duplicate normalized reference condition')
            normalized[k] = c
        elif c['dose_mode'] == 'raw' and c['dose_value'] == 1:
            if c['dose_metric'] != 'raw':
                raise ValueError('Reference contrasts require shared v2 raw k1')
            k = context + (c['changed_entry'], c['donor_assignment'])
            if k in raw:
                raise ValueError('Duplicate raw k1 reference condition')
            raw[k] = c
    if not contexts:
        return {'schema': SCHEMA, 'applicable': False, 'contrasts': [],
                'coverage': {'planned_score_contrasts': 0, 'planned_generation_contrasts': 0}}
    records, generations = list(records), list(generations)
    expected, eg = {}, {}
    for q, r, c in product(p['queries'], p['readouts'], p['conditions']):
        k = q['id'], r['response_format'], r['mapping'], c['name']
        if k in expected:
            raise ValueError('Duplicate planned reference input cell')
        expected[k] = q, r, c
        if q['generation_selected'] and c['generation']:
            eg[k] = expected[k]
    if len(expected) != p['expected_score_rows'] or len(eg) != p['expected_generation_rows']:
        raise ValueError('Reference parent-plan counts differ')
    scores = _index(records, expected, 'score', p)
    gens = _index(generations, eg, 'generation', p)
    prompts = {}
    for row in records + generations:
        if row['dose_metadata'].get('model_revision') != row['dose_calibration'].get('model_revision'):
            raise ValueError('Reference dose/calibration revision differs')
        prompt = row.get('prompt_sha256')
        k = row['id'], row['response_format'], row['mapping']
        if prompt is not None and k in prompts and prompts[k] != prompt:
            raise ValueError('Reference actual query/readout prompt differs')
        if prompt is not None:
            prompts[k] = prompt
    for k in scores.keys() & gens.keys():
        if scores[k]['dose_metadata'] != gens[k]['dose_metadata']:
            raise ValueError('Reference score/generation dose differs')
    model_keys = {row.get('model_key') for row in records + generations}
    revisions = {row['dose_metadata'].get('model_revision') for row in records + generations}
    if len(model_keys) > 1 or len(revisions) > 1:
        raise ValueError('Reference contrasts cannot pool models/revisions')
    model = p.get('model', {})
    model_key = model.get('key', next(iter(model_keys), None))
    revision = model.get('revision', next(iter(revisions), None))
    contrasts, used_scores, used_gens = [], set(), set()
    for q, r, group in product(p['queries'], p['readouts'], contexts):
        context, metric = group[:-1], group[-1]
        for target_role in ('later', 'main'):
            normalized_role = 'main' if target_role == 'later' else 'later'
            conditions = []
            for role, side in product(('main', 'later'), ('a', 'b')):
                conditions.append(normalized.get(group + (role, side))
                                  if role == normalized_role else raw.get(context + (role, side)))
            keys = [(q['id'], r['response_format'], r['mapping'], c['name'])
                    if c else None for c in conditions]
            used_scores.update(k for k in keys if k in expected)
            used_gens.update(k for k in keys if k in eg)
            bchoice = ('(A)', '(B)')[q['direction_b_index']]
            score = _measurement(keys, expected, scores, 'score', bchoice)
            generation = _measurement(keys, eg, gens, 'generation', bchoice)
            matched_indices = (0, 1) if normalized_role == 'main' else (2, 3)
            raw_indices = (2, 3) if normalized_role == 'main' else (0, 1)
            bindings = {kind: _pair_binding([index.get(keys[i]) for i in matched_indices],
                                           [index.get(keys[i]) for i in raw_indices], metric)
                        for kind, index in (('score', scores), ('generation', gens))}
            slots = []
            for k, c in zip(keys, conditions):
                slots.append({
                    'key': list(k) if k else None,
                    'changed_entry': c['changed_entry'] if c else None,
                    'donor_assignment': c['donor_assignment'] if c else None,
                    'planned_calibration_id': c['calibration_id'] if c else None,
                    'planned_score': k in expected, 'planned_generation': k in eg,
                    'score': _evidence(scores.get(k), metric),
                    'generation': _evidence(gens.get(k), metric),
                })
            contrasts.append({
                **{f: q[f] for f in ('id', 'scene_id', 'intervention_id', 'changed_object',
                                    'query_type', 'query_target')},
                'query_sha256': q.get('query_sha256'), **r, **contexts[group],
                'model_key': model_key, 'model_revision': revision,
                'dose_metric': metric, 'reference_target': target_role + '_native',
                'contrast': 'main_minus_later',
                'formula': ('E(main opposite-native matched-to-later) - E(later raw k1)'
                            if target_role == 'later' else
                            'E(main raw k1) - E(later opposite-native matched-to-main)'),
                'planned_anchor_query_id': p['calibration_anchors'][q['intervention_id']],
                'row_order': ['main_a', 'main_b', 'later_a', 'later_b'],
                'rows': slots, 'anchor_binding': bindings,
                'score': score, 'generation': generation,
                'score_effect': score['effect'], 'generation_effect': generation['effect'],
            })
    fields = ('model_key', 'model_revision', 'reference_target', 'dose_metric',
              'query_type', 'query_target', 'response_format', 'mapping') + CONTEXT
    coverage = {
        'parent_planned_score_rows': len(expected), 'parent_present_score_rows': len(scores),
        'parent_planned_generation_rows': len(eg), 'parent_present_generation_rows': len(gens),
        'planned_reference_cells': len(contrasts),
        'source_scenes': len({q['scene_id'] for q in p['queries']}),
        'unique_reference_score_rows': len(used_scores),
        'unique_reference_generation_rows': len(used_gens),
    }
    for kind in ('score', 'generation'):
        planned = [row for row in contrasts if row[kind]['planned']]
        slot_fields = ('planned_row_slots', 'present_row_slots', 'not_planned_rows',
                       'missing_rows', 'unmatched_rows', 'nonfinite_rows',
                       'invalid_rows', 'truncated_rows')
        coverage.update({
            'planned_' + kind + '_contrasts': len(planned),
            'not_planned_' + kind + '_contrasts': len(contrasts) - len(planned),
            'complete_' + kind + '_contrasts': sum(row[kind]['effect'] is not None for row in planned),
            kind + '_row_slot_counts': {
                field: sum(row[kind][field] for row in contrasts) for field in slot_fields},
        })
    return {
        'schema': SCHEMA, 'applicable': True, 'coverage': coverage, 'contrasts': contrasts,
        'score_summaries': _summary([row for row in contrasts if row['score']['planned']], fields, 'score_effect'),
        'generation_summaries': _summary([row for row in contrasts if row['generation']['planned']], fields, 'generation_effect'),
        'interpretation': 'Canonical-anchor reference-dose comparison, always main minus later. '
            'Effects are formed from four actual rows before any scene aggregation. '
            'Actual query/readout state-normalized forward doses may drift; no per-entry '
            'or per-forward equality claim. Existing opposite-policy role contrasts instead '
            'compare exchanged native targets.',
        'denominator_contract': 'Full parent plan; no inner join. Not-planned generation '
            'is distinct from missing planned rows. Raw k1 rows are shared across metrics '
            'and reference cells; row-slot counts are not unique executions. '
            'Invalid and truncated generation flags can overlap. Corrupt identity, '
            'canonical calibration, or provenance raises a technical error.',
    }
