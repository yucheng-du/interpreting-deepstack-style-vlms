#!/usr/bin/env python3
"""Standard-library analysis of measurement-v3 scores and generations.

All effects are paired within item/format/mapping; no validity rate is a gate.
Missing cells are reported against an explicit requested matrix, including on
partial runs. Conflicting duplicate cells are never silently selected.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import itertools
import json
import sys
import math
import os
from pathlib import Path
import statistics

FORMATS = ('base', 'strict_letters', 'strict_digits', 'prefix_digits')
MAPPINGS = ('identity', 'cyclic')
MASKS = ('111', '000')
SLOT_CONDITIONS = tuple(f'main_{main}__ds_{ds}' for main in ('original', 'swapped')
                        for ds in ('original', 'swapped', 'unrelated'))
SLOT_GEN_CONDITIONS = tuple(x for x in SLOT_CONDITIONS if not x.endswith('unrelated'))


def distribution(values):
    values = list(values)
    if not values:
        return {'n': 0, 'mean': None, 'mean_abs': None, 'median': None, 'min': None, 'max': None}
    return {'n': len(values), 'mean': statistics.fmean(values),
            'mean_abs': statistics.fmean(map(abs, values)), 'median': statistics.median(values),
            'min': min(values), 'max': max(values)}


def fraction(numerator, denominator):
    return {'n': numerator, 'denominator': denominator,
            'rate': numerator / denominator if denominator else None}


def pearson(x, y):
    if len(x) != len(y) or len(x) < 2:
        return None
    mx, my = statistics.fmean(x), statistics.fmean(y)
    xx, yy = sum((v-mx)**2 for v in x), sum((v-my)**2 for v in y)
    if xx == 0 or yy == 0:
        return None
    return sum((a-mx)*(b-my) for a, b in zip(x, y)) / math.sqrt(xx*yy)


def key(row):
    return (row['kind'], row['id'], row['format'], row['mapping'], row.get('mask', ''),
            row.get('condition', ''))


def signature(row):
    """Compare runtime inputs when recorded, without mistaking missing for equal."""
    source = row.get('input', {})
    if not isinstance(source, dict):
        source = {}
    result = {}
    for field in ('input_ids_sha256', 'prompt_sha256', 'pixel_values_sha256',
                  'image_grid_thw', 'sequence_length', 'visual_token_count'):
        if field in row:
            result[field] = row[field]
        elif field in source:
            result[field] = source[field]
    if 'grid' in row and 'image_grid_thw' not in result:
        result['image_grid_thw'] = row['grid']
    ids = row.get('input_ids', source.get('input_ids'))
    if ids is not None and 'input_ids_sha256' not in result:
        if len(ids) == 1 and isinstance(ids[0], list):
            ids = ids[0]
        result['input_ids_sha256'] = hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest()
    if 'prompt_sha256' not in result:
        if 'prompt_text' in row:
            result['prompt_sha256'] = hashlib.sha256(row['prompt_text'].encode()).hexdigest()
        elif row.get('option_tokens', {}).get('prompt_sha256'):
            result['prompt_sha256'] = row['option_tokens']['prompt_sha256']
    return result


def normalize(row):
    for required in ('kind', 'id', 'task', 'format', 'mapping', 'label_to_original',
                     'answer_label', 'original_answer_label'):
        if required not in row:
            raise ValueError(f'missing {required}: {row.get("id")}')
    if row['kind'] not in ('score', 'generation'):
        raise ValueError(f'unknown row kind: {row["kind"]}')
    mapping = row['label_to_original']
    if not isinstance(mapping, dict) or len(mapping) < 2 or len(set(mapping.values())) != len(mapping):
        raise ValueError(f'invalid label_to_original: {row["id"]}')
    answer = row['answer_label']
    expected_original = row.get('main_correct_original_choice', row['original_answer_label'])
    if mapping.get(answer) != expected_original:
        raise ValueError(f'answer/mapping mismatch: {row["id"]}')
    result = {field: row[field] for field in ('kind', 'id', 'task', 'format', 'mapping',
                                             'label_to_original', 'answer_label', 'original_answer_label')}
    result.update(mask=row.get('mask', ''), condition=row.get('condition', ''), input=signature(row),
                  protocol_sha256=row.get('protocol_sha256'),
                  output_head_precision=row.get('output_head_precision'), diagnostics=[])
    if row['kind'] == 'score':
        scored = row.get('scored', row)
        scores = scored.get('scores', row.get('scores'))
        if not isinstance(scores, dict) or set(scores) != set(mapping):
            raise ValueError(f'score labels/mapping mismatch: {row["id"]}')
        if any(type(x) not in (int, float) or not math.isfinite(x) for x in scores.values()):
            raise ValueError(f'nonfinite/nonnumeric score: {row["id"]}')
        best = max(scores.values())
        ties = [label for label in mapping if scores[label] == best]
        # Report ties explicitly. The runner chooses the first public labels entry.
        labels = row.get('labels', row.get('option_tokens', {}).get('labels', list(mapping)))
        if set(labels) != set(mapping) or len(labels) != len(mapping):
            raise ValueError(f'labels/mapping mismatch: {row["id"]}')
        prediction = max(labels, key=scores.__getitem__)
        margin = scores[answer] - max(value for label, value in scores.items() if label != answer)
        reported_margin = scored.get('correct_margin', scored.get('margin'))
        if reported_margin is not None and not math.isclose(float(reported_margin), margin, abs_tol=1e-6):
            result['diagnostics'].append('reported_margin_mismatch')
        reported_prediction = scored.get('prediction', row.get('prediction'))
        if reported_prediction is not None and reported_prediction != prediction:
            result['diagnostics'].append('reported_prediction_mismatch')
        mass = best + math.log(sum(math.exp(value-best) for value in scores.values()))
        result.update(scores=scores, prediction=prediction, original_prediction=mapping[prediction],
                      correct=prediction == answer, correct_margin=margin, top_tied=len(ties) > 1,
                      tied_labels=ties if len(ties) > 1 else [], raw_correct_logp=scores[answer],
                      log_label_mass=mass, normalized_correct_logp=scores[answer]-mass,
                      label_mass=math.exp(mass) if mass < 710 else None)
    else:
        parsed = row.get('parse', {})
        stop = row.get('stop', {field: row.get(field) for field in ('eos_reached', 'truncated', 'stop_reason')})
        if not isinstance(parsed, dict) or not isinstance(stop, dict):
            raise ValueError(f'parse and stop must be objects: {row["id"]}')
        label = parsed.get('label')
        parsed_ok = parsed.get('status', row.get('status')) == 'parsed' and label in mapping
        scorable = bool(parsed_ok and stop.get('eos_reached') and not stop.get('truncated')
                        and not row.get('resource_guard_error'))
        if 'scorable' in row and row['scorable'] != scorable:
            result['diagnostics'].append('reported_scorable_mismatch')
        if row.get('prediction') is not None and row['prediction'] != label:
            result['diagnostics'].append('reported_prediction_mismatch')
        exact = parsed.get('exact_label_only')
        if exact is not None and not isinstance(exact, bool):
            raise ValueError(f'exact_label_only must be boolean: {row["id"]}')
        result.update(prediction=label if parsed_ok else None,
                      original_prediction=mapping[label] if parsed_ok else None,
                      correct=bool(scorable and label == answer), scorable=scorable,
                      parsed=parsed_ok, parse_status=parsed.get('status', row.get('status', 'missing')),
                      exact_label_only=exact, eos_reached=bool(stop.get('eos_reached')),
                      truncated=bool(stop.get('truncated')), stop_reason=stop.get('stop_reason', 'missing'),
                      text=row.get('text', row.get('generated_text')), generated_tokens=row.get('generated_tokens'))
    for field in ('main_correct_original_choice', 'swap_correct_original_choice', 'swap_direction_log_odds'):
        if field in row:
            result[field] = row[field]
    return result


def grouped(rows, dimensions, summarizer):
    buckets = defaultdict(list)
    for row in rows:
        values = tuple(row[d] for d in dimensions)
        buckets[values + ('__all__',)].append(row)
        buckets[values + (row['task'],)].append(row)
    return [{**dict(zip((*dimensions, 'task'), values)), **summarizer(buckets[values])}
            for values in sorted(buckets)]


def score_summary(rows):
    return {'n': len(rows), 'accuracy': fraction(sum(r['correct'] for r in rows), len(rows)),
            'top_ties': fraction(sum(r['top_tied'] for r in rows), len(rows)),
            **{field: distribution(r[field] for r in rows) for field in
               ('correct_margin', 'raw_correct_logp', 'normalized_correct_logp', 'log_label_mass', 'label_mass')}}


def generation_summary(rows):
    n = len(rows)
    scorable = [r for r in rows if r['scorable']]
    exact_known = [r for r in rows if r['exact_label_only'] is not None]
    return {'n': n, 'parsed': fraction(sum(r['parsed'] for r in rows), n),
            'scorable': fraction(len(scorable), n),
            'accuracy_all_rows': fraction(sum(r['correct'] for r in rows), n),
            'accuracy_scorable_rows': fraction(sum(r['correct'] for r in scorable), len(scorable)),
            'truncated': fraction(sum(r['truncated'] for r in rows), n),
            'eos_reached': fraction(sum(r['eos_reached'] for r in rows), n),
            'noncompliant': fraction(sum(not r['exact_label_only'] for r in exact_known), len(exact_known)),
            'compliance_unknown': n-len(exact_known),
            'parse_status': dict(Counter(r['parse_status'] for r in rows)),
            'stop_reason': dict(Counter(r['stop_reason'] for r in rows))}


def input_comparison(left, right):
    common = sorted(set(left['input']) & set(right['input']))
    differing = [f for f in common if left['input'][f] != right['input'][f]]
    return {'compared_fields': common, 'different_fields': differing,
            'one_sided_fields': sorted(set(left['input']) ^ set(right['input'])),
            'status': 'different' if differing else ('equal_recorded_fields' if common else 'unknown')}


def concordance_summary(rows):
    usable = [r for r in rows if r['usable']]
    result = {'paired_rows': len(rows), 'scorable_pairs': len(usable),
            'agreement': fraction(sum(r['agreement'] for r in usable), len(usable)),
            'left_accuracy': fraction(sum(r['left_correct'] for r in usable), len(usable)),
            'right_accuracy': fraction(sum(r['right_correct'] for r in usable), len(usable)),
            'disagreement_ids': [r['id'] for r in usable if not r['agreement']],
            'unscorable_ids': [r['id'] for r in rows if not r['usable']],
            'input_comparison': dict(Counter(r['input_comparison']['status'] for r in rows))}
    continuous = [r for r in rows if 'left_score_metrics' in r and 'right_score_metrics' in r]
    if continuous:
        result['continuous_pairs'] = len(continuous)
        result['continuous_comparison'] = {}
        for field in EFFECT_METRICS:
            x = [r['left_score_metrics'][field] for r in continuous]
            y = [r['right_score_metrics'][field] for r in continuous]
            result['continuous_comparison'][field] = {'pearson': pearson(x, y),
                                                       'right_minus_left': distribution(b-a for a, b in zip(x, y))}
    return result


def compare_pair(left, right):
    usable = all(r['kind'] == 'score' or r['scorable'] for r in (left, right))
    result = {'id': left['id'], 'task': left['task'], 'format': left['format'],
            'mapping': left['mapping'], 'mask': left['mask'], 'condition': left['condition'],
            'kind': left['kind'], 'usable': usable,
            'agreement': left['original_prediction'] == right['original_prediction'] if usable else None,
            'left_correct': left['correct'] if usable else None,
            'right_correct': right['correct'] if usable else None,
            'left_original_prediction': left['original_prediction'],
            'right_original_prediction': right['original_prediction'],
            'input_comparison': input_comparison(left, right)}
    for side, row in (('left', left), ('right', right)):
        if row['kind'] == 'score':
            result[f'{side}_score_metrics'] = {field: row[field] for field in EFFECT_METRICS}
    return result


EFFECT_METRICS = ('correct_margin', 'raw_correct_logp', 'normalized_correct_logp', 'log_label_mass')


def effect_pair(baseline, ablated):
    result = {f: baseline[f] for f in ('id', 'task', 'format', 'mapping')}
    for field in EFFECT_METRICS:
        result[f'delta_{field}'] = ablated[field] - baseline[field]
    result.update(baseline_correct=baseline['correct'], ablated_correct=ablated['correct'],
                  prediction_flip=baseline['original_prediction'] != ablated['original_prediction'],
                  baseline_original_prediction=baseline['original_prediction'],
                  ablated_original_prediction=ablated['original_prediction'],
                  input_comparison=input_comparison(baseline, ablated))
    result['decomposition_residual'] = result['delta_raw_correct_logp'] - (
        result['delta_normalized_correct_logp'] + result['delta_log_label_mass'])
    return result


def effect_summary(rows):
    n = len(rows)
    return {'n': n,
            **{f'delta_{f}': distribution(r[f'delta_{f}'] for r in rows) for f in EFFECT_METRICS},
            'baseline_accuracy': fraction(sum(r['baseline_correct'] for r in rows), n),
            'ablated_accuracy': fraction(sum(r['ablated_correct'] for r in rows), n),
            'prediction_flips': fraction(sum(r['prediction_flip'] for r in rows), n),
            'correct_to_wrong': sum(r['baseline_correct'] and not r['ablated_correct'] for r in rows),
            'wrong_to_correct': sum(not r['baseline_correct'] and r['ablated_correct'] for r in rows),
            'max_abs_decomposition_residual': max((abs(r['decomposition_residual']) for r in rows), default=None),
            'input_comparison': dict(Counter(r['input_comparison']['status'] for r in rows))}


def mapping_effect_summary(rows):
    result = {'n': len(rows)}
    for field in EFFECT_METRICS:
        field = f'delta_{field}'
        x, y = [r['identity'][field] for r in rows], [r['cyclic'][field] for r in rows]
        result[field] = {'pearson': pearson(x, y),
                         'cyclic_minus_identity': distribution(b-a for a, b in zip(x, y)),
                         'same_sign_nonzero': fraction(sum(a*b > 0 for a, b in zip(x, y)),
                                                       sum(a != 0 and b != 0 for a, b in zip(x, y)))}
    return result


def analyze_rows(raw_rows, manifest, *, score_formats=FORMATS, generation_formats=FORMATS,
                 mappings=MAPPINGS, masks=MASKS, arm='readout',
                 slot_conditions=SLOT_CONDITIONS, slot_generation_conditions=SLOT_GEN_CONDITIONS):
    if len({r['id'] for r in manifest}) != len(manifest):
        raise ValueError('duplicate manifest IDs')
    source = {r['id']: r for r in manifest}
    unique, duplicates = {}, []
    for raw in raw_rows:
        row = normalize(raw)
        k = key(row)
        if k in unique:
            if row != unique[k]:
                raise ValueError(f'conflicting duplicate measurement cell: {k}')
            duplicates.append(k)
        else:
            unique[k] = row
    expected = set()
    for kind, formats in (('score', score_formats), ('generation', generation_formats)):
        coords = [(m, '') for m in masks] if arm == 'readout' else [('', c) for c in (
            slot_conditions if kind == 'score' else slot_generation_conditions)]
        expected.update((kind, ident, fmt, mapping, mask, condition) for ident, fmt, mapping, (mask, condition)
                        in itertools.product(source, formats, mappings, coords))
    observed = set(unique)
    unexpected = observed - expected
    eligible = {k: r for k, r in unique.items() if k in expected}
    errors = []
    for k, row in eligible.items():
        expected_row = source[row['id']]
        if row['task'] != expected_row['task'] or row['original_answer_label'] != expected_row['answer_label']:
            raise ValueError(f'manifest label/task mismatch: {k}')
        errors.extend({'cell': list(k), 'diagnostic': d} for d in row['diagnostics'])
    scores = [r for r in eligible.values() if r['kind'] == 'score']
    generations = [r for r in eligible.values() if r['kind'] == 'generation']
    dimensions = ('format', 'mapping', 'mask', 'condition')
    score_gen_pairs, mapping_pairs, mask_pairs, effects = [], [], [], []
    for k, row in sorted(eligible.items()):
        if row['kind'] == 'score':
            other = eligible.get(('generation', *k[1:]))
            if other is not None:
                score_gen_pairs.append(compare_pair(row, other))
        if row['mapping'] == 'identity':
            other = eligible.get((*k[:3], 'cyclic', *k[4:]))
            if other is not None:
                comparison = compare_pair(row, other)
                # Mapping legitimately changes text. Expose input diagnostics without calling them failures.
                mapping_pairs.append(comparison)
        if arm == 'readout' and row['mask'] == '111':
            other = eligible.get((*k[:4], '000', k[5]))
            if other is not None:
                mask_pairs.append(compare_pair(row, other))
                if row['kind'] == 'score':
                    effects.append(effect_pair(row, other))
    effects_by_key = {(r['id'], r['format'], r['mapping']): r for r in effects}
    mapping_effects = []
    for row in effects:
        if row['mapping'] == 'identity':
            other = effects_by_key.get((row['id'], row['format'], 'cyclic'))
            if other is not None:
                mapping_effects.append({'id': row['id'], 'task': row['task'], 'format': row['format'],
                                       'identity': row, 'cyclic': other})
    report = {
        'schema': 'measurement-v3-analysis-v1', 'arm': arm,
        'definitions': {
            'phase': 'exploratory; reused and new-to-readout panels are not unseen confirmatory samples',
            'correct_margin': 'score(current correct label) minus maximum wrong-label score',
            'delta': '000 minus 111, within the same question/format/mapping',
            'log_label_mass': 'logsumexp of candidate continuation log probabilities in the current conditioning context',
            'normalized_correct_logp': 'correct-label log probability minus log_label_mass',
            'decomposition': 'delta_raw_correct_logp = delta_normalized_correct_logp + delta_log_label_mass',
            'margin_normalization': 'subtracting the same log_label_mass from every candidate leaves correct_margin unchanged; the decomposition is for correct-label log probability, not margin',
            'prefix_context': 'prefix_digits scores are conditional on the supplied assistant prefix; raw mass is not directly the same event as unprefixed formats',
            'generation_scorable': 'parse.status == parsed, valid current label, EOS reached, not truncated',
            'generation_accuracy_all_rows': 'correct scorable generations / all generation rows; unscorable rows count as unsuccessful',
            'noncompliant': 'not parse.exact_label_only among rows with that explicit field; unknown separately reported',
            'agreement': 'original-choice predictions paired by question and condition; no averaging before comparison',
            'score_generation': 'both accuracies and agreement share the scorable-generation paired denominator',
            'mapping_comparison': 'identity versus cyclic per item after mapping labels back to original choices',
            'pearson': 'paired per-item effects; null when fewer than two pairs or either variance is zero',
            'validity': 'coverage, ties, parsing, compliance, truncation and consistency are reported dimensions, not arbitrary acceptance gates',
        },
        'validation': {'manifest_items': len(manifest), 'observed_unique_rows': len(unique),
                       'expected_rows': len(expected), 'included_rows': len(eligible),
                       'duplicate_identical_cells': [list(k) for k in duplicates],
                       'missing_cells': [list(k) for k in sorted(expected-observed)],
                       'unexpected_cells': [list(k) for k in sorted(unexpected)],
                       'complete_requested_matrix': not (expected-observed or unexpected),
                       'sample_status': dict(Counter(r.get('measurement_v3', {}).get('sample_status', 'unspecified') for r in manifest)),
                       'protocol_sha256_counts': dict(Counter(r['protocol_sha256'] or 'unrecorded' for r in eligible.values())),
                       'row_diagnostics': errors},
        'scores': grouped(scores, dimensions, score_summary),
        'generations': grouped(generations, dimensions, generation_summary),
        'score_generation_concordance': grouped(score_gen_pairs, dimensions, concordance_summary),
        'mapping_consistency': grouped(mapping_pairs, ('kind', 'format', 'mask', 'condition'), concordance_summary),
        'mask_consistency': grouped(mask_pairs, ('kind', 'format', 'mapping'), concordance_summary),
        'mask_effects': grouped(effects, ('format', 'mapping'), effect_summary),
        'cross_mapping_effects': grouped(mapping_effects, ('format',), mapping_effect_summary),
        'items': {'score_generation': score_gen_pairs, 'mapping_consistency': mapping_pairs,
                  'mask_consistency': mask_pairs, 'mask_effects': effects,
                  'cross_mapping_effects': mapping_effects},
    }
    if arm == 'slots':
        report['slot_contrasts'] = slot_contrasts(scores)
    return report


def slot_contrasts(scores):
    """All six cells share a fixed swapped-minus-original content contrast."""
    groups = defaultdict(dict)
    for row in scores:
        groups[(row['id'], row['task'], row['format'], row['mapping'])][row['condition']] = row
    comparisons = (
        ('ds_swap_main_original', 'main_original__ds_original', 'main_original__ds_swapped'),
        ('ds_swap_main_swapped', 'main_swapped__ds_original', 'main_swapped__ds_swapped'),
        ('main_swap_ds_original', 'main_original__ds_original', 'main_swapped__ds_original'),
        ('main_swap_ds_swapped', 'main_original__ds_swapped', 'main_swapped__ds_swapped'),
        ('unrelated_ds_main_original', 'main_original__ds_original', 'main_original__ds_unrelated'),
        ('unrelated_ds_main_swapped', 'main_swapped__ds_original', 'main_swapped__ds_unrelated'),
    )
    rows = []
    for (ident, task, fmt, mapping), cells in sorted(groups.items()):
        for name, left, right in comparisons:
            if left not in cells or right not in cells:
                continue
            a, b = cells[left], cells[right]
            if 'swap_direction_log_odds' not in a or 'swap_direction_log_odds' not in b:
                continue
            rows.append({'id': ident, 'task': task, 'format': fmt, 'mapping': mapping, 'contrast': name,
                         'delta_swap_direction_log_odds': b['swap_direction_log_odds'] - a['swap_direction_log_odds'],
                         'left_correct': a['correct'], 'right_correct': b['correct']})
        core = [f'main_{m}__ds_{d}' for m, d in (('original', 'original'), ('original', 'swapped'),
                                                ('swapped', 'original'), ('swapped', 'swapped'))]
        if all(c in cells and 'swap_direction_log_odds' in cells[c] for c in core):
            a, b, c, d = (cells[x]['swap_direction_log_odds'] for x in core)
            rows.append({'id': ident, 'task': task, 'format': fmt, 'mapping': mapping,
                         'contrast': 'main_by_ds_difference_in_differences',
                         'delta_swap_direction_log_odds': (d-c)-(b-a)})
    def summarize(values):
        return {'n': len(values), 'delta_swap_direction_log_odds': distribution(
            r['delta_swap_direction_log_odds'] for r in values)}
    return {'definition': 'fixed swapped-gold minus original-gold label score; contrast is right cell minus left cell. Main-swap contrasts do not subtract correctness margins with different targets.',
            'summary': grouped(rows, ('format', 'mapping', 'contrast'), summarize), 'items': rows}


def read_jsonl(path):
    with Path(path).open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def markdown(report):
    v = report['validation']
    lines = ['# Measurement v3 exploratory analysis', '',
             f"Requested matrix: {v['included_rows']}/{v['expected_rows']} cells; "
             f"{len(v['missing_cells'])} missing, {len(v['unexpected_cells'])} unexpected, "
             f"{len(v['duplicate_identical_cells'])} identical duplicates.", '',
             f"Sample status: {json.dumps(v['sample_status'], ensure_ascii=False)}.", '',
             'All tables retain format and mapping. Exact item pairs, denominators, diagnostics and task summaries are in summary.json.', '',
             '## Scores', '', '| Format | Mapping | Mask/condition | N | Correct | Mean margin | Mean log mass |',
             '| --- | --- | --- | ---: | ---: | ---: | ---: |']
    for r in report['scores']:
        if r['task'] == '__all__':
            lines.append(f"| {r['format']} | {r['mapping']} | {r['mask'] or r['condition']} | {r['n']} | {r['accuracy']['n']} | {r['correct_margin']['mean']:.6f} | {r['log_label_mass']['mean']:.6f} |")
    lines += ['', '## Generations', '', '| Format | Mapping | Mask/condition | N | Scorable | Correct/scorable | Truncated | Noncompliant/known |',
              '| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for r in report['generations']:
        if r['task'] == '__all__':
            a, c = r['accuracy_scorable_rows'], r['noncompliant']
            lines.append(f"| {r['format']} | {r['mapping']} | {r['mask'] or r['condition']} | {r['n']} | {r['scorable']['n']} | {a['n']}/{a['denominator']} | {r['truncated']['n']} | {c['n']}/{c['denominator']} |")
    lines += ['', '## Score–generation agreement on the same denominator', '',
              '| Format | Mapping | Mask/condition | Paired | Scorable | Agree | Score correct | Generation correct |',
              '| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for r in report['score_generation_concordance']:
        if r['task'] == '__all__':
            lines.append(f"| {r['format']} | {r['mapping']} | {r['mask'] or r['condition']} | {r['paired_rows']} | {r['scorable_pairs']} | {r['agreement']['n']} | {r['left_accuracy']['n']} | {r['right_accuracy']['n']} |")
    lines += ['', '## Identity–cyclic original-choice agreement', '',
              '| Kind | Format | Mask/condition | Paired | Scorable | Agree |',
              '| --- | --- | --- | ---: | ---: | ---: |']
    for r in report['mapping_consistency']:
        if r['task'] == '__all__':
            lines.append(f"| {r['kind']} | {r['format']} | {r['mask'] or r['condition']} | {r['paired_rows']} | {r['scorable_pairs']} | {r['agreement']['n']} |")
    if report['mask_effects']:
        lines += ['', '## Paired 000 minus 111 effects', '',
                  '| Format | Mapping | N | Mean Δmargin | Mean Δcorrect logp | Mean Δnormalized correct logp | Mean Δlog mass |',
                  '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
        for r in report['mask_effects']:
            if r['task'] == '__all__':
                lines.append(f"| {r['format']} | {r['mapping']} | {r['n']} | {r['delta_correct_margin']['mean']:.6f} | {r['delta_raw_correct_logp']['mean']:.6f} | {r['delta_normalized_correct_logp']['mean']:.6f} | {r['delta_log_label_mass']['mean']:.6f} |")
        lines += ['', 'The last two changes sum to the change in raw correct-label log probability. Candidate normalization leaves margin unchanged. Prefix scores condition on the supplied assistant prefix, so their raw label mass is a different event from unprefixed scores.']
    lines += ['', 'No validity-rate threshold is imposed. Generation accuracy on all rows includes unscorable rows as unsuccessful; conditional accuracy and score–generation comparisons use their explicit common denominator. Reused panels and additional140 remain exploratory.', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--results', type=Path, nargs='+', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--formats', nargs='+', default=list(FORMATS))
    parser.add_argument('--generation-formats', nargs='*', default=None,
                        help='default: same as --formats; empty list means score-only requested matrix')
    parser.add_argument('--mappings', nargs='+', default=list(MAPPINGS))
    parser.add_argument('--masks', nargs='+', default=list(MASKS))
    parser.add_argument('--arm', choices=('readout', 'slots'), default='readout')
    args = parser.parse_args()
    if not args.output_dir.is_absolute():
        raise ValueError('Use an explicit absolute analysis output path')
    manifest = list(read_jsonl(args.manifest))
    report = analyze_rows(itertools.chain.from_iterable(read_jsonl(p) for p in args.results), manifest,
                          score_formats=args.formats,
                          generation_formats=args.formats if args.generation_formats is None else args.generation_formats,
                          mappings=args.mappings, masks=args.masks, arm=args.arm)
    report['provenance'] = {'manifest': {'path': str(args.manifest), 'sha256': hashlib.sha256(args.manifest.read_bytes()).hexdigest()},
                            'results': [{'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in args.results]}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    (args.output_dir / 'report.md').write_text(markdown(report))
    print(json.dumps({'output': str(args.output_dir), **{k: v for k, v in report['validation'].items()
                     if not isinstance(v, list)}}, ensure_ascii=False))


if __name__ == '__main__':
    main()
