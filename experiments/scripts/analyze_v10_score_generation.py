#!/usr/bin/env python3
"""Join archived v10 natural-scene candidate scores and greedy answers (stdlib only).

Descriptive, exploratory analysis. Never filters on native success or effect sign.
The metadata execution plan supplies denominators; missing/invalid cells stay null.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import os as _path_os
DATA_ROOT = Path(_path_os.environ.get("DS_DATA_ROOT", "work")).expanduser().resolve()
from statistics import mean, median

MODELS = ('ibm4b', 'qwen8b', 'qwen32b')
SOURCE_FILES = (
    'cloud_prep/measurement_v10/runner.py',
    'cloud_prep/measurement_v10/natural_design.py',
    'cloud_prep/measurement_v10/natural_analysis.py',
    'cloud_prep/measurement_v9/runtime.py',
    'cloud_prep/measurement_v6/content_readout.py',
    'cloud_prep/measurement_v8/generation.py',
    'cloud_prep/measurement_v8/mechanism_analysis.py',
    'deepstack/scoring.py', 'deepstack/readout_v3.py',
    'scripts/prepare_readout_controls.py',
)
QUERY_FIELDS = ('scene_id', 'intervention_id', 'query_role', 'changed_object',
                'queried_object', 'direction_a_index', 'direction_b_index',
                'visual_source_signature', 'has_unchanged_query')
MATCH_FIELDS = ('condition_spec', 'prompt_sha256', 'input_ids_sha256',
                'visual_intervention_signature', 'visual_tensor_sha256',
                'source_contract_sha256', 'source_pixel_sha256',
                'source_image_sha256', 'source_a_label', 'source_b_label',
                'source_a_value', 'source_b_value', 'candidate_token_lengths')


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def jsonl(path, rows):
    with path.open('w') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')


def finite(value):
    return type(value) in (float, int) and math.isfinite(value)


def gen_valid(row):
    # Exact predicate used by the source-hash-verified formal analyzer.
    return bool(row and row.get('scorable') is True and not row.get('truncated', False)
                and row.get('original_prediction') in ('(A)', '(B)'))


def row_key(row):
    return row['id'], row['response_format'], row['mapping'], row['condition']


def load_rows(path, expected, kind, model):
    rows = {}
    h = hashlib.sha256()
    with path.open('rb') as f:
        for lineno, raw in enumerate(f, 1):
            h.update(raw)
            if not raw.strip():
                continue
            row = json.loads(raw)
            k = row_key(row)
            assert k in expected and k not in rows, (path, lineno, 'unplanned/duplicate')
            q, c, _ = expected[k]
            assert row['kind'] == kind and row['model_key'] == model
            assert row['schema'] == 'natural-scene-v10'
            assert all(row[f] == q[f] for f in QUERY_FIELDS)
            assert row['condition_spec'] == c
            assert (row['source_a_value'], row['source_b_value']) == ('red', 'blue')
            assert (row['source_a_original_choice'], row['source_b_original_choice']) == ('(A)', '(B)')
            if kind == 'score':
                scores = row['scores']
                a, b = row['source_a_label'], row['source_b_label']
                assert set(scores) == {a, b}
                if all(finite(v) for v in scores.values()):
                    assert math.isclose(scores[b] - scores[a], row['b_minus_a_log_odds'], abs_tol=1e-12)
                    pred = max(scores, key=scores.__getitem__)
                    assert row['prediction'] == pred
                    assert row['original_prediction'] == ('(A)' if pred == a else '(B)')
            else:
                assert row['scorable'] == (row['parse']['status'] == 'parsed'
                    and row['eos_reached'] and not row.get('resource_guard_error'))
                if gen_valid(row):
                    pred = row['parse']['label']
                    assert pred == row['prediction']
                    assert row['original_prediction'] == ('(A)' if pred == row['source_a_label'] else '(B)')
            row['_path'] = str(path)
            row['_line'] = lineno
            rows[k] = row
    return rows, h.hexdigest()


def make_pair(model, kind, qa, ca, readout, ka, kb, scores, gens):
    sa, sb = scores.get(ka), scores.get(kb)
    ga, gb = gens.get(ka), gens.get(kb)
    score_ok = all(r and finite(r.get('b_minus_a_log_odds')) for r in (sa, sb))
    generation_ok = gen_valid(ga) and gen_valid(gb)
    a = sa['b_minus_a_log_odds'] if sa and finite(sa.get('b_minus_a_log_odds')) else None
    b = sb['b_minus_a_log_odds'] if sb and finite(sb.get('b_minus_a_log_odds')) else None
    effect = b - a if score_ok else None
    gen_effect = int(gb['original_prediction'] == '(B)') - int(ga['original_prediction'] == '(B)') if generation_ok else None
    pair = dict(model=model, pair_kind=kind, id=qa['id'],
                **{f: qa[f] for f in QUERY_FIELDS}, **readout,
                main_strength=ca['main_strength'] if kind == 'roi' else None,
                main_layout=ca['main_layout'] if kind == 'roi' else None,
                condition_a=ka[-1], condition_b=kb[-1],
                margin_a=a, margin_b=b, effect=effect,
                abs_start_margin=abs(a) if a is not None else None,
                abs_effect=abs(effect) if effect is not None else None,
                effect_to_start_margin_ratio=abs(effect / a) if score_ok and a else None,
                toward_zero_from_a=(a * effect < 0) if score_ok else None,
                score_pair_valid=bool(score_ok), generation_pair_valid=generation_ok,
                candidate_boundary_crossed=(a * b < 0) if score_ok else None,
                candidate_tie=(a == 0 or b == 0) if score_ok else None,
                candidate_choice_changed=(sa['original_prediction'] != sb['original_prediction']) if score_ok else None,
                generation_changed=(ga['original_prediction'] != gb['original_prediction']) if generation_ok else None,
                generation_effect=gen_effect,
                missing_score_rows=sum(r is None for r in (sa, sb)),
                nonfinite_score_rows=sum(r is not None and not finite(r.get('b_minus_a_log_odds')) for r in (sa, sb)),
                missing_generation_rows=sum(r is None for r in (ga, gb)),
                invalid_generation_rows=sum(r is not None and not gen_valid(r) for r in (ga, gb)),
                truncated_generation_rows=sum(bool(r and r.get('truncated')) for r in (ga, gb)))
    for side, s, g in (('a', sa, ga), ('b', sb, gb)):
        pair['score_' + side] = None if s is None else {
            'path': s['_path'], 'line': s['_line'], 'scores': s['scores'],
            'original_prediction': s['original_prediction'], 'prediction': s['prediction'],
            'source_a_label': s['source_a_label'], 'source_b_label': s['source_b_label'],
            'candidate_token_lengths': s['candidate_token_lengths'],
            'token_ids': s['token_ids'], 'method': s['method']}
        pair['generation_' + side] = None if g is None else {
            'path': g['_path'], 'line': g['_line'],
            **{k: g[k] for k in ('original_prediction', 'prediction', 'generated_text',
                'generated_text_raw', 'generated_token_ids', 'parse', 'scorable', 'truncated', 'eos_reached', 'stop_reason')}}
        pair['score_generation_disagree_' + side] = (s['original_prediction'] != g['original_prediction']) if s and finite(s.get('b_minus_a_log_odds')) and gen_valid(g) else None
        if s and gen_valid(g):
            parsed_label = s['source_a_label'] if g['original_prediction'] == '(A)' else s['source_b_label']
            candidate_tokens = s['token_ids'][parsed_label]
            pair['generated_prefix_matches_parsed_candidate_' + side] = g['generated_token_ids'][:len(candidate_tokens)] == candidate_tokens
        else:
            pair['generated_prefix_matches_parsed_candidate_' + side] = None
    return pair


def distribution(values):
    v = sorted(x for x in values if x is not None)
    if not v:
        return {'n': 0}
    return {'n': len(v), 'mean': mean(v), 'median': median(v), 'min': v[0], 'max': v[-1]}


def summarize(rows):
    cross = Counter()
    for r in rows:
        s = 'missing_or_nonfinite' if not r['score_pair_valid'] else ('tie' if r['candidate_tie'] else ('crossed' if r['candidate_boundary_crossed'] else 'not_crossed'))
        g = 'invalid_or_missing' if not r['generation_pair_valid'] else ('changed' if r['generation_changed'] else 'unchanged')
        cross[s + '__' + g] += 1
    stay = [r for r in rows if r['candidate_boundary_crossed'] is False and r['candidate_tie'] is False and r['generation_changed'] is False]
    return {'planned_pairs': len(rows), 'source_photos': len({r['scene_id'] for r in rows}),
            'score_pairs_valid': sum(r['score_pair_valid'] for r in rows),
            'generation_pairs_valid': sum(r['generation_pair_valid'] for r in rows),
            'generation_flips': sum(r['generation_changed'] is True for r in rows),
            'generation_toward_b': sum(r['generation_effect'] == 1 for r in rows),
            'generation_toward_a': sum(r['generation_effect'] == -1 for r in rows),
            'candidate_boundary_crossings': sum(r['candidate_boundary_crossed'] is True for r in rows),
            'candidate_choice_changes': sum(r['candidate_choice_changed'] is True for r in rows),
            'cross_tab': dict(sorted(cross.items())),
            'missing_score_rows': sum(r['missing_score_rows'] for r in rows),
            'nonfinite_score_rows': sum(r['nonfinite_score_rows'] for r in rows),
            'missing_generation_rows': sum(r['missing_generation_rows'] for r in rows),
            'invalid_generation_rows': sum(r['invalid_generation_rows'] for r in rows),
            'truncated_generation_rows': sum(r['truncated_generation_rows'] for r in rows),
            'score_generation_disagree_sides': sum(r['score_generation_disagree_' + side] is True for r in rows for side in ('a', 'b')),
            'generated_prefix_differs_from_parsed_candidate_sides': sum(r['generated_prefix_matches_parsed_candidate_' + side] is False for r in rows for side in ('a', 'b')),
            'score_effect': distribution(r['effect'] for r in rows),
            'generation_effect': distribution(r['generation_effect'] for r in rows),
            'no_cross_no_flip': {'n': len(stay),
                'abs_start_margin': distribution(r['abs_start_margin'] for r in stay),
                'abs_effect': distribution(r['abs_effect'] for r in stay),
                'effect_to_start_margin_ratio': distribution(r['effect_to_start_margin_ratio'] for r in stay),
                'toward_boundary': sum(r['toward_zero_from_a'] is True for r in stay),
                'toward_boundary_effect_to_start_margin_ratio': distribution(r['effect_to_start_margin_ratio'] for r in stay if r['toward_zero_from_a']),
                'away_or_zero': sum(r['toward_zero_from_a'] is False for r in stay)}}


def grouped(rows, fields):
    groups = defaultdict(list)
    for r in rows:
        groups[tuple(r[f] for f in fields)].append(r)
    return [{**dict(zip(fields, k)), **summarize(v)} for k, v in sorted(groups.items(), key=lambda x: str(x[0]))]


def analyze_model(work, repo, model, provenance):
    key = model + '-natural-scene-all'
    run = work / 'entry-controls-v10-v1' / (key + '-main-s00')
    meta_path, formal_path = run / 'metadata.json', work / 'analysis' / key / 'summary.json'
    meta, formal = json.loads(meta_path.read_text()), json.loads(formal_path.read_text())
    for p in (meta_path, formal_path):
        provenance[str(p)] = {'sha256': digest(p), 'bytes': p.stat().st_size}
    for source in SOURCE_FILES:
        p = repo / source
        actual = digest(p)
        assert actual == meta['source_sha256'][source], ('runtime source hash differs', model, source)
        provenance[str(p)] = {'sha256': actual, 'matches_runtime_metadata': True}
    expected, gen_expected = {}, {}
    for q in meta['queries']:
        for r in meta['readouts']:
            for c in meta['conditions']:
                k = q['id'], r['response_format'], r['mapping'], c['name']
                assert k not in expected
                expected[k] = q, c, r
                if q['generation_selected'] and c['generation']:
                    gen_expected[k] = q, c, r
    assert len(expected) == meta['expected_score_rows']
    assert len(gen_expected) == meta['expected_generation_rows']
    scores, gens = {}, {}
    for filename, planned, kind in (('results.jsonl', expected, 'score'), ('generations.jsonl', gen_expected, 'generation')):
        path = run / filename
        rows, sha = load_rows(path, planned, kind, model)
        official_sha = next(s['sha256'] for s in formal['sources'] if s['path'].endswith('/' + filename))
        assert sha == official_sha, ('formal source hash mismatch', path)
        provenance[str(path)] = {'sha256': sha, 'bytes': path.stat().st_size, 'matches_formal_source': True}
        if kind == 'score': scores = rows
        else: gens = rows
    for k in scores.keys() & gens.keys():
        s, g = scores[k], gens[k]
        assert all(s[f] == g[f] for f in MATCH_FIELDS), ('score/gen joint identity mismatch', k)
        assert s['b_minus_a_log_odds'] == g['b_minus_a_log_odds']
    pairs, natives = [], []
    for ka, (q, c, r) in expected.items():
        if c['donor_assignment'] != 'a' or c['role'] not in ('natural_roi_pair', 'native_donor'):
            continue
        kind = 'roi' if c['role'] == 'natural_roi_pair' else 'native_donor'
        bname = c['name'][:-1] + 'b'
        kb = ka[:3] + (bname,)
        assert kb in expected and ka in gen_expected and kb in gen_expected
        pair = make_pair(model, kind, q, c, r, ka, kb, scores, gens)
        (pairs if kind == 'roi' else natives).append(pair)
    cov = {'planned_source_photos': len({q['scene_id'] for q in meta['queries']}),
        'paired_off_target_source_photos': len({q['scene_id'] for q in meta['queries'] if q['has_unchanged_query']}),
        'planned_score_rows': len(expected), 'present_score_rows': len(scores),
        'missing_score_rows': len(expected) - len(scores),
        'nonfinite_score_rows': sum(not finite(s.get('b_minus_a_log_odds')) for s in scores.values()),
        'planned_generation_rows': len(gen_expected), 'present_generation_rows': len(gens),
        'missing_generation_rows': len(gen_expected) - len(gens),
        'invalid_generation_rows': sum(not gen_valid(g) for g in gens.values()),
        'truncated_generation_rows': sum(bool(g.get('truncated')) for g in gens.values()),
        'planned_pairs': len(pairs), 'complete_pairs': sum(p['score_pair_valid'] for p in pairs),
        'planned_generation_pairs': len(pairs), 'valid_generation_pairs': sum(p['generation_pair_valid'] for p in pairs),
        'generation_flips': sum(p['generation_changed'] is True for p in pairs),
        'planned_native_donor_pairs': len(natives), 'complete_native_donor_pairs': sum(p['score_pair_valid'] for p in natives),
        'planned_native_donor_generation_pairs': len(natives), 'valid_native_donor_generation_pairs': sum(p['generation_pair_valid'] for p in natives)}
    assert cov == formal['coverage'], ('formal coverage differs', model, cov, formal['coverage'])
    # Recompute each mapping-specific score and generation mean without combining conditions.
    for pfield, target in (('effect', 'per_mapping_cell_summaries'), ('generation_effect', 'per_mapping_behavior_summaries')):
        for cell in formal[target]:
            fields = ('response_format', 'mapping', 'main_strength', 'main_layout', 'query_role')
            members = [p for p in pairs if all(p[f] == cell[f] for f in fields)]
            vals = [p[pfield] for p in members if p[pfield] is not None]
            assert len(members) == cell['planned_query_repeats'] and len(vals) == cell['available_query_repeats']
            assert len({p['scene_id'] for p in members}) == len(members)
            if vals:
                assert math.isclose(mean(vals), cell['statistics']['mean'], abs_tol=1e-10)
    return pairs, natives, {'model': model, 'coverage_matches_formal': True,
        'all_mapping_specific_means_match_formal': True, 'coverage': cov,
        'metadata_model': meta['model'], 'runtime': meta['runtime'],
        'generation_config': {k: meta['effective_generation_config'][k] for k in ('do_sample', 'num_beams', 'use_cache', 'max_new_tokens')}}




def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, default=(DATA_ROOT / 'runs/measurement-v10-cloud-20261014/remote/work'))
    parser.add_argument('--output', type=Path, default=(DATA_ROOT / 'analysis/v10-integration-20261016/score-generation'))
    args = parser.parse_args()

    assert args.output.resolve().is_relative_to((DATA_ROOT / 'analysis').resolve())
    repo = Path(__file__).resolve().parents[1]
    provenance, checks, pairs, natives = {}, [], [], []
    for model in MODELS:
        p, n, check = analyze_model(args.work, repo, model, provenance)
        pairs.extend(p); natives.extend(n); checks.append(check)
    args.output.mkdir(parents=True, exist_ok=True)
    roi_summary = grouped(pairs, ('model',))
    direct_rows = [p for p in pairs if p['response_format'] == 'direct' and p['query_role'] == 'changed' and p['main_strength'] == 1.0]
    direct = grouped(direct_rows, ('model', 'main_layout'))
    photos = grouped(pairs, ('model', 'scene_id'))
    native = grouped(natives, ('model', 'query_role', 'response_format', 'mapping'))
    strata = grouped(pairs, ('model', 'query_role', 'response_format', 'mapping', 'main_strength', 'main_layout'))
    examples = []
    for model in MODELS:
        # Descriptive diagnostic examples, selected by endpoint category only.
        for crossed, flipped in ((True, False), (False, False), (True, True), (False, True)):
            candidates = [p for p in pairs if p['model'] == model and p['candidate_boundary_crossed'] is crossed and p['generation_changed'] is flipped]
            candidates.sort(key=lambda p: (p['response_format'] != 'direct', p['query_role'] != 'changed', p['main_strength'] != 1.0, p['main_layout'] != 'M0', p['id'], p['mapping']))
            if candidates: examples.append(candidates[0])
    jsonl(args.output / 'roi_pairs.jsonl', pairs)
    jsonl(args.output / 'native_pairs.jsonl', natives)
    dump(args.output / 'stratified_summary.json', strata)
    dump(args.output / 'photo_summary.json', photos)
    dump(args.output / 'direct_changed_alpha1_photo_summary.json', grouped(direct_rows, ('model', 'scene_id')))
    dump(args.output / 'native_summary.json', native)
    dump(args.output / 'examples.json', examples)
    script_copy = args.output / Path(__file__).name
    script_copy.write_bytes(Path(__file__).read_bytes())
    provenance[str(Path(__file__).resolve())] = {'sha256': digest(Path(__file__))}
    dump(args.output / 'input_manifest.json', provenance)
    summary = {'schema': 'v10-score-generation-integration-v1', 'exploratory': True,
        'command': f'python3 {Path(__file__).resolve()} --work {args.work} --output {args.output}',
        'direction': 'a=red,b=blue; margin=logP(blue)-logP(red); E=margin_b-margin_a',
        'independence_unit': 'source photo scene_id; no condition-level inferential pooling',
        'roi_summary': roi_summary, 'direct_changed_alpha1': direct, 'validation': checks}
    dump(args.output / 'summary.json', summary)
    print(json.dumps({'output': str(args.output), 'roi_summary': roi_summary, 'direct_changed_alpha1': direct}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
