#!/usr/bin/env python3
"""Read-only P0 paired analysis of the accepted v11 BF16 dose records.

No tensor/model imports. The archive is restored separately with its official
verifier. Cached extraction is keyed to raw-file hashes in that verified snapshot.
New intervals use the original scene-mean / normal-95 convention; no marginal
intervals are subtracted and no generated token logits are inferred.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import os as _path_os
DATA_ROOT = Path(_path_os.environ.get("DS_DATA_ROOT", "work")).expanduser().resolve()
from statistics import mean, median, stdev

ROOT = Path(__file__).resolve().parents[1]
RESTORE = (DATA_ROOT / 'restored/v11-for-v12-p0-20261022')
OUTPUT = (DATA_ROOT / 'analysis/v12-p0-20261022/dose')
MODELS = ('ibm4b', 'qwen8b', 'qwen32b')
CTX = ('changing_layout', 'fixed_layout', 'changing_strength', 'fixed_strength', 'fixed_other_assignment', 'region')
CELL = ('model_key', 'query_type', 'query_target', 'response_format', 'mapping', *CTX)
QUERY = ('id', 'scene_id', 'intervention_id', 'changed_object')
BASE = (*CELL, *QUERY, 'changed_entry', 'calibration_id', 'pair_id', 'dose_mode', 'dose_value', 'dose_metric')
SCHEMA = 'v12-p0-dose-pairs-1'


def finite(x):
    return type(x) in (int, float) and math.isfinite(x)


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def dump_table(out, name, rows):
    rows = list(rows)
    path = out / (name + '.jsonl')
    with path.open('w') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, allow_nan=False, separators=(',', ':')) + '\n')
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with (out / (name + '.csv')).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: json.dumps(v, ensure_ascii=False, separators=(',', ':')) if isinstance(v, (dict, list, tuple)) else v for k, v in r.items()})
    return {'rows': len(rows), 'jsonl_sha256': sha(path)}


def ident(r, fields):
    return tuple(r[f] for f in fields)


def sub(r, fields):
    return {f: r[f] for f in fields}


def valid_gen(r):
    return bool(r and r.get('scorable') is True and not r.get('truncated', False)
                and r.get('original_prediction') in ('(A)', '(B)'))


def score_effect(a, b):
    return b['margin'] - a['margin'] if a and b and finite(a['margin']) and finite(b['margin']) else None


def gen_effect(a, b):
    return int(b['original_prediction'] == b['source_b_original_choice']) - int(a['original_prediction'] == a['source_b_original_choice']) if valid_gen(a) and valid_gen(b) else None


def stats(vals):
    vals = [v for v in vals if finite(v)]
    m = mean(vals) if vals else None
    se = stdev(vals) / math.sqrt(len(vals)) if len(vals) > 1 else None
    return {'complete_scene_n': len(vals), 'mean': m, 'ci_low': m-1.96*se if se is not None else None,
            'ci_high': m+1.96*se if se is not None else None}


def scene_summaries(rows, fields, values):
    groups = defaultdict(lambda: defaultdict(list))
    for r in rows:
        groups[ident(r, fields)][r['scene_id']].append(r)
    summary, units = [], []
    for key, scenes in groups.items():
        base = dict(zip(fields, key))
        for val in values:
            sv = []
            available = 0
            for sid, rs in sorted(scenes.items()):
                available += sum(finite(r.get(val)) for r in rs)
                # Each cell contains the two changed-object interventions.
                assert len(rs) == 2, (base, sid, len(rs))
                v = mean(r[val] for r in rs) if all(finite(r.get(val)) for r in rs) else None
                units.append({**base, 'estimand': val, 'scene_id': sid, 'query_pairs_planned': len(rs),
                              'query_pairs_available': sum(finite(r.get(val)) for r in rs), 'value': v})
                if v is not None:
                    sv.append(v)
            summary.append({**base, 'estimand': val, 'planned_scene_n': len(scenes), 'planned_query_pairs': sum(map(len, scenes.values())),
                            'available_query_pairs': available, **stats(sv), 'interval_method': 'scene mean +/- 1.96 sample SE; descriptive unadjusted interval'})
    return summary, units


def compact_entry(e):
    return {k: e.get(k) for k in ('entry', 'decoder_before', 'decoder_after', 'kind', 'selected_token_count', 'hidden_size')} | {
        'token_indices_sha256': hashlib.sha256(json.dumps(e.get('selected_token_indices'), separators=(',', ':')).encode()).hexdigest(),
        'source_delta_rms': e.get('source_delta', {}).get('rms'),
        'realized_delta_rms': e.get('realized_delta', {}).get('rms'),
        'anchor_a_state_rms': (e.get('baseline_state') or {}).get('rms')}


def compact_observations(delivery):
    return [{'entry': e['entry'], 'post_entry_state_rms': (e.get('post_entry_state') or {}).get('rms'),
             'selected_token_count': e.get('selected_token_count')}
            for e in (delivery or {}).get('dose_observations', [])]


def extract_model(model, restored, out):
    work = restored / 'remote/DeepStack/v11-work'
    campaign = work / 'dose-opposite-native-v11-both-v2-20261020' if work.is_dir() else restored
    cache = out / 'extracted' / f'{model}.jsonl'
    receipt = out / 'extracted' / f'{model}.receipt.json'
    if cache.exists() and receipt.exists():
        rec = json.loads(receipt.read_text())
        assert rec['schema'] == SCHEMA and rec['cache_sha256'] == sha(cache)
        if any(sha(restored / src['path']) != src['sha256'] for src in rec['sources']):
            raise ValueError('Input shards differ from extracted cache; choose a new output directory')
        print('Reading verified compact cache', model, flush=True)
        return [json.loads(x) for x in cache.open()], rec
    cache.parent.mkdir(exist_ok=True)
    shards = sorted(campaign.glob(model + '-dose-all-main-s*'))
    assert len(shards) == 4
    records, sources = [], []
    counts = Counter()
    with cache.open('w') as dst:
        for shard in shards:
            plan = json.loads((shard / 'metadata.json').read_text())
            assert plan['model']['key'] == model and plan['scene_templates'] == 4
            queries = {q['id']: q for q in plan['queries']}
            conditions = {c['name']: c for c in plan['conditions']}
            readouts = {(r['response_format'], r['mapping']) for r in plan['readouts']}
            for name, kind in (('results.jsonl', 'score'), ('generations.jsonl', 'generation')):
                source = shard / name
                h = hashlib.sha256()
                seen = set()
                with source.open('rb') as f:
                    for lineno, data in enumerate(f, 1):
                        h.update(data)
                        r = json.loads(data)
                        key = (r['id'], r['response_format'], r['mapping'], r['condition'])
                        assert key not in seen and key[0] in queries and key[1:3] in readouts
                        seen.add(key)
                        assert r['model_key'] == model and r['condition_spec'] == conditions[r['condition']]
                        assert r['schema'] == plan['schema'] and r['execution_schema'] == plan['execution_schema']
                        for field in ('scene_id', 'intervention_id', 'query_type', 'query_target', 'changed_object'):
                            assert r[field] == queries[r['id']][field]
                        dm = r['dose_metadata']
                        assert dm['model_revision'] == plan['model']['revision']
                        if kind == 'generation':
                            assert conditions[r['condition']]['generation'] and queries[r['id']]['generation_selected']
                        x = sub(r, BASE) | sub(r, ('condition', 'donor_assignment', 'original_prediction', 'source_a_original_choice', 'source_b_original_choice', 'protocol_status'))
                        x.update(kind=kind, margin=r.get('b_minus_a_log_odds'), scorable=r.get('scorable'), truncated=r.get('truncated', False),
                                 generated_text=r.get('generated_text'), stop_reason=r.get('stop_reason'), score_prediction=r.get('prediction'),
                                 method=r.get('method'), scores=r.get('scores'), token_ids=r.get('token_ids'),
                                 source_file=str(source.relative_to(restored)), source_line=lineno,
                                 reuse=r.get('evaluation_reuse'), actual_image_forward=r.get('actual_image_forward'),
                                 observations=compact_observations(r.get('visual_delivery')),
                                 audit_observations=compact_observations(r.get('fresh_audit_delivery')),
                                 decode_probe_steps_skipped=(r.get('visual_delivery') or {}).get('dose_decode_hook_steps_skipped'),
                                 entries=[compact_entry(e) for e in dm.get('entries', [])], effective_k=dm['effective_k'],
                                 native_self=dm['native_self_by_metric'], target_native=dm['target_other_native_by_metric'],
                                 realized_anchor=dm.get('realized_baseline_state_by_metric'), resolved_target=dm.get('resolved_target'),
                                 anchor_query_id=dm['anchor_query_id'], anchor_prompt_sha256=dm['anchor_prompt_sha256'],
                                 visual_signature=dm['visual_signature'], revision=dm['model_revision'],
                                 matching_status=dm.get('matching', {}).get('status'))
                        records.append(x)
                        dst.write(json.dumps(x, separators=(',', ':'), allow_nan=False) + '\n')
                        counts[kind] += 1
                expected = plan['expected_score_rows' if kind == 'score' else 'expected_generation_rows']
                assert len(seen) == expected, (model, shard, kind, len(seen), expected)
                sources.append({'path': str(source.relative_to(restored)), 'sha256': h.hexdigest(), 'rows': len(seen)})
            print('Extracted', shard.name, flush=True)
    rec = {'schema': SCHEMA, 'model': model, 'row_counts': dict(counts), 'sources': sources,
           'cache_sha256': sha(cache), 'input_root': str(restored)}
    save_json(receipt, rec)
    return records, rec


def effects(records):
    grouped = defaultdict(dict)
    for r in records:
        k = (r['kind'], r['id'], r['response_format'], r['mapping'], r['pair_id'])
        assert r['donor_assignment'] not in grouped[k]
        grouped[k][r['donor_assignment']] = r
    result = []
    for group in grouped.values():
        assert set(group) == {'a', 'b'}
        a, b = group['a'], group['b']
        e = score_effect(a, b) if a['kind'] == 'score' else gen_effect(a, b)
        result.append(sub(b, BASE) | {'kind': b['kind'], 'effect': e, 'a': a, 'b': b})
    return result


def pair_analyses(effs):
    groups = defaultdict(dict)
    for e in effs:
        key = (e['kind'], e['id'], e['response_format'], e['mapping'], *ident(e, CTX))
        groups[key][(e['changed_entry'], e['dose_mode'], e['dose_value'], e['dose_metric'])] = e
    raw_pairs, matched_pairs, transitions = [], [], []
    for g in groups.values():
        first = next(iter(g.values()))
        base = sub(first, (*CELL, *QUERY)) | {'kind': first['kind']}
        for role in ('main', 'later'):
            one = g[(role, 'raw', 1, 'raw')]
            for k in ((.25, .5, 2, 4) if first['kind'] == 'score' else (4,)):
                high = g[(role, 'raw', k, 'raw')]
                assert one['a']['margin'] == high['a']['margin']
                assert one['a']['original_prediction'] == high['a']['original_prediction']
                if first['kind'] == 'generation':
                    assert one['a']['generated_text'] == high['a']['generated_text']
                x, y = one['effect'], high['effect']
                valid = finite(x) and finite(y)
                raw_pairs.append({**base, 'changed_entry': role, 'high_k': k,
                                  'original_E1': x, 'original_Ek': y,
                                  'common_E1': x if valid else None, 'common_Ek': y if valid else None,
                                  'paired_Ek_minus_E1': y-x if valid else None,
                                  'r_lin_Ek_minus_kE1': y-k*x if valid else None})
                if first['kind'] == 'generation':
                    b1, b4, a1, a4 = one['b'], high['b'], one['a'], high['a']
                    p1 = gen_state(b1)
                    p4 = gen_state(b4)
                    transitions.append({**base, 'changed_entry': role, 'transition': p1 + '->' + p4,
                                        'state_k1': p1, 'state_k4': p4, 'complete_a_b_both_k': valid,
                                        'same_parsed_answer': b1['original_prediction'] == b4['original_prediction'] if valid_gen(b1) and valid_gen(b4) else None,
                                        'same_generated_text': b1['generated_text'] == b4['generated_text'],
                                        'baseline_state_k1': gen_state(a1), 'baseline_state_k4': gen_state(a4),
                                        'text_k1': b1['generated_text'], 'text_k4': b4['generated_text'],
                                        'truncated_k1': b1['truncated'], 'truncated_k4': b4['truncated'],
                                        'source_k1': f"{b1['source_file']}:{b1['source_line']}",
                                        'source_k4': f"{b4['source_file']}:{b4['source_line']}"})
        for target in ('main_native', 'later_native'):
            deltas = {}
            es = {}
            for metric in ('rss', 'l1'):
                m = g[('main', 'raw', 1, 'raw')] if target == 'main_native' else g[('main', 'baseline_state', 'opposite_native', metric)]
                l = g[('later', 'raw', 1, 'raw')] if target == 'later_native' else g[('later', 'baseline_state', 'opposite_native', metric)]
                es[metric] = (m['effect'], l['effect'])
                deltas[metric] = m['effect']-l['effect'] if all(finite(x) for x in es[metric]) else None
            valid = all(finite(v) for v in deltas.values())
            matched_pairs.append({**base, 'reference_target': target, 'original_delta_rss': deltas['rss'], 'original_delta_l1': deltas['l1'],
                                  'common_delta_rss': deltas['rss'] if valid else None, 'common_delta_l1': deltas['l1'] if valid else None,
                                  'paired_delta_rss_minus_l1': deltas['rss']-deltas['l1'] if valid else None,
                                  'original_main_rss': es['rss'][0], 'original_later_rss': es['rss'][1],
                                  'original_main_l1': es['l1'][0], 'original_later_l1': es['l1'][1]})
    # Joint candidate scores vs actual free generation, keeping raw candidate
    # sequence margins distinct from the unobserved generation-token logits.
    si = {(e['id'], e['response_format'], e['mapping'], e['pair_id']): e for e in effs if e['kind'] == 'score'}
    joint = []
    for e in effs:
        if e['kind'] != 'generation':
            continue
        s = si[(e['id'], e['response_format'], e['mapping'], e['pair_id'])]
        for side in ('a', 'b'):
            sr, gr = s[side], e[side]
            assert sr['effective_k'] == gr['effective_k'] and sr['visual_signature'] == gr['visual_signature']
            joint.append(sub(e, BASE) | {'donor_assignment': side, 'margin': sr['margin'],
                                         'candidate_preference': margin_state(sr['margin']), 'generation_state': gen_state(gr),
                                         'score_method': sr['method'], 'score_candidate_token_ids': sr['token_ids'],
                                         'actual_generated_text': gr['generated_text'], 'truncated': gr['truncated'],
                                         'same_preference': margin_state(sr['margin']) == gen_state(gr) if valid_gen(gr) and finite(sr['margin']) and sr['margin'] != 0 else None})
    ji = {(r['id'], r['response_format'], r['mapping'], *ident(r, CTX), r['changed_entry'], r['dose_value']): r
          for r in joint if r['dose_mode'] == 'raw' and r['donor_assignment'] == 'b'}
    for t in transitions:
        key = (t['id'], t['response_format'], t['mapping'], *ident(t, CTX), t['changed_entry'])
        s1, s4 = ji[key+(1,)], ji[key+(4,)]
        t.update(margin_k1=s1['margin'], margin_k4=s4['margin'], candidate_state_k1=s1['candidate_preference'],
                 candidate_state_k4=s4['candidate_preference'],
                 candidate_order_changed=s1['candidate_preference'] != s4['candidate_preference'],
                 score_generation_agree_k1=s1['same_preference'], score_generation_agree_k4=s4['same_preference'])
    return raw_pairs, matched_pairs, transitions, joint


def margin_state(x):
    return 'b' if finite(x) and x > 0 else 'a' if finite(x) and x < 0 else 'tie' if x == 0 else 'missing'


def gen_state(r):
    return ('b' if r['original_prediction'] == r['source_b_original_choice'] else 'a') if valid_gen(r) else ('truncated' if r.get('truncated') else 'invalid')


def resolved_observations(r, index):
    visited = set()
    source = r
    while not source['observations']:
        k = (source['kind'], source['id'], source['response_format'], source['mapping'], source['condition'])
        if k in visited:
            return [], None, 'reuse_cycle_or_missing_probe'
        visited.add(k)
        reuse = source.get('reuse') or {}
        if not reuse.get('reused'):
            return source['audit_observations'], source, 'fresh_audit_fallback' if source['audit_observations'] else 'fresh_probe_missing'
        target = k[:-1] + (reuse['source_condition'],)
        if target not in index:
            return [], None, 'reuse_source_missing'
        source = index[target]
        # Reuse is always the same query, readout and actual visual dose.
        assert source['visual_signature'] == r['visual_signature']
    return source['observations'], source, 'reused' if source is not r else 'fresh'


def dose_tables(records, effs):
    index = {(r['kind'], r['id'], r['response_format'], r['mapping'], r['condition']): r for r in records}
    canon, kin, actual = {}, {}, []
    checks = Counter()
    max_native_error = 0.
    for e in effs:
        a, b = e['a'], e['b']
        calkey = (e['model_key'], e['intervention_id'], e['calibration_id'])
        ratios = []
        for ent in b['entries']:
            den, delta = ent['anchor_a_state_rms'], ent['source_delta_rms']
            rj = delta/den if finite(delta) and finite(den) and den > 0 else None
            ratios.append(rj)
            c = sub(b, ('model_key', 'scene_id', 'intervention_id', 'changed_object', 'changed_entry', 'calibration_id', *CTX)) | {
                **{k: v for k, v in ent.items() if k != 'realized_delta_rms'}, 'canonical_native_r_j': rj,
                'anchor_query_id': b['anchor_query_id'], 'anchor_prompt_sha256': b['anchor_prompt_sha256'], 'revision': b['revision']}
            ck = calkey+(ent['entry'],)
            if ck in canon:
                assert canon[ck] == c
            else:
                canon[ck] = c
        if all(finite(x) for x in ratios):
            native_rss, native_l1 = math.sqrt(sum(x*x for x in ratios)), sum(ratios)
            max_native_error = max(max_native_error, abs(native_rss-b['native_self']['rss']), abs(native_l1-b['native_self']['l1']))
        if e['kind'] == 'score':
            kk = calkey+(b['dose_mode'], b['dose_value'], b['dose_metric'])
            kr = sub(b, ('model_key', 'scene_id', 'intervention_id', 'changed_object', 'changed_entry', 'calibration_id', *CTX, 'dose_mode', 'dose_value', 'dose_metric')) | {
                'effective_k': b['effective_k'], 'native_rss': b['native_self']['rss'], 'native_l1': b['native_self']['l1'],
                'native_l1_over_rss': b['native_self']['l1']/b['native_self']['rss'] if b['native_self']['rss'] else None,
                'target_native_rss': b['target_native']['rss'], 'target_native_l1': b['target_native']['l1'],
                'resolved_target': b['resolved_target'], 'realized_anchor_rss': (b['realized_anchor'] or {}).get('rss'),
                'realized_anchor_l1': (b['realized_anchor'] or {}).get('l1'), 'matching_status': b['matching_status']}
            if kk in kin:
                assert kin[kk] == kr
            else:
                kin[kk] = kr
        obs, src, status = resolved_observations(a, index)
        checks[status] += 1
        by_entry = defaultdict(list)
        expected_tokens = {en['entry']:en['selected_token_count'] for en in b['entries']}
        for ob in obs:
            if ob['entry'] in expected_tokens:
                assert ob['selected_token_count'] == expected_tokens[ob['entry']]
            if finite(ob['post_entry_state_rms']) and ob['post_entry_state_rms'] > 0:
                by_entry[ob['entry']].append(ob['post_entry_state_rms'])
        active = [x for x in b['entries']]
        actual_r, anchor_r, denratios, dispersions = [], [], [], []
        for ent in active:
            ds = by_entry[ent['entry']]
            if ds and finite(ent['realized_delta_rms']):
                den = mean(ds)
                # Every observed score candidate/prefill shares the same image
                # causal prefix. Check, do not assume, scalar denominators.
                dispersion = (max(ds)-min(ds))/den
                dispersions.append(dispersion)
                actual_r.append(ent['realized_delta_rms']/den)
                anchor_r.append(ent['realized_delta_rms']/ent['anchor_a_state_rms'])
                denratios.append(den/ent['anchor_a_state_rms'])
            else:
                actual_r.append(None)
                anchor_r.append(None)
        complete = all(finite(x) for x in actual_r) and len(actual_r) == len(active) and bool(active)
        arss = math.sqrt(sum(x*x for x in actual_r)) if complete else None
        al1 = sum(actual_r) if complete else None
        crss = math.sqrt(sum(x*x for x in anchor_r)) if complete else None
        cl1 = sum(anchor_r) if complete else None
        actual.append(sub(e, BASE) | {'kind': e['kind'], 'effective_k': b['effective_k'], 'status': status,
                                     'actual_a_dose_rss': arss, 'actual_a_dose_l1': al1, 'anchor_dose_rss': crss, 'anchor_dose_l1': cl1,
                                     'actual_over_anchor_rss': arss/crss if complete and crss else None,
                                     'actual_over_anchor_l1': al1/cl1 if complete and cl1 else None,
                                     'actual_a_over_anchor_state_min': min(denratios) if denratios else None,
                                     'actual_a_over_anchor_state_max': max(denratios) if denratios else None,
                                     'within_a_forward_relative_spread_max': max(dispersions) if dispersions else None,
                                     'entry_order': [en['entry'] for en in active], 'actual_r_j': actual_r, 'anchor_r_j': anchor_r,
                                     'actual_a_state_observation_n': len(obs),
                                     'a_source_file': src['source_file'] if src else None, 'a_source_line': src['source_line'] if src else None,
                                     'b_source_file': b['source_file'], 'b_source_line': b['source_line'],
                                     'denominator_contract': 'actual a-side post-entry state, never b-side post-state'})
    assert max_native_error < 1e-10
    return list(canon.values()), list(kin.values()), actual, {
        'reuse_resolution_counts': dict(checks), 'max_native_reconstruction_error': max_native_error,
        'actual_dose_pairs':len(actual), 'missing_actual_a_dose_pairs':sum(not finite(r['actual_a_dose_rss']) for r in actual),
        'max_within_a_forward_relative_spread':max((r['within_a_forward_relative_spread_max'] or 0) for r in actual)}


def aggregate_descriptive(rows, fields, values):
    groups = defaultdict(list)
    for r in rows:
        groups[ident(r, fields)].append(r)
    result = []
    for key, rs in groups.items():
        for field in values:
            xs = sorted(r[field] for r in rs if finite(r.get(field)))
            result.append(dict(zip(fields, key)) | {'variable': field, 'planned_repeated_observations': len(rs), 'available_observations': len(xs),
                                                    'min': min(xs) if xs else None, 'median': median(xs) if xs else None,
                                                    'max': max(xs) if xs else None, 'mean': mean(xs) if xs else None,
                                                    'interpretation': 'descriptive paired repeats, not independent sample size'})
    return result


def actual_reference_pairs(rows):
    """Compare the actually delivered doses on the same query, not just anchors."""
    groups = defaultdict(dict)
    for r in rows:
        key = (r['kind'], r['id'], r['response_format'], r['mapping'], *ident(r, CTX))
        groups[key][(r['changed_entry'], r['dose_mode'], r['dose_value'], r['dose_metric'])] = r
    output = []
    for g in groups.values():
        first = next(iter(g.values()))
        for target, matched_role, native_role in (('main_native', 'later', 'main'), ('later_native', 'main', 'later')):
            native = g[(native_role, 'raw', 1, 'raw')]
            for metric in ('rss', 'l1'):
                changed = g[(matched_role, 'baseline_state', 'opposite_native', metric)]
                a = changed['actual_a_dose_'+metric]
                b = native['actual_a_dose_'+metric]
                ac = changed['anchor_dose_'+metric]
                bc = native['anchor_dose_'+metric]
                output.append(sub(first, (*CELL, *QUERY)) | {'kind':first['kind'], 'reference_target':target,
                    'dose_metric':metric, 'matched_role':matched_role, 'matched_effective_k':changed['effective_k'],
                    'actual_matched_dose':a, 'actual_native_reference_dose':b,
                    'actual_matched_over_native_reference':a/b if finite(a) and finite(b) and b>0 else None,
                    'canonical_matched_over_native_reference':ac/bc if finite(ac) and finite(bc) and bc>0 else None,
                    'denominator_contract':'each intervention uses its own actual a-side baseline; same query/readout; descriptive, not rematching'})
    return output


def run(restored, out, make_figures=True):
    out.mkdir(parents=True, exist_ok=True)
    tables = out / 'tables'
    tables.mkdir(exist_ok=True)
    receipt_path = restored / 'RESTORE_VERIFICATION.json'
    rr = json.loads(receipt_path.read_text()) if receipt_path.is_file() else None
    if rr is not None:
        assert rr['status'] == 'PASS' and rr['all_restored_sha256_match']
    all_raw, all_match, all_trans, canonical_all, k_all, actual_summary, checks, manifests = [], [], [], [], [], [], {}, {}
    receipts = []
    identities = {}
    actual_model_summary = []
    for model in MODELS:
        records, rec = extract_model(model, restored, out)
        receipts.append(rec)
        print('Pairing', model, flush=True)
        assert len({r['scene_id'] for r in records}) == 16
        assert len({r['id'] for r in records}) == 64
        assert len({ident(r, CTX) for r in records}) == 12
        identities[model] = {'scene_ids':sorted({r['scene_id'] for r in records}), 'query_ids':sorted({r['id'] for r in records}),
                             'model_revisions':sorted({r['revision'] for r in records}),
                             'readouts':sorted({(r['response_format'],r['mapping']) for r in records})}
        ef = effects(records)
        raw, matched, trans, joint = pair_analyses(ef)
        can, ks, actual, check = dose_tables(records, ef)
        checks[model] = check
        manifests[model+'_candidate_generation_joint'] = dump_table(tables, model+'_candidate_generation_joint', joint)
        manifests[model+'_actual_dose'] = dump_table(tables, model+'_actual_dose', actual)
        actual_pairs = actual_reference_pairs(actual)
        manifests[model+'_actual_reference_matches'] = dump_table(tables, model+'_actual_reference_matches', actual_pairs)
        actual_model_summary += aggregate_descriptive([r for r in actual if r['effective_k'] and r['effective_k'] > 0],
                                                     ('model_key', 'kind', 'changed_entry', 'query_target', 'response_format', 'mapping'),
                                                     ('actual_over_anchor_rss', 'actual_over_anchor_l1', 'within_a_forward_relative_spread_max'))
        actual_summary += aggregate_descriptive(actual_pairs, ('model_key', 'kind', 'query_target', 'response_format', 'mapping', 'reference_target', 'dose_metric'),
                                                ('actual_matched_over_native_reference', 'canonical_matched_over_native_reference', 'matched_effective_k'))
        all_raw += raw
        all_match += matched
        all_trans += trans
        canonical_all += can
        k_all += ks
        del records, ef, joint, actual
    raw_summary, raw_units = scene_summaries(all_raw, (*CELL, 'kind', 'changed_entry', 'high_k'),
                                           ('original_E1', 'original_Ek', 'common_E1', 'common_Ek', 'paired_Ek_minus_E1', 'r_lin_Ek_minus_kE1'))
    match_summary, match_units = scene_summaries(all_match, (*CELL, 'kind', 'reference_target'),
                                               ('original_delta_rss', 'original_delta_l1', 'common_delta_rss', 'common_delta_l1', 'paired_delta_rss_minus_l1'))
    trans_summary = []
    trans_groups = defaultdict(list)
    for r in all_trans:
        trans_groups[ident(r, (*CELL, 'changed_entry'))].append(r)
    for key, rows in trans_groups.items():
        trans_summary.append(dict(zip((*CELL, 'changed_entry'), key)) | {
            'planned_query_pairs': len(rows), 'planned_scenes': len({r['scene_id'] for r in rows}),
            'transition_counts': dict(Counter(r['transition'] for r in rows)),
            'parsed_changed': sum(r['same_parsed_answer'] is False for r in rows),
            'parsed_same': sum(r['same_parsed_answer'] is True for r in rows),
            'parsed_invalid_endpoint': sum(r['same_parsed_answer'] is None for r in rows),
            'text_changed': sum(not r['same_generated_text'] for r in rows),
            'candidate_order_changed': sum(r['candidate_order_changed'] for r in rows),
            'candidate_changed_gen_same': sum(r['candidate_order_changed'] and r['same_parsed_answer'] is True for r in rows),
            'candidate_same_gen_changed': sum(not r['candidate_order_changed'] and r['same_parsed_answer'] is False for r in rows),
            'candidate_gen_disagree_k1': sum(r['score_generation_agree_k1'] is False for r in rows),
            'candidate_gen_disagree_k4': sum(r['score_generation_agree_k4'] is False for r in rows)})
    for name, rows in (('raw_query_pairs', all_raw), ('raw_scene_pairs', raw_units), ('raw_paired_summary', raw_summary),
                       ('matched_metric_query_pairs', all_match), ('matched_metric_scene_pairs', match_units), ('matched_metric_summary', match_summary),
                       ('generation_transitions', all_trans), ('generation_transition_summary', trans_summary),
                       ('canonical_entry_composition', canonical_all), ('canonical_k_and_metrics', k_all), ('actual_dose_summary', actual_model_summary)):
        manifests[name] = dump_table(tables, name, rows)
    manifests['actual_reference_match_summary'] = dump_table(tables, 'actual_reference_match_summary', actual_summary)
    k_summary = aggregate_descriptive([r for r in k_all if r['dose_mode'] != 'raw'], ('model_key', 'changed_entry', 'dose_metric'),
                                      ('effective_k', 'native_l1_over_rss', 'realized_anchor_rss', 'realized_anchor_l1'))
    manifests['canonical_k_summary'] = dump_table(tables, 'canonical_k_summary', k_summary)
    validation = validate_original(raw_summary, match_summary)
    snapshot_path = (DATA_ROOT / 'runs/measurement-v11-20261020/preservation/source-snapshot.json')
    if snapshot_path.is_file():
        snapshot = json.loads(snapshot_path.read_text())['entries']
        for receipt in receipts:
            for source in receipt['sources']:
                assert snapshot[source['path']]['sha256'] == source['sha256']
        validation['raw_source_files_sha256_matched_to_accepted_snapshot'] = sum(len(r['sources']) for r in receipts)
        validation['accepted_snapshot_sha256'] = sha(snapshot_path)
    code_root = restored/'remote/DeepStack/v11-work/candidates/dose-opposite-native-v2/code/cloud_prep/measurement_v11'
    if code_root.is_dir():
        validation['original_v2_code_sha256'] = {name:sha(code_root/name) for name in ('dose_analysis.py','reference_contrasts.py','dose_adapter.py','runner.py')}
    assert all(identities[m]['scene_ids'] == identities[MODELS[0]]['scene_ids'] and identities[m]['query_ids'] == identities[MODELS[0]]['query_ids'] for m in MODELS)
    validation.update(schema=SCHEMA, archive=rr, extraction_receipts=receipts, dose_checks=checks,
                      coverage={'models': 3, 'scenes_per_model': 16, 'queries_per_model': 64, 'backgrounds': 12, 'readouts': 3}, source_identities=identities,
                      analysis_script_sha256=sha(Path(__file__)), intervals='original scene-normal convention, new within-scene paired differences, unadjusted',
                      no_models_or_new_inference=True, no_generated_token_logits_inferred=True)
    save_json(out/'validation.json', validation)
    save_json(out/'manifest.json', manifests)
    digest = make_digest(raw_summary, match_summary, trans_summary, k_summary, actual_model_summary)
    save_json(out/'digest.json', digest)
    if make_figures:
        figures(out, raw_summary, match_summary, trans_summary, k_all, actual_model_summary)
    print(json.dumps(digest, indent=2, ensure_ascii=False), flush=True)


def validate_original(raw_summary, match_summary):
    source = (DATA_ROOT / 'runs/measurement-v11-20261020/readable/analysis/v2-dose-cross-model-final.json')
    if not source.is_file():
        return {'original_summary_comparison': 'reference summary not supplied'}
    old = json.loads(source.read_text())
    oldrows = {}
    for model in old['models']:
        mk = model['model']['key']
        for kind, field in (('score', 'score_scene_summaries'), ('generation', 'generation_scene_summaries')):
            for r in model['raw_curves'][field]:
                oldrows[(mk, kind, *ident(r, CELL[1:]), r['changed_entry'], r['dose_value'])] = r
    errors = []
    checked_keys = set()
    for r in raw_summary:
        if r['estimand'] not in ('original_E1', 'original_Ek'):
            continue
        k = 1 if r['estimand'] == 'original_E1' else r['high_k']
        o = oldrows[(r['model_key'], r['kind'], *ident(r, CELL[1:]), r['changed_entry'], k)]
        checked_keys.add((r['model_key'], r['kind'], *ident(r, CELL[1:]), r['changed_entry'], k))
        assert o['complete_unit_n'] == r['complete_scene_n']
        errors.append(abs(o['mean']-r['mean']))
    oldmatched = {}
    for m in old['models']:
        for kind, field in (('score', 'score_summaries'), ('generation', 'generation_summaries')):
            for r in m['canonical_native_reference_contrasts'][field]:
                oldmatched[(m['model']['key'], kind, *ident(r, CELL[1:]), r['reference_target'], r['dose_metric'])] = r
    matched_errors = []
    for r in match_summary:
        if r['estimand'] not in ('original_delta_rss', 'original_delta_l1'):
            continue
        metric = r['estimand'].split('_')[-1]
        o = oldmatched[(r['model_key'], r['kind'], *ident(r, CELL[1:]), r['reference_target'], metric)]
        assert o['complete_unit_n'] == r['complete_scene_n']
        matched_errors.append(abs(o['mean']-r['mean']))
    assert max(errors) < 1e-10 and max(matched_errors) < 1e-10
    return {'original_summary_source': str(source), 'original_summary_sha256': sha(source),
            'raw_summary_comparisons': len(errors), 'unique_original_raw_cells_reproduced':len(checked_keys), 'max_raw_mean_error': max(errors),
            'matched_summary_cells_reproduced': len(matched_errors), 'max_matched_mean_error': max(matched_errors)}


def make_digest(raw, matched, trans, ks, actual):
    answer = {}
    for model in MODELS:
        select = lambda rows, **kw: [r for r in rows if r['model_key'] == model and all(r[k] == v for k, v in kw.items())]
        score = select(raw, kind='score', changed_entry='main', query_target='changed', response_format='direct', high_k=4, estimand='paired_Ek_minus_E1')
        gen = select(raw, kind='generation', changed_entry='main', query_target='changed', response_format='direct', high_k=4, estimand='paired_Ek_minus_E1')
        tr = select(trans, changed_entry='main', query_target='changed', response_format='direct')
        answer[model] = {'main_changed_direct_score_k4_minus_k1': {'backgrounds': len(score), 'negative_means': sum(r['mean']<0 for r in score),
                                                                'negative_interval': sum(r['ci_high']<0 for r in score), 'positive_interval': sum(r['ci_low']>0 for r in score)},
                         'main_changed_direct_generation_common_k4_minus_k1': {'zero_means': sum(r['mean']==0 for r in gen),
                                                                            'min_complete_scenes': min(r['complete_scene_n'] for r in gen),
                                                                            'max_complete_scenes': max(r['complete_scene_n'] for r in gen)},
                         'main_changed_direct_generation_transitions': {f: sum(r[f] for r in tr) for f in ('planned_query_pairs', 'parsed_changed', 'parsed_same', 'parsed_invalid_endpoint', 'candidate_order_changed', 'candidate_same_gen_changed', 'candidate_changed_gen_same')},
                         'metric_pairs': []}
        for target in ('main_native', 'later_native'):
            rs = select(matched, kind='score', query_target='changed', response_format='direct', reference_target=target, estimand='paired_delta_rss_minus_l1')
            answer[model]['metric_pairs'].append({'target': target, 'backgrounds': len(rs), 'positive_mean': sum(r['mean']>0 for r in rs),
                                                 'negative_mean': sum(r['mean']<0 for r in rs), 'positive_interval': sum(r['ci_low']>0 for r in rs),
                                                 'negative_interval': sum(r['ci_high']<0 for r in rs)})
    return answer


def figures(out, raw, matched, trans, ks, actual):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.lines import Line2D
    import sys
    sys.path.insert(0, str(ROOT/'scripts'))
    from plot_v11_dose_curves import geometry
    os.environ.setdefault('MPLCONFIGDIR', str(out/'matplotlib-cache'))
    plt.rcParams.update({'font.size': 8, 'axes.titlesize': 9, 'axes.labelsize': 8, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
                         'font.family': 'DejaVu Sans', 'pdf.fonttype': 42})
    labels = {'ibm4b': 'Granite Vision 4.1-4B', 'qwen8b': 'Qwen3-VL-8B', 'qwen32b': 'Qwen3-VL-32B'}
    colors = {'main': '#0072B2', 'later': '#D55E00'}
    qa = []
    contexts = sorted({ident(r, CTX) for r in raw})
    # Fixed 7-inch standalone width, >=8 pt; figures are diagnostic, not yet
    # embedded in manuscript. All three readouts are exported as separate pages.
    with PdfPages(out/'paired_dose_diagnostics.pdf') as pdf:
        for response, mapping, name in (('direct', 'none', 'Direct'), ('label', 'identity', 'Label: identity'), ('label', 'cyclic', 'Label: cyclic')):
            fig, axes = plt.subplots(2, 3, figsize=(7.1, 5.8), dpi=160)
            fig.subplots_adjust(left=.09, right=.98, top=.86, bottom=.16, wspace=.42, hspace=.56)
            for col, model in enumerate(MODELS):
                for row, kind in enumerate(('score', 'generation')):
                    ax = axes[row, col]
                    for role, offset in (('main', -.12), ('later', .12)):
                        rr = [next(r for r in raw if r['model_key']==model and r['kind']==kind and r['changed_entry']==role
                                   and r['query_target']=='changed' and r['response_format']==response and r['mapping']==mapping
                                   and r['high_k']==4 and r['estimand']=='paired_Ek_minus_E1' and ident(r, CTX)==ctx) for ctx in contexts]
                        ax.errorbar([i+offset for i in range(12)], [r['mean'] for r in rr],
                                    yerr=[[r['mean']-r['ci_low'] for r in rr], [r['ci_high']-r['mean'] for r in rr]],
                                    fmt='o', ms=2.4, elinewidth=.7, capsize=1.5, color=colors[role])
                    ax.axhline(0, color='.5', lw=.6)
                    ax.spines[['top','right']].set_visible(False)
                    ax.set_xticks([0,3,7,11], ['1','4','8','12'])
                    ax.set_xlim(-.7,11.7)
                    if row == 0: ax.set_title(labels[model], pad=9)
                    if row == 1: ax.set_xlabel('Background index')
                    if col == 0: ax.set_ylabel('Score: E(4) - E(1)' if kind=='score' else 'Generation: E(4) - E(1)')
            fig.suptitle('Paired dose change | changed-object queries | '+name, y=.97, fontsize=10)
            fig.legend(handles=[Line2D([],[],marker='o',color=c,ls='none',label=role.capitalize()) for role,c in colors.items()],
                       loc='upper center', bbox_to_anchor=(.5,.935), ncol=2, frameon=False)
            fig.text(.5,.035,'Scene-paired normal 95% intervals; generation uses the common valid scenes.\nAll 12 backgrounds retained; context key is supplied in the accompanying table.',ha='center',va='bottom',fontsize=8)
            g = geometry(fig, name)
            qa.append(g)
            assert g['passed'], g
            pdf.savefig(fig)
            fig.savefig(out/('paired_dose_'+mapping+'.png'), dpi=180)
            plt.close(fig)
    save_json(out/'figure_layout_qa.json', {'target_width_inches': 7.1, 'manuscript_embedding_verified': False,
                                         'all_passed': all(x['passed'] for x in qa), 'pages': qa})
    dump_table(out/'tables', 'figure_background_key', [{'background_index':i+1, **dict(zip(CTX, c))} for i,c in enumerate(contexts)])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--restored', type=Path, default=RESTORE)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--no-figures', action='store_true')
    args = parser.parse_args()
    os.environ.setdefault('MPLCONFIGDIR', str(args.output/'matplotlib-cache'))
    run(args.restored, args.output, not args.no_figures)
