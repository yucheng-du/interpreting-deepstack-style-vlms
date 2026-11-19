#!/usr/bin/env python3
"""Supplement the shared analyzer with paired slot contrasts and item uncertainty.

Pure Python only. Each task, format and mapping stays separate. Generation
coverage requires an explicit fixed-panel manifest, not the observed ID set.
Use the exact-protocol collector before pooling completed attempts; this
supplement does not replace its provenance or cross-backend reuse checks.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import itertools
import json
from pathlib import Path
import random
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.analyze_measurement_v3 import (
    analyze_rows, normalize, read_jsonl, pearson, fraction, SLOT_CONDITIONS,
    SLOT_GEN_CONDITIONS, MAPPINGS,
)

SHORT = {'m0d0':'main_original__ds_original', 'm0d1':'main_original__ds_swapped',
         'm0du':'main_original__ds_unrelated', 'm1d0':'main_swapped__ds_original',
         'm1d1':'main_swapped__ds_swapped', 'm1du':'main_swapped__ds_unrelated'}
DEFINITIONS = {
    'z':'score(swapped-gold slot) minus score(original-gold slot), with the same direction in all cells',
    'ds_swap_main_original':'z(m0,d1) - z(m0,d0)',
    'ds_swap_main_swapped':'z(m1,d1) - z(m1,d0)',
    'directed_swap_vs_unrelated_main_original':'z(m0,d1) - z(m0,du)',
    'reverse_swap_vs_unrelated_main_swapped':'z(m1,du) - z(m1,d0)',
    'symmetric_donor_comparison_mean':'one half of the sum of the preceding two donor comparisons; descriptive only',
    'full_native_swap':'z(m1,d1) - z(m0,d0); descriptive scale, not a contribution denominator',
    'main_by_ds_difference_in_differences':'[z(m1,d1)-z(m1,d0)] - [z(m0,d1)-z(m0,d0)]',
    'uncertainty':'95% percentile bootstrap of question-level paired values within task/format/mapping; empirical item-resampling uncertainty, not a guarantee of generalization beyond this exploratory panel',
    'mapping_retest':'identity and cyclic effects paired by question before averaging; mapping means are retained separately, with paired cyclic-minus-identity differences, correlation and signs',
    'prediction_identity':'original choice/slot after undoing the answer-label permutation; not an image-object identity transported through a main swap',
    'generation':'fixed manifest defines all planned rows/pairs, including missing or invalid outputs; same-cell score-generation agreement uses the common scorable denominator',
}


def percentile(values, p):
    x = sorted(values)
    index = (len(x)-1)*p
    lo = int(index)
    hi = min(lo+1, len(x)-1)
    return x[lo]*(hi-index) + x[hi]*(index-lo) if hi != lo else x[lo]


def mean_uncertainty(values, *, draws=2000, seed=20261003):
    values = list(values)
    if not values:
        return {'n':0, 'mean':None, 'bootstrap_percentile_95':None}
    result = {'n':len(values), 'mean':statistics.fmean(values),
              'min':min(values), 'max':max(values), 'positive':sum(x>0 for x in values),
              'negative':sum(x<0 for x in values), 'zero':sum(x==0 for x in values)}
    if len(values) < 2:
        result['bootstrap_percentile_95'] = None
        result['uncertainty_note'] = 'Only one item; resampling cannot characterize item variation.'
    else:
        rng = random.Random(seed)
        means = [statistics.fmean(rng.choices(values, k=len(values))) for _ in range(draws)]
        result['bootstrap_percentile_95'] = [percentile(means,.025), percentile(means,.975)]
    return result


def slot_values(cells):
    """Return only estimable contrasts; missing cells never become zero."""
    z = {short:cells[name]['swap_direction_log_odds'] for short,name in SHORT.items() if name in cells}
    result = {f'z_{k}':v for k,v in z.items()}
    pairs = {
        'ds_swap_main_original':('m0d1','m0d0'),
        'ds_swap_main_swapped':('m1d1','m1d0'),
        'directed_swap_vs_unrelated_main_original':('m0d1','m0du'),
        'reverse_swap_vs_unrelated_main_swapped':('m1du','m1d0'),
        'full_native_swap':('m1d1','m0d0'),
    }
    for name,(left,right) in pairs.items():
        if left in z and right in z:
            result[name] = z[left]-z[right]
    if all(k in result for k in ('ds_swap_main_original','ds_swap_main_swapped')):
        result['main_by_ds_difference_in_differences'] = result['ds_swap_main_swapped']-result['ds_swap_main_original']
    donor = ('directed_swap_vs_unrelated_main_original','reverse_swap_vs_unrelated_main_swapped')
    if all(k in result for k in donor):
        result['symmetric_donor_comparison_mean'] = sum(result[k] for k in donor)/2
    return result


def summarize(raw, manifest, *, formats=('strict_digits',), mappings=MAPPINGS,
              generation_manifest=None, generation_conditions=SLOT_GEN_CONDITIONS,
              draws=2000, seed=20261003):
    raw = list(raw)
    if any(r.get('kind') not in ('score','generation') for r in raw):
        raise ValueError('Every input row must be a score or generation measurement')
    score_raw = [r for r in raw if r['kind']=='score']
    gen_raw = [r for r in raw if r['kind']=='generation']
    if gen_raw and generation_manifest is None:
        raise ValueError('Generation rows require --generation-manifest for their fixed planned denominator')
    score_report = analyze_rows(score_raw, manifest, score_formats=formats,
        generation_formats=(), mappings=mappings, arm='slots')
    v = score_report['validation']
    if v['unexpected_cells'] or v['row_diagnostics']:
        raise ValueError('Unexpected score cells or row diagnostics; inspect shared analyzer first')
    normalized = {}
    for r in score_raw:
        row = normalize(r)
        if 'swap_direction_log_odds' not in row:
            raise ValueError('Missing fixed swap direction in a score row')
        normalized[(row['id'],row['format'],row['mapping'],row['condition'])] = row
    item_rows = []
    for item,fmt,mapping in itertools.product(manifest,formats,mappings):
        cells = {c:normalized[(item['id'],fmt,mapping,c)] for c in SLOT_CONDITIONS
                 if (item['id'],fmt,mapping,c) in normalized}
        item_rows.append({'id':item['id'],'task':item['task'],'format':fmt,'mapping':mapping,
                          'values':slot_values(cells)})
    grouped = defaultdict(list)
    for r in item_rows:
        for name,value in r['values'].items():
            grouped[(r['task'],r['format'],r['mapping'],name)].append(value)
    effect_summaries = [{**dict(zip(('task','format','mapping','metric'),key)),
        **mean_uncertainty(values,draws=draws,seed=seed)} for key,values in sorted(grouped.items())]
    index = {(r['id'],r['format'],r['mapping']):r for r in item_rows}
    retest_groups = defaultdict(list)
    if 'identity' in mappings and 'cyclic' in mappings:
        for r in item_rows:
            if r['mapping'] != 'identity':
                continue
            other = index[(r['id'],r['format'],'cyclic')]
            for name in sorted(set(r['values']) & set(other['values'])):
                retest_groups[(r['task'],r['format'],name)].append(
                    {'id':r['id'],'identity':r['values'][name],'cyclic':other['values'][name]})
    retest = []
    for key,pairs in sorted(retest_groups.items()):
        x,y = [p['identity'] for p in pairs],[p['cyclic'] for p in pairs]
        retest.append({**dict(zip(('task','format','metric'),key)), 'n':len(pairs),
            'identity_mean':statistics.fmean(x),'cyclic_mean':statistics.fmean(y),
            'pearson':pearson(x,y),
            'same_sign_nonzero':fraction(sum(a*b>0 for a,b in zip(x,y)),sum(a!=0 and b!=0 for a,b in zip(x,y))),
            'cyclic_minus_identity':mean_uncertainty([b-a for a,b in zip(x,y)],draws=draws,seed=seed),
            'items':pairs})
    report = {'schema':'measurement-v3-slot-effects-supplement-v1','definitions':DEFINITIONS,
        'bootstrap':{'draws':draws,'seed':seed,'unit':'question; within each task, format and mapping'},
        'score_validation':v,'score_mapping_consistency':score_report['mapping_consistency'],
        'single_mapping_effects':effect_summaries,'cross_mapping_retest':retest,'items':item_rows}
    if generation_manifest is not None:
        main = {r['id']:r for r in manifest}
        if len({r['id'] for r in generation_manifest}) != len(generation_manifest):
            raise ValueError('Duplicate IDs in generation manifest')
        if any(r['id'] not in main or (r['task'],r['answer_label']) !=
               (main[r['id']]['task'],main[r['id']]['answer_label']) for r in generation_manifest):
            raise ValueError('Generation manifest must be a label/task-matching subset of the score manifest')
        gids = {r['id'] for r in generation_manifest}
        gen_report = analyze_rows([r for r in score_raw if r['id'] in gids]+gen_raw,
            generation_manifest, score_formats=formats,generation_formats=formats,
            mappings=mappings,arm='slots',slot_generation_conditions=generation_conditions)
        if gen_report['validation']['unexpected_cells'] or gen_report['validation']['row_diagnostics']:
            raise ValueError('Unexpected generation cells or diagnostics outside the fixed generation plan')
        gens = {(r['id'],r['format'],r['mapping'],r['condition']):normalize(r) for r in gen_raw}
        coverage,transitions = [],[]
        pair_defs = [('ds_swap_main_original','m0d0','m0d1'),
                     ('ds_swap_main_swapped','m1d0','m1d1'),
                     ('full_native_swap','m0d0','m1d1')]
        for task,fmt,mapping in itertools.product(sorted({r['task'] for r in generation_manifest}),formats,mappings):
            ids = [r['id'] for r in generation_manifest if r['task']==task]
            planned = [(ident,fmt,mapping,c) for ident,c in itertools.product(ids,generation_conditions)]
            seen = [gens[k] for k in planned if k in gens]
            usable = [r for r in seen if r['scorable']]
            paired = [(r,normalized[(r['id'],fmt,mapping,r['condition'])]) for r in usable
                      if (r['id'],fmt,mapping,r['condition']) in normalized]
            coverage.append({'task':task,'format':fmt,'mapping':mapping,'planned':len(planned),
                'observed':len(seen),'missing':len(planned)-len(seen),
                'parse_status':dict(Counter(r['parse_status'] for r in seen)),
                'parsed_over_planned':fraction(sum(r['parsed'] for r in seen),len(planned)),
                'scorable_over_planned':fraction(len(usable),len(planned)),
                'exact_label_only_over_planned':fraction(sum(r['exact_label_only'] is True for r in seen),len(planned)),
                'truncated_over_planned':fraction(sum(r['truncated'] for r in seen),len(planned)),
                'correct_over_planned':fraction(sum(r['correct'] for r in seen),len(planned)),
                'score_generation_common_pairs':len(paired),
                'score_generation_agreement':fraction(sum(a['original_prediction']==b['original_prediction'] for a,b in paired),len(paired)),
                'score_accuracy_common':fraction(sum(b['correct'] for a,b in paired),len(paired)),
                'generation_accuracy_common':fraction(sum(a['correct'] for a,b in paired),len(paired))})
            for name,left,right in pair_defs:
                lc,rc=SHORT[left],SHORT[right]
                if lc not in generation_conditions or rc not in generation_conditions:
                    continue
                pairs = [(gens.get((i,fmt,mapping,lc)),gens.get((i,fmt,mapping,rc))) for i in ids]
                valid = [(a,b) for a,b in pairs if a and b and a['scorable'] and b['scorable']]
                changed = sum(a['original_prediction']!=b['original_prediction'] for a,b in valid)
                transitions.append({'task':task,'format':fmt,'mapping':mapping,'contrast':name,
                    'planned_pairs':len(ids),'both_observed':sum(bool(a and b) for a,b in pairs),
                    'both_scorable':len(valid),'changed_valid_predictions_over_planned_pairs':fraction(changed,len(ids)),
                    'prediction_change_on_both_scorable':fraction(changed,len(valid)),
                    'unscorable_or_missing_pairs':len(ids)-len(valid)})
        report['generation'] = {'validation':gen_report['validation'],'planned_denominator_summary':coverage,
            'prediction_transitions':transitions,'mapping_consistency':gen_report['mapping_consistency']}
    return report


def markdown(report):
    lines = ['# Slot effect supplement','',
        'Exploratory item bootstrap intervals describe this panel; each task and mapping remains separate. Full native swap is a descriptive scale. No ratio or semantic-specificity claim is inferred.','',
        '| Task | Format | Mapping | Contrast | Items | Mean | 95% item bootstrap |',
        '| --- | --- | --- | --- | ---: | ---: | --- |']
    for row in report['single_mapping_effects']:
        if row['metric'].startswith('z_'):
            continue
        ci=row['bootstrap_percentile_95']
        interval=f'[{ci[0]:.5f}, {ci[1]:.5f}]' if ci else 'unavailable'
        lines.append(f"| {row['task']} | {row['format']} | {row['mapping']} | {row['metric']} | {row['n']} | {row['mean']:.5f} | {interval} |")
    lines += ['','Six-cell values, all item values, cross-mapping retest and fixed-denominator generation diagnostics are in slot_effects.json. Missing cells are retained in validation and never imputed.','']
    return '\n'.join(lines)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--results',type=Path,nargs='+',required=True)
    parser.add_argument('--generation-manifest',type=Path)
    parser.add_argument('--generation-conditions',nargs='+',choices=SLOT_CONDITIONS,default=list(SLOT_GEN_CONDITIONS))
    parser.add_argument('--formats',nargs='+',default=['strict_digits'])
    parser.add_argument('--mappings',nargs='+',choices=MAPPINGS,default=list(MAPPINGS))
    parser.add_argument('--bootstrap-draws',type=int,default=2000)
    parser.add_argument('--seed',type=int,default=20261003)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args(argv)
    if args.bootstrap_draws<100:
        parser.error('Use at least 100 bootstrap draws')
    if not args.output_dir.is_absolute():
        parser.error('Use an explicit absolute analysis output path')
    report=summarize(itertools.chain.from_iterable(read_jsonl(p) for p in args.results),list(read_jsonl(args.manifest)),
        formats=args.formats,mappings=args.mappings,generation_manifest=list(read_jsonl(args.generation_manifest)) if args.generation_manifest else None,
        generation_conditions=args.generation_conditions,draws=args.bootstrap_draws,seed=args.seed)
    paths=[args.manifest,*args.results,Path(__file__),ROOT/'scripts/analyze_measurement_v3.py']
    if args.generation_manifest:
        paths.append(args.generation_manifest)
    report['provenance']={'files':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
                          'arguments':{k:str(v) if isinstance(v,Path) else [str(p) for p in v] if k=='results' else v for k,v in vars(args).items()}}
    args.output_dir.mkdir(parents=True,exist_ok=True)
    (args.output_dir/'slot_effects.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    (args.output_dir/'slot_effects.md').write_text(markdown(report))
    print(json.dumps({'output':str(args.output_dir),'score_complete':report['score_validation']['complete_requested_matrix'],
                      'single_mapping_summaries':len(report['single_mapping_effects']),'cross_mapping_retests':len(report['cross_mapping_retest'])}))


if __name__=='__main__':
    main()
