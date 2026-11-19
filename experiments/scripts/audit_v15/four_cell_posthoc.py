#!/usr/bin/env python3
"""Post-hoc four-cell diagnostics for selected v15 cases (no inference).

Streams archived v15 task-pack observations and computes, per case:
single-entry losses U_M = CC - DC and U_L = CC - CD with paired cluster intervals,
paired differences between cases on shared units, and per-item DD -> DC transitions.
Synthetic units are templates (Student-t, df 31); benchmark units are images or
documents (normal, as the v15 scorer). Exploratory and post hoc.
"""
import importlib.util
import io
import json
import statistics
import os
import subprocess
import tarfile
from collections import defaultdict
from pathlib import Path

PACKS = Path(os.environ.get('DS_DATA_ROOT', 'work')) / 'runs/measurement-v15-cloud-20261029/packs'
SCORER_PATH = Path(__file__).resolve().parents[1]/'score_followup_benchmark_v15.py'
OUT = Path(os.environ.get('DS_DATA_ROOT', 'work')) / 'analysis/calibrated/four_cell_posthoc.json'
T975_DF31 = 2.03951344639637  # the v15 analysis quantile for 32 templates

spec = importlib.util.spec_from_file_location('scorer', SCORER_PATH)
scorer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scorer)

JOBS = {('qwen32b', 'synthetic'): 'qwen32b-synthetic-evaluation.attempt-2',
        ('qwen8b', 'synthetic'): 'qwen8b-synthetic-evaluation.attempt-2',
        ('ibm4b', 'synthetic'): 'ibm4b-synthetic-evaluation.attempt-3',
        ('deepstack7b', 'synthetic'): 'deepstack7b-synthetic-evaluation.attempt-2',
        ('qwen32b', 'docvqa'): 'qwen32b-docvqa-evaluation-fresh.attempt-2',
        ('qwen8b', 'docvqa'): 'qwen8b-docvqa-evaluation-fresh.attempt-2',
        ('ibm4b', 'docvqa'): 'ibm4b-docvqa-evaluation-fresh.attempt-2',
        ('deepstack7b', 'docvqa'): 'deepstack7b-docvqa-evaluation-fresh.attempt-2',
        ('ibm4b', 'gqa_attribute'): 'ibm4b-gqa-attribute-evaluation-fresh.attempt-2',
        ('deepstack7b', 'gqa_attribute'): 'deepstack7b-gqa-attribute-evaluation-fresh.attempt-2'}
# name: (model, source, task, response_format, mapping, kind, label, metric)
CASES = {
    'qwen32b_stripe_identity': ('qwen32b', 'synthetic', 'stripe_orientation', 'label', 'identity', 'generation', 'blurrel-period0.43', None),
    'qwen32b_text_identity': ('qwen32b', 'synthetic', 'text_reading', 'label', 'identity', 'generation', 'lowres-font2.5', None),
    'qwen32b_stripe_direct': ('qwen32b', 'synthetic', 'stripe_orientation', 'direct', 'none', 'generation', 'blurrel-period0.43', None),
    'qwen8b_stripe_direct': ('qwen8b', 'synthetic', 'stripe_orientation', 'direct', 'none', 'generation', 'blurrel-period0.43', None),
    'qwen32b_text_direct': ('qwen32b', 'synthetic', 'text_reading', 'direct', 'none', 'generation', 'lowres-font2.5', None),
    'qwen32b_color_direct': ('qwen32b', 'synthetic', 'color', 'direct', 'none', 'generation', 'desat-c0.05', None),
    'ibm4b_position_direct': ('ibm4b', 'synthetic', 'relative_position', 'direct', 'none', 'generation', 'blurrel-dotsep0.3', None),
    'ibm4b_text_direct': ('ibm4b', 'synthetic', 'text_reading', 'direct', 'none', 'generation', 'lowres-font2.5', None),
    'deepstack7b_color_direct': ('deepstack7b', 'synthetic', 'color', 'direct', 'none', 'generation', 'desat-c0.05', None),
    'qwen32b_docvqa': ('qwen32b', 'docvqa', 'docvqa_document_qa', 'freeform', 'none', 'generation', 'lowres-long180', 'docvqa_anls_organizer_framework'),
    'qwen8b_docvqa': ('qwen8b', 'docvqa', 'docvqa_document_qa', 'freeform', 'none', 'generation', 'lowres-long180', 'docvqa_anls_organizer_framework'),
    'ibm4b_docvqa': ('ibm4b', 'docvqa', 'docvqa_document_qa', 'freeform', 'none', 'generation', 'lowres-long180', 'docvqa_anls_organizer_framework'),
    'deepstack7b_docvqa': ('deepstack7b', 'docvqa', 'docvqa_document_qa', 'freeform', 'none', 'generation', 'lowres-long180', 'docvqa_anls_organizer_framework'),
    'ibm4b_gqa_attribute': ('ibm4b', 'gqa_attribute', 'gqa_attribute', 'freeform', 'none', 'generation', 'lowres-long5', 'gqa_case_punct_normalized'),
    'deepstack7b_gqa_attribute': ('deepstack7b', 'gqa_attribute', 'gqa_attribute', 'freeform', 'none', 'generation', 'lowres-long5', 'gqa_case_punct_normalized'),
}
CONTRASTS = [('qwen32b_stripe_identity', 'qwen32b_text_identity'), ('qwen32b_stripe_direct', 'qwen8b_stripe_direct'),
             ('qwen32b_stripe_direct', 'qwen32b_text_direct'), ('qwen32b_docvqa', 'qwen8b_docvqa')]


def observations(job):
    raw_dir = Path(os.environ.get('DS_DATA_ROOT', 'work')) / 'quality/calibrated/observations'
    local = raw_dir / job.split('.attempt-')[0] / 'observations.jsonl'
    if local.is_file():
        with local.open() as rows:
            for line in rows:
                if line.strip():
                    yield json.loads(line)
        return
    raw = subprocess.run(['zstd', '-dc', str(PACKS/f'{job}.tar.zst')], check=True, capture_output=True).stdout
    with tarfile.open(fileobj=io.BytesIO(raw)) as tar:
        member = next(m for m in tar.getmembers() if m.name.endswith('/observations.jsonl'))
        for line in tar.extractfile(member):
            yield json.loads(line)


def interval(values, synthetic):
    values = list(values)
    n, mean = len(values), statistics.mean(values)
    se = statistics.stdev(values)/n**.5 if n > 1 else 0.
    q = T975_DF31 if synthetic else 1.96
    if synthetic and n != 32:
        raise ValueError('Synthetic intervals assume 32 templates')
    return {'n': n, 'mean': mean, 'se': se, 'ci_low': mean-q*se if se else None, 'ci_high': mean+q*se if se else None,
            'se_zero': se == 0., 'interval': 't (df 31)' if synthetic else 'normal'}


def item_value(row, metric):
    if metric is None:
        value = row['result'].get('clean_reference_correct')
        return float(value) if value is not None else 0.  # invalid counts as incorrect
    text = row['result'].get('generated_text')
    if not isinstance(text, str) or scorer.invalid_reasons(row):
        return 0.
    bench = 'gqa' if metric.startswith('gqa') else 'docvqa'
    return scorer.metric_values(bench, text, row['reference_answers'])[metric]


def main():
    needed = defaultdict(list)
    for name, case in CASES.items():
        needed[(case[0], case[1])].append(name)
    units = {}
    result = {'cases': {}, 'contrasts': {}}
    for job_key, names in needed.items():
        rows = defaultdict(dict)
        for row in observations(JOBS[job_key]):
            if row['condition']['family'] != 'recovery':
                continue
            rows[(row['task'], row['readout']['response_format'], row['readout']['mapping'], row['kind'], row['id'])][row['condition']['name']] = row
        for name in names:
            model, source, task, fmt, mapping, kind, label, metric = CASES[name]
            synthetic = source == 'synthetic'
            cells = {'CC': 'recovery__clean__clean', 'CD': f'recovery__clean__{label}',
                     'DC': f'recovery__{label}__clean', 'DD': f'recovery__{label}__{label}'}
            by_unit, items = defaultdict(list), {}
            for (t, f, m, k, qid), conds in rows.items():
                if (t, f, m, k) != (task, fmt, mapping, kind):
                    continue
                if not all(c in conds for c in cells.values()):
                    raise ValueError(f'{name}: incomplete item {qid}')
                v = {c: item_value(conds[n], metric) for c, n in cells.items()}
                v.update(Lambda=v['CC']-v['DD'], G_D=v['DC']-v['DD'], U_M=v['CC']-v['DC'], U_L=v['CC']-v['CD'],
                         I_rec=(v['DC']-v['DD'])-(v['CC']-v['CD']))
                by_unit[conds[cells['CC']]['unit_id']].append(v)
                # edit1 recovery rows reuse edit0 inputs; count each template x object once
                key = qid.replace('-edit1-', '-edit0-') if synthetic else qid
                record = (v['DD'], v['DC'], v['CC'], v['CD'])
                outputs = tuple(conds[n]['result'].get('generated_text') for n in cells.values())
                if key in items and (items[key][:4] != record or items[key][4] != outputs):
                    raise ValueError(f'{name}: duplicated recovery item {qid} differs from its edit0 copy')
                items.setdefault(key, (*record, outputs))
            unit_means = {u: {k: statistics.mean(x[k] for x in vs) for k in vs[0]} for u, vs in by_unit.items()}
            units[name] = unit_means
            entry = {k: interval((u[k] for u in unit_means.values()), synthetic)
                     for k in ('CC', 'CD', 'DC', 'DD', 'Lambda', 'G_D', 'U_M', 'U_L', 'I_rec')}
            if synthetic:
                tr = {'w2r': 0, 'r2w': 0, 'r2r': 0, 'w2w': 0}
                rescued = broken = 0
                for dd, dc, cc, _cd, _out in items.values():
                    tr[('r' if dd else 'w')+'2'+('r' if dc else 'w')] += 1
                    if cc and not dd:
                        broken += 1
                        rescued += int(bool(dc))
                entry['transitions_distinct_items'] = {**tr, 'n_items': len(items), 'duplicates_identical': True,
                                                       'cc_right_dd_wrong': broken, 'rescued_by_dc': rescued}
            else:
                up = sum(dc > dd for dd, dc, *_ in items.values())
                down = sum(dc < dd for dd, dc, *_ in items.values())
                entry['question_changes_dd_to_dc'] = {'improved': up, 'worsened': down, 'n_questions': len(items)}
            entry['case'] = dict(zip(('model', 'source', 'task', 'response_format', 'mapping', 'kind', 'label', 'metric'),
                                     CASES[name]))
            result['cases'][name] = entry
    for a, b in CONTRASTS:
        shared = sorted(set(units[a]) & set(units[b]))
        synthetic = CASES[a][1] == 'synthetic'
        result['contrasts'][f'{a} - {b}'] = {k: interval((units[a][u][k]-units[b][u][k] for u in shared), synthetic)
                                            for k in ('G_D', 'Lambda')}
        result['contrasts'][f'{a} - {b}']['shared_units'] = len(shared)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=1)+'\n')

    def fmt(s):
        return f"{s['mean']:.3f}" + (f" [{s['ci_low']:.3f}, {s['ci_high']:.3f}]" if s['ci_low'] is not None else ' (SE=0)')
    for name, e in result['cases'].items():
        print(name, 'G_D', fmt(e['G_D']), 'U_M', fmt(e['U_M']), 'U_L', fmt(e['U_L']), 'I', fmt(e['I_rec']),
              e.get('transitions_distinct_items') or e.get('question_changes_dd_to_dc'))
    for name, e in result['contrasts'].items():
        print(name, 'dG_D', fmt(e['G_D']), 'dLambda', fmt(e['Lambda']), e['shared_units'])


if __name__ == '__main__':
    main()
