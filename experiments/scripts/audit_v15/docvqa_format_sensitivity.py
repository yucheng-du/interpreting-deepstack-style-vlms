#!/usr/bin/env python3
"""Post-hoc DocVQA format check: exact matching with the scorer's case-and-punctuation rule.

Streams archived v15 task-pack observations (no inference) and reuses the v15 scorer's
invalid policy, normalization and clustered normal intervals. Exploratory and post hoc.
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
SCORER = Path(__file__).resolve().parents[1]/'score_followup_benchmark_v15.py'
OUT = Path(os.environ.get('DS_DATA_ROOT', 'work')) / 'analysis/calibrated/docvqa_format_sensitivity.json'
JOBS = {'ibm4b': 'ibm4b-docvqa-evaluation-fresh.attempt-2', 'qwen8b': 'qwen8b-docvqa-evaluation-fresh.attempt-2',
        'qwen32b': 'qwen32b-docvqa-evaluation-fresh.attempt-2', 'deepstack7b': 'deepstack7b-docvqa-evaluation-fresh.attempt-2'}
LABELS = ('lowres-long360', 'lowres-long256', 'lowres-long180')

spec = importlib.util.spec_from_file_location('scorer', SCORER)
scorer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scorer)


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


def score(row, rule):
    text = row['result'].get('generated_text')
    if not isinstance(text, str) or scorer.invalid_reasons(row):
        return 0.
    refs = row['reference_answers']
    if rule == 'exact_lower_strip':
        return scorer.metric_values('docvqa', text, refs)['docvqa_exact_lower_strip']
    if rule == 'anls_organizer':
        return scorer.metric_values('docvqa', text, refs)['docvqa_anls_organizer_framework']
    if rule == 'exact_case_punct':
        p = scorer.gqa_case_punct(text)
        return float(bool(p) and p != 'none' and p in [scorer.gqa_case_punct(r) for r in refs])
    # Decomposition of the case_punct rule: one trailing period only, other final marks only, outer quotes only.
    norm = {'exact_period_only': lambda s: s.lower().strip()[:-1].rstrip() if s.lower().strip().endswith('.') else s.lower().strip(),
            'exact_other_marks_only': lambda s: s.lower().strip().rstrip('!?').strip(),
            'exact_quotes_only': lambda s: (lambda x: x[1:-1].strip() if len(x) >= 2 and (x[0], x[-1]) in scorer.QUOTES else x)(s.lower().strip())}
    if rule in norm:
        p = norm[rule](text)
        return float(bool(p) and p != 'none' and p in [norm[rule](r) for r in refs])
    raise ValueError(rule)


def trailing_period(row):
    text = row['result'].get('generated_text')
    return float(isinstance(text, str) and text.strip().endswith('.'))


def main():
    result = {}
    for model, job in JOBS.items():
        rows = defaultdict(dict)
        for row in observations(job):
            rows[row['id']][row['condition']['name']] = row
        result[model] = {}
        for label in LABELS:
            cells = {'CC': 'recovery__clean__clean', 'CD': f'recovery__clean__{label}',
                     'DC': f'recovery__{label}__clean', 'DD': f'recovery__{label}__{label}'}
            entry = {}
            for rule in ('anls_organizer', 'exact_lower_strip', 'exact_case_punct', 'exact_period_only',
                         'exact_other_marks_only', 'exact_quotes_only', 'trailing_period'):
                by_unit = defaultdict(list)
                for qid, conds in rows.items():
                    if not all(name in conds for name in cells.values()):
                        raise ValueError(f'{model} {label}: incomplete question {qid}')
                    vals = {c: (trailing_period(conds[n]) if rule == 'trailing_period' else score(conds[n], rule))
                            for c, n in cells.items()}
                    vals['G_D'] = vals['DC']-vals['DD']
                    vals['I_rec'] = (vals['DC']-vals['DD'])-(vals['CC']-vals['CD'])
                    vals['Lambda'] = vals['CC']-vals['DD']
                    by_unit[conds[cells['CC']]['unit_id']].append(vals)
                units = {u: {k: statistics.mean(v[k] for v in qs) for k in qs[0]} for u, qs in by_unit.items()}
                entry[rule] = {k: scorer.stat(u[k] for u in units.values()) for k in ('CC', 'CD', 'DC', 'DD', 'Lambda', 'G_D', 'I_rec')}
                entry[rule]['n_questions'] = sum(len(v) for v in by_unit.values())
            result[model][label] = entry
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=1)+'\n')
    for model in result:
        e = result[model]['lowres-long180']
        print(model, {r: {k: round(e[r][k]['mean'], 3) for k in ('CC', 'CD', 'DC', 'DD', 'I_rec')} | {'I_ci': [round(e[r]['I_rec']['ci_low'], 3), round(e[r]['I_rec']['ci_high'], 3)]} for r in e if r != 'n'})


if __name__ == '__main__':
    main()
