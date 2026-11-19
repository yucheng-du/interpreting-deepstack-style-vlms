#!/usr/bin/env python3
"""Write the calibrated follow-up's supplementary tables from the archived v15 final tables.

Every printed value is copied from main_table.csv, sensitivities.csv, grid.csv,
capability_calibrated.csv or sample_sizes.csv and rounded to three decimals with
Python formatting, as in the v15 report; nothing is re-estimated.
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from analysis.paths import INPUT, OUT as RENDERED, output_dir
PAPER = RENDERED
V15 = INPUT / 'calibrated'
SOURCES = {name: V15/f'{name}.csv' for name in
           ('main_table', 'sensitivities', 'grid', 'capability_calibrated', 'sample_sizes')}
BUILD = PAPER
OUTPUT = PAPER/'calibrated_conflict.tex'
CSV_DIR = PAPER/'supplementary_data'
GRID_OUTPUT = PAPER/'calibrated_grid.tex'
QA = PAPER/'qa/calibrated_tables.json'

MODELS = [('qwen8b', 'Qwen8'), ('qwen32b', 'Qwen32'), ('ibm4b', 'Granite'), ('deepstack7b', 'DeepStack')]
MODEL_NAME = dict(MODELS)
SYNTHETIC = [('color', 'Color'), ('stripe_orientation', 'Stripes'),
             ('relative_position', 'Relative position'), ('text_reading', 'Text')]
BENCHMARKS = [('gqa_attribute', 'GQA attribute'), ('gqa_spatial', 'GQA spatial'),
              ('docvqa_document_qa', 'DocVQA')]
TASK_NAME = dict(SYNTHETIC+BENCHMARKS)
SHORT_TASK = {'relative_position': 'Rel.\\ position', 'text_reading': 'Text', 'gqa_attribute': 'GQA attr.',
              'gqa_spatial': 'GQA spatial', 'docvqa_document_qa': 'DocVQA'}
READOUTS = [('direct/none', 'Direct'), ('label/identity', 'Identity'), ('label/cyclic', 'Cyclic')]
KINDS = [('generation', 'Gen.'), ('score', 'Score')]
# calibrate.py: no native-capable primary entry, excluded from the grid median.
NON_NATIVE = {('deepstack7b', t) for t in ('stripe_orientation', 'relative_position', 'text_reading')}
METRICS = {'gqa_exact': 'Exact', 'gqa_case_punct_normalized': 'C+P',
           'gqa_lower_strip_sensitivity': 'L+S',
           'docvqa_anls_organizer_framework': 'ANLS', 'docvqa_anls_strict_paper_cutoff': 'Strict ANLS',
           'docvqa_exact_lower_strip': 'Exact'}
TRACE = []
EXPORTS = []


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(name):
    with SOURCES[name].open() as handle:
        return list(csv.DictReader(handle))


def num(row, column, source, key):
    """Format one copied value and record where it came from."""
    text = row[column]
    if text in ('', 'NA'):
        return None
    value = f'{float(text):.3f}'
    TRACE.append({'source': source, 'row': key, 'column': column, 'raw': text, 'printed': value})
    return value


def math(value):
    return '--' if value is None else f'${value}$'


def interval(row, stem, source, key, se_zero):
    mean = num(row, stem+'_mean', source, key)
    low, high = num(row, stem+'_ci_low', source, key), num(row, stem+'_ci_high', source, key)
    if mean is None:
        return '--'
    if low is None:
        if not se_zero:
            raise ValueError(f'Missing interval without SE=0: {key} {stem}')
        # Zero standard error: a zero-width interval, the notation of the main text.
        return f'${mean}$ $[{mean}, {mean}]$'
    return f'${mean}$ $[{low}, {high}]$'


def level_text(task, label, setting):
    value = float(setting)
    text = f'{value:g}'
    return text if task in dict(SYNTHETIC) else f'{text} px'


def se_index():
    index = {}
    for row in read('sample_sizes'):
        key = (row['source'], row['model'], row['task'], row['readout'], row['kind'],
               row['row_type'], row['severity_label'], row['main_table_column'])
        index[key] = row
    return index


def longtable(columns, header, caption, label, size='\\footnotesize'):
    count = len(header)
    head = ' & '.join(header)+' \\\\'
    return [
        '{', size, '\\setlength{\\tabcolsep}{3.2pt}', '\\renewcommand{\\arraystretch}{1.02}',
        '\\setlength{\\LTleft}{\\fill}', '\\setlength{\\LTright}{\\fill}',
        f'\\begin{{longtable}}{{{columns}}}',
        f'\\caption{{{caption}}}\\label{{{label}}}\\\\',
        '\\toprule', '\\rowcolor{DSTableHeader}', head, '\\midrule', '\\endfirsthead',
        f'\\multicolumn{{{count}}}{{c}}{{\\tablename\\ \\thetable{{}} -- continued}}\\\\',
        '\\toprule', '\\rowcolor{DSTableHeader}', head, '\\midrule', '\\endhead',
        f'\\midrule\\multicolumn{{{count}}}{{r}}{{Continued on next page}}\\\\', '\\endfoot',
        '\\bottomrule', '\\endlastfoot']


def block(title, count):
    return f'\\rowcolor{{DSTableAccent}}\\multicolumn{{{count}}}{{l}}{{\\textbf{{{title}}}}}\\\\*'


def write_csv(name, fields, rows):
    """Copy source values verbatim (full precision) into one supplementary CSV."""
    CSV_DIR.mkdir(exist_ok=True)
    path = CSV_DIR/name
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), lineterminator='\n')
        writer.writeheader()
        for row in rows:
            writer.writerow({out: ('' if row.get(src) in (None, 'NA') else row[src]) for out, src in fields.items()})
    EXPORTS.append({'file': name, 'rows': len(rows), 'sha256': sha(path)})


def stat_fields(prefix, stem):
    return {prefix: stem+'_mean', prefix+'_ci_low': stem+'_ci_low', prefix+'_ci_high': stem+'_ci_high'}


def export_csvs(main_rows, sensitivities, capability):
    synthetic = sorted((r for r in main_rows if r['source'] == 'analysis'),
                       key=lambda r: (r['model'], r['task'], r['readout'], r['kind'], r['row_type'] != 'reference',
                                      int(r['severity_rank']) if r['severity_rank'] not in ('', 'NA') else 0))
    keys = {'model': 'model', 'task': 'task', 'readout': 'readout', 'kind': 'kind'}
    level = {'level': 'severity_label', 'level_value': 'setting_value', 'in_shared_grid': 'in_shared_grid'}
    recovery = dict(keys, **level, cc='cc_mean', cd='cd_mean', dc='dc_mean', dd='dd_mean',
                    joint_loss='cc_minus_dd_mean', **stat_fields('g_d', 'dc_minus_dd'),
                    **stat_fields('i_rec', 'recovery_interaction'), n_templates='dc_minus_dd_n')
    write_csv('calibrated_synthetic_recovery.csv', recovery, [r for r in synthetic if r['row_type'] == 'label'])
    conflict = dict(keys, main_state='row_type', **level, **stat_fields('target_effect', 'conflict_T'),
                    **stat_fields('control_effect', 'conflict_C'), **stat_fields('selectivity_r', 'conflict_R'),
                    n_r='conflict_R_n', **stat_fields('i_quality', 'delta_R'), n_i_quality='delta_R_n')
    write_csv('calibrated_synthetic_conflict.csv', conflict, synthetic)
    bench = {'model': 'model', 'task': 'task', **level, 'metric': 'metric', 'policy': 'policy',
             'n_clusters': 'cc_minus_dd_n', 'cc': 'cc_mean', 'cd': 'cd_mean', 'dc': 'dc_mean', 'dd': 'dd_mean',
             **stat_fields('joint_loss', 'cc_minus_dd'), **stat_fields('g_d', 'dc_minus_dd'),
             **stat_fields('i_rec', 'recovery_interaction')}
    write_csv('calibrated_benchmarks.csv', bench,
              sorted(sensitivities, key=lambda r: (r['task'], r['model'], int(r['severity_rank']), r['metric'], r['policy'])))
    cap = {'task': 'calibration_task', 'model': 'calibration_model', 'level': 'calibrated_label',
           'level_value': 'calibrated_value', 'retained_ability_r': 'calibration_r_mean',
           'in_shared_grid': 'label_in_shared_grid'}
    unique = {}
    for row in capability:
        unique.setdefault((row['calibration_task'], row['calibration_model']), row)
    write_csv('calibrated_capability_levels.csv', cap, [unique[k] for k in sorted(unique)])


def grid_table():
    rows = [r for r in read('grid') if r['in_shared_grid'] == 'True']
    lines = ['\\begin{table}[!htb]', '\\centering\\footnotesize', '\\setlength{\\tabcolsep}{3pt}',
             '\\renewcommand{\\arraystretch}{1.02}',
             '\\caption{Shared degradation levels chosen on calibration units from clean and jointly degraded '
             'cells only. Retained ability is $r=(A_{DD}-\\text{chance})/(A_{CC}-\\text{chance})$, with chance '
             '$0.5$ for two-choice tasks and $0$ for free-form tasks; each model averages its native-capable '
             'readouts. The three levels jointly minimize the summed distance of the cross-model median $r$ '
             'from $0.75$, $0.5$ and $0.25$. Levels give retained chroma (color), blur $\\sigma$ relative to '
             'the stripe period or dot separation, equivalent font size, or long side. NN marks non-native '
             'models excluded from the median. Q8, Q32, Gr.\\ and DS abbreviate the four models.}',
             '\\label{tab:v15_grid}',
             '\\begin{tabular}{@{}llrrrrr@{}}', '\\toprule', '\\rowcolor{DSTableHeader}',
             'Task & Level & Median $r$ & Q8 & Q32 & Gr. & DS \\\\', '\\midrule']
    order = [t for t, _ in SYNTHETIC+BENCHMARKS]
    for task in order:
        selected = sorted((r for r in rows if r['task'] == task), key=lambda r: int(r['grid_position']))
        if len(selected) != 3:
            raise ValueError('Expected three grid levels for '+task)
        for i, row in enumerate(selected):
            key = {'task': task, 'label': row['label']}
            cells = [SHORT_TASK.get(task, TASK_NAME[task]) if i == 0 else '', level_text(task, row['label'], row['value']),
                     math(num(row, 'median_r', 'grid', key))]
            for model, _ in MODELS:
                value = num(row, 'r_'+model, 'grid', key)
                cells.append('NN' if value is None and (model, task) in NON_NATIVE else math(value))
            lines.append(' & '.join(cells)+' \\\\')
        if task != order[-1]:
            lines.append('\\addlinespace[2pt]')
    lines += ['\\bottomrule', '\\end{tabular}', '\\end{table}']
    return lines


def recovery_rows(rows, se):
    """Group synthetic recovery rows; merge kinds only when every printed field is identical."""
    fields = ('cc_mean', 'cd_mean', 'dc_mean', 'dd_mean', 'cc_minus_dd_mean', 'dc_minus_dd_mean',
              'dc_minus_dd_ci_low', 'dc_minus_dd_ci_high', 'recovery_interaction_mean',
              'recovery_interaction_ci_low', 'recovery_interaction_ci_high')
    lines = []
    for task, task_name in SYNTHETIC:
        for model, model_name in MODELS:
            title = f'{model_name} -- {task_name}'+(' (non-native)' if (model, task) in NON_NATIVE else '')
            lines.append(block(title, 9))
            for readout, readout_name in READOUTS:
                by_kind = {kind: sorted((r for r in rows if r['model'] == model and r['task'] == task and
                                         r['readout'] == readout and r['kind'] == kind),
                                        key=lambda r: int(r['severity_rank'])) for kind, _ in KINDS}
                same = all(all(a[f] == b[f] for f in fields) for a, b in zip(by_kind['generation'], by_kind['score']))
                kinds = [('both', 'Both')] if same else KINDS
                for kind, kind_name in kinds:
                    for row in by_kind['generation' if kind == 'both' else kind]:
                        key = {k: row[k] for k in ('model', 'task', 'readout', 'kind', 'severity_label')}
                        zero = lambda column: se[('analysis', model, task, readout, row['kind'], 'label',
                                                  row['severity_label'], column)]['se_zero'] == 'True'
                        mark = '' if row['in_shared_grid'] == 'True' else '$^\\dagger$'
                        cells = [f'{kind_name} {readout_name}',
                                 level_text(task, row['severity_label'], row['setting_value'])+mark]
                        cells += [math(num(row, c+'_mean', 'main_table', key)) for c in ('cc', 'cd', 'dc', 'dd')]
                        cells.append(math(num(row, 'cc_minus_dd_mean', 'main_table', key)))
                        cells.append(interval(row, 'dc_minus_dd', 'main_table', key, zero('dc_minus_dd')))
                        cells.append(interval(row, 'recovery_interaction', 'main_table', key,
                                              zero('recovery_interaction')))
                        lines.append(' & '.join(cells)+' \\\\*[0pt]')
            lines[-1] = lines[-1].replace('\\\\*[0pt]', '\\\\[0pt]')
    return lines


def conflict_rows(rows, references, se):
    lines, exceptions = [], []
    for task, task_name in SYNTHETIC:
        for model, model_name in MODELS:
            title = f'{model_name} -- {task_name}'+(' (non-native)' if (model, task) in NON_NATIVE else '')
            lines.append(block(title, 5))
            for kind, kind_name in KINDS:
                for readout, readout_name in READOUTS:
                    ref = references[(model, task, readout, kind)]
                    ref_key = {k: ref[k] for k in ('model', 'task', 'readout', 'kind', 'row_type')}
                    ref_zero = se[('analysis', model, task, readout, kind, 'reference', 'NA',
                                   'conflict_R')]['se_zero'] == 'True'
                    cells = [f'{kind_name} {readout_name}',
                             interval(ref, 'conflict_R', 'main_table', ref_key, ref_zero)]
                    if ref['conflict_R_n'] not in ('32',):
                        exceptions.append((model, task, readout, kind, None, ref['conflict_R_n']))
                    labels = sorted((r for r in rows if r['model'] == model and r['task'] == task and
                                     r['readout'] == readout and r['kind'] == kind and r['in_shared_grid'] == 'True'),
                                    key=lambda r: int(r['severity_rank']))
                    for row in labels:
                        key = {k: row[k] for k in ('model', 'task', 'readout', 'kind', 'severity_label')}
                        zero = se[('analysis', model, task, readout, kind, 'label', row['severity_label'],
                                   'delta_R')]['se_zero'] == 'True'
                        cells.append(interval(row, 'delta_R', 'main_table', key, zero))
                        if row['delta_R_n'] != '32':
                            exceptions.append((model, task, readout, kind,
                                               level_text(task, row['severity_label'], row['setting_value']),
                                               row['delta_R_n']))
                    lines.append(' & '.join(cells)+' \\\\*[0pt]')
            lines[-1] = lines[-1].replace('\\\\*[0pt]', '\\\\[0pt]')
    return lines, exceptions


def benchmark_rows(sensitivities):
    """Primary metric plus named metric sensitivities; identical metric rows share one line."""
    fields = ('cc_mean', 'cd_mean', 'dc_mean', 'dd_mean', 'cc_minus_dd_mean', 'cc_minus_dd_ci_low',
              'cc_minus_dd_ci_high', 'dc_minus_dd_mean', 'dc_minus_dd_ci_low', 'dc_minus_dd_ci_high',
              'recovery_interaction_mean', 'recovery_interaction_ci_low', 'recovery_interaction_ci_high',
              'cc_minus_dd_n')
    metric_order = {'gqa_attribute': ['gqa_exact', 'gqa_case_punct_normalized', 'gqa_lower_strip_sensitivity'],
                    'gqa_spatial': ['gqa_exact', 'gqa_case_punct_normalized', 'gqa_lower_strip_sensitivity'],
                    'docvqa_document_qa': ['docvqa_anls_organizer_framework', 'docvqa_anls_strict_paper_cutoff',
                                           'docvqa_exact_lower_strip']}
    lines, policy_lines = [], []
    for task, task_name in BENCHMARKS:
        for model, model_name in MODELS:
            lines.append(block(f'{model_name} -- {task_name}', 9))
            labels = sorted({(int(r['severity_rank']), r['severity_label']) for r in sensitivities
                             if r['task'] == task and r['model'] == model})
            for _, label in labels:
                group = {r['metric']: r for r in sensitivities if r['task'] == task and r['model'] == model and
                         r['severity_label'] == label and r['policy'] == 'invalid_as_zero'}
                printed = []
                for metric in metric_order[task]:
                    row = group[metric]
                    match = next((p for p in printed if all(group[p[0]][f] == row[f] for f in fields)), None)
                    if match:
                        match.append(metric)
                    else:
                        printed.append([metric])
                for names in printed:
                    row = group[names[0]]
                    key = {k: row[k] for k in ('model', 'task', 'severity_label', 'metric', 'policy')}
                    mark = '' if row['in_shared_grid'] == 'True' else '$^\\dagger$'
                    cells = ['='.join(METRICS[m] for m in names),
                             level_text(task, label, row['setting_value'])+mark]
                    cells += [math(num(row, c+'_mean', 'sensitivities', key)) for c in ('cc', 'cd', 'dc', 'dd')]
                    cells += [interval(row, s, 'sensitivities', key, False)
                              for s in ('cc_minus_dd', 'dc_minus_dd', 'recovery_interaction')]
                    lines.append(' & '.join(cells)+' \\\\*[0pt]')
                # Policy variants: print only those that differ from the same metric's primary-policy row.
                for metric in metric_order[task]:
                    base = group[metric]
                    for policy in ('valid_cells_only', 'score_returned_strings'):
                        row = next(r for r in sensitivities if r['task'] == task and r['model'] == model and
                                   r['severity_label'] == label and r['metric'] == metric and r['policy'] == policy)
                        if all(row[f] == base[f] for f in fields+('cc_n',)):
                            continue
                        key = {k: row[k] for k in ('model', 'task', 'severity_label', 'metric', 'policy')}
                        cells = [f'{model_name}', level_text(task, label, row['setting_value']),
                                 METRICS[metric]+(', valid cells' if policy == 'valid_cells_only' else ', returned'),
                                 row['cc_minus_dd_n']]
                        cells += [math(num(row, c+'_mean', 'sensitivities', key)) for c in ('cc', 'cd', 'dc', 'dd')]
                        cells += [interval(row, s, 'sensitivities', key, False)
                                  for s in ('dc_minus_dd', 'recovery_interaction')]
                        policy_lines.append((task, ' & '.join(cells)+' \\\\'))
            lines[-1] = lines[-1].replace('\\\\*[0pt]', '\\\\[0pt]')
    return lines, policy_lines


def off_note(rows):
    parts = [f"{TASK_NAME[r['task']].lower()} {dict(READOUTS)[r['readout']].lower()} generation $I_{{\\rm quality}}$ "
             f"uses {r['delta_R_n']} templates" for r in rows if r['delta_R_n'] != '32' or r['conflict_R_n'] != '32']
    return 'invalid generations reduce $n$: '+'; '.join(parts)


def capability_table(rows):
    entries = defaultdict(dict)
    for row in rows:
        task, model = row['calibration_task'], row['calibration_model']
        key = {'calibration_task': task, 'calibration_model': model}
        r = num(row, 'calibration_r_mean', 'capability_calibrated', key)
        level = level_text(task, row['calibrated_label'], row['calibrated_value'])
        text = f'{level} ({r})'+('' if row['label_in_shared_grid'] == 'True' else '$^\\dagger$')
        previous = entries[task].get(model)
        if previous is not None and previous != text:
            raise ValueError('Inconsistent capability-calibrated label')
        entries[task][model] = text
    lines = ['\\begin{table}[htbp]', '\\centering\\small', '\\setlength{\\tabcolsep}{6pt}',
             '\\renewcommand{\\arraystretch}{1.04}',
             '\\caption{Model-specific capability-calibrated levels: for each model and task, the calibration '
             'level whose mean retained ability $r$ is nearest $0.5$ (shown in parentheses). This is a separate, '
             'model-specific estimand. Levels already in the shared grid reuse the grid rows of '
             'Tables~\\ref{tab:v15_recovery}--\\ref{tab:v15_benchmark}; $^\\dagger$ marks the four levels run '
             'outside the grid, whose rows are marked there. Non-native entries have none.}',
             '\\label{tab:v15_capability}', '\\begin{tabular}{@{}lllll@{}}', '\\toprule', '\\rowcolor{DSTableHeader}',
             'Task & Qwen8 & Qwen32 & Granite & DeepStack \\\\', '\\midrule']
    for task, name in SYNTHETIC+BENCHMARKS:
        cells = [name]+[entries[task].get(model, 'NN' if (model, task) in NON_NATIVE else '--')
                        for model, _ in MODELS]
        lines.append(' & '.join(cells)+' \\\\')
    lines += ['\\bottomrule', '\\end{tabular}', '\\end{table}']
    return lines


def main():
    main_rows, se = read('main_table'), se_index()
    synthetic = [r for r in main_rows if r['source'] == 'analysis' and r['row_type'] == 'label']
    references = {(r['model'], r['task'], r['readout'], r['kind']): r for r in main_rows
                  if r['source'] == 'analysis' and r['row_type'] == 'reference'}
    sensitivities = read('sensitivities')
    header = '% Generated by v15_revision/build_v15_tables.py from the archived v15 final tables; do not edit.'
    export_csvs(main_rows, sensitivities, read('capability_calibrated'))
    GRID_OUTPUT.write_text('\n'.join([header]+grid_table())+'\n')
    # Compact table behind the main text's color-score statement; every other slice is in the CSV files.
    color = [r for r in synthetic if r['task'] == 'color' and r['kind'] == 'score' and r['in_shared_grid'] == 'True']
    out = [header, '\\begin{table*}[t]', '\\centering\\footnotesize', '\\setlength{\\tabcolsep}{4pt}',
           '\\renewcommand{\\arraystretch}{1.04}',
           '\\caption{\\textbf{Degrading main raises later color score selectivity for every model, readout and level: all 36 '
           '$I_{\\rm quality}$ intervals lie above zero.} '
           'Color conflict scores in the calibrated experiment. $R$ is the later target-minus-control effect '
           'on the candidate score with clean main; $I_{\\rm quality}=R(\\text{degraded main})-R(\\text{clean main})$ '
           'at the three shared desaturation levels of Table~\\ref{tab:v15_grid}, mild to severe. Intervals are descriptive, unadjusted 95\\% Student-$t$ '
           'intervals over 32 templates. Every task, readout, kind, level and the target and control components are '
           'in \\texttt{calibrated\\_\\allowbreak{}synthetic\\_\\allowbreak{}conflict.csv}.}',
           '\\label{tab:v15_conflict}', '\\begin{tabular}{@{}llcccc@{}}', '\\toprule', '\\rowcolor{DSTableHeader}',
           'Model & Readout & $R$, clean main & $I_{\\rm quality}$, level 1 & Level 2 & Level 3 \\\\', '\\midrule']
    lows = []
    for model, model_name in MODELS:
        for j, (readout, readout_name) in enumerate(READOUTS):
            ref = references[(model, 'color', readout, 'score')]
            ref_key = {k: ref[k] for k in ('model', 'task', 'readout', 'kind', 'row_type')}
            ref_zero = se[('analysis', model, 'color', readout, 'score', 'reference', 'NA', 'conflict_R')]['se_zero'] == 'True'
            cells = [model_name if j == 0 else '', readout_name, interval(ref, 'conflict_R', 'main_table', ref_key, ref_zero)]
            for row in sorted((r for r in color if r['model'] == model and r['readout'] == readout),
                              key=lambda r: int(r['severity_rank'])):
                key = {k: row[k] for k in ('model', 'task', 'readout', 'kind', 'severity_label')}
                zero = se[('analysis', model, 'color', readout, 'score', 'label', row['severity_label'],
                           'delta_R')]['se_zero'] == 'True'
                cells.append(interval(row, 'delta_R', 'main_table', key, zero))
                lows.append(float(row['delta_R_ci_low']))
            out.append(' & '.join(cells)+' \\\\')
        if model != MODELS[-1][0]:
            out.append('\\addlinespace[2pt]')
    out += ['\\bottomrule', '\\end{tabular}', '\\end{table*}']
    if len(lows) != 36 or min(lows) <= 0:
        raise ValueError('The caption states that all 36 color score intervals lie above zero')
    exceptions = []
    OUTPUT.write_text('\n'.join(out)+'\n')
    QA.parent.mkdir(parents=True, exist_ok=True)
    QA.write_text(json.dumps({'schema': 'v15-paper-table-sources-1',
                              'generator': str(Path(__file__).resolve()), 'generator_sha256': sha(Path(__file__)),
                              'sources': {k: {'path': str(v), 'sha256': sha(v)} for k, v in SOURCES.items()},
                              'output': str(OUTPUT), 'output_sha256': sha(OUTPUT),
                              'grid_output': str(GRID_OUTPUT), 'grid_output_sha256': sha(GRID_OUTPUT),
                              'printed_values': len(TRACE), 'conflict_n_exceptions': exceptions,
                              'csv_exports': EXPORTS,
                              'trace': TRACE}, indent=1)+'\n')
    print(json.dumps({'output': str(OUTPUT), 'printed_values': len(TRACE), 'lines': len(out),
                      'csv_exports': [(e['file'], e['rows']) for e in EXPORTS]}))


if __name__ == '__main__':
    output_dir()
    main()
