#!/usr/bin/env python3
"""Typeset accepted v14 estimates and paired CPU analyses; never re-estimate."""
from pathlib import Path
import csv
import hashlib
import json

from analysis.paths import INPUT, OUT as RENDERED, output_dir
HERE = RENDERED / 'fixed'
SOURCE = INPUT / 'fixed'
OUT = HERE
MODELS = ['qwen8b', 'qwen32b', 'ibm4b', 'deepstack7b']
NAMES = {'qwen8b': 'Qwen8', 'qwen32b': 'Qwen32', 'ibm4b': 'Granite', 'deepstack7b': 'DeepStack'}
FORMATS = ['direct', 'identity', 'cyclic']
FNAMES = {'direct': 'Direct', 'identity': 'Identity', 'cyclic': 'Cyclic'}
TASKS = ['color', 'relative_position', 'stripe_orientation', 'text_reading', 'natural_scene_color']
TNAMES = {'color': 'Color', 'relative_position': 'Relative position', 'stripe_orientation': 'Stripes',
          'text_reading': 'Text', 'natural_scene_color': 'Natural color'}
CELLS = ['clean__clean', 'clean__degraded', 'degraded__clean', 'degraded__degraded']
recorded = []
# Complete estimates ship as CSV in the supplementary archive; the PDF keeps compact tables.
CSV_DIR = HERE.parent / 'supplementary_data'
exports = []


def stat_columns(prefix, s):
    """Mean, archived interval and SE=0 flag of one statistic object, unrounded."""
    if not s or s.get('mean') is None:
        return {prefix: '', prefix + '_ci_low': '', prefix + '_ci_high': '', prefix + '_se_zero': ''}
    ci = s.get('ci95_t')
    if ci is None and s.get('ci_low') is not None:
        ci = [s['ci_low'], s['ci_high']]
    zero = bool(s.get('se_zero') or s.get('se') == 0)
    low, high = (ci or ['', ''])
    return {prefix: s['mean'], prefix + '_ci_low': '' if zero else low, prefix + '_ci_high': '' if zero else high,
            prefix + '_se_zero': zero}


def write_csv(name, rows):
    CSV_DIR.mkdir(exist_ok=True)
    path = CSV_DIR / name
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    exports.append({'file': name, 'rows': len(rows), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})


def escape(text):
    return str(text).replace('_', r'\_').replace('%', r'\%').replace('&', r'\&')


def mean(s, scale=1, digits=3, source=''):
    if not s or s.get('mean') is None or s.get('n', s.get('n_clusters', 1)) == 0:
        return '--'
    recorded.append({'source': source, 'stat': s, 'scale': scale, 'presentation': 'mean'})
    return f"{s['mean'] * scale:.{digits}f}"


def estimate(s, scale=1, digits=3, source='', compact=True):
    if not s or s.get('mean') is None or s.get('n', s.get('n_clusters', 1)) == 0:
        return '--'
    recorded.append({'source': source, 'stat': s, 'scale': scale, 'presentation': 'mean_interval'})
    m = s['mean'] * scale
    first = f'${m:.{digits}f}$'
    ci = s.get('ci95_t')
    if ci is None and 'ci_low' in s and s['ci_low'] is not None:
        ci = [s['ci_low'], s['ci_high']]
    if s.get('se_zero') or s.get('se') == 0:
        # Zero standard error: a zero-width interval, the notation of the main text.
        z = m + 0.0
        second = f'$[{z:.{digits}f}, {z:.{digits}f}]$'
    elif ci is not None:
        second = f'$[{ci[0] * scale:.{digits}f}, {ci[1] * scale:.{digits}f}]$'
    else:
        second = r'\textit{CI unavailable}'
    if compact:
        return first + ' ' + second
    return r'\begin{tabular}[c]{@{}c@{}}' + first + r'\\' + second + r'\end{tabular}'


def one(items, **conditions):
    found = [x for x in items if all(x.get(k) == v for k, v in conditions.items())]
    assert len(found) == 1, (conditions, len(found))
    return found[0]


def group_index(groups):
    return {(g['model'], g['material'], g['task'], g['kind'], g['readout']): g
            for g in groups if 'line1' in g['job']}




def color_cells_table(groups):
    ix = group_index(groups)
    lines = [r'\begin{table}[!htb]', r'\centering\footnotesize',
             r'\setlength{\tabcolsep}{5pt}', r'\renewcommand{\arraystretch}{1.02}',
             r"\caption{\textbf{Qwen's color accuracy stays at ceiling except under cyclic labels.} "
             r'Synthetic-color generation accuracy (\%) at contrast $0.12$. '
             r'C and D denote clean and degraded sources, with main first and later second. '
             r'All 32 templates contribute in every format; invalid answers count as incorrect. '
             r'Paired contrasts and intervals for every task are in \texttt{fixed\_\allowbreak{}contrast\_\allowbreak{}recovery.csv}.}',
             r'\label{tab:v14_color_cells}',
             r'\begin{tabular}{@{}llrrrr@{}}', r'\toprule', r'\rowcolor{DSTableHeader}',
             r'Model & Format & CC & CD & DC & DD \\', r'\midrule']
    for model in MODELS:
        for j, fmt in enumerate(FORMATS):
            g = ix[model, 'synthetic', 'color', 'generation', fmt]
            cs = g['analysis']['recovery_clean_reference_accuracy_invalid_as_incorrect']
            cells = {x['main_quality'] + '__' + x['later_quality']: x for x in cs}
            if j == 0 and model != MODELS[0]: lines.append(r'\addlinespace[2pt]')
            vals = [NAMES[model] if j == 0 else '', FNAMES[fmt]]
            vals += [mean(cells[k], 100, 1, g['group_id'] + '/' + k) for k in CELLS]
            lines.append(' & '.join(vals) + r' \\[0pt]')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    (HERE / 'table_color_cells.tex').write_text('\n'.join(lines) + '\n')


def longtable(title, label, columns, header, ncols, rows):
    # Each shaded group is short enough to fit a page. Keep its identity with all rows.
    rows = list(rows)
    for i, row in enumerate(rows[:-1]):
        if not rows[i + 1].startswith(r'\rowcolor{DSTableAccent}'):
            rows[i] = row.replace(r'\\[', r'\\*[')
    return ['{', r'\small', r'\setlength{\tabcolsep}{3.0pt}', r'\renewcommand{\arraystretch}{1.08}',
            r'\setlength{\LTleft}{\fill}', r'\setlength{\LTright}{\fill}',
            r'\begin{longtable}{' + columns + '}',
            r'\caption{' + title + r'}\label{' + label + r'}\\',
            r'\toprule', r'\rowcolor{DSTableHeader}', header + r' \\', r'\midrule', r'\endfirsthead',
            r'\multicolumn{' + str(ncols) + r'}{c}{\tablename\ \thetable{} -- continued}\\',
            r'\toprule', r'\rowcolor{DSTableHeader}', header + r' \\', r'\midrule', r'\endhead',
            r'\midrule\multicolumn{' + str(ncols) + r'}{r}{Continued on next page}\\', r'\endfoot',
            r'\bottomrule', r'\endlastfoot'] + rows + [r'\end{longtable}', '}']


def grouped_rows(groups, kind=None):
    rows = []
    for model in MODELS:
        for task in TASKS:
            match = [g for g in groups if 'line1' in g['job'] and g['model'] == model and g['task'] == task and (kind is None or g['kind'] == kind)]
            if match:
                rows.append((model, task, sorted(match, key=lambda g: (['generation', 'score'].index(g['kind']), FORMATS.index(g['readout'])))))
    return rows


def appendix_line1(groups):
    line1 = sorted((g for g in groups if 'line1' in g['job']),
                   key=lambda g: (MODELS.index(g['model']), TASKS.index(g['task']), g['kind'], FORMATS.index(g['readout'])))
    recovery, conflict = [], []
    for g in line1:
        a = g['analysis']
        base = {'model': g['model'], 'material': g['material'], 'task': g['task'], 'kind': g['kind'], 'readout': g['readout']}
        cs = {x['main_quality'] + '__' + x['later_quality']: x for x in a['recovery_clean_reference_accuracy_invalid_as_incorrect']}
        rc = a['recovery_contrasts']['accuracy_invalid_as_incorrect']
        row = dict(base, cc=cs['clean__clean']['mean'], cd=cs['clean__degraded']['mean'],
                   dc=cs['degraded__clean']['mean'], dd=cs['degraded__degraded']['mean'])
        row.update(stat_columns('g_d', rc['clean_later_benefit_with_degraded_main'][0]))
        row.update(stat_columns('i_rec', rc['quality_interaction'][0]))
        recovery.append(row)
        r = one(a['selectivity_R'], main_quality='degraded')
        row = dict(base, n_r=r['n'])
        row.update(stat_columns('target_effect', one(a['conflict_query_effects'], main_quality='degraded', query_role='changed')))
        row.update(stat_columns('control_effect', one(a['conflict_query_effects'], main_quality='degraded', query_role='unchanged')))
        row.update(stat_columns('selectivity_r', r))
        row.update(stat_columns('i_quality', one(a['selectivity_interaction'], main_quality='degraded')))
        conflict.append(row)
    write_csv('fixed_contrast_recovery.csv', recovery)
    write_csv('fixed_contrast_conflict.csv', conflict)
    columns = [('color', 'score', 'direct'), ('color', 'score', 'identity'), ('color', 'score', 'cyclic'),
               ('color', 'generation', 'direct'), ('natural_scene_color', 'score', 'direct')]
    lines = [r'\Needspace{6\baselineskip}\subsection{Conflict quality interactions}',
             r'Table~\ref{tab:v14_conflict_full} gives the quality interactions cited in the main text. '
             r'The files \texttt{fixed\_\allowbreak{}contrast\_\allowbreak{}conflict.csv} and \texttt{fixed\_\allowbreak{}contrast\_\allowbreak{}recovery.csv} '
             r'give every task, kind and readout, with target and control components and four-cell recovery estimates.',
             r'\begin{table*}[t]', r'\centering\footnotesize', r'\setlength{\tabcolsep}{4pt}', r'\renewcommand{\arraystretch}{1.04}',
             r'\caption{\textbf{Degrading main raises later score selectivity on synthetic color in all four models.} '
             r'Quality interaction $I_{\rm quality}=R_{\later}(D)-R_{\later}(C)$ in the fixed-contrast experiment: '
             r'the change in later target-minus-control selectivity when main is degraded to contrast 0.12 while the later '
             r'features stay fixed. Score values are candidate log odds; generation values are directed answer-frequency '
             r'changes. Synthetic color uses 32 templates and photographs 64, with complete units only. Intervals are '
             r'descriptive, unadjusted 95\% $t$ intervals.}',
             r'\label{tab:v14_conflict_full}', r'\begin{tabular}{@{}lccccc@{}}', r'\toprule',
             r' & \multicolumn{4}{c}{Synthetic color} & Photographs \\', r'\cmidrule(lr){2-5}\cmidrule(l){6-6}',
             r'\rowcolor{DSTableHeader}', r'Model & Score, direct & Score, identity & Score, cyclic & Gen., direct & Score, direct \\',
             r'\midrule']
    for model in MODELS:
        cells = [NAMES[model]]
        for task, kind, fmt in columns:
            g = one(line1, model=model, task=task, kind=kind, readout=fmt)
            cells.append(estimate(one(g['analysis']['selectivity_interaction'], main_quality='degraded'),
                                  source=g['group_id'] + '/Iquality'))
        lines.append(' & '.join(cells) + r' \\')
    return lines + [r'\bottomrule', r'\end{tabular}', r'\end{table*}']


def appendix_marginal(paper, groups):
    rows, lines = [], []
    for task in TASKS[:4]:
        for kind in ['generation', 'score']:
            for fmt in FORMATS:
                g = one(paper['granite_conditional_marginal'], task=task, kind=kind, readout=fmt)
                m = g['marginal'][0]
                native = one(groups, model='ibm4b', material='synthetic', task=task, kind=kind, readout=fmt,
                             job='ibm4b-synthetic-line1-batched')['analysis']['native_clean_reference_accuracy']['accuracy_invalid_as_incorrect']
                row = {'task': task, 'kind': kind, 'readout': fmt, 'n': m['target_minus_control']['n'],
                       'native_accuracy': native['mean']}
                row.update(stat_columns('target_effect', one(m['query_effects'], query_role='changed')))
                row.update(stat_columns('control_effect', one(m['query_effects'], query_role='unchanged')))
                row.update(stat_columns('marginal', m['target_minus_control']))
                rows.append(row)
                if kind == 'generation':
                    lines.append(' & '.join([TNAMES[task] if fmt == 'direct' else '', FNAMES[fmt], str(m['target_minus_control']['n']),
                                             mean(native, source=g['group_id'] + '/native-line1'),
                                             estimate(m['target_minus_control'], source=g['group_id'] + '/M')]) + r' \\')
            if task != TASKS[3] and kind == 'generation':
                lines.append(r'\addlinespace[2pt]')
    write_csv('fixed_contrast_granite_marginal.csv', rows)
    return [r'\Needspace{6\baselineskip}\subsection{Granite conditional marginal}',
            r"Table~\ref{tab:v14_marginal_full} gives the generation effect of adding Granite's entry 21 while entries 15 and 18 stay fixed.",
            r'\begin{table}[!htb]', r'\centering\footnotesize', r'\setlength{\tabcolsep}{3.5pt}', r'\renewcommand{\arraystretch}{1.02}',
            r'\caption{\textbf{Entry 21 adds selectivity in three of the twelve task and readout cells.} '
            r'Granite entry-21 marginal given fixed entries 15 and 18: paired target-minus-control generation effect, '
            r'not a share of a joint effect. Native accuracy comes from the separate fixed-contrast condition. Units and $t$ '
            r'intervals follow that panel; score marginals and components are in \texttt{fixed\_\allowbreak{}contrast\_\allowbreak{}granite\_\allowbreak{}marginal.csv}.}',
            r'\label{tab:v14_marginal_full}', r'\begin{tabular}{@{}llrrc@{}}', r'\toprule', r'\rowcolor{DSTableHeader}',
            r'Task & Readout & $n$ & Native & Marginal \\', r'\midrule'] + lines + [r'\bottomrule', r'\end{tabular}', r'\end{table}']


def appendix_pairs(paper):
    rows = []
    for g in paper['paired_color_recovery']:
        row = {k: g.get(k, '') for k in ('model', 'material', 'task', 'kind', 'metric', 'invalid_policy', 'contrast')}
        row['n'] = g.get('n', '')
        row.update(stat_columns('difference', g))
        rows.append(row)
    write_csv('fixed_contrast_format_pairs.csv', rows)
    return []


def appendix_benchmark(bench):
    rows, lines = [], []
    reports = sorted(bench['reports'], key=lambda x: (x['task'], MODELS.index(x['model'])))
    for report in reports:
        for policy, data in report['policies'].items():
            for metric, values in data['metrics'].items():
                e = values['estimates']
                row = {'model': report['model'], 'task': report['task'], 'metric': metric, 'policy': policy,
                       'primary_metric': metric == report['primary_metric'], 'n_clusters': e['clean__clean'].get('n_clusters', ''),
                       'cc': e['clean__clean']['mean'], 'cd': e['clean__degraded']['mean'],
                       'dc': e['degraded__clean']['mean'], 'dd': e['degraded__degraded']['mean']}
                row.update(stat_columns('g_d', e['later_recovery_with_main_degraded']))
                row.update(stat_columns('i_rec', e['main_quality_by_later_quality_interaction']))
                rows.append(row)
        e = report['policies']['invalid_as_zero']['metrics'][report['primary_metric']]['estimates']
        src = report['job'] + '/' + report['primary_metric']
        task = {'gqa_attribute': 'GQA attr.', 'gqa_spatial': 'GQA spatial'}.get(report['task'], 'DocVQA')
        lines.append(' & '.join([task if report['model'] == MODELS[0] else '', NAMES[report['model']]]
                                + [mean(e[k], source=src + '/' + k) for k in CELLS]
                                + [estimate(e['later_recovery_with_main_degraded'], source=src + '/B'),
                                   estimate(e['main_quality_by_later_quality_interaction'], source=src + '/J')]) + r' \\')
        if report['model'] == MODELS[-1] and report is not reports[-1]:
            lines.append(r'\addlinespace[2pt]')
    write_csv('fixed_contrast_benchmarks.csv', rows)
    return [r'\Needspace{6\baselineskip}\subsection{Standard-task cells}',
            r'Table~\ref{tab:v14_benchmark_full} gives the four cells for GQA and DocVQA under blur, '
            r'with later recovery $G_D$ and its interaction $I_{\rm rec}$.',
            r'\begin{table*}[t]', r'\centering\footnotesize', r'\setlength{\tabcolsep}{4pt}', r'\renewcommand{\arraystretch}{1.02}',
            r'\caption{\textbf{Blur at radius 2 costs little GQA or DocVQA accuracy, and no positive later recovery is resolved.} '
            r'Fixed-contrast standard-task cells under Gaussian blur (radius 2.0), over 256 image or document clusters per task. '
            r'The primary metric is strict matching for GQA and organizer ANLS for DocVQA, with invalid outputs scored zero. $G_D$ is later '
            r'recovery with degraded main and $I_{\rm rec}$ its difference from the clean-main gain; intervals are '
            r'descriptive normal 95\% intervals. Named metric sensitivities and invalid-answer policies are in '
            r'\texttt{fixed\_\allowbreak{}contrast\_\allowbreak{}benchmarks.csv}. Granite and DeepStack capitalize their GQA answers, so their strict GQA rows '
            r'are a format floor.}',
            r'\label{tab:v14_benchmark_full}', r'\begin{tabular}{@{}llrrrrcc@{}}', r'\toprule', r'\rowcolor{DSTableHeader}',
            r'Task & Model & CC & CD & DC & DD & $G_D$ & $I_{\rm rec}$ \\', r'\midrule'] + lines + [r'\bottomrule', r'\end{tabular}', r'\end{table*}']


def main():
    HERE.mkdir(parents=True, exist_ok=True); OUT.mkdir(parents=True, exist_ok=True)
    summary = json.loads((SOURCE / 'summary.json').read_text())
    paper = json.loads((SOURCE / 'paper_numbers.json').read_text())
    bench = json.loads((SOURCE / 'benchmark_numbers.json').read_text())
    color_cells_table(summary['groups'])
    lines = appendix_line1(summary['groups']) + appendix_marginal(paper, summary['groups']) + appendix_pairs(paper) + appendix_benchmark(bench)
    (HERE / 'appendix_results.tex').write_text('\n'.join(lines) + '\n')
    manifest = {'generator': str(Path(__file__).resolve()),
                'sources': {name: {'path': str(SOURCE / name), 'sha256': hashlib.sha256((SOURCE / name).read_bytes()).hexdigest()}
                            for name in ['summary.json', 'paper_numbers.json', 'benchmark_numbers.json']},
                'copied_statistic_presentations': len(recorded), 'values': recorded, 'csv_exports': exports,
                'note': 'Copied estimates only. No inference, rescoring or uncertainty recomputation. SE=0 is explicit. Exact duplicate recovery/policy summaries share a display or an explicit pointer to the primary table; all source objects remain listed.'}
    (OUT / 'source_manifest_tables.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Wrote compact appendix tables and {len(exports)} CSV files; {len(recorded)} copied statistic presentations.')


if __name__ == '__main__':
    output_dir()
    main()
