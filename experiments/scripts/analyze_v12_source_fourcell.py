#!/usr/bin/env python3
"""Exploratory fixed-physical-source four-cell analysis of accepted v10 outputs.

J is never substituted for the source-role-swap contrast. No inference is run.
Intervals are scene-level descriptive normal 95% intervals, as in v10.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import os as _path_os
DATA_ROOT = Path(_path_os.environ.get("DS_DATA_ROOT", "work")).expanduser().resolve()
from statistics import mean, stdev

MODELS = ('ibm4b', 'qwen8b', 'qwen32b')
NAMES = {'ibm4b': 'Granite 4B', 'qwen8b': 'Qwen 8B', 'qwen32b': 'Qwen 32B'}
PHYSICAL = ('main_layout', 'later_layout', 'main_strength', 'later_strength')
READOUT = ('response_format', 'mapping')
GROUP = READOUT + PHYSICAL + ('query_type', 'query_target')
META = ('id', 'scene_id', 'intervention_id', 'changed_object', 'queried_object',
        'changed_index', 'queried_index', 'direction_a_index', 'direction_b_index') + GROUP
EDGES = (('main', 'a'), ('main', 'b'), ('later', 'a'), ('later', 'b'))
EDGE_METRICS = ('E_main_other_a', 'E_main_other_b', 'E_later_other_a', 'E_later_other_b')
DEFAULT_SOURCE = (DATA_ROOT / 'runs/measurement-v10-cloud-20261014/remote/work/analysis')
DEFAULT_OUT = (DATA_ROOT / 'analysis/v12-p0-20261022/fourcell')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''): h.update(b)
    return h.hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def finite(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def describe(values):
    n = len(values)
    m = mean(values) if n else None
    se = stdev(values) / math.sqrt(n) if n > 1 else None
    return {'n': n, 'mean': m, 'se': se,
            'ci_low': m - 1.96 * se if se is not None else None,
            'ci_high': m + 1.96 * se if se is not None else None,
            'positive_units': sum(x > 0 for x in values),
            'negative_units': sum(x < 0 for x in values),
            'zero_units': sum(x == 0 for x in values)}


def load_edges(path, kind):
    groups = defaultdict(dict)
    count = 0
    with Path(path).open() as f:
        for line in f:
            if not line.strip(): continue
            r = json.loads(line); count += 1
            key = (r['id'],) + tuple(r[k] for k in READOUT + PHYSICAL)
            edge = r['changed_entry'], r['fixed_other_assignment']
            if edge not in EDGES or edge in groups[key]:
                raise ValueError(f'Invalid/duplicate four-cell edge: {key}, {edge}')
            if kind == 'score' and finite(r['a_score']) and finite(r['b_score']):
                assert math.isclose(r['effect'], r['b_score'] - r['a_score'], abs_tol=1e-10)
            groups[key][edge] = r
    return groups, count


def fourcell(edges, kind):
    if set(edges) != set(EDGES): raise ValueError('Planned physical four-cell table is incomplete')
    anchor = edges[EDGES[0]]
    if any(any(r[k] != anchor[k] for k in META) for r in edges.values()):
        raise ValueError('Physical or query metadata changed across edges')
    vals = {name: edges[edge]['effect'] for name, edge in zip(EDGE_METRICS, EDGES)}
    complete = all(finite(v) for v in vals.values())
    if kind == 'generation':
        assert complete == all(r['both_valid'] for r in edges.values())
    metrics = dict(vals)
    if complete:
        j1 = vals['E_main_other_b'] - vals['E_main_other_a']
        j2 = vals['E_later_other_b'] - vals['E_later_other_a']
        assert math.isclose(j1, j2, abs_tol=1e-9), (j1, j2)
        d1 = vals['E_main_other_a'] - vals['E_later_other_a']
        d2 = vals['E_main_other_b'] - vals['E_later_other_b']
        assert math.isclose(d1, d2, abs_tol=1e-9)
        metrics.update(J=j1, D_physical=d1)
    else:
        metrics.update(J=None, D_physical=None)
    if kind == 'score':
        aa, ba = edges[('main', 'a')]['a_score'], edges[('main', 'a')]['b_score']
        ab, bb = edges[('main', 'b')]['a_score'], edges[('main', 'b')]['b_score']
        for v, w in zip((aa, ab, ba, bb), (edges[('later', 'a')]['a_score'],
                      edges[('later', 'a')]['b_score'], edges[('later', 'b')]['a_score'],
                      edges[('later', 'b')]['b_score'])):
            assert v == w or (finite(v) and finite(w) and math.isclose(v, w, abs_tol=1e-9))
        metrics.update(y_aa=aa, y_ba=ba, y_ab=ab, y_bb=bb)
        if complete:
            assert math.isclose(metrics['J'], bb - ba - ab + aa, abs_tol=1e-9)
            assert math.isclose(metrics['D_physical'], ba - ab, abs_tol=1e-9)
    return {**{k: anchor[k] for k in META}, 'kind': kind, 'common_four_valid': complete,
            'invalid_edge_count': sum(not finite(r['effect']) for r in edges.values()),
            'metrics': metrics}


def aggregate(rows, model):
    grouped = defaultdict(lambda: defaultdict(list))
    for r in rows: grouped[tuple(r[k] for k in GROUP)][r['scene_id']].append(r)
    summaries, units = [], []
    for group, scenes in sorted(grouped.items()):
        # Each original scene contains two changed-object interventions per query role.
        assert {len(v) for v in scenes.values()} == {2}
        metrics = tuple(next(iter(scenes.values()))[0]['metrics'])
        for validity in ('common_fourcell', 'original_pair'):
            for metric in metrics:
                if validity == 'original_pair' and metric not in EDGE_METRICS: continue
                values = []
                for scene, repeats in sorted(scenes.items()):
                    valid = all(finite(r['metrics'][metric]) for r in repeats)
                    if validity == 'common_fourcell': valid &= all(r['common_four_valid'] for r in repeats)
                    value = mean(r['metrics'][metric] for r in repeats) if valid else None
                    units.append({'model': model, 'kind': repeats[0]['kind'], **dict(zip(GROUP, group)),
                                  'scene_id': scene, 'metric': metric, 'validity': validity,
                                  'planned_repeats': len(repeats), 'value': value})
                    if value is not None: values.append(value)
                summaries.append({'model': model, 'kind': repeats[0]['kind'], **dict(zip(GROUP, group)),
                                  'metric': metric, 'validity': validity,
                                  'planned_units': len(scenes), 'incomplete_units': len(scenes) - len(values),
                                  **describe(values)})
    return summaries, units


def write_csv(path, rows):
    with Path(path).open('w', newline='') as f:
        w = csv.DictWriter(f, list(rows[0])); w.writeheader(); w.writerows(rows)


def reference(rows, kind, metric, model, validity='common_fourcell'):
    selected = [r for r in rows if r['model'] == model and r['kind'] == kind and r['metric'] == metric
                and r['validity'] == validity and r['response_format'] == 'direct'
                and r['query_type'] == 'identity' and r['query_target'] == 'changed'
                and r['main_layout'] == r['later_layout'] == 'M0'
                and r['main_strength'] == r['later_strength'] == 1.]
    assert len(selected) == 1
    return selected[0]


def plots(summaries, out):
    os.environ.setdefault('MPLCONFIGDIR', str(out / 'mpl-cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 9, 'axes.labelsize': 9, 'axes.titlesize': 10,
                         'xtick.labelsize': 9, 'ytick.labelsize': 8, 'pdf.fonttype': 42})
    fig, axs = plt.subplots(2, 3, figsize=(7.2, 6.6))
    fig.subplots_adjust(left=.12, right=.985, top=.79, bottom=.13, wspace=.46, hspace=1.1)
    colors = ('#2471A3', '#D97924')
    for col, model in enumerate(MODELS):
        ax = axs[0, col]
        for i, (fixed, keys) in enumerate((('a', ('y_aa', 'y_ba')), ('b', ('y_ab', 'y_bb')))):
            points = [reference(summaries, 'score', k, model) for k in keys]
            ax.errorbar([0, 1], [p['mean'] for p in points],
                        yerr=[1.96 * p['se'] for p in points], color=colors[i],
                        marker=('o', 's')[i], capsize=3, label=f'Later = {fixed}')
        j = reference(summaries, 'score', 'J', model)
        ax.set_title(f"{NAMES[model]} · J = {j['mean']:.3f}\n95% CI [{j['ci_low']:.4f}, {j['ci_high']:.4f}]", pad=9, fontsize=9)
        ax.set_xticks([0, 1], ['a', 'b']); ax.set_xlabel('Main source'); ax.set_xlim(-.18, 1.18)
        if col == 0: ax.set_ylabel('Candidate log-probability contrast')
        ax = axs[1, col]
        for i, entry in enumerate(('main', 'later')):
            points = [reference(summaries, 'generation', f'E_{entry}_other_{f}', model) for f in ('a', 'b')]
            ax.errorbar([0, 1], [p['mean'] for p in points],
                        yerr=[1.96 * p['se'] if p['se'] is not None else 0 for p in points],
                        marker=('o', 's')[i], color=colors[i], capsize=3, label=f'Change {entry}')
        j = reference(summaries, 'generation', 'J', model)
        ax.set_title(f"Generation J = {j['mean']:.3f}\n95% CI [{j['ci_low']:.3f}, {j['ci_high']:.3f}]\nn = {j['n']}/{j['planned_units']}", pad=9, fontsize=9)
        ax.set_xticks([0, 1], ['a', 'b']); ax.set_xlabel('Other source'); ax.set_xlim(-.18, 1.18)
        if col == 0: ax.set_ylabel('Conditional generation effect', labelpad=10)
    for ax in axs.flat:
        ax.spines[['top', 'right']].set_visible(False); ax.axhline(0, color='.7', linewidth=.6, zorder=0)
    fig.suptitle('Fixed physical source configuration: M0/M0, α = 1/1\nDirect identity query to the edited object', y=.985, fontsize=11)
    fig.legend(*axs[0, 0].get_legend_handles_labels(), loc='upper center', bbox_to_anchor=(.54, .92), ncol=2, frameon=False, fontsize=8)
    fig.legend(*axs[1, 0].get_legend_handles_labels(), loc='upper center', bbox_to_anchor=(.54, .50), ncol=2, frameon=False, fontsize=8)
    fig.text(.105, .025, 'Scene-paired descriptive 95% intervals. Score: 64 scenes. Generation: common-valid four cells.\nAxes use model-specific scales; J measures output nonadditivity, not a unique neural mechanism.', fontsize=8)
    fig.canvas.draw(); renderer = fig.canvas.get_renderer(); width, height = fig.canvas.get_width_height()
    from matplotlib.text import Text
    texts = []
    for t in fig.findobj(Text):
        if not t.get_visible() or not t.get_text().strip(): continue
        b = t.get_window_extent(renderer)
        assert b.x0 >= 0 and b.y0 >= 0 and b.x1 <= width and b.y1 <= height, t.get_text()
        texts.append({'text': t.get_text(), 'font_pt': t.get_fontsize(), 'box': list(b.extents)})
    overlaps = []
    for i, a in enumerate(texts):
        for b in texts[i+1:]:
            x = min(a['box'][2], b['box'][2]) - max(a['box'][0], b['box'][0])
            y = min(a['box'][3], b['box'][3]) - max(a['box'][1], b['box'][1])
            if x > .5 and y > .5: overlaps.append((a['text'], b['text']))
    assert not overlaps, overlaps
    fig.savefig(out / 'source_fourcell_reference.pdf')
    fig.savefig(out / 'source_fourcell_reference.png', dpi=180)
    dump(out / 'figure_geometry.json', {'width_inches': 7.2, 'height_inches': 6.6,
         'text_containment': 'PASS', 'text_overlap': 'PASS', 'minimum_font_points': min(t['font_pt'] for t in texts),
         'embedding': 'standalone; manuscript not written/verified', 'texts': texts})
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, default=DEFAULT_SOURCE)
    p.add_argument('--output', type=Path, default=DEFAULT_OUT)
    p.add_argument('--no-plots', action='store_true')
    a = p.parse_args(argv)
    if not a.output.resolve().is_relative_to(DATA_ROOT): raise ValueError('Use project external output')
    a.output.mkdir(parents=True, exist_ok=True)
    all_summary, manifest, checks = [], [], []
    with (a.output / 'scene_estimates.jsonl').open('w') as units_f, (a.output / 'query_fourcells.jsonl').open('w') as query_f:
        for model in MODELS:
            root = a.source / f'{model}-full-control-all'
            old = json.loads((root / 'summary.json').read_text())
            for name, kind, countkey in [('query_effects.jsonl', 'score', 'planned_pairs'),
                                         ('behavior_pairs.jsonl', 'generation', 'planned_generation_pairs')]:
                path = root / name; groups, n = load_edges(path, kind)
                assert n == old['coverage'][countkey]
                rows = [fourcell(edges, kind) for edges in groups.values()]
                summaries, units = aggregate(rows, model)
                assert len(summaries) == (2688 if kind == 'score' else 1920), (kind, len(summaries))
                for r in units: units_f.write(json.dumps(r, allow_nan=False) + '\n')
                for r in rows: query_f.write(json.dumps({'model': model, **r}, allow_nan=False) + '\n')
                all_summary += summaries
                checks.append({'model': model, 'kind': kind, 'edge_rows': n,
                               'physical_query_tables': len(rows),
                               'complete_query_tables': sum(r['common_four_valid'] for r in rows),
                               'planned_scenes': len({r['scene_id'] for r in rows}),
                               'four_edge_closure_and_direction': 'PASS', 'paired_repeats_per_scene': 2})
                manifest.append({'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha(path)})
                del groups, rows, units
    write_csv(a.output / 'summaries.csv', all_summary)
    dump(a.output / 'summary.json', {'schema': 'v12-p0-source-fourcell-v1', 'summaries': all_summary})
    dump(a.output / 'validation.json', {'state': 'PASS', 'checks': checks,
        'estimand': 'J at fixed PHYSICAL main/later layouts and strengths; original D_swap unchanged',
        'validity': 'common_fourcell primary; original_pair conditional effects retained separately',
        'interval': 'scene mean of both interventions, normal 95%, descriptive not simultaneous',
        'no_model_runs': True, 'inputs': manifest, 'script_sha256': sha(__file__)})
    if not a.no_plots: plots(all_summary, a.output)
    print(json.dumps({'output': str(a.output), 'checks': checks}, ensure_ascii=False))


if __name__ == '__main__': main()
