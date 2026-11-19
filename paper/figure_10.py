"""Render the calibrated four-cell or recovery panel from its unrounded summary tables."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from analysis.paths import INPUT, OUT as RENDERED, output_dir
V15 = INPUT / "calibrated"
MAIN_TABLE, SAMPLE_SIZES = V15/"main_table.csv", V15/"sample_sizes.csv"
OUTPUT = RENDERED

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.text import Text

from paper import plot_style as style

WIDTH_PT, HEIGHT_PT = 495.0, 160.0
# Color and marker identify the model. Qwen8 and Qwen32 get different hues (blue, purple);
# Granite and DeepStack avoid the later teal and the target coral of the other figures.
MODELS = [('qwen8b', 'Qwen8', 'o', style.MAIN), ('qwen32b', 'Qwen32', 's', '#8A5AA8'),
          ('ibm4b', 'Granite', 'D', '#B9822C'), ('deepstack7b', 'DeepStack', '^', '#222222')]
MARKER_SIZE = {'o': 4.3, 's': 3.9, 'D': 3.6, '^': 4.5}
# Panel (a): direct-format generation for synthetic tasks; organiser ANLS for DocVQA.
TASKS = [('color', 'Color, direct', 'analysis', 'direct/none', 'accuracy_invalid_as_incorrect'),
         ('stripe_orientation', 'Stripes, identity', 'analysis', 'label/identity', 'accuracy_invalid_as_incorrect'),
         ('text_reading', 'Text, identity', 'analysis', 'label/identity', 'accuracy_invalid_as_incorrect'),
         ('docvqa_document_qa', 'DocVQA', 'benchmark', 'freeform/none',
          'docvqa_anls_organizer_framework|invalid_as_zero')]
# Non-native on synthetic text (calibrate.py); its rows remain in the appendix table.
OMIT = {('deepstack7b', 'text_reading'), ('deepstack7b', 'stripe_orientation')}
# Drawn hollow and not read as effects: no measurable loss, or a main-entry no-op.
HOLLOW = {('qwen32b', 'color', 'desat-c0.2'): 'no net loss observed (DD = CC)',
          ('qwen32b', 'stripe_orientation', 'blurrel-period0.37'): 'no net loss observed (DD = CC)',
          ('deepstack7b', 'docvqa_document_qa', 'lowres-long360'):
              'near no-op for the 336-px main view'}
# Format abbreviations avoid C/D, which denote clean/degraded quality in panel (a).
READOUTS = [('direct', 'Dir'), ('identity', 'Id'), ('cyclic', 'Cyc')]
OFFSETS = (-.29, 0., .29)
LIMIT = (-.07, .69)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def number(text):
    return None if text in ('', 'NA') else float(text)


def load_v15():
    zero_se = {}
    with SAMPLE_SIZES.open() as handle:
        for row in csv.DictReader(handle):
            if row['row_type'] == 'label' and row['main_table_column'] == 'dc_minus_dd':
                key = (row['model'], row['task'], row['readout'], row['kind'], row['severity_label'])
                zero_se[key] = row['se_zero'] == 'True'
    selected = {}
    with MAIN_TABLE.open() as handle:
        for row in csv.DictReader(handle):
            for task, _, source, readout, measure in TASKS:
                if (row['task'] == task and row['source'] == source and row['readout'] == readout
                        and row['kind'] == 'generation' and row['row_type'] == 'label'
                        and row['in_shared_grid'] == 'True' and row['recovery_measure'] == measure):
                    selected.setdefault((row['model'], task), []).append(row)
    curves = {}
    for model, *_ in MODELS:
        for task, *_ in TASKS:
            rows = sorted(selected.get((model, task), []), key=lambda r: int(r['severity_rank']))
            if len(rows) != 3:
                raise ValueError(f'Expected three shared-grid labels for {model} {task}')
            points = []
            for row in rows:
                low, high = number(row['dc_minus_dd_ci_low']), number(row['dc_minus_dd_ci_high'])
                se_zero = zero_se.get((model, task, row['readout'], 'generation', row['severity_label']))
                if row['source'] == 'benchmark':
                    se_zero = False
                if (low is None) != bool(se_zero):
                    raise ValueError(f'Unexplained missing interval: {model} {task} {row["severity_label"]}')
                points.append({'label': row['severity_label'], 'rank': int(row['severity_rank']),
                               'loss': float(row['cc_minus_dd_mean']),
                               'loss_ci': [number(row['cc_minus_dd_ci_low']), number(row['cc_minus_dd_ci_high'])],
                               'recovery': float(row['dc_minus_dd_mean']), 'recovery_ci': [low, high],
                               'recovery_n': int(row['dc_minus_dd_n']), 'se_zero': bool(se_zero),
                               'interaction': float(row['recovery_interaction_mean']),
                               'cells': {c: float(row[c+'_mean']) for c in ('cc', 'cd', 'dc', 'dd')},
                               'hollow': HOLLOW.get((model, task, row['severity_label'])),
                               'source_row': {k: row[k] for k in ('source', 'model', 'task', 'readout', 'kind',
                                                                  'row_type', 'severity_label', 'recovery_measure')}})
            curves[(model, task)] = points
    return curves




def style_axis(ax, xlim, ylim, xticks, yticks, xticklabels=None, ylabels=True):
    ax.set_facecolor('white')
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xticks(xticks, xticklabels if xticklabels is not None else
                  [('0' if t == 0 else f'{t:.1f}'.lstrip('0')) for t in xticks])
    ax.set_yticks(yticks, [('0' if t == 0 else f'{t:.1f}'.lstrip('0') if isinstance(t, float) else str(t))
                           for t in yticks] if ylabels else [])
    ax.tick_params(axis='both', labelsize=8.2, length=2.5, width=.6, pad=2.4)
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_linewidth(.6)
    ax.grid(color=style.BORDER, linewidth=.45, zorder=0)


def draw_curves(ax, curves, task):
    ax.plot(LIMIT, LIMIT, color=style.MUTED, linewidth=.8, linestyle=(0, (3.2, 2.2)), zorder=1)
    ax.axhline(0, color=style.MUTED, linewidth=.75, zorder=1)
    drawn = []
    for model, _, marker, color in MODELS:
        if (model, task) in OMIT:
            continue
        points = curves[(model, task)]
        ax.plot([p['loss'] for p in points], [p['recovery'] for p in points], color=color,
                linewidth=.95, alpha=.9, zorder=2)
        for p in points:
            low, high = p['recovery_ci']
            if low is not None:
                ax.errorbar(p['loss'], p['recovery'], yerr=[[p['recovery']-low], [high-p['recovery']]],
                            fmt='none', ecolor=color, elinewidth=.9, capsize=1.6, capthick=.75, zorder=3)
            ax.plot(p['loss'], p['recovery'], marker=marker, markersize=MARKER_SIZE[marker],
                    markeredgewidth=.75 if p['hollow'] else .5,
                    markeredgecolor=color if p['hollow'] else 'white',
                    markerfacecolor='white' if p['hollow'] else color, linestyle='none', zorder=4)
            drawn.append({'model': model, 'task': task, **p,
                          'display': {'marker': marker, 'color': color,
                                      'fill': 'hollow' if p['hollow'] else 'filled'}})
    return drawn




def check_geometry(fig, cards, axes):
    fig.canvas.draw()
    renderer, canvas = fig.canvas.get_renderer(), fig.bbox
    texts = [t for t in fig.findobj(Text) if t.get_visible() and t.get_text()]
    outside = [t.get_text() for t in texts
               if not (canvas.contains(*t.get_window_extent(renderer).p0)
                       and canvas.contains(*t.get_window_extent(renderer).p1))]
    if outside:
        raise ValueError('Text outside canvas: '+repr(outside))
    for card in cards:
        region = card['surface'].get_window_extent(renderer)
        for key in ('letter', 'title'):
            box = card[key].get_window_extent(renderer)
            if not (region.x0 < box.x0 < box.x1 < region.x1 and region.y0 < box.y0 < box.y1 < region.y1):
                raise ValueError('Panel heading outside its card')
    overlaps = []
    for i, first in enumerate(texts):
        a = first.get_window_extent(renderer)
        for second in texts[i+1:]:
            b = second.get_window_extent(renderer)
            if min(a.x1, b.x1)-max(a.x0, b.x0) > .2 and min(a.y1, b.y1)-max(a.y0, b.y0) > .2:
                overlaps.append((first.get_text(), second.get_text()))
    if overlaps:
        raise ValueError('Independent text overlap: '+repr(overlaps))
    for ax in axes:
        for line in ax.get_lines():
            xs, ys = line.get_xdata(), line.get_ydata()
            if len(xs) and line.get_linestyle() == 'None':
                (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
                if any(not (x0 < float(x) < x1 and y0 < float(y) < y1) for x, y in zip(xs, ys)):
                    raise ValueError('Marker outside axes')
        for collection in ax.collections:
            for segment in collection.get_segments():
                (y0, y1) = ax.get_ylim()
                if any(not (y0 < float(y) < y1) for _, y in segment):
                    raise ValueError('Clipped interval')
    if min(t.get_fontsize() for t in texts) < 8:
        raise ValueError('Normal-reading text fell below 8 pt')
    return {'text_inside_canvas': True, 'headings_inside_cards': True,
            'independent_text_nonoverlap': True, 'markers_and_intervals_inside_axes': True,
            'minimum_font_pt': min(t.get_fontsize() for t in texts)}


def main():
    curves = load_v15()
    style.apply_paper_style()
    plt.rcParams.update({'font.size': 8.2, 'mathtext.fontset': 'dejavusans', 'axes.unicode_minus': True})
    fig = plt.figure(figsize=(WIDTH_PT/72, HEIGHT_PT/72))
    card_y, card_h = 25, HEIGHT_PT-30
    cards = [style.panel_card(fig, (5, card_y, 485, card_h), None, 'Later recovery against joint loss',
                              title_size=8.9)]
    bottom, height = 54/HEIGHT_PT, 69/HEIGHT_PT
    recovery_axes, drawn = [], []
    for i, (task, title, *_rest) in enumerate(TASKS):
        ax = fig.add_axes([(44+i*112)/WIDTH_PT, bottom, 92/WIDTH_PT, height])
        style_axis(ax, LIMIT, (-.18, LIMIT[1]), [0, .2, .4, .6], [0, .2, .4, .6], ylabels=(i == 0))
        ax.set_title(title, fontsize=8.4, pad=3.0, color=style.INK)
        drawn += draw_curves(ax, curves, task)
        recovery_axes.append(ax)
    recovery_axes[0].set_ylabel('Recovery $G_D$', fontsize=8.2, labelpad=2.5)
    fig.text(.5, 27.5/HEIGHT_PT, 'Joint loss $A_{CC}-A_{DD}$', fontsize=8.2, ha='center', va='bottom', color=style.INK)
    handles = [Line2D([], [], color=color, marker=marker, linestyle='-', linewidth=.95,
                      markersize=MARKER_SIZE[marker]+.3, markeredgewidth=.5, markeredgecolor='white',
                      label=label) for _, label, marker, color in MODELS]
    handles.append(Line2D([], [], color=style.MUTED, linewidth=.8, linestyle=(0, (3.2, 2.2)),
                          label='Full recovery ($A_{DC}=A_{CC}$)'))
    handles.append(Line2D([], [], color=style.MUTED, marker='o', linestyle='none', markersize=4.6,
                          markeredgewidth=.75, markerfacecolor='white', label='Hollow: not interpreted'))
    fig.legend(handles=handles, loc='center', bbox_to_anchor=(.5, 10/HEIGHT_PT), ncol=6, frameon=False,
               fontsize=8.2, handlelength=1.6, handletextpad=.35, columnspacing=.95,
               borderaxespad=0, borderpad=0)
    geometry = check_geometry(fig, cards, recovery_axes)
    for suffix in ('pdf', 'svg', 'png'):
        fig.savefig(OUTPUT/('supp_recovery_curves.'+suffix), dpi=240, bbox_inches=None)
    plt.close(fig)
    manifest = {
        'schema': 'v15-recovery-curves-supp-1',
        'source': {'main_table': str(MAIN_TABLE), 'main_table_sha256': sha(MAIN_TABLE),
                   'sample_sizes': str(SAMPLE_SIZES), 'sample_sizes_sha256': sha(SAMPLE_SIZES),
                   'selection': 'generation; color direct/none, stripes and text label/identity; benchmark '
                                'organiser ANLS with invalid_as_zero for DocVQA; shared-grid labels'},
        'generator': str(Path(__file__).resolve()), 'generator_sha256': sha(Path(__file__)),
        'style': str(Path(style.__file__).resolve()), 'style_sha256': sha(Path(style.__file__)),
        'size_points': [WIDTH_PT, HEIGHT_PT], 'font': 'Arial',
        'points': len(drawn), 'interval_bars': sum(p['recovery_ci'][0] is not None for p in drawn),
        'zero_se_points': sum(p['se_zero'] for p in drawn),
        'hollow_points': [{'model': p['model'], 'task': p['task'], 'label': p['label'], 'reason': p['hollow']}
                          for p in drawn if p['hollow']],
        'omitted': [{'model': m, 'task': t, 'reason': 'non-native task (calibrate.py)'} for m, t in OMIT],
        'x_intervals': 'not drawn; synthetic loss is a difference of copied means without an interval, '
                       'benchmark loss intervals are in the supplement table',
        'geometry_checks': geometry, 'records': drawn,
        'outputs': {s: {'path': str(OUTPUT/('supp_recovery_curves.'+s)),
                        'sha256': sha(OUTPUT/('supp_recovery_curves.'+s))} for s in ('pdf', 'svg', 'png')}}
    (OUTPUT/'source_manifest_recovery_curves_supp.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k: manifest[k] for k in ('size_points', 'points', 'interval_bars', 'zero_se_points',
                                               'hollow_points', 'geometry_checks')}))


if __name__ == '__main__':
    output_dir()
    main()
