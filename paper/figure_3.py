"""Render the calibrated four-cell or recovery panel from its unrounded summary tables."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from analysis.paths import INPUT, OUT as RENDERED, output_dir
V15 = INPUT / "calibrated"
MAIN_TABLE, SENSITIVITIES = V15/"main_table.csv", V15/"sensitivities.csv"
OUTPUT = RENDERED

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.text import Text
from matplotlib.transforms import offset_copy

from paper import plot_style as style

WIDTH_PT, HEIGHT_PT = 495.0, 150.0
# Cell colours follow which entry is clean: blue main, teal later (as in Figures 1-2).
# Lightness runs from dark CC to light DD, and the two cells with degraded later input are
# hatched, so the cells separate in grey-scale and for colour-blind readers.
# key, legend label, fill, hatch, hatch colour
CELLS = [('cc', 'Both clean (CC)', style.CELL_CC, None, None),
         ('cd', 'Main clean (CD)', style.MAIN, '//////', style.HATCH_LIGHT),
         ('dc', 'Later clean (DC)', style.LATER, None, None),
         ('dd', 'Both degraded (DD)', style.CELL_DD, '//////', style.HATCH_DARK)]
# name, (model, readout or metric, task), table, row filter, x position
# Qwen32's stripe cells are identical under direct answers and identity labels at this level
# (means and intervals); direct answers are plotted so that Qwen8 and Qwen32 share a readout.
CASES = [
    ('qwen8b_stripes', ('Qwen8', 'direct', 'Stripes'), 'main', dict(source='analysis', model='qwen8b', task='stripe_orientation',
     readout='direct/none', kind='generation', row_type='label', severity_label='blurrel-period0.43'), 0.0),
    ('qwen32b_stripes', ('Qwen32', 'direct', 'Stripes'), 'main', dict(source='analysis', model='qwen32b', task='stripe_orientation',
     readout='direct/none', kind='generation', row_type='label', severity_label='blurrel-period0.43'), 1.0),
    ('qwen32b_text', ('Qwen32', 'identity', 'Text'), 'main', dict(source='analysis', model='qwen32b', task='text_reading',
     readout='label/identity', kind='generation', row_type='label', severity_label='lowres-font2.5'), 2.0),
    ('ibm4b_position', ('Granite', 'direct', 'Position'), 'main', dict(source='analysis', model='ibm4b', task='relative_position',
     readout='direct/none', kind='generation', row_type='label', severity_label='blurrel-dotsep0.3'), 3.0),
    ('qwen32b_docvqa', ('Qwen32', 'ANLS', 'DocVQA'), 'main', dict(source='benchmark', model='qwen32b', task='docvqa_document_qa',
     row_type='label', severity_label='lowres-long180'), 4.45),
    ('ibm4b_docvqa', ('Granite', 'ANLS', 'DocVQA'), 'main', dict(source='benchmark', model='ibm4b', task='docvqa_document_qa',
     row_type='label', severity_label='lowres-long180'), 5.45),
    ('deepstack7b_gqa', ('DeepStack', 'accuracy', 'GQA attr.'), 'sens', dict(model='deepstack7b', task='gqa_attribute',
     severity_label='lowres-long5', metric='gqa_case_punct_normalized', policy='invalid_as_zero'), 6.6),
]
# Synthetic groups lie left of this x, benchmark groups right of it.
SEPARATOR_X = 3.725
READOUTS = [('direct', 'Dir', 'o'), ('identity', 'Id', 's'), ('cyclic', 'Cyc', 'D')]
OFFSETS = (-.29, 0., .29)
MODELS = [('qwen8b', 'Qwen8'), ('qwen32b', 'Qwen32'), ('ibm4b', 'Granite'), ('deepstack7b', 'DeepStack')]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def number(text):
    return None if text in ('', 'NA') else float(text)


def load_cases():
    tables = {name: list(csv.DictReader(path.open())) for name, path in (('main', MAIN_TABLE), ('sens', SENSITIVITIES))}
    loaded = []
    for name, label, table, match, x in CASES:
        rows = [r for r in tables[table] if all(r[k] == v for k, v in match.items())]
        if table == 'main' and match['source'] == 'benchmark':
            rows = [r for r in rows if r['recovery_measure'] == 'docvqa_anls_organizer_framework|invalid_as_zero']
        if len(rows) != 1:
            raise ValueError(f'{name}: expected one row, found {len(rows)}')
        row = rows[0]
        cells = {c: {'mean': float(row[c+'_mean']), 'ci': [number(row[c+'_ci_low']), number(row[c+'_ci_high'])]}
                 for c, *_ in CELLS}
        loaded.append({'name': name, 'label': label, 'x': x, 'table': table, 'row': match, 'cells': cells,
                       'G_D': float(row['dc_minus_dd_mean']), 'I_rec': float(row['recovery_interaction_mean']),
                       'Lambda': float(row['cc_minus_dd_mean'])})
    return loaded


def load_stripes():
    rows = [r for r in csv.DictReader(MAIN_TABLE.open()) if r['model'] == 'qwen32b' and r['task'] == 'stripe_orientation'
            and r['readout'] == 'direct/none' and r['kind'] == 'generation']
    ref = next(r for r in rows if r['row_type'] == 'reference')
    labels = ('blurrel-period0.37', 'blurrel-period0.4', 'blurrel-period0.43')
    lab = {r['severity_label']: r for r in rows if r['row_type'] == 'label'}
    first = lab[labels[0]]
    def stat(r, stem):
        return {'mean': float(r[stem+'_mean']), 'ci': [number(r[stem+'_ci_low']), number(r[stem+'_ci_high'])]}
    clean = {'mean': float(first['cc_mean']), 'ci': [number(first['cc_ci_low']), number(first['cc_ci_high'])]}
    return {'levels': ['Clean', '0.37', '0.40', '0.43'],
            'r': [stat(ref, 'conflict_R')]+[stat(lab[l], 'conflict_R') for l in labels],
            'dc': [clean]+[stat(lab[l], 'dc') for l in labels],
            'dd': [clean]+[stat(lab[l], 'dd') for l in labels]}




def check_geometry(fig, cards, axes):
    fig.canvas.draw()
    renderer, canvas = fig.canvas.get_renderer(), fig.bbox
    texts = [t for t in fig.findobj(Text) if t.get_visible() and t.get_text()]
    outside = [t.get_text() for t in texts if not (canvas.contains(*t.get_window_extent(renderer).p0)
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
        (y0, y1) = ax.get_ylim()
        for collection in ax.collections:
            for segment in collection.get_segments():
                if any(not (y0 <= float(y) <= y1) for _, y in segment):
                    raise ValueError('Clipped interval')
    if min(t.get_fontsize() for t in texts) < 7:
        raise ValueError('Text fell below 7 pt')
    return {'text_inside_canvas': True, 'headings_inside_cards': True, 'independent_text_nonoverlap': True,
            'intervals_inside_axes': True, 'minimum_font_pt': min(t.get_fontsize() for t in texts)}


def main():
    cases = load_cases()
    style.apply_paper_style()
    plt.rcParams.update({'font.size': 8.2, 'mathtext.fontset': 'dejavusans', 'axes.unicode_minus': True,
                         'hatch.linewidth': .55})
    fig = plt.figure(figsize=(WIDTH_PT/72, HEIGHT_PT/72))
    ax_y, ax_h = 50, 70
    card_y = 18
    card_h = ax_y+ax_h+6+19-card_y
    cards = [style.panel_card(fig, (5, card_y, 327, card_h), 'a', 'Four quality cells, strongest level', title_size=8.9),
             style.panel_card(fig, (338, card_y, 152, card_h), 'b', 'Qwen32 stripes, direct', title_size=8.9)]
    a = fig.add_axes([34/WIDTH_PT, ax_y/HEIGHT_PT, 293/WIDTH_PT, ax_h/HEIGHT_PT])
    a.set_facecolor('white')
    width = .2
    for case in cases:
        for j, (cell, _, color, hatch, hatch_color) in enumerate(CELLS):
            x = case['x']+(j-1.5)*width
            value = case['cells'][cell]
            a.bar(x, value['mean'], width=width*.86, color=color, hatch=hatch,
                  edgecolor=hatch_color if hatch else 'none', linewidth=0, zorder=2)
            low, high = value['ci']
            if low is not None:
                a.errorbar(x, value['mean'], yerr=[[value['mean']-low], [high-value['mean']]], fmt='none',
                           ecolor=style.INK, elinewidth=.6, capsize=1.1, capthick=.5, zorder=3)
        model, readout, _ = case['label']
        a.annotate(model, (case['x'], 0), xycoords=a.get_xaxis_transform(), xytext=(0, -3),
                   textcoords='offset points', ha='center', va='top', fontsize=8.0)
        a.annotate(readout, (case['x'], 0), xycoords=a.get_xaxis_transform(), xytext=(0, -12.5),
                   textcoords='offset points', ha='center', va='top', fontsize=7.0, color=style.MUTED)
    # Task line: one label per task, under a thin rule spanning that task's groups.
    rule = offset_copy(a.get_xaxis_transform(), fig=fig, y=-22.5, units='points')
    tasks = []
    for case in cases:
        task = case['label'][2]
        if tasks and tasks[-1][0] == task:
            tasks[-1][2] = case['x']
        else:
            tasks.append([task, case['x'], case['x']])
    for task, x0, x1 in tasks:
        a.plot([x0-.43, x1+.43], [0, 0], transform=rule, color=style.MUTED, linewidth=.5, clip_on=False)
        a.annotate(task, ((x0+x1)/2, 0), xycoords=a.get_xaxis_transform(), xytext=(0, -24.5),
                   textcoords='offset points', ha='center', va='top', fontsize=8.0)
    # Thin rule between the synthetic tasks (left) and the benchmarks (right).
    a.axvline(SEPARATOR_X, ymin=0, ymax=1, color='#9AAEC0', linewidth=.6, zorder=1)
    a.set_xlim(-.52, cases[-1]['x']+.52)
    a.set_ylim(0, 1.08)
    a.set_xticks([])
    a.set_yticks([0, .5, 1], ['0', '.5', '1'])
    a.tick_params(axis='y', labelsize=8.2, length=2.5, width=.6, pad=2.4)
    a.spines[['top', 'right']].set_visible(False)
    a.spines[['left', 'bottom']].set_linewidth(.6)
    a.grid(axis='y', color=style.BORDER, linewidth=.45, zorder=0)
    a.set_ylabel('Accuracy or ANLS', fontsize=8.2, labelpad=2.5)

    b = fig.add_axes([366/WIDTH_PT, ax_y/HEIGHT_PT, 116/WIDTH_PT, ax_h/HEIGHT_PT])
    b.set_facecolor('white')
    series = load_stripes()
    xs = range(len(series['levels']))
    b.set_xlim(-.35, len(series['levels'])-.65)
    b.set_ylim(0, 1.1)
    b.set_xticks(list(xs), ['Clean', '.37', '.40', '.43'])
    b.set_yticks([0, .5, 1], ['0', '.5', '1'])
    b.tick_params(axis='both', labelsize=8.2, length=2.5, width=.6, pad=2.4)
    b.spines[['top', 'right']].set_visible(False)
    b.spines[['left', 'bottom']].set_linewidth(.6)
    b.grid(axis='y', color=style.BORDER, linewidth=.45, zorder=0)
    b.set_xlabel('Main blur ($\\sigma$ / stripe period)', fontsize=8.2, labelpad=2.0)
    b.set_ylabel('Accuracy or $R$', fontsize=8.2, labelpad=2.5)
    for key, color, marker, z in (('dd', style.DD_LINE, 'o', 2), ('dc', style.LATER, 'o', 3), ('r', style.INK, 's', 4)):
        values = series[key]
        # Both-degraded is dashed, echoing the hatching of degraded-later cells in panel (a).
        b.plot(list(xs), [v['mean'] for v in values], color=color, linewidth=1.1,
               linestyle=(0, (3, 1.6)) if key == 'dd' else '-', marker=marker,
               markersize=3.6, markeredgewidth=.5, markeredgecolor='white', zorder=z)
        for x, v in zip(xs, values):
            if v['ci'][0] is not None:
                b.errorbar(x, v['mean'], yerr=[[v['mean']-v['ci'][0]], [v['ci'][1]-v['mean']]], fmt='none',
                           ecolor=color, elinewidth=.8, capsize=1.4, capthick=.6, zorder=z)
    # Direct labels name the three lines.
    b.annotate('DC', (1.5, series['dc'][2]['mean']), xytext=(0, 2.5), textcoords='offset points',
               ha='center', va='bottom', fontsize=7.5, color=style.LATER_INK, weight='bold')
    b.annotate('DD', (3, series['dd'][3]['mean']), xytext=(-5, -2), textcoords='offset points',
               ha='right', va='top', fontsize=7.5, color=style.DD_LINE, weight='bold')
    b.annotate('$R$', (0, series['r'][0]['mean']), xytext=(5, -1), textcoords='offset points',
               ha='left', va='center', fontsize=7.5, color=style.INK)
    records = series

    handles = [Patch(facecolor=color, hatch=hatch, edgecolor=hatch_color if hatch else 'none', linewidth=0,
                     label=label) for _, label, color, hatch, hatch_color in CELLS]
    handles.append(Line2D([], [], color=style.INK, linestyle='-', marker='s', markersize=3.6, markeredgewidth=.5,
                          markeredgecolor='white', linewidth=1.1, label='Later conflict effect $R$'))
    fig.legend(handles=handles, loc='center', bbox_to_anchor=(.5, 7/HEIGHT_PT), ncol=5, frameon=False,
               fontsize=8.2, handlelength=2.0, handleheight=1.0, handletextpad=.45, columnspacing=1.1,
               borderaxespad=0, borderpad=0)
    geometry = check_geometry(fig, cards, (a, b))
    for suffix in ('pdf', 'svg', 'png'):
        fig.savefig(OUTPUT/('fig3_fourcell.'+suffix), dpi=240, bbox_inches=None)
    plt.close(fig)
    manifest = {'schema': 'v15-fourcell-figure-1',
                'panel_a_sources': {'main_table': str(MAIN_TABLE), 'main_table_sha256': sha(MAIN_TABLE),
                                    'sensitivities': str(SENSITIVITIES), 'sensitivities_sha256': sha(SENSITIVITIES)},
                'panel_b_source': {'main_table': str(MAIN_TABLE), 'rows': 'qwen32b stripe_orientation direct/none generation: '
                                   'reference row (normal main) and blurrel-period0.37/0.4/0.43 label rows'},
                'generator': str(Path(__file__).resolve()), 'generator_sha256': sha(Path(__file__)),
                'style': str(Path(style.__file__).resolve()), 'size_points': [WIDTH_PT, HEIGHT_PT],
                'panel_a_cases': cases, 'panel_b_series': records, 'geometry_checks': geometry,
                'outputs': {s: {'path': str(OUTPUT/('fig3_fourcell.'+s)), 'sha256': sha(OUTPUT/('fig3_fourcell.'+s))}
                            for s in ('pdf', 'svg', 'png')}}
    (OUTPUT/'source_manifest_fourcell_v15.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({'cases': [(c['name'], {k: round(v['mean'], 3) for k, v in c['cells'].items()}) for c in cases],
                      'geometry': geometry}))


if __name__ == '__main__':
    output_dir()
    main()
