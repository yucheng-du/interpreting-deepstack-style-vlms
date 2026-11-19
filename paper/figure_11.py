from __future__ import annotations


from analysis.coding_csv import read_estimates, family
from analysis.paths import output_dir

import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FixedLocator, NullLocator  # noqa: E402

from paper import plot_style as style  # noqa: E402


WIDTH_PT, HEIGHT_PT = 495.0, 166.0
# Paper model order: Qwen first, then DeepStack.
PANELS = [('q8', 'Qwen8, GQA', 'Accuracy (%)'), ('q32a', 'Qwen32, GQA', None),
          ('q32d', 'Qwen32, DocVQA', 'ANLS × 100'), ('ds', 'DeepStack, GQA', 'Accuracy (%)')]
# Main only in the main-entry blue, later offsets in the later teal, shared QP in neutral ink.
OFFSET_GREY = '#7F8C99'
SERIES = [('shared', 'Shared QP', style.INK, 'o', '-', 3.4, True),
          ('scale_normalized', 'Scale-normalized', style.LATER, 's', '-', 3.1, True),
          ('offset', 'Other later offsets', OFFSET_GREY, 'x', '', 3.4, False),
          ('main_only', 'Main only, later zeroed', style.MAIN, 'D', '', 3.6, False)]
FULL_LINE = style.MUTED
# The DeepStack pair that Section 5.5 reports: main only at QP -4 against shared QP -2.
MARKED = {'ds': ('q-4|Z', 'q-2|q-2')}


def series_of(c):
    f = family(c)
    if f in ('later_coarser', 'later_finer'):
        return 'offset'
    return f


def draw(ax, E, cell, title, ylabel):
    pts = {}
    for k, e in E.items():
        if not k.startswith(f'score/{cell}/'):
            continue
        s = series_of(e['config'])
        if s in ('shared', 'scale_normalized', 'offset', 'main_only'):
            pts.setdefault(s, []).append(e)
    full = E[f'score/{cell}/C|C']
    ax.axhline(100*full['est'], color=FULL_LINE, lw=.7, ls=(0, (3, 2)), zorder=1)
    used = []
    for key, label, color, marker, ls, ms, line in SERIES:
        es = sorted(pts.get(key, []), key=lambda e: e['bytes'])
        if not es:
            continue
        x = [e['bytes']/1000 for e in es]
        y = [100*e['est'] for e in es]
        yerr = [[100*(e['est']-e['lo']) for e in es], [100*(e['hi']-e['est']) for e in es]]
        ax.errorbar(x, y, yerr=yerr, fmt='none', ecolor=color, elinewidth=.55, capsize=0, alpha=.9, zorder=2)
        mfc = 'white' if key == 'scale_normalized' else color
        ax.plot(x, y, ls=ls if line else 'none', lw=.9, color=color, marker=marker, ms=ms, mfc=mfc,
                mec=color, mew=.9 if marker in 'xs' else .6, zorder=3 if key != 'main_only' else 4)
        used += [dict(series=key, config=e['config'], kb=e['bytes']/1000, score=100*e['est'], lo=100*e['lo'],
                      hi=100*e['hi']) for e in es]
    if cell in MARKED:
        ring = [u for c in MARKED[cell] for u in used if u['config'] == c]
        assert len(ring) == 2
        ax.plot([u['kb'] for u in ring], [u['score'] for u in ring], ls='none', marker='o', ms=8.2,
                mfc='none', mec=style.INK, mew=.7, zorder=5)
        ax.text(.97, .03, 'Ringed: main-text pair', transform=ax.transAxes,
                ha='right', va='bottom', fontsize=7.6, color=style.INK)
    ax.set_xscale('log')
    xs = [u['kb'] for u in used]
    lo, hi = min(xs)/1.5, max(xs)*1.5
    ax.set_xlim(lo, hi)
    ticks = [t for t in (1, 3, 10, 30, 100, 300, 1000, 3000) if lo <= t <= hi]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels([f'{t:,}' for t in ticks])
    ax.set_ylim(0, 88)
    ax.set_yticks([0, 20, 40, 60, 80])
    ax.tick_params(axis='both', labelsize=7.6, length=2.4, width=.6, pad=2)
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_linewidth(.6)
    ax.grid(color=style.BORDER, linewidth=.45, zorder=0)
    ax.set_xlabel('KB per image' if cell != 'q32d' else 'KB per document', fontsize=7.8, labelpad=1.5)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=7.8, labelpad=2)
    return dict(cell=cell, title=title, full_input=100*full['est'], points=used)


def main():
    E = read_estimates()
    tree = output_dir()
    style.apply_paper_style()
    fig = plt.figure(figsize=(WIDTH_PT/72, HEIGHT_PT/72))
    left, right, bottom, top_axes = 30.0, 6.0, 25.0, 128.0
    gap = 24.0
    w = (WIDTH_PT-left-right-3*gap)/4
    manifest = []
    for i, (cell, title, ylabel) in enumerate(PANELS):
        x0 = left+i*(w+gap)
        ax = fig.add_axes([x0/WIDTH_PT, bottom/HEIGHT_PT, w/WIDTH_PT, (top_axes-bottom)/HEIGHT_PT])
        manifest.append(draw(ax, E, cell, title, ylabel))
        style.panel_card(fig, (x0-6, top_axes+1, w+10, 16), None, title, title_size=8.2, header_height=16)
    handles = [Line2D([], [], color=c, marker=m, ls=ls if line else 'none', lw=.9, ms=ms,
                      mfc='white' if k == 'scale_normalized' else c, mec=c, mew=.9 if m in 'xs' else .6)
               for k, lab, c, m, ls, ms, line in SERIES]
    labels = [lab for _, lab, *_ in SERIES]
    handles.append(Line2D([], [], color=FULL_LINE, lw=.7, ls=(0, (3, 2))))
    labels.append('Full input, uncoded')
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, 1.0), ncol=5, frameon=False, fontsize=7.6,
               handlelength=2.0, columnspacing=1.4, handletextpad=.5, borderaxespad=.2)
    out = tree/'supp_coding_rate.pdf'
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, metadata={'CreationDate': None, 'ModDate': None})
    plt.close(fig)
    print(out)


if __name__ == '__main__':
    output_dir()
    main()
