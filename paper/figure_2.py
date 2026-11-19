"""Figure 2. Read paired spatial estimates from the three local-role summaries."""
from __future__ import annotations

import json

from analysis.paths import INPUT, OUT as RENDERED, output_dir
OUT = RENDERED
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from paper.plot_style import apply_paper_style, panel_card, MAIN, LATER, LATER_INK

BLUE, ORANGE, INK, PURPLE = '#326A9B', '#C27B35', '#29323B', '#79558D'
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 8,
    'axes.labelsize': 8, 'axes.titlesize': 8.5,
    'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'legend.fontsize': 8, 'pdf.fonttype': 42, 'ps.fonttype': 42,
    'svg.fonttype': 'none', 'axes.linewidth': .6,
    'axes.edgecolor': '#7E8991', 'text.color': INK,
    'axes.labelcolor': INK, 'xtick.color': INK, 'ytick.color': INK,
    'xtick.major.width': .6, 'ytick.major.width': .6,
    'xtick.major.size': 2, 'ytick.major.size': 2,
    'axes.unicode_minus': True,
})


def style(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.axhline(0, color='#A4ADB5', lw=.6, zorder=0)
    ax.tick_params(pad=2)
    ax.set_axisbelow(True)


def eb(ax, xs, rows, *, color, marker, ls='-', label=None, offset=0):
    means = [r['mean'] for r in rows]
    ci = [r['ci95'] for r in rows]
    return ax.errorbar([x + offset for x in xs], means,
        yerr=[[m-c[0] for m,c in zip(means,ci)], [c[1]-m for m,c in zip(means,ci)]],
        color=color, marker=marker, ls=ls, lw=.9, elinewidth=.7, capsize=1.7,
        capthick=.7, markersize=3.3, markerfacecolor='white', markeredgewidth=.9,
        label=label, zorder=3)


def txt(fig, x, y, text, height=154, **kw):
    return fig.text(x/504, y/height, text, **kw)


qa = {}


def save(fig, stem):
    fig.canvas.draw()
    for ext in ['pdf', 'svg', 'png']:
        fig.savefig(OUT / f'{stem}.{ext}', dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)


def spatial_figure():
    # Scope the new theme to the requested main-paper figure. The archived
    # supplementary boundary figure keeps its original presentation.
    with plt.rc_context():
        apply_paper_style()
        _spatial_figure()


# Figure 2 height after trimming the empty space under the tick and layout labels (was 154 pt).
FIG2_H = 141


def _spatial_figure():
    selected = []
    for model in ('ibm4b','qwen8b','qwen32b'):
        path = INPUT / 'spatial' / model / 'summary.json'
        for row in json.loads(path.read_text())['per_mapping_contrast_summaries']:
            if (row['contrast'], row['response_format'], row['mapping'], row['main_strength']) != (
                    'changed_minus_unchanged', 'direct', 'none', 0.08):
                continue
            stats = row['statistics']
            selected.append({'display_row': dict(row, model=model), 'n': stats['n'],
                             'mean': stats['mean'], 'ci95': stats['normal_95_interval']})
    assert len(selected) == 24 and all(p['n']==64 for p in selected)
    H = FIG2_H
    fig = plt.figure(figsize=(7,H/72), dpi=144)
    # Model order and short names follow the text and Tables 2-3.
    specs = [('qwen8b','Qwen8',(-21,43),[-20,0,20,40]),
             ('qwen32b','Qwen32',(-41,52),[-40,0,40]),
             ('ibm4b','Granite',(-3.4,4.4),[-2,0,2,4])]
    starts = [37,200,363]
    axes_bottom, axes_height = 27, 78
    card_top = axes_bottom+axes_height+19
    cards = [(18,2,160,card_top-2),(181,2,160,card_top-2),(344,2,157,card_top-2)]
    fig._paper_panel_cards = []
    # One panel per model: titles name the models, so the panels carry no letters.
    for (model,title,ylim,yticks),left,card,letter in zip(specs,starts,cards,[None]*3):
        fig._paper_panel_cards.append(panel_card(fig,card,letter,title,
                                                 title_size=8.5,header_height=18))
        ax = fig.add_axes([left/504,axes_bottom/H,132/504,axes_height/H])
        style(ax)
        ax.set_ylim(*ylim); ax.set_yticks(yticks)
        ax.set_xlim(-.3,3.5)
        ax.set_xticks([0,1,2.2,3.2],['Orig.','Opp.','Orig.','Opp.'])
        ax.set_facecolor('white')
        for entry,color,marker in [('main',MAIN,'o'),('later',LATER,'s')]:
            for layout,xx,ls in [('M0',[0,1],'-'),('M1',[2.2,3.2],'--')]:
                rr=[p for p in selected if p['display_row']['model']==model
                    and p['display_row']['changed_entry']==entry
                    and p['display_row']['main_layout']==layout]
                rr.sort(key=lambda p: ['original','opposite'].index(p['display_row']['source_destination']))
                eb(ax,xx,rr,color=color,marker=marker,ls=ls)
                if model=='qwen8b' and entry=='later':
                    for x,p in zip(xx,rr):
                        ax.annotate(f"{p['mean']:.2f}".replace('-', '\u2212'),(x,p['mean']),
                            xytext=(0,6),textcoords='offset points',
                            ha='center',va='bottom',fontsize=8,color=LATER_INK,
                            bbox={'facecolor':'white','edgecolor':'none','pad':.2})
        # Layout labels sit directly under the tick labels (fixed offset in points).
        for x,label in [(.5,'M0'),(2.7,'M1')]:
            ax.annotate(label,(x,0),xycoords=ax.get_xaxis_transform(),xytext=(0,-15.5),
                        textcoords='offset points',ha='center',va='top',fontsize=8)
    txt(fig,6,axes_bottom+axes_height/2,'Relative selectivity R (score)',height=H,
        rotation=90,va='center',fontsize=8)
    handles=[Line2D([0],[0],color=MAIN,marker='o',mfc='white',lw=.9,ms=3.3,label='Main'),
             Line2D([0],[0],color=LATER,marker='s',mfc='white',lw=.9,ms=3.3,label='All later')]
    legend_y = card_top+8
    fig.legend(handles=handles,loc='center left',bbox_to_anchor=(31/504,legend_y/H),
               ncol=2,frameon=False,handlelength=1.7,columnspacing=1.5,borderaxespad=0)
    txt(fig,495,legend_y,'Color \u00b7 \u03b1 = .08 \u00b7 direct score \u00b7 n = 64',height=H,
        ha='right',va='center',fontsize=8)
    save(fig,'fig2_spatial')


if __name__ == '__main__':
    output_dir()
    spatial_figure()
