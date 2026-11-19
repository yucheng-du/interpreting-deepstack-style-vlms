#!/usr/bin/env python3
"""Compact four accepted v12 support plots; copy estimates, never recompute them."""
from __future__ import annotations
import hashlib
import itertools
import json
from pathlib import Path

from analysis.paths import INPUT, OUT as RENDERED, output_dir
SOURCE = INPUT / "matched"
OUTPUT = RENDERED

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.text import Text
from matplotlib.ticker import MaxNLocator
from paper import plot_style as style
MODELS = ["qwen8b","qwen32b","ibm4b"]
NAMES = dict(qwen8b="Qwen8",qwen32b="Qwen32",ibm4b="Granite")
TICKS = ["Raw","M/R","M/L","L/R","L/L"]
def errorbars(ax, x, rows, color, marker='o', label=None):
    x = list(x)
    ax._v12_intervals = getattr(ax, '_v12_intervals', []) + list(zip(x, rows))
    y = [r['mean'] for r in rows]
    error = [[r['mean'] - r['ci95'][0] for r in rows],
             [r['ci95'][1] - r['mean'] for r in rows]]
    ax.errorbar(x, y, yerr=error, fmt=marker, color=color, label=label,
                ms=4.1, elinewidth=.85, capsize=2, capthick=.85, zorder=3)


def base_axis(ax):
    ax.axhline(0, color='#777777', lw=.65, zorder=0)
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_linewidth(.6)
    ax.tick_params(width=.6, length=2.5, pad=2)
    ax.grid(axis='y', color='#E9E9E9', lw=.5, zorder=0)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))


WIDTH = 495
DATA = {}
MANIFEST = []


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def select(rows, **query):
    found = [r for r in rows if all(r[k] == v for k,v in query.items())]
    if len(found) != 1:
        raise ValueError(('Nonunique existing point', query, len(found)))
    return found[0]


def load(name, count):
    path = SOURCE/(name+'.json')
    rows = json.loads(path.read_text())
    if len(rows) != count:
        raise ValueError('Unexpected accepted point inventory')
    for row in rows:
        original=row['original_row']
        stats=original.get('statistics', original)
        assert row['mean'] == stats['mean']
        assert row['ci95'][0] <= row['mean'] <= row['ci95'][1]
    return rows


def text(fig,x,y,label,**kwargs):
    return fig.text(x/WIDTH,y/(fig.get_figheight()*72),label,fontsize=8.1,**kwargs)


def axis(fig,x,y,w,h,lim,ticks,labels,yticks,ylabel=None):
    height = fig.get_figheight()*72
    ax = fig.add_axes([x/WIDTH,y/height,w/WIDTH,h/height])
    base_axis(ax)
    ax.set_facecolor('white')
    ax.set_xlim(-.45,len(ticks)-.55)
    ax.set_ylim(*lim)
    ax.set_xticks(ticks,labels)
    ax.set_yticks(yticks)
    ax.tick_params(labelsize=8.1,pad=2,length=2.2)
    ax.spines['bottom'].set_visible(False)
    if ylabel:
        ax.set_ylabel(ylabel,fontsize=8.1,labelpad=3)
    return ax


def legend(fig,y,items):
    handles=[Line2D([],[],color=c,marker=m,ls='',label=l,ms=4.1) for c,m,l in items]
    return fig.legend(handles=handles,loc='center',bbox_to_anchor=(.5,y/(fig.get_figheight()*72)),
                      ncol=len(items),frameon=False,fontsize=8.1,handletextpad=.3,
                      columnspacing=1.7,borderaxespad=0,borderpad=0)


def save(fig,name,rows,notes,cards):
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer(); canvas=fig.bbox
    # Legacy errorbar() retains exactly the row objects actually plotted.
    drawn=[r for ax in fig.axes for _,r in getattr(ax,'_v12_intervals',[])]
    encode=lambda r: json.dumps(r,sort_keys=True)
    if sorted(map(encode,drawn)) != sorted(map(encode,rows)):
        raise ValueError('Point inventory changed: '+name)
    outside=[]; visible=[]; skip=set()
    for ax in fig.axes:
        for x,r in getattr(ax,'_v12_intervals',[]):
            if not (ax.get_xlim()[0] <= x <= ax.get_xlim()[1] and
                    ax.get_ylim()[0] < r['ci95'][0] <= r['ci95'][1] < ax.get_ylim()[1]):
                raise ValueError(('Clipped point/interval',name,r))
        for obj,lim in [(ax.xaxis,ax.get_xlim()),(ax.yaxis,ax.get_ylim())]:
            for tick in obj.get_major_ticks()+obj.get_minor_ticks():
                if not min(lim)<=tick.get_loc()<=max(lim):
                    skip.update((id(tick.label1),id(tick.label2)))
    for artist in fig.findobj(Text):
        if not artist.get_visible() or not artist.get_text() or id(artist) in skip:
            continue
        box=artist.get_window_extent(renderer)
        if not canvas.contains(box.x0,box.y0) or not canvas.contains(box.x1,box.y1):
            outside.append(artist.get_text())
        visible.append((artist,box))
    if outside:
        raise ValueError(('Text outside canvas',name,outside))
    overlap=[]
    for (a,aa),(b,bb) in itertools.combinations(visible,2):
        if min(aa.x1,bb.x1)-max(aa.x0,bb.x0)>.5 and min(aa.y1,bb.y1)-max(aa.y0,bb.y0)>.5:
            overlap.append((a.get_text(),b.get_text()))
    if overlap:
        raise ValueError(('Text overlap',name,overlap))
    for card in cards:
        outer=card['surface'].get_window_extent(renderer)
        for key in ['title','letter']:
            bb=card[key].get_window_extent(renderer)
            if not (outer.x0<bb.x0<bb.x1<outer.x1 and outer.y0<bb.y0<bb.y1<outer.y1):
                raise ValueError('Heading outside its card')
    minimum=min(a.get_fontsize() for a,_ in visible)
    if minimum<8:
        raise ValueError('Text smaller than 8 pt')
    outname='supp_'+('dose_reference' if name == 'dose_reference_45' else name)
    for suffix in ['pdf','svg','png']:
        fig.savefig(OUTPUT/(outname+'.'+suffix),dpi=240,bbox_inches=None)
    # This 144-dpi view is the final 495-pt asset rasterized at 2 pixels/point.
    fig.savefig(OUTPUT/(outname+'_target.png'),dpi=144,bbox_inches=None)
    MANIFEST.append({'name': outname, 'point_count': len(rows)})
    plt.close(fig)


def natural(kind):
    name='natural_'+kind; rows=load(name,33)
    dy=14  # one extra note line at the bottom that decodes the counts
    fig=plt.figure(figsize=(WIDTH/72,(240+dy)/72)); cards=[]; notes=[]
    legend(fig,229+dy,[(style.MAIN,'o','Main entry'),(style.LATER,'s','Later entries'),
                    (style.INK,'o','Paired main − later')])
    # Axis ranges are keyed by model (they were indexed by the old Granite-first column order).
    score_top={'ibm4b':((-.1,1.8),[0,.8,1.6]),'qwen8b':((-1,22),[0,10,20]),'qwen32b':((-1.2,26),[0,10,20])}
    score_bottom={'ibm4b':((-1.05,.6),[-1,0,.5]),'qwen8b':((-1,21),[0,10,20]),'qwen32b':((-1,24),[0,10,20])}
    for col,model in enumerate(MODELS):
        left=5+164*col
        cards.append(style.panel_card(fig,(left,40+dy,157,177),None,NAMES[model],title_size=8.7,header_height=17))
        top_lim,top_ticks=((- .12,.88),[0,.4,.8]) if kind=='generation' else score_top[model]
        bottom_lim,bottom_ticks=((- .44,.88),[-.4,0,.4,.8]) if kind=='generation' else score_bottom[model]
        top=axis(fig,left+38,147+dy,113,47,top_lim,range(3),['Target','Control','R'],top_ticks,
                 ('Generation\neffect' if kind=='generation' else 'Score effect') if col==0 else None)
        counts=[]
        for entry,offset,color,marker in [('main',-.1,style.MAIN,'o'),('later',.1,style.LATER,'s')]:
            selected=[select(rows,model=model,setting=0,endpoint=f'{entry}.{e}') for e in ['E_target','E_control','R']]
            errorbars(top,[i+offset for i in range(3)],selected,color,marker)
            counts.append('/'.join(str(r['n']) for r in selected))
        note=f'n: M {counts[0]}; L {counts[1]}'
        text(fig,left+78.5,125+dy,note,ha='center',va='center');notes.append(note)
        lower=[select(rows,model=model,setting=s,endpoint='main_minus_later.R') for s in range(5)]
        pair_note='Paired n = '+ '/'.join(map(str,sorted(set(r['n'] for r in lower))))
        text(fig,left+78.5,113+dy,pair_note,ha='center',va='center');notes.append(pair_note)
        bottom=axis(fig,left+38,61+dy,113,43,bottom_lim,range(5),TICKS,bottom_ticks,
                    'Paired ΔR' if col==0 else None)
        errorbars(bottom,range(5),lower,style.INK)
    context='R = target − control; ΔR = R(main) − R(later). Direct answers; M0; α = 1.'
    dose='M/R: main-native RSS   M/L: main-native L1   L/R: later-native RSS   L/L: later-native L1'
    counts='n: complete photos for target / control / R; M: main entry, L: later entries.'
    text(fig,8,26+dy,context,va='center');text(fig,8,12+dy,dose,va='center');text(fig,8,12,counts,va='center')
    notes += [context,dose,counts]
    save(fig,name,rows,notes,cards)


def stripe():
    name='stripe_spatial_interaction';rows=load(name,27)
    fig=plt.figure(figsize=(WIDTH/72,155/72));cards=[]
    for col,model in enumerate(MODELS):
        left=5+164*col
        cards.append(style.panel_card(fig,(left,36,157,114),None,NAMES[model],title_size=8.7,header_height=17))
        ax=axis(fig,left+34,56,117,70,(-2.1,2.1),range(3),['Direct','Identity','Cyclic'],[-2,0,2],
                'Generation interaction' if col==0 else None)
        for entry,offset,color,marker in [('main',-.18,style.MAIN,'o'),('later',0,style.LATER,'s'),('main_minus_later',.18,style.INK,'D')]:
            selected=[select(rows,model=model,entry=entry,readout=fmt) for fmt in ['Direct','Identity','Cyclic']]
            errorbars(ax,[i+offset for i in range(3)],selected,color,marker)
    legend(fig,25,[(style.MAIN,'o','Main'),(style.LATER,'s','Later'),(style.INK,'D','Main − later')])
    note='Layout × destination × query; α = 1.  n = 64 shared templates for every point.'
    text(fig,8,9,note,va='center')
    save(fig,name,rows,[note],cards)


def dose():
    name='dose_reference_45';rows=load(name,45)
    # Rows are 75 pt apart (68 before) so that each row title clears the tick labels above it.
    fig=plt.figure(figsize=(WIDTH/72,280/72));cards=[];notes=[]
    for col,model in enumerate(MODELS):
        left=5+164*col
        cards.append(style.panel_card(fig,(left,25,157,250),None,NAMES[model],title_size=8.7,header_height=17))
        for ri,(panel,label) in enumerate([('natural','Natural photos'),('color','Synthetic color'),('stripe_orientation','Synthetic stripes')]):
            selected=[select(rows,model=model,panel=panel,setting=s) for s in range(5)]
            y=196-75*ri
            ax=axis(fig,left+34,y,117,43,(-.85,1.13),range(5),TICKS,[-.5,0,.5,1],
                    'ΔR' if col==0 else None)
            errorbars(ax,range(5),selected,style.INK)
            note=label+' · n = '+ '/'.join(map(str,sorted(set(r['n'] for r in selected))))
            text(fig,left+78.5,y+52,note,ha='center',va='center');notes.append(note)
    dose_note='M/R: main-native RSS   M/L: main-native L1   L/R: later-native RSS   L/L: later-native L1'
    text(fig,8,18,dose_note,va='center')
    note='ΔR > 0: main more selective than later. Direct answers; α = 1; synthetic rows use fixed source a.'
    text(fig,8,5,note,va='center')
    notes += [dose_note,note,'Generation ΔR; direct answers; α = 1; synthetic rows use fixed source a.']
    save(fig,name,rows,notes,cards)


def main():
    style.apply_paper_style();plt.rcParams.update({'font.size':8.1})
    natural('generation');natural('score');stripe();dose()
    print(MANIFEST)

if __name__ == '__main__':
    output_dir()
    main()
