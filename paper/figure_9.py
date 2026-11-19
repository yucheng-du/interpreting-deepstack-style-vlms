"""Draw Figure 9 from matched-target generation summaries at the two fixed-source settings."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

from analysis.paths import INPUT, OUT as RENDERED, output_dir
OUT = RENDERED

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from paper import plot_style as paper_style
# Paper palette: main blue, later teal, paired differences in ink; the second fixed source is a
# neutral grey with a dashed line and a triangle (coral stays reserved for the edited object).
BLUE, ORANGE, INK, PURPLE = paper_style.MAIN, paper_style.LATER, paper_style.INK, '#7F8C99'
paper_style.apply_paper_style()
plt.rcParams.update({
    'font.size':8, 'axes.labelsize':8,
    'axes.titlesize':8.5, 'xtick.labelsize':8, 'ytick.labelsize':8,
    'legend.fontsize':8, 'pdf.fonttype':42, 'ps.fonttype':42,
    'svg.fonttype':'none', 'axes.linewidth':.6, 'axes.edgecolor':'#8DA3B5',
    'text.color':INK, 'axes.labelcolor':INK, 'xtick.color':INK, 'ytick.color':INK,
    'xtick.major.width':.6, 'ytick.major.width':.6,
    'xtick.major.size':2, 'ytick.major.size':2, 'axes.unicode_minus':True,
})

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def pointer(data,path):
    for key in path.strip('/').split('/'):
        key=key.replace('~1','/').replace('~0','~')
        data=data[int(key)] if isinstance(data,list) else data[key]
    return data

def style(ax):
    ax.spines[['top','right']].set_visible(False)
    ax.axhline(0,color='#A4ADB5',lw=.6,zorder=0)
    ax.tick_params(pad=2)
    ax.set_axisbelow(True)

def eb(ax,xs,rows,color,marker,offset=0,ls='-'):
    mm=[r['mean'] for r in rows];ci=[r['ci95'] for r in rows]
    ax.errorbar([v+offset for v in xs],mm,
        yerr=[[m-c[0] for m,c in zip(mm,ci)],[c[1]-m for m,c in zip(mm,ci)]],
        color=color,marker=marker,ls=ls,lw=.9,elinewidth=.7,capsize=1.7,
        capthick=.7,markersize=3.3,markerfacecolor='white',markeredgewidth=.9,zorder=3)

def txt(fig,x,y,text,**kw):
    return fig.text(x/504,y/154,text,**kw)

def main():
    points=[]
    for row in json.loads((INPUT/'matched/dose_condition_boundaries.json').read_text()):
        if row['model']=='qwen32b' and row['fixed']=='a':
            points.append(dict(panel='a', ci95=row['ci95'], mean=row['mean'], n=row['n'], display_row=row))
    assert len(points)==6
    summary=json.loads((INPUT/'cosmos_super_dose.json').read_text())
    path='/by_attribute/stripe_orientation/generation_matched_target_selectivity/summaries'
    for row in pointer(summary,path):
        if (row['response_format'], row['mapping'], row['reference_target'], row['endpoint']) != (
                'direct','none','later_native','R_main_minus_R_later'):
            continue
        assert row['changing_layout']==row['fixed_layout']=='M0'
        assert row['changing_strength']==row['fixed_strength']==1 and row['region']=='all'
        stats=row['statistics']
        points.append(dict(panel='b', mean=stats['mean'], ci95=stats['normal_95_interval'], n=stats['n'],
                           display_row=dict(fixed=row['fixed_other_assignment'],metric=row['dose_metric'])))
    assert len(points)==10 and all(p['n']==32 for p in points)

    fig=plt.figure(figsize=(7,154/72),dpi=144)
    left=fig.add_axes([35/504,39/154,199/504,76/154]);style(left)
    left.set_ylim(-.25,.85);left.set_yticks([-.2,0,.2,.4,.6,.8]);left.set_xlim(-.4,1.4)
    left.set_xticks([0,1],['RSS','L1'])
    for entry,color,marker,offset in [('main',BLUE,'o',-.045),('later',ORANGE,'s',0),('main_minus_later',INK,'D',.045)]:
        rr=[p for p in points if p['panel']=='a' and p['display_row']['entry']==entry]
        rr.sort(key=lambda p:p['display_row']['setting'])
        eb(left,[0,1],rr,color,marker,offset)
    left.set_ylabel('Generation R or ΔR',labelpad=1.5)
    paper_style.panel_card(fig,(30,124,240,30),'a','Qwen32 · stripes · fixed source a',title_size=8.5,header_height=18)
    txt(fig,134.5,8,'Main-native target',ha='center',fontsize=8)
    handles=[Line2D([0],[0],color=c,marker=m,mfc='white',lw=.9,ms=3.3,label=l)
        for c,m,l in [(BLUE,'o','Main R'),(ORANGE,'s','Later R'),(INK,'D','Paired ΔR')]]
    fig.legend(handles=handles,loc='center',bbox_to_anchor=(134/504,129/154),ncol=3,
        frameon=False,handlelength=1.35,columnspacing=.8,borderaxespad=0,handletextpad=.4)
    right=fig.add_axes([294/504,39/154,199/504,76/154]);style(right)
    right.set_ylim(-.35,.60);right.set_yticks([-.2,0,.2,.4,.6]);right.set_xlim(-.4,1.4)
    right.set_xticks([0,1],['RSS','L1'])
    for fixed,color,marker,ls,offset in [('a',INK,'D','-',-.025),('b',PURPLE,'^','--',.025)]:
        rr=[p for p in points if p['panel']=='b' and p['display_row']['fixed']==fixed]
        rr.sort(key=lambda p:['rss','l1'].index(p['display_row']['metric']))
        eb(right,[0,1],rr,color,marker,offset,ls)
    right.set_ylabel('Generation ΔR',labelpad=3)
    paper_style.panel_card(fig,(289,124,210,30),'b','Super · stripes',title_size=8.5,header_height=18)
    txt(fig,393.5,8,'Later-native target',ha='center',fontsize=8)
    handles=[Line2D([0],[0],color=c,marker=m,ls=ls,mfc='white',lw=.9,ms=3.3,label=l)
        for c,m,ls,l in [(INK,'D','-','Fixed source a'),(PURPLE,'^','--','Fixed source b')]]
    fig.legend(handles=handles,loc='center',bbox_to_anchor=(393/504,129/154),ncol=2,
        frameon=False,handlelength=1.8,columnspacing=1.5,borderaxespad=0)
    fig.canvas.draw()
    for ax,which in ((left,'a'),(right,'b')):
        low,high=ax.get_ylim()
        assert all(low<=p['ci95'][0]<=p['mean']<=p['ci95'][1]<=high for p in points if p['panel']==which)
    for ext in ('pdf','svg','png'):
        fig.savefig(OUT/('supp_dose_boundaries.'+ext),dpi=200,facecolor='white')
    plt.close(fig)


if __name__ == '__main__':
    output_dir()
    main()
