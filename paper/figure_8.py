#!/usr/bin/env python3
"""Render target/control components and direction-aligned composition estimates; no model evaluations."""
import json
from analysis.paths import INPUT, OUT as RENDERED, output_dir
FIGS = RENDERED
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from paper.plot_style import apply_paper_style, panel_card, CANVAS, LATER, CORAL, BORDER, INK, MUTED
SOURCE=INPUT / 'components.json'
def main():
    j=json.loads(SOURCE.read_text()); plotted=[]
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.3,'axes.labelsize':8.7,'axes.titlesize':9.3,
        'xtick.labelsize':8,'ytick.labelsize':8.1,'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    apply_paper_style()
    # Three aligned panels use the manuscript width without the old two-storey gap.
    # Native font sizes are retained; only layout and non-data row offsets change.
    fig=plt.figure(figsize=(6.875,2.65),facecolor=CANVAS)
    cards=[]
    for bounds,letter,title in [((3,21,184,167),'a','Natural photos'),
                                ((192,21,151,167),'b','Qwen32 score'),
                                ((348,21,144,167),'c','Qwen32 generation')]:
        cards.append(panel_card(fig,bounds_pt=bounds,letter=letter,title=title,title_size=9.3,header_height=19))
    a=fig.add_axes([.153,.255,.215,.615])
    b=fig.add_axes([.495,.255,.191,.615])
    c=fig.add_axes([.791,.255,.191,.615])
    def point(ax,stat,y,color,marker,sign=1,key='',hollow=False):
        mean=sign*stat['mean']; lo,hi=sorted([sign*stat['ci_low'],sign*stat['ci_high']])
        ax.errorbar(mean,y,xerr=[[mean-lo],[hi-mean]],fmt=marker,color=color,markersize=4.3,elinewidth=.9,capsize=1.8,lw=0,
                    markerfacecolor='white' if hollow else color,markeredgewidth=.9 if hollow else None)
        plotted.append({'source':key,'mean':mean,'ci_low':lo,'ci_high':hi,'n':stat['n'],'direction_multiplier':sign})
    models=[('qwen8b','Qwen8'),('qwen32b','Qwen32'),('ibm4b','Granite'),('deepstack7b','DeepStack')]
    ylabels=[]
    for index,(model,label) in enumerate(models):
        for route,off in [('main',0),('later',1)]:
            y=7-(index*2+off); ylabels.append((y,label+' '+route))
            for target,dy,col,marker in [('target',.18,CORAL,'o'),('control',-.18,MUTED,'s')]:
                key=route+'_'+target
                point(a,j['p1_components'][model]['estimates'][key],y+dy,col,marker,key=f'p1_components/{model}/estimates/{key}')
    for y in [1.5,3.5,5.5]:a.axhline(y,color=BORDER,lw=.6)
    a.set_yticks([x[0] for x in ylabels],[x[1] for x in ylabels]);a.set_ylim(-.6,7.65);a.set_xlim(-.055,.9)
    a.set_xlabel('Directed answer change',labelpad=3)
    a.set_xticks([0,.4,.8])
    a.tick_params(axis='y',length=0,pad=4)
    a.axvline(0,color='#c5c5c5',lw=.7,zorder=0)
    fig.legend(handles=[Line2D([],[],marker='o',color=CORAL,ls='',label='Target'),Line2D([],[],marker='s',color=MUTED,ls='',label='Control')],loc='lower center',bbox_to_anchor=(.235,.012),ncol=2,frameon=False,columnspacing=.9,handletextpad=.3,handlelength=.9)
    colors={'all':LATER,'sum_singletons':LATER,'C':INK}; markers={'all':'o','sum_singletons':'s','C':'D'}; hollow={'sum_singletons'}
    contexts=['M0_s008','M1_s008','M0_s100','M1_s100']
    for ax,kind,title in [(b,'score','(b) Qwen32 score'),(c,'generation','(c) Qwen32 generation')]:
        for i,context in enumerate(contexts):
            sign=-1 if context.startswith('M1') else 1
            for name,dy in [('all',.18),('sum_singletons',0),('C',-.18)]:
                point(ax,j['qwen32_composition'][kind]['contexts'][context]['estimates'][name],3-i+dy,colors[name],markers[name],sign,
                      f'qwen32_composition/{kind}/contexts/{context}/estimates/{name}',hollow=name in hollow)
        ax.set_yticks([3,2,1,0],['M0, .08','M1*, .08','M0, 1','M1*, 1']);ax.set_ylim(-.5,3.5)
        ax.axvline(0,color='#bfbfbf',lw=.7,zorder=0)
        # The unchanged title is now in its panel's shared heading band.
        ax.set_xlabel('Aligned effect'+(' (nats)' if kind=='score' else ''),labelpad=3)
        ax.tick_params(axis='y',length=0,pad=4)
    b.set_xlim(-3.5,14);c.set_xlim(-.34,.62)  # retain the full lowest generation interval
    b.set_xticks([0,4,8,12]);c.set_xticks([-.2,0,.2,.4,.6])
    fig.legend(handles=[Line2D([],[],marker=markers[k],color=colors[k],ls='',label=label,
                               markerfacecolor='white' if k in hollow else colors[k]) for k,label in [('all','Joint'),('sum_singletons','Singleton sum'),('C','Residual C')]],loc='lower center',bbox_to_anchor=(.74,.012),ncol=3,frameon=False,columnspacing=.75,handletextpad=.25,handlelength=.9)
    for ext in ['pdf','svg','png']:
        fig.savefig(FIGS/f'supp_entry_components.{ext}',dpi=180,facecolor=CANVAS)
    plt.close(fig)


if __name__ == '__main__':
    output_dir()
    main()
