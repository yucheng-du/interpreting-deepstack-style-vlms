"""Rebuild the protocol tables and selected rows from the paired summaries."""
import argparse
import json
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from analysis.paths import INPUT, output_dir
from analysis.prepare_matched import one, photo
from analysis.coding_csv import records

MODELS = {'qwen2b':'Qwen3-VL-2B','qwen4b':'Qwen3-VL-4B','qwen8b':'Qwen3-VL-8B',
          'qwen30b_a3b':'Qwen3-VL-30B-A3B','qwen32b':'Qwen3-VL-32B','cosmos_nano':'Cosmos3-Nano',
          'cosmos_super':'Cosmos3-Super','ibm4b':'Granite Vision 4.1-4B','deepstack7b':'DeepStack-L-Vicuna-7B'}
FOUR = {'qwen8b':'Qwen8','qwen32b':'Qwen32','ibm4b':'Granite','deepstack7b':'DeepStack'}


def rounded(x, digits=3):
    return str(Decimal(str(x)).quantize(Decimal(1).scaleb(-digits), rounding=ROUND_HALF_UP))


def row(*cells):
    return ' & '.join(map(str,cells)) + r' \\'


def ci(r):
    return ','.join(rounded(r[k]) for k in ('ci_low','ci_high'))


def natural():
    for model,name in MODELS.items():
        reports=json.loads((INPUT/'photographs'/model/'generation.json').read_text())['summaries']
        if model in FOUR:
            yield r'\rowcolor{DSTableAccent}'
        vals=[]
        for entry in ('main','later','main_minus_later'):
            x=photo(reports,0,entry,'R')
            vals.extend([r'\('+rounded(x['mean'])+r'\;['+ci(x)+r']\)',x['complete_photos']])
        yield row(name,*vals)+'[1pt]'
        if model=='cosmos_super':
            yield r'\addlinespace[2pt]'


def corners():
    scores=list(records('configurations'))
    names={'C':'image','Z':'zeros','M':'mean'}
    for kind in ('C|C','C|Z','C|M','Z|C','M|C','Z|Z','M|M'):
        vals=[]
        for model in FOUR:
            image_set='granite999' if model=='ibm4b' else 'confirm1000'
            s=one(scores,model=model,image_set=image_set,config=kind)
            vals.append(rounded(s['score'],2))
        yield row(*(names[k] for k in kind.split('|')),*vals)


def fractions(part, totals):
    return f'{part}/{totals}'


def transitions(number):
    photos=json.loads((INPUT/'components.json').read_text())['p1_components']
    for model,name in FOUR.items():
        for entry in ('main','later'):
            count=photos[model]['common_complete_counts']
            t,c=(count[f'{entry}_{role}'] for role in ('target','control'))
            t_all=fractions(t['forward'],t['query_pairs'])
            c_all=fractions(c['forward'],c['query_pairs'])
            if number==6:
                yield row(name,entry.title(),t_all,c_all,fractions(t['photos_any_change'],t['photos']))
            else:
                yield row(name if entry=='main' else '',entry.title(),t['photos'] if entry=='main' else '',
                          fractions(t['forward'],t['at_risk']),t_all,fractions(c['forward'],c['at_risk']),c_all)


def composition(number):
    j=json.loads((INPUT/'components.json').read_text())['qwen32_composition']
    if number==8:
        for layout,lab in [('M0','Original'),('M1','Exchanged')]:
            for alpha,stamp in [('.08','s008'),('1','s100')]:
                rr=[j[k]['contexts'][f'{layout}_{stamp}']['estimates']['C'] for k in ('score','generation')]
                yield row(lab,alpha,*['$'+rounded(r['mean'])+r'\ ['+ci(r)+']$' for r in rr],rr[0]['n'])
    else:
        for kind in ('generation','score'):
            for alpha,stamp in [('.08','s008'),('1','s100')]:
                r=j[kind]['aligned_layout'][stamp]['estimates']['C']
                yield row(kind.title(),alpha,'$'+rounded(r['mean'])+'$','$['+ci(r)+']$')


def common_low():
    for model,label in [('qwen8b','Qwen8'),('qwen32b','Qwen32')]:
        data={}
        for kind in ('score','generation'):
            j=json.loads((INPUT/'decomposition'/model/(kind+'.json')).read_text())
            data[kind]={k:j[k] for k in ('summary','contrast_summary')}
        yield r'\multicolumn{7}{@{}l}{\textbf{'+label+r'}} \\[2pt]'
        for mode,lab in [('raw','Raw ($k=1$)'),('common_low_rss','Low RSS'),('common_low_l1','Low L1')]:
            vals=[]
            for kind in ('score','generation'):
                params=dict(attribute='color',kind=kind,response_format='direct',mapping='none',context='M0_s100',mode=mode)
                selected=[one(data[kind]['summary'],**params,measure='R',group=g) for g in ('main','all')]
                selected.append(one(data[kind]['contrast_summary'],**params,contrast='main_minus_group',left='main',right='all'))
                for r in selected:
                    m=rounded(r['mean'])
                    interval=','.join(rounded(x) for x in r['normal_95_interval'])
                    vals.append(r'\begin{tabular}[c]{@{}c@{}}\('+m+r'\)\\\(['+interval+r']\)\end{tabular}')
            if mode=='common_low_rss':
                yield r'\rowcolor{DSTableAccent}'
            yield row(lab,*vals)+'[3pt]'


def render(number):
    source=Path(__file__).resolve().parent/'templates'/f'table_{number}.tex'
    t=source.read_text()
    if '@@ROWS@@' in t:
        routines={2:natural,3:corners,6:lambda:transitions(6),7:lambda:transitions(7),
                  8:lambda:composition(8),9:lambda:composition(9),10:common_low}
        t=t.replace('@@ROWS@@','\n'.join(routines[number]()))
    doc=(r'\documentclass{article}'+'\n'+r'\usepackage[T1]{fontenc}\usepackage{times}\usepackage[margin=0.75in]{geometry}'+'\n'
         +r'\usepackage{booktabs,colortbl,tabularx,amsmath,xcolor}'+'\n'
         +r'\definecolor{DSTableHeader}{HTML}{E9EEF3}\definecolor{DSTableAccent}{HTML}{E8F2F2}'+'\n'
         +r'\newcommand{\main}{\mathrm{main}}\newcommand{\later}{\mathrm{later}}'+'\n'
         +r'\begin{document}\pagestyle{empty}'+f'\\setcounter{{table}}{{{number-1}}}'+ '\n'
         +t+r'\end{document}'+'\n')
    (output_dir()/f'table_{number}.tex').write_text(doc)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('numbers',nargs='+',type=int,choices=range(1,11))
    for n in parser.parse_args().numbers:
        render(n)
