#!/usr/bin/env python3
"""CPU image preparation for new, editable v14 conditions; never runs a model.

Every generated endpoint pair changes one named object. Source a, source b,
and one deterministic degraded a share image geometry. Existing v12 materials
can also be imported for NEW conditions; their original files remain untouched.
"""
from __future__ import annotations
import argparse, hashlib, json, math, random, shutil
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageFilter

TASKS = ('color','stripe_orientation','relative_position','text_reading')
SIZE = (384,256)

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def key(seed, *parts): return int.from_bytes(hashlib.sha256(str((seed,parts)).encode()).digest()[:8],'big')
def dump(path, value): Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
def degrade(image, kind, strength):
    if kind == 'contrast': return ImageEnhance.Contrast(image).enhance(strength)
    if kind == 'blur': return image.filter(ImageFilter.GaussianBlur(strength))
    if kind == 'grayscale': return Image.blend(image,image.convert('L').convert('RGB'),strength)
    raise ValueError(kind)
def font_path():
    for p in ('/System/Library/Fonts/Supplemental/Arial.ttf','/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'):
        if Path(p).is_file(): return Path(p)
    raise ValueError('Set --font to a TrueType font installed on this host')
def save(image,path):
    image.save(path)
    return {'sha256':sha(path),'width':image.width,'height':image.height,'bytes':path.stat().st_size}
def shape(draw,name,box,**kwargs):
    (draw.ellipse if name=='circle' else draw.rectangle)(box,**kwargs)
def render(spec,task,choices,changed,b, font):
    image=Image.new('RGB',SIZE,(spec['background'],)*3)
    for j,name in enumerate(('circle','square')):
        box=spec['boxes'][j]; val=spec['values'][j] ^ (b and j==changed)
        mask=Image.new('L',SIZE); shape(ImageDraw.Draw(mask),name,box,fill=255)
        texture=Image.new('RGB',SIZE,(205,205,205)); d=ImageDraw.Draw(texture)
        cx,cy=(box[0]+box[2])/2,(box[1]+box[3])/2
        if task=='color':
            texture.paste(choices[val],(0,0,*SIZE))
        elif task=='stripe_orientation':
            texture.paste((235,235,235),(0,0,*SIZE)); d=ImageDraw.Draw(texture)
            for off in range(spec['phase']-14,max(SIZE),14):
                d.line((off,0,off,SIZE[1]) if val==0 else (0,off,SIZE[0],off),fill=(45,45,45),width=6)
        elif task=='relative_position':
            direction=-1 if val==0 else 1
            d.ellipse((cx-5,cy-5,cx+5,cy+5),fill='white',outline=(110,110,110))
            x=cx+direction*24
            d.ellipse((x-6,cy-6,x+6,cy+6),fill='black')
        else:
            f=ImageFont.truetype(str(font),spec['font_size'])
            d.text((cx,cy),choices[val],font=f,fill='black',anchor='mm')
        image.paste(texture,(0,0),mask)
        shape(ImageDraw.Draw(image),name,box,outline=(65,65,65),width=2)
    return image

def prepare(args):
    root=args.output.resolve()
    root.mkdir(parents=True,exist_ok=False); (root/'images').mkdir()
    rows=[]; specs=[]; image_count=0
    font=Path(args.font) if args.font else font_path()
    for n in range(args.templates):
        rng=random.Random(key(args.seed,n)); radius=rng.randrange(44,51)
        y=rng.randrange(104,153); shift=rng.randrange(-8,9); slots=(0,1) if n%2==0 else (1,0)
        boxes=[[96+192*s+shift-radius,y-radius,96+192*s+shift+radius,y+radius] for s in slots]
        spec={'boxes':boxes,'values':[n%2,1-n%2],'background':rng.choice([224,232,240]),'phase':rng.randrange(14),'font_size':rng.choice([18,22,26])}
        tid=f'v14-s{args.seed}-t{n:04d}'; specs.append({'unit_id':tid,**spec})
        for task in args.tasks:
            choices={'color':[['red','blue'],['green','purple'],['orange','blue']][n%3],
                     'stripe_orientation':['vertical stripes','horizontal stripes'],
                     'relative_position':['left','right'],
                     'text_reading':[['R7','M2'],['K4','N8'],['P3','T6'],['B9','H5']][n%4]}[task]
            for changed in range(2):
                iid=f'{tid}-{task}-edit{changed}'; paths={}; meta={}
                a=render(spec,task,choices,changed,False,font); b=render(spec,task,choices,changed,True,font)
                da=degrade(a,args.degradation,args.strength)
                for name,img in [('a',a),('b',b),('degraded_a',da)]:
                    rel=Path('images')/f'{iid}-{name}.png'; paths[name]=str(rel); meta[name]=save(img,root/rel); image_count+=1
                # Pair differences must be contained by the named object's ROI.
                import PIL.ImageChops
                actual=PIL.ImageChops.difference(a,b).getbbox(); box=boxes[changed]
                if actual is None or not (box[0]<=actual[0] and box[1]<=actual[1] and actual[2]<=box[2]+1 and actual[3]<=box[3]+1):
                    raise AssertionError('Counterfactual escaped its object ROI')
                for queried,name in enumerate(('circle','square')):
                    query={'color':f'What color is the {name}?','stripe_orientation':f'What pattern is inside the {name}?',
                           'relative_position':f'Is the black dot to the left or right of the white dot inside the {name}?',
                           'text_reading':f'What two-character code is written inside the {name}?'}[task]
                    val=spec['values'][queried]
                    rows.append({'schema':'followup-material-v14','id':f'{iid}-q{queried}','unit_id':tid,'template_id':tid,'scene_id':tid,
                        'intervention_id':iid,'task':task,'attribute':task,'query':query,'choices':choices,'option_labels':['(A)','(B)'],
                        'answer_index':val,'answer_label':['(A)','(B)'][val],'reference_answers':[choices[val]],
                        'direction_a_index':spec['values'][changed],'direction_b_index':1-spec['values'][changed],
                        'query_role':'changed' if queried==changed else 'unchanged','query_target':'changed' if queried==changed else 'unchanged',
                        'changed_index':changed,'changed_object':['circle','square'][changed],'queried_index':queried,'queried_object':name,
                        'donor_roi_bbox_xyxy':[box[0],box[1],box[2]+1,box[3]+1],'donor_roi_image_size':list(SIZE),
                        'source_images':paths,'source_image_metadata':meta,'source_answer_indices':{'a':val,'b':val^(queried==changed),'degraded_a':None},
                        'degradation':{'kind':args.degradation,'strength':args.strength,'same_realization_for_all_paired_conditions':True},
                        'reference_semantics':'clean a task recovery; mixed a/b states have no unique gold answer',
                        'source':{'dataset':'DeepStack procedural followup','revision':'v14','seed':args.seed,'split':'exploratory','generator_sha256':sha(__file__),
                                  'font_sha256':sha(font),'selection_uses_model_outputs':False}})
                a.close();b.close();da.close()
    (root/'materials.jsonl').write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in rows))
    dump(root/'specs.json',specs)
    report={'schema':'followup-preparation-v14','manifest':'materials.jsonl','manifest_sha256':sha(root/'materials.jsonl'),
            'units':args.templates,'query_rows':len(rows),'tasks':args.tasks,'image_files':image_count,
            'degradation':{'kind':args.degradation,'strength':args.strength},'pixel_geometry':list(SIZE),
            'validated':'pair edits contained in ROI; same-sized endpoints; unique query ids; paired source provenance',
            'model_or_processor_validation':'pending cloud; same pixel sizes do not alone prove postprocessor grid alignment',
            'font':str(font),'font_sha256':sha(font),'model_runs':False}
    dump(root/'preparation.json',report)
    # A small actual-input contact sheet, one shared template across all tasks.
    chosen=[r for r in rows if r['unit_id']==rows[0]['unit_id'] and r['changed_index']==0 and r['queried_index']==0]
    sheet=Image.new('RGB',(384*3, (256+34)*len(chosen)), 'white');d=ImageDraw.Draw(sheet)
    for i,row in enumerate(chosen):
        for j,name in enumerate(('a','b','degraded_a')):
            d.text((j*384+5,i*290+5),row['task']+' / '+name,fill='black')
            with Image.open(root/row['source_images'][name]) as im:sheet.paste(im,(j*384,i*290+30))
    sheet.save(root/'contact_sheet.png'); print(json.dumps(report,indent=2))

def import_existing(args):
    """Reuse accepted material pixels in new conditions, not old model runs."""
    root=args.output.resolve();root.mkdir(parents=True,exist_ok=False);(root/'images').mkdir()
    rows=[json.loads(x) for x in args.manifest.read_text().split('\n') if x.strip()];out=[];cache={}  # physical lines, not splitlines
    for old in rows:
        r=dict(old); paths={};meta={}
        for name,source in [('a',args.a_source),('b',args.b_source)]:
            p=Path(old['source_images'][source]);p=p if p.is_absolute() else args.manifest.parent/p
            identity=sha(p); rel=Path('images')/f'{identity}.png'
            if not (root/rel).exists():
                with Image.open(p) as im: save(im.convert('RGB'),root/rel)
            with Image.open(root/rel) as im:meta[name]={'sha256':sha(root/rel),'width':im.width,'height':im.height,'bytes':(root/rel).stat().st_size}
            paths[name]=str(rel)
        p=root/paths['a'];identity=sha(p);rel=Path('images')/f'{identity}-degraded.png'
        if not (root/rel).exists():
            with Image.open(p) as im:save(degrade(im,args.degradation,args.strength),root/rel)
        with Image.open(root/rel) as im:meta['degraded_a']={'sha256':sha(root/rel),'width':im.width,'height':im.height,'bytes':(root/rel).stat().st_size}
        paths['degraded_a']=str(rel)
        r.update(source_images=paths,source_image_metadata=meta,unit_id=old.get('unit_id',old.get('template_id',old['scene_id'])),
                 degradation={'kind':args.degradation,'strength':args.strength},families=['conflict','recovery'])
        r['source']={**old.get('source',{}),'followup_original_manifest':str(args.manifest),'followup_original_manifest_sha256':sha(args.manifest),
                     'reuse_scope':'same materials, NEW fixed-later conditional interventions; not independent source replication'}
        if old.get('source_answer_indices',{}).get(args.a_source) is not None:
            r['answer_index']=old['source_answer_indices'][args.a_source];r['answer_label']=r['option_labels'][r['answer_index']]
        r['source_answer_indices']={'a':r['answer_index'],'b':old.get('source_answer_indices',{}).get(args.b_source),'degraded_a':None}
        r['reference_answers']=[r['choices'][r['answer_index']]]
        out.append(r)
    (root/'materials.jsonl').write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in out))
    report={'rows':len(out),'units':len(set(r['unit_id'] for r in out)),'original_manifest':str(args.manifest),'original_sha256':sha(args.manifest),
            'new_manifest_sha256':sha(root/'materials.jsonl'),'degradation':{'kind':args.degradation,'strength':args.strength},'model_runs':False}
    dump(root/'preparation.json',report);print(json.dumps(report,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--templates',type=int,default=32);p.add_argument('--seed',type=int,default=2026102801)
    p.add_argument('--tasks',nargs='+',choices=TASKS,default=list(TASKS));p.add_argument('--font')
    p.add_argument('--degradation',choices=['contrast','blur','grayscale'],default='contrast');p.add_argument('--strength',type=float,default=.12)
    p.add_argument('--manifest',type=Path);p.add_argument('--a-source',default='d_a');p.add_argument('--b-source',default='d_b')
    args=p.parse_args()
    if args.templates<1 or args.strength<0 or (args.degradation=='grayscale' and args.strength>1):p.error('Invalid count/strength')
    (import_existing if args.manifest else prepare)(args)
