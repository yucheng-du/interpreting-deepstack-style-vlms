"""Patch-aligned identity/position routing stimuli with pixel-exact half swaps."""
from __future__ import annotations
import argparse
from copy import deepcopy
import hashlib
from itertools import product
import json
from pathlib import Path
import random

from cloud_prep.portable_data import _guard_path, sha256_file
from cloud_prep.measurement_v6.content_intervention import (SCHEMA,STRENGTHS,POSES,main_source,donor_source)

SEED=2026100506
SIZE=(384,256)
ATTRIBUTES={'color':('red','blue'),'stripe_orientation':('vertical stripes','horizontal stripes')}
SHAPE_PAIRS=(('circle','square'),('triangle','diamond'))


def _rng(key,seed=SEED):
    seed=int.from_bytes(hashlib.sha256(f'{seed}|{key}'.encode()).digest()[:8],'big')
    return random.Random(seed)


def scene_spec(attribute,shape_pair,layout,assignment,replicate,seed=SEED):
    if attribute not in ATTRIBUTES or shape_pair not in (0,1) or any(x not in (0,1) for x in (layout,assignment)) or type(replicate) is not int or replicate < 0:
        raise ValueError('Unknown scene design value')
    # Both halves share local geometry; every scene has an independent draw.
    # A cross source is an exact half swap, not an independently redrawn image.
    rng=_rng(f'{attribute}|{shape_pair}|{layout}|{assignment}|{replicate}|geometry',seed)
    x,y=96+rng.randrange(-12,13),128+rng.randrange(-28,29)
    centers=[[x,y],[x+SIZE[0]//2,y]]
    radius=rng.randrange(43,53)
    background=_rng(f'{attribute}|{replicate}|background',seed).choice((220,232,244))
    stripe_phase=_rng(f'{attribute}|{replicate}|phase',seed).randrange(14)
    return {'seed':seed,'attribute':attribute,'shape_pair':shape_pair,'objects':list(SHAPE_PAIRS[shape_pair]),
        'layout':layout,'assignment':assignment,'replicate':replicate,'centers':centers,'radius':radius,
        'background':background,'stripe_phase':stripe_phase,'permutation_geometry':{'kind':'swap_equal_horizontal_halves','half_width':SIZE[0]//2,'image_size':list(SIZE),'patch_size':16,'merge_size':2},
        'object_to_slot':{obj:(index^layout) for index,obj in enumerate(SHAPE_PAIRS[shape_pair])},
        'object_values_a':{obj:(index^assignment) for index,obj in enumerate(SHAPE_PAIRS[shape_pair])}}


def source_specs(strengths=STRENGTHS):
    sources={main_source(strength):{'outline':False,'strength':strength/100,'pose':'same','assignment':'a','type':'main'}
             for strength in sorted(set(strengths)|{0,100})}
    for pose,value in product(POSES,('a','b')):
        sources[donor_source(pose,value)]={'outline':False,'strength':1.,'pose':pose,'assignment':value,'type':'donor'}
    sources['blank']={'outline':False,'strength':0.,'pose':'same','assignment':'a','type':'blank'}
    return sources


def swap_image_halves(image):
    if image.width % 64 or image.height % 32:
        raise ValueError('Image halves must align to merged patch32 grid')
    half=image.width//2
    result=image.copy()
    with image.crop((half,0,image.width,image.height)) as right, image.crop((0,0,half,image.height)) as left:
        result.paste(right,(0,0));result.paste(left,(half,0))
    return result


def object_box(spec,obj,pose='same'):
    slot=spec['object_to_slot'][obj]^(pose=='cross')
    x,y=spec['centers'][slot];r=spec['radius']
    return [x-r,y-r,x+r,y+r]


def _shape(draw,obj,box,**kwargs):
    x0,y0,x1,y1=box
    if obj=='circle':
        draw.ellipse(box,**kwargs)
    elif obj=='square':
        draw.rectangle(box,**kwargs)
    else:
        points=([(int((x0+x1)/2),y0),(x1,y1),(x0,y1)] if obj=='triangle' else
                [(int((x0+x1)/2),y0),(x1,int((y0+y1)/2)),(int((x0+x1)/2),y1),(x0,int((y0+y1)/2))])
        draw.polygon(points,**kwargs)


def render_scene(spec,source_spec):
    from PIL import Image,ImageDraw
    if source_spec['pose']=='cross':
        with render_scene(spec,{**source_spec,'pose':'same'}) as original:
            return swap_image_halves(original)
    image=Image.new('RGB',SIZE,(spec['background'],)*3)
    if source_spec['type']=='blank':
        return image
    for obj in spec['objects']:
        box=object_box(spec,obj,source_spec['pose'])
        mask=Image.new('L',SIZE,0)
        _shape(ImageDraw.Draw(mask),obj,box,fill=255)
        value=spec['object_values_a'][obj]^(source_spec['assignment']=='b')
        neutral=Image.new('RGB',SIZE,(150,150,150))
        if spec['attribute']=='color':
            full=Image.new('RGB',SIZE,((214,43,49),(39,87,207))[value])
        else:
            full=Image.new('RGB',SIZE,(224,224,224))
            draw=ImageDraw.Draw(full)
            for offset in range(spec['stripe_phase']-14,max(SIZE),14):
                xy=(offset,0,offset,SIZE[1]) if value==0 else (0,offset,SIZE[0],offset)
                draw.line(xy,fill=(55,55,55),width=6)
        texture=Image.blend(neutral,full,source_spec['strength'])
        image.paste(texture,(0,0),mask)
        if source_spec['outline']:
            _shape(ImageDraw.Draw(image),obj,box,outline=(25,25,25),width=3)
        mask.close();neutral.close();full.close();texture.close()
    return image


def build(output,strengths=STRENGTHS,seed=SEED,replicates=4,overwrite=False):
    output=_guard_path(Path(output),write=True)
    output.mkdir(parents=True,exist_ok=overwrite);(output/'images').mkdir(exist_ok=overwrite)
    if type(replicates) is not int or replicates < 1:raise ValueError('Positive independent replicate count required')
    sources=source_specs(strengths);rows=[]
    for attribute,shape_pair,layout,assignment,replicate in product(ATTRIBUTES,range(2),range(2),range(2),range(replicates)):
        scene_id=f'{attribute}-p{shape_pair}-l{layout}-a{assignment}-r{replicate}'
        spec=scene_spec(attribute,shape_pair,layout,assignment,replicate,seed)
        paths,metadata={},{}
        for name,sspec in sources.items():
            relative=f'images/{scene_id}-{name}.png'
            with render_scene(spec,sspec) as image:
                image.save(output/relative)
            paths[name]=relative
            metadata[name]={'source_field':'image_1','width':SIZE[0],'height':SIZE[1],
                'bytes':(output/relative).stat().st_size,'sha256':sha256_file(output/relative)}
        for target_index,obj in enumerate(spec['objects']):
            choices=list(ATTRIBUTES[attribute])
            query=f'What color is the {obj}?' if attribute=='color' else f'What pattern is inside the {obj}?'
            prompt=query+'\n'+''.join(f'({chr(65+i)}) {choice}\n' for i,choice in enumerate(choices))
            value=spec['object_values_a'][obj]
            answers={name:None if ss['strength']==0 else value^(ss['assignment']=='b') for name,ss in sources.items()}
            slot_values={name:None if ss['strength']==0 else answers[name]^(ss['pose']=='cross') for name,ss in sources.items()}
            rows.append({'schema':SCHEMA,'id':scene_id+'-'+obj,'scene_id':scene_id,'task':'content_'+attribute,
                'attribute':attribute,'shape_pair':shape_pair,'layout':layout,'assignment':assignment,'replicate':replicate,
                'target_object':obj,'target_index':target_index,'target_main_slot':spec['object_to_slot'][obj],
                'scene':spec,'query':query,'prompt':prompt,'choices':choices,'option_labels':['(A)','(B)'],
                'answer_index':value,'answer_label':f'({chr(65+value)})',
                'source_answer_indices':answers,'source_slot_answer_indices':slot_values,
                'source_specs':sources,'source_images':paths,'source_image_metadata':metadata,
                'source':{'dataset':'DeepStack procedural counterfactuals','revision':SCHEMA,'split':'exploratory',
                          'seed':seed,'generator_sha256':sha256_file(Path(__file__))}})
    manifest=output/'manifest.jsonl'
    manifest.write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in rows))
    result=validate(manifest)
    (output/'dataset.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def load_manifest(manifest,image_root=None,*,check_hashes=True):
    manifest=_guard_path(Path(manifest));root=_guard_path(Path(image_root)) if image_root else manifest.parent
    rows=[json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    if not rows or len({r['id'] for r in rows})!=len(rows):
        raise ValueError('Empty/duplicate manifest')
    checked={};result=[]
    for row in rows:
        row_strengths=[round(s['strength']*100) for s in row['source_specs'].values() if s['type']=='main']
        declared_sources=source_specs(row_strengths)
        if row['schema']!=SCHEMA or row['source_specs']!=declared_sources:
            raise ValueError('Unknown schema or source design')
        spec=scene_spec(*(row[k] for k in ('attribute','shape_pair','layout','assignment','replicate')),seed=row['source']['seed'])
        if spec!=row['scene'] or row['target_object'] not in spec['objects']:
            raise ValueError('Scene/object disagreement')
        value=spec['object_values_a'][row['target_object']]
        expected={n:None if s['strength']==0 else value^(s['assignment']=='b') for n,s in declared_sources.items()}
        if row['source_answer_indices']!=expected or row['source_slot_answer_indices']!={n:None if v is None else v^(declared_sources[n]['pose']=='cross') for n,v in expected.items()}:
            raise ValueError('Object-versus-spatial source answers disagree')
        if row['answer_index']!=value or row['answer_label']!=row['option_labels'][value] or row['choices']!=list(ATTRIBUTES[row['attribute']]):
            raise ValueError('Content answer/choice mismatch')
        loaded=deepcopy(row)
        for name,path in row['source_images'].items():
            rel=Path(path)
            if rel.is_absolute() or '..' in rel.parts:
                raise ValueError('Images must use contained portable paths')
            path=_guard_path(root/rel)
            if not path.is_relative_to(root) or not path.is_file():
                raise ValueError('Image missing or escaping data root')
            meta=row['source_image_metadata'][name]
            if (meta['width'],meta['height'])!=SIZE:
                raise ValueError('Fixed source geometry changed')
            if check_hashes:
                if path not in checked:
                    checked[path]=(path.stat().st_size,sha256_file(path))
                if checked[path]!=(meta['bytes'],meta['sha256']):
                    raise ValueError('Image bytes/SHA changed')
            loaded['source_images'][name]=str(path)
        if set(row['source_images'])!=set(declared_sources):
            raise ValueError('Missing/extra source images')
        result.append(loaded)
    return result


def model_row(row,source=None):
    source=source or main_source(100)
    return {**row,'images':[row['source_images'][source]],'image_metadata':[row['source_image_metadata'][source]]}


def validate(manifest,image_root=None):
    rows=load_manifest(manifest,image_root)
    images={p for r in rows for p in r['source_images'].values()}
    return {'schema':SCHEMA,'manifest':str(manifest),'manifest_sha256':sha256_file(manifest),'items':len(rows),
        'scene_templates':len({r['scene_id'] for r in rows}),'unique_image_paths':len(images),
        'image_bytes':sum(Path(p).stat().st_size for p in images),'status':'passed','model_results':False,
        'seed':rows[0]['source']['seed'],'source_count_per_scene':len(rows[0]['source_specs']),
        'semantics':'Independent scenes; query, mapping, response format, main strength, donor pose and token permutation are within-scene repeats. Cross pixels are exact equal-half swaps.'}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('build','validate'))
    p.add_argument('--strengths',type=int,nargs='+',default=list(STRENGTHS));p.add_argument('--seed',type=int,default=SEED)
    p.add_argument('--replicates',type=int,default=4);p.add_argument('--output',type=Path);p.add_argument('--manifest',type=Path);p.add_argument('--image-root',type=Path)
    a=p.parse_args(argv)
    if a.command=='build' and a.output is None:p.error('build requires --output')
    if a.command=='validate' and a.manifest is None:p.error('validate requires --manifest')
    if any(not 0<=s<=100 for s in a.strengths):p.error('Strengths use 0..100 percent')
    print(json.dumps(build(a.output,a.strengths,a.seed,a.replicates) if a.command=='build' else validate(a.manifest,a.image_root),indent=2))

if __name__=='__main__':main()
