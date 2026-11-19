"""v7 line B: natural-object spatial-correspondence bridge built from GQA scene graphs + Visual Genome crops.

Every scene is a CONTROLLED RECOMPOSITION: four real object crops (GQA boxes, annotated colors) letterboxed
into identical fixed boxes on a neutral 384x256 canvas. It is never an unedited natural whole image. Object
identity (category), layout (slot), target attribute (annotated color) and injection route (main/donor source)
are separate fields. Selection = declared rules + seeded hash order; data-label checks are model-independent
and recorded before any model output. CPU/PIL/numpy only; no torch/transformers imports.

  python -I -B cloud_prep/measurement_v7/natobj_data.py plan-fetch --gqa-root <gqa-official> --output <fetch_plan.json>
  python -I -B cloud_prep/measurement_v7/natobj_gqa.py fetch --root <gqa-official> --ids-file <fetch_plan.json> --workers 4
  python -I -B cloud_prep/measurement_v7/natobj_data.py build --gqa-root <gqa-official> --output <dataset dir>
  python -I -B cloud_prep/measurement_v7/natobj_data.py validate --manifest <dataset dir>/manifest.jsonl --pixels
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import gzip
import hashlib
from itertools import product
import json
from pathlib import Path
import os as _path_os
DATA_ROOT = Path(_path_os.environ.get("DS_DATA_ROOT", "work")).expanduser().resolve()
import shutil
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cloud_prep.portable_data import _guard_path, sha256_file
from cloud_prep.measurement_v6.content_data import swap_image_halves

SCHEMA='natobj-gqa-v7-1'
CONSTRUCTION='controlled_recomposition'
CONSTRUCTION_NOTE=('Real GQA/Visual Genome object crops (annotated box + context margin), aspect-preserving letterboxed '
                   'into identical fixed boxes on a uniform neutral canvas. Not an unedited natural whole image.')
CONFIG=Path(__file__).resolve().parent/'configs'/'natobj_gqa_v7.json'
POSES=('same','cross')
CONFIG_KEYS={'seed','splits','basic_colors','categories','category_pairs','color_pairs','quads_per_stratum','pool_per_cell',
             'spare_per_cell','max_crop_uses','max_image_uses','annotation','pixel_check','canvas','strengths',
             'manual_exclusions','visual_review','native_panel'}
CATEGORY_KEYS={'group','synonyms','context_exclude','occlusion_check'}
REASONS=('multi_color','both_pair_colors','bad_bbox','too_small','aspect','border_cut','duplicate_color_conflict',
         'other_instance_in_crop','context_conflict','occluded','manual_exclusion')


# ---------------------------------------------------------------- configuration

def read_config(path=None,overrides=()):
    """Declared selection/build parameters; ``overrides`` are 'key=<json>' strings (exploration stays adjustable)."""
    config=json.loads(Path(path or CONFIG).read_text())
    for item in overrides:
        key,sep,value=item.partition('=')
        if not sep or key not in CONFIG_KEYS:raise ValueError(f'Unknown override {item!r}')
        config[key]=json.loads(value)
    if set(config)!=CONFIG_KEYS:raise ValueError(f'Config keys differ: {sorted(set(config)^CONFIG_KEYS)}')
    cats=config['categories'];pair_colors={c for p in config['color_pairs'] for c in p}
    if any(set(v)!=CATEGORY_KEYS for v in cats.values()):raise ValueError(f'Each category needs {sorted(CATEGORY_KEYS)}')
    if any(len(p)!=2 or p[0]==p[1] or p[0] not in cats or p[1] not in cats for p in config['category_pairs']):
        raise ValueError('Category pairs need two distinct declared categories')
    if len({tuple(p) for p in config['category_pairs']})!=len(config['category_pairs']):raise ValueError('Duplicate category pair')
    if any(len(p)!=2 or p[0]==p[1] for p in config['color_pairs']) or not pair_colors<=set(config['pixel_check']['hue_bins']):
        raise ValueError('Color pairs need two distinct colors with declared hue bins')
    if not pair_colors<=set(config['basic_colors']):raise ValueError('Pair colors must be basic colors')
    for key in ('quads_per_stratum','pool_per_cell','max_crop_uses','max_image_uses'):
        if type(config[key]) is not int or config[key]<1:raise ValueError(f'{key} must be a positive int')
    if type(config['spare_per_cell']) is not int or config['spare_per_cell']<0:raise ValueError('spare_per_cell must be >= 0')
    s=config['strengths']
    if not s or any(type(x) is not int or not 0<=x<=100 for x in s) or len(set(s))!=len(s):raise ValueError('Unique 0..100 strengths')
    c=config['canvas'];(w,h),box=c['size'],c['box']
    if w%64 or h%32 or not 0<box<=min(w//2,h):raise ValueError('Canvas halves must align to the merged 32 px grid and hold the box')
    if type(c['background']) is not int or not 0<=c['background']<=255:raise ValueError('Neutral background level 0..255')
    return config


def config_sha(config):return hashlib.sha256(json.dumps(config,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def strata(config):
    return [{'id':f'{x}-{y}-{c0}-{c1}'.replace(' ','_'),'objects':[x,y],'colors':[c0,c1]}
            for (x,y),(c0,c1) in product(config['category_pairs'],config['color_pairs'])]


def cells(config):
    return sorted({(o,c) for s in strata(config) for o in s['objects'] for c in s['colors']})


def needed_per_cell(config):
    need=Counter()
    for s in strata(config):
        for o,c in product(s['objects'],s['colors']):need[(o,c)]+=config['quads_per_stratum']
    return need


def rank(seed,image_id,object_id):return hashlib.sha256(f'{seed}|{image_id}|{object_id}'.encode()).hexdigest()


# ---------------------------------------------------------------- annotation checks (model independent)

def _area(b):return max(0,b[2])*max(0,b[3])


def _inter(a,b):
    w=min(a[0]+a[2],b[0]+b[2])-max(a[0],b[0]);h=min(a[1]+a[3],b[1]+b[3])-max(a[1],b[1])
    return max(0,w)*max(0,h)


def _iou(a,b):
    i=_inter(a,b);u=_area(a)+_area(b)-i
    return i/u if u>0 else 0.


def central_box(bbox,frac):
    x,y,w,h=bbox
    return [round(x+w*(1-frac)/2),round(y+h*(1-frac)/2),round(x+w*(1+frac)/2),round(y+h*(1+frac)/2)]


def crop_box(bbox,image_size,margin):
    """Annotated box + margin*max(w,h) context on each side, clipped to the image: [x0,y0,x1,y1]."""
    x,y,w,h=bbox;W,H=image_size;m=round(margin*max(w,h))
    return [max(0,x-m),max(0,y-m),min(W,x+w+m),min(H,y+h+m)]


def annotation_reasons(obj,image,category,color,config):
    """Exclusion reasons for one GQA object (empty list = annotation sanity passed)."""
    a=config['annotation'];oid,name,x,y,w,h,attrs=obj;W,H=image['width'],image['height']
    basic=set(config['basic_colors']);pair_colors={c for p in config['color_pairs'] for c in p}
    colors={c for c in attrs if c in basic};reasons=[]
    if colors!={color}:reasons.append('both_pair_colors' if len(colors&pair_colors)>1 else 'multi_color')
    if w<=0 or h<=0 or x<0 or y<0 or x+w>W or y+h>H:reasons.append('bad_bbox')
    if w>0 and h>0:
        if min(w,h)<a['min_side'] or max(w,h)<a['min_long_side']:reasons.append('too_small')
        if max(w/h,h/w)>a['max_aspect']:reasons.append('aspect')
    m=a['border_margin']
    if x<m or y<m or x+w>W-m or y+h>H-m:reasons.append('border_cut')
    cat=config['categories'][category];synonyms=set(cat['synonyms'])|{category}
    cx0,cy0,cx1,cy1=crop_box([x,y,w,h],(W,H),config['canvas']['context_margin']);crop=[cx0,cy0,cx1-cx0,cy1-cy0]
    kx0,ky0,kx1,ky1=central_box([x,y,w,h],config['pixel_check']['central_frac']);core=[kx0,ky0,kx1-kx0,ky1-ky0]
    occluders=set(a['occluder_names']) if cat['occlusion_check'] else set()
    for other in image['objects']:
        if other[0]==oid:continue
        ob=other[2:6]
        if other[1] in synonyms:
            if _iou([x,y,w,h],ob)>=a['duplicate_iou']:
                if {c for c in other[6] if c in basic}-{color}:reasons.append('duplicate_color_conflict')
            elif _area(ob)>0 and _inter(crop,ob)>=a['other_instance_crop_frac']*_area(ob):reasons.append('other_instance_in_crop')
        if other[1] in cat['context_exclude']:reasons.append('context_conflict')
        if other[1] in occluders and _area(core)>0 and _inter(core,ob)>=a['occlusion_max']*_area(core):reasons.append('occluded')
    if f'{image["image_id"]}/{oid}' in config['manual_exclusions']:reasons.append('manual_exclusion')
    return sorted(set(reasons),key=REASONS.index)


def iter_index(gqa_root):
    with gzip.open(Path(gqa_root)/'index'/'objects_by_image.jsonl.gz','rt') as handle:
        for line in handle:yield json.loads(line)


def candidate_population(gqa_root,config):
    """All objects named exactly like a declared category and carrying a needed color, with reasons."""
    needed=defaultdict(set)
    for o,c in cells(config):needed[o].add(c)
    splits=set(config['splits']);out=defaultdict(list)
    for image in iter_index(gqa_root):
        if image['split'] not in splits:continue
        for obj in image['objects']:
            if obj[1] not in needed:continue
            for color in needed[obj[1]]&set(obj[6]):
                out[(obj[1],color)].append({'candidate_id':f'{image["image_id"]}-{obj[0]}','image_id':image['image_id'],
                    'object_id':obj[0],'split':image['split'],'name':obj[1],'category':obj[1],'color':color,
                    'attributes':obj[6],'bbox':obj[2:6],'image_size':[image['width'],image['height']],
                    'rank':rank(config['seed'],image['image_id'],obj[0]),
                    'annotation_reasons':annotation_reasons(obj,image,obj[1],color,config),
                    'manual_note':config['manual_exclusions'].get(f'{image["image_id"]}/{obj[0]}'),
                    'same_category_in_image':sum(o[1] in set(config['categories'][obj[1]]['synonyms'])|{obj[1]} for o in image['objects'])})
    for key in out:out[key].sort(key=lambda r:r['rank'])
    return dict(out)


def population_counts(population,config):
    return {'/'.join(cell):{'population':len(population.get(cell,[])),
            'annotation_pass':sum(not r['annotation_reasons'] for r in population.get(cell,[])),
            'annotation_reasons':dict(Counter(x for r in population.get(cell,[]) for x in r['annotation_reasons']))}
            for cell in cells(config)}


def walk(population,config,gqa_root,status,*,planning=False):
    """Per cell, seeded hash order: annotation failures are recorded; annotation passes need their photo and are
    pixel-checked; stop at needed+spare label passes (or pool_per_cell annotation passes). The examined prefix is
    therefore a deterministic function of config + photos, independent of how many extra photos were fetched.
    ``planning`` lists unfetched photos (counted as tentative passes) instead of failing."""
    from PIL import Image
    need=needed_per_cell(config);examined=[];fetch=[]
    for cell in cells(config):
        target=need[cell]+config['spare_per_cell'];passes=tentative=annotation_ok=0
        for position,row in enumerate(population.get(cell,[])):
            if passes+tentative>=target or annotation_ok>=config['pool_per_cell']:break
            row=dict(row,hash_position=position,fetch=None,pixel=None,label_pass=False)
            if not row['annotation_reasons']:
                annotation_ok+=1;record=status.get(row['image_id'])
                if record is None:
                    if not planning:raise ValueError(f'Photo {row["image_id"]} not fetched; run plan-fetch then natobj_gqa fetch')
                    fetch.append(row['image_id']);tentative+=1;row['fetch']='planned'
                elif record['status']!='ok':row['fetch']='failed'
                else:
                    row['fetch']='ok';row['source']={k:record[k] for k in ('url','bytes','sha256','size','last_modified','file')}
                    with Image.open(Path(gqa_root)/'images'/record['file']) as im:
                        if list(im.size)!=row['image_size']:raise ValueError(f'Size mismatch {row["image_id"]}')
                        row['pixel']=pixel_check(im,row['bbox'],row['color'],config)
                    row['label_pass']=row['pixel']['passed'];passes+=row['label_pass']
            examined.append(row)
    return examined,sorted(set(fetch),key=lambda k:(len(k),k))


# ---------------------------------------------------------------- pixel label check

def hsv_degrees(arr):
    import numpy as np
    maxc=arr.max(-1);minc=arr.min(-1);delta=maxc-minc;safe=np.maximum(delta,1e-12)
    r,g,b=arr[...,0],arr[...,1],arr[...,2]
    hue=np.where(maxc==r,((g-b)/safe)%6,np.where(maxc==g,(b-r)/safe+2,(r-g)/safe+4))*60.
    return np.where(delta>0,hue,0.),np.where(maxc>0,delta/np.maximum(maxc,1e-12),0.),maxc


def hue_fractions(arr,pc):
    import numpy as np
    hue,sat,val=hsv_degrees(arr);chroma=(sat>=pc['s_min'])&(val>=pc['v_min'])
    frac={c:float(np.mean(chroma&np.any([(hue>=lo)&(hue<hi) for lo,hi in bins],axis=0))) for c,bins in sorted(pc['hue_bins'].items())}
    return frac,float(chroma.mean())


def pixel_check(image,bbox,color,config):
    """Central box region: own hue-bin fraction must dominate every other declared hue bin (ratio + margin rule);
    whole annotated box: own hue bin is the plurality chromatic bin (e.g. rejects a yellow bus labelled blue for its
    windshield); displayed crop: the other pair color(s) are capped (e.g. a red umbrella on blue sea). Model independent."""
    import numpy as np
    pc=config['pixel_check'];x,y,w,h=bbox
    central=central_box(bbox,pc['central_frac']);shown=crop_box(bbox,image.size,config['canvas']['context_margin'])
    rgb=image.convert('RGB')
    frac,chromatic=hue_fractions(np.asarray(rgb.crop(central),dtype=np.float64)/255.,pc)
    crop_frac,_=hue_fractions(np.asarray(rgb.crop(shown),dtype=np.float64)/255.,pc)
    box_frac,_=hue_fractions(np.asarray(rgb.crop((x,y,x+w,y+h)),dtype=np.float64)/255.,pc)
    plurality=all(box_frac[color]>=v for k,v in box_frac.items() if k!=color)
    pair_rivals=sorted({c for p in config['color_pairs'] for c in p}-{color})
    rival=max(sorted(set(frac)-{color}),key=lambda c:(frac[c],c));own=frac[color]
    crop_rival=max(crop_frac[c] for c in pair_rivals)
    passed=(own>=pc['min_frac'] and own>=pc['ratio']*frac[rival] and own-frac[rival]>=pc['margin']
            and crop_rival<=pc['crop_rival_max'] and (plurality or not pc['bbox_plurality']))
    luma=np.asarray(rgb.convert('L').crop((x,y,x+w,y+h)),dtype=np.float64)
    return {'passed':bool(passed),'color':color,'own_frac':round(own,4),'rival':rival,'rival_frac':round(frac[rival],4),
            'crop_rival_frac':round(crop_rival,4),'bbox_plurality':bool(plurality),'fractions':{k:round(v,4) for k,v in frac.items()},
            'bbox_fractions':{k:round(v,4) for k,v in box_frac.items()},
            'crop_fractions':{k:round(v,4) for k,v in crop_frac.items()},'chromatic_frac':round(chromatic,4),
            'central_box':central,'mean_luma_bbox':round(float(luma.mean()),2)}


# ---------------------------------------------------------------- selection

def select_quads(checks,config):
    """Seeded hash order within each (category,color) cell; distinct source images inside a quad; reuse caps."""
    passing=defaultdict(list)
    for row in checks:
        if row['label_pass']:passing[(row['category'],row['color'])].append(row)
    crop_uses=Counter();image_uses=Counter();quads=[];deficits=[]
    for stratum in strata(config):
        (x,y),(c0,c1)=stratum['objects'],stratum['colors']
        for k in range(config['quads_per_stratum']):
            picks={};images=set()
            for role,(obj,value,col) in zip(('X_c0','X_c1','Y_c0','Y_c1'),((x,0,c0),(x,1,c1),(y,0,c0),(y,1,c1))):
                for cand in passing[(obj,col)]:
                    if (crop_uses[cand['candidate_id']]<config['max_crop_uses'] and image_uses[cand['image_id']]<config['max_image_uses']
                            and cand['image_id'] not in images):
                        picks[role]=cand;images.add(cand['image_id']);break
                else:deficits.append(f'{stratum["id"]} quad {k}: no unused passing {obj}/{col} crop')
            if len(picks)<4:continue
            for cand in picks.values():crop_uses[cand['candidate_id']]+=1;image_uses[cand['image_id']]+=1
            quads.append({'quad_id':f'{stratum["id"]}-q{k:02d}','stratum':stratum['id'],'objects':[x,y],'colors':[c0,c1],
                          'layout':k%2,'replicate':k,'crops':picks})
    if deficits:
        raise ValueError('Insufficient label-passing crops; raise pool_per_cell, fetch, rebuild:\n'+'\n'.join(deficits))
    return quads


def units(quads):
    """Union-find over source image ids: quads sharing any source image share an independence unit."""
    parent={q['quad_id']:q['quad_id'] for q in quads};owner={}
    def find(a):
        while parent[a]!=a:parent[a]=parent[parent[a]];a=parent[a]
        return a
    for q in quads:
        for cand in q['crops'].values():
            image=cand['image_id']
            if image in owner:
                a,b=find(q['quad_id']),find(owner[image]);parent[max(a,b)]=min(a,b)
            else:owner[image]=q['quad_id']
    roots=sorted({find(q) for q in parent});label={r:f'u{i:03d}' for i,r in enumerate(roots)}
    return {q:label[find(q)] for q in parent}


# ---------------------------------------------------------------- rendering

def geometry(config):
    c=config['canvas'];(w,h),box=c['size'],c['box'];half=w//2;x0,y0=(half-box)//2,(h-box)//2
    return {'size':[w,h],'half_width':half,'box':box,'background':c['background'],'context_margin':c['context_margin'],
            'slot_boxes':[[x0,y0,x0+box,y0+box],[half+x0,y0,half+x0+box,y0+box]],'resample':'LANCZOS',
            'patch_size':16,'merge_size':2,'expected_qwen_grid':[1,h//16,w//16]}


def render_tile(source,crop,geo):
    """Crop [x0,y0,x1,y1] from the source image, resize aspect-preserving into the box, centre on background."""
    from PIL import Image
    box,bg=geo['box'],geo['background']
    with Image.open(source) as im:region=im.convert('RGB').crop(tuple(crop))
    cw,ch=region.size;scale=box/max(cw,ch);nw,nh=max(1,round(cw*scale)),max(1,round(ch*scale))
    tile=Image.new('RGB',(box,box),(bg,)*3);offset=[(box-nw)//2,(box-nh)//2]
    with region.resize((nw,nh),Image.Resampling.LANCZOS) as resized:tile.paste(resized,tuple(offset))
    region.close()
    return tile,{'scale':round(scale,6),'resized':[nw,nh],'offset':offset}


def compose(tiles,geo):
    """tiles[slot] -> full canvas; identical box geometry in both halves."""
    from PIL import Image
    image=Image.new('RGB',tuple(geo['size']),(geo['background'],)*3)
    for slot,tile in enumerate(tiles):image.paste(tile,tuple(geo['slot_boxes'][slot][:2]))
    return image


def gray_blend(image,strength):
    """s=0 luminance only (structure kept, color removed); s=1 identical to the input."""
    from PIL import Image
    if strength>=1:return image.copy()
    with image.convert('L').convert('RGB') as gray:
        return gray.copy() if strength<=0 else Image.blend(gray,image,strength)


def main_source(strength,main_layout=0):return f'm_s{strength:03d}'+('_M1' if main_layout else '')
def donor_source(pose,value):return f'd_{pose}_{value}'


def source_specs(strengths):
    """Scene sources. 'pose' = pixel arrangement relative to the scene's ORIGINAL layout (cross = exact half swap)."""
    sources={}
    for strength,main_layout in product(sorted(set(strengths)|{0,100}),(0,1)):
        sources[main_source(strength,main_layout)]={'type':'main','strength':strength/100,'pose':POSES[main_layout],
            'main_layout':main_layout,'assignment':'a','outline':False}
    for pose,value in product(POSES,('a','b')):
        sources[donor_source(pose,value)]={'type':'donor','strength':1.,'pose':pose,'assignment':value,'outline':False}
    sources['blank']={'type':'blank','strength':0.,'pose':'same','assignment':'a','outline':False}
    return sources


def scene_spec(quad,assignment,geo):
    x,y=quad['objects']
    if quad['layout'] not in (0,1) or assignment not in (0,1):raise ValueError('Binary layout/assignment required')
    return {'objects':[x,y],'colors':list(quad['colors']),'layout':quad['layout'],'assignment':assignment,
            'object_to_slot':{x:quad['layout'],y:1-quad['layout']},'object_values_a':{x:assignment,y:1-assignment},
            'crop_roles':{x:['X_c0','X_c1'],y:['Y_c0','Y_c1']},'canvas':geo,
            'permutation_geometry':{'kind':'swap_equal_horizontal_halves','half_width':geo['half_width'],
                                    'image_size':geo['size'],'patch_size':16,'merge_size':2}}


def expected_answers(spec,target,sources):
    """Object-identity answers, original-slot answers and target slots for every source."""
    value=spec['object_values_a'][target];slot=spec['object_to_slot'][target]
    obj={n:None if s['type']=='blank' or s['strength']==0 else value^(s['assignment']=='b') for n,s in sources.items()}
    flip=lambda s:s['pose']=='cross'
    return (obj,{n:None if v is None else v^flip(sources[n]) for n,v in obj.items()},
            {n:None if s['type']=='blank' else slot^flip(s) for n,s in sources.items()})


def prompt_for(query,choices):return query+'\n'+''.join(f'({chr(65+i)}) {c}\n' for i,c in enumerate(choices))


def crop_record(cand,geo,tile_meta,role):
    obj,col=cand['category'],cand['color']
    return {'role':role,'candidate_id':cand['candidate_id'],'category':obj,'color':col,'gqa_split':cand['split'],
            'image_id':cand['image_id'],'object_id':cand['object_id'],'name':cand['name'],'attributes':cand['attributes'],
            'bbox_xywh':cand['bbox'],'crop_box_xyxy':crop_box(cand['bbox'],cand['image_size'],geo['context_margin']),
            'source_size':cand['image_size'],'source_image':f'sources/{cand["image_id"]}.jpg','source_url':cand['source']['url'],
            'source_sha256':cand['source']['sha256'],'source_bytes':cand['source']['bytes'],
            'source_last_modified':cand['source']['last_modified'],'tile':tile_meta,'hash_position':cand['hash_position'],
            'label_check':{'annotation_reasons':cand['annotation_reasons'],'pixel':cand['pixel'],'passed':cand['label_pass'],
                           'same_category_in_image':cand['same_category_in_image']}}


def render_quad(quad,geo,gqa_root,output,strengths):
    """Files shared by the quad's two assignment scenes: arrangement D_k puts X at value k and Y at 1-k."""
    x,y=quad['objects'];tiles={};meta={};files={}
    for role,cand in quad['crops'].items():
        crop=crop_box(cand['bbox'],cand['image_size'],geo['context_margin'])
        tiles[role],tile_meta=render_tile(Path(gqa_root)/'images'/cand['source']['file'],crop,geo)
        meta[role]=crop_record(cand,geo,tile_meta,role)
    for k in (0,1):
        slots=[None,None];slots[quad['layout']]=tiles[f'X_c{k}'];slots[1-quad['layout']]=tiles[f'Y_c{1-k}']
        with compose(slots,geo) as same:
            renders={f'D{k}-same':same.copy(),f'D{k}-cross':swap_image_halves(same)}
            for s in sorted(set(strengths)|{0,100}):
                with gray_blend(same,s/100) as m0:
                    renders[f'D{k}-m_s{s:03d}']=m0.copy();renders[f'D{k}-m_s{s:03d}_M1']=swap_image_halves(m0)
        for name,image in renders.items():
            rel=f'images/{quad["quad_id"]}-{name}.png';image.save(Path(output)/rel);image.close();files[name]=rel
    for tile in tiles.values():tile.close()
    return files,meta


def file_meta(output,rel,width,height):
    path=Path(output)/rel
    return {'width':width,'height':height,'bytes':path.stat().st_size,'sha256':sha256_file(path)}


def scene_rows(quad,unit,files,crop_meta,geo,strengths,output,provenance,metas):
    sources=source_specs(strengths);rows=[];x,y=quad['objects'];c0,c1=quad['colors']
    for assignment in (0,1):
        spec=scene_spec(quad,assignment,geo);scene_id=f'{quad["quad_id"]}-a{assignment}'
        paths={}
        for name,s in sources.items():
            if s['type']=='blank':paths[name]='images/blank.png'
            elif s['type']=='main':
                paths[name]=files[f'D{assignment}-m_s{round(s["strength"]*100):03d}'+('_M1' if s['main_layout'] else '')]
            else:paths[name]=files[f'D{assignment^(s["assignment"]=="b")}-{s["pose"]}']
        for path in set(paths.values()):
            if path not in metas:metas[path]=file_meta(output,path,*geo['size'])
        for target_index,target in enumerate((x,y)):
            query=f'What color is the {target}?';choices=[c0,c1]
            obj,slot_values,slots=expected_answers(spec,target,sources);value=spec['object_values_a'][target]
            rows.append({'schema':SCHEMA,'id':f'{scene_id}-{target.replace(" ","_")}','scene_id':scene_id,'quad_id':quad['quad_id'],
                'unit_id':unit,'stratum':quad['stratum'],'task':'natobj_color','attribute':'color','category_pair':[x,y],
                'color_pair':[c0,c1],'layout':quad['layout'],'assignment':assignment,'replicate':quad['replicate'],
                'target_object':target,'target_index':target_index,'objects':[x,y],'target_main_slot':spec['object_to_slot'][target],
                'scene':spec,'query':query,'prompt':prompt_for(query,choices),'choices':choices,'option_labels':['(A)','(B)'],
                'answer_index':value,'answer_label':f'({chr(65+value)})','source_answer_indices':obj,
                'source_slot_answer_indices':slot_values,'source_object_slots':slots,'source_specs':sources,'source_images':paths,
                'source_image_metadata':{n:dict(metas[p],source_field='image_1') for n,p in paths.items()},
                'natural':{'construction':CONSTRUCTION,'note':CONSTRUCTION_NOTE,'crops':crop_meta,
                           'target_crop_roles':{'a':spec['crop_roles'][target][value],'b':spec['crop_roles'][target][1-value]},
                           'source_image_ids':sorted({c['image_id'] for c in crop_meta.values()},key=lambda k:(len(k),k))},
                'source':provenance})
    return rows


# ---------------------------------------------------------------- native panel and contact sheet

def render_whole(source,geo,dest):
    from PIL import Image
    w,h=geo['size']
    with Image.open(source) as im:
        image=im.convert('RGB');scale=min(w/image.width,h/image.height)
        nw,nh=max(1,round(image.width*scale)),max(1,round(image.height*scale))
        canvas=Image.new('RGB',(w,h),(geo['background'],)*3)
        with image.resize((nw,nh),Image.Resampling.LANCZOS) as resized:canvas.paste(resized,((w-nw)//2,(h-nh)//2))
        image.close()
    canvas.save(dest);canvas.close()
    return {'scale':round(scale,6),'resized':[nw,nh],'offset':[(w-nw)//2,(h-nh)//2]}


def native_panel(quads,unit_of,geo,gqa_root,output,provenance):
    """Unedited source images (content unedited; resized + letterboxed) with exactly one object of the category."""
    rows=[];(Path(output)/'images'/'native').mkdir(exist_ok=True)
    for quad in quads:
        for role,cand in quad['crops'].items():
            if cand['same_category_in_image']!=1:continue
            rel=f'images/native/{cand["candidate_id"]}.png'
            fit=render_whole(Path(gqa_root)/'images'/cand['source']['file'],geo,Path(output)/rel)
            choices=list(quad['colors']);value=choices.index(cand['color']);query=f'What color is the {cand["category"]}?'
            rows.append({'schema':SCHEMA,'id':f'native-{cand["candidate_id"]}','panel':'native_unedited_whole',
                'construction':'unedited_whole_image_resized_letterboxed','quad_id':quad['quad_id'],'unit_id':unit_of[quad['quad_id']],
                'crop_role':role,'task':'natobj_color_native','attribute':'color','target_object':cand['category'],
                'query':query,'prompt':prompt_for(query,choices),'choices':choices,'option_labels':['(A)','(B)'],
                'answer_index':value,'answer_label':f'({chr(65+value)})','source_answer_indices':{'native_whole':value},
                'source_specs':{'native_whole':{'type':'native_whole','strength':1.,'pose':'same','assignment':'a','outline':False}},
                'source_images':{'native_whole':rel},'source_image_metadata':{'native_whole':dict(file_meta(output,rel,*geo['size']),source_field='image_1')},
                'natural':{'image_id':cand['image_id'],'object_id':cand['object_id'],'bbox_xywh':cand['bbox'],
                           'source_image':f'sources/{cand["image_id"]}.jpg','source_sha256':cand['source']['sha256'],'fit':fit,
                           'same_category_in_image':1},'source':provenance})
    return rows


def contact_sheets(rows,output,per_page=8):
    """Review pages: four crop tiles with labels, then D0 same / gray main / s008 main / D0 cross per quad."""
    from PIL import Image,ImageDraw,ImageFont
    font=ImageFont.load_default(size=12);by_quad={}
    for r in rows:
        if r['assignment']==0 and r['target_index']==0:by_quad[r['quad_id']]=r
    quads=sorted(by_quad);pages=[];tile=128;thumb=(192,128);pad=6;label=30
    row_h=max(tile,thumb[1])+label+pad;width=4*(tile+pad)+4*(thumb[0]+pad)+pad
    for start in range(0,len(quads),per_page):
        chunk=quads[start:start+per_page];sheet=Image.new('RGB',(width,len(chunk)*row_h+pad),(255,255,255))
        draw=ImageDraw.Draw(sheet)
        for i,qid in enumerate(chunk):
            r=by_quad[qid];top=pad+i*row_h;x=pad;crops=r['natural']['crops']
            for role in ('X_c0','X_c1','Y_c0','Y_c1'):
                c=crops[role]
                with render_tile(Path(output)/c['source_image'],c['crop_box_xyxy'],r['scene']['canvas'])[0] as t:
                    sheet.paste(t.resize((tile,tile)),(x,top+label))
                draw.text((x,top),f'{role} {c["category"]}/{c["color"]}\nimg {c["image_id"]} f={c["label_check"]["pixel"]["own_frac"]:.2f}',fill=(0,0,0),font=font)
                x+=tile+pad
            for name,caption in (('d_same_a','D same a (clear)'),('m_s000','main gray s0'),('m_s008','main s.08'),('d_cross_a','D cross a')):
                with Image.open(Path(output)/r['source_images'][name]) as im:sheet.paste(im.resize(thumb),(x,top+label))
                draw.text((x,top),f'{caption}\n{qid} L{r["layout"]} {r["unit_id"]}' if name=='d_same_a' else caption,fill=(0,0,0),font=font)
                x+=thumb[0]+pad
        path=Path(output)/f'contact_sheet_p{len(pages)+1:02d}.png';sheet.save(path);sheet.close();pages.append(path.name)
    return pages


# ---------------------------------------------------------------- build

GENERATED=('manifest.jsonl','label_checks.jsonl','native_panel.jsonl','dataset.json','config.json')


def clear_generated(output):
    """--overwrite removes only builder outputs (images/*.png, sources/*.jpg, sheets, JSON); anything else aborts."""
    for sub,suffix in (('images','.png'),('sources','.jpg')):
        root=output/sub
        if not root.exists():continue
        for path in sorted(root.rglob('*'),key=lambda q:len(q.parts),reverse=True):
            if path.is_file() and path.suffix==suffix and not path.is_symlink():path.unlink()
            elif path.is_dir() and not path.is_symlink():path.rmdir()
            else:raise ValueError(f'Unexpected file in generated tree: {path}')
    for path in [*output.glob('contact_sheet_p*.png'),*(output/n for n in GENERATED)]:
        if path.is_file():path.unlink()


def build(gqa_root,output,config=None,*,overwrite=False,log_name='fetch_log.jsonl'):
    from cloud_prep.measurement_v7.natobj_gqa import fetch_status
    config=config or read_config();gqa_root=_guard_path(Path(gqa_root));output=_guard_path(Path(output),write=True)
    if (output/'manifest.jsonl').exists() and not overwrite:raise FileExistsError('Use a new output directory (or overwrite)')
    if overwrite:clear_generated(output)
    output.mkdir(parents=True,exist_ok=True);(output/'images').mkdir(exist_ok=True);(output/'sources').mkdir(exist_ok=True)
    geo=geometry(config);strengths=config['strengths']
    population=candidate_population(gqa_root,config)
    checks,_=walk(population,config,gqa_root,fetch_status(gqa_root,log_name))
    quads=select_quads(checks,config);unit_of=units(quads)
    selected=defaultdict(list)
    for q in quads:
        for role,cand in q['crops'].items():selected[cand['candidate_id']].append(f'{q["quad_id"]}:{role}')
    index_summary=json.loads((gqa_root/'index'/'index_summary.json').read_text())
    provenance={'dataset':'GQA scene graphs (Hudson & Manning 2019) on Visual Genome images; controlled recomposition',
        'license_note':'GQA scene graphs and Visual Genome images are distributed under CC BY 4.0 by their authors',
        'scene_graphs_url':'https://downloads.cs.stanford.edu/nlp/data/gqa/sceneGraphs.zip','image_url_templates':
        ['https://cs.stanford.edu/people/rak248/VG_100K/{id}.jpg','https://cs.stanford.edu/people/rak248/VG_100K_2/{id}.jpg'],
        'scene_graph_files':index_summary['scene_graph_files'],'index_sha256':index_summary['index_sha256'],
        'revision':SCHEMA,'split':'exploratory','seed':config['seed'],'config_sha256':config_sha(config),
        'generator_sha256':sha256_file(Path(__file__)),'construction':CONSTRUCTION}
    from PIL import Image
    Image.new('RGB',tuple(geo['size']),(geo['background'],)*3).save(output/'images'/'blank.png')
    metas={};rows=[]
    for q in quads:
        for cand in q['crops'].values():
            dest=output/'sources'/f'{cand["image_id"]}.jpg'
            if not dest.exists():shutil.copyfile(gqa_root/'images'/cand['source']['file'],dest)
            if sha256_file(dest)!=cand['source']['sha256']:raise ValueError('Source image bytes changed')
        files,crop_meta=render_quad(q,geo,gqa_root,output,strengths)
        rows+=scene_rows(q,unit_of[q['quad_id']],files,crop_meta,geo,strengths,output,provenance,metas)
    (output/'manifest.jsonl').write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in rows))
    for row in checks:row['selected_as']=selected.get(row['candidate_id'],[])
    (output/'label_checks.jsonl').write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in checks))
    native=native_panel(quads,unit_of,geo,gqa_root,output,provenance) if config['native_panel']['enabled'] else []
    (output/'native_panel.jsonl').write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in native))
    (output/'config.json').write_text(json.dumps(config,indent=2,sort_keys=True)+'\n')
    pages=contact_sheets(rows,output)
    result=validate(output/'manifest.jsonl',pixels=True)
    reason_counts=Counter(x for r in checks for x in r['annotation_reasons'])
    examined=Counter('annotation_fail' if r['annotation_reasons'] else 'fetch_failed' if r['fetch']!='ok' else
                     'pixel_fail' if not r['label_pass'] else 'label_pass' for r in checks)
    by_cell=defaultdict(Counter)
    for r in checks:by_cell['/'.join((r['category'],r['color']))]['examined']+=1;by_cell['/'.join((r['category'],r['color']))][
        'annotation_fail' if r['annotation_reasons'] else 'fetch_failed' if r['fetch']!='ok' else 'pixel_fail' if not r['label_pass'] else 'label_pass']+=1
    result.update({'label_checks':{'file':'label_checks.jsonl','examined':len(checks),'outcomes':dict(examined),
        'annotation_reasons_examined':dict(reason_counts),'by_cell':{k:dict(v) for k,v in sorted(by_cell.items())},
        'needed_per_cell':{'/'.join(k):v for k,v in sorted(needed_per_cell(config).items())},'spare_per_cell':config['spare_per_cell'],
        'population_by_cell':population_counts(population,config),
        'rule':'declared before any model output; seeded hash order; first label-passing crops are used'},
        'native_panel':{'rows':len(native),'file':'native_panel.jsonl','eligibility':'selected crop whose source image has exactly one object named in the category synonym set'},
        'visual_review':{**{k:v for k,v in config['visual_review'].items() if k!='reviewed_selected'},
            'manual_exclusions_declared':len(config['manual_exclusions']),
            'manual_exclusions_in_examined_prefix':sum('manual_exclusion' in r['annotation_reasons'] for r in checks),
            'selected_reviewed':sum(c in set(config['visual_review']['reviewed_selected']) for c in selected),
            'selected_unreviewed':sorted(c for c in selected if c not in set(config['visual_review']['reviewed_selected']))},
        'contact_sheets':pages,'config':'config.json','config_sha256':config_sha(config),
        'crop_reuse':{'max_uses':max(Counter(c for q in quads for c in (x['candidate_id'] for x in q['crops'].values())).values()),
                      'source_images':len({c['image_id'] for q in quads for c in q['crops'].values()}),'crops':sum(len(q['crops']) for q in quads)}})
    (output/'dataset.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    return result


# ---------------------------------------------------------------- loader / validator

def _contained(root,rel):
    rel=Path(rel)
    if rel.is_absolute() or '..' in rel.parts:raise ValueError('Images must use contained portable paths')
    path=_guard_path(root/rel)
    if not path.is_relative_to(root) or not path.is_file():raise ValueError(f'Missing or escaping file: {rel}')
    return path


def load_manifest(manifest,image_root=None,*,check_hashes=True):
    """Validate design/answers/provenance and return rows with absolute source paths (v6 content-row compatible)."""
    manifest=_guard_path(Path(manifest));root=_guard_path(Path(image_root)) if image_root else manifest.parent
    rows=[json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    if not rows or len({r['id'] for r in rows})!=len(rows):raise ValueError('Empty/duplicate manifest')
    checked={};result=[];quad_units={};image_units={}
    for row in rows:
        strengths=sorted({round(s['strength']*100) for s in row['source_specs'].values() if s['type']=='main'})
        sources=source_specs(strengths)
        if row['schema']!=SCHEMA or row['source_specs']!=sources or set(row['source_images'])!=set(sources):
            raise ValueError('Unknown schema or source design')
        spec=row['scene'];x,y=row['objects']
        if (spec['objects']!=[x,y] or row['category_pair']!=[x,y] or x==y or spec['layout']!=row['layout']
                or spec['assignment']!=row['assignment'] or spec['object_to_slot']!={x:row['layout'],y:1-row['layout']}
                or spec['object_values_a']!={x:row['assignment'],y:1-row['assignment']} or row['target_object']!=[x,y][row['target_index']]
                or row['target_main_slot']!=spec['object_to_slot'][row['target_object']]):
            raise ValueError('Scene/object disagreement')
        canvas=spec['canvas']
        if canvas['size']!=[canvas['half_width']*2,canvas['size'][1]] or canvas['half_width']%32 or canvas['size'][1]%32:
            raise ValueError('Canvas halves must align to the merged grid')
        if spec['permutation_geometry']['half_width']!=canvas['half_width'] or canvas['slot_boxes'][1]!=[
                canvas['slot_boxes'][0][0]+canvas['half_width'],canvas['slot_boxes'][0][1],canvas['slot_boxes'][0][2]+canvas['half_width'],canvas['slot_boxes'][0][3]]:
            raise ValueError('Slot boxes must be identical in both halves')
        obj,slot_values,slots=expected_answers(spec,row['target_object'],sources);value=spec['object_values_a'][row['target_object']]
        if (row['source_answer_indices']!=obj or row['source_slot_answer_indices']!=slot_values or row['source_object_slots']!=slots):
            raise ValueError('Object-versus-spatial source answers disagree')
        if (row['answer_index']!=value or row['answer_label']!=row['option_labels'][value] or row['choices']!=row['color_pair']
                or spec['colors']!=row['color_pair'] or row['query']!=f'What color is the {row["target_object"]}?'
                or row['prompt']!=prompt_for(row['query'],row['choices'])):
            raise ValueError('Content answer/choice mismatch')
        nat=row['natural'];crops=nat['crops']
        if nat['construction']!=CONSTRUCTION or set(crops)!={'X_c0','X_c1','Y_c0','Y_c1'}:raise ValueError('Natural provenance missing')
        for role,c in crops.items():
            if c['category']!=(x if role[0]=='X' else y) or c['color']!=row['color_pair'][int(role[-1])] or not c['label_check']['passed']:
                raise ValueError('Crop role/label disagreement')
        if len({c['image_id'] for c in crops.values()})!=4:raise ValueError('A quad needs four distinct source images')
        roles=spec['crop_roles'][row['target_object']]
        if nat['target_crop_roles']!={'a':roles[value],'b':roles[1-value]}:raise ValueError('Target crop roles disagree')
        if quad_units.setdefault(row['quad_id'],row['unit_id'])!=row['unit_id']:raise ValueError('Quad split across units')
        for c in crops.values():
            if image_units.setdefault(c['image_id'],row['unit_id'])!=row['unit_id']:raise ValueError('Shared source image across units')
        loaded=deepcopy(row)
        for name,rel in row['source_images'].items():
            path=_contained(root,rel);meta=row['source_image_metadata'][name]
            if [meta['width'],meta['height']]!=canvas['size']:raise ValueError('Fixed source geometry changed')
            if check_hashes:
                if path not in checked:checked[path]=(path.stat().st_size,sha256_file(path))
                if checked[path]!=(meta['bytes'],meta['sha256']):raise ValueError(f'Image bytes/SHA changed: {rel}')
            loaded['source_images'][name]=str(path)
        if check_hashes:
            for c in crops.values():
                path=_contained(root,c['source_image'])
                if path not in checked:checked[path]=(path.stat().st_size,sha256_file(path))
                if checked[path]!=(c['source_bytes'],c['source_sha256']):raise ValueError('Source photo bytes/SHA changed')
        result.append(loaded)
    return result


def model_row(row,source=None):
    source=source or main_source(100)
    return {**row,'images':[row['source_images'][source]],'image_metadata':[row['source_image_metadata'][source]]}


def verify_pixels(rows,tolerance=0):
    """Pixel-exact design checks per scene, plus crop traceability against packaged source photos."""
    import numpy as np
    from PIL import Image
    def load(path):
        with Image.open(path) as im:return np.asarray(im.convert('RGB'),dtype=np.int16)
    swap=lambda a:np.concatenate([a[:,a.shape[1]//2:],a[:,:a.shape[1]//2]],axis=1)
    counts=Counter();seen=set();traced=set()
    for row in rows:
        if row['scene_id'] in seen:continue
        seen.add(row['scene_id']);p=row['source_images'];img={n:load(v) for n,v in p.items()}
        canvas=row['scene']['canvas'];blank=img['blank'];inside=np.zeros(blank.shape[:2],bool)
        for x0,y0,x1,y1 in canvas['slot_boxes']:inside[y0:y1,x0:x1]=True
        if blank.shape!=(canvas['size'][1],canvas['size'][0],3) or (blank!=canvas['background']).any():raise ValueError('Blank is not uniform')
        for name,a in img.items():
            if (a[~inside]!=canvas['background']).any():raise ValueError(f'Background outside boxes changed: {name}')
        for value in ('a','b'):
            if not np.array_equal(img[f'd_cross_{value}'],swap(img[f'd_same_{value}'])):raise ValueError('Cross donor is not an exact half swap')
        for name,s in row['source_specs'].items():
            if s['type']=='main' and s['main_layout']==1:
                if not np.array_equal(img[name],swap(img[name[:-3]])):raise ValueError('M1 main is not an exact half swap')
        top=main_source(100)
        if not np.array_equal(img[top],img['d_same_a']):raise ValueError('Strength-1 main differs from d_same_a')
        gray=img[main_source(0)]
        if not (np.array_equal(gray[...,0],gray[...,1]) and np.array_equal(gray[...,1],gray[...,2])):raise ValueError('Gray main has chroma')
        with Image.open(p['d_same_a']) as clear:
            for name,s in row['source_specs'].items():
                if s['type']=='main' and s['main_layout']==0:
                    with gray_blend(clear.convert('RGB'),s['strength']) as expected:
                        if not np.array_equal(img[name],np.asarray(expected,dtype=np.int16)):raise ValueError(f'{name} is not the declared gray blend of d_same_a')
        counts['scenes']+=1
        # Crop traceability: re-render each tile from its source photo and compare inside its slot box.
        spec=row['scene']
        for k,path in (('a',p['d_same_a']),('b',p['d_same_b'])):
            for obj in spec['objects']:
                v=spec['object_values_a'][obj]^(k=='b');role=spec['crop_roles'][obj][v];crop=row['natural']['crops'][role]
                key=(path,role)
                if key in traced:continue
                traced.add(key);x0,y0,x1,y1=canvas['slot_boxes'][spec['object_to_slot'][obj]]
                root=Path(path).parents[1];expected_crop=crop_box(crop['bbox_xywh'],crop['source_size'],canvas['context_margin'])
                if crop['crop_box_xyxy']!=expected_crop:raise ValueError('Crop box does not follow bbox + margin rule')
                tile,_=render_tile(root/crop['source_image'],crop['crop_box_xyxy'],canvas)
                diff=np.abs(np.asarray(tile,dtype=np.int16)-img[f'd_same_{k}'][y0:y1,x0:x1]).max();tile.close()
                if diff>tolerance:raise ValueError(f'Tile not traceable to source crop: {role} max diff {diff}')
                counts['traced_tiles']+=1
    return dict(counts)


def validate(manifest,image_root=None,*,pixels=False):
    rows=load_manifest(manifest,image_root)
    images={p for r in rows for p in r['source_images'].values()}
    quads={r['quad_id'] for r in rows};units_={r['unit_id'] for r in rows}
    crops={c['candidate_id'] for r in rows for c in r['natural']['crops'].values()}
    photos={c['image_id'] for r in rows for c in r['natural']['crops'].values()}
    out={'schema':SCHEMA,'manifest':Path(manifest).name,'manifest_sha256':sha256_file(manifest),'items':len(rows),
        'scenes':len({r['scene_id'] for r in rows}),'quads':len(quads),'independent_units':len(units_),
        'unit_field':'unit_id','strata':sorted({r['stratum'] for r in rows}),'crops':len(crops),'source_photos':len(photos),
        'unique_image_paths':len(images),'image_bytes':sum(Path(p).stat().st_size for p in images),
        'source_count_per_scene':len(rows[0]['source_specs']),'status':'passed','model_results':False,'seed':rows[0]['source']['seed'],
        'construction':CONSTRUCTION,'semantics':('Controlled recomposition of real crops. Independent unit = unit_id (union-find over '
            'source photos). Assignment 0/1 counterbalance the same four crops; target, mapping, format, main strength, main '
            'layout, donor pose and token permutation are within-unit repeats. Cross donors and M1 mains are exact half swaps.')}
    if pixels:out['pixel_checks']=verify_pixels(rows)
    return out


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('command',choices=('plan-fetch','build','validate'))
    p.add_argument('--gqa-root',type=Path,default=(DATA_ROOT / 'datasets/gqa-official'))
    p.add_argument('--config',type=Path,default=CONFIG);p.add_argument('--set',action='append',default=[],metavar='KEY=JSON')
    p.add_argument('--output',type=Path);p.add_argument('--overwrite',action='store_true')
    p.add_argument('--manifest',type=Path);p.add_argument('--image-root',type=Path);p.add_argument('--pixels',action='store_true')
    a=p.parse_args(argv)
    if a.command=='validate':
        if a.manifest is None:p.error('validate requires --manifest')
        result=validate(a.manifest,a.image_root,pixels=a.pixels)
    else:
        if a.output is None:p.error(f'{a.command} requires --output')
        config=read_config(a.config,a.set)
        if a.command=='build':result=build(a.gqa_root,a.output,config,overwrite=a.overwrite)
        else:
            from cloud_prep.measurement_v7.natobj_gqa import fetch_status
            root=_guard_path(a.gqa_root);population=candidate_population(root,config)
            examined,ids=walk(population,config,root,fetch_status(root),planning=True)
            out=_guard_path(a.output,write=True);out.parent.mkdir(parents=True,exist_ok=True)
            passes=Counter('/'.join((r['category'],r['color'])) for r in examined if r['label_pass'])
            result={'schema':SCHEMA,'config_sha256':config_sha(config),'image_ids':ids,'label_passes_so_far':dict(passes),
                    'needed_per_cell':{'/'.join(k):v for k,v in needed_per_cell(config).items()},
                    'population_counts':population_counts(population,config)}
            out.write_text(json.dumps(result,indent=1,sort_keys=True)+'\n')
            result={'fetch_plan':str(out),'photos_to_fetch':len(ids),'label_passes_so_far':dict(passes)}
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
