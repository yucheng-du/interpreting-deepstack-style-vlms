"""Full-scene GQA/VG color-region counterfactuals; Pillow/NumPy CPU only.

No instance segmentation is asserted: masks select annotated-color pixels inside
an accepted object's annotation bbox. The photograph is retained, fit into a
384-square canvas, and separately mirrored. Selection never reads model output.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path
import os as _path_os
DATA_ROOT = Path(_path_os.environ.get("DS_DATA_ROOT", "work")).expanduser().resolve()
import shutil

from cloud_prep.portable_data import _guard_path, sha256_file

ROW_SCHEMA = 'natural-scene-data-v10'
FAMILY = 'natural_scene'
SIZE = (384, 384)
DEFAULT_ROOT = (DATA_ROOT / 'datasets/natural-scene-v10')
DEFAULT_SOURCE = (DATA_ROOT / 'datasets/natural-object-v7/native_panel.jsonl')
DEFAULT_INDEX = (DATA_ROOT / 'datasets/gqa-official/index/objects_by_image.jsonl.gz')
SEED = 2026101403
CATEGORIES = ('bag','bus','car','chair','pillow','train','truck','umbrella')
COLORS = {'red','blue','green','yellow','orange','pink','purple','white','black','gray','grey','brown','silver','gold','tan','beige','maroon','navy','teal','turquoise','cream','khaki'}
# Concrete, referable categories only; exclude sky, lettering, hair and part labels.
CONTROL_NAMES = {'bag','backpack','bus','car','chair','pillow','train','truck','umbrella','shirt','jacket','coat','jeans','shorts','cup','cooler','van','taxi','fire hydrant','sofa','carpet','rug','bedspread'}
# Review exclusions are image-only decisions, recorded with the final selection.
VISUAL_EXCLUSIONS = {
 '285841':'two bicycle pannier bags make unqualified target ambiguous',
 '767':'target car chromatic region too small/dark for reliable full-image label',
 '2364384':'target bbox contains adjacent same-category beach chairs',
 '2344556':'multiple chairs without a stable unqualified referent',
 '2363090':'multiple adjacent same-color chairs without stable unqualified reference',
 '2327305':'multiple cafe chairs make unqualified reference ambiguous',
 '1592028':'multiple party chairs make unqualified reference ambiguous',
 '1591846':'several different chairs require additional reference disambiguation',
 '2334847':'chair color obscured by foreground furniture',
 '2393785':'truck is too occluded/distant to identify reliably in full photograph',
 '2374714':'connected red color region leaks onto nearby person clothing',
 '2387274':'umbrella is mostly white with a small pale-blue sector',
}
QUERY_DESCRIPTIONS = {
 '150294':"bag hanging from the woman's hand",
 '1592325':'truck behind the police officers',
 '2335923':'chair the man is sitting on',
 '2410954':'pillow on the armchair',
 '1592362':"pillow next to the woman's arm",
 '1592000':'pillow on top of the sofa back',
 '2382462':'upper cargo section of the truck',
 '2348088':'caboose at the end of the train',
}
COLOR_REGION_POLYGONS = {
 '2410323':[(0,160),(82,105),(174,93),(244,107),(321,143),(279,149),(252,171),(209,185),(161,205),(115,222),(56,237),(0,254)],
 '2324015':[(54,146),(77,114),(133,92),(211,87),(244,105),(253,118),(199,125),(147,132),(97,141)],
}
CONTROL_EXCLUSIONS = {
 '2380885':'multiple street cars make the unqualified control query ambiguous',
 '2342009':'shirt color/referent insufficiently clear in full photograph',
}


def _dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True)+'\n')


def _hash_order(value):
    return hashlib.sha256(f'{SEED}:{value}'.encode()).hexdigest()


def fit_image(image):
    from PIL import Image
    image = image.convert('RGB')
    scale = min(SIZE[0]/image.width, SIZE[1]/image.height)
    target = (round(image.width*scale), round(image.height*scale))
    offset = ((SIZE[0]-target[0])//2, (SIZE[1]-target[1])//2)
    out = Image.new('RGB', SIZE, (127,127,127))
    out.paste(image.resize(target, Image.Resampling.LANCZOS), offset)
    return out, {'original_size':list(image.size),'resized':list(target),'offset':list(offset),
                 'scale_xy':[target[0]/image.width,target[1]/image.height]}


def transform_bbox(box, fit):
    x,y,w,h = box
    sx,sy = fit['scale_xy']; ox,oy = fit['offset']
    return [max(0,round(x*sx+ox)), max(0,round(y*sy+oy)),
            min(SIZE[0],round((x+w)*sx+ox)), min(SIZE[1],round((y+h)*sy+oy))]


def mirror_bbox(box):
    x0,y0,x1,y1 = box
    return [SIZE[0]-x1,y0,SIZE[0]-x0,y1]


def color_mask(image, bbox, color, clip_polygon=None):
    """Nominal bbox-constrained chromatic pixels, not a segmentation model."""
    import numpy as np
    from PIL import Image
    hsv = np.asarray(image.convert('HSV'))
    h,s,v = hsv[:,:,0],hsv[:,:,1],hsv[:,:,2]
    hue = ((h<=18)|(h>=237)) if color == 'red' else ((h>=135)&(h<=185))
    if color not in ('red','blue'): raise ValueError('Only red/blue masks')
    eligible = hue & (s>=65) & (v>=35)
    roi = np.zeros(eligible.shape, dtype=bool)
    x0,y0,x1,y1 = bbox; roi[y0:y1,x0:x1] = True
    # Keep the largest 8-connected color component. This reduces leakage into
    # separately colored objects/skin/background inside the annotation box.
    # It is still a heuristic color-region mask, never an instance mask.
    from collections import deque
    remaining=eligible & roi
    if clip_polygon:
        from PIL import ImageDraw
        clip=Image.new('L',image.size,0);ImageDraw.Draw(clip).polygon(clip_polygon,fill=255)
        remaining &= np.asarray(clip)>0
    seen=np.zeros(remaining.shape,dtype=bool); largest=[]
    for yy,xx in zip(*np.nonzero(remaining)):
        if seen[yy,xx]:continue
        queue=deque([(int(yy),int(xx))]);seen[yy,xx]=True;component=[]
        while queue:
            cy,cx=queue.popleft();component.append((cy,cx))
            for dy in (-1,0,1):
                for dx in (-1,0,1):
                    ny,nx=cy+dy,cx+dx
                    if 0<=ny<remaining.shape[0] and 0<=nx<remaining.shape[1] and remaining[ny,nx] and not seen[ny,nx]:
                        seen[ny,nx]=True;queue.append((ny,nx))
        if len(component)>len(largest):largest=component
    kept=np.zeros(remaining.shape,dtype='uint8')
    for yy,xx in largest:kept[yy,xx]=255
    return Image.fromarray(kept)


def recolor(image, mask, color):
    """Retain per-pixel HSV saturation/value; change hue only inside mask."""
    import numpy as np
    from PIL import Image
    array = np.asarray(image.convert('HSV')).copy()
    selected = np.asarray(mask)>0
    array[:,:,0][selected] = 0 if color == 'red' else 170
    shifted = Image.fromarray(array, 'HSV').convert('RGB')
    # Composite preserves exact RGB outside the mask (avoids round-trip drift).
    return Image.composite(shifted, image, mask)


def _disjoint(a,b):
    return min(a[2],b[2])<=max(a[0],b[0]) or min(a[3],b[3])<=max(a[1],b[1])


def _candidate(source, graph, source_root):
    from PIL import Image
    image_path = source_root / source['natural']['source_image']
    with Image.open(image_path) as raw: image, fit = fit_image(raw)
    box = transform_bbox(source['natural']['bbox_xywh'], fit)
    color = source['choices'][source['answer_index']]
    mask = color_mask(image,box,color,COLOR_REGION_POLYGONS.get(source['natural']['image_id']))
    count = sum(mask.histogram()[1:]); area=(box[2]-box[0])*(box[3]-box[1])
    # Image-only adequacy checks; no native accuracy or intervention-outcome gate.
    reasons=[]
    if min(box[2]-box[0],box[3]-box[1])<24: reasons.append('object_below_24px_side')
    if count<160 or count/max(area,1)<.12: reasons.append('too_little_annotated_color')
    if source['natural']['image_id'] in VISUAL_EXCLUSIONS: reasons.append(VISUAL_EXCLUSIONS[source['natural']['image_id']])
    counts=Counter(o[1] for o in graph['objects']); controls=[]
    for obj in graph['objects']:
        oid,name,x,y,w,h,attrs=obj
        colors=set(attrs)&COLORS
        if name not in CONTROL_NAMES or name==source['target_object'] or counts[name]!=1 or len(colors)!=1 or not colors&{'red','blue'}: continue
        control_box=transform_bbox([x,y,w,h],fit)
        if not _disjoint(box,control_box) or min(control_box[2]-control_box[0],control_box[3]-control_box[1])<20: continue
        control_color=next(iter(colors)); cmask=color_mask(image,control_box,control_color)
        n=sum(cmask.histogram()[1:]); control_area=(control_box[2]-control_box[0])*(control_box[3]-control_box[1])
        if n<100 or n/max(control_area,1)<.12: continue
        controls.append({'object_id':oid,'name':name,'color':control_color,'bbox_xyxy':control_box,
                         'bbox_xywh_original':[x,y,w,h], 'color_region_pixels':n})
    controls.sort(key=lambda o:(-o['color_region_pixels'],_hash_order(o['object_id'])))
    if source['natural']['image_id'] in CONTROL_EXCLUSIONS: controls=[]
    return {'source':source,'graph':{'image_id':graph['image_id'],'split':graph['split']},'image_path':image_path,
            'fit':fit,'bbox':box,'color':color,'mask_pixels':count,'bbox_pixels':area,'reasons':reasons,
            'control':controls[0] if controls else None}


def visual_signature(row):
    value={'sources':{k:v['sha256'] for k,v in row['source_image_metadata'].items()},
           'roi':row['donor_roi_bbox_xyxy'],'mask':row['mask_sha256'],'changed':row['changed_object_id']}
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


def build(output=DEFAULT_ROOT, source_manifest=DEFAULT_SOURCE, index=DEFAULT_INDEX, per_category=4):
    from PIL import Image, ImageDraw, ImageOps
    output=_guard_path(Path(output),write=True); source_manifest=_guard_path(Path(source_manifest)); index=_guard_path(Path(index))
    if output.exists(): raise FileExistsError('Use a new versioned output; never overwrite a built panel')
    source=[json.loads(s) for s in source_manifest.read_text().splitlines() if s]
    ids={r['natural']['image_id'] for r in source}; graphs={}
    with gzip.open(index,'rt') as f:
        for line in f:
            obj=json.loads(line)
            if obj['image_id'] in ids:graphs[obj['image_id']]=obj
    candidates=[_candidate(r,graphs[r['natural']['image_id']],source_manifest.parent) for r in source]
    selected=[]
    for category in CATEGORIES:
        eligible=[c for c in candidates if c['source']['target_object']==category and not c['reasons']]
        eligible.sort(key=lambda c:(c['control'] is None,_hash_order(c['source']['id'])))
        if len(eligible)<per_category: raise ValueError(f'{category}: only {len(eligible)} eligible')
        selected.extend(eligible[:per_category])
    output.mkdir(parents=True);(output/'images').mkdir();(output/'masks').mkdir();(output/'sources').mkdir()
    manifest=[];selected_meta=[];sheet_rows=[]
    for number,c in enumerate(selected):
        old=c['source']; image_id=old['natural']['image_id']; sid=f'vg-{image_id}'
        with Image.open(c['image_path']) as raw:main,fit=fit_image(raw)
        mask=color_mask(main,c['bbox'],c['color'],COLOR_REGION_POLYGONS.get(image_id)); mask_path=Path('masks')/f'{sid}.png';mask.save(output/mask_path)
        gray=Image.new('RGB',SIZE,(127,127,127)); donor_a=recolor(main,mask,'red');donor_b=recolor(main,mask,'blue')
        images={'d_a':donor_a,'d_b':donor_b}; specs={}
        for layout in ('M0','M1'):
            arranged=main if layout=='M0' else ImageOps.mirror(main)
            for strength in (8,100):
                name=f'{layout.lower()}_s{strength:03d}'
                images[name]=Image.blend(gray,arranged,strength/100)
                specs[name]={'type':'main','strength':strength/100,'layout':layout,'assignment':None}
        specs.update({f'd_{a}':{'type':'donor','strength':1.,'layout':'M0','assignment':a} for a in ('a','b')})
        paths={};metadata={}
        for name,image in images.items():
            relative=Path('images')/f'{sid}-{name}.png';image.save(output/relative)
            paths[name]=str(relative);p=output/relative
            metadata[name]={'width':SIZE[0],'height':SIZE[1],'bytes':p.stat().st_size,'sha256':sha256_file(p),'source_field':'image_1'}
        original=Path('sources')/f'{image_id}.jpg';shutil.copyfile(c['image_path'],output/original)
        target={'object_id':old['natural']['object_id'],'name':old['target_object'],'color':c['color'],'bbox_xyxy':c['bbox']}
        query_objects=[('changed',target)]+([('unchanged',c['control'])] if c['control'] else [])
        for role,obj in query_objects:
            ans=('red','blue').index(obj['color'])
            source_answers={n:None if role=='changed' and n.startswith('d_') else ans for n in images}
            row={'schema':ROW_SCHEMA,'family':FAMILY,'task':'natural_scene_color','id':f'{sid}-{role}',
                 'scene_id':sid,'unit_id':sid,'intervention_id':sid,'attribute':'color','generation_group':number%4,
                 'query_role':role,'query_description':QUERY_DESCRIPTIONS.get(image_id,obj['name']) if role=='changed' else obj['name'],'query':f"What color is the {QUERY_DESCRIPTIONS.get(image_id,obj['name']) if role=='changed' else obj['name']}?",'choices':['red','blue'],'option_labels':['(A)','(B)'],
                 'answer_index':ans,'answer_label':('(A)','(B)')[ans],'direction_a_index':0,'direction_b_index':1,
                 'changed_object':target['name'],'changed_object_id':target['object_id'],'changed_index':0,
                 'queried_object':obj['name'],'queried_object_id':obj['object_id'],'queried_index':0 if role=='changed' else 1,
                 'target_object':obj['name'],'target_index':0 if role=='changed' else 1,'target_main_slot':0,
                 'target_main_slot_by_layout':{'M0':0,'M1':0},'source_roi_slot':0,
                 'donor_roi_bbox_xyxy':c['bbox'],'donor_roi_image_size':list(SIZE),'tile_grid':[1,1],'geometry':'natural_full',
                 'changed_bbox_by_layout':{'M0':c['bbox'],'M1':mirror_bbox(c['bbox'])},
                 'queried_bbox_by_layout':{'M0':obj['bbox_xyxy'],'M1':mirror_bbox(obj['bbox_xyxy'])},
                 'source_images':paths,'source_image_metadata':metadata,'source_specs':specs,'source_answer_indices':source_answers,
                 'mask_path':str(mask_path),'mask_sha256':sha256_file(output/mask_path),'mask_pixels':c['mask_pixels'],'mask_clip_polygon':COLOR_REGION_POLYGONS.get(image_id),
                 'has_unchanged_query':c['control'] is not None,'fit':fit,'source_photo':str(original),
                 'source':{**deepcopy(old['source']),'revision':ROW_SCHEMA,'construction':'whole_scene_bbox_color_region_edit',
                           'parent_manifest_sha256':sha256_file(source_manifest),'parent_native_id':old['id'],
                           'source_image_sha256':sha256_file(output/original),'image_id':image_id,
                           'object_id':target['object_id'],'seed':SEED,'split':'exploratory_reuse'},
                 'record_fields':{'native_color_annotation':obj['color'],'query_object_id':obj['object_id']}}
            row['prompt']=row['query']+'\n(A) red\n(B) blue\n';row['visual_source_signature']=visual_signature(row);manifest.append(row)
        selected_meta.append({'scene_id':sid,'image_id':image_id,'category':target['name'],'original_color':c['color'],
                              'bbox':c['bbox'],'mask_pixels':c['mask_pixels'],'bbox_pixels':c['bbox_pixels'],
                              'control':c['control'],'source_sha256':sha256_file(output/original)})
        # Full image overview; inset overlays are for review only and never model inputs.
        overlay=main.copy();overlay.paste(Image.new('RGB',SIZE,(255,0,255)),mask=mask.point(lambda x:90 if x else 0))
        d=ImageDraw.Draw(overlay);d.rectangle(tuple(c['bbox']),outline='yellow',width=2)
        if c['control']:d.rectangle(tuple(c['control']['bbox_xyxy']),outline='lime',width=2)
        sheet_rows.append((sid,target['name'],c['control']['name'] if c['control'] else '-', [main,donor_a,donor_b,overlay,ImageOps.mirror(main)]))
    (output/'manifest.jsonl').write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in manifest))
    checks=[{'source_id':c['source']['id'],'image_id':c['source']['natural']['image_id'],'category':c['source']['target_object'],
             'reasons':c['reasons'],'has_candidate_control':c['control'] is not None,
             'mask_pixels':c['mask_pixels'],'bbox_pixels':c['bbox_pixels'],
             'selected':c['source']['natural']['image_id'] in {m['image_id'] for m in selected_meta}} for c in candidates]
    _dump(output/'selection.json',{'seed':SEED,'selection':'8 categories x 4; eligible off-target first, then seeded hash; image-only selection',
                                 'candidates':checks,'selected':selected_meta,'visual_exclusions':VISUAL_EXCLUSIONS,'query_descriptions':QUERY_DESCRIPTIONS,'control_exclusions':CONTROL_EXCLUSIONS,'manual_color_region_clips':COLOR_REGION_POLYGONS})
    sheets=[]
    for start in range(0,len(sheet_rows),8):
        sheet=Image.new('RGB',(5*192,8*218),(240,240,240));draw=ImageDraw.Draw(sheet)
        for j,(sid,category,control,images) in enumerate(sheet_rows[start:start+8]):
            draw.text((5,j*218+2),f'{sid}: {category}; unchanged={control}',fill='black')
            for k,img in enumerate(images):sheet.paste(img.resize((192,192)),(k*192,j*218+22))
        name=f'contact_sheet_{start//8+1:02d}.jpg';sheet.save(output/name,quality=94);sheets.append(name)
    report=validate(output/'manifest.jsonl');report.update(contact_sheets=sheets,selection_file='selection.json',
        visual_review_status='pending',mask_semantics='HSV color-region pixels within GQA bbox; not instance segmentation',
        model_results=False,preparation_scope='Exploratory whole-scene controlled editing; not unedited natural task performance')
    _dump(output/'dataset.json',report)
    return report


def validate_query_groups(rows):
    if not rows or len({r['id'] for r in rows})!=len(rows):raise ValueError('Unique nonempty natural queries required')
    grouped=defaultdict(list)
    for r in rows:grouped[r['scene_id']].append(r)
    for members in grouped.values():
        expected={'changed','unchanged'} if members[0]['has_unchanged_query'] else {'changed'}
        if {r['query_role'] for r in members}!=expected or len(members)!=len(expected):raise ValueError('Incomplete natural query group')
        for r in members:
            for f in ('intervention_id','direction_a_index','direction_b_index','donor_roi_bbox_xyxy','visual_source_signature','has_unchanged_query'):
                if r[f]!=members[0][f]:raise ValueError(f'Query pair differs in {f}')


def load_manifest(manifest,image_root=None,*,check_hashes=True):
    manifest=_guard_path(Path(manifest));root=_guard_path(Path(image_root)) if image_root else manifest.parent
    rows=[json.loads(s) for s in manifest.read_text().splitlines() if s];checked={}
    for r in rows:
        if r['schema']!=ROW_SCHEMA or r['family']!=FAMILY:raise ValueError('Wrong natural schema')
        if r['choices']!=['red','blue'] or r['option_labels']!=['(A)','(B)'] or (r['direction_a_index'],r['direction_b_index'])!=(0,1):raise ValueError('Fixed red-to-blue direction required')
        if r['tile_grid']!=[1,1] or r['donor_roi_image_size']!=list(SIZE):raise ValueError('Natural full-scene geometry mismatch')
        if r['query']!=f"What color is the {r['query_description']}?":raise ValueError('Query identity mismatch')
        if r['visual_source_signature']!=visual_signature(r):raise ValueError('Visual signature mismatch')
        for n,relative in r['source_images'].items():
            rel=Path(relative)
            if rel.is_absolute() or '..' in rel.parts:raise ValueError('Portable contained paths required')
            p=_guard_path(root/rel)
            if not p.is_relative_to(root) or not p.is_file():raise ValueError('Missing/escaping image')
            meta=r['source_image_metadata'][n]
            if (meta['width'],meta['height'])!=SIZE:raise ValueError('Image metadata size mismatch')
            if check_hashes:
                if p not in checked:checked[p]=(p.stat().st_size,sha256_file(p))
                if checked[p]!=(meta['bytes'],meta['sha256']):raise ValueError('Image bytes/hash mismatch')
            r['source_images'][n]=str(p)
        for field in ('mask_path','source_photo'):
            relative=Path(r[field])
            if relative.is_absolute() or '..' in relative.parts:raise ValueError('Contained portable auxiliary path required')
            path=_guard_path(root/relative)
            if not path.is_relative_to(root) or not path.is_file():raise ValueError('Missing/escaping auxiliary file')
            r[field]=str(path)
        original=r['answer_index']
        if original not in (0,1) or r['answer_label']!=r['option_labels'][original]:raise ValueError('Invalid native label')
        expected_answers={n:None if r['query_role']=='changed' and n.startswith('d_') else original for n in r['source_images']}
        if r['source_answer_indices']!=expected_answers:raise ValueError('Native source answer contract mismatch')
        if r['changed_bbox_by_layout']!={'M0':r['donor_roi_bbox_xyxy'],'M1':mirror_bbox(r['donor_roi_bbox_xyxy'])}:raise ValueError('Mirror bbox contract mismatch')
    validate_query_groups(rows)
    return rows


def select_panel(rows,panel):
    if panel=='main':return rows
    if panel!='smoke':raise ValueError('Unknown natural panel')
    scenes=[]
    # One with off-target and one without, when available; keep complete groups.
    for present in (True,False):
        ids=sorted({r['scene_id'] for r in rows if r['has_unchanged_query']==present})
        if ids:scenes.append(ids[0])
    result=[r for r in rows if r['scene_id'] in scenes];validate_query_groups(result);return result


def validate(manifest,image_root=None):
    from PIL import Image,ImageOps
    import numpy as np
    rows=load_manifest(manifest,image_root);scenes={r['scene_id']:r for r in rows}
    for r in scenes.values():
        arrays={n:np.asarray(Image.open(p).convert('RGB')) for n,p in r['source_images'].items()}
        mask=np.asarray(Image.open(r['mask_path']))>0
        if sha256_file(r['mask_path'])!=r['mask_sha256'] or int(mask.sum())!=r['mask_pixels']:raise ValueError('Mask hash/size mismatch')
        x0,y0,x1,y1=r['donor_roi_bbox_xyxy'];roi=np.zeros(mask.shape,bool);roi[y0:y1,x0:x1]=True
        if np.any(mask & ~roi):raise ValueError('Mask escaped nominal bbox')
        diff=np.any(arrays['d_a']!=arrays['d_b'],axis=2)
        if not np.array_equal(diff,mask):raise ValueError('Only and all declared mask pixels must differ a/b')
        main=arrays['m0_s100']
        if any(np.any(arrays[n][~mask]!=main[~mask]) for n in ('d_a','d_b')):raise ValueError('Editing changed background pixels')
        if not np.array_equal(arrays['m1_s100'],main[:,::-1]):raise ValueError('M1 must be exact horizontal pixel mirror')
        with Image.open(r['source_photo']) as raw:expected,fit=fit_image(raw)
        if not np.array_equal(np.asarray(expected),main) or fit!=r['fit']:raise ValueError('Main must preserve full original photograph under declared fit')
        for strength in (8,):
            for m in ('m0','m1'):
                expected=np.asarray(Image.blend(Image.new('RGB',SIZE,(127,127,127)),Image.fromarray(arrays[f'{m}_s100']),strength/100))
                if not np.array_equal(expected,arrays[f'{m}_s{strength:03d}']):raise ValueError('Strength pixels mismatch')
        for q in [x for x in rows if x['scene_id']==r['scene_id']]:
            if q['query_role']=='unchanged' and not _disjoint(q['queried_bbox_by_layout']['M0'],r['donor_roi_bbox_xyxy']):raise ValueError('Off-target bbox overlaps edit ROI')
    return {'schema':ROW_SCHEMA,'status':'passed','manifest':str(manifest),'manifest_sha256':sha256_file(manifest),
            'items':len(rows),'independent_source_photos':len(scenes),'changed_queries':sum(r['query_role']=='changed' for r in rows),
            'unchanged_queries':sum(r['query_role']=='unchanged' for r in rows),'pixel_pairs_verified':len(scenes),
            'unique_image_paths':len({p for r in rows for p in r['source_images'].values()}),'canvas':list(SIZE),
            'independence_unit':'source photo scene_id; query, layout, strength and mapping are repeats'}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('build','validate'));p.add_argument('--output',type=Path,default=DEFAULT_ROOT);a=p.parse_args(argv)
    print(json.dumps(build(a.output) if a.command=='build' else validate(a.output/'manifest.jsonl'),indent=2))

if __name__=='__main__':main()
