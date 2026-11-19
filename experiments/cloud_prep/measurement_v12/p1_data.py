"""CPU-only P1 natural-photo materials: two edits, two queries per edit.

This is an image experiment construction, not an instance-segmentation method.
Selection and review never inspect model responses. Raw GQA annotations only
nominate candidates; the reviewed selection records are a separate authority.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import io
import itertools
import json
from pathlib import Path
import os as _path_os
DATA_ROOT = Path(_path_os.environ.get("DS_DATA_ROOT", "work")).expanduser().resolve()
import shutil
import urllib.request

from cloud_prep.portable_data import _guard_path, sha256_file
from cloud_prep.measurement_v10.natural_data import (
    SIZE, COLORS, fit_image, transform_bbox, mirror_bbox, color_mask, recolor, _disjoint)

ROOT = (DATA_ROOT / 'datasets/natural-selectivity-v12')
DATASETS = ROOT.parent
GQA = DATASETS / 'gqa-official'
SCHEMA = 'natural-selectivity-v12-data-v1'
SEED = 2026102201
NAMES = set('bag|backpack|bus|car|chair|pillow|train|truck|umbrella|shirt|jacket|coat|jeans|shorts|cup|cooler|van|taxi|fire hydrant|sofa|carpet|rug|bedspread|hat|cap|helmet|pants|tie|dress|sweater|t-shirt|scarf|blanket|bowl|plate|vase|bottle|door|bench|boat|motorcycle|suitcase|suit|towel|frisbee|ball|surfboard|kite'.split('|'))
VG_URLS = ('https://cs.stanford.edu/people/rak248/VG_100K/{id}.jpg',
           'https://cs.stanford.edu/people/rak248/VG_100K_2/{id}.jpg')

def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True)+'\n')

def hash_order(value):
    return hashlib.sha256(f'{SEED}:{value}'.encode()).hexdigest()

def prior_sources(datasets=DATASETS):
    """Include every source ID in prior manifests/panels, including real crops."""
    ids, inputs = set(), []
    def walk(obj):
        if isinstance(obj, dict):
            for key, value in obj.items():
                if key == 'image_id': ids.add(str(value))
                walk(value)
        elif isinstance(obj, list):
            for value in obj: walk(value)
    for path in sorted(Path(datasets).rglob('*.jsonl')):
        if 'gqa-official' in path.parts or 'natural-selectivity-v12' in path.parts: continue
        if 'provenance' in path.parts or ('manifest' not in path.name and 'panel' not in path.name): continue
        before = len(ids)
        for line in path.read_text().splitlines():
            if line: walk(json.loads(line))
        inputs.append({'path':str(path),'sha256':sha256_file(path),'new_ids':len(ids)-before})
    return {'source_ids':sorted(ids), 'inputs':inputs,
            'rule':'all image_id fields in prior dataset manifests and panels; no model-result access'}

def make_index(root=ROOT, *, excluded_ids=None):
    root = _guard_path(Path(root), write=True); root.mkdir(parents=True, exist_ok=True)
    excluded = prior_sources() if excluded_ids is None else {'source_ids': list(excluded_ids)}
    used=set(excluded['source_ids']); rows=[]
    with gzip.open(GQA/'index/objects_by_image.jsonl.gz', 'rt') as handle:
        for line in handle:
            graph=json.loads(line)
            if graph['image_id'] in used: continue
            counts=Counter(o[1] for o in graph['objects']); objs=[]
            for oid,name,x,y,w,h,attrs in graph['objects']:
                cs=set(attrs)&COLORS
                if name not in NAMES or counts[name]!=1 or len(cs)!=1 or not cs&{'red','blue'}:continue
                if min(w,h)*SIZE[0]/max(graph['width'],graph['height']) < 24: continue
                objs.append({'object_id':oid,'name':name,'annotation_color':next(iter(cs)),
                             'bbox_xywh_original':[x,y,w,h], 'attributes':attrs})
            pairs=[]
            for a,b in itertools.combinations(objs,2):
                boxes=[[o['bbox_xywh_original'][0],o['bbox_xywh_original'][1],
                        o['bbox_xywh_original'][0]+o['bbox_xywh_original'][2],
                        o['bbox_xywh_original'][1]+o['bbox_xywh_original'][3]] for o in (a,b)]
                if _disjoint(*boxes):pairs.append([a,b])
            if pairs:
                pairs.sort(key=lambda p:(-min(o['bbox_xywh_original'][2]*o['bbox_xywh_original'][3] for o in p), hash_order(str(p))))
                rows.append({'image_id':graph['image_id'],'split':graph['split'],
                             'annotated_size':[graph['width'],graph['height']], 'pairs':pairs,
                             'cached':(GQA/'images'/f"{graph['image_id']}.jpg").is_file()})
    rows.sort(key=lambda x:(not x['cached'],hash_order(x['image_id'])))
    dump(root/'excluded_sources.json', excluded)
    dump(root/'candidate_index.json', {'schema':SCHEMA, 'seed':SEED, 'index_sha256':sha256_file(GQA/'index/objects_by_image.jsonl.gz'),
        'scene_graph_provenance':json.loads((GQA/'provenance/sceneGraphs.download.json').read_text()),
        'selection_basis':'unique named red/blue annotation; disjoint boxes; both >=24 canvas px; then pixels and visual review',
        'candidates':rows})
    return {'candidates':len(rows),'cached':sum(x['cached'] for x in rows),'excluded':len(used)}

def fetch_one(row, root):
    from PIL import Image
    image_id=row['image_id']; target=root/'sources'/f'{image_id}.jpg'
    cached=GQA/'images'/target.name
    if target.is_file(): return {'image_id':image_id,'status':'present','sha256':sha256_file(target)}
    if cached.is_file():
        shutil.copyfile(cached,target)
        return {'image_id':image_id,'status':'copied_cache','original_path':str(cached),'sha256':sha256_file(target)}
    attempts=[]
    for template in VG_URLS:
        url=template.format(id=image_id)
        try:
            request=urllib.request.Request(url,headers={'User-Agent':'DeepStack-research-subset-fetch/1.0'})
            with urllib.request.urlopen(request,timeout=25) as response:
                content=response.read();headers={k:response.headers.get(k) for k in ('Last-Modified','ETag','Content-Type')}
            with Image.open(io.BytesIO(content)) as image:
                image.verify()
            target.write_bytes(content)
            return {'image_id':image_id,'status':'downloaded','url':url,'headers':headers,
                    'bytes':len(content),'sha256':sha256_file(target),'attempts':attempts}
        except Exception as exc: attempts.append({'url':url,'error':f'{type(exc).__name__}: {exc}'})
    return {'image_id':image_id,'status':'failed','attempts':attempts}

def fetch_candidates(root=ROOT, limit=220):
    root=_guard_path(Path(root),write=True);(root/'sources').mkdir(exist_ok=True)
    rows=json.loads((root/'candidate_index.json').read_text())['candidates']
    # Cache first; new images prefer uncommon category pairs before repeats.
    pending=[r for r in rows if r['cached']]; remaining=[r for r in rows if not r['cached']]
    counts=Counter(o['name'] for r in pending for o in r['pairs'][0])
    while remaining and len(pending)<limit:
        r=min(remaining,key=lambda x:(sum(counts[o['name']] for o in x['pairs'][0]),hash_order(x['image_id'])))
        pending.append(r);remaining.remove(r);counts.update(o['name'] for o in r['pairs'][0])
    with ThreadPoolExecutor(max_workers=4) as pool:
        results=list(pool.map(lambda r:fetch_one(r,root),pending[:limit]))
    with (root/'fetch_log.jsonl').open('a') as handle:
        for row in results:handle.write(json.dumps(row,sort_keys=True)+'\n')
    return dict(Counter(r['status'] for r in results))

def pixel_candidates(root=ROOT):
    from PIL import Image
    import numpy as np
    root=_guard_path(Path(root),write=True); candidates=[]; rejected=[]
    for row in json.loads((root/'candidate_index.json').read_text())['candidates']:
        path=root/'sources'/f"{row['image_id']}.jpg"
        if not path.is_file():continue
        with Image.open(path) as raw:
            actual_size=list(raw.size); image,fit=fit_image(raw)
        pairs=[]
        for pair in row['pairs']:
            objects=[]
            for obj in pair:
                o=dict(obj);o['bbox_xyxy']=transform_bbox(o['bbox_xywh_original'],fit)
                mask=color_mask(image,o['bbox_xyxy'],o['annotation_color'])
                count=int((np.asarray(mask)>0).sum());x0,y0,x1,y1=o['bbox_xyxy']
                o['mask_pixels']=count;o['mask_fraction']=count/max((x1-x0)*(y1-y0),1)
                objects.append(o)
            if all(o['mask_pixels']>=200 and o['mask_fraction']>=.14 for o in objects):pairs.append(objects)
        if not pairs:rejected.append({'image_id':row['image_id'],'reason':'one or both color components <200px or <14% bbox'});continue
        pairs.sort(key=lambda p:(-min(o['mask_pixels'] for o in p),hash_order(str(p))))
        candidates.append({'image_id':row['image_id'],'split':row['split'],'objects':pairs[0],
                           'annotated_size':row['annotated_size'],'actual_size':actual_size,'fit':fit,
                           'source_sha256':sha256_file(path),'visual_review':'pending'})
    # Review diverse categories first, without native answers or effect filtering.
    ordered=[];counts=Counter()
    while candidates:
        row=min(candidates,key=lambda x:(sum(counts[o['name']] for o in x['objects']),hash_order(x['image_id'])))
        candidates.remove(row);ordered.append(row);counts.update(o['name'] for o in row['objects'])
    for i,r in enumerate(ordered):r['review_index']=i
    dump(root/'pixel_candidates.json',{'candidates':ordered,'rejected':rejected})
    return {'pixel_pass':len(ordered),'pixel_reject':len(rejected)}

def make_review(root=ROOT):
    from PIL import Image,ImageDraw
    import numpy as np
    root=_guard_path(Path(root),write=True);out=root/'review';out.mkdir(exist_ok=True)
    rows=json.loads((root/'pixel_candidates.json').read_text())['candidates'];total=len(rows)
    if (root/'visual_review.json').exists():
        reviewed=set(json.loads((root/'visual_review.json').read_text()).get('reviewed_source_ids',[]))
        rows=[r for r in rows if r['image_id'] not in reviewed]
    sheets=[]
    for start in range(0,len(rows),5):
        sheet=Image.new('RGB',(1152,5*242),(246,246,246));draw=ImageDraw.Draw(sheet)
        for i,row in enumerate(rows[start:start+5]):
            with Image.open(root/'sources'/f"{row['image_id']}.jpg") as raw:image,_=fit_image(raw)
            objs=row['objects'];masks=[color_mask(image,o['bbox_xyxy'],o['annotation_color']) for o in objs]
            overlay=image.copy();d=ImageDraw.Draw(overlay)
            for letter,o,color in zip('AB',objs,('yellow','lime')):
                d.rectangle(o['bbox_xyxy'],outline=color,width=3);d.text((o['bbox_xyxy'][0],o['bbox_xyxy'][1]),letter,fill=color)
            panels=[overlay,recolor(image,masks[0],'red'),recolor(image,masks[0],'blue'),recolor(image,masks[1],'red'),recolor(image,masks[1],'blue')]
            maskviz=Image.new('RGB',SIZE,'black');arr=np.asarray(maskviz).copy();arr[np.asarray(masks[0])>0]=[255,255,0];arr[np.asarray(masks[1])>0]=[0,255,0];panels.append(Image.fromarray(arr))
            draw.text((4,i*242+2),f"{row['review_index']:03d} / {row['image_id']}  A={objs[0]['name']}({objs[0]['annotation_color']}) B={objs[1]['name']}({objs[1]['annotation_color']})",fill='black')
            for j,(label,panel) in enumerate(zip(('Original A/B','A red','A blue','B red','B blue','edit masks'),panels)):
                draw.text((j*192+3,i*242+19),label,fill='black');sheet.paste(panel.resize((192,192)),(j*192,i*242+37))
        path=out/f'batch{total:03d}_{start//5:02d}.jpg';sheet.save(path,quality=94);sheets.append(str(path))
    dump(out/f'batch{total:03d}_index.json',{'sheets':sheets,'rows':rows})
    return {'sheets':sheets,'candidates':len(rows)}

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('index','fetch','pixels','review'))
    p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--limit',type=int,default=220);a=p.parse_args(argv)
    functions={'index':make_index,'fetch':lambda root:fetch_candidates(root,a.limit),'pixels':pixel_candidates,'review':make_review}
    print(json.dumps(functions[a.command](a.root),indent=2))

if __name__=='__main__':main()
