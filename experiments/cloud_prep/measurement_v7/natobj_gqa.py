"""GQA scene graphs and a Visual Genome image subset for the v7 natural-object bridge.

Standard library + Pillow only; no model imports. Every download is untrusted data:
each kind lives in its own directory on Research (raw/, extracted/, images/), this code
lives in the repo, and Python reading those files runs with ``-I`` from the repo. Nothing
inside a download directory is imported or executed. Downloads resume and are polite
(sequential by default, at most 4 concurrent, retry with backoff). The full 20 GB GQA
images.zip is never fetched on the Mac; only individually needed VG images.

  python -I -B cloud_prep/measurement_v7/natobj_gqa.py scene-graphs --root ./work/datasets/gqa-official
  python -I -B cloud_prep/measurement_v7/natobj_gqa.py extract --root ./work/datasets/gqa-official
  python -I -B cloud_prep/measurement_v7/natobj_gqa.py index --root ./work/datasets/gqa-official
  python -I -B cloud_prep/measurement_v7/natobj_gqa.py fetch --root ... --ids-file <fetch_ids.json> --workers 4
  python -I -B cloud_prep/measurement_v7/natobj_gqa.py import-local --root ... --ids-file <fetch_ids.json> --source <images.zip>
"""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import re
import sys
import threading
import time
import urllib.error
import urllib.request
import zipfile

SCENE_GRAPHS_URL='https://downloads.cs.stanford.edu/nlp/data/gqa/sceneGraphs.zip'
VG_URLS=('https://cs.stanford.edu/people/rak248/VG_100K/{id}.jpg','https://cs.stanford.edu/people/rak248/VG_100K_2/{id}.jpg')
SPLIT_FILES={'train':'train_sceneGraphs.json','val':'val_sceneGraphs.json'}
USER_AGENT='DeepStack-research-subset-fetch/1.0 (academic; sequential, <=4 concurrent)'
INDEX_SCHEMA='gqa-objects-by-image-v1'
# Basic color vocabulary used for annotation sanity (any other color term marks a conflict).
BASIC_COLORS=('red','blue','green','yellow','orange','pink','purple','white','black','gray','grey','brown',
              'silver','gold','tan','beige','maroon','navy','teal','turquoise','cream','khaki','blond')
MAX_WORKERS=4


def now():return datetime.now(timezone.utc).isoformat(timespec='seconds')


def sha256_file(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(1<<20),b''):digest.update(block)
    return digest.hexdigest()


def guard(path,write=False):
    return Path(path).expanduser().resolve()


def _request(url,headers=None,timeout=60,method=None):
    return urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':USER_AGENT,**(headers or {})},method=method),timeout=timeout)


def download(url,dest,*,retries=6,backoff=2.):
    """Resumable single-file download with HTTP Range; returns provenance."""
    dest=guard(dest,write=True);part=dest.with_name(dest.name+'.part')
    for attempt in range(retries):
        try:
            with _request(url,timeout=60,method='HEAD') as head:
                total=int(head.headers.get('Content-Length') or 0)
                meta={'url':url,'last_modified':head.headers.get('Last-Modified'),'etag':head.headers.get('ETag'),
                      'accept_ranges':head.headers.get('Accept-Ranges'),'content_length':total}
            break
        except (urllib.error.URLError,TimeoutError,ConnectionError) as exc:
            print(f'HEAD retry {attempt+1}: {exc}',file=sys.stderr);time.sleep(backoff**attempt)
    else:raise RuntimeError(f'HEAD failed: {url}')
    if dest.exists() and dest.stat().st_size==total:
        return {**meta,'bytes':total,'sha256':sha256_file(dest),'resumed':False,'already_present':True}
    for attempt in range(retries):
        try:
            have=part.stat().st_size if part.exists() else 0
            with _request(url,{'Range':f'bytes={have}-'} if have else None,timeout=120) as resp,part.open('ab' if have and resp.status==206 else 'wb') as out:
                while block:=resp.read(1<<20):out.write(block)
            if part.stat().st_size==total:break
        except (urllib.error.URLError,TimeoutError,ConnectionError) as exc:
            print(f'download retry {attempt+1}: {exc}',file=sys.stderr);time.sleep(backoff**attempt)
    else:raise RuntimeError(f'Download incomplete: {url}')
    part.replace(dest)
    return {**meta,'bytes':dest.stat().st_size,'sha256':sha256_file(dest),'downloaded_at':now()}


def scene_graphs(root,url=SCENE_GRAPHS_URL):
    root=guard(root,write=True);raw=root/'raw';prov=root/'provenance'
    raw.mkdir(parents=True,exist_ok=True);prov.mkdir(exist_ok=True)
    record=prov/'sceneGraphs.download.json';previous=json.loads(record.read_text()) if record.exists() else None
    info=download(url,raw/'sceneGraphs.zip')
    if previous and info.get('already_present'):
        if previous['sha256']!=info['sha256']:raise ValueError('sceneGraphs.zip differs from its recorded download')
        previous.setdefault('reverified',[]).append({'at':now(),'last_modified':info['last_modified'],'etag':info['etag'],'bytes':info['bytes']})
        info=previous
    record.write_text(json.dumps(info,indent=2)+'\n')
    return info


def extract(root,max_total=8<<30):
    """Extract only flat *.json/*.txt members after name/size checks; never execute contents."""
    root=guard(root,write=True);dest=root/'extracted';dest.mkdir(exist_ok=True)
    source=root/'raw'/'sceneGraphs.zip';out={'zip':str(source),'zip_sha256':sha256_file(source),'members':[]}
    with zipfile.ZipFile(source) as archive:
        infos=[i for i in archive.infolist() if not i.is_dir()]
        if sum(i.file_size for i in infos)>max_total:raise ValueError('Archive larger than declared limit')
        for info in infos:
            name=info.filename
            if name.startswith('__MACOSX/') or Path(name).name.startswith('._'):continue
            if not re.fullmatch(r'[A-Za-z0-9_\-]+\.(json|txt)',Path(name).name) or '..' in Path(name).parts or Path(name).is_absolute():
                raise ValueError(f'Unexpected archive member {name!r}')
            target=dest/Path(name).name
            if not target.exists() or target.stat().st_size!=info.file_size:
                with archive.open(info) as src,target.open('wb') as dst:
                    while block:=src.read(1<<20):dst.write(block)
            out['members'].append({'name':name,'file':target.name,'bytes':target.stat().st_size,'sha256':sha256_file(target)})
    (root/'provenance'/'sceneGraphs.extract.json').write_text(json.dumps(out,indent=2)+'\n')
    return out


def _norm(text):return re.sub(r'\s+',' ',str(text).strip().lower())


def index(root,splits=('train','val'),colors=BASIC_COLORS):
    """Compact per-image object lists: every object as [object_id, name, x, y, w, h, attributes].

    Names/attributes are lower-cased and whitespace-normalized; relations are dropped. Keeping all
    objects lets category, synonym, duplicate and overlap rules change later without re-indexing.
    """
    root=guard(root,write=True);out_dir=root/'index';out_dir.mkdir(exist_ok=True)
    color_set=set(colors);images={};single=Counter();n_objects=0
    out_path=out_dir/'objects_by_image.jsonl.gz'
    with gzip.GzipFile(out_path,'wb',compresslevel=6,mtime=0) as raw:
        for split in splits:
            data=json.loads((root/'extracted'/SPLIT_FILES[split]).read_text())
            for image_id in sorted(data,key=lambda k:(len(k),k)):
                graph=data[image_id];objects=graph.get('objects') or {};rows=[]
                for oid in sorted(objects,key=lambda k:(len(k),k)):
                    obj=objects[oid];attrs=list(dict.fromkeys(_norm(a) for a in obj.get('attributes') or []))
                    rows.append([oid,_norm(obj['name']),obj['x'],obj['y'],obj['w'],obj['h'],attrs])
                    found=[a for a in attrs if a in color_set]
                    if len(found)==1:single[(rows[-1][1],found[0])]+=1
                n_objects+=len(rows)
                images[image_id]={'split':split,'width':graph['width'],'height':graph['height'],'n_objects':len(rows)}
                raw.write((json.dumps({'image_id':image_id,'split':split,'width':graph['width'],'height':graph['height'],
                                       'objects':rows},separators=(',',':'))+'\n').encode())
            del data
    (out_dir/'images.json').write_text(json.dumps(images,sort_keys=True)+'\n')
    top=lambda c:[[n,k] for (n,cc),k in single.most_common() if cc==c][:80]
    summary={'schema':INDEX_SCHEMA,'splits':list(splits),'scene_graph_files':{s:{'file':SPLIT_FILES[s],
             'sha256':sha256_file(root/'extracted'/SPLIT_FILES[s])} for s in splits},'images':len(images),'objects':n_objects,
             'index':out_path.name,'index_sha256':sha256_file(out_path),'images_json_sha256':sha256_file(out_dir/'images.json'),
             'basic_colors':list(colors),'note':'Counts below: objects whose only basic-color attribute is the given color (no geometry filters).',
             'top_single_color_names':{c:top(c) for c in ('red','blue','green','yellow')}}
    (out_dir/'index_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary


def _fetch_one(image_id,size,dest,retries=5,backoff=2.,pause=.25):
    from PIL import Image
    record={'image_id':image_id,'scene_graph_size':list(size),'attempts':[]}
    final=dest/f'{image_id}.jpg';part=dest/f'.{image_id}.part'
    for template in VG_URLS:
        url=template.format(id=image_id)
        for attempt in range(retries):
            time.sleep(pause)
            try:
                with _request(url,timeout=60) as resp:
                    body=resp.read();status=resp.status;headers=resp.headers
                record['attempts'].append({'url':url,'status':status})
                break
            except urllib.error.HTTPError as exc:
                record['attempts'].append({'url':url,'status':exc.code})
                if exc.code==404:body=None;break
                time.sleep(backoff**attempt)
            except (urllib.error.URLError,TimeoutError,ConnectionError,OSError) as exc:
                record['attempts'].append({'url':url,'error':type(exc).__name__});time.sleep(backoff**attempt)
        else:body=None
        if body is None:continue
        part.write_bytes(body)
        try:
            with Image.open(part) as im:
                im.load();actual=list(im.size);mode=im.mode;fmt=im.format;orientation=im.getexif().get(0x0112)
        except Exception as exc:
            record['attempts'][-1]['decode_error']=type(exc).__name__;part.unlink(missing_ok=True);continue
        if actual!=list(size):
            record['attempts'][-1]['size_mismatch']=actual;part.unlink(missing_ok=True);continue
        part.replace(final)
        return {**record,'status':'ok','url':url,'file':final.name,'bytes':len(body),'sha256':hashlib.sha256(body).hexdigest(),
                'size':actual,'mode':mode,'format':fmt,'exif_orientation':orientation,'last_modified':headers.get('Last-Modified'),
                'content_type':headers.get('Content-Type'),'fetched_at':now()}
    return {**record,'status':'failed'}


def fetch(root,image_ids,workers=1,log_name='fetch_log.jsonl'):
    """Fetch only the listed VG images; verify decoded size == scene-graph size (GQA id <-> VG file)."""
    if not 1<=workers<=MAX_WORKERS:raise ValueError(f'Use 1..{MAX_WORKERS} workers')
    root=guard(root,write=True);dest=root/'images';dest.mkdir(exist_ok=True);log=root/'provenance'/log_name
    sizes=json.loads((root/'index'/'images.json').read_text())
    done={}
    if log.exists():
        for line in log.read_text().splitlines():
            r=json.loads(line)
            if r['status']=='ok' and (dest/r['file']).exists() and sha256_file(dest/r['file'])==r['sha256']:done[r['image_id']]=r
    todo=[i for i in dict.fromkeys(map(str,image_ids)) if i not in done]
    unknown=[i for i in todo if i not in sizes]
    if unknown:raise ValueError(f'Image ids absent from scene graphs: {unknown[:5]}')
    lock=threading.Lock();results=[]
    def run(image_id):
        r=_fetch_one(image_id,(sizes[image_id]['width'],sizes[image_id]['height']),dest)
        with lock:
            with log.open('a') as handle:handle.write(json.dumps(r,sort_keys=True)+'\n')
            results.append(r)
            if len(results)%25==0:print(f'fetched {len(results)}/{len(todo)}',file=sys.stderr)
    with ThreadPoolExecutor(max_workers=workers) as pool:list(pool.map(run,todo))
    status=Counter(r['status'] for r in results)
    return {'requested':len(set(map(str,image_ids))),'already_ok':len(done),'fetched_now':status.get('ok',0),
            'failed':sorted(r['image_id'] for r in results if r['status']!='ok'),'log':str(log)}


def import_local(root,image_ids,source,log_name='fetch_log.jsonl'):
    """Cloud/offline path: take photos from GQA images.zip (members images/<id>.jpg) or an unpacked directory.

    Same decode + scene-graph size identity check and the same append-only log as fetch(); only listed ids
    are read (no full extraction). Use when per-image VG downloads are undesirable (large rebuilds).
    """
    from PIL import Image
    import io
    root=guard(root,write=True);dest=root/'images';dest.mkdir(exist_ok=True);log=root/'provenance'/log_name
    sizes=json.loads((root/'index'/'images.json').read_text());source=Path(source);done=fetch_status(root,log_name)
    archive=zipfile.ZipFile(source) if source.is_file() else None;counts=Counter()
    try:
        names=set(archive.namelist()) if archive else None
        unknown=[i for i in dict.fromkeys(map(str,image_ids)) if i not in sizes]
        if unknown:raise ValueError(f'Image ids absent from scene graphs: {unknown[:5]}')
        for image_id in dict.fromkeys(map(str,image_ids)):
            if done.get(image_id,{}).get('status')=='ok':counts['already_ok']+=1;continue
            size=[sizes[image_id]['width'],sizes[image_id]['height']];record={'image_id':image_id,'scene_graph_size':size,'attempts':[]}
            member=f'images/{image_id}.jpg'
            if archive:body=archive.read(member) if member in names else None
            else:body=(source/f'{image_id}.jpg').read_bytes() if (source/f'{image_id}.jpg').is_file() else None
            url=f'local:{source.name}:{member if archive else image_id+".jpg"}'
            actual=None
            if body is not None:
                try:
                    with Image.open(io.BytesIO(body)) as im:im.load();actual=list(im.size);mode=im.mode;fmt=im.format;orientation=im.getexif().get(0x0112)
                except Exception as exc:
                    record['attempts'].append({'url':url,'decode_error':type(exc).__name__})
            if body is None:record.update(status='failed',attempts=[{'url':url,'status':'missing'}])
            elif actual is None:record['status']='failed'
            elif actual!=size:record.update(status='failed',attempts=[{'url':url,'size_mismatch':actual}])
            else:
                (dest/f'{image_id}.jpg').write_bytes(body)
                record.update(status='ok',url=url,file=f'{image_id}.jpg',bytes=len(body),sha256=hashlib.sha256(body).hexdigest(),
                              size=actual,mode=mode,format=fmt,exif_orientation=orientation,last_modified=None,fetched_at=now())
            counts[record['status']]+=1
            with log.open('a') as handle:handle.write(json.dumps(record,sort_keys=True)+'\n')
    finally:
        if archive:archive.close()
    return dict(counts)


def fetch_status(root,log_name='fetch_log.jsonl'):
    """Latest record per image id from the append-only fetch log; 'ok' records must still match file bytes."""
    root=guard(root);out={};path=root/'provenance'/log_name
    if path.exists():
        for line in path.read_text().splitlines():
            r=json.loads(line)
            if r['status']=='ok' or out.get(r['image_id'],{}).get('status')!='ok':out[r['image_id']]=r
    for r in out.values():
        if r['status']=='ok' and (not (root/'images'/r['file']).is_file() or (root/'images'/r['file']).stat().st_size!=r['bytes']):
            raise ValueError(f'Fetched photo missing or changed: {r["file"]}')
    return out


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('command',choices=('scene-graphs','extract','index','fetch','import-local'))
    p.add_argument('--root',type=Path,required=True);p.add_argument('--url',default=SCENE_GRAPHS_URL)
    p.add_argument('--splits',nargs='+',default=['train','val'],choices=sorted(SPLIT_FILES))
    p.add_argument('--ids-file',type=Path,help='JSON list of GQA/VG image ids, or object with "image_ids"')
    p.add_argument('--ids',nargs='*',default=[]);p.add_argument('--workers',type=int,default=1)
    p.add_argument('--source',type=Path,help='import-local: GQA images.zip or a directory of <id>.jpg')
    a=p.parse_args(argv)
    if a.command=='scene-graphs':result=scene_graphs(a.root,a.url)
    elif a.command=='extract':result=extract(a.root)
    elif a.command=='index':result=index(a.root,tuple(a.splits))
    else:
        ids=list(a.ids)
        if a.ids_file:
            loaded=json.loads(a.ids_file.read_text());ids+=loaded['image_ids'] if isinstance(loaded,dict) else loaded
        if not ids:p.error(f'{a.command} requires --ids or --ids-file')
        if a.command=='fetch':result=fetch(a.root,ids,a.workers)
        elif a.source is None:p.error('import-local requires --source')
        else:result=import_local(a.root,ids,a.source)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
