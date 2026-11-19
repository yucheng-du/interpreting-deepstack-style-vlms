"""Fetch only required JPEG members from the official 2019 GQA ZIP via Range.

Never downloads the 21.8GB archive. Validates HTTP ranges, original archive
identity, ZIP CRC and JPEG decoding, recording exact member/range provenance.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import struct
import threading
import time
import urllib.error
import urllib.request
import zipfile
import zlib

from cloud_prep.measurement_v12.p1_data import ROOT, hash_order, dump
from cloud_prep.portable_data import _guard_path, sha256_file

URL='https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip'
EXPECTED_SIZE=21817965542
EXPECTED_ETAG='"5c55f4cb-51473bbe6"'
REQUEST_LOCK=threading.Lock()

def byte_range(start,length):
    if length<=0:return b''
    if length>40_000_000:raise ValueError('Refuse large response; subset only')
    req=urllib.request.Request(URL,headers={'Range':f'bytes={start}-{start+length-1}',
        'If-Match':EXPECTED_ETAG,'User-Agent':'DeepStack-research-GQA-subset/1.0'})
    for attempt in range(4):
        try:
            # The archive server throttles burst Range requests; be sequential.
            with REQUEST_LOCK:
                time.sleep(.35)
                response=urllib.request.urlopen(req,timeout=20)
            with response:
                if response.status!=206:raise RuntimeError('Server did not honor Range; refusing full archive')
                expected=f'bytes {start}-{start+length-1}/{EXPECTED_SIZE}'
                if response.headers.get('Content-Range')!=expected:raise RuntimeError('Content-Range identity mismatch')
                if response.headers.get('ETag')!=EXPECTED_ETAG:raise RuntimeError('Archive ETag changed')
                content=response.read(length+1)
            break
        except (urllib.error.URLError,TimeoutError):
            if attempt==3:raise
            time.sleep(2**(attempt+1))
    if len(content)!=length:raise RuntimeError('Range size mismatch')
    return content

class RemoteZip(io.RawIOBase):
    def __init__(self):self.position=0
    def seekable(self):return True
    def tell(self):return self.position
    def seek(self,offset,whence=0):
        self.position=(0 if whence==0 else self.position if whence==1 else EXPECTED_SIZE)+offset
        return self.position
    def read(self,size=-1):
        size=EXPECTED_SIZE-self.position if size<0 else min(size,EXPECTED_SIZE-self.position)
        data=byte_range(self.position,size);self.position+=len(data);return data

def index(root):
    path=root/'gqa_image_zip_index.json'
    if path.exists():return json.loads(path.read_text())
    with zipfile.ZipFile(RemoteZip()) as archive:
        members={Path(info.filename).stem:{'name':info.filename,'offset':info.header_offset,
            'compressed_bytes':info.compress_size,'bytes':info.file_size,'compression':info.compress_type,'crc32':info.CRC}
            for info in archive.infolist() if info.filename.lower().endswith('.jpg')}
    value={'url':URL,'archive_bytes':EXPECTED_SIZE,'etag':EXPECTED_ETAG,'members':members}
    dump(path,value);return value

def fetch_member(image_id,info,root):
    from PIL import Image
    dest=root/'sources'/f'{image_id}.jpg'
    if dest.exists():return {'image_id':image_id,'status':'present','sha256':sha256_file(dest)}
    try:
        header=byte_range(info['offset'],30)
        fields=struct.unpack('<4s5H3L2H',header)
        if fields[0]!=b'PK\x03\x04':raise ValueError('Invalid local ZIP header')
        name_length,extra_length=fields[-2:]
        data_start=info['offset']+30+name_length+extra_length
        packed=byte_range(data_start,info['compressed_bytes'])
        if info['compression']==8:content=zlib.decompress(packed,-15)
        elif info['compression']==0:content=packed
        else:raise ValueError('Unexpected JPEG ZIP compression')
        if len(content)!=info['bytes'] or zlib.crc32(content)!=info['crc32']:raise ValueError('ZIP CRC/size mismatch')
        with Image.open(io.BytesIO(content)) as im:im.verify()
        dest.write_bytes(content)
        return {'image_id':image_id,'status':'downloaded','url':URL,'etag':EXPECTED_ETAG,'archive_member':info,
                'data_range':[data_start,data_start+len(packed)-1],'bytes':len(content),'sha256':hashlib.sha256(content).hexdigest()}
    except Exception as exc:return {'image_id':image_id,'status':'failed','error':f'{type(exc).__name__}: {exc}'}

def fetch(root=ROOT,limit=220):
    root=_guard_path(Path(root),write=True);(root/'sources').mkdir(exist_ok=True)
    rows=json.loads((root/'candidate_index.json').read_text())['candidates'];idx=index(root)
    selected=[r for r in rows if (root/'sources'/f"{r['image_id']}.jpg").is_file()]
    remaining=[r for r in rows if r not in selected];counts=Counter(o['name'] for r in selected for o in r['pairs'][0])
    while remaining and len(selected)<limit:
        r=min(remaining,key=lambda x:(sum(counts[o['name']] for o in x['pairs'][0]),hash_order(x['image_id'])))
        selected.append(r);remaining.remove(r);counts.update(o['name'] for o in r['pairs'][0])
    with ThreadPoolExecutor(max_workers=2) as pool, (root/'official_zip_fetch_log.jsonl').open('a') as handle:
        results=[]
        for result in pool.map(lambda row:fetch_member(row['image_id'],idx['members'][row['image_id']],root),selected[:limit]):
            results.append(result);handle.write(json.dumps(result,sort_keys=True)+'\n');handle.flush()
    return dict(Counter(r['status'] for r in results))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--limit',type=int,default=220);a=p.parse_args()
    print(json.dumps(fetch(a.root,a.limit),indent=2))
