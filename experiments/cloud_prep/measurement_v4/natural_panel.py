"""Prepare the complete BLINK Visual_Similarity panel for v4, without models."""
from __future__ import annotations
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil

from cloud_prep.common import sha256, write_json
from cloud_prep.portable_data import _guard_path, load_records

SCHEMA = 'natural-vs135-v4-1'
SEED = 20261004


def rank_id(value):
    return hashlib.sha256(f'{SCHEMA}:{SEED}:{value}'.encode()).hexdigest()


def select_rows(rows):
    selected = sorted((deepcopy(r) for r in rows if r['task'] == 'Visual_Similarity'), key=lambda r:r['id'])
    if not selected or len({r['id'] for r in selected}) != len(selected):
        raise ValueError('Empty or duplicate Visual_Similarity panel')
    generated = set(r['id'] for r in sorted(selected, key=lambda r:rank_id(r['id']))[:32])
    for row in selected:
        if len(row['images']) != 3 or len(row['option_labels']) != 2:
            raise ValueError('Expected one reference and two candidate images')
        dims = [(m['width'],m['height']) for m in row['image_metadata']]
        if dims[1] != dims[2]:
            raise ValueError('Candidate geometry differs: version a new alignment arm, never silently filter')
        row['natural_v4'] = {'schema':SCHEMA, 'candidate_slots':[1,2], 'slot_order':[0,2,1],
            'original_answer_label':row['answer_label'], 'swapped_answer_label':row['option_labels'][1-row['answer_index']],
            'generation_selected':row['id'] in generated,
            'selection':'all Visual_Similarity; generation lowest SHA256 ranks independent of outputs',
            'sample_status':'previously explored BLINK; natural-image transfer validation, not unseen confirmation'}
    return selected


def prepare(source, output):
    source = Path(source)
    rows = select_rows(load_records(source))
    output = _guard_path(Path(output), write=True)
    output.mkdir(parents=True, exist_ok=False)
    for row in rows:
        paths = []
        for path, meta in zip(row['images'],row['image_metadata']):
            src = Path(path)
            if sha256(src) != meta['sha256']:
                raise ValueError(f'Image bytes changed: {src}')
            relative = Path('images') / (meta['sha256'] + src.suffix.lower())
            dest = output / relative
            dest.parent.mkdir(exist_ok=True)
            if not dest.exists():
                shutil.copyfile(src,dest)
            paths.append(relative.as_posix())
        row['images'] = paths
    def emit(name, records):
        (output/name).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records))
    emit('natural_vs135.jsonl',rows)
    smoke = sorted(rows,key=lambda r:rank_id(r['id']))[:4]
    emit('natural_vs_smoke4.jsonl',smoke)
    emit('natural_vs_generation32.jsonl',[r for r in rows if r['natural_v4']['generation_selected']])
    info = {'schema':SCHEMA,'source_manifest':str(source),'source_sha256':sha256(source),
        'items':len(rows),'task_counts':dict(Counter(r['task'] for r in rows)),
        'unique_images':len(list((output/'images').iterdir())),
        'image_bytes':sum(p.stat().st_size for p in (output/'images').iterdir()),
        'generation_ids':[r['id'] for r in rows if r['natural_v4']['generation_selected']],
        'smoke_ids':[r['id'] for r in smoke],
        'manifests':{n:sha256(output/n) for n in ('natural_vs135.jsonl','natural_vs_smoke4.jsonl','natural_vs_generation32.jsonl')},
        'image_transform':'none; source bytes copied exactly; candidate shapes equal for every item',
        'cpu_validated':True,'cuda_validated':False}
    write_json(output/'manifest.json',info)
    return info


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(argv); print(json.dumps(prepare(a.source,a.output),indent=2))

if __name__=='__main__':main()
