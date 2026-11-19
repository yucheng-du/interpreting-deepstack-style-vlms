#!/usr/bin/env python3
"""Import local official GQA/Single-Document DocVQA validation annotations.

No downloads or model-dependent filtering. The output is a portable v14 recovery
manifest with one clean/degraded image pair shared by every question on an image.
Use a new output directory for materially changed selections or image operations.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import re
import sys

OFFICIAL = {
    'gqa': dict(dataset='https://cs.stanford.edu/people/dorarad/gqa/download.html',
        evaluation='https://cs.stanford.edu/people/dorarad/gqa/evaluate.html',
        executable='https://raw.githubusercontent.com/dorarad/gqa/9f88c1118598ba459266b9d3a6b74cd3db60bc44/evaluation/eval.py',
        executable_sha256='011dbe26a3de2e3843e0184e9660bc76041d40bd4aed1a3e5baba68379a97a5a'),
    'docvqa': dict(dataset='https://site.docvqa.org/datasets',
        evaluation='https://rrc.cvc.uab.es/?ch=17&com=tasks',
        executable='https://raw.githubusercontent.com/rubenpt91/MP-DocVQA-Framework/656d4e5a316720c83a1b1679fb39f424265655d7/metrics.py',
        executable_sha256='481fe17e8d443a6dcf852b8348fcf06b7973ac1e99336f230a749008e992bf70',
        original_metric='https://arxiv.org/html/1907.00490'),
}
SPATIAL_RELATIONS = ['left of', 'right of', 'above', 'below', 'behind', 'in front of',
                     'on top of', 'under', 'over', 'next to', 'near']


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''): h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def safe_output(path):
    path = path.resolve()
    return path


def exclusion_ids(path):
    if not path: return set()
    text = path.read_text()
    try:
        values = json.loads(text)
        if not isinstance(values, list) or not all(isinstance(v, (str,int)) for v in values):
            raise ValueError('Exclusion JSON must be an array of image IDs')
    except json.JSONDecodeError:
        values = [s.strip() for s in text.splitlines() if s.strip() and not s.startswith('#')]
    return {re.sub(r'^vg-', '', Path(str(v)).stem) for v in values}


def get_field(row, dotted):
    value = row
    for part in dotted.split('.'):
        if not isinstance(value, dict): return None
        value = value.get(part)
    return value


def gqa_group(row, spatial_relations):
    semantic = row.get('types', row.get('type', {})).get('semantic', '')
    if semantic in ('attribute', 'attr'): return 'attribute'
    if semantic in ('relation', 'rel'):
        for step in row.get('semantic', []):
            if not isinstance(step, dict): continue
            op = str(step.get('operation', '')).lower()
            argument = str(step.get('argument', '')).lower()
            if ('rel' in op) and any(re.search(r'(?<!\w)' + re.escape(rel) + r'(?!\w)', argument)
                                      for rel in spatial_relations):
                return 'spatial'
    return 'other'


def prepare_records(data, args, excluded, exact_filters):
    records, dropped = [], Counter()
    if args.benchmark == 'gqa':
        if not isinstance(data, dict) or 'data' in data:
            raise ValueError('GQA expects the official questionId -> question dictionary')
        for qid, row in data.items():
            if 'isBalanced' not in row and not args.gqa_assume_balanced:
                raise ValueError('Missing isBalanced; supply official balanced file or explicitly --gqa-assume-balanced')
            if row.get('isBalanced', True) is not True:
                dropped['unbalanced'] += 1; continue
            image_id = str(row['imageId'])
            if image_id in excluded:
                dropped['excluded_image'] += 1; continue
            group = gqa_group(row, args.spatial_relations)
            if args.gqa_group != 'all' and group != args.gqa_group:
                dropped['group_filter'] += 1; continue
            if any(get_field(row, key) not in allowed for key, allowed in exact_filters.items()):
                dropped['exact_metadata_filter'] += 1; continue
            records.append(dict(qid=str(qid), image_id=image_id, unit_id='gqa-image-'+image_id,
                image_path=args.images_root / (image_id + args.gqa_image_extension), query=row['question'],
                references=[row['answer']], group=group,
                official_metadata={key:row[key] for key in ('types','groups','semantic','isBalanced') if key in row}))
    else:
        # The official RRC Task-1 files name the dataset 'SP-DocVQA' (single-page DocVQA); older copies say 'docvqa'.
        if not isinstance(data, dict) or data.get('dataset_name', 'docvqa').lower() not in ('docvqa', 'sp-docvqa') or not isinstance(data.get('data'), list):
            raise ValueError('Expected official Single-Document DocVQA JSON object with data list')
        actual_split = data.get('dataset_split')
        aliases = {'val','validation'}
        if actual_split and actual_split not in aliases: raise ValueError('Expected validation annotations, found '+str(actual_split))
        for row in data['data']:
            if row.get('data_split', 'val') not in aliases: raise ValueError('Annotation contains a non-validation row')
            image_name = str(row['image'])
            image_id = Path(image_name).stem
            if image_id in excluded:
                dropped['excluded_image'] += 1; continue
            document = row.get('ucsf_document_id', row.get('docId'))
            if document is None:
                raise ValueError('DocVQA needs docId or ucsf_document_id for document-level inference')
            records.append(dict(qid=str(row['questionId']), image_id=image_id,
                unit_id='docvqa-document-'+str(document), image_path=args.images_root / image_name,
                query=row['question'], references=row['answers'], group='document_qa',
                official_metadata={key:row[key] for key in ('docId','ucsf_document_id','ucsf_document_page_no','question_types','data_split') if key in row}))
    for row in records:
        if not isinstance(row['query'], str) or not row['query'].strip(): raise ValueError('Empty question')
        if not row['references'] or not all(isinstance(x,str) and x for x in row['references']):
            raise ValueError('Nonempty reference answer strings required')
    return records, dict(dropped)


def import_dataset(args):
    from PIL import Image, ImageFilter, ImageOps, __version__ as pillow_version
    out = safe_output(args.output)
    manifest_path = out / 'manifest.jsonl'
    if manifest_path.exists() and not args.overwrite_manifest:
        raise ValueError('Manifest already exists; use a new directory or explicit --overwrite-manifest')
    data = json.loads(args.annotations.read_text())
    excluded = exclusion_ids(args.exclude_image_ids)
    exact_filters = json.loads(args.gqa_filter_json.read_text()) if args.gqa_filter_json else {}
    if not isinstance(exact_filters, dict) or any(not isinstance(v,list) for v in exact_filters.values()):
        raise ValueError('GQA filters must map dotted metadata paths to allowed-value lists')
    records, dropped = prepare_records(data,args,excluded,exact_filters)
    grouped=defaultdict(list)
    for row in records: grouped[row['unit_id']].append(row)
    rng=random.Random(args.seed)
    units=sorted(grouped);rng.shuffle(units)
    if args.max_units is not None: units=units[:args.max_units]
    selected=[]
    for unit in units:
        qq=sorted(grouped[unit],key=lambda r:r['qid']);rng.shuffle(qq)
        selected.extend(qq[:args.max_questions_per_unit] if args.max_questions_per_unit else qq)
    selected.sort(key=lambda r:(r['unit_id'],r['qid']))
    if not selected: raise ValueError('Selection is empty')
    out.mkdir(parents=True, exist_ok=True); (out/'images').mkdir(exist_ok=True)
    operation=dict(name=args.degradation, blur_radius=args.blur_radius if args.degradation=='blur' else None,
                   grayscale_mix=args.grayscale_mix if args.degradation=='grayscale' else None,
                   resize=False, pillow_version=pillow_version)
    operation_id=hashlib.sha256(json.dumps(operation,sort_keys=True).encode()).hexdigest()[:12]
    cache={};manifest=[]
    annotation_sha=sha(args.annotations)
    version=args.dataset_version or (data.get('dataset_version') if args.benchmark=='docvqa' else None)
    if not version: raise ValueError('Declare --dataset-version (DocVQA may provide it in JSON)')
    for record in selected:
        source=record['image_path'].resolve()
        if source not in cache:
            original_sha=sha(source)
            stem=original_sha[:20]
            clean=out/'images'/f'{stem}-clean.png'
            degraded=out/'images'/f'{stem}-{operation_id}.png'
            with Image.open(source) as opened:
                image=opened.convert('RGB')
                changed=(image.filter(ImageFilter.GaussianBlur(args.blur_radius)) if args.degradation=='blur'
                         else Image.blend(image,ImageOps.grayscale(image).convert('RGB'),args.grayscale_mix))
                assert image.size==changed.size
                image.save(clean);changed.save(degraded)
                w,h=image.size
            cache[source]=dict(paths={'a':clean.relative_to(out).as_posix(),'degraded_a':degraded.relative_to(out).as_posix()},
                metadata={name:dict(width=w,height=h,sha256=sha(path)) for name,path in [('a',clean),('degraded_a',degraded)]},
                original_sha256=original_sha,original_path=str(source))
        material=cache[source]
        manifest.append(dict(id=args.benchmark+'-'+record['qid'], unit_id=record['unit_id'],
            intervention_id=args.benchmark+'-'+record['qid'],task=args.benchmark+'_'+record['group'],
            query_role='task',query=record['query'],answer_instruction=args.answer_instruction,
            source_images=material['paths'],source_image_metadata=material['metadata'],
            reference_answers=record['references'],families=['recovery'],
            benchmark=dict(name=args.benchmark,version=str(version),split='validation',question_id=record['qid'],
                image_id=record['image_id'],cluster_id=record['unit_id'],group=record['group'],official_metadata=record['official_metadata']),
            source=dict(official=OFFICIAL[args.benchmark],annotations_sha256=annotation_sha,
                original_image_sha256=material['original_sha256'],original_image_path=material['original_path'],
                selection_seed=args.seed,degradation=operation,
                semantics='Same underlying clean image; all four quality combinations use its clean-reference answers')))
    with manifest_path.open('w') as f:
        for row in manifest:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
    metadata=dict(schema='followup-benchmark-import-v14', benchmark=args.benchmark,version=str(version),split='validation',
        official=OFFICIAL[args.benchmark],input_dataset_header={k:data[k] for k in ('dataset_name','dataset_split','dataset_version') if k in data},annotations_path=str(args.annotations.resolve()),annotations_sha256=annotation_sha,
        manifest_sha256=sha(manifest_path),selection_seed=args.seed,eligible_questions=len(records),eligible_units=len(grouped),
        selected_questions=len(manifest),selected_units=len(units),selected_images=len(cache),
        selection=dict(unit='image for GQA; source document for DocVQA',max_units=args.max_units,
            max_questions_per_unit=args.max_questions_per_unit,gqa_group=args.gqa_group,
            gqa_spatial_relations=args.spatial_relations,gqa_exact_metadata_filters=exact_filters,
            gqa_missing_balanced_flag_assumption=args.gqa_assume_balanced,excluded_image_ids=sorted(excluded),
            exclusion_file_sha256=sha(args.exclude_image_ids) if args.exclude_image_ids else None,
            output_independent=True),dropped=dropped,degradation=operation,
        readout='Freeform generation only; no artificial distractor or binary recoding',
        evaluation_note='GQA official exact equality is primary; lower+strip is a separately named sensitivity. DocVQA organizer-framework ANLS and strict-paper cutoff are distinguished by scorer.',
        limitation='Imported validation subset, not a full benchmark or a model result. Pixel sizes are preserved; processor token-grid identity is checked at runtime.')
    write_json(out/'import_metadata.json',metadata)
    return metadata


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--benchmark',choices=['gqa','docvqa'],required=True)
    p.add_argument('--annotations',type=Path,required=True);p.add_argument('--images-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--dataset-version')
    p.add_argument('--seed',type=int,default=20261028);p.add_argument('--max-units',type=int)
    p.add_argument('--max-questions-per-unit',type=int,default=2)
    p.add_argument('--exclude-image-ids',type=Path)
    p.add_argument('--gqa-group',choices=['all','attribute','spatial'],default='all')
    p.add_argument('--gqa-filter-json',type=Path);p.add_argument('--gqa-assume-balanced',action='store_true')
    p.add_argument('--gqa-image-extension',default='.jpg')
    p.add_argument('--spatial-relations',nargs='+',default=SPATIAL_RELATIONS)
    p.add_argument('--degradation',choices=['blur','grayscale'],default='blur')
    p.add_argument('--blur-radius',type=float,default=2.)
    p.add_argument('--grayscale-mix',type=float,default=.8)
    p.add_argument('--answer-instruction',default='Answer using only a short word or phrase.')
    p.add_argument('--overwrite-manifest',action='store_true')
    args=p.parse_args()
    if args.max_units is not None and args.max_units<1: p.error('--max-units must be positive')
    if args.max_questions_per_unit<0: p.error('Question cap must be >=0 (0 keeps all)')
    if not 0<=args.grayscale_mix<=1 or args.blur_radius<0: p.error('Invalid degradation magnitude')
    result=import_dataset(args)
    print(json.dumps({k:result[k] for k in ('benchmark','selected_questions','selected_units','selected_images','manifest_sha256')}))

if __name__=='__main__':main()
