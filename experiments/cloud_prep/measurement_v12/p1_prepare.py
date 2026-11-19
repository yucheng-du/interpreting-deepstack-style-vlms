"""Build reviewed P1 photo counterfactuals and validate their physical identity."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from cloud_prep.portable_data import _guard_path, sha256_file
from cloud_prep.measurement_v12.p1_data import ROOT,SCHEMA,SIZE,SEED,dump,fit_image,mirror_bbox,color_mask,recolor,_disjoint

def visual_signature(row):
    payload={'sources':{k:v['sha256'] for k,v in row['source_image_metadata'].items()},
             'roi':row['donor_roi_bbox_xyxy'],'mask':row['mask_sha256'],
             'changed_object_id':row['changed_object_id'],'intervention_id':row['intervention_id']}
    return hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()

def build(root=ROOT):
    from PIL import Image,ImageOps
    root=_guard_path(Path(root),write=True)
    review=json.loads((root/'visual_review.json').read_text())
    if review['status']!='complete':raise ValueError('Actual image review required before build')
    candidates={r['image_id']:r for r in json.loads((root/'pixel_candidates.json').read_text())['candidates']}
    selected=review['selected'];ids=[s['image_id'] for s in selected]
    if len(ids)!=len(set(ids)):raise ValueError('Duplicate source photo')
    excluded=set(json.loads((root/'excluded_sources.json').read_text())['source_ids'])
    if excluded & set(ids):raise ValueError('Previously used photo selected')
    (root/'images').mkdir(exist_ok=True);(root/'masks').mkdir(exist_ok=True)
    rows=[]
    for number,sel in enumerate(selected):
        c=deepcopy(candidates[sel['image_id']]);sid=f"vg-{c['image_id']}";original=root/'sources'/f"{c['image_id']}.jpg"
        with Image.open(original) as raw:main,fit=fit_image(raw)
        objects=c['objects']
        for index,o in enumerate(objects):
            o['query_description']=sel.get('query_descriptions',{}).get(str(index),o['name'])
            o['mask_clip_polygon']=sel.get('mask_clip_polygons',{}).get(str(index))
        for edit_index,target in enumerate(objects):
            iid=f'{sid}-edit-{edit_index}'
            mask=color_mask(main,target['bbox_xyxy'],target['annotation_color'],target['mask_clip_polygon'])
            maskpath=Path('masks')/f'{iid}.png';mask.save(root/maskpath)
            maskcount=sum(mask.histogram()[1:])
            images={'d_a':recolor(main,mask,'red'),'d_b':recolor(main,mask,'blue')}
            specs={f'd_{a}':{'type':'donor','strength':1.,'layout':'M0','assignment':a} for a in ('a','b')}
            for layout in ('M0','M1'):
                arranged=main if layout=='M0' else ImageOps.mirror(main)
                for strength in (8,100):
                    key=f'{layout.lower()}_s{strength:03d}'
                    images[key]=Image.blend(Image.new('RGB',SIZE,(127,127,127)),arranged,strength/100)
                    specs[key]={'type':'main','strength':strength/100,'layout':layout,'assignment':None}
            paths={};metadata={}
            for key,image in images.items():
                relative=Path('images')/f'{iid}-{key}.png';image.save(root/relative)
                paths[key]=str(relative);metadata[key]={'sha256':sha256_file(root/relative),'bytes':(root/relative).stat().st_size,'width':SIZE[0],'height':SIZE[1]}
            for query_index,obj in enumerate(objects):
                role='changed' if query_index==edit_index else 'unchanged';answer=('red','blue').index(obj['annotation_color'])
                source_answers={k:answer for k in images}
                if role=='changed':source_answers.update(d_a=0,d_b=1)
                row={'schema':SCHEMA,'family':'natural_selectivity','task':'natural_scene_color',
                    'id':f'{iid}-query-{query_index}','scene_id':sid,'unit_id':sid,'intervention_id':iid,
                    'attribute':'color','generation_group':number%4,'query_role':role,'query_type':'identity','query_target':role,
                    'query_description':obj['query_description'],'query':f"What color is the {obj['query_description']}?",
                    'choices':['red','blue'],'option_labels':['(A)','(B)'],'answer_index':answer,'answer_label':('(A)','(B)')[answer],
                    'direction_a_index':0,'direction_b_index':1,'changed_object':target['name'],
                    'changed_object_id':target['object_id'],'changed_index':edit_index,
                    'queried_object':obj['name'],'queried_object_id':obj['object_id'],'queried_index':query_index,
                    'target_object':obj['name'],'target_index':query_index,'target_main_slot':0,
                    'target_main_slot_by_layout':{'M0':0,'M1':0},'source_roi_slot':0,
                    'donor_roi_bbox_xyxy':target['bbox_xyxy'],'donor_roi_image_size':list(SIZE),
                    'donor_roi_coordinate_system':'pixel_half_open_xyxy','donor_reference_layout':'M0',
                    'tile_grid':[1,1],'geometry':'natural_full',
                    'changed_bbox_by_layout':{'M0':target['bbox_xyxy'],'M1':mirror_bbox(target['bbox_xyxy'])},
                    'queried_bbox_by_layout':{'M0':obj['bbox_xyxy'],'M1':mirror_bbox(obj['bbox_xyxy'])},
                    'source_images':paths.copy(),'source_image_metadata':deepcopy(metadata),'source_specs':deepcopy(specs),
                    'source_answer_indices':source_answers,'mask_path':str(maskpath),'mask_sha256':sha256_file(root/maskpath),
                    'mask_pixels':maskcount,'mask_clip_polygon':target['mask_clip_polygon'],'has_unchanged_query':True,
                    'fit':fit,'source_photo':str(original.relative_to(root)),
                    'source':{'revision':SCHEMA,'construction':'whole_scene_bbox_color_region_edit',
                              'source_image_sha256':c['source_sha256'],'image_id':c['image_id'],
                              'object_id':target['object_id'],'gqa_split':c['split'],'seed':SEED,'split':'new_source_photos_v12'},
                    'record_fields':{'native_color_annotation':obj['annotation_color'],'query_object_id':obj['object_id'],
                                     'visual_review':'accepted','review_note':sel.get('note','')},
                    'dose_contract':{'calibration_group':iid,'canonical_query_id':f'{iid}-query-{edit_index}',
                                     'canonical_readout':'direct','fixed_across_queries_and_readouts':['ROI','selected_tokens','a_baseline','delta','k'],
                                     'anchor':'a-side post-entry baseline-state; actual forward doses saved'}}
                row['prompt']=row['query']+'\n(A) red\n(B) blue\n'
                row['visual_source_signature']=visual_signature(row);rows.append(row)
    manifest=root/'manifest.jsonl';manifest.write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in rows))
    report=validate(manifest)
    report.update(visual_review_status='complete',visual_review_sha256=sha256_file(root/'visual_review.json'),
                  model_results=False,mask_semantics='largest 8-connected HSV color region within bbox, with reviewed optional polygon; not instance segmentation',
                  category_counts=dict(Counter(o['name'] for s in selected for o in candidates[s['image_id']]['objects'])))
    dump(root/'dataset.json',report)
    return report

def contact_sheets(root=ROOT):
    """Final, actual-model-input QA sheets: original, A red/blue, B red/blue, masks."""
    from PIL import Image,ImageDraw
    import numpy as np
    root=_guard_path(Path(root),write=True);rows=load_manifest(root/'manifest.jsonl')
    interventions,photos=validate_query_groups(rows);out=root/'qa';out.mkdir(exist_ok=True)
    scenes=list(photos);paths=[];index=[]
    for start in range(0,len(scenes),4):
        sheet=Image.new('RGB',(1536,4*300),(247,247,247));draw=ImageDraw.Draw(sheet)
        for j,sid in enumerate(scenes[start:start+4]):
            members=photos[sid];edits=[next(r for r in members if r['changed_index']==i) for i in (0,1)]
            first,second=edits;main=Image.open(first['source_images']['m0_s100']).convert('RGB');overlay=main.copy();d=ImageDraw.Draw(overlay)
            masks=[np.asarray(Image.open(r['mask_path']))>0 for r in edits]
            for letter,row,color in zip('AB',edits,('yellow','lime')):
                d.rectangle(row['donor_roi_bbox_xyxy'],outline=color,width=3);d.text(tuple(row['donor_roi_bbox_xyxy'][:2]),letter,fill=color)
            arr=np.zeros((384,384,3),dtype='uint8');arr[masks[0]]=[255,255,0];arr[masks[1]]=[0,255,0]
            panels=[overlay,*[Image.open(r['source_images'][key]).convert('RGB') for r in edits for key in ('d_a','d_b')],Image.fromarray(arr)]
            desc=[next(r for r in members if r['queried_index']==i)['query_description'] for i in (0,1)]
            draw.text((4,j*300+2),f'{start+j+1:02d} {sid} | A: {desc[0]} | B: {desc[1]}',fill='black')
            for k,(label,panel) in enumerate(zip(('Original A/B boxes','A red','A blue','B red','B blue','Masks A/B'),panels)):
                draw.text((k*256+4,j*300+19),label,fill='black');sheet.paste(panel.resize((256,256)),(k*256,j*300+38))
            index.append({'scene_id':sid,'sheet':f'selected_{start//4:02d}.jpg','row':j,'queries':desc})
        path=out/f'selected_{start//4:02d}.jpg';sheet.save(path,quality=95);paths.append(str(path))
    dump(out/'index.json',{'schema':SCHEMA,'sheets':paths,'photos':index,'status':'rendered from final manifest model inputs; all originals/edits already directly reviewed'})
    return {'sheets':paths,'photos':len(scenes)}

def validate_query_groups(rows):
    if not rows or len({r['id'] for r in rows})!=len(rows):raise ValueError('Unique nonempty rows required')
    interventions=defaultdict(list);photos=defaultdict(list)
    for row in rows:interventions[row['intervention_id']].append(row);photos[row['scene_id']].append(row)
    shared=('scene_id','unit_id','donor_roi_bbox_xyxy','visual_source_signature','mask_sha256','source_images','source_image_metadata','dose_contract')
    for members in interventions.values():
        if len(members)!=2 or {r['query_role'] for r in members}!={'changed','unchanged'}:raise ValueError('Each edit needs target and control queries')
        if len({r['queried_object_id'] for r in members})!=2:raise ValueError('Distinct queried objects required')
        for field in shared:
            if members[0][field]!=members[1][field]:raise ValueError(f'Same intervention identity broken: {field}')
    for members in photos.values():
        if len(members)!=4 or len({r['intervention_id'] for r in members})!=2:raise ValueError('Two edits / four queries per independent photo required')
        if len({r['changed_object_id'] for r in members})!=2:raise ValueError('Each object edited once')
    return interventions,photos

def load_manifest(manifest=ROOT/'manifest.jsonl',image_root=None,check_hashes=True):
    manifest=_guard_path(Path(manifest));root=_guard_path(Path(image_root)) if image_root else manifest.parent
    rows=[json.loads(s) for s in manifest.read_text().splitlines() if s];checked={}
    for row in rows:
        if row['schema']!=SCHEMA or row['family']!='natural_selectivity':raise ValueError('Wrong P1 schema')
        if row['visual_source_signature']!=visual_signature(row):raise ValueError('Visual signature mismatch')
        if row['choices']!=['red','blue'] or (row['direction_a_index'],row['direction_b_index'])!=(0,1):raise ValueError('Fixed red-to-blue estimand required')
        for key,relative in row['source_images'].items():
            rel=Path(relative)
            if rel.is_absolute() or '..' in rel.parts:raise ValueError('Portable contained path required')
            path=_guard_path(root/rel)
            if not path.is_file() or not path.is_relative_to(root):raise ValueError('Missing/escaping image')
            if check_hashes:
                if path not in checked:checked[path]=(path.stat().st_size,sha256_file(path))
                meta=row['source_image_metadata'][key]
                if checked[path]!=(meta['bytes'],meta['sha256']):raise ValueError('Image bytes/hash mismatch')
            row['source_images'][key]=str(path)
        for field in ('mask_path','source_photo'):
            rel=Path(row[field])
            if rel.is_absolute() or '..' in rel.parts:raise ValueError('Portable contained aux path required')
            path=_guard_path(root/rel)
            if not path.is_file() or not path.is_relative_to(root):raise ValueError('Missing/escaping auxiliary')
            row[field]=str(path)
    validate_query_groups(rows)
    return rows

def validate(manifest=ROOT/'manifest.jsonl'):
    from PIL import Image
    import numpy as np
    rows=load_manifest(manifest);interventions,photos=validate_query_groups(rows)
    for iid,members in interventions.items():
        row=members[0];arrays={key:np.asarray(Image.open(path).convert('RGB')) for key,path in row['source_images'].items()}
        mask=np.asarray(Image.open(row['mask_path']))>0
        if sha256_file(row['mask_path'])!=row['mask_sha256'] or int(mask.sum())!=row['mask_pixels']:raise ValueError('Mask bytes/pixels mismatch')
        x0,y0,x1,y1=row['donor_roi_bbox_xyxy'];roi=np.zeros(mask.shape,bool);roi[y0:y1,x0:x1]=True
        if np.any(mask&~roi):raise ValueError('Mask outside ROI')
        if not np.array_equal(np.any(arrays['d_a']!=arrays['d_b'],axis=2),mask):raise ValueError('a/b difference != mask')
        if any(np.any(arrays[k][~mask]!=arrays['m0_s100'][~mask]) for k in ('d_a','d_b')):raise ValueError('Unedited RGB pixels changed')
        if not np.array_equal(arrays['m1_s100'],arrays['m0_s100'][:,::-1]):raise ValueError('Mirror identity failed')
        with Image.open(row['source_photo']) as raw:expected,fit=fit_image(raw)
        if not np.array_equal(np.asarray(expected),arrays['m0_s100']) or fit!=row['fit']:raise ValueError('Full photograph preservation failed')
        for layout in ('m0','m1'):
            expected=Image.blend(Image.new('RGB',SIZE,(127,127,127)),Image.fromarray(arrays[f'{layout}_s100']),.08)
            if not np.array_equal(np.asarray(expected),arrays[f'{layout}_s008']):raise ValueError('Weak context identity failed')
        control=next(r for r in members if r['query_role']=='unchanged')
        if not _disjoint(row['donor_roi_bbox_xyxy'],control['queried_bbox_by_layout']['M0']):raise ValueError('Control bbox overlaps changed ROI')
    return {'schema':SCHEMA,'status':'passed','manifest':str(manifest),'manifest_sha256':sha256_file(manifest),
            'items':len(rows),'independent_source_photos':len(photos),'interventions':len(interventions),
            'changed_queries':sum(r['query_role']=='changed' for r in rows),
            'unchanged_queries':sum(r['query_role']=='unchanged' for r in rows),
            'pixel_pairs_verified':len(interventions),'canvas':list(SIZE),
            'independence_unit':'source photo scene_id; two edits, queries, conditions and mappings are repeats'}

def select_panel(rows,panel='main'):
    if panel=='main':return rows
    if panel!='smoke':raise ValueError('Unknown panel')
    scene=sorted({r['scene_id'] for r in rows})[0]
    return [r for r in rows if r['scene_id']==scene]

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('build','validate','contact-sheets'))
    p.add_argument('--root',type=Path,default=ROOT);a=p.parse_args(argv)
    print(json.dumps(build(a.root) if a.command=='build' else contact_sheets(a.root) if a.command=='contact-sheets' else validate(a.root/'manifest.jsonl'),indent=2))

if __name__=='__main__':main()
