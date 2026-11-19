"""Portable, tensor-free full-source competition stimuli with defined relation queries."""
from __future__ import annotations
import argparse
from collections import defaultdict
from copy import deepcopy
import hashlib
from itertools import product
import json
from pathlib import Path
import os as _path_os
DATA_ROOT = Path(_path_os.environ.get("DS_DATA_ROOT", "work")).expanduser().resolve()
from cloud_prep.portable_data import _guard_path, sha256_file
from cloud_prep.measurement_v8 import mechanism_data as v8

ROW_SCHEMA = 'full-source-data-v10'
FAMILY = 'full_source'
DEFAULT_ROOT = (DATA_ROOT / 'datasets/full-control-v10')
DEFAULT_SOURCE = v8.DEFAULT_ROOT / 'manifest.jsonl'
SIZE = (576, 192)
STRENGTHS = (0, 8, 100)
LAYOUTS = ('M0', 'M1')


def source_name(layout, strength, assignment):
    if layout not in LAYOUTS or strength not in STRENGTHS or assignment not in ('a','b'):
        raise ValueError('Unknown full-source image specification')
    return f's_{layout}_{strength:03d}_{assignment}'


def source_specs():
    return {source_name(m,s,a): {'layout':m,'strength':s/100,'assignment':a,'type':'full_source'}
            for m,s,a in product(LAYOUTS,STRENGTHS,('a','b'))}


def object_slot(scene, obj, layout):
    return 2 * (scene['object_to_slot'][obj] ^ (layout == 'M1'))


def bbox(scene, obj, layout='M0'):
    slot = object_slot(scene,obj,layout)
    original_slot = scene['object_to_slot'][obj]
    x = scene['centers'][original_slot][0] - 192*original_slot + 192*slot
    y = round(scene['centers'][original_slot][1]*.75)
    r = round(scene['radius']*.75)
    return [x-r,y-r,x+r+1,y+r+1]


def render(scene, changed, spec):
    from PIL import Image, ImageDraw
    image = Image.new('RGB',SIZE,(scene['background'],)*3)
    draw = ImageDraw.Draw(image)
    # The central reference remains achromatic and identical at every strength.
    draw.rectangle((280,63,296,129),fill=(110,110,110))
    draw.rectangle((255,88,321,104),fill=(110,110,110))
    for obj in scene['objects']:
        value = scene['object_values_a'][obj] ^ (obj == changed and spec['assignment'] == 'b')
        alpha = spec['strength']
        color = tuple(int(150*(1-alpha)+x*alpha) for x in v8.COLORS[value])
        box = bbox(scene,obj,spec['layout'])
        v8._shape(draw,obj,[box[0],box[1],box[2]-1,box[3]-1],fill=color)
    return image


def referred_object(row, spec):
    if row['query_type'] == 'identity':
        return row['queried_object']
    return next(obj for obj in row['scene']['objects'] if object_slot(row['scene'],obj,spec['layout']) == row['query_slot'])


def source_answers(row):
    return {name: None if spec['strength'] == 0 else
            row['scene']['object_values_a'][referred_object(row,spec)] ^
            (spec['assignment'] == 'b' and referred_object(row,spec) == row['changed_object'])
            for name,spec in row['source_specs'].items()}


def visual_signature(row):
    payload = {'images':{k:v['sha256'] for k,v in row['source_image_metadata'].items()},
               'changed_object':row['changed_object'],'roi':row['donor_roi_bbox_xyxy'],'tile_grid':[1,3]}
    return hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def query_text(row):
    if row['query_type'] == 'identity':
        return f"What color is the {row['queried_object']}?"
    return f"What color is the object to the {'left' if row['query_slot']==0 else 'right'} of the gray cross?"


def build(output=DEFAULT_ROOT, source_manifest=DEFAULT_SOURCE):
    source_manifest = _guard_path(Path(source_manifest))
    originals = v8.load_manifest(source_manifest)
    output = _guard_path(Path(output),write=True)
    output.mkdir(parents=True,exist_ok=False)
    (output/'images').mkdir()
    rows=[]
    specs=source_specs()
    for old in originals:
        scene=old['scene']; changed=old['target_object']; iid=f"full10-{old['scene_id']}-change-{changed}"
        images={}; metadata={}
        for name,ss in specs.items():
            relative=f'images/{iid}-{name}.png'; path=output/relative
            with render(scene,changed,ss) as image:
                image.save(path)
            images[name]=relative
            metadata[name]={'width':SIZE[0],'height':SIZE[1],'bytes':path.stat().st_size,
                            'sha256':sha256_file(path),'source_field':'image_1'}
        for query_type,queried_index in product(('identity','relation'),range(2)):
            queried=scene['objects'][queried_index]; value=scene['object_values_a'][queried]
            row={'schema':ROW_SCHEMA,'family':FAMILY,'id':f'{iid}-{query_type}-{queried}',
                'scene_id':old['scene_id'],'intervention_id':iid,'task':'full_source_color','attribute':'color',
                **{k:old[k] for k in ('shape_pair','layout','assignment','replicate','generation_group')},
                'changed_object':changed,'changed_index':old['target_index'],
                'queried_object':queried,'queried_index':queried_index,'query_type':query_type,
                'query_target':'changed' if queried==changed else 'unchanged',
                'query_role':'changed' if queried==changed else 'unchanged',
                'query_slot':object_slot(scene,queried,'M0'),
                'target_object':queried,'target_index':queried_index,'target_main_slot':object_slot(scene,queried,'M0'),
                'target_main_slot_by_layout':{m:object_slot(scene,queried,m) for m in LAYOUTS},
                'direction_a_index':scene['object_values_a'][changed],'direction_b_index':1-scene['object_values_a'][changed],
                'answer_index':value,'answer_label':('(A)','(B)')[value],'choices':['red','blue'],'option_labels':['(A)','(B)'],
                'tile_grid':[1,3],'donor_roi_bbox_xyxy':bbox(scene,changed),'donor_roi_image_size':list(SIZE),
                'donor_roi_coordinate_system':'pixel_half_open_xyxy','scene':deepcopy(scene),
                'source_specs':deepcopy(specs),'source_images':deepcopy(images),'source_image_metadata':deepcopy(metadata),
                'record_fields':{'source_cluster_id':old['scene_id']},
                'source':{'dataset':'DeepStack full-source controlled composition','revision':ROW_SCHEMA,
                    'parent_revision':v8.ROW_SCHEMA,'parent_manifest_sha256':sha256_file(source_manifest),
                    'seed':old['source']['seed'],'split':'exploratory','generator_sha256':sha256_file(Path(__file__)),
                    'reuse_scope':'v8 scene parameters, new 1x3 rendering with fixed center cross; not independent replication'}}
            row['query']=query_text(row); row['prompt']=row['query']+'\n(A) red\n(B) blue\n'
            row['source_answer_indices']=source_answers(row)
            row['referred_object_by_source']={n:referred_object(row,s) for n,s in specs.items()}
            row['visual_source_signature']=visual_signature(row)
            rows.append(row)
    manifest=output/'manifest.jsonl'
    manifest.write_text(''.join(json.dumps(row,sort_keys=True)+'\n' for row in rows))
    result=validate(manifest)
    (output/'dataset.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def validate_query_groups(rows):
    if not rows or len({r['id'] for r in rows})!=len(rows):
        raise ValueError('Nonempty unique full-source rows required')
    groups=defaultdict(list)
    for row in rows: groups[row['intervention_id']].append(row)
    for members in groups.values():
        if len(members)!=4 or {(r['query_type'],r['query_target']) for r in members}!={(t,q) for t,q in product(('identity','relation'),('changed','unchanged'))}:
            raise ValueError('Each intervention requires both identity and both relation queries')
        for field in ('scene_id','changed_object','changed_index','direction_a_index','direction_b_index','visual_source_signature','generation_group','source_images','source_specs','source_image_metadata'):
            if field in members[0] and any(r[field]!=members[0][field] for r in members):
                raise ValueError(f'Query changed visual contract: {field}')


def load_manifest(manifest,image_root=None,*,check_hashes=True):
    manifest=_guard_path(Path(manifest)); root=_guard_path(Path(image_root)) if image_root else manifest.parent
    rows=[json.loads(s) for s in manifest.read_text().splitlines() if s.strip()]; checked={}
    for row in rows:
        if row['schema']!=ROW_SCHEMA or row['family']!=FAMILY or row['source_specs']!=source_specs():
            raise ValueError('Full-source schema/specification mismatch')
        expected_scene=v8.scene_spec('color',*(row[k] for k in ('shape_pair','layout','assignment','replicate')),seed=row['source']['seed'])
        if row['scene']!=expected_scene: raise ValueError('Parent scene parameters changed')
        scene=row['scene']; changed=scene['objects'][row['changed_index']]; queried=scene['objects'][row['queried_index']]
        if row['changed_object']!=changed or row['queried_object']!=queried: raise ValueError('Object identity mismatch')
        if row['query_type'] not in ('identity','relation') or row['query_target']!=('changed' if changed==queried else 'unchanged'):
            raise ValueError('Query role mismatch')
        if row['query_slot']!=object_slot(scene,queried,'M0') or row['query']!=query_text(row) or row['prompt']!=row['query']+'\n(A) red\n(B) blue\n':
            raise ValueError('Query semantics mismatch')
        if row['choices']!=['red','blue'] or row['option_labels']!=['(A)','(B)']:
            raise ValueError('Readout choice mismatch')
        if row['direction_a_index']!=scene['object_values_a'][changed] or row['direction_b_index']!=1-row['direction_a_index']:
            raise ValueError('a/b must follow changed-object values, never entry role')
        if row['source_answer_indices']!=source_answers(row) or row['referred_object_by_source']!={n:referred_object(row,s) for n,s in row['source_specs'].items()}:
            raise ValueError('Source-aware relation answers mismatch')
        if row['tile_grid']!=[1,3] or row['donor_roi_image_size']!=list(SIZE) or row['donor_roi_bbox_xyxy']!=bbox(scene,changed):
            raise ValueError('Geometry mismatch')
        if set(row['source_images'])!=set(source_specs()) or set(row['source_image_metadata'])!=set(source_specs()) or row['visual_source_signature']!=visual_signature(row):
            raise ValueError('Visual source contract mismatch')
        for name,rel in row['source_images'].items():
            relative=Path(rel)
            if relative.is_absolute() or '..' in relative.parts: raise ValueError('Contained portable paths required')
            path=_guard_path(root/relative)
            if not path.is_relative_to(root) or not path.is_file(): raise ValueError('Missing/escaping image')
            meta=row['source_image_metadata'][name]
            if (meta['width'],meta['height'])!=SIZE: raise ValueError('Canvas mismatch')
            if check_hashes:
                if path not in checked: checked[path]=(path.stat().st_size,sha256_file(path))
                if checked[path]!=(meta['bytes'],meta['sha256']): raise ValueError('Image hash mismatch')
            row['source_images'][name]=str(path)
    validate_query_groups(rows)
    return rows


def select_panel(rows,panel):
    if panel=='main': return rows
    if panel!='smoke': raise ValueError('Unknown panel')
    selected=[r for r in rows if r['replicate']==0 and (r['shape_pair'],r['layout'],r['assignment']) in ((0,0,0),(1,1,1))]
    validate_query_groups(selected)
    return selected


def validate(manifest,image_root=None):
    from PIL import Image, ImageChops, ImageDraw
    rows=load_manifest(manifest,image_root); interventions={r['intervention_id']:r for r in rows}; checked=0
    for row in interventions.values():
        for layout,strength in product(LAYOUTS,STRENGTHS):
            a_name,b_name=[source_name(layout,strength,a) for a in ('a','b')]
            with Image.open(row['source_images'][a_name]) as a,Image.open(row['source_images'][b_name]) as b:
                for name,image in ((a_name,a),(b_name,b)):
                    with render(row['scene'],row['changed_object'],row['source_specs'][name]) as expected:
                        if ImageChops.difference(image,expected).getbbox(): raise ValueError('Rendered pixels disagree with declaration')
                changed=ImageChops.difference(a,b).convert('L').point(lambda x:255 if x else 0)
                mask=Image.new('L',SIZE,0)
                if strength:
                    box=bbox(row['scene'],row['changed_object'],layout)
                    v8._shape(ImageDraw.Draw(mask),row['changed_object'],[box[0],box[1],box[2]-1,box[3]-1],fill=255)
                if ImageChops.difference(changed,mask).getbbox(): raise ValueError('a/b changes outside changed object, or alpha0 is nonidentical')
                changed.close();mask.close(); checked+=1
    images={p for r in rows for p in r['source_images'].values()}
    return {'schema':ROW_SCHEMA,'status':'passed','model_results':False,'manifest':str(manifest),
        'manifest_sha256':sha256_file(manifest),'items':len(rows),'scene_templates':len({r['scene_id'] for r in rows}),
        'visual_interventions':len(interventions),'pixel_exact_ab_pairs':checked,'unique_image_paths':len(images),
        'image_bytes':sum(Path(p).stat().st_size for p in images),'query_types':['identity','relation'],
        'source_count_per_query':len(source_specs()),'independence_unit':'scene_id; v8 parameter reuse, not new independent replication',
        'alpha0_contract':'a/b identical achromatic targets; source native attribute truth unavailable',
        'relation_contract':'left/right of fixed gray cross, source-aware answer at every layout'}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('build','validate'))
    p.add_argument('--output',type=Path,default=DEFAULT_ROOT);p.add_argument('--source-manifest',type=Path,default=DEFAULT_SOURCE)
    p.add_argument('--manifest',type=Path);p.add_argument('--image-root',type=Path);a=p.parse_args(argv)
    if a.command=='validate' and a.manifest is None:p.error('--manifest required')
    print(json.dumps(build(a.output,a.source_manifest) if a.command=='build' else validate(a.manifest,a.image_root),indent=2))
if __name__=='__main__':main()
