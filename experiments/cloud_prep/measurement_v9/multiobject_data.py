"""New three-object scenes, exact tile cycles, and disjoint-source GQA recompositions.

CPU/Pillow/numpy only. Large artifacts stay on Research; cached GQA inputs are read-only.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import os as _path_os
DATA_ROOT = Path(_path_os.environ.get("DS_DATA_ROOT", "work")).expanduser().resolve()
import random
import shutil
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cloud_prep.portable_data import _guard_path, sha256_file
from cloud_prep.measurement_v9.multiobject_design import main_source, tile_permutation

ROW_SCHEMA = 'three-object-color-v9'
FAMILY = 'multiobject_color'
SEED = 2026101309
DEFAULT_ROOT = (DATA_ROOT / 'datasets/multiobject-v9')
GQA_ROOT = (DATA_ROOT / 'datasets/gqa-official')
OLD_MANIFEST = (DATA_ROOT / 'datasets/natural-object-v7/manifest.jsonl')
COLORS = ((214, 43, 49), (39, 87, 207))
GEOMETRIES = {'three_columns': [1, 3], 'two_dimensional': [2, 2]}
TILE = 192
TRIPLES = (('car', 'bus', 'umbrella'), ('car', 'train', 'bag'), ('truck', 'bus', 'chair'),
    ('truck', 'train', 'umbrella'), ('car', 'chair', 'bag'), ('bus', 'train', 'umbrella'),
    ('truck', 'chair', 'bag'), ('car', 'bus', 'train'))


def source_specs(strengths=(0, 8)):
    out = {main_source(layout, strength): dict(type='main', layout=layout, strength=strength / 100,
        assignment='a', pose='same', transform='cycle_forward' if layout == 'M1' else 'identity')
        for strength in sorted(set(strengths) | {100}) for layout in ('M0', 'M1')}
    out.update({f'd_{v}': dict(type='donor', layout='M0', strength=1., assignment=v,
        pose='same', transform='identity') for v in ('a', 'b')})
    return out


def slot_box(slot, grid):
    return [(slot % grid[1]) * TILE, (slot // grid[1]) * TILE,
            (slot % grid[1] + 1) * TILE, (slot // grid[1] + 1) * TILE]


def permute_image(image, grid, permutation):
    from PIL import Image
    if image.size != (grid[1] * TILE, grid[0] * TILE):
        raise ValueError('Image does not have exact equal-tile dimensions')
    out = Image.new('RGB', image.size)
    for dest, src in enumerate(tile_permutation(grid, permutation)):
        with image.crop(slot_box(src, grid)) as tile:
            out.paste(tile, tuple(slot_box(dest, grid)[:2]))
    return out


def _synthetic_scene(index, geometry, seed=SEED):
    rng = random.Random(seed + index * 7919 + list(GEOMETRIES).index(geometry) * 104729)
    groups = (('circle', 'square', 'triangle'), ('diamond', 'ellipse', 'pentagon'))
    objects = list(groups[(index // 8) % 2]); slots = list(range(3)); rng.shuffle(slots)
    assignment = index % 2
    return dict(objects=objects, object_to_slot=dict(zip(objects, slots)),
        object_values_a={o: (i + assignment) % 2 for i, o in enumerate(objects)},
        boxes={o: [rng.randint(30, 44), rng.randint(28, 42), rng.randint(147, 164), rng.randint(146, 164)] for o in objects},
        background=rng.randint(217, 231), seed=seed, scene_index=index)


def _draw(draw, kind, b, color):
    x0, y0, x1, y1 = b; cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    if kind in ('circle', 'ellipse'): draw.ellipse(b, fill=color)
    elif kind == 'square': draw.rectangle(b, fill=color)
    elif kind == 'triangle': draw.polygon([(cx, y0), (x1, y1), (x0, y1)], fill=color)
    elif kind == 'diamond': draw.polygon([(cx, y0), (x1, cy), (cx, y1), (x0, cy)], fill=color)
    elif kind == 'pentagon':
        import math
        draw.polygon([(cx + (x1 - x0) / 2 * math.cos(-math.pi / 2 + k * 2 * math.pi / 5),
                       cy + (y1 - y0) / 2 * math.sin(-math.pi / 2 + k * 2 * math.pi / 5)) for k in range(5)], fill=color)
    else: raise ValueError(kind)


def synthetic_render(scene, grid, changed_object=None, strength=1., layout='M0'):
    from PIL import Image, ImageDraw
    image = Image.new('RGB', (grid[1] * TILE, grid[0] * TILE), (scene['background'],) * 3)
    draw = ImageDraw.Draw(image)
    for obj in scene['objects']:
        value = scene['object_values_a'][obj] ^ (obj == changed_object)
        color = tuple(int(150 * (1 - strength) + x * strength) for x in COLORS[value])
        dx, dy = slot_box(scene['object_to_slot'][obj], grid)[:2]
        b = scene['boxes'][obj]
        _draw(draw, obj, [b[0] + dx, b[1] + dy, b[2] + dx, b[3] + dy], color)
    if layout == 'M1':
        out = permute_image(image, grid, 'cycle_forward'); image.close(); return out
    return image


def _save(image, output, relative):
    path = output / relative; path.parent.mkdir(parents=True, exist_ok=True); image.save(path)
    return relative, dict(width=image.width, height=image.height, bytes=path.stat().st_size,
        sha256=sha256_file(path), source_field='image_1')


def _rows_for_scene(output, sid, unit, family, geometry, scene, render, strengths, generation_group, provenance, natural=None):
    grid = GEOMETRIES[geometry]; sources = source_specs(strengths); paths, metadata = {}, {}
    for name, spec in sources.items():
        if spec['type'] == 'main':
            with render(None, spec['strength'], spec['layout']) as im:
                paths[name], metadata[name] = _save(im, output, f'images/{sid}-{name}.png')
    paths['d_a'], metadata['d_a'] = paths[main_source('M0', 100)], metadata[main_source('M0', 100)]
    rows = []
    for i, obj in enumerate(scene['objects']):
        pp, mm = deepcopy(paths), deepcopy(metadata)
        with render(obj, 1., 'M0') as im:
            pp['d_b'], mm['d_b'] = _save(im, output, f'images/{sid}-{obj}-d_b.png')
        value, slot = scene['object_values_a'][obj], scene['object_to_slot'][obj]
        moved = (slot + 1) % 3
        query = f'What color is the {obj}?'
        fields = dict(unit_id=unit, source_cluster_id=unit, panel_family=family, geometry=geometry,
            construction='exact_procedural_color_counterfactual' if family == 'synthetic' else 'controlled_recomposition')
        if natural:
            fields.update(source_photos='|'.join(sorted(c['image_id'] for c in natural['crops'].values())),
                          source_crops='|'.join(sorted(c['candidate_id'] for c in natural['crops'].values())))
        row = dict(schema=ROW_SCHEMA, family=FAMILY, id=f'{sid}-{obj}', scene_id=sid, **fields,
            task='multiobject_color', attribute='color', objects=scene['objects'], scene=scene,
            generation_group=generation_group, replicate=generation_group, tile_grid=grid, occupied_slots=[0, 1, 2],
            target_object=obj, changed_object=obj, queried_object=obj, target_index=i, target_main_slot=slot,
            target_main_slot_by_layout={'M0': slot, 'M1': moved}, query=query,
            prompt=query + '\n(A) red\n(B) blue\n', choices=['red', 'blue'], option_labels=['(A)', '(B)'],
            answer_index=value, answer_label=['(A)', '(B)'][value], direction_a_index=value, direction_b_index=1-value,
            donor_roi_bbox_xyxy=slot_box(slot, grid), donor_roi_coordinate_system='pixel_half_open_xyxy',
            donor_roi_image_size=[grid[1]*TILE, grid[0]*TILE], donor_reference_layout='M0',
            source_specs=sources, source_images=pp, source_image_metadata=mm,
            source_answer_indices={n: None if s['strength'] == 0 else value ^ (s['assignment'] == 'b') for n, s in sources.items()},
            source_object_slots={n: moved if s['layout'] == 'M1' else slot for n, s in sources.items()},
            record_fields=fields, source=provenance)
        if natural: row['natural'] = natural
        rows.append(row)
    return rows


def _finish(output, rows):
    manifest = output / 'manifest.jsonl'
    manifest.write_text(''.join(json.dumps(r, sort_keys=True) + '\n' for r in rows))
    result = validate(manifest)
    (output / 'dataset.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def build_synthetic(output=DEFAULT_ROOT / 'synthetic', scenes_per_geometry=32, seed=SEED, strengths=(0, 8)):
    output = _guard_path(Path(output), write=True); output.mkdir(parents=True, exist_ok=False); rows=[]
    if scenes_per_geometry < 1: raise ValueError('Positive scene count required')
    for geometry, grid in GEOMETRIES.items():
        for index in range(scenes_per_geometry):
            scene = _synthetic_scene(index, geometry, seed); sid=f'syn-{geometry}-{index:03d}'
            render = lambda changed, strength, layout: synthetic_render(scene, grid, changed, strength, layout)
            rows += _rows_for_scene(output, sid, sid, 'synthetic', geometry, scene, render, strengths, index % 8,
                dict(dataset='DeepStack new three-object procedural scenes', revision=ROW_SCHEMA, seed=seed,
                     generator_sha256=sha256_file(Path(__file__))))
    return _finish(output, rows)


def prior_sources(manifest=OLD_MANIFEST):
    rows=[json.loads(s) for s in Path(manifest).read_text().splitlines() if s.strip()]
    return {c['image_id'] for r in rows for c in r['natural']['crops'].values()}


def prepare_natural(output=DEFAULT_ROOT / 'preparation', gqa_root=GQA_ROOT, old_manifest=OLD_MANIFEST):
    """Fixed 8 category triples x 4 source clusters; target+4 spare color-qualified photos per cell.

    Examines seeded ranks without skipping un-fetched candidates. Only new necessary images
    download, at most four concurrently; all decisions precede model output.
    """
    from PIL import Image
    from cloud_prep.measurement_v7 import natobj_data as nd, natobj_gqa as ng
    output=_guard_path(Path(output),write=True); output.mkdir(parents=True,exist_ok=True)
    photos=output/'source_photos'; photos.mkdir(exist_ok=True)
    gqa_root=Path(gqa_root); config=nd.read_config();config['seed']=SEED
    exclude=prior_sources(old_manifest); population=nd.candidate_population(gqa_root,config)
    cached=ng.fetch_status(gqa_root); fresh={}; log=output/'fetch_log.jsonl'
    if log.exists():
        for line in log.read_text().splitlines():
            rec=json.loads(line)
            if rec['status']=='ok' or fresh.get(rec['image_id'],{}).get('status')!='ok':fresh[rec['image_id']]=rec
    targets=Counter(o for triple in TRIPLES for o in triple)
    targets={o:n*4 for o,n in targets.items()}
    checks=[]; passing=defaultdict(list)
    for category in sorted(targets):
        for color in ('red','blue'):
            cell=(category,color); candidates=[r for r in population[cell] if r['image_id'] not in exclude and not r['annotation_reasons']]
            pos=0; target=targets[category]+4
            while pos<len(candidates) and len(passing[cell])<target:
                batch=candidates[pos:pos+max(4,min(12,(target-len(passing[cell]))*2))]
                missing=list(dict.fromkeys(r['image_id'] for r in batch if r['image_id'] not in cached and r['image_id'] not in fresh))
                sizes={r['image_id']:r['image_size'] for r in batch}
                if missing:
                    with ThreadPoolExecutor(max_workers=4) as pool:
                        fetched=list(pool.map(lambda iid: ng._fetch_one(iid,sizes[iid],photos,retries=2,backoff=1.5,pause=.1),missing))
                    with log.open('a') as f:
                        for r in fetched:f.write(json.dumps(r,sort_keys=True)+'\n');fresh[r['image_id']]=r
                for cand in batch:
                    if len(passing[cell])>=target:break
                    pos+=1; rec=cached.get(cand['image_id'],fresh.get(cand['image_id']))
                    r=deepcopy(cand);r['label_pass']=False
                    if rec and rec['status']=='ok':
                        path=(gqa_root/'images' if cand['image_id'] in cached else photos)/rec['file']
                        if sha256_file(path)!=rec['sha256']:raise ValueError('Source photo SHA mismatch')
                        with Image.open(path) as im:
                            if list(im.size)!=cand['image_size']:raise ValueError('GQA/photo dimensions differ')
                            r['pixel']=nd.pixel_check(im,cand['bbox'],color,config)
                        r['label_pass']=r['pixel']['passed'];r['source']=rec;r['source_path']=str(path)
                        r['crop_box']=nd.crop_box(cand['bbox'],cand['image_size'],config['canvas']['context_margin'])
                        if r['label_pass']:passing[cell].append(r)
                    else:r['fetch_failed']=True
                    checks.append(r)
                (output/'label_checks.jsonl').write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in checks))
            print(f'{category}/{color}: {len(passing[cell])} passing / {pos} examined; target {target}',flush=True)
    # Deterministic category-stratum order, seeded candidate rank, no source photo reuse.
    used=set(); clusters=[]
    for ti,triple in enumerate(TRIPLES):
        for rep in range(4):
            picks={}
            for obj in triple:
                for color in ('red','blue'):
                    found=next((r for r in passing[(obj,color)] if r['image_id'] not in used),None)
                    if found is None:raise ValueError(f'Insufficient independent sources for {obj}/{color}; expand declared source pool')
                    picks[f'{obj}_{color}']=found;used.add(found['image_id'])
            clusters.append(dict(cluster_id=f'nat-{ti:02d}-{rep:02d}', objects=list(triple), crops=picks,
                generation_group=(ti*4+rep)%8, stratum=ti, replicate=rep))
    selection=dict(schema=ROW_SCHEMA,clusters=clusters,excluded_source_photos=sorted(exclude),
        old_manifest_sha256=sha256_file(old_manifest),gqa_index=json.loads((gqa_root/'index'/'index_summary.json').read_text()),
        gqa_archive=json.loads((gqa_root/'provenance'/'sceneGraphs.download.json').read_text()),config=config,
        selection_rule='8 declared category triples x4; first label-passing seeded ranks with no source photo reuse',
        selected_source_photos=len(used), source_overlap_with_v7_v8=len(used & exclude), model_results=False)
    (output/'selection.json').write_text(json.dumps(selection,indent=2)+'\n')
    contact_sheets(selection,output)
    return dict(clusters=len(clusters),photos=len(used),selection=str(output/'selection.json'),checks=len(checks),
        source_overlap_with_v7_v8=len(used & exclude),visual_review='pending')


def contact_sheets(selection, output):
    from PIL import Image,ImageDraw
    from cloud_prep.measurement_v7.natobj_data import render_tile
    crops=[c for cluster in selection['clusters'] for c in cluster['crops'].values()]
    for start in range(0,len(crops),48):
        sheet=Image.new('RGB',(8*160,6*190),'white');draw=ImageDraw.Draw(sheet)
        for k,c in enumerate(crops[start:start+48]):
            x=(k%8)*160;y=(k//8)*190
            tile,_=render_tile(c['source_path'],c['crop_box'],dict(box=152,background=224))
            sheet.paste(tile,(x+4,y));tile.close()
            draw.text((x+3,y+152),f'{c["category"]} {c["color"]}',fill='black')
            draw.text((x+3,y+167),c['candidate_id'],fill='black')
        sheet.save(Path(output)/f'contact_sheet_{start//48+1:02d}.png');sheet.close()


def apply_visual_exclusions(selection_path, exclusions):
    """Replace declared label/duplicate failures by next passing unused source; retain history."""
    selection_path=Path(selection_path);root=selection_path.parent;selection=json.loads(selection_path.read_text())
    checks=[json.loads(s) for s in (root/'label_checks.jsonl').read_text().splitlines()]
    original=root/'selection_before_visual_review.json'
    if not original.exists():original.write_text(selection_path.read_text())
    review=selection.setdefault('visual_review',dict(criteria={'C1':'Recognizable instance of named category',
        'C2':'Named color is dominant chromatic color of object',
        'D1':'No observed duplicate photo content within panel or with old source set'},exclusions={},replacements=[],status='pending'))
    review['exclusions'].update(exclusions)
    used={c['image_id'] for g in selection['clusters'] for c in g['crops'].values()}
    bad_photos={c['image_id'] for c in checks if c['candidate_id'] in review['exclusions']}
    replaced=[]
    for cluster in selection['clusters']:
        for role,c in list(cluster['crops'].items()):
            if c['candidate_id'] not in exclusions:continue
            replacement=next((r for r in checks if r.get('label_pass') and r['category']==c['category'] and
                r['color']==c['color'] and r['image_id'] not in used|bad_photos and r['candidate_id'] not in review['exclusions']),None)
            if replacement is None:raise ValueError(f'Need additional reviewed candidate for {role}')
            cluster['crops'][role]=replacement;used.add(replacement['image_id'])
            change=dict(cluster_id=cluster['cluster_id'],role=role,excluded=c['candidate_id'],replacement=replacement['candidate_id'])
            review['replacements'].append(change);replaced.append(change)
    selection_path.write_text(json.dumps(selection,indent=2)+'\n');contact_sheets(selection,root)
    return replaced


def build_natural(selection, output=DEFAULT_ROOT / 'natural', strengths=(0,8)):
    from PIL import Image
    from cloud_prep.measurement_v7.natobj_data import render_tile,gray_blend
    selection_path=Path(selection);selection=json.loads(selection_path.read_text());output=_guard_path(Path(output),write=True)
    if selection.get('visual_review',{}).get('status')!='passed':raise ValueError('Finish the declared pre-model visual label review before building')
    output.mkdir(parents=True,exist_ok=False); rows=[]
    for cluster in selection['clusters']:
        unit=cluster['cluster_id'];tiles={};crops=deepcopy(cluster['crops'])
        for key,c in crops.items():
            path=Path(c.pop('source_path'))
            if not path.is_absolute():path=selection_path.parent.parent/path
            dest=output/'sources'/path.name;dest.parent.mkdir(exist_ok=True)
            if not dest.exists():shutil.copyfile(path,dest)
            c['retained_source_file']=str(dest.relative_to(output))
            tiles[key],c['render']=render_tile(dest,c['crop_box'],dict(box=160,background=224))
        rng=random.Random(SEED+cluster['stratum']*101+cluster['replicate']);slots=list(range(3));rng.shuffle(slots)
        scene=dict(objects=cluster['objects'],object_to_slot=dict(zip(cluster['objects'],slots)),
            object_values_a={o:(i+cluster['replicate'])%2 for i,o in enumerate(cluster['objects'])},background=224)
        for geometry,grid in GEOMETRIES.items():
            sid=f'{unit}-{geometry}'
            def render(changed,strength,layout):
                im=Image.new('RGB',(grid[1]*TILE,grid[0]*TILE),(224,)*3)
                for obj in scene['objects']:
                    value=scene['object_values_a'][obj]^(obj==changed);tile=tiles[f'{obj}_{("red","blue")[value]}']
                    dx,dy=slot_box(scene['object_to_slot'][obj],grid)[:2];im.paste(tile,(dx+16,dy+16))
                out=gray_blend(im,strength);im.close()
                if layout=='M1':moved=permute_image(out,grid,'cycle_forward');out.close();return moved
                return out
            natural=dict(construction='controlled_recomposition',note='Real crop substitution changes source instance/context as well as color; not exact same-object color editing or unedited whole photographs.',crops=crops)
            rows+=_rows_for_scene(output,sid,unit,'natural_recomposition',geometry,scene,render,strengths,
                cluster['generation_group'],dict(dataset='GQA scene graphs / Visual Genome object crops',revision=ROW_SCHEMA,
                annotation_archive=selection['gqa_archive'],index_sha256=selection['gqa_index']['index_sha256'],
                old_manifest_sha256=selection['old_manifest_sha256'],excluded_source_photos=selection['excluded_source_photos'],
                generator_sha256=sha256_file(Path(__file__))),natural)
        for tile in tiles.values():tile.close()
    provenance=output/'provenance';provenance.mkdir()
    portable=deepcopy(selection)
    for cluster in portable['clusters']:
        for c in cluster['crops'].values():c['source_path']='sources/'+Path(c['source_path']).name
    (provenance/'selection.json').write_text(json.dumps(portable,indent=2)+'\n')
    (provenance/'config.json').write_text(json.dumps(selection['config'],indent=2)+'\n')
    (provenance/'visual_review.json').write_text(json.dumps(selection['visual_review'],indent=2)+'\n')
    # Selection/check evidence is small; unselected full photos remain in preparation only.
    for filename in ('label_checks.jsonl','fetch_log.jsonl','source_near_duplicates.json'):
        src=selection_path.parent/filename
        if src.exists():shutil.copyfile(src,provenance/filename)
    return _finish(output,rows)


def load_manifest(manifest,image_root=None,*,check_hashes=True):
    manifest=_guard_path(Path(manifest));root=_guard_path(Path(image_root)) if image_root else manifest.parent
    rows=[json.loads(s) for s in manifest.read_text().splitlines() if s.strip()];checked={};unit_sources={};owners={}
    if not rows or len({r['id'] for r in rows})!=len(rows):raise ValueError('Empty/duplicate queries')
    for r in rows:
        if r['schema']!=ROW_SCHEMA or r['family']!=FAMILY:raise ValueError('Unknown multiobject schema')
        grid=GEOMETRIES[r['geometry']];obj=r['objects'][r['target_index']];slot=r['scene']['object_to_slot'][obj]
        if r['tile_grid']!=grid or set(r['scene']['object_to_slot'].values())!={0,1,2}:raise ValueError('Three occupied distinct tiles required')
        value=r['scene']['object_values_a'][obj]
        if r['target_object']!=obj or r['changed_object']!=obj or r['queried_object']!=obj:raise ValueError('Wrong changed/query object')
        if r['answer_index']!=value or r['direction_a_index']!=value or r['direction_b_index']!=1-value:raise ValueError('Wrong a/b direction')
        if r['target_main_slot_by_layout']!={'M0':slot,'M1':(slot+1)%3} or r['donor_roi_bbox_xyxy']!=slot_box(slot,grid):raise ValueError('Wrong spatial mapping')
        expected={n:None if s['strength']==0 else value^(s['assignment']=='b') for n,s in r['source_specs'].items()}
        if r['source_answer_indices']!=expected:raise ValueError('Source answers disagree')
        if r['panel_family']=='synthetic' and r['unit_id']!=r['scene_id']:raise ValueError('Synthetic independent unit must be scene')
        if r['panel_family']=='natural_recomposition':
            photos={c['image_id'] for c in r['natural']['crops'].values()}
            if len(photos)!=6 or photos&set(r['source']['excluded_source_photos']):raise ValueError('Reused old or within-cluster photos')
            if unit_sources.setdefault(r['unit_id'],photos)!=photos:raise ValueError('Source cluster membership changed')
            for photo in photos:
                if owners.setdefault(photo,r['unit_id'])!=r['unit_id']:raise ValueError('Different clusters share source photos')
            for crop in r['natural']['crops'].values():
                rel=Path(crop['retained_source_file'])
                if rel.is_absolute() or '..' in rel.parts:raise ValueError('Portable contained original-photo path required')
                path=_guard_path(root/rel)
                if not path.is_relative_to(root) or not path.is_file():raise ValueError('Missing/escaping original source photo')
                if check_hashes:
                    if path not in checked:checked[path]=(path.stat().st_size,sha256_file(path))
                    if checked[path]!=(crop['source']['bytes'],crop['source']['sha256']):raise ValueError('Original source photo hash differs')
        if set(r['source_images'])!=set(r['source_specs']) or set(r['source_image_metadata'])!=set(r['source_specs']):raise ValueError('Source set mismatch')
        for name,rel in r['source_images'].items():
            rel=Path(rel)
            if rel.is_absolute() or '..' in rel.parts:raise ValueError('Portable contained paths required')
            path=_guard_path(root/rel)
            if not path.is_relative_to(root) or not path.is_file():raise ValueError('Missing/escaping image')
            meta=r['source_image_metadata'][name]
            if [meta['width'],meta['height']]!=[grid[1]*TILE,grid[0]*TILE]:raise ValueError('Wrong image dimensions')
            if check_hashes:
                if path not in checked:checked[path]=(path.stat().st_size,sha256_file(path))
                if checked[path]!=(meta['bytes'],meta['sha256']):raise ValueError('Image hash differs')
            r['source_images'][name]=str(path)
    return rows


def select_panel(rows,panel):
    if panel=='main':return rows
    if panel!='smoke':raise ValueError('Unknown panel')
    out=[]
    for family in sorted({r['panel_family'] for r in rows}):
        unit=next(r['unit_id'] for r in rows if r['panel_family']==family)
        for geometry in GEOMETRIES:
            first=next(r for r in rows if r['panel_family']==family and r['geometry']==geometry and
                       (family=='synthetic' or r['unit_id']==unit))
            out.extend(r for r in rows if r['scene_id']==first['scene_id'])
    return out


def validate(manifest,image_root=None):
    from PIL import Image,ImageChops
    rows=load_manifest(manifest,image_root);checked=set();pair_n=0
    for r in rows:
        with Image.open(r['source_images']['d_a']) as a,Image.open(r['source_images']['d_b']) as b:
            diff=ImageChops.difference(a,b);bbox=diff.getbbox();slot=r['donor_roi_bbox_xyxy']
            if bbox is None or bbox[0]<slot[0] or bbox[1]<slot[1] or bbox[2]>slot[2] or bbox[3]>slot[3]:raise ValueError('Donor pair changes outside target tile')
            # Clearing the target tile must erase every difference.
            diff.paste((0,0,0),tuple(slot))
            if diff.getbbox():raise ValueError('Unchanged objects altered')
            pair_n+=1
        for name,spec in r['source_specs'].items():
            if spec['type']!='main' or spec['layout']!='M0' or r['source_images'][name] in checked:continue
            with Image.open(r['source_images'][name]) as m0,Image.open(r['source_images'][main_source('M1',round(spec['strength']*100))]) as m1:
                with permute_image(m0,r['tile_grid'],'cycle_forward') as expected:
                    if ImageChops.difference(expected,m1).getbbox():raise ValueError('M1 is not exact tile cycle')
                with permute_image(m1,r['tile_grid'],'cycle_inverse') as recovered:
                    if ImageChops.difference(recovered,m0).getbbox():raise ValueError('Cycle inverse differs')
            checked.add(r['source_images'][name])
    images={p for r in rows for p in r['source_images'].values()}
    original_photos={c['image_id'] for r in rows if 'natural' in r for c in r['natural']['crops'].values()}
    return dict(schema=ROW_SCHEMA,manifest=str(manifest),manifest_sha256=sha256_file(manifest),status='passed',
        model_results=False,items=len(rows),scene_templates=len({r['scene_id'] for r in rows}),
        independent_units=len({r['unit_id'] for r in rows}),panel_families=dict(Counter(r['panel_family'] for r in rows)),
        geometries=dict(Counter(r['geometry'] for r in rows)),unique_image_paths=len(images),
        image_bytes=sum(Path(p).stat().st_size for p in images),single_target_tile_pairs=pair_n,exact_main_cycle_pairs=len(checked),
        source_original_photos_hash_verified=len(original_photos),
        synthetic_note='Only the named object color changes; exact procedural attribute counterfactual',
        natural_note='One named-category crop changes; source instance and context also change. Geometries repeat within source cluster.')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('build-synthetic','prepare-natural','build-natural','validate'))
    p.add_argument('--output',type=Path);p.add_argument('--manifest',type=Path);p.add_argument('--image-root',type=Path)
    p.add_argument('--selection',type=Path,default=DEFAULT_ROOT/'preparation'/'selection.json')
    p.add_argument('--gqa-root',type=Path,default=GQA_ROOT);p.add_argument('--old-manifest',type=Path,default=OLD_MANIFEST)
    p.add_argument('--scenes-per-geometry',type=int,default=32)
    a=p.parse_args(argv)
    if a.command=='build-synthetic':result=build_synthetic(a.output or DEFAULT_ROOT/'synthetic',a.scenes_per_geometry)
    elif a.command=='prepare-natural':result=prepare_natural(a.output or DEFAULT_ROOT/'preparation',a.gqa_root,a.old_manifest)
    elif a.command=='build-natural':result=build_natural(a.selection,a.output or DEFAULT_ROOT/'natural')
    elif a.manifest:result=validate(a.manifest,a.image_root)
    else:p.error('--manifest required')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
