"""Matched color/stripe single-object materials, with no model/tensor imports."""
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
from cloud_prep.measurement_v6 import content_data as old
from cloud_prep.measurement_v8.mechanism_design import main_source
from cloud_prep.measurement_v10.full_data import source_name

SCHEMA = 'paired-attribute-single-object-v12-v1'
SEED = 2026102202
SIZE = old.SIZE
ATTRIBUTES = tuple(old.ATTRIBUTES)
DEFAULT_ROOT = (DATA_ROOT / 'datasets/attribute-roles-v12')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def scene_spec(attribute, pair, layout, assignment, replicate, seed=SEED):
    # Draw geometry ONCE using the color-independent template key. Changing
    # attribute only selects its texture; all geometric/background fields match.
    spec = old.scene_spec('color', pair, layout, assignment, replicate, seed)
    spec['attribute'] = attribute
    return spec


def geometry_signature(spec):
    return digest({k: v for k, v in spec.items() if k != 'attribute'})


def bbox(spec, obj):
    x0, y0, x1, y1 = old.object_box(spec, obj)
    return [x0, y0, x1 + 1, y1 + 1]


def source_specs():
    specs = {source_name(layout, strength, assignment):
             dict(layout=layout, strength=strength / 100, assignment=assignment,
                  type='attribute_source')
             for layout, strength, assignment in product(('M0', 'M1'), (8, 100), ('a', 'b'))}
    for layout, strength in product(('M0', 'M1'), (8, 100)):
        specs[main_source(layout, strength)] = dict(specs[source_name(layout, strength, 'a')])
    for assignment in ('a', 'b'):
        specs['d_' + assignment] = dict(specs[source_name('M0', 100, assignment)])
    return specs


def render(spec, changed, source):
    """Exactly one object can change; M1 is a pixel-exact half swap."""
    from PIL import Image, ImageDraw
    image = Image.new('RGB', SIZE, (spec['background'],) * 3)
    for obj in spec['objects']:
        mask = Image.new('L', SIZE, 0)
        old._shape(ImageDraw.Draw(mask), obj, old.object_box(spec, obj), fill=255)
        value = spec['object_values_a'][obj] ^ (source['assignment'] == 'b' and obj == changed)
        neutral = Image.new('RGB', SIZE, (150, 150, 150))
        if spec['attribute'] == 'color':
            full = Image.new('RGB', SIZE, ((214, 43, 49), (39, 87, 207))[value])
        elif spec['attribute'] == 'stripe_orientation':
            full = Image.new('RGB', SIZE, (224, 224, 224))
            draw = ImageDraw.Draw(full)
            for offset in range(spec['stripe_phase'] - 14, max(SIZE), 14):
                line = (offset, 0, offset, SIZE[1]) if value == 0 else (0, offset, SIZE[0], offset)
                draw.line(line, fill=(55, 55, 55), width=6)
        else:
            raise ValueError('Unknown attribute')
        texture = Image.blend(neutral, full, source['strength'])
        image.paste(texture, (0, 0), mask)
        mask.close(); neutral.close(); full.close(); texture.close()
    if source['layout'] == 'M1':
        swapped = old.swap_image_halves(image)
        image.close()
        return swapped
    return image


def visual_signature(row):
    return digest(dict(images={k: v['sha256'] for k, v in row['source_image_metadata'].items()},
                       roi=row['donor_roi_bbox_xyxy'], changed=row['changed_object'],
                       attribute=row['attribute'], template_id=row['template_id']))


def query_text(attribute, obj):
    return f'What color is the {obj}?' if attribute == 'color' else f'What pattern is inside the {obj}?'


def dose_subset(rows, count=32):
    templates = {r['template_id']: r for r in rows}
    groups = defaultdict(list)
    for tid, r in templates.items():
        groups[(r['shape_pair'], r['layout'], r['assignment'])].append(tid)
    ranked = {g: sorted(ids, key=lambda tid: digest(['p2-dose-subset-v12-20261022', tid]))
              for g, ids in groups.items()}
    balanced = [ranked[g][rank] for rank in range(max(map(len, ranked.values())))
                for g in sorted(ranked) if rank < len(ranked[g])]
    if count > len(balanced) or count < 1:
        raise ValueError('Dose subset exceeds available templates')
    return balanced[:count]


def build(output=DEFAULT_ROOT, seed=SEED, replicates=8):
    if type(replicates) is not int or replicates < 1:
        raise ValueError('Positive replicate count required')
    output = _guard_path(Path(output), write=True)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'images').mkdir()
    rows, specs = [], source_specs()
    for pair, layout, assignment, replicate in product(range(2), range(2), range(2), range(replicates)):
        tid = f'attr12-s{seed}-p{pair}-l{layout}-a{assignment}-r{replicate}'
        for attribute in ATTRIBUTES:
            spec = scene_spec(attribute, pair, layout, assignment, replicate, seed)
            shared = {}
            for changed_index, changed in enumerate(spec['objects']):
                iid = f'{tid}-{attribute}-change-{changed}'
                paths, metadata = {}, {}
                for name, source in specs.items():
                    # Alias old local names to the same exact source image.
                    canonical = source_name(source['layout'], round(source['strength'] * 100), source['assignment'])
                    key = canonical if source['assignment'] == 'a' else f'{changed}-{canonical}'
                    if key not in shared:
                        rel = Path('images') / f'{tid}-{attribute}-{key}.png'
                        with render(spec, changed, source) as image:
                            image.save(output / rel)
                        shared[key] = (str(rel), dict(width=SIZE[0], height=SIZE[1], source_field='image_1',
                            bytes=(output / rel).stat().st_size, sha256=sha256_file(output / rel)))
                    paths[name], metadata[name] = shared[key]
                for queried_index, queried in enumerate(spec['objects']):
                    choices = list(old.ATTRIBUTES[attribute])
                    value, slot = spec['object_values_a'][queried], spec['object_to_slot'][queried]
                    changed_slot = spec['object_to_slot'][changed]
                    query = query_text(attribute, queried)
                    role = 'changed' if changed == queried else 'unchanged'
                    row = dict(schema=SCHEMA, family='paired_attributes', task='attribute_roles_' + attribute,
                        id=f'{iid}-query-{queried}', scene_id=tid, template_id=tid,
                        intervention_id=iid, attribute=attribute, shape_pair=pair, layout=layout,
                        assignment=assignment, replicate=replicate, generation_group=replicate,
                        scene=deepcopy(spec), geometry_signature=geometry_signature(spec),
                        changed_object=changed, changed_index=changed_index,
                        queried_object=queried, queried_index=queried_index,
                        target_object=queried, target_index=queried_index,
                        query_type='identity', query_target=role, query_role=role, query_slot=slot,
                        target_main_slot=slot, target_main_slot_by_layout={'M0':slot, 'M1':1-slot},
                        queried_main_slot_by_layout={'M0':slot, 'M1':1-slot},
                        changed_main_slot_by_layout={'M0':changed_slot, 'M1':1-changed_slot},
                        source_roi_slot=changed_slot, tile_grid=[1, 2],
                        donor_roi_bbox_xyxy=bbox(spec, changed), donor_roi_image_size=list(SIZE),
                        donor_roi_coordinate_system='pixel_half_open_xyxy', donor_reference_layout='M0',
                        direction_a_index=spec['object_values_a'][changed],
                        direction_b_index=1-spec['object_values_a'][changed],
                        query=query, choices=choices, option_labels=['(A)', '(B)'],
                        prompt=query+'\n'+''.join(f'({chr(65+i)}) {c}\n' for i,c in enumerate(choices)),
                        answer_index=value, answer_label=['(A)', '(B)'][value],
                        source_specs=deepcopy(specs), source_images=deepcopy(paths),
                        source_image_metadata=deepcopy(metadata),
                        source_answer_indices={n:value ^ (s['assignment']=='b' and changed==queried) for n,s in specs.items()},
                        referred_object_by_source={n:queried for n in specs},
                        record_fields={'source_cluster_id':tid, 'template_id':tid},
                        source=dict(dataset='DeepStack paired attribute single-object counterfactuals',
                            revision=SCHEMA, seed=seed, split='exploratory', generator_sha256=sha256_file(Path(__file__)),
                            texture_definition='v6 stripe period14 width6 and color constants; geometry paired across attributes',
                            reuse_scope='new seeded geometry templates; neither old v8 pixels nor outcome-selected'))
                    row['visual_source_signature'] = visual_signature(row)
                    rows.append(row)
    selected = dose_subset(rows, min(32, replicates * 8))
    for row in rows:
        row['dose_bridge_selected'] = row['template_id'] in selected
    manifest = output / 'manifest.jsonl'
    manifest.write_text(''.join(json.dumps(r, sort_keys=True) + '\n' for r in rows))
    (output / 'dose_subset.json').write_text(json.dumps(dict(template_ids=selected,
        count=len(selected), selector='round-robin shape_pair/layout/assignment strata; SHA256 rank within strata',
        ranking_key='p2-dose-subset-v12-20261022', uses_model_results=False), indent=2)+'\n')
    result = validate(manifest)
    (output / 'dataset.json').write_text(json.dumps(result, indent=2)+'\n')
    contact_sheets(load_manifest(manifest), output / 'qa')
    return result


def validate_query_groups(rows):
    if not rows or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Nonempty unique rows required')
    groups, templates = defaultdict(list), defaultdict(list)
    for row in rows:
        groups[row['intervention_id']].append(row)
        templates[row['template_id']].append(row)
    for members in groups.values():
        if len(members) != 2 or {r['query_target'] for r in members} != {'changed','unchanged'}:
            raise ValueError('Both same-intervention queries required')
        for field in ('visual_source_signature','scene','changed_object','direction_a_index','direction_b_index',
                      'source_images','source_specs','source_image_metadata','donor_roi_bbox_xyxy','dose_bridge_selected'):
            if members[0][field] != members[1][field]:
                raise ValueError('Query altered intervention: ' + field)
    expected = set(product(ATTRIBUTES, range(2), range(2)))
    for members in templates.values():
        if len(members) != 8 or {(r['attribute'],r['changed_index'],r['queried_index']) for r in members} != expected:
            raise ValueError('Whole paired-attribute template required, not split query/attribute shards')
        if len({r['geometry_signature'] for r in members}) != 1:
            raise ValueError('Attributes must share identical geometry')


def load_manifest(manifest, image_root=None, *, check_hashes=True):
    manifest = _guard_path(Path(manifest))
    root = _guard_path(Path(image_root)) if image_root else manifest.parent
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    checked = {}
    for row in rows:
        if row['schema'] != SCHEMA or row['source_specs'] != source_specs():
            raise ValueError('Unknown schema/source specs')
        spec = scene_spec(row['attribute'], *(row[k] for k in ('shape_pair','layout','assignment','replicate')), seed=row['source']['seed'])
        changed, queried = spec['objects'][row['changed_index']], spec['objects'][row['queried_index']]
        if row['scene'] != spec or row['geometry_signature'] != geometry_signature(spec):
            raise ValueError('Geometry or seed mismatch')
        if row['changed_object'] != changed or row['queried_object'] != queried or row['target_object'] != queried:
            raise ValueError('Object mismatch')
        if row['choices'] != list(old.ATTRIBUTES[row['attribute']]) or row['query'] != query_text(row['attribute'],queried):
            raise ValueError('Attribute candidates/query mismatch')
        if (row['direction_a_index'],row['direction_b_index']) != (spec['object_values_a'][changed],1-spec['object_values_a'][changed]):
            raise ValueError('Direction changed with query')
        if row['answer_index'] != spec['object_values_a'][queried] or row['query_target'] != ('changed' if changed==queried else 'unchanged'):
            raise ValueError('Native query truth mismatch')
        answers = {n:spec['object_values_a'][queried] ^ (s['assignment']=='b' and changed==queried) for n,s in source_specs().items()}
        if row['source_answer_indices'] != answers or row['donor_roi_bbox_xyxy'] != bbox(spec,changed):
            raise ValueError('Source truth/ROI mismatch')
        if row['visual_source_signature'] != visual_signature(row):
            raise ValueError('Visual signature mismatch')
        if set(row['source_images']) != set(source_specs()) or set(row['source_image_metadata']) != set(source_specs()):
            raise ValueError('Missing/extra sources')
        for name, rel in row['source_images'].items():
            relative = Path(rel)
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError('Contained relative image paths required')
            path = _guard_path(root / relative)
            if not path.is_relative_to(root) or not path.is_file():
                raise ValueError('Missing/escaping image')
            meta = row['source_image_metadata'][name]
            if (meta['width'],meta['height']) != SIZE:
                raise ValueError('Canvas mismatch')
            if check_hashes:
                checked.setdefault(path, (path.stat().st_size, sha256_file(path)))
                if checked[path] != (meta['bytes'],meta['sha256']):
                    raise ValueError('Image bytes/hash mismatch')
            row['source_images'][name] = str(path)
    validate_query_groups(rows)
    return rows


def select_rows(rows, panel='main', dose=False, template_ids=None):
    tids = sorted({r['template_id'] for r in rows if not dose or r['dose_bridge_selected']})
    if panel == 'smoke':
        # Preserve both geometry families and BOTH attributes, edits and queries.
        first = {r['shape_pair']:r['template_id'] for r in reversed(rows)
                 if r['template_id'] in tids and r['layout']==r['assignment']==0}
        tids = sorted(first.values())
    elif panel != 'main':
        raise ValueError('Unknown panel')
    if template_ids is not None:
        if set(template_ids)-set(tids): raise ValueError('Requested templates outside selected panel')
        tids = template_ids
    result = [r for r in rows if r['template_id'] in tids]
    validate_query_groups(result)
    return result


def validate(manifest, image_root=None):
    from PIL import Image, ImageChops, ImageDraw
    rows = load_manifest(manifest,image_root)
    pairs = {r['intervention_id']:r for r in rows}
    changed_pixels = []
    for row in pairs.values():
        for layout,strength in product(('M0','M1'),(8,100)):
            names = [source_name(layout,strength,s) for s in ('a','b')]
            mask = Image.new('L',SIZE,0)
            old._shape(ImageDraw.Draw(mask),row['changed_object'],old.object_box(row['scene'],row['changed_object']),fill=255)
            if layout == 'M1':
                new_mask=old.swap_image_halves(mask); mask.close(); mask=new_mask
            with Image.open(row['source_images'][names[0]]) as a, Image.open(row['source_images'][names[1]]) as b:
                diff=ImageChops.difference(a,b).convert('L').point(lambda x:255 if x else 0)
                if not diff.getbbox() or ImageChops.subtract(diff,mask).getbbox():
                    raise ValueError('Change must be nonempty and contained within exactly one object')
                changed_pixels.append(dict(intervention_id=row['intervention_id'],layout=layout,strength=strength,
                    changed_pixel_count=diff.histogram()[255],mask_pixel_count=mask.histogram()[255],
                    outside_changed_object_pixels=0,unchanged_object_pixels_changed=0))
                diff.close()
            mask.close()
        for name,source in row['source_specs'].items():
            if not name.startswith('s_'): continue
            with Image.open(row['source_images'][name]) as actual, render(row['scene'],row['changed_object'],source) as expected:
                if ImageChops.difference(actual,expected).getbbox():
                    raise ValueError('Pixels disagree with declared rendering')
            if source['layout']=='M0':
                other=source_name('M1',round(source['strength']*100),source['assignment'])
                with Image.open(row['source_images'][name]) as a, Image.open(row['source_images'][other]) as b, old.swap_image_halves(a) as swapped:
                    if ImageChops.difference(swapped,b).getbbox(): raise ValueError('M1 is not exact half swap')
    images={p for r in rows for p in r['source_images'].values()}
    result=dict(schema=SCHEMA,status='passed',model_results=False,manifest=str(manifest),manifest_sha256=sha256_file(manifest),
        rows=len(rows),independent_templates=len({r['template_id'] for r in rows}),
        attribute_scenes=len({(r['template_id'],r['attribute']) for r in rows}),interventions=len(pairs),
        same_intervention_dual_queries=len(pairs),pixel_pair_checks=len(changed_pixels),
        unique_images=len(images),image_bytes=sum(Path(p).stat().st_size for p in images),
        dose_templates=len({r['template_id'] for r in rows if r['dose_bridge_selected']}),
        independence_unit='template_id; color/stripe, both edits, both queries and every condition are paired repeats',
        mask_note='Stripe orientations overlap at some pixels; require nonempty changed support contained in target mask, not every target pixel different')
    report=Path(manifest).parent/'pixel_validation.jsonl'
    report.write_text(''.join(json.dumps(x,sort_keys=True)+'\n' for x in changed_pixels))
    return result


def contact_sheets(rows, output):
    from PIL import Image, ImageDraw
    output.mkdir(parents=True,exist_ok=True)
    templates=sorted({r['template_id'] for r in rows})
    for page in range((len(templates)+7)//8):
        selected=templates[page*8:(page+1)*8]
        canvas=Image.new('RGB',(1160, 34+len(selected)*220),'white')
        draw=ImageDraw.Draw(canvas)
        draw.text((8,8),'Each row: COLOR a / edit object 0 / edit object 1 | STRIPE a / edit object 0 / edit object 1',fill='black')
        for i,tid in enumerate(selected):
            members=[r for r in rows if r['template_id']==tid]
            draw.text((8,36+i*220),tid + (' [dose]' if members[0]['dose_bridge_selected'] else ''),fill='black')
            for attr_index,attribute in enumerate(ATTRIBUTES):
                for j,(changed_index,source) in enumerate(((0,'d_a'),(0,'d_b'),(1,'d_b'))):
                    row=next(r for r in members if r['attribute']==attribute and r['changed_index']==changed_index)
                    with Image.open(row['source_images'][source]) as im:
                        im=im.resize((192,128))
                        canvas.paste(im,((attr_index*3+j)*192+4,56+i*220)); im.close()
                    text='a: unchanged' if source=='d_a' else 'edit '+row['changed_object']
                    draw.text(((attr_index*3+j)*192+6,188+i*220),text,fill='black')
            draw.text((8,205+i*220),'Queries: '+members[0]['scene']['objects'][0]+' / '+members[0]['scene']['objects'][1]+'; M1 is exact half swap; alpha .08/1',fill='black')
        canvas.save(output/f'contact-{page+1:02d}.png'); canvas.close()
    (output/'README.txt').write_text('Eight templates per page. Six images per template: unchanged and each one-object edit for color and stripe. Pixel validation covers all layouts and strengths. Visual inspection is independent of model outcomes.\n')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('build','validate'))
    p.add_argument('--output',type=Path,default=DEFAULT_ROOT)
    p.add_argument('--manifest',type=Path)
    p.add_argument('--seed',type=int,default=SEED)
    p.add_argument('--replicates',type=int,default=8)
    a=p.parse_args(argv)
    if a.command=='validate' and a.manifest is None: p.error('--manifest required')
    print(json.dumps(build(a.output,a.seed,a.replicates) if a.command=='build' else validate(a.manifest),indent=2))


if __name__=='__main__': main()
