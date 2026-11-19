"""Portable v10 local-role queries over the accepted v8 pixels (PIL CPU only)."""
from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import os as _path_os
DATA_ROOT = Path(_path_os.environ.get("DS_DATA_ROOT", "work")).expanduser().resolve()
import shutil

from cloud_prep.portable_data import _guard_path, sha256_file
from cloud_prep.measurement_v8 import mechanism_data as v8

ROW_SCHEMA = 'local-role-data-v10'
FAMILY = 'local_role'
DEFAULT_ROOT = (DATA_ROOT / 'datasets/local-role-v10')
DEFAULT_SOURCE = v8.DEFAULT_ROOT / 'manifest.jsonl'


def visual_signature(row):
    """Query-independent declared visual intervention identity; no prompt/label input."""
    payload = {'sources': {k: v['sha256'] for k, v in row['source_image_metadata'].items()},
               'roi': row['donor_roi_bbox_xyxy'], 'size': row['donor_roi_image_size'],
               'changed_object': row['changed_object']}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def _expand(source, source_manifest_sha256):
    """One v8 changed-object pair becomes two queries with identical visual input."""
    rows = []
    for original in source:
        spec = original['scene']
        changed, changed_index = original['target_object'], original['target_index']
        for queried_index, queried in enumerate(spec['objects']):
            row = deepcopy(original)
            value = spec['object_values_a'][queried]
            slot = spec['object_to_slot'][queried]
            changed_slot = spec['object_to_slot'][changed]
            row.update(schema=ROW_SCHEMA, family=FAMILY, task='local_role_color',
                id=f"local10-{original['scene_id']}-change-{changed}-query-{queried}",
                intervention_id=f"{original['scene_id']}-change-{changed}",
                reused_v8_query_id=original['id'], changed_object=changed, changed_index=changed_index,
                queried_object=queried, queried_index=queried_index,
                query_role='changed' if queried == changed else 'unchanged',
                target_object=queried, target_index=queried_index, target_main_slot=slot,
                target_main_slot_by_layout={'M0': slot, 'M1': 1-slot},
                queried_main_slot_by_layout={'M0': slot, 'M1': 1-slot},
                changed_main_slot_by_layout={'M0': changed_slot, 'M1': 1-changed_slot},
                source_roi_slot=changed_slot,
                direction_a_index=spec['object_values_a'][changed],
                direction_b_index=1-spec['object_values_a'][changed],
                answer_index=value, answer_label=row['option_labels'][value],
                query=f'What color is the {queried}?',
                prompt=f'What color is the {queried}?\n(A) red\n(B) blue\n')
            row['source_specs']['d_b']['changed_object'] = changed
            row['source_answer_indices'] = {name: None if ss['strength'] == 0 else
                value ^ (ss['assignment'] == 'b' and queried == changed)
                for name, ss in row['source_specs'].items()}
            row['source'] = {**row['source'], 'revision': ROW_SCHEMA,
                'parent_revision': v8.ROW_SCHEMA, 'parent_manifest_sha256': source_manifest_sha256,
                'reuse_scope': 'All 64 v8 scenes and pixels; no new independent scene replication',
                'expander_sha256': sha256_file(Path(__file__))}
            row['visual_source_signature'] = visual_signature(row)
            rows.append(row)
    return rows


def build(output=DEFAULT_ROOT, source_manifest=DEFAULT_SOURCE):
    """Copy accepted image bytes once, then add off-target queries without rendering."""
    source_manifest = _guard_path(Path(source_manifest))
    source = v8.load_manifest(source_manifest)
    output = _guard_path(Path(output), write=True)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'images').mkdir()
    rows = _expand(source, sha256_file(source_manifest))
    copied = {}
    for row in rows:
        for name, path in row['source_images'].items():
            path = Path(path)
            relative = Path('images') / path.name
            if relative in copied and copied[relative] != path:
                raise ValueError('Source image basename collision')
            if relative not in copied:
                shutil.copyfile(path, output / relative)
                copied[relative] = path
            row['source_images'][name] = str(relative)
    manifest = output / 'manifest.jsonl'
    manifest.write_text(''.join(json.dumps(row, sort_keys=True)+'\n' for row in rows))
    result = validate(manifest)
    (output / 'dataset.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


def validate_query_groups(rows):
    """Every planned scene contains the full changed x queried 2x2 pairing."""
    if not rows or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Nonempty unique rescue queries required')
    scenes = defaultdict(list)
    for row in rows:
        scenes[row['scene_id']].append(row)
    for members in scenes.values():
        if len(members) != 4 or {(r['changed_index'], r['queried_index']) for r in members} != {(0,0),(0,1),(1,0),(1,1)}:
            raise ValueError('Each scene must retain all four changed-object x queried-object rows')
        for changed in (0, 1):
            pair = [r for r in members if r['changed_index'] == changed]
            for field in ('intervention_id', 'changed_object', 'direction_a_index', 'direction_b_index',
                          'source_roi_slot', 'donor_roi_bbox_xyxy', 'visual_source_signature',
                          'generation_group'):
                if pair[0][field] != pair[1][field]:
                    raise ValueError(f'Same intervention query pair differs: {field}')
            for field in ('source_images', 'source_image_metadata', 'source_specs'):
                if field in pair[0] and pair[0][field] != pair[1][field]:
                    raise ValueError(f'Same intervention query pair differs: {field}')


def load_manifest(manifest, image_root=None, *, check_hashes=True):
    manifest = _guard_path(Path(manifest))
    root = _guard_path(Path(image_root)) if image_root else manifest.parent
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    checked = {}
    for row in rows:
        if row['schema'] != ROW_SCHEMA or row['family'] != FAMILY:
            raise ValueError('Unknown rescue data schema')
        spec = v8.scene_spec('color', *(row[k] for k in ('shape_pair', 'layout', 'assignment', 'replicate')), seed=row['source']['seed'])
        if row['scene'] != spec or row['source']['parent_revision'] != v8.ROW_SCHEMA:
            raise ValueError('Scene or reuse provenance mismatch')
        changed, queried = spec['objects'][row['changed_index']], spec['objects'][row['queried_index']]
        if (row['changed_object'], row['queried_object'], row['target_object'], row['target_index']) != (changed, queried, queried, row['queried_index']):
            raise ValueError('Changed/queried identity mismatch')
        if row['query_role'] != ('changed' if changed == queried else 'unchanged'):
            raise ValueError('Changed/unchanged query role mismatch')
        if row['query'] != f'What color is the {queried}?' or row['prompt'] != row['query']+'\n(A) red\n(B) blue\n':
            raise ValueError('Query text mismatch')
        if row['choices'] != ['red', 'blue'] or row['option_labels'] != ['(A)', '(B)']:
            raise ValueError('Choices changed')
        value = spec['object_values_a'][queried]
        if (row['answer_index'], row['answer_label']) != (value, row['option_labels'][value]):
            raise ValueError('Native queried-object answer mismatch')
        if (row['direction_a_index'], row['direction_b_index']) != (spec['object_values_a'][changed], 1-spec['object_values_a'][changed]):
            raise ValueError('Direction must be fixed to changed object, never queried object')
        strengths = [round(s['strength']*100) for s in row['source_specs'].values() if s['type'] == 'main']
        sources = v8.source_specs(strengths)
        sources['d_b']['changed_object'] = changed
        if row['source_specs'] != sources:
            raise ValueError('Source specification mismatch')
        answers = {n: None if s['strength'] == 0 else value ^ (s['assignment'] == 'b' and queried == changed) for n,s in sources.items()}
        if row['source_answer_indices'] != answers:
            raise ValueError('Native donor answer must flip only for the changed-object query')
        for obj, field in ((changed, 'changed_main_slot_by_layout'), (queried, 'queried_main_slot_by_layout')):
            slot = spec['object_to_slot'][obj]
            if row[field] != {'M0': slot, 'M1': 1-slot}:
                raise ValueError('Main layout slot mismatch')
        if row['source_roi_slot'] != spec['object_to_slot'][changed] or row['donor_roi_bbox_xyxy'] != v8.bbox(spec, changed):
            raise ValueError('Source ROI must remain the changed object in donor M0 coordinates')
        if row['donor_roi_image_size'] != list(v8.SIZE) or row['generation_group'] != row['replicate']:
            raise ValueError('Image geometry or generation selection changed')
        if row['visual_source_signature'] != visual_signature(row):
            raise ValueError('Declared visual signature mismatch')
        if set(row['source_images']) != set(sources) or set(row['source_image_metadata']) != set(sources):
            raise ValueError('Source image set mismatch')
        for name, relative in row['source_images'].items():
            relative = Path(relative)
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError('Contained portable image paths required')
            path = _guard_path(root / relative)
            if not path.is_relative_to(root) or not path.is_file():
                raise ValueError('Missing or escaping source image')
            meta = row['source_image_metadata'][name]
            if (meta['width'], meta['height']) != v8.SIZE:
                raise ValueError('Source image size mismatch')
            if check_hashes:
                if path not in checked:
                    checked[path] = path.stat().st_size, sha256_file(path)
                if checked[path] != (meta['bytes'], meta['sha256']):
                    raise ValueError('Source image bytes/hash mismatch')
            row['source_images'][name] = str(path)
    validate_query_groups(rows)
    return rows


def select_panel(rows, panel):
    if panel == 'main':
        return rows
    if panel != 'smoke':
        raise ValueError('Unknown panel')
    selected = [r for r in rows if r['replicate'] == 0 and
                (r['shape_pair'], r['layout'], r['assignment']) in ((0,0,0),(1,1,1))]
    validate_query_groups(selected)
    return selected


def validate(manifest, image_root=None):
    from PIL import Image, ImageChops, ImageDraw
    rows = load_manifest(manifest, image_root)
    pairs = {}
    for row in rows:
        pairs.setdefault(row['intervention_id'], row)
    for row in pairs.values():
        with Image.open(row['source_images']['d_a']) as a, Image.open(row['source_images']['d_b']) as b:
            with ImageChops.difference(a,b) as diff:
                changed = diff.convert('L').point(lambda x: 255 if x else 0)
            mask = Image.new('L', v8.SIZE, 0)
            v8._shape(ImageDraw.Draw(mask), row['changed_object'], v8.object_box(row['scene'],row['changed_object']), fill=255)
            with ImageChops.difference(changed, mask) as mismatch:
                if mismatch.getbbox() or changed.getbbox() != tuple(row['donor_roi_bbox_xyxy']):
                    raise ValueError('Exactly the declared changed-object pixels must differ')
            changed.close(); mask.close()
    images = {p for r in rows for p in r['source_images'].values()}
    return {'schema': ROW_SCHEMA, 'manifest': str(manifest), 'manifest_sha256': sha256_file(manifest),
        'status': 'passed', 'model_results': False, 'items': len(rows),
        'scene_templates': len({r['scene_id'] for r in rows}), 'intervention_pairs': len(pairs),
        'identical_visual_two_query_pairs': len(pairs), 'only_changed_object_pixel_pairs': len(pairs),
        'unique_image_paths': len(images), 'image_bytes': sum(Path(p).stat().st_size for p in images),
        'independence_unit': 'scene_id; v8 scenes reused, not independent new scene replication',
        'source_manifest_sha256': rows[0]['source']['parent_manifest_sha256'],
        'direction_contract': 'Both queried objects use changed-object a/b colors; queried native labels remain independent'}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('build','validate'))
    p.add_argument('--output', type=Path, default=DEFAULT_ROOT)
    p.add_argument('--source-manifest', type=Path, default=DEFAULT_SOURCE)
    p.add_argument('--manifest', type=Path)
    p.add_argument('--image-root', type=Path)
    a = p.parse_args(argv)
    if a.command == 'validate' and a.manifest is None:
        p.error('--manifest is required for validate')
    print(json.dumps(build(a.output,a.source_manifest) if a.command == 'build' else validate(a.manifest,a.image_root), indent=2))


if __name__ == '__main__':
    main()
