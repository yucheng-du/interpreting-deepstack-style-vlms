"""Build one-object-only color counterfactuals; all image work is lightweight PIL CPU."""
from __future__ import annotations

import argparse
from copy import deepcopy
from itertools import product
import json
from pathlib import Path
import os as _path_os
DATA_ROOT = Path(_path_os.environ.get("DS_DATA_ROOT", "work")).expanduser().resolve()

from cloud_prep.portable_data import _guard_path, sha256_file
from cloud_prep.measurement_v6.content_data import SIZE, scene_spec, object_box, _shape, swap_image_halves
from cloud_prep.measurement_v8.mechanism_design import main_source, LAYOUTS

ROW_SCHEMA = 'single-object-color-v8'
FAMILY = 'single_object_color'
SEED = 2026100708
COLORS = ((214, 43, 49), (39, 87, 207))
DEFAULT_ROOT = (DATA_ROOT / 'datasets/single-object-v8')


def source_specs(strengths=(0, 8)):
    result = {}
    for strength, layout in product(sorted(set(strengths) | {100}), LAYOUTS):
        result[main_source(layout, strength)] = {'type': 'main', 'layout': layout,
            'strength': strength / 100, 'assignment': 'a', 'pose': 'same',
            'transform': 'swap_equal_horizontal_halves' if layout == 'M1' else None}
    for value in ('a', 'b'):
        result[f'd_{value}'] = {'type': 'donor', 'layout': 'M0', 'strength': 1.,
            'assignment': value, 'pose': 'same', 'transform': None,
            'changed_object': 'query_target' if value == 'b' else None}
    return result


def render(spec, source_spec, target_object=None):
    from PIL import Image, ImageDraw
    image = Image.new('RGB', SIZE, (spec['background'],) * 3)
    for obj in spec['objects']:
        value = spec['object_values_a'][obj]
        if source_spec['assignment'] == 'b' and obj == target_object:
            value = 1 - value
        strength = source_spec['strength']
        color = tuple(int(150 * (1 - strength) + x * strength) for x in COLORS[value])
        _shape(ImageDraw.Draw(image), obj, object_box(spec, obj), fill=color)
    if source_spec['layout'] == 'M1':
        swapped = swap_image_halves(image)
        image.close()
        return swapped
    return image


def bbox(spec, target):
    x0, y0, x1, y1 = object_box(spec, target)
    return [x0, y0, x1 + 1, y1 + 1]  # Pillow drawing uses inclusive right/bottom.


def build(output=DEFAULT_ROOT, strengths=(0, 8), seed=SEED, replicates=8):
    output = _guard_path(Path(output), write=True)
    if type(replicates) is not int or replicates < 1:
        raise ValueError('Positive replicate count required')
    if any(type(x) is not int or not 0 <= x <= 100 for x in strengths):
        raise ValueError('Strengths must be integer percentages')
    output.mkdir(parents=True, exist_ok=False)
    (output / 'images').mkdir()
    sources, rows = source_specs(strengths), []

    def save(image, name):
        relative = f'images/{name}.png'
        image.save(output / relative)
        return relative, {'width': SIZE[0], 'height': SIZE[1], 'source_field': 'image_1',
            'bytes': (output / relative).stat().st_size, 'sha256': sha256_file(output / relative)}

    for pair, layout, assignment, replicate in product(range(2), range(2), range(2), range(replicates)):
        sid = f'onecolor-p{pair}-l{layout}-a{assignment}-r{replicate}'
        spec = scene_spec('color', pair, layout, assignment, replicate, seed)
        common_paths, common_metadata = {}, {}
        for name, ss in sources.items():
            if ss['type'] == 'main':
                with render(spec, ss) as image:
                    common_paths[name], common_metadata[name] = save(image, f'{sid}-{name}')
        common_paths['d_a'] = common_paths[main_source('M0', 100)]
        common_metadata['d_a'] = common_metadata[main_source('M0', 100)]
        for target_index, obj in enumerate(spec['objects']):
            paths, metadata = deepcopy(common_paths), deepcopy(common_metadata)
            with render(spec, sources['d_b'], obj) as image:
                paths['d_b'], metadata['d_b'] = save(image, f'{sid}-{obj}-d_b')
            value = spec['object_values_a'][obj]
            query = f'What color is the {obj}?'
            answers = {name: None if ss['strength'] == 0 else value ^ (ss['assignment'] == 'b')
                       for name, ss in sources.items()}
            target_slot = spec['object_to_slot'][obj]
            rows.append({'schema': ROW_SCHEMA, 'family': FAMILY, 'id': f'{sid}-{obj}', 'scene_id': sid,
                'task': 'single_object_color', 'attribute': 'color', 'shape_pair': pair, 'layout': layout,
                'assignment': assignment, 'replicate': replicate, 'generation_group': replicate,
                'target_object': obj, 'target_index': target_index, 'target_main_slot': target_slot,
                'target_main_slot_by_layout': {'M0': target_slot, 'M1': 1 - target_slot},
                'query': query, 'prompt': query + '\n(A) red\n(B) blue\n',
                'choices': ['red', 'blue'], 'option_labels': ['(A)', '(B)'],
                'answer_index': value, 'answer_label': ['(A)', '(B)'][value],
                'donor_roi_bbox_xyxy': bbox(spec, obj), 'donor_roi_coordinate_system': 'pixel_half_open_xyxy',
                'donor_roi_image_size': list(SIZE), 'donor_reference_layout': 'M0',
                'changed_attribute': 'color', 'changed_object': obj,
                'unchanged_object': spec['objects'][1 - target_index], 'scene': spec,
                'source_specs': sources, 'source_images': paths, 'source_image_metadata': metadata,
                'source_answer_indices': answers, 'record_fields': {'source_cluster_id': sid},
                'source': {'dataset': 'DeepStack procedural one-object counterfactuals', 'revision': ROW_SCHEMA,
                    'split': 'exploratory', 'seed': seed, 'generator_sha256': sha256_file(Path(__file__))}})
    manifest = output / 'manifest.jsonl'
    manifest.write_text(''.join(json.dumps(r, sort_keys=True) + '\n' for r in rows))
    result = validate(manifest)
    (output / 'dataset.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def load_manifest(manifest, image_root=None, *, check_hashes=True):
    manifest = _guard_path(Path(manifest))
    root = _guard_path(Path(image_root)) if image_root else manifest.parent
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    if not rows or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Empty or duplicate mechanism queries')
    checked = {}
    for row in rows:
        if row['schema'] != ROW_SCHEMA or row['family'] != FAMILY:
            raise ValueError('Unknown mechanism data schema')
        spec = scene_spec('color', *(row[k] for k in ('shape_pair', 'layout', 'assignment', 'replicate')), seed=row['source']['seed'])
        target = spec['objects'][row['target_index']]
        value = spec['object_values_a'][target]
        strengths = [round(ss['strength'] * 100) for ss in row['source_specs'].values() if ss['type'] == 'main']
        if row['scene'] != spec or row['source_specs'] != source_specs(strengths):
            raise ValueError('Scene/source specification mismatch')
        if row['target_object'] != target or row['donor_roi_bbox_xyxy'] != bbox(spec, target):
            raise ValueError('Target or donor ROI mismatch')
        if row['changed_object'] != target or row['unchanged_object'] != spec['objects'][1 - row['target_index']]:
            raise ValueError('Only the query target may change')
        expected = {n: None if s['strength'] == 0 else value ^ (s['assignment'] == 'b') for n, s in row['source_specs'].items()}
        if row['source_answer_indices'] != expected or row['answer_index'] != value or row['answer_label'] != row['option_labels'][value]:
            raise ValueError('Source answer identity mismatch')
        slot = spec['object_to_slot'][target]
        if row['target_main_slot'] != slot or row['target_main_slot_by_layout'] != {'M0': slot, 'M1': 1 - slot}:
            raise ValueError('M1 may only move the main layout')
        if row['choices'] != ['red', 'blue'] or row['option_labels'] != ['(A)', '(B)'] or row['generation_group'] != row['replicate']:
            raise ValueError('Choices or fixed generation subset changed')
        if set(row['source_images']) != set(row['source_specs']) or set(row['source_image_metadata']) != set(row['source_specs']):
            raise ValueError('Source image set mismatch')
        for name, rel in row['source_images'].items():
            rel = Path(rel)
            if rel.is_absolute() or '..' in rel.parts:
                raise ValueError('Contained portable image paths required')
            path = _guard_path(root / rel)
            if not path.is_relative_to(root) or not path.is_file():
                raise ValueError('Missing/escaping image')
            meta = row['source_image_metadata'][name]
            if (meta['width'], meta['height']) != SIZE:
                raise ValueError('Image geometry changed')
            if check_hashes:
                if path not in checked:
                    checked[path] = (path.stat().st_size, sha256_file(path))
                if checked[path] != (meta['bytes'], meta['sha256']):
                    raise ValueError('Image bytes/hash mismatch')
            row['source_images'][name] = str(path)
    return rows


def select_panel(rows, panel):
    if panel == 'main':
        return rows
    if panel != 'smoke':
        raise ValueError('Unknown panel')
    return [next(r for r in rows if r['shape_pair'] == pair and r['layout'] == layout
                 and r['assignment'] == assignment and r['replicate'] == 0 and r['target_index'] == target)
            for pair, layout, assignment, target in ((0, 0, 0, 0), (1, 1, 1, 1))]


def validate(manifest, image_root=None):
    from PIL import Image, ImageChops, ImageDraw
    rows = load_manifest(manifest, image_root)
    checked_main_pairs = set()
    for row in rows:
        with Image.open(row['source_images']['d_a']) as a, Image.open(row['source_images']['d_b']) as b:
            diff = ImageChops.difference(a, b)
            changed = diff.convert('L').point(lambda x: 255 if x else 0)
            expected_mask = Image.new('L', SIZE, 0)
            _shape(ImageDraw.Draw(expected_mask), row['target_object'], object_box(row['scene'], row['target_object']), fill=255)
            if changed.getbbox() != tuple(row['donor_roi_bbox_xyxy']) or ImageChops.difference(changed, expected_mask).getbbox():
                raise ValueError('Donor pair must change exactly the target object pixels')
            # Check unchanged object and color values by rendering the declared single change.
            for name in ('d_a', 'd_b'):
                with render(row['scene'], row['source_specs'][name], row['target_object']) as expected:
                    if ImageChops.difference(a if name == 'd_a' else b, expected).getbbox():
                        raise ValueError('Donor pixels disagree with scene identity')
            diff.close(); changed.close(); expected_mask.close()
        for name, ss in row['source_specs'].items():
            if ss['type'] != 'main' or ss['layout'] != 'M0':
                continue
            pair = (row['source_images'][name], row['source_images'][main_source('M1', round(ss['strength'] * 100))])
            if pair in checked_main_pairs:
                continue
            with Image.open(pair[0]) as m0, Image.open(pair[1]) as m1, swap_image_halves(m0) as swapped, render(row['scene'], ss) as expected:
                if ImageChops.difference(m0, expected).getbbox() or ImageChops.difference(swapped, m1).getbbox():
                    raise ValueError('M1 must be the exact half swap of the unchanged M0 image')
            checked_main_pairs.add(pair)
    images = {p for row in rows for p in row['source_images'].values()}
    return {'schema': ROW_SCHEMA, 'family': FAMILY, 'manifest': str(manifest),
        'manifest_sha256': sha256_file(manifest), 'status': 'passed', 'model_results': False,
        'items': len(rows), 'scene_templates': len({r['scene_id'] for r in rows}),
        'unique_image_paths': len(images), 'image_bytes': sum(Path(p).stat().st_size for p in images),
        'only_one_object_changed_pairs': len(rows), 'exact_main_half_swap_pairs': len(checked_main_pairs),
        'bbox_pixel_masks_checked': len(rows), 'generation_subset_rule': 'replicate in configured generation_groups',
        'independence_unit': 'scene_id', 'source_count_per_query': len(rows[0]['source_specs']),
        'roi_scope': 'Donor-coordinate nominal token footprint, not a finite effective receptive field'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('build', 'validate'))
    parser.add_argument('--output', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--image-root', type=Path)
    parser.add_argument('--strengths', type=int, nargs='+', default=[0, 8])
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--replicates', type=int, default=8)
    args = parser.parse_args(argv)
    if args.command == 'validate' and args.manifest is None:
        parser.error('--manifest required for validate')
    result = build(args.output, args.strengths, args.seed, args.replicates) if args.command == 'build' else validate(args.manifest, args.image_root)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
