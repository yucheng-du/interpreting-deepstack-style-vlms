"""Main-layout reference rows: v6 scenes + exact M1 main half swaps, behind a small family registry.

Normalized row contract (one query on one composed 384x256 two-half canvas; the
runner, planner and analyzer only rely on these fields; families may add more):

  schema, family        row schema string and registered family name
  id, scene_id, task    unique query id; canvas id shared by both target queries
  attribute             analysis stratum (e.g. color, stripe_orientation)
  target_object         object identity asked about; target_index 0/1 on the canvas
  target_main_slot      target's slot (0 left, 1 right) in the ORIGINAL layout (M0 main, same donors)
  target_main_slot_by_layout  {'M0': t, 'M1': 1 - t}
  query / prompt        question without options / question + '(A) x\\n(B) y\\n' block
  choices, option_labels, answer_index, answer_label
                        two content values; answer_index is the target's 'a' value
                        (its value in M0/M1 mains and the a donors); 'b' is the other
  source_specs          {name: {type main|donor|blank, layout M0|M1|None, strength 0..1,
                        pose same|cross, assignment a|b, ...}}; names: mains m_sNNN (M0) and
                        m1_sNNN (M1, derived_from m_sNNN, exact pixel half swap), donors
                        d_same_a/b and d_cross_a/b (D relative to the original layout;
                        cross = exact half swap of same), optional layout-invariant 'blank'
  source_images         {name: absolute path after loading}; manifests store contained relative paths
  source_image_metadata {name: {width 384, height 256, bytes, sha256, source_field}}
  source_answer_indices object identity: target value in that source (None without attribute evidence)
  source_original_slot_answer_indices  content at the target's original slot
  donor_current_slot_answer_indices    {M: {donor: content at the target's slot in main layout M}}
                        (all three maps must equal answer_maps(answer_index, source_specs);
                        the two canvas objects carry opposite values of the queried attribute)
  generation_group      int; design generation_groups selects the fixed generation subset
  record_fields         flat JSON scalars copied into every output record (unit keys such as
                        source_cluster_id live here; must not collide with core fields)
  source                provenance dict

register_family(name, schemas, load, smoke=None) adds a validator returning normalized
rows; load_rows dispatches on the row schema, an explicit family, or a dataset.json
'family_module' under cloud_prep.measurement_v7 next to the manifest, then runs check_rows.
normalize_rows(rows, family, record_fields=..., generation_group=...) adapts v6-style rows
(any main names, spec 'main_layout' 0/1 or 'layout' M0/M1) and cross-checks their answers.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import importlib
from itertools import product
import json
from pathlib import Path
import os as _path_os
DATA_ROOT = Path(_path_os.environ.get("DS_DATA_ROOT", "work")).expanduser().resolve()
import shutil

from cloud_prep.portable_data import _guard_path, sha256_file
from cloud_prep.measurement_v6.content_data import (SIZE, ATTRIBUTES, scene_spec, swap_image_halves,
    load_manifest as load_v6_manifest)
from cloud_prep.measurement_v6.content_intervention import (STRENGTHS, POSES, donor_source,
    select_panel as v6_select_panel)
from cloud_prep.measurement_v7.spatial_design import DESIGN_SCHEMA, LAYOUTS, main_source

ROW_SCHEMA = 'main-layout-reference-v7-shapes'
FAMILY = 'synthetic_shapes'
MODULE = 'cloud_prep.measurement_v7.spatial_data'  # stable even when run with -m
V6_MANIFEST = (DATA_ROOT / 'datasets/content-routing-v6/manifest.jsonl')
DONORS = tuple(donor_source(pose, value) for pose, value in product(POSES, ('a', 'b')))
CORE_FIELDS = ('schema', 'family', 'id', 'scene_id', 'task', 'attribute', 'target_object', 'target_index',
    'target_main_slot', 'target_main_slot_by_layout', 'query', 'prompt', 'choices', 'option_labels',
    'answer_index', 'answer_label', 'source_specs', 'source_images', 'source_image_metadata',
    'source_answer_indices', 'source_original_slot_answer_indices', 'donor_current_slot_answer_indices',
    'generation_group', 'record_fields', 'source')
# Top-level keys written by qwen_spatial records; record_fields may not shadow them.
RESERVED_RECORD_FIELDS = frozenset(CORE_FIELDS) | {'row_schema', 'execution_schema', 'kind', 'arm', 'mapping',
    'response_format', 'condition', 'condition_spec', 'main_source', 'main_layout', 'main_kind', 'main_strength',
    'donor_pose', 'token_permutation', 'ds_sources', 'ds_code', 'template_signs', 'original_prediction',
    'content_prediction', 'prediction', 'scores', 'b_minus_a_log_odds', 'native_source', 'native_answer_label',
    'native_correct', 'generation_planned', 'unit', 'unit_key'}
FAMILIES = {}


def source_specs(strengths=STRENGTHS, include_blank=True):
    sources = {}
    for s in sorted(set(strengths)):
        m0, m1 = main_source('M0', s), main_source('M1', s)
        sources[m0] = {'type': 'main', 'layout': 'M0', 'strength': s / 100, 'pose': 'same', 'assignment': 'a',
                       'outline': False, 'derived_from': None, 'transform': None}
        sources[m1] = {**sources[m0], 'layout': 'M1', 'derived_from': m0, 'transform': 'swap_equal_horizontal_halves'}
    for pose, value in product(POSES, ('a', 'b')):
        cross = pose == 'cross'
        sources[donor_source(pose, value)] = {'type': 'donor', 'layout': None, 'strength': 1., 'pose': pose,
            'assignment': value, 'outline': False, 'derived_from': donor_source('same', value) if cross else None,
            'transform': 'swap_equal_horizontal_halves' if cross else None}
    if include_blank:
        sources['blank'] = {'type': 'blank', 'layout': None, 'strength': 0., 'pose': 'same', 'assignment': 'a',
                            'outline': False, 'derived_from': None, 'transform': None, 'layout_invariant': True}
    return sources


def answer_maps(value, sources):
    """Object-identity, original-slot and current-main-slot answers for the target.

    value is the target's 'a' content index. Moving pixels (cross donor, M1 main)
    carries the target with them, so object identity is unchanged while the
    content found at the target's original slot becomes the other object's value.
    """
    identity, original = {}, {}
    for name, s in sources.items():
        if s['type'] == 'blank' or s['strength'] == 0:
            identity[name] = original[name] = None
            continue
        identity[name] = value ^ (s['assignment'] == 'b')
        moved = s['pose'] == 'cross' if s['type'] == 'donor' else s['layout'] == 'M1'
        original[name] = identity[name] ^ moved
    current = {layout: {name: original[name] ^ (layout == 'M1') for name, s in sources.items() if s['type'] == 'donor'}
               for layout in LAYOUTS}
    return identity, original, current


def normalize_rows(rows, family, *, record_fields=(), generation_group='replicate', slot_field='source_slot_answer_indices'):
    """Adapter for v6-style family rows into the normalized contract.

    Mains are recognized by type 'main' plus 'layout' (M0/M1) or 'main_layout' (0/1); their
    names are rebuilt as m_sNNN / m1_sNNN, so any family naming works. Donor/blank names stay.
    Answer maps are recomputed with answer_maps and must equal the family's own object-identity
    map and (if present) original-slot map, after renaming. record_fields names scalar row
    fields to carry (e.g. unit_id); generation_group is a row field name or a callable.
    """
    out = []
    for row in rows:
        rename, specs = {}, {}
        for name, s in row['source_specs'].items():
            if s['type'] == 'main':
                raw = s.get('layout', s.get('main_layout'))
                layout = {0: 'M0', 'M0': 'M0', None: 'M0', 1: 'M1', 'M1': 'M1'}.get(raw)
                strength = round(s['strength'] * 100)
                if layout is None or abs(s['strength'] * 100 - strength) > 1e-9:
                    raise ValueError(f'{row["id"]}: main {name} needs layout M0/M1 and a percent strength')
                new = main_source(layout, strength)
                spec = {k: v for k, v in s.items() if k != 'main_layout'}
                spec.update(layout=layout, pose='same', derived_from=main_source('M0', strength) if layout == 'M1' else None,
                            transform='swap_equal_horizontal_halves' if layout == 'M1' else None)
            else:
                new, spec = name, {**s, 'layout': None}
            if new in specs:
                raise ValueError(f'{row["id"]}: two sources normalize to {new}')
            rename[name], specs[new] = new, spec
        identity, original, current = answer_maps(row['answer_index'], specs)
        if {rename[n]: v for n, v in row['source_answer_indices'].items()} != identity or \
                (slot_field in row and {rename[n]: v for n, v in row[slot_field].items()} != original):
            raise ValueError(f'{row["id"]}: family answers disagree with the normalized object/original-slot maps')
        t = row['target_main_slot']
        drop = {'source_specs', 'source_images', 'source_image_metadata', 'source_answer_indices', slot_field}
        out.append({**{k: v for k, v in row.items() if k not in drop}, 'family': family, 'source_specs': specs,
            'source_images': {rename[n]: p for n, p in row['source_images'].items()},
            'source_image_metadata': {rename[n]: m for n, m in row['source_image_metadata'].items()},
            'source_answer_indices': identity, 'source_original_slot_answer_indices': original,
            'donor_current_slot_answer_indices': current, 'target_main_slot_by_layout': {'M0': t, 'M1': 1 - t},
            'generation_group': generation_group(row) if callable(generation_group) else row[generation_group],
            'record_fields': {k: row[k] for k in record_fields}, 'source_name_map': rename})
    return out


def _image_meta(path, origin):
    return {'source_field': 'image_1', 'width': SIZE[0], 'height': SIZE[1], 'bytes': path.stat().st_size,
            'sha256': sha256_file(path), 'origin': origin}


def build(output, v6_manifest=V6_MANIFEST, overwrite=False):
    from PIL import Image
    output = _guard_path(Path(output), write=True)
    output.mkdir(parents=True, exist_ok=overwrite)
    (output / 'images').mkdir(exist_ok=overwrite)
    v6_manifest = _guard_path(Path(v6_manifest))
    v6_rows = load_v6_manifest(v6_manifest)
    strengths = sorted(round(s['strength'] * 100) for s in v6_rows[0]['source_specs'].values() if s['type'] == 'main')
    sources = source_specs(strengths)
    v6_sha, generator_sha = sha256_file(v6_manifest), sha256_file(Path(__file__))
    scenes, rows = {}, []
    for v6 in v6_rows:
        sid = v6['scene_id']
        if sid not in scenes:
            paths, metadata = {}, {}
            for name, spec in sources.items():
                relative = f'images/{sid}-{name}.png'
                if spec['layout'] == 'M1':
                    with Image.open(v6['source_images'][spec['derived_from']]) as m0, swap_image_halves(m0) as m1:
                        m1.save(output / relative)
                    metadata[name] = _image_meta(output / relative, 'v7_exact_half_swap_of_' + spec['derived_from'])
                else:
                    shutil.copyfile(v6['source_images'][name], output / relative)
                    metadata[name] = _image_meta(output / relative, 'v6_byte_copy')
                    if (metadata[name]['bytes'], metadata[name]['sha256']) != tuple(v6['source_image_metadata'][name][k] for k in ('bytes', 'sha256')):
                        raise ValueError('Copied v6 image differs from the v6 manifest')
                paths[name] = relative
            scenes[sid] = paths, metadata
        paths, metadata = scenes[sid]
        value, slot = v6['answer_index'], v6['target_main_slot']
        identity, original, current = answer_maps(value, sources)
        if any(v6['source_answer_indices'][n] != identity[n] or v6['source_slot_answer_indices'][n] != original[n]
               for n in v6['source_answer_indices']):
            raise ValueError('v7 answer maps disagree with v6 on reused sources')
        rows.append({'schema': ROW_SCHEMA, 'family': FAMILY, 'design_schema': DESIGN_SCHEMA, 'id': v6['id'],
            'scene_id': sid, 'task': 'main_reference_' + v6['attribute'],
            **{k: v6[k] for k in ('attribute', 'shape_pair', 'layout', 'assignment', 'replicate', 'target_object',
                                  'target_index', 'scene', 'query', 'prompt', 'choices', 'option_labels',
                                  'answer_index', 'answer_label')},
            'target_main_slot': slot, 'target_main_slot_by_layout': {'M0': slot, 'M1': 1 - slot},
            'source_answer_indices': identity, 'source_original_slot_answer_indices': original,
            'donor_current_slot_answer_indices': current, 'source_specs': sources, 'source_images': paths,
            'source_image_metadata': metadata, 'generation_group': v6['replicate'],
            'record_fields': {'shape_pair': v6['shape_pair'], 'scene_layout': v6['layout'],
                              'assignment': v6['assignment'], 'replicate': v6['replicate']},
            'v6_pairing': {'v6_schema': v6['schema'], 'v6_id': v6['id'], 'v6_manifest_sha256': v6_sha,
                           'reused_sources': sorted(v6['source_images'])},
            'source': {'dataset': 'DeepStack procedural counterfactuals', 'revision': ROW_SCHEMA, 'split': 'exploratory',
                       'seed': v6['source']['seed'], 'generator_sha256': generator_sha,
                       'v6_generator_sha256': v6['source']['generator_sha256']}})
    manifest = output / 'manifest.jsonl'
    manifest.write_text(''.join(json.dumps(r, sort_keys=True) + '\n' for r in rows))
    result = {**validate(manifest), 'v6_pairing': verify_v6_pairing(manifest, v6_manifest),
              'family_module': MODULE, 'portable_paths': 'images/<scene>-<source>.png relative to the manifest'}
    (output / 'dataset.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def load_manifest(manifest, image_root=None, *, check_hashes=True):
    """Synthetic family validator: recompute scenes, specs, answers; check contained images."""
    manifest = _guard_path(Path(manifest))
    root = _guard_path(Path(image_root)) if image_root else manifest.parent
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    if not rows or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Empty/duplicate manifest')
    checked, result = {}, []
    for row in rows:
        if row.get('schema') != ROW_SCHEMA or row.get('family') != FAMILY:
            raise ValueError('Unknown schema/family for the synthetic loader')
        strengths = [round(s['strength'] * 100) for s in row['source_specs'].values() if s['type'] == 'main' and s['layout'] == 'M0']
        declared = source_specs(strengths, 'blank' in row['source_specs'])
        if row['source_specs'] != declared:
            raise ValueError('Unknown source design')
        spec = scene_spec(*(row[k] for k in ('attribute', 'shape_pair', 'layout', 'assignment', 'replicate')), seed=row['source']['seed'])
        if spec != row['scene'] or row['target_object'] not in spec['objects'] or spec['objects'].index(row['target_object']) != row['target_index']:
            raise ValueError('Scene/object disagreement')
        value, slot = spec['object_values_a'][row['target_object']], spec['object_to_slot'][row['target_object']]
        identity, original, current = answer_maps(value, declared)
        if (row['source_answer_indices'], row['source_original_slot_answer_indices'], row['donor_current_slot_answer_indices']) != (identity, original, current):
            raise ValueError('Object/original-slot/current-slot answers disagree')
        if row['answer_index'] != value or row['answer_label'] != row['option_labels'][value] or row['choices'] != list(ATTRIBUTES[row['attribute']]):
            raise ValueError('Content answer/choice mismatch')
        if row['target_main_slot'] != slot or row['target_main_slot_by_layout'] != {'M0': slot, 'M1': 1 - slot}:
            raise ValueError('Target slot mismatch')
        if row['generation_group'] != row['replicate'] or row['record_fields'] != {'shape_pair': row['shape_pair'],
                'scene_layout': row['layout'], 'assignment': row['assignment'], 'replicate': row['replicate']}:
            raise ValueError('Generation group/record fields mismatch')
        if set(row['source_images']) != set(declared) or set(row['source_image_metadata']) != set(declared):
            raise ValueError('Missing/extra source images')
        loaded = deepcopy(row)
        for name, path in row['source_images'].items():
            rel = Path(path)
            if rel.is_absolute() or '..' in rel.parts:
                raise ValueError('Images must use contained portable paths')
            path = _guard_path(root / rel)
            if not path.is_relative_to(root) or not path.is_file():
                raise ValueError('Image missing or escaping data root')
            meta = row['source_image_metadata'][name]
            if (meta['width'], meta['height']) != SIZE:
                raise ValueError('Fixed source geometry changed')
            if check_hashes:
                if path not in checked:
                    checked[path] = (path.stat().st_size, sha256_file(path))
                if checked[path] != (meta['bytes'], meta['sha256']):
                    raise ValueError('Image bytes/SHA changed')
            loaded['source_images'][name] = str(path)
        result.append(loaded)
    return result


def check_rows(rows):
    """Family-independent normalized-row contract (see module docstring)."""
    if not rows or len({r.get('id') for r in rows}) != len(rows):
        raise ValueError('Empty rows or duplicate ids')
    canvases = {}
    for row in rows:
        missing = [k for k in CORE_FIELDS if k not in row]
        if missing:
            raise ValueError(f'{row.get("id")}: missing contract fields {missing}')
        if len(row['choices']) != 2 or row['option_labels'] != ['(A)', '(B)'] or row['answer_index'] not in (0, 1) \
                or row['answer_label'] != row['option_labels'][row['answer_index']]:
            raise ValueError(f'{row["id"]}: two choices with (A)/(B) labels and an a-value answer required')
        t = row['target_main_slot']
        if row['target_index'] not in (0, 1) or t not in (0, 1) or row['target_main_slot_by_layout'] != {'M0': t, 'M1': 1 - t}:
            raise ValueError(f'{row["id"]}: invalid target slot/index')
        specs = row['source_specs']
        for name, s in specs.items():
            if not {'type', 'layout', 'strength', 'pose', 'assignment'} <= set(s) or s['type'] not in ('main', 'donor', 'blank') \
                    or not 0 <= s['strength'] <= 1:
                raise ValueError(f'{row["id"]}: invalid source spec {name}')
            if s['type'] == 'main':
                strength = round(s['strength'] * 100)
                if abs(s['strength'] * 100 - strength) > 1e-9 or name != main_source(s['layout'], strength) \
                        or s['pose'] != 'same' or s['assignment'] != 'a':
                    raise ValueError(f'{row["id"]}: main source naming/spec mismatch {name}')
                other = main_source('M1' if s['layout'] == 'M0' else 'M0', strength)
                if other not in specs or (s['layout'] == 'M1' and s.get('derived_from') != other):
                    raise ValueError(f'{row["id"]}: every strength needs an M0 main and its exact M1 half swap')
            elif s['type'] == 'donor':
                if name not in DONORS or name != donor_source(s['pose'], s['assignment']) or s['layout'] is not None or s['strength'] == 0:
                    raise ValueError(f'{row["id"]}: donor naming/spec mismatch {name}')
            elif name != 'blank' or s['strength'] != 0:
                raise ValueError(f'{row["id"]}: only a zero-strength source named blank may be blank')
        if set(DONORS) - set(specs) or set(row['source_images']) != set(specs) or set(row['source_image_metadata']) != set(specs):
            raise ValueError(f'{row["id"]}: source names, images and metadata must match and include all donors')
        for name, path in row['source_images'].items():
            meta = row['source_image_metadata'][name]
            if (meta.get('width'), meta.get('height')) != SIZE or not Path(path).is_absolute() or not Path(path).is_file():
                raise ValueError(f'{row["id"]}: {name} must be a loaded 384x256 image path')
        expected = answer_maps(row['answer_index'], specs)
        if (row['source_answer_indices'], row['source_original_slot_answer_indices'], row['donor_current_slot_answer_indices']) != expected:
            raise ValueError(f'{row["id"]}: answer maps differ from answer_maps(answer_index, source_specs)')
        if type(row['generation_group']) is not int or row['generation_group'] < 0:
            raise ValueError(f'{row["id"]}: nonnegative integer generation_group required')
        fields = row['record_fields']
        if not isinstance(fields, dict) or set(fields) & RESERVED_RECORD_FIELDS \
                or any(not isinstance(v, (str, int, float, bool, type(None))) for v in fields.values()):
            raise ValueError(f'{row["id"]}: record_fields must be flat JSON scalars outside reserved names')
        images = json.dumps(row['source_images'], sort_keys=True)
        if canvases.setdefault(row['scene_id'], images) != images:
            raise ValueError(f'{row["scene_id"]}: queries of one canvas must share source images')
    return rows


def register_family(name, schemas, load, smoke=None, module=None):
    if not name or not schemas or not callable(load) or (smoke is not None and not callable(smoke)):
        raise ValueError('Family needs a name, schemas and a callable loader')
    for other in FAMILIES.values():
        if other['name'] != name and set(other['schemas']) & set(schemas):
            raise ValueError('Schema already registered by another family')
    FAMILIES[name] = {'name': name, 'schemas': tuple(schemas), 'load': load, 'smoke': smoke, 'module': module}
    return FAMILIES[name]


def import_family_module(module):
    if not module.startswith('cloud_prep.measurement_v7.'):
        raise ValueError('Family modules must live under cloud_prep.measurement_v7')
    return importlib.import_module(module)


def family_for_schema(schema):
    names = [f['name'] for f in FAMILIES.values() if schema in f['schemas']]
    if len(names) != 1:
        raise ValueError(f'No registered family for row schema {schema!r}; pass --family-module')
    return names[0]


def load_rows(manifest, image_root=None, *, family=None, modules=(), check_hashes=True):
    """Dispatch to a registered family validator, then enforce the normalized contract."""
    for module in modules:
        import_family_module(module)
    manifest = Path(manifest)
    first = next((line for line in manifest.read_text().splitlines() if line.strip()), None)
    if first is None:
        raise ValueError('Empty manifest')
    schema = json.loads(first)['schema']
    if family is None and not any(schema in f['schemas'] for f in FAMILIES.values()):
        described = manifest.parent / 'dataset.json'
        if described.is_file() and json.loads(described.read_text()).get('family_module'):
            import_family_module(json.loads(described.read_text())['family_module'])
    family = family or family_for_schema(schema)
    if family not in FAMILIES or schema not in FAMILIES[family]['schemas']:
        raise ValueError('Manifest schema does not belong to the selected family')
    return family, check_rows(FAMILIES[family]['load'](manifest, image_root, check_hashes=check_hashes))


def default_smoke(rows):
    """First query per attribute with alternating target index (at least two queries); fixed before model output."""
    chosen = []
    for i, attribute in enumerate(sorted({r['attribute'] for r in rows})):
        members = [r for r in rows if r['attribute'] == attribute]
        chosen.append(next((r for r in members if r['target_index'] == i % 2), members[0]))
    if len(chosen) == 1:
        extra = [r for r in rows if r['scene_id'] != chosen[0]['scene_id']]
        chosen += [next((r for r in extra if r['target_index'] != chosen[0]['target_index']), extra[0])] if extra else []
    return chosen


def select_panel(rows, panel, family=FAMILY):
    if panel == 'main':
        return rows
    if panel != 'smoke':
        raise ValueError('Unknown panel')
    return (FAMILIES[family]['smoke'] or default_smoke)(rows)


def verify_v6_pairing(manifest, v6_manifest=V6_MANIFEST):
    """Reused v6 sources byte-identical; every M1 main an exact pixel half swap of its M0 main."""
    from PIL import Image, ImageChops
    rows, v6 = load_manifest(manifest), {r['id']: r for r in load_v6_manifest(v6_manifest)}
    if set(v6) != {r['id'] for r in rows}:
        raise ValueError('v7 rows must pair one-to-one with v6 queries')
    copied = swapped = 0
    for row in rows:
        old = v6[row['id']]
        if row['scene'] != old['scene'] or any(row[k] != old[k] for k in ('query', 'prompt', 'choices', 'answer_index', 'target_main_slot')):
            raise ValueError('Scene/query changed relative to v6')
        for name, spec in row['source_specs'].items():
            if spec['layout'] == 'M1':
                with Image.open(row['source_images'][name]) as m1, Image.open(row['source_images'][spec['derived_from']]) as m0, swap_image_halves(m0) as expected:
                    if m1.size != SIZE or ImageChops.difference(m1.convert('RGB'), expected.convert('RGB')).getbbox() is not None:
                        raise ValueError('M1 main is not an exact half swap')
                swapped += 1
            else:
                if sha256_file(row['source_images'][name]) != old['source_image_metadata'][name]['sha256'] \
                        or row['source_image_metadata'][name]['sha256'] != old['source_image_metadata'][name]['sha256']:
                    raise ValueError('Reused source differs from v6 bytes')
                copied += 1
    return {'v6_manifest': str(v6_manifest), 'v6_manifest_sha256': sha256_file(v6_manifest), 'paired_queries': len(rows),
            'reused_source_checks': copied, 'reused_unique_images': len({r['source_images'][n] for r in rows for n in r['source_images'] if r['source_specs'][n]['layout'] != 'M1'}),
            'm1_half_swap_checks': swapped, 'status': 'passed'}


def validate(manifest, image_root=None, family=None, modules=()):
    family, rows = load_rows(manifest, image_root, family=family, modules=modules)
    images = {p for r in rows for p in r['source_images'].values()}
    m1 = {r['source_images'][n] for r in rows for n, s in r['source_specs'].items() if s['layout'] == 'M1'}
    return {'schema': rows[0]['schema'], 'family': family, 'design_schema': DESIGN_SCHEMA, 'manifest': str(manifest),
        'manifest_sha256': sha256_file(manifest), 'items': len(rows), 'scene_templates': len({r['scene_id'] for r in rows}),
        'unique_image_paths': len(images), 'm1_main_images': len(m1),
        'image_bytes': sum(Path(p).stat().st_size for p in images), 'status': 'passed', 'model_results': False,
        'source_count_per_scene': len(rows[0]['source_specs']),
        'main_strengths': sorted(round(s['strength'] * 100) for s in rows[0]['source_specs'].values() if s['type'] == 'main' and s['layout'] == 'M0'),
        'semantics': 'Same 64 v6 scenes, queries and donors. M1 = exact left/right pixel half swap of the M0 main only; D stays relative to the original layout. Scene is the independent unit; M is a within-scene manipulation, not a new sample.'}


register_family(FAMILY, (ROW_SCHEMA,), load_manifest, smoke=lambda rows: v6_select_panel(rows, 'smoke'), module=MODULE)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('command', choices=('build', 'validate'))
    p.add_argument('--output', type=Path)
    p.add_argument('--manifest', type=Path)
    p.add_argument('--image-root', type=Path)
    p.add_argument('--v6-manifest', type=Path, default=V6_MANIFEST)
    p.add_argument('--family')
    p.add_argument('--family-module', action='append', default=[])
    a = p.parse_args(argv)
    if a.command == 'build' and a.output is None:
        p.error('build requires --output')
    if a.command == 'validate' and a.manifest is None:
        p.error('validate requires --manifest')
    result = build(a.output, a.v6_manifest) if a.command == 'build' else validate(a.manifest, a.image_root, a.family, a.family_module)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
