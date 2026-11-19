#!/usr/bin/env python3
"""CPU synthetic materials for v15 severity curves; never runs a model.

New-seed templates on a 448x336 canvas, sized for native legibility: large
objects, 20-24 px stripe periods, 32-40 px dot separations, 36-44 px codes.
Each intervention changes one named object (source b). Every severity of the
task's family (cloud_prep/measurement_v15/severity.py) degrades the clean source
a once, on the same canvas, and all paired cells name that file. Calibration and
evaluation are separate runs with their own seeds and unit ids; both carry the
full candidate grid, so the post-calibration evaluation grid is a run-config
choice, not a regeneration. --check validates written manifests (v15 reader,
three two-choice readouts, re-derived degradations, split disjointness).
"""
from __future__ import annotations
import argparse, datetime, functools, hashlib, itertools, json, random, sys
from collections import Counter
from pathlib import Path
import PIL
from PIL import Image, ImageChops, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from cloud_prep.measurement_v15 import severity  # noqa: E402

TASKS = ('color', 'stripe_orientation', 'relative_position', 'text_reading')
SIZE = (448, 336)
OBJECTS = ('circle', 'square')
SPLITS = {'calibration': dict(short='cal', templates=16, seed=2026102911),
          'evaluation': dict(short='eval', templates=32, seed=2026102921)}
# ITU-R 601 luma 96-97 for every color: desat-c0 leaves no luminance cue to the answer.
COLORS = {'red': (230, 40, 40), 'blue': (45, 92, 255), 'green': (20, 150, 20), 'purple': (170, 40, 190)}
COLOR_PAIRS = (('red', 'blue'), ('green', 'purple'), ('red', 'green'))
LETTERS, DIGITS = 'ACEFHJKLMNPRTUVXY', '234679'
QUERIES = {'color': 'What color is the {}?', 'stripe_orientation': 'What pattern is inside the {}?',
           'relative_position': 'Is the black dot to the left or right of the white dot inside the {}?',
           'text_reading': 'What two-character code is written inside the {}?'}
READOUTS = (('direct', 'none'), ('label', 'identity'), ('label', 'cyclic'))
STIMULUS = dict(canvas=list(SIZE), object_radius_px=[72, 84], slot_center_x_px=[112, 336], center_x_jitter_px=12,
    center_y_px=[150, 186], background_gray=[224, 244], object_fill_gray=dict(relative_position=128, text_reading=205),
    outline=dict(gray=65, width_px=3),
    color_rgb=COLORS, color_luma='ITU-R 601 (Pillow L) 96-97 for all colors', color_pairs=[list(p) for p in COLOR_PAIRS], stripe_period_px=[20, 22, 24], stripe_line_px='period//2, gray 45 on 235',
    dot_separation_px=[32, 36, 40], dot_radius_px=10, dots='white at the object center, black left/right of it',
    font_px=[36, 38, 40, 42, 44],
    codes='letter+digit; both characters differ between the two objects', code_letters=LETTERS, code_digits=DIGITS,
    balance='circle on the left iff template index is even; circle value = (index//2)%2; the square takes the other option',
    font_px_meaning='TrueType em size in pixels', rng='random.Random(sha256((seed, (split, template_index))))')


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def key(seed, *parts): return int.from_bytes(hashlib.sha256(str((seed, parts)).encode()).digest()[:8], 'big')
def dump(path, value): Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
def read_rows(path): return [json.loads(x) for x in Path(path).read_text().split('\n') if x.strip()]  # physical lines
@functools.lru_cache(maxsize=None)
def truetype(path, size): return ImageFont.truetype(str(path), size)
def font_path():
    for p in ('/System/Library/Fonts/Supplemental/Arial.ttf', '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'):
        if Path(p).is_file(): return Path(p)
    raise ValueError('Set --font to a TrueType font installed on this host')
def save(image, path):
    image.save(path)
    return {'width': image.width, 'height': image.height, 'sha256': sha(path), 'bytes': path.stat().st_size}
def shape(draw, name, box, **kwargs):
    (draw.ellipse if name == 'circle' else draw.rectangle)(box, **kwargs)
def output_dir(path):
    path = Path(path).resolve()
    return path


def template(seed, split, n):
    rng = random.Random(key(seed, split, n)); order, value = n % 2, (n//2) % 2
    radius = rng.randrange(72, 85); boxes = []
    for slot in (order, 1-order):  # circle, square
        cx, cy = 112+224*slot+rng.randrange(-12, 13), rng.randrange(150, 187)
        boxes.append([cx-radius, cy-radius, cx+radius, cy+radius])
    period = rng.choice((20, 22, 24)); letters, digits = rng.sample(LETTERS, 2), rng.sample(DIGITS, 2)
    return dict(order=order, boxes=boxes, values=[value, 1-value], background=rng.randrange(224, 245),
        color_pair=list(rng.choice(COLOR_PAIRS)), stripe_period_px=period, stripe_line_px=period//2, phase=rng.randrange(period),
        dot_separation_px=rng.choice((32, 36, 40)), font_px=rng.randrange(36, 45, 2), codes=[a+b for a, b in zip(letters, digits)])

def choices(spec, task):
    return {'color': spec['color_pair'], 'stripe_orientation': ['vertical stripes', 'horizontal stripes'],
            'relative_position': ['left', 'right'], 'text_reading': spec['codes']}[task]

def scales(spec, task):
    """Stimulus scales of one task; DEFAULT_GRIDS[task]['reference_kind'] names the severity reference."""
    found = {'long_side_px': max(SIZE), 'object_radius_px': (spec['boxes'][0][2]-spec['boxes'][0][0])//2}
    found.update({'stripe_orientation': {k: spec[k] for k in ('stripe_period_px', 'stripe_line_px')},
                  'relative_position': {'dot_separation_px': spec['dot_separation_px']},
                  'text_reading': {'font_px': spec['font_px']}}.get(task, {}))
    return found

def render(spec, task, font, changed=None):
    """Clean source a, or counterfactual b when `changed` names the flipped object."""
    image = Image.new('RGB', SIZE, (spec['background'],)*3)
    for j, name in enumerate(OBJECTS):
        box = spec['boxes'][j]; val = spec['values'][j] ^ (j == changed); cx, cy = (box[0]+box[2])/2, (box[1]+box[3])/2
        texture = Image.new('RGB', SIZE, (128 if task == 'relative_position' else 205,)*3); d = ImageDraw.Draw(texture)
        if task == 'color':
            texture.paste(COLORS[spec['color_pair'][val]], (0, 0, *SIZE))
        elif task == 'stripe_orientation':
            texture.paste((235,)*3, (0, 0, *SIZE)); p = spec['stripe_period_px']
            for off in range(spec['phase']-p, max(SIZE)+p, p):
                d.line((off, 0, off, SIZE[1]) if val == 0 else (0, off, SIZE[0], off), fill=(45,)*3, width=spec['stripe_line_px'])
        elif task == 'relative_position':
            x = cx+(-1 if val == 0 else 1)*spec['dot_separation_px']
            d.ellipse((cx-10, cy-10, cx+10, cy+10), fill='white')  # +-127 around the fill: both dots fade alike
            d.ellipse((x-10, cy-10, x+10, cy+10), fill='black')
        else:
            d.text((cx, cy), spec['codes'][val], font=truetype(font, spec['font_px']), fill='black', anchor='mm')
        mask = Image.new('L', SIZE); shape(ImageDraw.Draw(mask), name, box, fill=255)
        image.paste(texture, (0, 0), mask)
        shape(ImageDraw.Draw(image), name, box, outline=(65,)*3, width=3)
    return image


def exclusions(manifests):
    units, clean, info = set(), set(), []
    for m in manifests:
        rows = read_rows(m); u = {r['unit_id'] for r in rows}; c = {r['source_image_metadata']['a']['sha256'] for r in rows}
        units |= u; clean |= c
        info.append(dict(manifest=str(Path(m).resolve()), sha256=sha(m), splits=sorted({str(r.get('split')) for r in rows}),
                         units=len(u), clean_images=len(c)))
    return units, clean, info


def prepare(args):
    root = output_dir(args.output); stage = root.with_name(root.name+'.partial')
    previews = output_dir(args.previews) if args.previews else None
    if root.exists() or stage.exists(): raise FileExistsError(f'Refusing to overwrite {root} (or {stage.name}); choose a new sibling')
    font = Path(args.font).resolve() if args.font else font_path()
    labels = {t: [s['label'] for s in severity.grid(t, args.severities, args.values.get(t))] for t in args.tasks}
    short = SPLITS[args.split]['short']; units = [f'v15-{short}-s{args.seed}-t{n:04d}' for n in range(args.templates)]
    excluded_units, excluded_clean, excluded_info = exclusions(args.disjoint_from)
    if excluded_units & set(units): raise ValueError('Unit ids overlap a --disjoint-from manifest: '+str(sorted(excluded_units & set(units))[:3]))
    sheets = [previews/f'{args.split}-{t}.png' for t in args.tasks]+[previews/f'{args.split}-overview.png'] if previews else []
    if any(p.exists() for p in sheets): raise FileExistsError('Refusing to overwrite contact sheets in '+str(previews))
    stage.mkdir(parents=True); (stage/'images').mkdir()
    provenance = {'dataset': 'DeepStack procedural followup', 'revision': 'v15', 'seed': args.seed, 'split': args.split,
        'generator_sha256': sha(__file__), 'severity_module_sha256': sha(severity.__file__), 'severity_generator': severity.GENERATOR,
        'font_sha256': sha(font), 'selection_uses_model_outputs': False}
    rows, specs, images, owners, realized = [], [], Counter(), {}, Counter()
    for n, tid in enumerate(units):
        spec = template(args.seed, args.split, n); specs.append({'unit_id': tid, **spec})
        realized.update([('stripe_period_px', spec['stripe_period_px']), ('dot_separation_px', spec['dot_separation_px']),
                         ('font_px', spec['font_px']), ('color_pair', '/'.join(spec['color_pair'])), ('order_value', f"{spec['order']}{spec['values'][0]}")])
        for task in args.tasks:
            ch, sc = choices(spec, task), scales(spec, task); kind = severity.DEFAULT_GRIDS[task]['reference_kind']
            a = render(spec, task, font); degraded = {x: severity.apply(a, x, None if kind is None else sc[kind]) for x in labels[task]}
            for changed in range(2):
                iid = f'{tid}-{task}-edit{changed}'; b = render(spec, task, font, changed)
                box = spec['boxes'][changed]; diff = ImageChops.difference(a, b).getbbox()
                if diff is None or not (box[0] <= diff[0] and box[1] <= diff[1] and diff[2] <= box[2]+1 and diff[3] <= box[3]+1):
                    raise AssertionError('Counterfactual escaped its object ROI: '+iid)
                paths, meta = {}, {}
                for name, image in [('a', a), ('b', b), *((x, degraded[x][0]) for x in labels[task])]:
                    rel = f'images/{iid}-{name}.png'; paths[name] = rel; meta[name] = save(image, stage/rel)
                    images['degraded' if name in degraded else name] += 1
                if len({(m['width'], m['height']) for m in meta.values()}) != 1: raise AssertionError('Sources differ in canvas: '+iid)
                clean = meta['a']['sha256']
                if clean in excluded_clean: raise ValueError(f'Clean image of {iid} also occurs in a --disjoint-from manifest')
                if owners.setdefault((task, clean), tid) != tid: raise ValueError(f'{iid} repeats the clean image of {owners[(task, clean)]}')
                for queried, name in enumerate(OBJECTS):
                    val = spec['values'][queried]; role = 'changed' if queried == changed else 'unchanged'
                    rows.append({'schema': 'followup-material-v15', 'id': f'{iid}-q{queried}', 'unit_id': tid, 'template_id': tid,
                        'scene_id': tid, 'intervention_id': iid, 'split': args.split, 'material_set': f'v15-synthetic-{args.split}',
                        'task': task, 'attribute': task, 'query': QUERIES[task].format(name), 'choices': ch, 'option_labels': ['(A)', '(B)'],
                        'answer_index': val, 'answer_label': ['(A)', '(B)'][val], 'reference_answers': [ch[val]],
                        'direction_a_index': spec['values'][changed], 'direction_b_index': 1-spec['values'][changed],
                        'query_role': role, 'query_target': role, 'changed_index': changed, 'changed_object': OBJECTS[changed],
                        'queried_index': queried, 'queried_object': name,
                        'donor_roi_bbox_xyxy': [box[0], box[1], box[2]+1, box[3]+1], 'donor_roi_image_size': list(SIZE),
                        'object_bbox_xyxy': {o: [x0, y0, x1+1, y1+1] for o, (x0, y0, x1, y1) in zip(OBJECTS, spec['boxes'])},
                        'stimulus_scales': sc, 'severity_sources': {x: x for x in labels[task]},
                        'source_images': paths, 'source_image_metadata': meta,
                        'source_answer_indices': {'a': val, 'b': val ^ (queried == changed), **{x: None for x in labels[task]}},
                        'degradations': {x: degraded[x][1] for x in labels[task]},
                        'reference_semantics': 'clean a task recovery; mixed a/b states have no unique gold answer; '
                                               'each severity source is one realization of a',
                        'source': provenance})
                b.close()
            for image in (a, *(x[0] for x in degraded.values())): image.close()
    if len({r['id'] for r in rows}) != len(rows): raise AssertionError('Duplicate query ids')
    (stage/'materials.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))  # key order keeps severities mild->severe
    dump(stage/'specs.json', specs)
    if previews: previews.mkdir(parents=True, exist_ok=True); contact_sheets(stage, rows, args.tasks, args.split, font, sheets)
    report = {'schema': 'followup-preparation-v15', 'split': args.split, 'material_set': f'v15-synthetic-{args.split}',
        'manifest': 'materials.jsonl', 'manifest_sha256': sha(stage/'materials.jsonl'), 'seed': args.seed, 'tasks': args.tasks,
        'counts': {'units': len(units), 'interventions': images['a'], 'query_rows': len(rows),
                   'query_rows_by_task': dict(Counter(r['task'] for r in rows)), 'images': {**images, 'total': sum(images.values())}},
        'severity_grid': {t: 'explicit' if t in args.values else args.severities for t in args.tasks},
        'labels': labels, 'default_start_labels': {t: [s['label'] for s in severity.grid(t, 'start')] for t in args.tasks},
        'families': {t: {k: severity.DEFAULT_GRIDS[t][k] for k in ('family', 'reference_kind')} for t in args.tasks},
        'canvas': list(SIZE), 'stimulus': STIMULUS,
        'realized_template_parameters': {f'{k}={v}': c for (k, v), c in sorted(realized.items(), key=str)},
        'disjoint_from': [{**x, 'overlapping_units': 0, 'overlapping_clean_images': 0} for x in excluded_info],
        'validated': ['a/b differences inside the changed object ROI', 'one canvas for all sources of an intervention',
                      'unique query ids', 'clean images unique across templates per task',
                      'unit ids and clean images disjoint from --disjoint-from manifests'],
        'generator': 'scripts/prepare_followup_v15.py', 'generator_sha256': provenance['generator_sha256'],
        'severity_module_sha256': provenance['severity_module_sha256'], 'severity_generator': severity.GENERATOR,
        'font': str(font), 'font_sha256': provenance['font_sha256'], 'pillow': PIL.__version__, 'python': sys.version.split()[0],
        'cli': {k: (str(v) if isinstance(v, Path) else [str(x) for x in v] if k == 'disjoint_from' else v) for k, v in vars(args).items()},
        'contact_sheets': [str(p) for p in sheets], 'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
        'model_runs': False, 'native_capability': 'intended; verify on calibration units in the cloud',
        'model_or_processor_validation': 'pending cloud; same pixel sizes do not alone prove processor geometry'}
    dump(stage/'preparation.json', report); stage.rename(root)
    return report


def contact_sheets(stage, rows, tasks, split, font, paths):
    """Actual input files: per task two templates x (a, b, each severity); one overview across tasks."""
    cap = truetype(font, 13); first = {}
    for r in rows: first.setdefault(r['intervention_id'], r)
    units = sorted({r['unit_id'] for r in rows}); pick = lambda t, u, e: first[f'{u}-{t}-edit{e}']
    def caption(r):
        return (f"{r['intervention_id']}  changed {r['changed_object']}: "
                f"{r['choices'][r['direction_a_index']]} -> {r['choices'][r['direction_b_index']]}")
    def draw(group, scale, path, title):
        w, h = round(SIZE[0]*scale), round(SIZE[1]*scale); cols = max(len(r['source_images']) for r in group)
        sheet = Image.new('RGB', (w*cols, 22+len(group)*(h+34)), 'white'); d = ImageDraw.Draw(sheet)
        d.text((4, 3), title, font=cap, fill='black')
        for i, r in enumerate(group):
            y = 22+i*(h+34); d.text((4, y+1), caption(r), font=cap, fill='black')
            for j, name in enumerate(r['source_images']):
                d.text((j*w+4, y+17), name, font=cap, fill=(70, 70, 70))
                with Image.open(stage/r['source_images'][name]) as im:
                    sheet.paste(im.convert('RGB').resize((w, h), Image.Resampling.BOX), (j*w, y+34))
        sheet.save(path)
    for task, path in zip(tasks, paths):
        draw([pick(task, units[0], 0), pick(task, units[min(1, len(units)-1)], 1)], .5, path,
             f'{split} / {task}: actual input files at 1/2 scale; degraded columns mild -> severe')
    draw([pick(t, units[0], 0) for t in tasks], .375, paths[-1], f'{split} overview: actual input files at 3/8 scale')


def reapply(rows):
    """Re-derive every degraded file from its clean a; returns mismatching (intervention, label)."""
    bad, done = [], set()
    for r in rows:
        if r['intervention_id'] in done: continue
        done.add(r['intervention_id'])
        with Image.open(r['source_images']['a']) as im: a = im.convert('RGB')
        for label, name in r['severity_sources'].items():
            spec = r['degradations'][label]; image, again = severity.apply(a, label, spec['reference_px'])
            with Image.open(r['source_images'][name]) as im: same = im.convert('RGB').tobytes() == image.tobytes()
            if not same or again != spec: bad.append((r['intervention_id'], label))
    return bad


def check(manifests):
    """Load with the v15 reader; check readout alignment, degraded identities and split disjointness."""
    from cloud_prep.measurement_v15 import design
    from cloud_prep.measurement_v6.content_readout import make_variant
    result, ids, clean = {'pillow': PIL.__version__, 'manifests': []}, {}, {}
    for manifest in manifests:
        rows = design.load_rows(manifest, design.settings())
        for r in rows:
            short = SPLITS[r['split']]['short']
            if not r['unit_id'].startswith(f'v15-{short}-') or r['material_set'] != f"v15-synthetic-{r['split']}":
                raise AssertionError(r['id']+': split, material set and unit id disagree')
            if r['source_answer_indices']['a'] != r['answer_index'] or r['reference_answers'] != [r['choices'][r['answer_index']]]:
                raise AssertionError(r['id']+': answer fields disagree')
            for fmt, mapping in READOUTS:
                v = make_variant(r, response_format=fmt, mapping=mapping)
                if (v['label_to_original'][v['answer_label']] != r['option_labels'][r['answer_index']]
                        or v['original_to_label'][r['answer_label']] != v['answer_label']
                        or v['display_choices'][v['labels'].index(v['answer_label'])] != r['choices'][r['answer_index']]):
                    raise AssertionError(f"{r['id']}: {fmt}/{mapping} does not map the answer back to answer_index")
        bad = reapply(rows); ids[manifest] = {r['unit_id'] for r in rows}
        clean[manifest] = {r['source_image_metadata']['a']['sha256'] for r in rows}
        result['manifests'].append(dict(manifest=str(manifest), sha256=sha(manifest), splits=sorted({r['split'] for r in rows}),
            rows=len(rows), units=len(ids[manifest]), interventions=len({r['intervention_id'] for r in rows}),
            sources=len({m['sha256'] for r in rows for m in r['source_image_metadata'].values()}),
            readouts_checked=[f'{f}/{m}' for f, m in READOUTS], reapply_mismatches=bad[:20], reapply_mismatch_count=len(bad)))
    for x, y in itertools.combinations(manifests, 2):
        if ids[x] & ids[y] or clean[x] & clean[y]: raise AssertionError(f'{x} and {y} share unit ids or clean images')
    result['pairwise_disjoint'] = len(manifests) > 1
    result['ok'] = not any(m['reapply_mismatch_count'] for m in result['manifests'])
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--output', type=Path); p.add_argument('--split', choices=SPLITS)
    p.add_argument('--templates', type=int, help='default 16 (calibration) / 32 (evaluation)')
    p.add_argument('--seed', type=int, help='default 2026102911 (calibration) / 2026102921 (evaluation)')
    p.add_argument('--tasks', nargs='+', choices=TASKS, default=list(TASKS))
    p.add_argument('--severities', choices=('candidates', 'start'), default='candidates', help='severity.DEFAULT_GRIDS list per task')
    p.add_argument('--values', nargs='+', default=[], metavar='TASK=V1,V2', help='explicit settings of a task (overrides --severities)')
    p.add_argument('--font'); p.add_argument('--previews', type=Path, help='directory for contact sheets (named by split)')
    p.add_argument('--disjoint-from', nargs='+', type=Path, default=[], metavar='MANIFEST',
                   help='other split manifests: abort if unit ids or clean images overlap')
    p.add_argument('--check', nargs='+', type=Path, metavar='MANIFEST', help='validate written manifests instead of generating')
    args = p.parse_args(argv)
    if args.check:
        result = check(args.check); print(json.dumps(result, indent=2)); return result
    if not args.output or not args.split: p.error('--output and --split are required')
    args.templates = SPLITS[args.split]['templates'] if args.templates is None else args.templates
    args.seed = SPLITS[args.split]['seed'] if args.seed is None else args.seed
    if args.templates < 1: p.error('--templates must be positive')
    values = {}
    for item in args.values:
        task, _, text = item.partition('=')
        if task not in args.tasks or task in values: p.error('--values takes TASK=V1,V2,... once per generated task: '+item)
        try: values[task] = [float(x) for x in text.split(',')]
        except ValueError: p.error('--values needs numbers: '+item)
    args.values = values
    report = prepare(args)
    print(json.dumps({k: report[k] for k in ('split', 'seed', 'manifest_sha256', 'counts', 'labels', 'contact_sheets')}, indent=2))
    return report


if __name__ == '__main__':
    out = main()
    sys.exit(0 if out.get('ok', True) else 1)
