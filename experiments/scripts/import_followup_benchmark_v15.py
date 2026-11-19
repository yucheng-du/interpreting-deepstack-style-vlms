#!/usr/bin/env python3
"""Import GQA / Single-Document DocVQA validation units as v15 severity-curve materials.

No downloads or model-dependent filtering. Every question on an image shares one
clean image 'a' and one realization per severity label, written on the clean
canvas by cloud_prep/measurement_v15/severity.py with reference_px = the clean
long side. Rows follow the v15 design contract (split, material_set,
severity_sources ordered mild to severe, source_images {'a', label...},
source_image_metadata, degradations); no 'degraded_a' and no counterfactual 'b'.

Modes
  official import  --benchmark B --annotations F [--annotations-member M] --images-root D --split S --output DIR
  selection only   the same with --selection-only --output PLAN.json (ids, counts, sha256s; no images, text or answers)
  recorded plan    --unit-ids-file PLAN.json imports exactly that plan after checking its input sha256
  v14 reuse        --from-v14-manifest MANIFEST: named calibration material reused from an executed v14 import
  exclusion list   --exclusions-only merges v14 manifests and id files into one exclusion JSON
  contact sheet    --contact-sheet MANIFEST --output SHEET.jpg

A unit falls in the calibration pool when its sha256 draw keyed by (seed, benchmark,
unit) is below --calibration-fraction and in the evaluation pool otherwise, whatever
the exclusions, caps or run order. Within a pool, units are taken in sha256 rank
order, so a larger --max-units extends a selection without changing it. The
DocVQA unit is the source document: excluding any page excludes the document.
Use a new output directory for materially changed selections or image operations.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
import zipfile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cloud_prep.measurement_v15 import severity  # noqa: E402

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
# The executed v14 instructions (-lc GQA and DocVQA manifests).
ANSWER_INSTRUCTIONS = {'gqa': 'Answer with a single lowercase word or short phrase, without punctuation.',
                       'docvqa': 'Answer using only a short word or phrase.'}
REUSE_SCOPE = 'v14 benchmark material reused for calibration; not independent replication'
UNIT_PREFIX = {'gqa': 'gqa-image-', 'docvqa': 'docvqa-document-'}
SEED = 20261029
PLAN_SCHEMA = 'followup-benchmark-selection-plan-v15'
SEMANTICS = 'Same underlying clean image; every clean/degraded cell uses its clean-reference answers'
RULE = ("pool: sha256('seed|split|benchmark|unit_id') first 15 hex / 16**15 < calibration_fraction -> calibration, "
        "else evaluation; units: ascending sha256('seed|unit|task|unit_id'); questions: ascending sha256('seed|question|task|question_id')")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''): h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def safe_output(path):
    path = Path(path).resolve()
    return path


def jsonl(path):
    # Physical lines: str.splitlines also splits at U+2028/U+2029/U+0085 inside JSON strings.
    return [json.loads(line) for line in Path(path).read_text().split('\n') if line.strip()]


def draw(seed, *parts):
    return hashlib.sha256('|'.join(str(x) for x in (seed,) + parts).encode()).hexdigest()


def split_pool(seed, benchmark, unit_id, calibration_fraction):
    return 'calibration' if int(draw(seed, 'split', benchmark, unit_id)[:15], 16)/16**15 < calibration_fraction else 'evaluation'


def load_annotations(path, member=None):
    """Official JSON, read in place (also from a zip member). sha256 is that of the JSON bytes."""
    path = Path(path)
    if zipfile.is_zipfile(path):
        if not member: raise ValueError('Name the JSON member of an annotation zip with --annotations-member')
        with zipfile.ZipFile(path) as z: raw = z.read(member)
        container = sha(path)
    else:
        if member: raise ValueError('--annotations-member applies to zip files only')
        raw, container = path.read_bytes(), None
    info = dict(kind='official_annotations', file_name=path.name, member=member,
                sha256=hashlib.sha256(raw).hexdigest(), container_sha256=container)
    return json.loads(raw), info


def image_key(value):
    return re.sub(r'^vg-', '', Path(str(value)).stem)


def read_ids(path, key):
    """Ids from a JSON array, a JSON object's unit_ids/image_ids list, a v15 selection plan (unit ids),
    a manifest (.jsonl) or text lines."""
    path = Path(path); text = path.read_text()
    if path.suffix == '.jsonl':
        rows = [json.loads(x) for x in text.split('\n') if x.strip()]
        values = [r['unit_id'] if key == 'unit_ids' else r['benchmark']['image_id'] for r in rows]
    else:
        try: values = json.loads(text)
        except json.JSONDecodeError: values = [s.strip() for s in text.splitlines() if s.strip() and not s.startswith('#')]
        if isinstance(values, dict) and values.get('schema') == PLAN_SCHEMA:
            if key != 'unit_ids': raise ValueError(f'{path}: use a selection plan with --exclude-unit-ids')
            values = [u['unit_id'] for u in values['units']]
        elif isinstance(values, dict):
            if key not in values: raise ValueError(f'{path} has no {key} list')
            values = values[key]
    if not isinstance(values, list) or not all(isinstance(v, (str, int)) for v in values):
        raise ValueError(f'{path}: expected a list of ids')
    if not values: raise ValueError(f'{path} lists no {key}')
    if key == 'image_ids': return {image_key(v) for v in values}
    values = {str(v) for v in values}
    bad = sorted(v for v in values if not v.startswith(tuple(UNIT_PREFIX.values())))
    if bad: raise ValueError('Unit ids look like gqa-image-<id> or docvqa-document-<id>, not '+repr(bad[0]))
    return values


def exclusion_files(args):
    """Merged unit/image ids plus one record and id set per file."""
    found = dict(unit_ids=set(), image_ids=set()); files = []; sets = []
    for key in found:
        for path in getattr(args, 'exclude_'+key) or []:
            ids = read_ids(path, key); found[key] |= ids; sets.append(ids)
            files.append(dict(flag='--exclude-'+key.replace('_', '-'), file_name=Path(path).name, sha256=sha(path), ids=len(ids)))
    return found['unit_ids'], found['image_ids'], files, sets


def apply_exclusions(records, units, images, files=(), sets=()):
    """Drop whole units: listed unit ids, and every unit containing an excluded image (a DocVQA page drops its document)."""
    by_image = defaultdict(set)
    for r in records: by_image[r['image_id']].add(r['unit_id'])
    def via(ids): return set().union(*(by_image[i] for i in ids if i in by_image))
    present, via_image = {r['unit_id'] for r in records}, via(images)
    for f, ids in zip(files, sets):  # each file's own effect on the eligible units
        f['units_matched'] = len(present & ids) if f['flag'] == '--exclude-unit-ids' else len(via(ids))
    kept, dropped = [], Counter()
    for r in records:
        if r['unit_id'] in units: dropped['excluded_unit'] += 1
        elif r['unit_id'] in via_image: dropped['excluded_image'] += 1
        else: kept.append(r)
    return kept, dropped


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


def official_records(data, args, exact_filters):
    """Eligible questions before exclusions (v14 checks; GQA group filter; DocVQA document units)."""
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
            group = gqa_group(row, args.spatial_relations)
            if group != args.gqa_group:
                dropped['group_filter'] += 1; continue
            if any(get_field(row, key) not in allowed for key, allowed in exact_filters.items()):
                dropped['exact_metadata_filter'] += 1; continue
            records.append(dict(qid=str(qid), image_id=image_id, unit_id=UNIT_PREFIX['gqa']+image_id,
                image_file=image_id + args.gqa_image_extension, query=row['question'], references=[row['answer']], group=group,
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
            document = row.get('ucsf_document_id', row.get('docId'))
            if document is None:
                raise ValueError('DocVQA needs docId or ucsf_document_id for document-level inference')
            records.append(dict(qid=str(row['questionId']), image_id=Path(image_name).stem,
                unit_id=UNIT_PREFIX['docvqa']+str(document), image_file=image_name,
                query=row['question'], references=row['answers'], group='document_qa',
                official_metadata={key:row[key] for key in ('docId','ucsf_document_id','ucsf_document_page_no','question_types','data_split') if key in row}))
    for row in records:
        if not isinstance(row['query'], str) or not row['query'].strip(): raise ValueError('Empty question')
        if not row['references'] or not all(isinstance(x,str) and x for x in row['references']):
            raise ValueError('Nonempty reference answer strings required')
    if len({r['qid'] for r in records}) != len(records): raise ValueError('Duplicate question id')
    return records, dropped


def v14_records(manifest):
    """Questions of an executed v14 import; the clean 'a' image is the only source reused."""
    rows = jsonl(manifest); records = []
    for row in rows:
        if row.get('query_role', 'task') != 'task' or 'a' not in row['source_images']: raise ValueError('Not a v14 benchmark row: '+row['id'])
        records.append(dict(qid=row['benchmark']['question_id'], image_id=str(row['benchmark']['image_id']), unit_id=row['unit_id'],
            image_file=row['source_images']['a'], group=row['benchmark']['group'], v14_row=row))
    if len({r['v14_row']['id'] for r in records}) != len(records): raise ValueError('Duplicate v14 query id')
    tasks = {r['v14_row']['task'] for r in records}
    if len(tasks) != 1: raise ValueError('One task per v14 manifest expected, found '+str(sorted(tasks)))
    meta_path = Path(manifest).parent / 'import_metadata.json'; meta = None
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta.get('manifest_sha256') != sha(manifest): raise ValueError('v14 import_metadata.json does not describe this manifest')
    info = dict(kind='v14_manifest', file_name=Path(manifest).name, path=str(Path(manifest).resolve()), sha256=sha(manifest),
        rows=len(rows), task=tasks.pop(), v14_import_metadata=None if meta is None else dict(
            schema=meta.get('schema'), manifest_sha256=meta.get('manifest_sha256'), annotations_sha256=meta.get('annotations_sha256'),
            selection_seed=meta.get('selection_seed'), selected_units=meta.get('selected_units'), degradation=meta.get('degradation')))
    return records, info


def select(records, *, seed, task, max_units, max_questions, benchmark, split=None, calibration_fraction=None):
    grouped = defaultdict(list)
    for r in records: grouped[r['unit_id']].append(r)
    pool = [u for u in grouped if split is None or split_pool(seed, benchmark, u, calibration_fraction) == split]
    order = sorted(pool, key=lambda u: draw(seed, 'unit', task, u))
    chosen = order[:max_units] if max_units else order
    selected = []
    for unit in chosen:
        qq = sorted(grouped[unit], key=lambda r: draw(seed, 'question', task, r['qid']))
        selected.extend(qq[:max_questions] if max_questions else qq)
    selected.sort(key=lambda r: (r['unit_id'], r['qid']))
    counts = dict(eligible_questions=len(records), eligible_units=len(grouped), pool_units=len(pool),
        other_pool_units=len(grouped)-len(pool), selected_units=len(chosen), remaining_pool_units=len(pool)-len(chosen),
        selected_questions=len(selected), selected_images=len({r['image_file'] for r in selected}))
    return selected, counts


def from_plan(records, plan, input_info, task):
    if plan.get('schema') != PLAN_SCHEMA: raise ValueError('Not a v15 selection plan')
    if plan['input']['sha256'] != input_info['sha256']:
        raise ValueError('Plan input sha256 differs from the supplied annotations/manifest')
    if plan['task'] != task: raise ValueError(f"Plan is for task {plan['task']}, not {task}")
    by_id = {r['qid']: r for r in records}; selected = []
    for unit in plan['units']:
        for qid in unit['question_ids']:
            r = by_id.get(qid)
            if r is None or r['unit_id'] != unit['unit_id']:
                raise ValueError('Planned question is not eligible here (excluded, filtered or another unit): '+qid)
            selected.append(r)
    if not selected: raise ValueError('Empty plan')
    selected.sort(key=lambda r: (r['unit_id'], r['qid']))
    counts = dict(plan['counts']); counts.pop('dropped', None)
    counts.update(selected_units=len({r['unit_id'] for r in selected}), selected_questions=len(selected),
                  selected_images=len({r['image_file'] for r in selected}))
    return selected, counts


def plan_value(args, mode, task, input_info, selected, counts, dropped, files, header):
    units = defaultdict(lambda: dict(question_ids=[], image_files=set()))
    for r in selected:
        units[r['unit_id']]['question_ids'].append(r['qid']); units[r['unit_id']]['image_files'].add(r['image_file'])
    return dict(schema=PLAN_SCHEMA, mode=mode, benchmark=args.benchmark, task=task, split=args.split, material_set=args.material_set,
        seed=args.seed, calibration_fraction=args.calibration_fraction, max_units=args.max_units,
        max_questions_per_unit=args.max_questions_per_unit, rule=RULE, input=input_info, input_dataset_header=header,
        exclusions=files, counts={**counts, 'dropped': dict(dropped)},
        units=[dict(unit_id=u, question_ids=v['question_ids'], image_files=sorted(v['image_files'])) for u, v in sorted(units.items())],
        importer_sha256=sha(__file__), content='ids, counts and sha256s only: no question text, answers or URLs')


def settings_for(task, args):
    which = 'values' if args.values else args.severities
    chosen = severity.grid(task, 'candidates' if which == 'candidates' else 'start', args.values)
    return chosen, which


def materialize(selected, out, settings, clean_writer):
    """One clean file and one realization per setting for every distinct source image, shared by its questions."""
    from PIL import Image
    (out/'images').mkdir(parents=True, exist_ok=True); cache = {}; noop = Counter()
    for r in selected:
        if r['image_file'] in cache: continue
        stem, clean_path, provenance = clean_writer(r)
        with Image.open(clean_path) as opened: clean = opened.convert('RGB')
        width, height = clean.size
        paths = {'a': clean_path.relative_to(out).as_posix()}
        meta = {'a': dict(width=width, height=height, sha256=sha(clean_path), role='clean')}; specs = {}
        for s in settings:
            image, spec = severity.apply(clean, s, reference_px=max(width, height))
            path = out/'images'/f"{stem}-{s['label']}.png"; image.save(path)
            paths[s['label']] = path.relative_to(out).as_posix()
            meta[s['label']] = dict(width=width, height=height, sha256=sha(path), role='degraded',
                                    severity_label=s['label'], spec_id=spec['spec_id'])
            specs[s['label']] = spec; noop[s['label']] += int(bool(spec['realized'].get('noop')))
        cache[r['image_file']] = dict(paths=paths, metadata=meta, specs=specs, provenance=provenance)
    return cache, noop


def v15_fields(material, split, material_set, settings):
    labels = [s['label'] for s in settings]
    return dict(split=split, material_set=material_set, severity_sources={x: x for x in labels},
        source_images=material['paths'], source_image_metadata=material['metadata'], degradations=material['specs'])


def import_or_plan(args):
    reuse = bool(args.from_v14_manifest)
    plan = json.loads(Path(args.unit_ids_file).read_text()) if args.unit_ids_file else None
    if plan:
        for key in ('split', 'material_set', 'seed', 'calibration_fraction', 'max_units', 'max_questions_per_unit'):
            given = getattr(args, key)
            if given is not None and given != plan[key]: raise ValueError(f'--{key.replace("_","-")} differs from the plan')
            setattr(args, key, plan[key])
        if not reuse: args.benchmark = args.benchmark or plan['benchmark']
    exact_filters = {}
    if reuse:
        if len(args.from_v14_manifest) != 1: raise ValueError('Import one v14 manifest at a time')
        if args.annotations: raise ValueError('--from-v14-manifest replaces --annotations')
        records, input_info = v14_records(args.from_v14_manifest[0]); dropped = Counter()
        task = input_info['task']; args.benchmark = records[0]['v14_row']['benchmark']['name']; header = {}
        if args.split not in (None, 'calibration'): raise ValueError('Reused v14 material is calibration material only')
        args.split = 'calibration'
    else:
        if not (args.benchmark and args.annotations): raise ValueError('Official import needs --benchmark and --annotations')
        data, input_info = load_annotations(args.annotations, args.annotations_member)
        if args.benchmark == 'gqa':
            args.gqa_group = args.gqa_group or (plan['task'].split('_', 1)[1] if plan else None)
            if args.gqa_group not in ('attribute', 'spatial'): raise ValueError('GQA needs --gqa-group attribute or spatial')
            exact_filters = json.loads(Path(args.gqa_filter_json).read_text()) if args.gqa_filter_json else {}
            if not isinstance(exact_filters, dict) or any(not isinstance(v,list) for v in exact_filters.values()):
                raise ValueError('GQA filters must map dotted metadata paths to allowed-value lists')
            task = 'gqa_' + args.gqa_group
        else:
            task = 'docvqa_document_qa'
        records, dropped = official_records(data, args, exact_filters)
        header = {k: data[k] for k in ('dataset_name','dataset_split','dataset_version') if k in data}
        if args.split is None: raise ValueError('Official import needs --split calibration|evaluation')
    args.seed = SEED if args.seed is None else args.seed
    if args.calibration_fraction is None: args.calibration_fraction = None if reuse else .25
    if not reuse and not 0 <= args.calibration_fraction <= 1: raise ValueError('--calibration-fraction must be in [0,1]')
    args.max_questions_per_unit = 2 if args.max_questions_per_unit is None else args.max_questions_per_unit
    args.material_set = args.material_set or f"{args.split}-{'v14reuse' if reuse else 'fresh'}-{task}"
    units, images, files, sets = exclusion_files(args)
    records, excluded = apply_exclusions(records, units, images, files, sets); dropped.update(excluded)
    if plan: selected, counts = from_plan(records, plan, input_info, task)
    else:
        selected, counts = select(records, seed=args.seed, task=task, max_units=args.max_units, max_questions=args.max_questions_per_unit,
            benchmark=args.benchmark, split=None if reuse else args.split, calibration_fraction=args.calibration_fraction)
    if not selected: raise ValueError('Selection is empty')
    mode = 'v14_reuse' if reuse else 'official'
    if args.selection_only:
        out = safe_output(args.output)
        if out.exists() and not args.overwrite: raise ValueError('Plan exists; use a new path or --overwrite')
        out.parent.mkdir(parents=True, exist_ok=True)
        value = plan_value(args, mode, task, input_info, selected, counts, dropped, files, header)
        write_json(out, value); return value
    out = safe_output(args.output); manifest_path = out/'manifest.jsonl'
    if manifest_path.exists() and not args.overwrite: raise ValueError('Manifest already exists; use a new directory or explicit --overwrite')
    settings, which = settings_for(task, args)
    out.mkdir(parents=True, exist_ok=True)
    if reuse:
        root = Path(args.from_v14_manifest[0]).resolve().parent
        def clean_writer(r):
            row = r['v14_row']; source = root/row['source_images']['a']; target = out/'images'/Path(row['source_images']['a']).name
            expected = row['source_image_metadata']['a']['sha256']
            if sha(source) != expected: raise ValueError('v14 clean image differs from its manifest: '+str(source))
            shutil.copyfile(source, target)
            return row['source']['original_image_sha256'][:20], target, dict(v14_clean_sha256=expected)
    else:
        if not args.images_root: raise ValueError('Official import needs --images-root')
        def clean_writer(r):
            from PIL import Image
            source = (Path(args.images_root)/r['image_file']).resolve(); original = sha(source)
            target = out/'images'/f'{original[:20]}-clean.png'
            with Image.open(source) as opened: opened.convert('RGB').save(target)
            return original[:20], target, dict(original_image_sha256=original, original_image_path=str(source))
    version = selected[0]['v14_row']['benchmark']['version'] if reuse else (args.dataset_version or header.get('dataset_version'))
    if not version: raise ValueError('Declare --dataset-version (DocVQA may provide it in JSON)')
    cache, noop = materialize(selected, out, settings, clean_writer)
    manifest = []
    for r in selected:
        material = cache[r['image_file']]; fields = v15_fields(material, args.split, args.material_set, settings)
        if reuse:
            old = r['v14_row']; src = old['source']
            manifest.append(dict(id=old['id'], unit_id=old['unit_id'], intervention_id=old['intervention_id'], task=old['task'],
                query_role='task', query=old['query'], answer_instruction=old['answer_instruction'], **fields,
                reference_answers=old['reference_answers'], families=['recovery'], benchmark=old['benchmark'],
                source=dict(official=src['official'], annotations_sha256=src['annotations_sha256'],
                    original_image_sha256=src['original_image_sha256'], original_image_path=src['original_image_path'],
                    selection_seed=args.seed, v14_selection_seed=src.get('selection_seed'),
                    reuse=dict(scope=REUSE_SCOPE, v14_manifest_sha256=input_info['sha256'], v14_query_id=old['id'],
                               **material['provenance']), semantics=SEMANTICS)))
        else:
            qid = args.benchmark+'-'+r['qid']
            manifest.append(dict(id=qid, unit_id=r['unit_id'], intervention_id=qid, task=task, query_role='task', query=r['query'],
                answer_instruction=args.answer_instruction or ANSWER_INSTRUCTIONS[args.benchmark], **fields,
                reference_answers=r['references'], families=['recovery'],
                benchmark=dict(name=args.benchmark, version=str(version), split='validation', question_id=r['qid'],
                    image_id=r['image_id'], cluster_id=r['unit_id'], group=r['group'], official_metadata=r['official_metadata']),
                source=dict(official=OFFICIAL[args.benchmark], annotations_sha256=input_info['sha256'], **material['provenance'],
                    selection_seed=args.seed, calibration_fraction=args.calibration_fraction, semantics=SEMANTICS)))
    with manifest_path.open('w') as f:
        for row in manifest: f.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
    from PIL import __version__ as pillow_version
    grid = severity.DEFAULT_GRIDS[task]
    metadata = dict(schema='followup-benchmark-import-v15', mode=mode, benchmark=args.benchmark, task=task,
        version=str(version), split='validation', v15_split=args.split, material_set=args.material_set,
        official=OFFICIAL[args.benchmark], input=input_info, input_dataset_header=header, manifest_sha256=sha(manifest_path),
        counts={**counts, 'manifest_rows': len(manifest), 'images_written': len(cache)*(1+len(settings))}, dropped=dict(dropped),
        selection=dict(seed=args.seed, calibration_fraction=args.calibration_fraction, max_units=args.max_units,
            max_questions_per_unit=args.max_questions_per_unit, rule=RULE, unit='image for GQA; source document for DocVQA',
            plan=None if not plan else dict(file_name=Path(args.unit_ids_file).name, sha256=sha(args.unit_ids_file)),
            gqa_group=args.gqa_group if args.benchmark == 'gqa' and not reuse else None,
            gqa_spatial_relations=args.spatial_relations if args.benchmark == 'gqa' and not reuse else None,
            gqa_exact_metadata_filters=exact_filters, gqa_missing_balanced_flag_assumption=args.gqa_assume_balanced),
        exclusions=files,
        severity=dict(which=which, family=grid['family'], reference_kind=grid['reference_kind'],
            reference_px='clean image long side, max(width, height)', labels=[s['label'] for s in settings],
            values=[s['value'] for s in settings], default_grid=grid, generator=severity.GENERATOR, pillow=pillow_version,
            noop_images={s['label']: noop[s['label']] for s in settings}),
        reuse=None if not reuse else dict(scope=REUSE_SCOPE, receipts=[json.loads(Path(x).read_text()) for x in args.v14_pack_receipt or []]),
        importer_sha256=sha(__file__),
        readout='Freeform generation only; no artificial distractor or binary recoding',
        evaluation_note='GQA official exact equality is primary; lower+strip and case+punctuation are separately named sensitivities. DocVQA organizer-framework ANLS and strict-paper cutoff are distinguished by scorer.',
        limitation='Imported validation subset, not a full benchmark or a model result. Pixel sizes are preserved; processor token-grid identity is checked at runtime.'
            + (' ' + REUSE_SCOPE + '.' if reuse else ''))
    write_json(out/'import_metadata.json', metadata)
    return metadata


def write_exclusions(args):
    units, images, files, _ = exclusion_files(args); sources = [dict(kind='id_file', **f) for f in files]
    for path in args.from_v14_manifest or []:
        records, info = v14_records(path)
        units |= {r['unit_id'] for r in records}; images |= {image_key(r['image_id']) for r in records}
        sources.append(dict(info, units=len({r['unit_id'] for r in records}), images=len({r['image_id'] for r in records})))
    out = safe_output(args.output)
    if out.exists() and not args.overwrite: raise ValueError('Exclusion file exists; use a new path or --overwrite')
    out.parent.mkdir(parents=True, exist_ok=True)
    value = dict(schema='followup-benchmark-exclusions-v15', description=args.description, unit_ids=sorted(units), image_ids=sorted(images),
        counts=dict(unit_ids=len(units), image_ids=len(images)), sources=sources,
        receipts=[json.loads(Path(x).read_text()) for x in args.v14_pack_receipt or []],
        usage='--exclude-unit-ids reads unit_ids; --exclude-image-ids reads image_ids. An excluded image drops its whole unit (a DocVQA page drops its document).')
    write_json(out, value); return value


def contact_sheet(args):
    """Clean plus every severity of a few calibration images; DocVQA adds a 1:1 crop of its densest text window."""
    from PIL import Image, ImageDraw, ImageFilter, ImageStat
    out = safe_output(args.output); manifest = Path(args.contact_sheet).resolve(); rows = jsonl(manifest)
    firsts = {}
    for r in rows: firsts.setdefault(r['source_images']['a'], r)
    seed = SEED if args.seed is None else args.seed
    picks = sorted(firsts.values(), key=lambda r: draw(seed, 'preview', r['source_images']['a']))[:args.preview_images]
    names = ['a'] + list(picks[0]['severity_sources']); zoom = picks[0]['benchmark']['name'] == 'docvqa'
    tile, crop_h, head = args.preview_px, int(args.preview_px*.75), 18
    def rgb(path):
        with Image.open(path) as im: return im.convert('RGB')
    blocks = []
    for r in picks:
        opened = {n: rgb(manifest.parent/r['source_images'][n]) for n in names}
        w, h = opened['a'].size; scale = min(tile/w, tile/h)
        thumbs = [opened[n].resize((max(1, round(w*scale)), max(1, round(h*scale))), Image.Resampling.LANCZOS) for n in names]
        crops = None
        if zoom:  # 1:1 pixels of the clean window with the densest edges (text), away from scan borders
            cw, ch = min(tile, w), min(crop_h, h); best = None
            edges = opened['a'].convert('L').filter(ImageFilter.FIND_EDGES)
            x0, y0, x1, y1 = int(.05*w), int(.05*h), max(int(.05*w), int(.95*w)-cw), max(int(.05*h), int(.95*h)-ch)
            for fy in range(10):
                for fx in range(8):
                    box = (x0+(x1-x0)*fx//7, y0+(y1-y0)*fy//9); box += (box[0]+cw, box[1]+ch)
                    density = ImageStat.Stat(edges.crop(box)).mean[0]
                    if best is None or density > best[0]: best = (density, box)
            crops = [opened[n].crop(best[1]) for n in names]
        blocks.append((r, thumbs, crops))
    width = len(names)*(tile+6)+6
    height = head+sum(tile+head+6+((crop_h+6) if zoom else 0) for _ in blocks)+6
    sheet = Image.new('RGB', (width, height), 'white'); pen = ImageDraw.Draw(sheet)
    for i, n in enumerate(names): pen.text((6+i*(tile+6), 3), 'clean' if n == 'a' else n, fill='black')
    y = head
    for r, thumbs, crops in blocks:
        meta = r['source_image_metadata']['a']
        pen.text((6, y), f"{r['unit_id']}  canvas {meta['width']}x{meta['height']}" + ('; second row: 1:1 crop' if zoom else ''), fill='black')
        y += head
        for i, t in enumerate(thumbs): sheet.paste(t, (6+i*(tile+6), y))
        y += tile+6
        if crops:
            for i, c in enumerate(crops): sheet.paste(c, (6+i*(tile+6), y))
            y += crop_h+6
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, quality=88) if out.suffix.lower() in ('.jpg', '.jpeg') else sheet.save(out)
    return dict(output=str(out), images=len(picks), columns=names, zoom_crop=zoom, size=list(sheet.size))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--benchmark', choices=['gqa','docvqa'])
    p.add_argument('--annotations', type=Path); p.add_argument('--annotations-member')
    p.add_argument('--images-root', type=Path); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--dataset-version')
    p.add_argument('--split', choices=['calibration','evaluation'])
    p.add_argument('--material-set'); p.add_argument('--seed', type=int, help=f'default {SEED}')
    p.add_argument('--calibration-fraction', type=float, help='share of units in the calibration pool (default .25)')
    p.add_argument('--max-units', type=int); p.add_argument('--max-questions-per-unit', type=int, help='default 2; 0 keeps all')
    p.add_argument('--exclude-unit-ids', type=Path, action='append'); p.add_argument('--exclude-image-ids', type=Path, action='append')
    p.add_argument('--severities', choices=['candidates','start'], default='start'); p.add_argument('--values', type=float, nargs='+')
    p.add_argument('--selection-only', action='store_true'); p.add_argument('--unit-ids-file', type=Path)
    p.add_argument('--from-v14-manifest', type=Path, action='append'); p.add_argument('--v14-pack-receipt', type=Path, action='append')
    p.add_argument('--exclusions-only', action='store_true'); p.add_argument('--description', default='')
    p.add_argument('--contact-sheet', type=Path); p.add_argument('--preview-images', type=int, default=3)
    p.add_argument('--preview-px', type=int, default=256)
    p.add_argument('--gqa-group', choices=['attribute','spatial'])
    p.add_argument('--gqa-filter-json', type=Path); p.add_argument('--gqa-assume-balanced', action='store_true')
    p.add_argument('--gqa-image-extension', default='.jpg')
    p.add_argument('--spatial-relations', nargs='+', default=SPATIAL_RELATIONS)
    p.add_argument('--answer-instruction'); p.add_argument('--overwrite', action='store_true')
    args = p.parse_args(argv)
    if args.max_units is not None and args.max_units < 1: p.error('--max-units must be positive')
    if args.max_questions_per_unit is not None and args.max_questions_per_unit < 0: p.error('Question cap must be >=0 (0 keeps all)')
    if args.contact_sheet: result = contact_sheet(args)
    elif args.exclusions_only: result = write_exclusions(args); result = dict(output=str(args.output), **result['counts'])
    else:
        result = import_or_plan(args)
        result = {k: result[k] for k in ('schema','mode','task','v15_split','split','material_set','counts','manifest_sha256') if k in result}
    print(json.dumps(result))
    return result

if __name__ == '__main__': main()
