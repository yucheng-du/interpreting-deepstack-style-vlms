#!/usr/bin/env python3
"""Add severity levels to a packaged v15 manifest (synthetic or benchmark) on the cloud, without re-rendering any stimulus.

New degraded files are derived from each intervention's packaged clean image ``a`` with ``severity.apply`` exactly as
``prepare_followup_v15.py:reapply()`` re-derives the packaged levels (run with Pillow 12.2.0, the version that drew the
packaged levels). Nothing in the package is modified: the extended manifest and the new files go to --output-dir; the
extended rows reference the packaged files by absolute path (path relocation does not change observation keys) and
the new files by absolute path in --output-dir/images.

Before writing, every intervention of an extended task is checked: the clean a, the counterfactual b and every
packaged level file must have the packaged sha256, and every packaged level must re-derive pixel-identically with an
identical spec (otherwise the run stops). Rows of other tasks are copied unchanged apart from absolute image paths.
Each extended row keeps its ``source`` provenance dict and every packaged field; only the new labels are added to
severity_sources / source_images / source_image_metadata / source_answer_indices (synthetic only) / degradations, mild to
severe. Derivation is once per clean image (benchmark questions of one image share files); the reference scale is the
row's packaged specs' reference_px (synthetic: stimulus scale; benchmarks: clean long side). New files are named like the
packaged ones: <clean stem without -a/-clean>-<label>.png.

  python scripts/extend_synthetic_levels_v15.py --manifest DATA/synthetic/calibration/materials.jsonl \
      --output-dir work/materials/synthetic-ext1/calibration --values text_reading=4,3,2 ...
"""
import argparse, copy, hashlib, json, os, platform, sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cloud_prep.measurement_v15 import severity  # noqa: E402


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--manifest', required=True, type=Path)
    p.add_argument('--output-dir', required=True, type=Path)
    p.add_argument('--values', nargs='+', required=True, metavar='TASK=V1,V2')
    p.add_argument('--note', default='')
    a = p.parse_args(argv)
    from PIL import Image, __version__ as pillow
    if pillow != '12.2.0': sys.exit(f'Pillow {pillow}: derive with Pillow 12.2.0 (the ibm env), the version of the packaged levels')
    manifest = a.manifest.resolve(); base = manifest.parent
    out = a.output_dir.resolve()
    if out.exists(): sys.exit(f'refusing to reuse {out}; choose a new directory')
    values = {}
    for item in a.values:
        task, _, vs = item.partition('=')
        if task not in severity.DEFAULT_GRIDS or not vs: sys.exit('bad --values item ' + item)
        values[task] = [float(v) for v in vs.split(',')]
    rows = [json.loads(x) for x in manifest.read_text().split('\n') if x.strip()]
    stage = out.with_name(out.name + '.partial'); (stage / 'images').mkdir(parents=True)
    new_labels = {}
    for task, vs in values.items():
        g = severity.DEFAULT_GRIDS[task]
        new_labels[task] = [severity.setting(g['family'], v, g['reference_kind'])['label'] for v in vs]
    derived, by_clean, checked_levels, checked_files = {}, {}, 0, 0
    for r in rows:
        if r['task'] not in values or r['intervention_id'] in derived: continue
        iid, task = r['intervention_id'], r['task']; kind = severity.DEFAULT_GRIDS[task]['reference_kind']
        clean_rel = r['source_images']['a']
        if clean_rel in by_clean:
            made_before, labels_before = by_clean[clean_rel]
            if labels_before != new_labels[task]: sys.exit(f'{iid}: clean image shared across tasks with different new labels')
            derived[iid] = made_before; continue
        clash = [L for L in new_labels[task] if L in r['severity_sources'] or L in r['source_images']]
        if clash: sys.exit(f'{iid}: labels already present {clash}')
        for name, rel in r['source_images'].items():
            path = Path(rel) if Path(rel).is_absolute() else base / rel
            if sha(path) != r['source_image_metadata'][name]['sha256']: sys.exit(f'{iid}/{name}: packaged file sha256 differs')
            checked_files += 1
        with Image.open(base / r['source_images']['a']) as im: clean = im.convert('RGB')
        for label, name in r['severity_sources'].items():
            spec = r['degradations'][label]
            image, again = severity.apply(clean, label, spec['reference_px'])
            with Image.open(base / r['source_images'][name]) as im: same = im.convert('RGB').tobytes() == image.tobytes()
            if not same or again != spec: sys.exit(f'{iid}/{label}: packaged level does not re-derive identically (pixels {same}, spec {again == spec})')
            checked_levels += 1
        refs = {spec['reference_px'] for spec in r['degradations'].values()}
        if len(refs) != 1: sys.exit(f'{iid}: packaged specs disagree on reference_px {refs}')
        ref = refs.pop()
        if kind is None: ref = None
        elif 'stimulus_scales' in r and float(r['stimulus_scales'][kind]) != float(ref): sys.exit(f'{iid}: reference_px differs from stimulus scale')
        stem = Path(clean_rel).stem
        for suffix in ('-a', '-clean'):
            if stem.endswith(suffix): stem = stem[:-len(suffix)]; break
        else: sys.exit(f'{iid}: unexpected clean file name {clean_rel}')
        made = {}
        for label in new_labels[task]:
            image, spec = severity.apply(clean, label, ref)
            path = stage / 'images' / f'{stem}-{label}.png'
            if path.exists(): sys.exit(f'refusing to overwrite {path}')
            image.save(path)
            made[label] = dict(spec=spec, file=out / 'images' / path.name,
                               meta={'width': image.width, 'height': image.height, 'sha256': sha(path), 'bytes': path.stat().st_size})
        derived[iid] = made; by_clean[clean_rel] = (made, new_labels[task])
    extended = []
    for r in rows:
        x = copy.deepcopy(r)
        x['source_images'] = {k: (v if Path(v).is_absolute() else str((base / v).resolve())) for k, v in r['source_images'].items()}
        if r['task'] in values:
            made = derived[r['intervention_id']]
            order = severity.ordered(list(r['severity_sources']) + list(made))
            x['severity_sources'] = {L: r['severity_sources'].get(L, L) for L in order}
            images = {k: v for k, v in x['source_images'].items() if k not in r['severity_sources']}
            meta = {k: v for k, v in r['source_image_metadata'].items() if k not in r['severity_sources']}
            answers = {k: v for k, v in r.get('source_answer_indices', {}).items() if k not in r['severity_sources']}
            for L in order:
                if L in made:
                    images[L] = str(made[L]['file']); meta[L] = made[L]['meta']; answers[L] = None
                else:
                    images[L] = x['source_images'][L]; meta[L] = r['source_image_metadata'][L]
                    if 'source_answer_indices' in r: answers[L] = r['source_answer_indices'][L]
            x['source_images'], x['source_image_metadata'] = images, meta
            if 'source_answer_indices' in r: x['source_answer_indices'] = answers
            x['degradations'] = {L: (made[L]['spec'] if L in made else r['degradations'][L]) for L in order}
        # every packaged field and value is kept (only the image paths are made absolute)
        for k, v in r.items():
            if k in ('source_images',): continue
            if isinstance(v, dict) and k in ('severity_sources', 'source_image_metadata', 'source_answer_indices', 'degradations'):
                if any(x[k].get(kk) != vv for kk, vv in v.items()): sys.exit(f'{r["id"]}: packaged {k} changed')
            elif x[k] != v: sys.exit(f'{r["id"]}: packaged field {k} changed')
        extended.append(x)
    text = ''.join(json.dumps(x) + '\n' for x in extended)
    (stage / 'materials.jsonl').write_text(text)
    prov = dict(schema='v15-synthetic-level-extension-1', created_at=datetime.now(timezone.utc).isoformat(timespec='seconds'),
                packaged_manifest=str(manifest), packaged_manifest_sha256=sha(manifest), output_manifest=str(out / 'materials.jsonl'),
                output_manifest_sha256=hashlib.sha256(text.encode()).hexdigest(), values=values, labels_added=new_labels,
                interventions_extended=len(derived), clean_images_extended=len(by_clean),
                new_files=sum(len(m) for m, _ in by_clean.values()),
                packaged_files_sha_checked=checked_files, packaged_levels_rederived_identically=checked_levels,
                rows=len(extended), pillow=pillow, python=platform.python_version(), severity_module_sha256=sha(severity.__file__),
                severity_generator=severity.GENERATOR, script_sha256=sha(__file__),
                method='severity.apply on the packaged clean a (as prepare_followup_v15.py:reapply); no stimulus re-rendering; '
                       'packaged rows, source dicts and existing level files unchanged; packaged paths made absolute',
                note=a.note)
    (stage / 'extension.json').write_text(json.dumps(prov, indent=1) + '\n')
    os.replace(stage, out)
    print(json.dumps({k: prov[k] for k in ('output_manifest', 'output_manifest_sha256', 'labels_added', 'interventions_extended', 'clean_images_extended',
                                            'new_files', 'packaged_files_sha_checked', 'packaged_levels_rederived_identically', 'rows')}))


if __name__ == '__main__':
    main()
