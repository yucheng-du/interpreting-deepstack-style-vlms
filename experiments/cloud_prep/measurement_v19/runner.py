"""v19 stream-coding runner (model environment).

Modes
  extract       vision encoder only: save per-unit stream tensors (float32 .npy) + metadata
  means         level-M vectors from the manifest's images (not resumable, no shards)
  stats         vision encoder only: per-unit stream n/sum/sumsq (stats_units.jsonl, resumable), then the
                offsets JSON (offset_axis.offsets_table) from all stats files in --out; no features saved
  check-native  validity (i): unmodified model vs (C,C) through the new path, generated text + prefill logits
  cc-eval       validity (ii): (C,C) on evaluation-fresh questions, for comparison with stored cloud CC
  sweep         per unit: encode once, then loop configs; resumable JSONL
  merge         no model: fold shard files in --out into the base file (dedupe by key); stats: rewrite offsets

Config names (levels.py): shared 'A|B', offset 'q8|k+2' (needs --offsets), explicit 'A|L1,L2,..'.
Device: --device (default common.DEFAULT_DEVICE: mps on macOS, cuda elsewhere). Quantization runs on the
model's device on CUDA (--quant-path auto) and on CPU on MPS (as in all local results).
Shards: --shard i/n keeps the units at positions j with j % n == i of the deterministic unit order (after
--min-questions/--max-units); each shard writes <base>.shard-i-of-n.jsonl in the same --out, resume reads
every result file in --out, `--mode merge` concatenates.

Levels are applied through the adapter tensor interface (measurement_v15.adapter.tensors/wrap), so the
same code serves Qwen (1+3), Granite (1+7 entries) and DeepStack (1+geometry-dependent groups; inactive
zero-padded groups are left untouched and never count as payload).
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import time

from cloud_prep.measurement_v19 import common, levels

BASENAMES = {'extract': 'features.jsonl', 'means': 'means_units.jsonl', 'stats': 'stats_units.jsonl',
             'check-native': 'check_native.jsonl', 'cc-eval': 'cc_eval.jsonl', 'sweep': 'observations.jsonl'}


def inactive_keys(visual, family):
    if family == 'deepstack':
        active = visual.grid[0] * visual.grid[1]
        return {f'later{i}' for i in range(active, 4)}
    return set()


# ---- pure helpers (tested without a model) ---------------------------------------------------

def parse_shard(text):
    """'i/n' -> (i, n) with 0 <= i < n; None -> None."""
    if text is None:
        return None
    try:
        i, n = (int(x) for x in str(text).split('/'))
    except ValueError as exc:
        raise ValueError(f'--shard must be i/n, got {text!r}') from exc
    if n < 1 or not 0 <= i < n:
        raise ValueError(f'--shard needs 0 <= i < n, got {text!r}')
    return i, n


def shard_units(units, shard):
    """Units at positions j % n == i (positions in the given deterministic order)."""
    if shard is None:
        return list(units)
    i, n = shard
    return [u for j, u in enumerate(units) if j % n == i]


def out_name(mode, shard=None):
    base = BASENAMES[mode]
    if shard is None:
        return base
    stem = base[:-len('.jsonl')]
    return f'{stem}.shard-{shard[0]}-of-{shard[1]}.jsonl'


def result_files(out_dir, mode):
    """Base file and every shard file of this mode in out_dir (existing ones only, base first)."""
    out_dir = Path(out_dir)
    base = out_dir / BASENAMES[mode]
    stem = BASENAMES[mode][:-len('.jsonl')]
    return [p for p in [base] if p.exists()] + sorted(out_dir.glob(f'{stem}.shard-*-of-*.jsonl'))


def done_keys(paths):
    """Keys already written, over one path or several (shards). A truncated line stops the run."""
    if isinstance(paths, (str, Path)):
        paths = [paths]
    found = set()
    for path in paths:
        if not Path(path).exists():
            continue
        with open(path) as f:
            for n, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f'{path}:{n} incomplete; repair tail before resuming') from exc
                found.add(r['key'])
    return found


def sweep_key(row_id, config):
    return f'{row_id}::{config}'


def feature_key(unit_id, row, first_image):
    """Key of one coded image: the unit id for the unit's first image (all GQA units; old extract dirs),
    unit@sha16 for any other page image of a DocVQA document. Feature dirs and byte tables use it."""
    if row['source_images']['a'] == first_image:
        return unit_id
    return f"{unit_id}@{row['source_image_metadata']['a']['sha256'][:16]}"


def read_configs(configs=None, configs_file=None):
    """--configs plus --configs-file (JSON list, or one name per line; '#' comments), order kept, de-duplicated."""
    out = list(configs or [])
    if configs_file:
        text = Path(configs_file).read_text()
        if text.lstrip().startswith('['):
            out += json.loads(text)
        else:
            out += [l.split('#')[0].strip() for l in text.splitlines() if l.split('#')[0].strip()]
    seen, uniq = set(), []
    for c in out:
        if c not in seen:
            seen.add(c); uniq.append(c)
    return uniq


def coarse_configs(qps):
    cfg = ['C|C']
    for q in qps:
        cfg += [f'q{q}|C', f'C|q{q}', f'q{q}|q{q}']
    cfg += ['C|Z', 'Z|C', 'Z|Z', 'C|M', 'M|C', 'M|M']
    # Keep one quantized stream and zero the other.
    for q in qps:
        cfg += [f'q{q}|Z', f'Z|q{q}']
    return cfg


def byte_table_from_json(bt):
    """bytes.json -> {unit: {(stream, qp|'raw'): bytes}}."""
    return {u: {(r['stream'], r['qp'] if r['qp'] == 'raw' else int(r['qp'])): r['bytes'] for r in recs}
            for u, recs in bt['per_unit'].items()}


def table_keys(bt, main_key, offsets=None):
    """Keys for QP resolution: main + offsets' later keys if given, else every stream in the byte table."""
    if offsets is not None:
        return [main_key] + list(levels.offset_map(offsets))
    ks = {r['stream'] for recs in bt['per_unit'].values() for r in recs} if bt else set()
    return [main_key] + levels.later_keys(ks | {main_key}, main_key)


def written_qps(name):
    """QPs written in a shared/explicit name (no key set needed)."""
    form, x, y = levels.parse_config(name)
    if form == 'offset':
        raise ValueError(f'Offset config {name!r} needs keys and offsets')
    parts = [x] + (y if form == 'explicit' else [y])
    return {levels.parse_level(p)[1] for p in parts} - {None}


def check_configs(configs, *, main_key, keys=None, offsets=None, table_qps=None, allow_missing_bytes=False,
                  table_qps_by_stream=None):
    """Validate names and resolution; returns the sorted QPs the configs need (on ``keys`` when given,
    else as written). With table_qps_by_stream (bytes.json 'qps_by_stream') the check is per key."""
    for c in configs:
        form = levels.parse_config(c)[0]
        if form == 'offset' and offsets is None:
            raise ValueError(f'Offset config {c!r} needs --offsets')
    if keys is None:
        need = sorted(set().union(*[written_qps(c) for c in configs])) if configs else []
    else:
        need = levels.config_qps(configs, keys, main_key, offsets)
    if need and not allow_missing_bytes:
        if table_qps is None:
            raise ValueError('Quantized configs need --bytes (or --allow-missing-bytes)')
        if table_qps_by_stream and keys is not None:
            miss = set()
            for c in configs:
                for k, lv in levels.resolve_config(c, keys, main_key, offsets).items():
                    kind, qp = levels.parse_level(lv)
                    if kind == 'q' and k in table_qps_by_stream and qp not in table_qps_by_stream[k]:
                        miss.add((k, qp))
            if miss:
                raise ValueError(f'Byte table lacks (key, QP) {sorted(miss)}')
            return need
        missing = set(need) - {int(q) for q in table_qps}
        if missing:
            raise ValueError(f'Byte table lacks QPs {sorted(missing)}')
    return need


def config_fields(name, keys, main_key, offsets=None, inactive=()):
    """Record fields for one config: as-written parts, form, resolved per-key levels, offset q/k."""
    form, x, y = levels.parse_config(name)
    ml, ll = levels.split_config(name)
    return dict(config=name, main_level=ml, later_level=ll, config_form=form,
                levels=levels.resolve_config(name, keys, main_key, offsets),
                inactive_keys=sorted(inactive),
                offset_q=x if form == 'offset' else None, offset_k=y if form == 'offset' else None)


def record_bytes(table, raw, name, keys, main_key, *, inactive=(), offsets=None, allow_missing=False):
    """(bytes {main, later}, bytes_per_key) or (None, None) when allowed and the table lacks an entry.
    C counts raw BF16 (2 bytes/element) from ``raw``; q levels need ``table``."""
    t = {**raw, **(table or {})}
    try:
        b = levels.config_bytes(t, main_key, name, None, keys, inactive=inactive, offsets=offsets, detail=True)
    except ValueError:
        if allow_missing:
            return None, None
        raise
    per = b.pop('per_key')
    return b, per


def load_jsonl_by_key(paths):
    out = {}
    for p in paths:
        with open(p) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    out.setdefault(r['key'], r)
    return out


def merge_files(out_dir, mode):
    """Fold every shard file into the base file (dedupe by key, base rows first). Returns counts."""
    files = result_files(out_dir, mode)
    rows, order, conflicts = {}, [], 0
    for p in files:
        with open(p) as f:
            for n, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f'{p}:{n} incomplete; repair before merging') from exc
                k = r['key']
                if k in rows:
                    conflicts += rows[k] != line.strip()
                    continue
                rows[k] = line.strip(); order.append(k)
    base = Path(out_dir) / BASENAMES[mode]
    tmp = base.with_suffix('.jsonl.tmp')
    with open(tmp, 'w') as f:
        for k in order:
            f.write(rows[k] + '\n')
    tmp.replace(base)
    return dict(files=[str(p) for p in files], rows=len(order), duplicate_keys_with_different_content=conflicts)


def aggregate_stats(paths):
    """StreamStats over every stats_units line (one per unit; first line per unit wins)."""
    from cloud_prep.measurement_v19.offset_axis import StreamStats
    total, seen, model, main_key = StreamStats(), set(), None, None
    for p in paths:
        with open(p) as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                if r['unit_id'] in seen:
                    continue
                seen.add(r['unit_id']); model = model or r.get('model'); main_key = main_key or r.get('main_key')
                total.merge(StreamStats.from_json(r['stats']))
    return total, model, main_key


def write_offsets(out_dir, model=None, main_key=None, dest=None, source=None):
    """Offsets JSON from every stats file in out_dir (model / main key default to those in the lines)."""
    from cloud_prep.measurement_v19 import offset_axis
    st, m, mk = aggregate_stats(result_files(out_dir, 'stats'))
    model = model or m
    main_key = main_key or mk or ('layer0' if str(model).startswith('ibm') else 'main')
    t = offset_axis.offsets_table(st, model, main_key, source=source or f'runner stats {Path(out_dir).name} ({len(st.units)} units)')
    offset_axis.save_offsets(t, dest or Path(out_dir) / 'offsets.json')
    return t


def device_info(torch, device):
    info = dict(device=device, torch=torch.__version__)
    if device == 'cuda':
        p = torch.cuda.get_device_properties(0)
        info.update(gpu=torch.cuda.get_device_name(0), gpu_count=torch.cuda.device_count(),
                    capability=list(torch.cuda.get_device_capability(0)), gpu_total_memory=p.total_memory,
                    cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                    cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                    allow_tf32_matmul=torch.backends.cuda.matmul.allow_tf32, allow_tf32_cudnn=torch.backends.cudnn.allow_tf32)
        try:
            import subprocess
            info['driver'] = subprocess.run(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader', '-i', '0'],
                                            capture_output=True, text=True, timeout=20).stdout.strip() or None
        except Exception:
            info['driver'] = None
    else:
        info.update(mps_available=torch.backends.mps.is_available())
    return info


def library_versions(names=('torch', 'transformers', 'Pillow', 'numpy', 'accelerate', 'peft', 'safetensors')):
    out = {}
    for n in names:
        try:
            out[n] = importlib.metadata.version(n)
        except importlib.metadata.PackageNotFoundError:
            if n in ('torch', 'transformers', 'Pillow'):
                raise
    return out


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--mode', required=True, choices=['extract', 'means', 'stats', 'check-native', 'cc-eval', 'sweep', 'merge'])
    p.add_argument('--model', default='qwen8b')
    p.add_argument('--manifest', type=Path)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--device', choices=['mps', 'cuda'], help='default: DEEPSTACK_V19_DEVICE / platform default')
    p.add_argument('--features', type=Path, help='extract output dir (sweep: feature sha check)')
    p.add_argument('--bytes', type=Path, help='rate table JSON from rate_audit.py')
    p.add_argument('--allow-missing-bytes', action='store_true',
                   help='quantized configs without a byte entry are still run; their bytes fields are null')
    p.add_argument('--means', type=Path, help='mean vectors .npz (sweep: read; means mode: written)')
    p.add_argument('--offsets', type=Path, help='offsets JSON (offset_axis); required by offset configs q<q>|k<k>')
    p.add_argument('--offsets-out', type=Path, help='stats mode: offsets JSON path (default OUT/offsets.json)')
    p.add_argument('--qps', type=int, nargs='*', default=[])
    p.add_argument('--configs', nargs='*', help='explicit config names; default = coarse sweep set from --qps')
    p.add_argument('--configs-file', type=Path, help='JSON list or one config per line; appended to --configs')
    p.add_argument('--min-questions', type=int)
    p.add_argument('--max-units', type=int)
    p.add_argument('--limit-questions', type=int)
    p.add_argument('--shard', help='i/n: this process takes units j with j %% n == i')
    p.add_argument('--quant-path', choices=['auto', 'cpu', 'device'], default='auto',
                   help='auto: device on cuda, cpu on mps (the local results)')
    p.add_argument('--replica-check-units', type=int,
                   help='device quantization: bitwise check vs CPU replica on the first N units (default 2 on '
                        'the device path, 0 on cpu); a mismatch stops the run')
    p.add_argument('--lora', type=Path, help='LoRA directory (train.lora_utils.attach_lora); evaluated unmerged')
    p.add_argument('--warmup', action='store_true',
                   help='one discarded (C,C) generation on the first pending question before writing anything')
    return p.parse_args(argv)


def merge_main(a):
    common.require_under_root(a.out)
    report = {}
    for mode in BASENAMES:
        if mode == 'means':
            continue
        if any('.shard-' in p.name for p in result_files(a.out, mode)):
            report[mode] = merge_files(a.out, mode)
    if result_files(a.out, 'stats'):
        t = write_offsets(a.out, dest=a.offsets_out)
        report['offsets'] = dict(path=str(a.offsets_out or a.out / 'offsets.json'), model=t['model'], d=t['d'],
                                 n_units=t['n_units'])
    print(json.dumps(report, indent=1))
    return 0


def main(argv=None):
    a = parse_args(argv)
    common.require_research()
    a.out = common.require_under_root(a.out)
    if a.mode == 'merge':
        return merge_main(a)
    shard = parse_shard(a.shard)
    if shard and a.mode == 'means':
        raise ValueError('means mode is not sharded (it averages in memory)')
    device = common.device_name(a.device)
    a.out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('TMPDIR', str(common.RUN_ROOT / 'tmp'))
    import numpy as np
    import torch
    from deepstack.readout_v3 import fp32_output_head
    from cloud_prep.measurement_v15.adapter import tensors, wrap, geometry

    if device == 'cuda':
        # As the accepted v15 CUDA runs (stored cloud CC): no TF32.
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        common.reset_peak(torch, device)
    on_device = a.quant_path == 'device' or (a.quant_path == 'auto' and device == 'cuda')
    n_replica = a.replica_check_units if a.replica_check_units is not None else (2 if on_device else 0)

    offsets = None
    if a.offsets:
        from cloud_prep.measurement_v19.offset_axis import load_offsets
        offsets = load_offsets(a.offsets)
        if offsets.get('model') not in (None, a.model):
            raise ValueError(f"Offsets table is for {offsets.get('model')}, not {a.model}")

    rows = common.load_rows(a.manifest or common.CAL_MANIFEST)
    units = shard_units(common.take_units(rows, a.min_questions, a.max_units), shard)
    started = time.monotonic()
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    lora_meta = None
    if a.lora:
        lm = a.lora / 'lora_meta.json'
        lora_meta = dict(path=str(a.lora.resolve()), meta=json.loads(lm.read_text()) if lm.exists() else None)
    meta = dict(mode=a.mode, model=a.model, manifest=str(a.manifest or common.CAL_MANIFEST), run_id=run_id,
                units=len(units), questions=sum(len(q) for _, q in units), platform=platform.platform(),
                libraries=library_versions(),
                generate_tokens=common.GENERATE_TOKENS,
                metric='; '.join(common.metric_label(b) for b in sorted({r['benchmark']['name'] for r in rows})),
                device=device, device_info=device_info(torch, device), dtype='bfloat16', batch_size=1,
                shard=list(shard) if shard else None, max_units=a.max_units, min_questions=a.min_questions,
                quantization=dict(path='device' if on_device else 'cpu', replica_check_units=n_replica,
                                  note='float32 replica of NNCodec URQ (levels.quantize), cast to the model dtype'),
                offsets=offsets, offsets_path=str(a.offsets) if a.offsets else None, lora=lora_meta,
                allow_missing_bytes=a.allow_missing_bytes, status='loading')
    metapath = a.out / (f'run-{a.mode}-{run_id}' + (f'-shard-{shard[0]}-of-{shard[1]}' if shard else '') + '.json')
    common.write_json(metapath, meta)
    t = time.monotonic()
    adapter, family, ameta, snapshot = common.load_model(a.model, device=device)
    if a.lora:
        from cloud_prep.measurement_v19.train.lora_utils import attach_lora
        attach_lora(adapter, family, a.lora)
    model, processor = adapter.model, adapter.processor
    common.sync(torch, device)
    meta.update(load_seconds=time.monotonic() - t, adapter=ameta, snapshot=snapshot,
                image_processor=processor.image_processor.to_dict(), status='running')
    genconfig, eos = common.make_genconfig(model)
    meta['generation_config'] = genconfig.to_dict()
    main_key = common.FAMILY_MAIN_KEY[family]
    peak = 0

    configs = read_configs(a.configs, a.configs_file) or coarse_configs(a.qps)
    means = None
    byte_table = None
    bt = None
    if a.mode == 'sweep':
        if any('M' in c for c in configs):
            if not a.means:
                raise ValueError('M configs need --means')
            z = np.load(a.means)
            means = {k: torch.from_numpy(z[k]) for k in z.files}
        bt = json.loads(a.bytes.read_text()) if a.bytes else None
        meta['config_qps'] = check_configs(configs, main_key=main_key,
                                           keys=table_keys(bt, main_key, offsets) if (bt or offsets) else None,
                                           offsets=offsets, table_qps=bt['qps'] if bt else None,
                                           allow_missing_bytes=a.allow_missing_bytes,
                                           table_qps_by_stream=(bt or {}).get('qps_by_stream'))
        if bt is not None:
            byte_table = byte_table_from_json(bt)
        meta['configs'] = configs
    feature_recs = None
    if a.mode == 'sweep' and a.features is not None:
        feature_recs = load_jsonl_by_key(result_files(a.features, 'extract'))     # by feature_key

    if a.mode == 'means':
        if not a.means:
            raise ValueError('--means output path required')
        acc = {}
        (a.out / 'means_units.jsonl').unlink(missing_ok=True)   # means are recomputed in full, never resumed
    out_path = a.out / out_name(a.mode, shard)
    done = done_keys(result_files(a.out, a.mode)) if a.mode != 'means' else set()
    n_questions = 0
    replica_checked = 0
    meta['device_replica_checks'] = []

    def replica_check(unit_id, vals, inactive):
        nonlocal replica_checked
        if not on_device or replica_checked >= n_replica or a.mode != 'sweep':
            return
        replica_checked += 1
        for k, v in vals.items():
            if k in inactive:
                continue
            qps = sorted({levels.parse_level(levels.resolve_config(c, list(vals), main_key, offsets)[k])[1]
                          for c in configs} - {None})
            if not qps:
                continue
            r = levels.device_replica_check(v, qps)
            meta['device_replica_checks'].append(dict(unit_id=unit_id, key=k, qps=qps, all_equal=r['all_equal'],
                                                      mismatches=sum(x['mismatches'] for x in r['per_qp'].values())))
            if not r['all_equal']:
                common.write_json(metapath, meta)
                raise RuntimeError(f'Device quantization differs from the CPU replica ({unit_id}, {k}); '
                                   'rerun with --quant-path cpu')

    try:
        with out_path.open('a', buffering=1) as out, torch.inference_mode(), fp32_output_head(model):
            if a.warmup and a.mode == 'sweep':
                first = next(((u, r) for u, qs in units for r in qs
                              if any(sweep_key(r['id'], c) not in done for c in configs)), None)
                if first is not None:
                    wrow = first[1]
                    winp, _ = adapter.prepare(wrow, common.variant(wrow), 'a')
                    wsrc = adapter.encode(winp)
                    wvals = tensors(wsrc, family)
                    wvis = wrap(wsrc, levels.apply_config(wvals, main_key, 'C', 'C', inactive=inactive_keys(wsrc, family)), family, {})
                    if family == 'ibm':
                        winp, _ = adapter.prepare(wrow, common.variant(wrow), 'a')
                    wres = common.generate_text(adapter, family, winp, genconfig, eos, visual=wvis)
                    meta['warmup'] = dict(id=wrow['id'], generated_text=wres['generated_text'][:200], discarded=True)
                    del wsrc, wvals, wvis
            for unit_id, qrows in units:
                if a.limit_questions is not None and n_questions >= a.limit_questions:
                    break
                if a.mode == 'means':
                    # Level-M vectors from calibration images only: per stream, image-weighted mean of
                    # per-image token means over images where the stream is active. Nothing is saved per token.
                    # every distinct image of the unit (DocVQA documents can have several page images)
                    for img_row in {r['source_images']['a']: r for r in qrows}.values():
                        inp, _ = adapter.prepare(img_row, common.variant(img_row), 'a')
                        visual = adapter.encode(inp)
                        vals = tensors(visual, family); inactive = inactive_keys(visual, family)
                        for k, v in vals.items():
                            if k not in inactive:
                                acc.setdefault(k, []).append(v.detach().float().mean(dim=0).cpu().numpy())
                        out.write(json.dumps(dict(key=unit_id, unit_id=unit_id, image=img_row['source_images']['a'],
                                                  active=sorted(set(vals) - inactive))) + '\n')
                    n_questions += len(qrows)
                    print(json.dumps(dict(event='unit', mode=a.mode, unit_id=unit_id, seconds=time.monotonic() - started)), flush=True)
                    continue
                if a.mode == 'stats':
                    # Offsets: population n/sum/sumsq per stream over every distinct image of the unit
                    # (GQA: one image), active streams only; float64 on CPU (MPS has no float64).
                    if unit_id in done:
                        continue
                    from cloud_prep.measurement_v19.offset_axis import StreamStats
                    st = StreamStats()
                    imgs = list({r['source_images']['a']: r for r in qrows}.values())
                    t_enc = time.monotonic()
                    for img_row in imgs:
                        inp, _ = adapter.prepare(img_row, common.variant(img_row), 'a')
                        visual = adapter.encode(inp)
                        vals = tensors(visual, family); inactive = inactive_keys(visual, family)
                        for k, v in vals.items():
                            if k not in inactive:
                                st.update(k, v.detach().float().cpu().numpy(), unit=unit_id)
                    out.write(json.dumps(dict(key=unit_id, unit_id=unit_id, model=a.model, main_key=main_key,
                                              images=[r['source_images']['a'] for r in imgs],
                                              encode_seconds=time.monotonic() - t_enc, stats=st.to_json())) + '\n')
                    n_questions += len(qrows)
                    print(json.dumps(dict(event='unit', mode=a.mode, unit_id=unit_id, seconds=time.monotonic() - started)), flush=True)
                    continue
                if a.mode == 'extract':
                    # One record per distinct image (DocVQA pages): key = feature_key (unit id for the first image).
                    first_img = qrows[0]['source_images']['a']
                    for page, img_row in enumerate({r['source_images']['a']: r for r in qrows}.values()):
                        fkey = feature_key(unit_id, img_row, first_img)
                        if fkey in done:
                            continue
                        inp, _ = adapter.prepare(img_row, common.variant(img_row), 'a')
                        visual = adapter.encode(inp)
                        vals = tensors(visual, family)
                        udir = a.out / 'features' / fkey.replace('/', '_')
                        udir.mkdir(parents=True, exist_ok=True)
                        rec = dict(key=fkey, unit_id=unit_id, page=page, image=img_row['source_images']['a'],
                                   image_sha256=img_row['source_image_metadata']['a']['sha256'],
                                   geometry=geometry(visual, family), inactive=sorted(inactive_keys(visual, family)),
                                   streams={})
                        for k, v in vals.items():
                            np.save(udir / f'{k}.npy', v.detach().float().cpu().numpy())
                            rec['streams'][k] = dict(shape=list(v.shape), dtype=str(v.dtype), sha256_bf16=common.tensor_sha(v))
                        out.write(json.dumps(rec) + '\n')
                    n_questions += len(qrows)
                    print(json.dumps(dict(event='unit', mode=a.mode, unit_id=unit_id, seconds=time.monotonic() - started)), flush=True)
                    continue

                pending = []
                for row in qrows:
                    if a.mode == 'sweep':
                        pending += [(row, c) for c in configs if sweep_key(row['id'], c) not in done]
                    elif row['id'] not in done:
                        pending.append((row, 'C|C'))
                if not pending:
                    n_questions += len(qrows)
                    continue
                inp0, _ = adapter.prepare(qrows[0], common.variant(qrows[0]), 'a')
                t_enc = time.monotonic()
                source = adapter.encode(inp0)
                common.sync(torch, device)
                enc_seconds = time.monotonic() - t_enc
                vals = tensors(source, family)
                inactive = inactive_keys(source, family)
                shas = {k: common.tensor_sha(v) for k, v in vals.items()}
                first_img = cur_img = qrows[0]['source_images']['a']
                fkey = unit_id

                def check_features():
                    if feature_recs is not None:
                        frec = feature_recs.get(fkey)
                        if frec is None or any(frec['streams'][k]['sha256_bf16'] != shas[k] for k in vals):
                            raise ValueError('Runner features differ from the coded features: ' + fkey)
                check_features()
                replica_check(unit_id, vals, inactive)
                visuals = {}
                for row in qrows:
                    if a.limit_questions is not None and n_questions >= a.limit_questions:
                        break
                    n_questions += 1
                    todo = [c for r, c in pending if r is row]
                    if not todo:
                        continue
                    if row['source_images']['a'] != cur_img:
                        # DocVQA: questions of one document can use different page images. Re-encode for this
                        # page (GQA units have one image, so this never triggers there); bytes and the feature
                        # check use this page's feature_key.
                        inp_img, _ = adapter.prepare(row, common.variant(row), 'a')
                        t_enc = time.monotonic(); source = adapter.encode(inp_img); common.sync(torch, device)
                        enc_seconds = time.monotonic() - t_enc
                        cur_img = row['source_images']['a']
                        fkey = feature_key(unit_id, row, first_img)
                        vals = tensors(source, family); inactive = inactive_keys(source, family)
                        shas = {k: common.tensor_sha(v) for k, v in vals.items()}
                        check_features()
                        visuals = {}
                    inp, prompt = adapter.prepare(row, common.variant(row), 'a')
                    if a.mode == 'check-native':
                        cc = wrap(source, levels.apply_config(vals, main_key, 'C', 'C', inactive=inactive), family, {})
                        common.sync(torch, device); t0 = time.monotonic()
                        native = common.generate_text(adapter, family, inp, genconfig, eos)
                        common.sync(torch, device); t1 = time.monotonic()
                        patched = common.generate_text(adapter, family, inp, genconfig, eos, visual=cc)
                        common.sync(torch, device); t2 = time.monotonic()
                        ln = common.prefill_logits(adapter, inp)
                        lp = common.prefill_logits(adapter, inp, visual=cc)
                        diff = (ln - lp).abs().max().item()
                        rec = dict(key=row['id'], id=row['id'], unit_id=unit_id, native=native, patched=patched,
                                   text_equal=native['generated_text'] == patched['generated_text'],
                                   tokens_equal=native['generated_token_ids'] == patched['generated_token_ids'],
                                   prefill_logits_max_abs_diff=diff, prefill_logits_argmax_equal=int(ln.argmax()) == int(lp.argmax()),
                                   native_seconds=t1 - t0, patched_seconds=t2 - t1, device=device)
                        out.write(json.dumps(rec) + '\n')
                        continue
                    keys = list(vals)
                    raw = {(k, 'raw'): 2 * v.numel() for k, v in vals.items()}
                    for c in todo:
                        cf = config_fields(c, keys, main_key, offsets, inactive)
                        common.sync(torch, device); t0 = time.monotonic()
                        if c not in visuals:
                            visuals[c] = wrap(source, levels.apply_levels(vals, cf['levels'], means=means, inactive=inactive,
                                                                          on_device=on_device), family, {'config': c})
                        common.sync(torch, device); t1 = time.monotonic()
                        if family == 'ibm':
                            # Granite: fresh processor inputs per config. One calibration run failed the adapter's
                            # placeholder-count check on reused inputs after 35 configs; a failed check is retried
                            # once with freshly prepared inputs and recorded.
                            inp, _ = adapter.prepare(row, common.variant(row), 'a')
                            retried = False
                            try:
                                result = common.generate_text(adapter, family, inp, genconfig, eos, visual=visuals[c])
                            except ValueError as exc:
                                if 'placeholder' not in str(exc):
                                    raise
                                inp, _ = adapter.prepare(row, common.variant(row), 'a')
                                result = common.generate_text(adapter, family, inp, genconfig, eos, visual=visuals[c])
                                retried = True
                            result['granite_geometry_retry'] = retried
                        else:
                            result = common.generate_text(adapter, family, inp, genconfig, eos, visual=visuals[c])
                        common.sync(torch, device); t2 = time.monotonic()
                        sc = common.score(row, result)
                        rec = dict(key=sweep_key(row['id'], c) if a.mode == 'sweep' else row['id'], id=row['id'],
                                   unit_id=unit_id, cluster_id=row['benchmark'].get('cluster_id', unit_id),
                                   task=row['task'], split=row.get('split'), **cf,
                                   query=row['query'], reference_answers=row['reference_answers'],
                                   result=result, **sc, transform_seconds=t1 - t0, generate_seconds=t2 - t1,
                                   encode_seconds_unit=enc_seconds, feature_sha256_bf16=shas, run_id=run_id,
                                   image=row['source_images']['a'], feature_key=fkey, device=device,
                                   quant_path='device' if on_device else 'cpu',
                                   lora=lora_meta['path'] if lora_meta else None)
                        if a.mode == 'sweep':
                            rec['bytes'], rec['bytes_per_key'] = record_bytes(
                                byte_table.get(fkey) if byte_table is not None else None, raw, c, keys, main_key,
                                inactive=inactive, offsets=offsets, allow_missing=a.allow_missing_bytes)
                        out.write(json.dumps(rec, ensure_ascii=False) + '\n')
                    peak = max(peak, common.peak_bytes(torch, device))
                    print(json.dumps(dict(event='question', id=row['id'], configs=len(todo), seconds=time.monotonic() - started)), flush=True)
                visuals.clear()
                peak = max(peak, common.peak_bytes(torch, device))
                meta.update(memory=common.mem(torch, device), questions_seen=n_questions, peak_bytes=peak)
                meta['peak_mps_driver_bytes' if device == 'mps' else 'peak_cuda_reserved_bytes'] = peak
                common.write_json(metapath, meta)
        if a.mode == 'means':
            np.savez(a.means, **{k: np.mean(v, axis=0).astype(np.float32) for k, v in acc.items()})
            meta['means_images'] = {k: len(v) for k, v in acc.items()}
        if a.mode == 'stats' and shard is None:
            tbl = write_offsets(a.out, a.model, main_key, a.offsets_out)
            meta['offsets_written'] = dict(path=str(a.offsets_out or a.out / 'offsets.json'), d=tbl['d'],
                                           n_units=tbl['n_units'], units_sha256=tbl['units_sha256'])
        elif a.mode == 'stats':
            meta['offsets_written'] = 'sharded: run --mode merge --out OUT after all shards'
        meta['status'] = 'complete'
    except BaseException as exc:
        meta.update(status='failed_or_interrupted', error=repr(exc))
        raise
    finally:
        try:
            peak = max(peak, common.peak_bytes(torch, device))
        except Exception:
            pass
        meta.update(wall_seconds=time.monotonic() - started, peak_bytes=peak, questions_seen=n_questions,
                    finished_at=datetime.now(timezone.utc).isoformat())
        meta['peak_mps_driver_bytes' if device == 'mps' else 'peak_cuda_reserved_bytes'] = peak
        common.write_json(metapath, meta)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
