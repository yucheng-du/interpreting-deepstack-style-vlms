"""Cross-evaluation commands: base model + 4 arms x {short, long} checkpoints, one config list, one manifest.

Each evaluation is a runner sweep with real NNCodec bytes (--bytes from rate_audit on the evaluation features) on
the same manifest, the same --configs-file and the training offsets; trained arms add --lora <checkpoint dir>.
Fixed layout:
  <out-root>/<model>[/<tag>]/plan.json  configs.txt  commands.sh  [commands-gpu<g>.sh]
  <out-root>/<model>[/<tag>]/<name>/    runner --out of one evaluation; name = base | <arm>-<label>
Checkpoints: <train-root>/<model>[/<tag>]/<arm>/checkpoints/<label>-step*/. lora_meta.json is checked across arms:
same model, seed, training rows and order, LoRA init, hyperparameters, offsets; compression arms also the same
budget, |k| distribution, q grid, full-input mix and rate table. Config list: --configs-file (one name per line)
or offset_axis families from --qs/--ks (corners incl. C|C, S, the whole offset axis, single-stream).

  python -m cloud_prep.measurement_v19.train.eval_plan --model qwen8b --manifest EVAL.jsonl --bytes BYTES.json \
      --means MEANS.npz --qs 0 1 2 3 4 5 6 8 --ks 2 4 [--tag dev1] [--gpus 4] [--runner-args "--max-units 120"]
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shlex

from cloud_prep.measurement_v19 import levels, offset_axis
from cloud_prep.measurement_v19.train import lora_utils, schedule

ALL_KEYS = ('model', 'family', 'seed', 'role', 'sample_order_sha256', 'train_manifest_sha256', 'n_samples',
            'hyperparameters', 'base', 'lora.init_sha256', 'lora.modules', 'offsets.sha256')
COMPRESSION_KEYS = ('budget', 'k_abs', 'k_weights', 'q_grid', 'full_mix', 'rate_table.sha256')


def get(meta, dotted):
    v = meta
    for p in dotted.split('.'):
        v = v.get(p) if isinstance(v, dict) else None
    return v


def find_checkpoints(train_dir, arms, labels):
    """{(arm, label): (path, meta)}; exactly one <label>-step* dir per arm and label."""
    found = {}
    for arm in arms:
        for label in labels:
            ds = sorted((Path(train_dir) / arm / 'checkpoints').glob(f'{label}-step*'))
            ds = [d for d in ds if not d.name.endswith(('.tmp', '.old'))]
            if len(ds) != 1:
                raise ValueError(f'{arm}: expected one {label}-step* checkpoint, found {[d.name for d in ds]}')
            d = ds[0]
            for f in ('adapter_config.json', 'lora_meta.json'):
                if not (d / f).exists():
                    raise ValueError(f'{d} lacks {f}')
            meta = lora_utils.read_lora_meta(d)
            if meta['arm'] != arm or meta['checkpoint']['label'] != label:
                raise ValueError(f'{d}: lora_meta says {meta["arm"]}/{meta["checkpoint"]["label"]}')
            found[(arm, label)] = (d, meta)
    return found


def check_matching(found):
    """List of mismatches across checkpoints (empty when the arms are comparable)."""
    bad = []
    metas = [(k, m) for k, (_, m) in sorted(found.items())]
    for key in ALL_KEYS:
        vals = {json.dumps(get(m, key), sort_keys=True) for _, m in metas}
        if len(vals) > 1:
            bad.append(dict(key=key, values=sorted(vals)))
    comp = [(k, m) for k, m in metas if k[0] != 'sft']
    for key in COMPRESSION_KEYS:
        vals = {json.dumps(get(m, key), sort_keys=True) for _, m in comp}
        if len(vals) > 1:
            bad.append(dict(key=key, values=sorted(vals), arms='compression'))
    by_label = {}
    for (arm, label), m in metas:
        by_label.setdefault(label, set()).add(m['checkpoint']['step'])
    for label, steps in by_label.items():
        if len(steps) > 1:
            bad.append(dict(key=f'checkpoint step ({label})', values=sorted(steps)))
    return bad


def eval_configs(a, role):
    if a.configs_file:
        names = [l.strip() for l in a.configs_file.read_text().splitlines() if l.strip() and not l.startswith('#')]
    else:
        fam = offset_axis.config_families(a.qs, a.ks, role, mean_corners=not a.no_mean_corners)
        names = offset_axis.flat_configs(fam, a.families)
    for n in names:
        levels.parse_config(n)
    if 'C|C' not in names:
        raise ValueError('The evaluation list must contain the full input C|C')
    if len(set(names)) != len(names):
        raise ValueError('Duplicate config names')
    return names


def command(a, out, configs_file, offsets, lora=None):
    parts = shlex.split(a.runner) + ['--mode', 'sweep', '--model', a.model, '--manifest', str(a.manifest),
                                     '--out', str(out), '--bytes', str(a.bytes), '--offsets', str(offsets),
                                     '--configs-file', str(configs_file)]
    if a.means:
        parts += ['--means', str(a.means)]
    if a.features:
        parts += ['--features', str(a.features)]
    if lora:
        parts += ['--lora', str(lora)]
    parts += shlex.split(a.runner_args or '')
    return ' '.join(shlex.quote(p) for p in parts)


def build(a):
    from cloud_prep.measurement_v19 import common
    tag = [a.tag] if a.tag else []
    train_dir = Path(a.train_root or common.RUN_ROOT / 'train').joinpath(a.model, *tag)
    out_dir = Path(a.out_root or common.RUN_ROOT / 'train-eval').joinpath(a.model, *tag)
    found = find_checkpoints(train_dir, a.arms, a.labels) if a.arms else {}
    bad = check_matching(found) if found else []
    if bad and not a.allow_mismatch:
        raise ValueError('Checkpoints are not comparable: ' + json.dumps(bad)[:2000])
    metas = [m for _, m in found.values()]
    role = metas[0]['role'] if metas else json.loads(schedule.default_config(a.model).read_text())['role']
    configs = eval_configs(a, role)
    train_budget = next((m['budget'] for m in metas if m['arm'] != 'sft'), None)
    refs = set((train_budget or {}).get('train_refs') or []) if (train_budget or {}).get('mode') == 'ref_config' else set()
    evals = []
    if not a.no_base:
        evals.append(dict(name='base', arm=None, label=None, step=None, lora=None))
    for (arm, label), (d, m) in sorted(found.items(), key=lambda kv: (a.arms.index(kv[0][0]), a.labels.index(kv[0][1]))):
        evals.append(dict(name=f'{arm}-{label}', arm=arm, label=label, step=m['checkpoint']['step'], lora=str(d)))
    _, offsets = schedule.load_offsets(a.offsets or a.model)
    offsets = offsets.resolve()
    for m in metas:
        if m['offsets']['sha256'] != schedule.file_sha256(offsets):
            raise ValueError(f'Evaluation offsets {offsets} differ from the training offsets of {m["arm"]}')
    for e in evals:
        e['out'] = str(out_dir / e['name'])
        e['command'] = command(a, e['out'], out_dir / 'configs.txt', offsets, e['lora'])
    fams = {c: offset_axis.config_family(c, role)['family'] for c in configs}
    plan = dict(model=a.model, tag=a.tag, manifest=str(a.manifest), manifest_sha256=schedule.file_sha256(a.manifest),
                bytes=str(a.bytes), means=str(a.means) if a.means else None, offsets=str(offsets), configs=configs,
                configs_sha256=hashlib.sha256('\n'.join(configs).encode()).hexdigest(), config_family=fams,
                train_refs=sorted(refs), configs_also_train_refs=[c for c in configs if c in refs],
                train_eval_refs=(train_budget or {}).get('eval_refs'), role=role, mismatches=bad,
                train_dir=str(train_dir), out_dir=str(out_dir), evaluations=evals)
    return plan


def write(plan, gpus):
    out = Path(plan['out_dir'])
    out.mkdir(parents=True, exist_ok=True)
    lora_utils.write_json(out / 'plan.json', plan)
    (out / 'configs.txt').write_text('\n'.join(plan['configs']) + '\n')
    head = '#!/usr/bin/env bash\nset -euo pipefail\n'
    (out / 'commands.sh').write_text(head + ''.join(e['command'] + '\n' for e in plan['evaluations']))
    if gpus > 1:
        for g in range(gpus):
            mine = plan['evaluations'][g::gpus]
            (out / f'commands-gpu{g}.sh').write_text(head + ''.join(f"CUDA_VISIBLE_DEVICES={g} {e['command']}\n" for e in mine))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', required=True)
    p.add_argument('--manifest', type=Path, required=True, help='evaluation manifest, the same for every evaluation')
    p.add_argument('--bytes', type=Path, required=True, help='real NNCodec byte table of the evaluation features')
    p.add_argument('--means', type=Path, help='level-M means (needed when M corners are in the list)')
    p.add_argument('--features', type=Path, help='runner extract dir for the sweep feature check')
    p.add_argument('--offsets', help='offsets JSON or model key (default offsets/<model>.json; must match training)')
    p.add_argument('--configs-file', type=Path)
    p.add_argument('--qs', type=int, nargs='*', default=[])
    p.add_argument('--ks', type=int, nargs='*', default=[])
    p.add_argument('--families', nargs='*', default=['corner', 'S', 'axis', 'single'])
    p.add_argument('--no-mean-corners', action='store_true')
    p.add_argument('--train-root', type=Path, help='default RUN_ROOT/train')
    p.add_argument('--out-root', type=Path, help='default RUN_ROOT/train-eval')
    p.add_argument('--tag')
    p.add_argument('--arms', nargs='*', default=list(schedule.ARMS), choices=schedule.ARMS)
    p.add_argument('--labels', nargs='+', default=['short', 'long'])
    p.add_argument('--no-base', action='store_true')
    p.add_argument('--runner', default='python -m cloud_prep.measurement_v19.runner')
    p.add_argument('--runner-args', default='', help='extra runner arguments, appended to every command')
    p.add_argument('--gpus', type=int, default=1, help='also write round-robin commands-gpu<g>.sh')
    p.add_argument('--allow-mismatch', action='store_true')
    a = p.parse_args(argv)
    if not a.configs_file and not a.qs:
        p.error('give --configs-file or --qs (and --ks)')
    if a.means is None and not a.no_mean_corners and not a.configs_file:
        p.error('M corners need --means (or --no-mean-corners)')
    plan = build(a)
    write(plan, a.gpus)
    print(json.dumps(dict(out=plan['out_dir'], evaluations=len(plan['evaluations']), configs=len(plan['configs']),
                          mismatches=len(plan['mismatches']))))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
