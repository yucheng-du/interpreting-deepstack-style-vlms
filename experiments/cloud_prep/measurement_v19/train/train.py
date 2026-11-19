"""v19 compression-aware LoRA pilot trainer (model environment, CUDA).

One run = one arm (sft | generic | role | anti) of one model (qwen8b | ibm4b): a single pass over the ordered
training rows, batch 1 with gradient accumulation, LoRA on the language model only (lora_utils), vision encoder
and projectors frozen and untouched. Per sample, with the evaluation prompt, template and delivery path:
  adapter.prepare(row, variant(row), 'a') -> frozen adapter.encode under no_grad -> measurement_v15 tensors ->
  the sample's config from schedule.py, coded on the device by levels.quantize_for_training (exact replica) ->
  wrap -> adapter.use(visual, inputs) delivers the coded streams natively (Qwen visual.forward + DeepStack
  injection; Granite get_image_features + pre-layer injection) -> language-model forward WITH grad on
  prompt + answer tokens (deepstack.scoring._continuation_inputs, as the option scorer) -> cross-entropy on the
  answer tokens only (reference answer + the model's end-of-turn token, eos[0] of the eval generation config).
Base BF16; LoRA weights and AdamW states float32; forward under BF16 autocast (config autocast); output head in
float32 with autocast off (config fp32_head, as fp32_output_head in evaluation). Granite gets fresh processor
inputs for the forward, as the runner does.

Outputs, <out> = --out or RUN_ROOT/train/<model>[/<tag>]/<arm>:
  config.json  run_identity.json  schedule.jsonl  schedule_summary.json
  train_log.jsonl   per sample: i, id, unit_id, config, q, k, kabs, budget, bytes_*, feasible, full, loss, ...
  steps.jsonl       per optimizer step: step, lr, loss_mean, grad_norm, peak_bytes, seconds
  run-<run_id>.json meta (status, versions, LoRA info, memory)   timing.json
  checkpoints/short-stepNNNNN/ and long-stepNNNNN/: adapter_model.safetensors, adapter_config.json, lora_meta.json
  state/latest/     resumable state; a rerun with the same arguments resumes from it (identity checked)

First cloud step (timing; stops after N samples, no checkpoint):
  python -m cloud_prep.measurement_v19.train.train --model qwen8b --arm role --manifest TRAIN.jsonl \
      --rate-table TABLE.jsonl --tag timing --max-samples 48 --timing
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import shutil
import time

from cloud_prep.measurement_v19 import levels
from cloud_prep.measurement_v19.train import lora_utils, schedule

LOG_KEYS = ('i', 'id', 'unit_id', 'arm', 'config', 'q', 'k', 'kabs', 'sign', 'full', 'budget_ref', 'budget',
            'bytes_main', 'bytes_later', 'bytes_total', 'raw_bytes', 'feasible')


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', required=True, choices=['qwen8b', 'ibm4b'])
    p.add_argument('--arm', required=True, choices=schedule.ARMS)
    p.add_argument('--manifest', type=Path, required=True, help='ordered training manifest (runner row format)')
    p.add_argument('--rate-table', type=Path, help='training code-length table (rate_table.py); required except sft')
    p.add_argument('--config', type=Path, help='default: train/configs/<model>.json')
    p.add_argument('--set', action='append', default=[], metavar='KEY=JSON',
                   help='override a config value, dotted keys for nested, e.g. --set lr=5e-5 '
                        '--set budget.train_refs=\'["q2|k0"]\'')
    p.add_argument('--offsets', help='offsets JSON or model key (default: cloud_prep/measurement_v19/offsets/<model>.json)')
    p.add_argument('--out', type=Path)
    p.add_argument('--tag', help='extra directory level for dev-stage variants')
    p.add_argument('--device', choices=['cuda', 'mps'])
    p.add_argument('--max-samples', type=int, help='stop after this many samples in this invocation (timing)')
    p.add_argument('--timing', action='store_true', help='synchronize and time encode/quantize/forward/backward')
    p.add_argument('--dry-run', action='store_true', help='write config and schedule, load no model')
    p.add_argument('--fresh', action='store_true', help='refuse to resume (output dir must be empty)')
    a = p.parse_args(argv)
    if a.arm != 'sft' and a.rate_table is None:
        p.error('compression arms need --rate-table')
    return a


# ---- per-sample pieces (kept separate so a tiny model can exercise them) -------------------------

def answer_text(row, mode='first_reference'):
    if mode != 'first_reference':
        raise ValueError('answer mode must be first_reference')
    ans = row['reference_answers'][0]
    if not isinstance(ans, str) or not ans.strip():
        raise ValueError(f"Empty reference answer for {row['id']}")
    return ans


def target_ids(tok, answer, end_id):
    """Answer tokens as the model would generate them after the prompt, then the end-of-turn token."""
    ids = list(tok(answer, add_special_tokens=False)['input_ids'])
    if not ids:
        raise ValueError('Answer tokenizes to nothing')
    return ids + [int(end_id)]


def inactive_keys(visual, family):
    if family == 'deepstack':
        return {f'later{i}' for i in range(visual.grid[0] * visual.grid[1], 4)}
    return set()


def check_raw(vals, unit_table):
    """The table was coded from the same encoder geometry: raw bytes == 2*numel per stream."""
    if unit_table is None:
        return
    for k, v in vals.items():
        if (k, 'raw') not in unit_table:
            raise ValueError(f'Rate table has no stream {k}')
        if unit_table[(k, 'raw')] != 2 * v.numel():
            raise ValueError(f'Rate table raw bytes for {k} differ from the training features (encoder geometry)')


@contextmanager
def fp32_head(model, enabled, device_type):
    """Output head in float32 with autocast off; the BF16 weights are not changed."""
    if not enabled:
        yield
        return
    import torch
    import torch.nn.functional as F
    head = model.lm_head
    w = head.weight.detach().float()
    b = head.bias.detach().float() if getattr(head, 'bias', None) is not None else None

    def forward(hidden):
        with torch.autocast(device_type=device_type, enabled=False):
            return F.linear(hidden.float(), w, b)
    head.forward = forward
    try:
        yield
    finally:
        del head.forward


def lm_loss(adapter, inputs, visual, tgt, *, autocast, device_type):
    """Language-model forward with grad on prompt + target, coded streams delivered natively; mean CE on tgt."""
    import torch
    import torch.nn.functional as F
    from deepstack.scoring import _continuation_inputs, _forward_kwargs
    model = adapter.model
    ext = _continuation_inputs(inputs, tuple(tgt))
    kw = _forward_kwargs(model, ext, len(tgt) + 1)
    with adapter.use(visual, ext) as d, torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=bool(autocast)):
        out = model(**kw)
    if d['calls'] != 1:
        raise ValueError('Training forward needs exactly one visual delivery')
    logits = out.logits[0, -(len(tgt) + 1):-1].float()
    return F.cross_entropy(logits, torch.tensor(tgt, device=logits.device))


def code_streams(source, family, config, offsets):
    """Adapter tensors of a frozen encode -> coded streams for ``config`` (device replica, no grad) -> visual."""
    from cloud_prep.measurement_v15.adapter import tensors, wrap
    vals = tensors(source, family)
    lm = levels.resolve_config(config, list(vals), offsets['main_key'], offsets)
    coded = levels.quantize_for_training(vals, lm, inactive=inactive_keys(source, family))
    return vals, wrap(source, coded, family, {'config': config})


class Clock:
    def __init__(self, torch, device, on):
        from cloud_prep.measurement_v19 import common
        self.common, self.torch, self.device, self.on = common, torch, device, on
        self.t, self.last = {}, time.monotonic()

    def __call__(self, name):
        if self.on:
            self.common.sync(self.torch, self.device)
        now = time.monotonic()
        self.t[name] = self.t.get(name, 0.0) + now - self.last
        self.last = now


def sample_loss(adapter, family, row, config, offsets, unit_table, end_id, cfg, torch, device, clock):
    from cloud_prep.measurement_v19 import common
    var = common.variant(row)
    inp, _ = adapter.prepare(row, var, 'a')
    with torch.no_grad():
        source = adapter.encode(inp)
    clock('encode')
    vals, visual = code_streams(source, family, config, offsets)
    check_raw(vals, unit_table)
    clock('quantize')
    if family == 'ibm':
        inp, _ = adapter.prepare(row, var, 'a')
    tgt = target_ids(adapter.processor.tokenizer, answer_text(row, cfg['answer']), end_id)
    loss = lm_loss(adapter, inp, visual, tgt, autocast=cfg['autocast'], device_type=device)
    clock('forward')
    return loss, len(tgt)


# ---- state -------------------------------------------------------------------------------------

def save_state(pm, opt, sch, out, step, next_sample, ident):
    import torch
    tmp = out / 'state' / 'latest.tmp'
    if tmp.exists():
        shutil.rmtree(tmp)
    pm.save_pretrained(str(tmp))
    rng = dict(cpu=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
    torch.save(dict(optimizer=opt.state_dict(), scheduler=sch.state_dict(), rng=rng), tmp / 'trainer.pt')
    lora_utils.write_json(tmp / 'trainer_state.json', dict(step=step, next_sample=next_sample, **ident))
    lora_utils.replace_dir(tmp, out / 'state' / 'latest')


def load_state(pm, opt, sch, st):
    import torch
    from peft.utils import set_peft_model_state_dict
    from safetensors.torch import load_file
    p = Path(st['path'])
    res = set_peft_model_state_dict(pm, load_file(str(p / 'adapter_model.safetensors')))
    if getattr(res, 'unexpected_keys', None):
        raise RuntimeError(f'Resume adapter has unexpected keys: {res.unexpected_keys[:3]}')
    blob = torch.load(p / 'trainer.pt', map_location='cpu', weights_only=False)
    opt.load_state_dict(blob['optimizer'])
    sch.load_state_dict(blob['scheduler'])
    torch.set_rng_state(blob['rng']['cpu'])
    if blob['rng'].get('cuda') is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(blob['rng']['cuda'])


def truncate_jsonl(path, keep):
    path = Path(path)
    if not path.exists():
        return
    lines = [l for l in path.read_text().splitlines() if l.strip()]
    kept = []
    for l in lines:
        try:
            r = json.loads(l)
        except json.JSONDecodeError:
            continue                     # partial last line of an interrupted run
        if keep(r):
            kept.append(l)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(''.join(l + '\n' for l in kept))
    tmp.replace(path)


def meta_fields(*, model, family, arm, cfg, ident, manifest, n_samples, total, table, rate_table, offsets,
                offsets_path, sched_sha, linfo, base, run_id, libs):
    """lora_meta.json fields shared by every checkpoint of a run (checkpoint and created_at are added per save)."""
    hyper = {k: cfg[k] for k in lora_utils.HYPER_KEYS if k in cfg}
    hyper.update(batch_size=1, total_steps=total)
    return dict(model=model, family=family, arm=arm, role=cfg['role'], seed=cfg['seed'], hyperparameters=hyper,
                train_manifest=str(manifest), train_manifest_sha256=ident['train_manifest_sha256'],
                sample_order_sha256=ident['sample_order_sha256'], n_samples=n_samples,
                budget=cfg['budget'], k_abs=cfg['k_abs'], k_weights=cfg.get('k_weights'), q_grid=cfg['q_grid'],
                full_mix=cfg.get('full_mix', 0.0),
                rate_table=dict(path=str(rate_table) if rate_table else None, sha256=ident['rate_table_sha256'],
                                kind=table['kind'] if table else None),
                offsets=dict(path=str(offsets_path), sha256=ident['offsets_sha256'], d=offsets['d']),
                schedule_sha256=sched_sha,
                lora=dict(modules=linfo['modules'], parameters=linfo['parameters'], init_sha256=linfo['init_sha256']),
                base=base, config_sha256=ident['config_sha256'], run_id=run_id, libraries=libs)


def default_out(model, arm, tag=None):
    from cloud_prep.measurement_v19 import common
    base = common.RUN_ROOT / 'train' / model
    return (base / tag if tag else base) / arm


def language_model(model):
    lm = getattr(getattr(model, 'model', None), 'language_model', None)
    if lm is None:
        raise TypeError('Expected model.model.language_model (Qwen3-VL / Granite Vision)')
    return lm


# ---- main --------------------------------------------------------------------------------------

def main(argv=None):
    a = parse_args(argv)
    cfg = schedule.load_config(a.config or schedule.default_config(a.model), schedule.parse_set(a.set))
    if cfg['model'] != a.model:
        raise ValueError(f"config is for {cfg['model']}, not {a.model}")
    from cloud_prep.measurement_v19 import common
    common.require_research()
    out = common.require_under_root(a.out or default_out(a.model, a.arm, a.tag))
    rows = common.load_rows(a.manifest)
    order = schedule.sample_order(rows, cfg['n_samples'], cfg['seed'], cfg['order'])
    offsets, offsets_path = schedule.load_offsets(a.offsets or a.model)
    table = schedule.load_rate_table(a.rate_table) if a.rate_table else None
    sched = schedule.build_schedule(rows, order, a.arm, cfg, table, offsets)
    accum = int(cfg['accum'])
    total = len(order) // accum
    ckpts = lora_utils.checkpoint_steps(cfg, total)
    saves = set(lora_utils.save_steps(total, cfg.get('save_every'), ckpts))

    out.mkdir(parents=True, exist_ok=True)
    state = lora_utils.latest_state(out)
    if a.fresh and any(out.iterdir()):
        raise ValueError(f'--fresh: {out} is not empty')
    sched_sha = schedule.write_jsonl(out / 'schedule.jsonl', sched)
    ident = dict(model=a.model, arm=a.arm, config_sha256=schedule.config_sha256(cfg), schedule_sha256=sched_sha,
                 train_manifest_sha256=schedule.file_sha256(a.manifest),
                 sample_order_sha256=schedule.ids_sha256([rows[j]['id'] for j in order]),
                 rate_table_sha256=schedule.file_sha256(a.rate_table) if a.rate_table else None,
                 offsets_sha256=schedule.file_sha256(offsets_path))
    idp = out / 'run_identity.json'
    if idp.exists():
        old = json.loads(idp.read_text())
        diff = {k for k in ident if old.get(k) != ident[k]}
        if diff:
            raise ValueError(f'{out} holds a run with different {sorted(diff)}; use a new --out/--tag')
    lora_utils.write_json(idp, ident)
    lora_utils.write_json(out / 'config.json', cfg)
    lora_utils.write_json(out / 'schedule_summary.json', schedule.summary(sched))
    print(json.dumps(dict(event='schedule', out=str(out), total_steps=total, checkpoints=ckpts, **schedule.summary(sched))), flush=True)
    if a.dry_run:
        return 0

    import torch
    device = common.device_name(a.device)
    os.environ.setdefault('TMPDIR', str(out / 'tmp'))
    (out / 'tmp').mkdir(exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    started = time.monotonic()
    libs = {}
    for n in ('torch', 'transformers', 'peft', 'Pillow'):
        try:
            libs[n] = importlib.metadata.version(n)
        except importlib.metadata.PackageNotFoundError:
            libs[n] = None
    meta = dict(mode='train', run_id=run_id, out=str(out), manifest=str(a.manifest), rate_table=str(a.rate_table),
                identity=ident, platform=platform.platform(), libraries=libs, device=device, dtype='bfloat16',
                batch_size=1, accum=accum, total_steps=total, checkpoints=ckpts, status='loading',
                resumed_from=state and dict(step=state['step'], next_sample=state['next_sample']))
    metapath = out / f'run-{run_id}.json'
    lora_utils.write_json(metapath, meta)
    step, processed, stopped = 0, 0, False
    clock_tot = {}
    sample_seconds = []
    try:
        t = time.monotonic()
        adapter, family, ameta, snapshot = common.load_model(a.model, device)
        if family not in ('qwen', 'ibm'):
            raise ValueError('The training pilot covers Qwen and Granite')
        model = adapter.model
        pm, linfo = lora_utils.make_lora(model, cfg, cfg['seed'])
        lm = language_model(model)
        if cfg['grad_ckpt']:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
            if not any(getattr(m, 'gradient_checkpointing', False) for m in lm.modules()):
                raise RuntimeError('Gradient checkpointing did not reach the language-model layers')
        model.eval()
        lm.train()                       # LoRA dropout + checkpointing in the LM only; vision/projectors stay eval
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=cfg['lr'], betas=tuple(cfg['adam_betas']), weight_decay=cfg['weight_decay'])
        sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lora_utils.lr_multiplier(
            s, total, cfg['warmup_steps'], cfg['lr_schedule'], cfg.get('min_lr_ratio', 0.0)))
        start = 0
        if state:
            if a.fresh:
                raise ValueError('--fresh with an existing state')
            load_state(pm, opt, sch, state)
            step, start = int(state['step']), int(state['next_sample'])
        truncate_jsonl(out / 'train_log.jsonl', lambda r: r['i'] < start)
        truncate_jsonl(out / 'steps.jsonl', lambda r: r['step'] <= step)
        genconfig, eos = common.make_genconfig(model)
        end_id = eos[0]
        tok = adapter.processor.tokenizer
        common.reset_peak(torch, device)
        meta.update(load_seconds=time.monotonic() - t, adapter=ameta, snapshot=snapshot, family=family, lora=linfo,
                    end_token=dict(id=int(end_id), text=tok.convert_ids_to_tokens(int(end_id))), status='running',
                    start_sample=start, start_step=step)
        lora_utils.write_json(metapath, meta)
        base = dict(repo=common.SNAPSHOT_REVS[a.model][0], revision=common.SNAPSHOT_REVS[a.model][1],
                    snapshot=str(snapshot.get('path') if isinstance(snapshot, dict) else snapshot))
        meta_common = meta_fields(model=a.model, family=family, arm=a.arm, cfg=cfg, ident=ident, manifest=a.manifest,
                                  n_samples=len(order), total=total, table=table, rate_table=a.rate_table,
                                  offsets=offsets, offsets_path=offsets_path, sched_sha=sched_sha, linfo=linfo,
                                  base=base, run_id=run_id, libs=libs)
        losses, t_step = [], time.monotonic()
        with open(out / 'train_log.jsonl', 'a', buffering=1) as log, open(out / 'steps.jsonl', 'a', buffering=1) as slog, \
                fp32_head(model, cfg['fp32_head'], device):
            for i, boundary in lora_utils.iter_samples(len(order), accum, start):
                s, row = sched[i], rows[order[i]]
                ut = table['per_unit'][row['unit_id']] if table else None
                clock = Clock(torch, device, a.timing)
                t0 = time.monotonic()
                loss, ntok = sample_loss(adapter, family, row, s['config'], offsets, ut, end_id, cfg, torch, device, clock)
                (loss / accum).backward()
                lv = float(loss.detach())
                clock('backward')
                if not math.isfinite(lv):
                    raise FloatingPointError(f"Non-finite loss at sample {i} ({row['id']}, {s['config']})")
                sec = time.monotonic() - t0
                sample_seconds.append(sec)
                for k, v in clock.t.items():
                    clock_tot[k] = clock_tot.get(k, 0.0) + v
                log.write(json.dumps({**{k: s[k] for k in LOG_KEYS}, 'loss': lv, 'n_target': ntok, 'step': step,
                                      'seconds': round(sec, 4), **({'timing': clock.t} if a.timing else {})}) + '\n')
                losses.append(lv)
                processed += 1
                if boundary:
                    gn = float(torch.nn.utils.clip_grad_norm_(params, cfg['max_grad_norm']))
                    lr = sch.get_last_lr()[0]
                    opt.step(); sch.step(); opt.zero_grad(set_to_none=True)
                    step = boundary
                    slog.write(json.dumps(dict(step=step, samples_seen=step * accum, lr=lr, loss_mean=sum(losses) / len(losses),
                                               grad_norm=gn, peak_bytes=common.peak_bytes(torch, device),
                                               seconds=round(time.monotonic() - t_step, 3))) + '\n')
                    print(json.dumps(dict(event='step', step=step, total_steps=total, loss_mean=round(sum(losses) / len(losses), 5),
                                          seconds=round(time.monotonic() - t_step, 2))), flush=True)
                    losses, t_step = [], time.monotonic()
                    if step in ckpts:
                        label = ckpts[step]
                        lm_meta = lora_utils.make_lora_meta(
                            checkpoint=dict(label=label, step=step, samples_seen=step * accum, total_steps=total),
                            created_at=datetime.now(timezone.utc).isoformat(), **meta_common)
                        lora_utils.save_checkpoint(pm, out / 'checkpoints' / lora_utils.ckpt_dirname(label, step), lm_meta)
                    if step in saves:
                        save_state(pm, opt, sch, out, step, step * accum, ident)
                    meta.update(step=step, memory=common.mem(torch, device), peak_bytes=common.peak_bytes(torch, device))
                    lora_utils.write_json(metapath, meta)
                if a.max_samples and processed >= a.max_samples:
                    stopped = True
                    break
        meta['status'] = 'stopped_max_samples' if stopped else 'complete'
    except BaseException as exc:
        meta.update(status='failed_or_interrupted', error=repr(exc))
        raise
    finally:
        n = len(sample_seconds)
        timing = dict(samples=n, seconds_per_sample=sum(sample_seconds) / n if n else None,
                      seconds_per_sample_after_first=(sum(sample_seconds[1:]) / (n - 1)) if n > 1 else None,
                      components_per_sample={k: v / n for k, v in clock_tot.items()} if n and a.timing else None,
                      projected_hours_full_pass=(sum(sample_seconds[1:]) / (n - 1)) * len(order) / 3600 if n > 1 else None,
                      synchronized=a.timing)
        try:
            timing['peak_bytes'] = common.peak_bytes(torch, device)
        except Exception:
            timing['peak_bytes'] = None
        lora_utils.write_json(out / 'timing.json', timing)
        meta.update(wall_seconds=time.monotonic() - started, processed=processed, step=step, timing=timing,
                    finished_at=datetime.now(timezone.utc).isoformat())
        lora_utils.write_json(metapath, meta)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
