"""Language-side LoRA (peft), checkpoint layout, lora_meta.json and the resumable step plan. torch/peft lazy.

Eval entry (runner --lora): attach_lora(adapter, family, path) loads a checkpoint dir saved here
(adapter_model.safetensors + adapter_config.json + lora_meta.json) onto adapter.model in place, unmerged,
LoRA weights float32 next to the BF16 base, eval mode, nothing trainable. Returns None.

Run dir (train.py):  <out>/checkpoints/<label>-step<NNNNN>/   eval checkpoints (label short | long)
                     <out>/state/latest/                      resumable state (adapter, trainer.pt, trainer_state.json)
"""
from __future__ import annotations

import json
from pathlib import Path
import shutil

DEFAULT_LM_PREFIX = 'model.language_model.'
EXCLUDE = ('visual', 'vision', 'projector', 'merger', 'lm_head', 'embed', 'qformer')
META_SCHEMA = 'v19-train-lora-1'
LORA_META_KEYS = ('schema', 'model', 'family', 'arm', 'role', 'checkpoint', 'seed', 'hyperparameters',
                  'train_manifest', 'train_manifest_sha256', 'sample_order_sha256', 'n_samples', 'budget',
                  'k_abs', 'k_weights', 'q_grid', 'full_mix', 'rate_table', 'offsets', 'schedule_sha256',
                  'lora', 'base', 'config_sha256', 'run_id', 'created_at', 'libraries')
HYPER_KEYS = ('lr', 'lr_schedule', 'min_lr_ratio', 'warmup_steps', 'weight_decay', 'adam_betas', 'max_grad_norm',
              'accum', 'lora_rank', 'lora_alpha', 'lora_dropout', 'lora_targets', 'lm_prefix', 'grad_ckpt',
              'autocast', 'fp32_head', 'answer', 'order', 'ckpt_short', 'ckpt_long', 'save_every')


# ---- target modules ----------------------------------------------------------------------------

def lora_targets(names, targets, prefix=DEFAULT_LM_PREFIX):
    """Full module names to wrap: under the language-model prefix, last component in ``targets``, and no
    vision/projector/merger/head/embedding component (Granite's SigLIP tower also has q_proj/k_proj/v_proj)."""
    targets = set(targets)
    out = []
    for n in names:
        parts = n.split('.')
        if not n.startswith(prefix) or parts[-1] not in targets:
            continue
        if any(x in p for p in parts for x in EXCLUDE):
            continue
        out.append(n)
    if not out:
        raise ValueError(f'No LoRA targets under {prefix!r} for {sorted(targets)}')
    return out


def linear_names(model):
    import torch
    return [n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)]


def make_lora(model, cfg, seed):
    """Freeze ``model``, inject LoRA (float32 adapters) into the language model in place; same seed -> same init.
    Returns (peft_model, info). Call model(...) directly: the PeftModel wrapper is kept only for saving."""
    import torch
    from peft import LoraConfig, get_peft_model
    model.requires_grad_(False)
    names = lora_targets(linear_names(model), cfg['lora_targets'], cfg.get('lm_prefix', DEFAULT_LM_PREFIX))
    conf = LoraConfig(r=int(cfg['lora_rank']), lora_alpha=cfg['lora_alpha'], lora_dropout=cfg['lora_dropout'],
                      target_modules=names, bias='none', task_type=None)
    torch.manual_seed(int(seed))
    pm = get_peft_model(model, conf, autocast_adapter_dtype=True)
    info = check_lora(model)
    if info['modules'] != len(names):
        raise RuntimeError(f"Injected {info['modules']} LoRA modules for {len(names)} targets")
    return pm, {**info, 'targets': len(names), 'init_sha256': lora_sha256(model)}


def lora_modules(model):
    return [(n, m) for n, m in model.named_modules() if hasattr(m, 'lora_A') and len(getattr(m, 'lora_A', {}))]


def check_lora(model, prefix=DEFAULT_LM_PREFIX):
    """LoRA only on language-model linears, float32 adapters; returns counts."""
    mods = lora_modules(model)
    bad = [n for n, _ in mods if not n.startswith(prefix) or any(x in n for x in EXCLUDE)]
    if bad:
        raise RuntimeError(f'LoRA outside the language model: {bad[:3]}')
    params = [(n, p) for n, p in model.named_parameters() if 'lora_' in n]
    if any(p.dtype.is_floating_point and str(p.dtype) != 'torch.float32' for _, p in params):
        raise RuntimeError('LoRA weights must be float32')
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    if any('lora_' not in n for n in trainable):
        raise RuntimeError('Non-LoRA parameters are trainable')
    return dict(modules=len(mods), parameters=sum(p.numel() for _, p in params), trainable=len(trainable))


def lora_sha256(model):
    import hashlib
    h = hashlib.sha256()
    for n, p in sorted((n, p) for n, p in model.named_parameters() if 'lora_' in n):
        h.update(n.encode())
        h.update(p.detach().float().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


# ---- meta and checkpoints ----------------------------------------------------------------------

def make_lora_meta(**fields):
    meta = dict(schema=META_SCHEMA, **fields)
    missing = [k for k in LORA_META_KEYS if k not in meta]
    if missing:
        raise ValueError(f'lora_meta lacks {missing}')
    ck = meta['checkpoint']
    if not {'label', 'step', 'samples_seen'} <= set(ck):
        raise ValueError('checkpoint needs label, step, samples_seen')
    return meta


def read_lora_meta(path):
    return json.loads((Path(path) / 'lora_meta.json').read_text())


def ckpt_dirname(label, step):
    return f'{label}-step{int(step):05d}'


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=1, default=str))
    tmp.replace(path)


def replace_dir(tmp, final):
    """Move a fully written tmp dir into place; the previous version is removed only afterwards."""
    tmp, final = Path(tmp), Path(final)
    old = final.with_name(final.name + '.old')
    if old.exists():
        shutil.rmtree(old)
    if final.exists():
        final.rename(old)
    tmp.rename(final)
    if old.exists():
        shutil.rmtree(old)


def save_checkpoint(peft_model, path, meta):
    """Eval checkpoint: adapter weights + adapter_config.json + lora_meta.json, written atomically."""
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    if tmp.exists():
        shutil.rmtree(tmp)
    peft_model.save_pretrained(str(tmp))
    write_json(tmp / 'lora_meta.json', meta)
    replace_dir(tmp, path)
    return path


def attach_lora(adapter, family, path):
    """Runner --lora: load the checkpoint onto adapter.model in place (not merged), eval mode, frozen."""
    from peft import PeftModel
    path = Path(path)
    meta = read_lora_meta(path)
    if meta.get('family') not in (None, family):
        raise ValueError(f"LoRA checkpoint is for family {meta.get('family')}, model family is {family}")
    model = adapter.model
    if getattr(model, '_v19_lora_path', None) or lora_modules(model):
        raise RuntimeError('A LoRA adapter is already attached')
    model.requires_grad_(False)
    PeftModel.from_pretrained(model, str(path), is_trainable=False, autocast_adapter_dtype=True)
    info = check_lora(model, meta.get('hyperparameters', {}).get('lm_prefix', DEFAULT_LM_PREFIX))
    expect = meta.get('lora', {}).get('modules')
    if expect is not None and info['modules'] != expect:
        raise RuntimeError(f"Attached {info['modules']} LoRA modules, checkpoint recorded {expect}")
    model.eval()
    model._v19_lora_path = str(path)


# ---- step plan and resume ----------------------------------------------------------------------

def checkpoint_steps(cfg, total_steps):
    """{optimizer step: label} for the short and long eval checkpoints."""
    s, l = int(cfg['ckpt_short']), int(cfg['ckpt_long'])
    if not 0 < s < l <= total_steps:
        raise ValueError(f'need 0 < ckpt_short ({s}) < ckpt_long ({l}) <= total steps ({total_steps})')
    return {s: 'short', l: 'long'}


def save_steps(total_steps, save_every, ckpt):
    """Optimizer steps after which the resumable state is written (every save_every, the checkpoints, the end)."""
    out = set(ckpt) | {total_steps}
    if save_every:
        out |= set(range(int(save_every), total_steps + 1, int(save_every)))
    return sorted(out)


def iter_samples(n_samples, accum, start=0):
    """Yield (sample index, optimizer step completed after it or None). The trailing partial accumulation
    group is dropped, so a step always averages exactly ``accum`` samples. start must be a step boundary."""
    if start % accum:
        raise ValueError('resume must start at an accumulation boundary')
    usable = (n_samples // accum) * accum
    for i in range(start, usable):
        yield i, ((i + 1) // accum if (i + 1) % accum == 0 else None)


def latest_state(run_dir):
    """trainer_state.json of <run_dir>/state/latest (None if absent); half-written tmp dirs are ignored."""
    p = Path(run_dir) / 'state' / 'latest' / 'trainer_state.json'
    if not p.exists():
        old = Path(run_dir) / 'state' / 'latest.old' / 'trainer_state.json'
        if not old.exists():
            return None
        p = old                         # crash between the two renames of replace_dir
    st = json.loads(p.read_text())
    st['path'] = str(p.parent)
    return st


def lr_multiplier(step, total, warmup, schedule='cosine', min_ratio=0.0):
    import math
    if warmup and step < warmup:
        return (step + 1) / warmup
    if schedule == 'constant':
        return 1.0
    t = (step - warmup) / max(1, total - warmup)
    t = min(max(t, 0.0), 1.0)
    if schedule == 'linear':
        return min_ratio + (1 - min_ratio) * (1 - t)
    if schedule == 'cosine':
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * t))
    raise ValueError('lr_schedule must be cosine, linear or constant')
