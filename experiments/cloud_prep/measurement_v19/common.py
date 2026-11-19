"""Shared, model-side helpers for v19 stream coding (model envs only; no nncodec import).

Model loading, prompts, greedy generation and scoring reuse the accepted v15/v16 code:
  - load: cloud_prep.measurement_v16.runner.load_adapter (local MPS Qwen/Granite/DeepStack bridges)
  - streams: cloud_prep.measurement_v15.adapter.tensors / wrap / geometry
  - prompt: measurement_v15.runner.make_variant (freeform: query + newline + answer_instruction)
  - generation: measurement_v15.runner.generate (native greedy), fp32 output head, 64 new tokens
  - scoring: scripts/score_followup_benchmark_v15.py metric_values / invalid_reasons
    (gqa_case_punct_normalized, invalid_as_zero), cluster-weighted mean as in the paper.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import sys

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

RUN_NAME = 'stream-coding-v19-20261110'


def resolve_paths(env=None, system=None):
    import platform
    env = os.environ if env is None else env
    system = system or platform.system()
    root = Path(env.get('DEEPSTACK_V19_ROOT') or env.get('DS_DATA_ROOT') or 'work').resolve()
    hf = Path(env.get('DEEPSTACK_V19_HF_ROOT') or env.get('HF_HOME') or root / 'huggingface')
    return dict(root=root, run_root=Path(env.get('DEEPSTACK_V19_RUN_ROOT') or root / 'runs' / RUN_NAME),
                data_root=Path(env.get('DEEPSTACK_V19_DATA_ROOT') or root / 'data'), hf_root=hf.resolve(),
                device=env.get('DEEPSTACK_V19_DEVICE') or ('mps' if system == 'Darwin' else 'cuda'))


_P = resolve_paths()
ROOT, RUN_ROOT, DATA_ROOT, HF_ROOT, DEFAULT_DEVICE = _P['root'], _P['run_root'], _P['data_root'], _P['hf_root'], _P['device']
BENCH = DATA_ROOT / 'followup-v15-20261029/benchmarks'
CAL_MANIFEST = BENCH / 'calibration-v14reuse/gqa-attribute/manifest.jsonl'
EVAL_MANIFEST = RUN_ROOT / 'materials/attempt-1/materials/manifest.jsonl'
CLOUD_CC = RUN_ROOT / 'cloud-cc/attempt-2/observations.jsonl'
SNAPSHOT_REVS = {
    'qwen8b': ('Qwen/Qwen3-VL-8B-Instruct', '0c351dd01ed87e9c1b53cbc748cba10e6187ff3b'),
    'deepstack7b': ('menglc/deepstack-l-vicuna-7b', '76f7d20cd835dfb59652be061bf8e3732190781f'),
    'ibm4b': ('ibm-granite/granite-vision-4.1-4b', '37d591f06319e8f1638b5adcf58bdf50e0f84f7a'),
    'qwen32b': ('Qwen/Qwen3-VL-32B-Instruct', '0cfaf48183f594c314753d30a4c4974bc75f3ccb'),
}


def snapshot_path(model_key, hf_root=None):
    repo, rev = SNAPSHOT_REVS[model_key]
    return Path(hf_root or HF_ROOT) / 'hub' / ('models--' + repo.replace('/', '--')) / 'snapshots' / rev


SNAPSHOTS = {k: snapshot_path(k) for k in SNAPSHOT_REVS}
TASK_MANIFESTS = {
    'gqa-attribute': (CAL_MANIFEST, EVAL_MANIFEST),
    'gqa-spatial': (BENCH / 'calibration-v14reuse/gqa-spatial/manifest.jsonl',
                    RUN_ROOT / 'materials-gqa-spatial/attempt-1/materials/manifest.jsonl'),
    'docvqa': (BENCH / 'calibration-v14reuse/docvqa/manifest.jsonl',
               RUN_ROOT / 'materials-docvqa/attempt-1/materials/manifest.jsonl'),
}


def cloud_cc_path(model, task):
    import glob
    name = ('cloud-cc' if model == 'qwen8b' else f'cloud-cc-{model}') if task == 'gqa-attribute' else f'cloud-cc-{model}-{task}'
    found = sorted(glob.glob(str(RUN_ROOT / f'{name}/attempt-*/observations.jsonl')))
    return Path(found[-1]) if found else None


CLOUD_CC_BY_MODEL = {'qwen8b': CLOUD_CC,
                     'deepstack7b': RUN_ROOT / 'cloud-cc-deepstack7b/attempt-2/observations.jsonl',
                     'ibm4b': RUN_ROOT / 'cloud-cc-ibm4b/attempt-2/observations.jsonl'}
GENERATE_TOKENS = 64
METRIC = 'gqa_case_punct_normalized'
FAMILY_MAIN_KEY = {'qwen': 'main', 'deepstack': 'main', 'ibm': 'layer0'}


def require_research():
    """Require an existing output root before model execution."""
    if not ROOT.is_dir():
        raise RuntimeError(f'Storage root {ROOT} does not exist')


def require_under_root(path, root=None):
    """Resolved path; raises unless it lies under the storage root (outputs, temp files)."""
    p = Path(path).resolve()
    r = Path(root or ROOT).resolve()
    if not p.is_relative_to(r):
        raise ValueError(f'{path} must be under {r}')
    return p


def device_name(device=None):
    d = device or DEFAULT_DEVICE
    if d not in ('mps', 'cuda'):
        raise ValueError('device must be mps or cuda, got ' + repr(d))
    return d


def sync(torch, device=None):
    (torch.cuda.synchronize if device_name(device) == 'cuda' else torch.mps.synchronize)()


def mem(torch, device=None):
    if device_name(device) == 'cuda':
        return dict(cuda_allocated=torch.cuda.memory_allocated(), cuda_reserved=torch.cuda.memory_reserved(),
                    cuda_peak_allocated=torch.cuda.max_memory_allocated(), cuda_peak_reserved=torch.cuda.max_memory_reserved())
    return dict(mps_current=torch.mps.current_allocated_memory(), mps_driver=torch.mps.driver_allocated_memory())


def peak_bytes(torch, device=None):
    """CUDA: max reserved since the last reset_peak. MPS: current driver allocation (callers keep the max)."""
    if device_name(device) == 'cuda':
        return torch.cuda.max_memory_reserved()
    return torch.mps.driver_allocated_memory()


def reset_peak(torch, device=None):
    if device_name(device) == 'cuda':
        torch.cuda.reset_peak_memory_stats()


def load_rows(path):
    path = Path(path)
    rows = []
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            r['source_images'] = {k: str((path.parent / v).resolve()) for k, v in r['source_images'].items()}
            rows.append(r)
    return rows


def unit_hash(unit_id, salt='v19'):
    return hashlib.sha256(f'{salt}|{unit_id}'.encode()).hexdigest()


def order_units(rows, salt='v19'):
    """Rows grouped by unit, units in deterministic hash order; questions in file order."""
    units = {}
    for r in rows:
        units.setdefault(r['unit_id'], []).append(r)
    return [(u, units[u]) for u in sorted(units, key=lambda u: unit_hash(u, salt))]


def take_units(rows, min_questions=None, max_units=None, salt='v19'):
    out = []
    for i, (u, qs) in enumerate(order_units(rows, salt)):
        if max_units is not None and i >= max_units:
            break
        if min_questions is not None and sum(len(x) for _, x in out) >= min_questions:
            break
        out.append((u, qs))
    return out


def load_model(model_key, device=None, dtype='bfloat16'):
    """(adapter, family, meta, snapshot) through the accepted v16 loader.

    mps: v16 local MPS loaders. cuda: v16 delegates to the accepted v15 CUDA loaders (BF16 only,
    device 'cuda' i.e. the first visible GPU; select the card with CUDA_VISIBLE_DEVICES).
    """
    require_research()
    device = device_name(device)
    os.environ.setdefault('HF_DEACTIVATE_ASYNC_LOAD', '1')
    from types import SimpleNamespace
    from cloud_prep.measurement_v15.runner import model_spec
    from cloud_prep.measurement_v16.runner import load_adapter
    spec = model_spec(model_key)
    a = SimpleNamespace(snapshot=SNAPSHOTS[model_key], device=device, dtype=dtype,
                        cache_root=HF_ROOT, local_files_only=True)
    adapter, family, meta, snapshot = load_adapter(a, spec, {'qwen_pixel_limits': None})
    return adapter, family, meta, snapshot


def make_genconfig(model):
    from copy import deepcopy
    from cloud_prep.measurement_v8.generation import resolve_greedy_config
    g = deepcopy(model.generation_config)
    g.max_new_tokens = GENERATE_TOKENS
    g.do_sample = False; g.num_beams = 1; g.num_return_sequences = 1; g.use_cache = True
    g = resolve_greedy_config(model, g)
    eos = g.eos_token_id
    eos = [eos] if isinstance(eos, int) else list(eos or [])
    if not eos:
        raise ValueError('Explicit EOS required')
    return g, eos


def variant(row):
    from cloud_prep.measurement_v15.runner import make_variant
    return make_variant(row, {'response_format': 'freeform', 'mapping': 'none'})


def generate_text(adapter, family, inputs, genconfig, eos, visual=None):
    """Greedy generation; with ``visual`` the cached/transformed streams are delivered natively."""
    from contextlib import nullcontext
    from cloud_prep.measurement_v15.runner import generate
    from scripts.prepare_readout_controls import generation_stop
    ctx = adapter.use(visual, inputs) if visual is not None else nullcontext({'calls': None})
    with ctx as d:
        ids = generate(adapter.model, inputs, genconfig, family, None).detach().cpu().tolist()
    if visual is not None and d['calls'] != 1:
        raise ValueError('Generation needs exactly one visual prefill')
    tok = adapter.processor.tokenizer
    return dict(generated_text=tok.decode(ids, skip_special_tokens=True),
                generated_token_ids=ids, **generation_stop(ids, eos, GENERATE_TOKENS))


def prefill_logits(adapter, inputs, visual=None):
    from contextlib import nullcontext
    from deepstack.scoring import _forward_kwargs
    model = adapter.model
    ctx = adapter.use(visual, inputs) if visual is not None else nullcontext()
    with ctx:
        out = model(**_forward_kwargs(model, inputs, 1))
    return out.logits[0, -1].float().cpu()


def metric_label(bench):
    """Label of the per-row 'correct' metric for one benchmark (the same strings score() writes per row)."""
    return ('docvqa_anls_organizer_framework' if bench == 'docvqa' else METRIC) + '|invalid_as_zero'


def score(row, result):
    from scripts.score_followup_benchmark_v15 import metric_values, invalid_reasons
    record = {'result': result}
    invalid = invalid_reasons(record)
    text = result.get('generated_text')
    bench = row['benchmark']['name']
    if bench == 'docvqa':
        # Paper default: organizer-framework ANLS; strict-paper-cutoff ANLS kept alongside.
        if not isinstance(text, str) or invalid:
            m = dict(docvqa_anls_organizer_framework=0.0, docvqa_anls_strict_paper_cutoff=0.0)
        else:
            m = metric_values(bench, text, row['reference_answers'])
        return dict(correct=m['docvqa_anls_organizer_framework'], anls_strict=m['docvqa_anls_strict_paper_cutoff'],
                    invalid_reasons=invalid, metric='docvqa_anls_organizer_framework|invalid_as_zero')
    if not isinstance(text, str) or invalid:
        value = 0.0
    else:
        value = metric_values(bench, text, row['reference_answers'])[METRIC]
    return dict(correct=value, invalid_reasons=invalid, metric=METRIC + '|invalid_as_zero')


def wilson(k, n, z=1.96):
    if n == 0:
        return (None, None)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def cluster_stats(pairs, n_boot=2000, seed=20261110):
    """pairs: [(cluster_id, value)]. Cluster-weighted mean, normal CI (as scorer) and cluster bootstrap CI."""
    by = {}
    for c, v in pairs:
        by.setdefault(c, []).append(v)
    means = [statistics.mean(v) for v in by.values()]
    n = len(means)
    m = statistics.mean(means) if n else None
    se = statistics.stdev(means) / math.sqrt(n) if n > 1 else None
    rng = random.Random(seed)
    boots = sorted(statistics.mean(rng.choices(means, k=n)) for _ in range(n_boot)) if n > 1 else []
    lo = boots[int(0.025 * n_boot)] if boots else None
    hi = boots[int(0.975 * n_boot) - 1] if boots else None
    qw = statistics.mean(v for _, v in pairs) if pairs else None
    return dict(n_questions=len(pairs), n_clusters=n, acc_cluster=m, se_cluster=se,
                ci_boot_low=lo, ci_boot_high=hi, acc_question=qw)


def tensor_sha(t):
    import torch
    return hashlib.sha256(t.detach().to('cpu').contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=str))
    tmp.replace(path)
