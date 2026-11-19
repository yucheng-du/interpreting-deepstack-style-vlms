"""Per-stream coding levels: C (untouched), q<QP> (NNC uniform quantization), Z (zeros), M (fixed mean).

The quantizer replicates NNCodec 2.1.3 deepCABAC URQ with dq_flag=0 and lambda_scale=0
(extensions/deepCABAC/source/Lib/CommonLib/Quant.cpp, quantizeURQ / deQuantize; bindings.cpp
quantLayer / dequantLayer):

    k = 1 << qp_density;  step = float32((k + (qp & (k-1))) * 2**((qp >> qp_density) - qp_density))
    level = round_half_away_from_zero(float32(float32(1)/step) * x)    # float32 product
    rec   = step * float32(level)                                       # float32 product

Config names (main part | later part):
  shared   'A|B'            A, B in C, Z, M, q<int>; every later entry gets B (original v19 names)
  offset   'q<q>|k<k>'      main QP q, later entry i QP q + d_i + k (d from the model's offsets table);
                            k is 'k0' or a signed nonzero integer ('k+2', 'k-4'); k0 is N
  explicit 'A|L1,L2,...'    one level per later entry, in later_keys() order (ascending trailing integer)

Model-agnostic: levels act on the adapter's ``tensors`` dict ({main_key: T, later_i: T}).
Torch is imported lazily so that the pure helpers stay importable without it.
"""
from __future__ import annotations

import re

QP_DENSITY = 2
LEVEL_RE = re.compile(r'^(C|Z|M|q(-?\d+))$')
K_RE = re.compile(r'^k(0|[+-][1-9]\d*)$')


def step_size(qp, qp_density=QP_DENSITY):
    k = 1 << qp_density
    mul = k + (qp & (k - 1))          # python & / >> on negative ints = two's complement, as in C++
    shift = qp >> qp_density
    return mul * 2.0 ** (shift - qp_density)


def parse_level(level):
    m = LEVEL_RE.match(level)
    if not m:
        raise ValueError('Unknown level ' + repr(level))
    return ('q', int(m.group(2))) if m.group(2) is not None else (level, None)


def config_name(main_level, later_level):
    return f'{main_level}|{later_level}'


def offset_name(q, k):
    return f'q{int(q)}|k{int(k):+d}' if k else f'q{int(q)}|k0'


def parse_config(name):
    """('shared', main, later) | ('offset', q, k) | ('explicit', main, [later levels])."""
    if not isinstance(name, str) or name.count('|') != 1:
        raise ValueError('Config needs exactly one "|": ' + repr(name))
    a, b = name.split('|')
    kind, qp = parse_level(a)
    if b.startswith('k'):
        m = K_RE.match(b)
        if not m:
            raise ValueError('Offset part must be k0 or k+N / k-N (N>0): ' + repr(name))
        if kind != 'q':
            raise ValueError('Offset configs need a q<int> main level: ' + repr(name))
        return ('offset', qp, int(m.group(1)))
    if ',' in b:
        parts = b.split(',')
        for p in parts:
            parse_level(p)
        return ('explicit', a, parts)
    parse_level(b)
    return ('shared', a, b)


def split_config(name):
    """(main part, later part) as written; validates any config form."""
    parse_config(name)
    a, b = name.split('|')
    return a, b


def later_keys(keys, main_key):
    """Non-main keys ordered by trailing integer (later0..3; Granite layer3..layer21)."""
    def num(k):
        m = re.search(r'(\d+)$', k)
        return (int(m.group(1)) if m else -1, k)
    return sorted((k for k in keys if k != main_key), key=num)


def offset_map(offsets):
    """{later key: int d} from a plain dict or an offsets table (offset_axis.offsets_table)."""
    if offsets is None:
        return None
    if isinstance(offsets, dict) and 'd' in offsets and isinstance(offsets['d'], dict):
        offsets = offsets['d']
    return {k: int(v) for k, v in offsets.items()}


def resolve_config(name, keys, main_key, offsets=None):
    """Config name -> {key: level} for every key in ``keys`` (inactive keys included; callers skip them)."""
    keys = list(keys)
    if main_key not in keys:
        raise ValueError(f'main key {main_key!r} not in keys')
    lk = later_keys(keys, main_key)
    form, x, y = parse_config(name)
    if form == 'shared':
        out = {k: (x if k == main_key else y) for k in keys}
    elif form == 'explicit':
        if len(y) != len(lk):
            raise ValueError(f'{name!r} gives {len(y)} later levels for {len(lk)} later keys {lk}')
        out = {main_key: x, **dict(zip(lk, y))}
    else:
        d = offset_map(offsets)
        if d is None:
            raise ValueError(f'Offset config {name!r} needs an offsets table')
        if set(d) != set(lk):
            raise ValueError(f'Offsets keys {sorted(d)} do not match later keys {lk}')
        out = {main_key: f'q{x}', **{k: f'q{x + d[k] + y}' for k in lk}}
    return {k: out[k] for k in keys}


def config_qps(names, keys, main_key, offsets=None):
    """Sorted set of QPs the configs need on these keys (byte table / rate audit)."""
    need = set()
    for n in names:
        for lv in resolve_config(n, keys, main_key, offsets).values():
            kind, qp = parse_level(lv)
            if kind == 'q':
                need.add(qp)
    return sorted(need)


def quantize(x, qp, qp_density=QP_DENSITY):
    """Torch replica of NNCodec encode->decode for one tensor. Returns float32 on x's device.

    Exact IEEE float32 operations only; ties (fraction exactly .5) are rounded away from zero
    as C ``round`` does, unlike torch.round (half to even).
    """
    import torch
    step = torch.tensor(step_size(qp, qp_density), dtype=torch.float32)
    scale = torch.tensor(1.0, dtype=torch.float32) / step       # float32 reciprocal, as C++
    p = x.to(torch.float32) * scale.to(x.device)
    r = torch.round(p)
    t = torch.trunc(p)
    tie = (p - t).abs() == 0.5
    r = torch.where(tie, t + torch.sign(p), r)
    # level 0 decodes to +0.0 (int level * step); "+ 0.0" turns torch's -0.0 into +0.0 for bit equality.
    return r * step.to(x.device) + 0.0


def quantize_levels(x, qp, qp_density=QP_DENSITY):
    """Integer levels (int64) of the replica, for zero-fraction statistics."""
    import torch
    step = torch.tensor(step_size(qp, qp_density), dtype=torch.float32)
    p = x.to(torch.float32) * (torch.tensor(1.0, dtype=torch.float32) / step).to(x.device)
    r = torch.round(p); t = torch.trunc(p)
    return torch.where((p - t).abs() == 0.5, t + torch.sign(p), r).to(torch.int64)


def transform(value, level, *, mean=None, active=True, on_device=False):
    """Apply one level to one tensor; returns a tensor of value's dtype/device.

    Quantization runs in float32 (exact replica) and is cast back to the model dtype: on CPU by
    default, on value's own device with ``on_device=True`` (same float32 ops; check with
    device_replica_check). ``active=False`` marks a structurally inactive (zero-padded) slot: it is
    returned unchanged and never counts as payload.
    """
    import torch
    kind, qp = parse_level(level)
    if not active or kind == 'C':
        return value
    if kind == 'Z':
        return torch.zeros_like(value)
    if kind == 'M':
        if mean is None:
            raise ValueError('Level M needs a calibration mean vector')
        m = mean.to(device=value.device, dtype=value.dtype)
        if m.ndim != 1 or m.shape[0] != value.shape[-1]:
            raise ValueError('Mean vector must be hidden-size')
        return m.expand_as(value).clone()
    if on_device:
        return quantize(value.detach(), qp).to(dtype=value.dtype)
    return quantize(value.detach().to('cpu'), qp).to(device=value.device, dtype=value.dtype)


def apply_levels(values, level_map, *, means=None, inactive=(), on_device=False):
    """values: adapter tensors dict; level_map: {key: level} (resolve_config)."""
    means = means or {}
    missing = set(values) - set(level_map)
    if missing:
        raise ValueError(f'No level for keys {sorted(missing)}')
    return {key: transform(value, level_map[key], mean=means.get(key), active=key not in inactive, on_device=on_device)
            for key, value in values.items()}


def apply_config(values, main_key, main_level, later_level=None, *, means=None, inactive=(), offsets=None,
                 on_device=False):
    """Old call: (main_level, later_level). New: later_level=None and main_level a full config name."""
    name = main_level if later_level is None else config_name(main_level, later_level)
    return apply_levels(values, resolve_config(name, list(values), main_key, offsets), means=means,
                        inactive=inactive, on_device=on_device)


def quantize_for_training(values, level_map, *, means=None, inactive=(), dtype=None):
    """Training forward: {key: tensor already on device} -> coded tensors on the same device, no grad.

    Same float32 replica as decoding, computed on the tensors' device, cast once to ``dtype``
    (default: each input's dtype, e.g. BF16, as the inference path delivers it; float32 keeps the
    decoded values exactly).
    """
    import torch
    missing = set(values) - set(level_map)
    if missing:
        raise ValueError(f'No level for keys {sorted(missing)}')
    means = means or {}
    out = {}
    with torch.no_grad():
        for k, v in values.items():
            v = v.detach()
            kind, qp = parse_level(level_map[k])
            if kind == 'q' and k not in inactive:
                y = quantize(v, qp)
            else:
                y = transform(v, level_map[k], mean=means.get(k), active=k not in inactive)
            out[k] = y.to(dtype or v.dtype)
    return out


def device_replica_check(x, qps, device=None):
    """Bitwise compare quantize() on ``device`` (default: x's device) with the CPU replica, per QP."""
    import torch
    dev = torch.device(device) if device is not None else x.device
    xc = x.detach().to('cpu')
    xd = x.detach().to(dev)
    per = {}
    for qp in qps:
        a = quantize(xc, qp).contiguous().view(torch.int32)
        b = quantize(xd, qp).to('cpu').contiguous().view(torch.int32)
        n = int((a != b).sum())
        per[int(qp)] = dict(equal=n == 0, mismatches=n)
    return dict(device=str(dev), numel=int(x.numel()), all_equal=all(v['equal'] for v in per.values()), per_qp=per)


def levels_bytes(byte_table, level_map, main_key, *, inactive=(), detail=False):
    """Payload bytes for {key: level}. See config_bytes for the byte table."""
    parts = {'main': 0, 'later': 0}
    per_key = {}
    for key, level in level_map.items():
        if key in inactive:
            continue
        kind, qp = parse_level(level)
        if kind in ('C', 'q'):
            idx = (key, 'raw') if kind == 'C' else (key, qp)
            if idx not in byte_table:
                raise ValueError(f'Byte table lacks {key} at ' + ('raw' if kind == 'C' else f'QP {qp}'))
            b = byte_table[idx]
        else:
            b = 0
        per_key[key] = b
        parts['main' if key == main_key else 'later'] += b
    return {**parts, 'per_key': per_key} if detail else parts


def config_bytes(byte_table, main_key, main_level, later_level, keys, *, inactive=(), offsets=None, detail=False):
    """Payload bytes of one image for a config.

    byte_table[(key, qp)] -> NNCodec bitstream bytes (with headers). C counts raw BF16
    (2 bytes/element, from byte_table[(key,'raw')]); Z and M carry no per-image payload.
    Inactive (zero-padded) slots never count. later_level=None: main_level is a full config name.
    detail=True adds 'per_key' (active keys only).
    """
    name = main_level if later_level is None else config_name(main_level, later_level)
    return levels_bytes(byte_table, resolve_config(name, keys, main_key, offsets), main_key,
                        inactive=inactive, detail=detail)
