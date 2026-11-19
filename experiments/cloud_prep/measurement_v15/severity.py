"""Physical degradation settings for v15 severity curves (CPU, Pillow only).

A setting is one named physical operation, shared by every row and model of a
task family: e.g. ``desat-c0.2`` keeps 20% of chroma, ``blurrel-period0.25``
blurs with sigma = 0.25 stripe periods, ``lowres-long160`` re-renders the image
at an effective 160-pixel long side. Defining a setting relative to a stimulus
or image scale makes it comparable across source images of different sizes,
which a raw pixel radius is not. Every model receives the same physical image
on the same canvas, which helps comparability, but each native processor still
resizes, tiles or pads it differently and backends differ, so the realized
damage is model-specific: native clean/joint-degraded ability and the actual
processor geometry must be calibrated and reported on the cloud.

The degraded file keeps the clean canvas, is written once per source image and
setting, and is named by every paired cell that uses it (main-degraded,
later-degraded and joint-degraded cells share one realization). Per-row specs
record the realized reference scale and pixel parameter; the file hash in the
row metadata is the identity the runner checks.
"""
from __future__ import annotations
import hashlib
import json
import math

GENERATOR = 'deepstack-v15-severity-1'
FAMILIES = {
    'desaturate': dict(short='desat', references={None: 'c'},
        meaning='keep fraction c of chroma; ITU-R 601 luma preserved (c=1 clean, c=0 gray)', more_severe='lower'),
    'blur_rel': dict(short='blurrel', references={'stripe_period_px': 'period', 'dot_separation_px': 'dotsep', 'long_side_px': 'long'},
        meaning='Pillow Gaussian blur, sigma_px = value x reference_px', more_severe='higher'),
    'lowres': dict(short='lowres', references={'font_px': 'font', 'long_side_px': 'long'},
        meaning='area (BOX) downsample so reference_px maps to value px, bicubic upsample back to the same canvas', more_severe='lower'),
    'contrast': dict(short='contrast', references={None: 'f'},
        meaning='Pillow ImageEnhance.Contrast factor (v14 anchor family)', more_severe='lower'),
}

# Editable starting grids. ``candidates`` serve calibration on separate units;
# ``start`` is the initial three-setting evaluation grid, to be replaced by the
# calibration choice (clean/joint-degraded ability only) without regenerating
# clean or counterfactual images.
DEFAULT_GRIDS = {
    'color': dict(family='desaturate', reference_kind=None,
        candidates=[0.4, 0.2, 0.1, 0.05, 0.025, 0.0], start=[0.2, 0.1, 0.05]),
    'stripe_orientation': dict(family='blur_rel', reference_kind='stripe_period_px',
        candidates=[0.1, 0.18, 0.25, 0.32, 0.4, 0.5], start=[0.18, 0.25, 0.32]),
    'relative_position': dict(family='blur_rel', reference_kind='dot_separation_px',
        candidates=[0.1, 0.2, 0.3, 0.45, 0.6, 0.8], start=[0.2, 0.3, 0.45]),
    'text_reading': dict(family='lowres', reference_kind='font_px',
        candidates=[16, 12, 10, 8, 6, 5], start=[10, 8, 6]),
    'gqa_attribute': dict(family='lowres', reference_kind='long_side_px',
        candidates=[224, 160, 112, 80, 56, 40], start=[160, 80, 40]),
    'gqa_spatial': dict(family='lowres', reference_kind='long_side_px',
        candidates=[224, 160, 112, 80, 56, 40], start=[160, 80, 40]),
    'docvqa_document_qa': dict(family='lowres', reference_kind='long_side_px',
        candidates=[1024, 720, 512, 360, 256, 180], start=[720, 360, 180]),
}


def number(value):
    """Canonical number text: 0.2 -> '0.2', 160.0 -> '160'."""
    value = float(value)
    if not math.isfinite(value): raise ValueError('Finite severity value required')
    return format(value, 'g')


def setting(family, value, reference_kind=None):
    if family not in FAMILIES: raise ValueError('Unknown degradation family '+repr(family))
    refs = FAMILIES[family]['references']
    if reference_kind not in refs:
        raise ValueError(f'{family} supports reference kinds {sorted(map(str, refs))}, not {reference_kind!r}')
    value = float(value)
    if family == 'desaturate' and not 0 <= value <= 1: raise ValueError('chroma must be in [0,1]')
    if family == 'contrast' and not 0 <= value <= 1: raise ValueError('contrast factor must be in [0,1]')
    if family == 'blur_rel' and not value > 0: raise ValueError('relative blur must be positive')
    if family == 'lowres' and not value >= 1: raise ValueError('effective resolution must be >= 1 px')
    label = f"{FAMILIES[family]['short']}-{refs[reference_kind]}{number(value)}"
    return dict(label=label, family=family, value=value, reference_kind=reference_kind)


def parse_label(label):
    family_short, _, rest = label.partition('-')
    for family, info in FAMILIES.items():
        if info['short'] != family_short: continue
        for kind, prefix in sorted(info['references'].items(), key=lambda kv: -len(kv[1])):
            if rest.startswith(prefix):
                found = setting(family, float(rest[len(prefix):]), kind)
                if found['label'] != label: raise ValueError('Non-canonical severity label '+label)
                return found
    raise ValueError('Unknown severity label '+repr(label))


def severity_rank_key(item):
    """Sort key, mild to severe, for settings of one family/reference kind."""
    s = parse_label(item) if isinstance(item, str) else item
    return -s['value'] if FAMILIES[s['family']]['more_severe'] == 'lower' else s['value']


def ordered(labels):
    settings = [parse_label(x) for x in labels]
    if len({(s['family'], s['reference_kind']) for s in settings}) > 1:
        raise ValueError('One task orders settings of one family/reference kind')
    if len({s['label'] for s in settings}) != len(settings): raise ValueError('Duplicate severity label')
    return [s['label'] for s in sorted(settings, key=severity_rank_key)]


def grid(task, which='start', values=None):
    """Settings for one task: the default 'start'/'candidates' list or explicit values."""
    if task not in DEFAULT_GRIDS: raise ValueError('No default degradation grid for task '+repr(task))
    spec = DEFAULT_GRIDS[task]
    chosen = spec[which] if values is None else values
    labels = [setting(spec['family'], v, spec['reference_kind'])['label'] for v in chosen]
    return [parse_label(x) for x in ordered(labels)]


def spec_id(spec):
    body = {k: v for k, v in spec.items() if k != 'spec_id'}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(',', ':')).encode()).hexdigest()[:16]


def apply(image, chosen, reference_px=None):
    """Return (degraded RGB image on the same canvas, per-row spec)."""
    from PIL import Image, ImageEnhance, ImageFilter, __version__ as pillow
    chosen = parse_label(chosen) if isinstance(chosen, str) else chosen
    image = image.convert('RGB'); width, height = image.size
    family, value, kind = chosen['family'], chosen['value'], chosen['reference_kind']
    if kind is not None:
        if reference_px is None or not float(reference_px) > 0:
            raise ValueError(chosen['label']+' needs a positive reference_px')
        reference_px = float(reference_px)
    elif reference_px is not None:
        raise ValueError(chosen['label']+' takes no reference scale')
    realized = {}
    if family == 'desaturate':
        gray = image.convert('L').convert('RGB')
        out = Image.blend(gray, image, value)
        realized['luma'] = 'ITU-R 601-2 (Pillow L)'
    elif family == 'contrast':
        out = ImageEnhance.Contrast(image).enhance(value)
    elif family == 'blur_rel':
        sigma = value*reference_px
        out = image.filter(ImageFilter.GaussianBlur(sigma))
        realized.update(sigma_px=sigma, kernel='Pillow GaussianBlur(radius=sigma), extended-box approximation')
    elif family == 'lowres':
        scale = min(1., value/reference_px)
        small = (max(1, round(width*scale)), max(1, round(height*scale)))
        if small == (width, height):
            out = image.copy()
        else:
            out = image.resize(small, Image.Resampling.BOX).resize((width, height), Image.Resampling.BICUBIC)
        realized.update(scale=scale, effective_size=list(small), noop=small == (width, height))
    else:
        raise ValueError(family)
    if out.size != image.size or out.mode != 'RGB': raise AssertionError('Degradation changed the canvas')
    spec = dict(label=chosen['label'], family=family, value=value, reference_kind=kind, reference_px=reference_px,
        realized=realized, canvas=[width, height], generator=GENERATOR, pillow=pillow,
        same_realization_for_all_paired_cells=True)
    spec['spec_id'] = spec_id(spec)
    return out, spec
