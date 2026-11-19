"""Small, torch-free configuration and provenance helpers."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + '\n')
    temporary.replace(path)


def read_config(path, overrides=()):
    config = json.loads(Path(path).read_text())
    for override in overrides:
        key, separator, encoded = override.partition('=')
        if not separator:
            raise ValueError('Use --set dotted.key=JSON (quote string values)')
        names = key.split('.')
        target = config
        for name in names[:-1]:
            if name not in target or not isinstance(target[name], dict):
                raise ValueError(f'Unknown configuration section: {key}')
            target = target[name]
        # New keys inside an existing section allow new official generation or
        # processor options; the complete resolved config is always recorded.
        target[names[-1]] = json.loads(encoded)
    model = config['model']
    if not model['id'] or not model['revision'] or model['total_parameters'] <= 0:
        raise ValueError('Explicit model identity, revision and parameter budget required')
    if config['runtime']['dtype'] not in ('bfloat16', 'float16', 'float32'):
        raise ValueError('Unsupported dtype')
    if config['validation']['audit_items'] < 1 or config['validation']['score_atol'] < 0:
        raise ValueError('At least one semantic audit item and nonnegative tolerance required')
    if config['scoring']['max_sequence'] < 1:
        raise ValueError('max_sequence must be positive')
    generation = config['generation']
    if generation.get('enabled') and generation['max_new_tokens'] < 1:
        raise ValueError('max_new_tokens must be positive')
    if generation.get('num_return_sequences', 1) != 1:
        raise ValueError('This batch-one pilot records one generation per item')
    conditions = config['conditions']
    if not conditions or len(conditions) != len(set(conditions)):
        raise ValueError('Conditions must be nonempty and unique')
    return config


def select_rows(rows, selection):
    ids = selection.get('ids', [])
    if ids:
        if len(ids) != len(set(ids)):
            raise ValueError('Duplicate requested IDs')
        lookup = {row['id']: row for row in rows}
        missing = set(ids) - lookup.keys()
        if missing:
            raise ValueError(f'Missing requested IDs: {sorted(missing)}')
        return [lookup[item] for item in ids]
    offset, limit = selection.get('offset', 0), selection.get('limit')
    if offset < 0 or (limit is not None and limit < 1):
        raise ValueError('Invalid offset/limit')
    selected = rows[offset:] if limit is None else rows[offset:offset + limit]
    if not selected or (limit is not None and len(selected) != limit):
        raise ValueError('Selection is empty or shorter than the requested limit')
    return selected


def check_local_write(path, *, cache=False):
    return Path(path).expanduser().resolve()


def parse_generation(text, labels, strategy):
    if strategy in ('strict_final', 'existing_explicit'):
        from scripts.prepare_readout_controls import parse_generated_label
        return parse_generated_label(text, labels)
    if strategy == 'exact_label':
        clean = text.strip()
        return {'status': 'parsed' if clean in labels else 'missing',
                'label': clean if clean in labels else None}
    raise ValueError(f'Unknown parser: {strategy}; edit this helper to add one')


def generation_kwargs(config):
    result = deepcopy(config)
    result.pop('enabled', None)
    result.setdefault('do_sample', False)
    result.setdefault('num_beams', 1)
    result.setdefault('num_return_sequences', 1)
    result.setdefault('use_cache', True)
    result.update(return_dict_in_generate=False, output_scores=False, output_logits=False)
    return result
