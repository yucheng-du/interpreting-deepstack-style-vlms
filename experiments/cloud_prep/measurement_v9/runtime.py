"""v9 cloud runtime and query-independent visual input identities, tensor-free."""
from __future__ import annotations

import hashlib
import json

from cloud_prep.measurement_v8.runtime import (
    plan_runtime, prepare_runtime, resolve_snapshot, sha256, write_json,
)


def direction_indices(row):
    """Always orient by changed donor attribute, never off-target native truth."""
    a, b = row['direction_a_index'], row['direction_b_index']
    if type(a) is not int or type(b) is not int or a == b or min(a, b) < 0 or max(a, b) >= len(row['choices']):
        raise ValueError('Distinct valid donor a/b direction indices required')
    return a, b


def direction_readout(row, variant, scores=None):
    a, b = direction_indices(row)
    original_a, original_b = row['option_labels'][a], row['option_labels'][b]
    label_a, label_b = variant['original_to_label'][original_a], variant['original_to_label'][original_b]
    fields = {'direction_a_index': a, 'direction_b_index': b,
              'source_a_original_choice': original_a, 'source_b_original_choice': original_b,
              'source_a_value': row['choices'][a], 'source_b_value': row['choices'][b],
              'source_a_label': label_a, 'source_b_label': label_b}
    if scores is not None:
        fields['b_minus_a_log_odds'] = scores[label_b] - scores[label_a]
    return fields


def stable_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def visual_contract(row):
    a, b = direction_indices(row)
    return {'images': {name: meta['sha256'] for name, meta in row['source_image_metadata'].items()},
            'canvases': {name: [meta['width'], meta['height']] for name, meta in row['source_image_metadata'].items()},
            'roi': row.get('donor_roi_bbox_xyxy'), 'tile_grid': row.get('tile_grid', [1, 2]),
            'direction_values': [row['choices'][a], row['choices'][b]],
            'changed_object': row.get('changed_object', row.get('target_object'))}


def cache_identity(row):
    return str(row.get('intervention_id', row['id'])), stable_digest(visual_contract(row))


def visual_signature(row, condition, pixel_hashes):
    """Query-free identity of all inputs to deterministic adapter composition."""
    return stable_digest({'source_contract': visual_contract(row), 'processed_sources': pixel_hashes,
        'main': condition['main'], 'donor': condition.get('donor_source'),
        'entry_group': condition.get('entry_group'), 'region': condition.get('region'),
        'token_permutation': condition.get('token_permutation', 'identity')})


def validate_shared_interventions(rows):
    contracts = {}; previous = None
    for row in rows:
        key, signature = cache_identity(row)
        if key in contracts and contracts[key] != signature:
            raise ValueError(f'Queries disagree on shared visual intervention {key}')
        if key != previous and key in contracts:
            raise ValueError('Shared intervention queries must be contiguous to reuse exactly one feature cache')
        contracts[key] = signature
        previous = key
    return {'unique_visual_interventions': len(contracts),
            'shared_visual_contracts_validated': True,
            'shared_visual_queries_contiguous': True,
            'direction_contract': 'changed donor object a/b labels; independent of queried native truth'}
