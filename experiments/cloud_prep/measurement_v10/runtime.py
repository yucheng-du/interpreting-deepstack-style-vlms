"""Versioned entry-role identities; CPU planning never imports a tensor library."""
from cloud_prep.measurement_v9.runtime import (
    plan_runtime, prepare_runtime, resolve_snapshot, sha256, write_json,
    direction_indices, direction_readout, stable_digest, visual_contract,
    cache_identity, validate_shared_interventions as _validate_shared_interventions,
)


def visual_signature(row, condition, pixel_hashes):
    """Include every composition input, while excluding query/readout labels."""
    keys = ('main', 'donor_source', 'ds_sources', 'context_source', 'entry_group',
            'region', 'token_permutation', 'intervention_type', 'changed_entry')
    return stable_digest({'source_contract': visual_contract(row),
        'processed_sources': pixel_hashes,
        'composition': {k: condition.get(k) for k in keys}})


def required_sources(conditions):
    names = set()
    for c in conditions:
        names.add(c['main'])
        names.update(c.get('ds_sources') or [])
        for key in ('donor_source', 'context_source'):
            if c.get(key) is not None:
                names.add(c[key])
    return names


def validate_shared_interventions(rows):
    result = _validate_shared_interventions(rows)
    result['direction_contract'] = 'Changed-object a/b semantics fixed across queries and entry roles; never flip direction merely because the changed entry changes'
    return result
