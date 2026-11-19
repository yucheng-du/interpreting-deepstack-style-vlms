"""Candidate identities and scoped registration; importing performs no tensor work.

Defaults record the official revisions inspected on 2026-10-24. This extension
does not modify the historical registry on disk or claim CUDA acceptance.
"""
from contextlib import contextmanager
from copy import deepcopy
from cloud_prep.measurement_v8 import model_registry as registry

QWEN2B = registry._qwen('Qwen/Qwen3-VL-2B-Instruct',
    '89644892e4d85e24eaac8bacfd4f463576704203', 2127532032,
    vision_indexes=(5, 11, 17))
QWEN2B.update(parameter_count_status='official_Hub_safetensors_metadata; CUDA_loaded_count_pending',
    preparation_status='v12_adapter_reused; actual_BF16_smoke_pending')
DEEPSTACK7B = dict(model='menglc/deepstack-l-vicuna-7b',
    revision='76f7d20cd835dfb59652be061bf8e3732190781f',
    family='deepstack_original', backend='deepstack_llama', vendor='MengLcool',
    base_family='Vicuna-7B-v1.5', total_parameters=7062902784,
    checkpoint_weight_bytes=14125805568,
    parameter_count_status='official_Hub_safetensors_metadata; CUDA_loaded_count_pending',
    loader_repository='https://github.com/MengLcool/DeepStack-VL',
    loader_revision='3aab69bff6d7ce587b839f518dda40d3379e522d',
    transformers_version='4.39.3',
    clip_repository='openai/clip-vit-large-patch14-336',
    clip_revision='ce19dc912ca5cd21c8a653c79e251e808ccabcd1',
    main_entry='low_resolution_projected_CLIP_grid_at_LM_input',
    later_entries=[1, 2, 3, 4], num_crops=4,
    visual_group_semantics='2D phase groups of high-resolution patch grid; unused groups zero-padded',
    preparation_status='independent_loader_geometry_delivery_and_dose_smoke; common_campaign_integration_pending')


def candidate_models():
    return {key: {'key': key, **deepcopy(value)}
            for key, value in (('qwen2b', QWEN2B), ('deepstack7b', DEEPSTACK7B))}


@contextmanager
def candidate_registry():
    """Expose Qwen2 to existing v12 planners/loaders in this process only.

    DeepStack is deliberately not registered in a factory that would silently
    route it to Qwen's geometry and dose hooks. Safe for nested serial use;
    use one CLI process per job, as in the existing campaign.
    """
    key = 'qwen2b'
    aliases = ('qwen2', QWEN2B['model'].lower())
    sentinel = object()
    previous = registry.MODELS.get(key, sentinel)
    old_aliases = {name: registry.ALIASES.get(name, sentinel) for name in aliases}
    if previous is not sentinel and previous != QWEN2B:
        raise ValueError('qwen2b already has a different process-local identity')
    registry.MODELS[key] = deepcopy(QWEN2B)
    registry.ALIASES.update({name: key for name in aliases})
    try:
        yield
    finally:
        if previous is sentinel:
            registry.MODELS.pop(key, None)
        else:
            registry.MODELS[key] = previous
        for name, value in old_aliases.items():
            if value is sentinel:
                registry.ALIASES.pop(name, None)
            else:
                registry.ALIASES[name] = value


def check_qwen2_config(config):
    vision, text = config['vision_config'], config['text_config']
    expected = (config['model_type'], vision['patch_size'], vision['spatial_merge_size'],
                vision['deepstack_visual_indexes'], text['hidden_size'], text['tie_word_embeddings'])
    if expected != ('qwen3_vl', 16, 2, [5, 11, 17], 2048, True):
        raise ValueError('Qwen2 checkpoint geometry/weight tying differs from inspected candidate')
    return {'vision_indexes': [5, 11, 17], 'later_LM_entries': [1, 2, 3],
            'patch_size': 16, 'spatial_merge_size': 2, 'hidden_size': 2048}
