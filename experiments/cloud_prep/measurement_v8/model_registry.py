"""Pinned v8 checkpoints and loading contracts; import is tensor-free.

Counts for new towers are preflight expectations, not measured v8 CUDA results.
Cosmos checkpoint counts include generation/audio/action parameters; the loader
instantiates only its Qwen-derived understanding tower.
"""
from __future__ import annotations
from copy import deepcopy

COSMOS_LOADER_REVISION = 'cf5d68c00d97ccd2480a2320ed652b92dec63102'
COSMOS_LOADER_REPOSITORY = 'https://github.com/NVIDIA/cosmos-framework'
COSMOS_LOADER_SHA256 = {
    'model.py': '192e32053e3e6d30054ef50687eb686e3800cb63ad9212ed3269a4790b542201',
    'config.py': '002982501692343a5be7a61fa23989a03426b996f84b44beac5d96275f9c03d7',
    '__init__.py': '8f8b24e2458f2c31439c8c5fbf1511adab5f7ccfcc6c79884dc4d520a3e7376e',
}
QWEN_TRANSFORMERS_VERSION = '5.8.0'


def _qwen(model, revision, count, *, backend='qwen_dense', vendor='Qwen', checkpoint_count=None,
          vision_indexes=(8, 16, 24)):
    return dict(model=model, revision=revision, family='qwen', backend=backend,
        vendor=vendor, base_family='Qwen3-VL', total_parameters=count,
        parameter_count_status='expected_from_fixed_config; verify_loaded_count_on_CUDA',
        checkpoint_parameters=checkpoint_count or count,
        checkpoint_weight_bytes=2 * (checkpoint_count or count),
        bf16_parameter_bytes=2 * count, transformers_version=QWEN_TRANSFORMERS_VERSION,
        main_entry='post_final_vision_merger', later_entries=[0, 1, 2],
        entry_groups={'all': [0, 1, 2]}, vision_indexes=list(vision_indexes),
        image_processor={'patch_size': 16, 'merge_size': 2, 'min_pixels': 4096, 'max_pixels': 262144},
        training_background='Cosmos3 continued multimodal / Physical AI training' if vendor=='NVIDIA' else 'Qwen3-VL Instruct')


MODELS = {
    'qwen4b': _qwen('Qwen/Qwen3-VL-4B-Instruct', 'ebb281ec70b05090aa6165b016eac8ec08e71b17',
                    4437815808, vision_indexes=(5, 11, 17)),
    'qwen8b': _qwen('Qwen/Qwen3-VL-8B-Instruct', '0c351dd01ed87e9c1b53cbc748cba10e6187ff3b', 8767123696),
    'qwen32b': _qwen('Qwen/Qwen3-VL-32B-Instruct', '0cfaf48183f594c314753d30a4c4974bc75f3ccb', 33357390064),
    'qwen30b_a3b': _qwen('Qwen/Qwen3-VL-30B-A3B-Instruct', '9c4b90e1e4ba969fd3b5378b57d966d725f1b86c', 31070754032, backend='qwen_moe'),
    'cosmos_nano': _qwen('nvidia/Cosmos3-Nano', 'e59a53c25979a090fa8706c9acc0c254a6e89b92', 8767123696,
                          backend='cosmos3', vendor='NVIDIA', checkpoint_count=15750057456),
    'cosmos_super': _qwen('nvidia/Cosmos3-Super', 'f543c56225b2e04d0ad141e29655be3a45d9c455', 33357390064,
                           backend='cosmos3', vendor='NVIDIA', checkpoint_count=64615003632),
    'ibm4b': dict(model='ibm-granite/granite-vision-4.1-4b', revision='37d591f06319e8f1638b5adcf58bdf50e0f84f7a',
                  family='granite', backend='granite', vendor='IBM', base_family='Granite4.1',
                  training_background='Granite Vision 4.1', total_parameters=3997203904,
                  parameter_count_status='fixed_checkpoint_index; verify_loaded_count_on_CUDA',
                  checkpoint_parameters=3997203904, checkpoint_weight_bytes=8508215168,
                  bf16_parameter_bytes=7994407808, main_entry=0,
                  later_entries=[3,6,9,12,15,18,21],
                  entry_groups={'all':[3,6,9,12,15,18,21], 'layer':[3,6,9], 'spatial':[12,15,18,21]}),
}
for _key in ('cosmos_nano', 'cosmos_super'):
    MODELS[_key].update(loader_revision=COSMOS_LOADER_REVISION,
                       loader_repository=COSMOS_LOADER_REPOSITORY,
                       loader_subdirectory='packages/transformers-cosmos3',
                       loader_source_sha256=deepcopy(COSMOS_LOADER_SHA256),
                       parameter_count_status='understanding_tower_estimate_from_fixed_config; verify_loaded_count_on_CUDA')
MODELS['qwen8b']['parameter_count_status'] = 'verified_in_v7; v8_backend_bridge_pending'
# Same checkpoint/count as measurement_v3/snapshot_qwen4b.json and completed v7.
# The v12 modern-backend bridge remains a separate CUDA check.
MODELS['qwen4b']['parameter_count_status'] = 'verified_in_v7; v12_modern_backend_bridge_pending'
NEW_MODELS = ('ibm4b', 'cosmos_nano', 'cosmos_super', 'qwen32b', 'qwen30b_a3b')
MECHANISM_MODELS = ('qwen8b', 'qwen32b', 'ibm4b')
ALIASES = {'qwen4': 'qwen4b', 'qwen8': 'qwen8b', 'qwen32': 'qwen32b', 'qwen30moe': 'qwen30b_a3b',
           'qwen30b': 'qwen30b_a3b', 'qwen30b-a3b': 'qwen30b_a3b',
           'cosmos-nano': 'cosmos_nano', 'cosmos-super': 'cosmos_super',
           'cosmos3-nano': 'cosmos_nano', 'cosmos3-super': 'cosmos_super',
           'granite4b': 'ibm4b', 'granite': 'ibm4b', 'ibm': 'ibm4b'}
ALIASES.update({v['model'].lower(): k for k, v in MODELS.items()})


def get_model(key):
    """Return an independent JSON-serializable spec with canonical ``key``."""
    canonical = ALIASES.get(str(key).lower(), str(key).lower())
    if canonical not in MODELS:
        raise ValueError(f'Unknown v8 checkpoint {key!r}; choose {tuple(MODELS)}')
    return {'key': canonical, **deepcopy(MODELS[canonical])}


def validate_spec(spec):
    """Prevent a supplied mutable spec from silently changing checkpoint lineage."""
    fixed = get_model(spec.get('key', spec.get('model')))
    for field in ('model', 'revision', 'backend', 'family'):
        if spec.get(field) != fixed[field]:
            raise ValueError(f'Pinned model spec mismatch for {field}')
    return fixed
