"""Qwen native images/readout for the common real-image and natural-answer arms.

The historical v3 DS/readout/slot runners remain separate and unchanged.
"""
from contextlib import nullcontext


def supported_conditions(spec):
    return ['native']


def supported_generation_conditions(spec):
    return ['native']


def load_backend(spec, snapshot_path, device='cuda'):
    import sys
    if sys.platform != 'linux' or device != 'cuda':
        raise RuntimeError('Linux CUDA only; no local model execution')
    import torch
    from transformers import Qwen3VLForConditionalGeneration
    from scripts.run_pilot import make_processor
    model = Qwen3VLForConditionalGeneration.from_pretrained(snapshot_path, local_files_only=True,
        torch_dtype=torch.bfloat16, attn_implementation='sdpa').eval().to(device)
    count = sum(p.numel() for p in model.parameters())
    if count != spec['total_parameters']:
        raise ValueError('Qwen loaded parameter count differs')
    settings = spec['processor']
    processor = make_processor(snapshot_path, settings['min_pixels'], settings['max_pixels'])
    return model, processor, {'actual_loaded_parameters':count, 'processor':processor.image_processor.to_dict(),
        'architecture':'Qwen3-VL native; DS intervention uses existing readout/slots runners'}


def make_inputs(processor, row, variant, spec, device='cuda'):
    import sys
    if sys.platform != 'linux' or device != 'cuda':
        raise RuntimeError('Linux CUDA only; no local tensor execution')
    from deepstack.readout_v3 import make_inputs as prepare
    inputs, prompt = prepare(processor, row, variant, device=device)
    return inputs, prompt, {'image_grid_thw':inputs['image_grid_thw'].detach().cpu().tolist()}


def intervention(model, inputs, spec, condition, audit=False):
    if condition != 'native':
        raise ValueError('Use the v3 DS/slot runner for Qwen mechanism interventions')
    return nullcontext()
