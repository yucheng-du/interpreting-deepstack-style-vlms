"""Official processor construction; model libraries are imported lazily."""
from __future__ import annotations

import json
from pathlib import Path


def make_processor(snapshot, config):
    settings = dict(config['processor'])
    kind = settings.pop('kind')
    snapshot = Path(snapshot)
    if kind == 'qwen':
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(snapshot, local_files_only=True,
                                                  trust_remote_code=False, use_fast=settings.pop('use_fast', False))
        image = processor.image_processor
        minimum, maximum = settings.pop('min_pixels'), settings.pop('max_pixels')
        image.min_pixels, image.max_pixels = minimum, maximum
        image.size = {'shortest_edge': minimum, 'longest_edge': maximum}
        if settings:
            raise ValueError(f'Unknown Qwen processor settings: {settings}')
        return processor
    import transformers
    family = 'Gemma4' if kind == 'gemma4' else 'Gemma4Unified' if kind == 'gemma4_unified' else None
    if family is None:
        raise ValueError(f'Unknown processor kind: {kind}')
    raw = json.loads((snapshot / 'processor_config.json').read_text())
    backend = settings.pop('backend', 'pil' if kind == 'gemma4' else 'torchvision')
    expected_backend = 'pil' if kind == 'gemma4' else 'torchvision'
    if backend != expected_backend:
        raise ValueError(f'This constructor uses {expected_backend}; edit make_processor to change backend explicitly')
    if kind == 'gemma4':
        from transformers.models.gemma4.image_processing_pil_gemma4 import Gemma4ImageProcessorPil
        image_class = Gemma4ImageProcessorPil
    else:
        image_class = getattr(transformers, family + 'ImageProcessor')
    processor_class = getattr(transformers, family + 'Processor')
    extras = {k: v for k, v in raw.items() if k not in (
        'processor_class', 'image_processor', 'video_processor', 'feature_extractor',
        'tokenizer', 'chat_template', 'auto_map')}
    return processor_class(
        image_processor=image_class.from_dict(raw['image_processor'], **settings),
        video_processor=getattr(transformers, family + 'VideoProcessor').from_dict(raw['video_processor']),
        feature_extractor=getattr(transformers, family + 'AudioFeatureExtractor').from_dict(raw['feature_extractor']),
        tokenizer=transformers.AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False),
        chat_template=(snapshot / 'chat_template.jinja').read_text(), **extras)


def prepare_inputs(processor, row, config, *, tensor_type):
    from PIL import Image
    images = []
    try:
        for path in row['images']:
            with Image.open(path) as im:
                images.append(im.convert('RGB'))
        prompt = row['prompt'] + config['prompt'].get('suffix', '')
        messages = [{'role': 'user', 'content': [{'type': 'text', 'text': prompt},
                    *[{'type': 'image'} for _ in images]]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                              **config['prompt'].get('chat_template_kwargs', {}))
        image_arg = images if config['processor']['kind'] == 'qwen' else [images]
        inputs = processor(text=[text], images=image_arg, return_tensors=tensor_type, padding=False)
    finally:
        for image in images:
            image.close()
    ids = inputs['input_ids'].tolist()
    if len(ids) != 1 or not ids[0] or inputs['attention_mask'].tolist() != [[1] * len(ids[0])]:
        raise ValueError('Expected one nonempty unpadded prompt')
    if any(name in inputs for name in ('position_ids', 'past_key_values', 'inputs_embeds')):
        raise ValueError('Shared scorer requires fresh processor inputs without explicit text positions')
    prefix = processor.tokenizer.encode(text, add_special_tokens=False)
    suffixes = []
    for label in row['option_labels']:
        full = processor.tokenizer.encode(text + label, add_special_tokens=False)
        if full[:len(prefix)] != prefix or len(full) == len(prefix):
            raise ValueError(f'Label crosses prompt boundary: {label}')
        suffixes.append(full[len(prefix):])
    if len({tuple(x) for x in suffixes}) != len(suffixes):
        raise ValueError('Legal labels collide after tokenization')
    generated = config['generation']['max_new_tokens'] if config['generation']['enabled'] else 0
    if len(ids[0]) + max(generated, max(map(len, suffixes)) - 1) > config['scoring']['max_sequence']:
        raise ValueError('Prompt plus continuation exceeds max_sequence; change the explicit budget')
    info = {'prompt_text': text, 'input_ids': ids[0], 'sequence_length': len(ids[0]),
            'option_token_ids': suffixes, 'processor_inputs': {
                key: {'shape': list(value.shape), 'dtype': str(value.dtype)} for key, value in inputs.items()}}
    for name in ('image_grid_thw', 'image_position_ids', 'mm_token_type_ids'):
        if name in inputs:
            info[name] = inputs[name].tolist()
    visual_count = ids[0].count(processor.image_token_id)
    if visual_count < 1:
        raise ValueError('Processor produced no image placeholder tokens')
    if config['processor']['kind'] == 'qwen':
        grids = info['image_grid_thw']
        patch = processor.image_processor.patch_size
        merge = processor.image_processor.merge_size
        counts = [t * h * w // (merge * merge) for t, h, w in grids]
        pixels = [h * w * patch * patch for _, h, w in grids]
        if len(grids) != len(row['images']) or sum(counts) != visual_count:
            raise ValueError('Qwen image order/count/grid and placeholder count disagree')
        if any(value > config['processor']['max_pixels'] for value in pixels):
            raise ValueError('Actual Qwen image grid exceeds requested max_pixels')
        info.update(image_pixels=pixels, image_visual_tokens=counts)
    else:
        positions = info['image_position_ids']
        patches = [sum(pair != [-1, -1] for pair in image) for image in positions]
        # Unified merges 3x3 patches in the image processor; E models pool in
        # the vision encoder. Both expose the actual placeholder count.
        factor = processor.image_processor.pooling_kernel_size ** 2 if config['processor']['kind'] == 'gemma4' else 1
        counts = [n // factor for n in patches]
        if len(positions) != len(row['images']) or any(n % factor for n in patches) or sum(counts) != visual_count:
            raise ValueError('Gemma image order/count/positions and placeholder count disagree')
        if any(n > config['processor']['max_soft_tokens'] for n in counts):
            raise ValueError('Actual Gemma image tokens exceed requested soft-token budget')
        info.update(image_patch_counts=patches, image_visual_tokens=counts)
    info['visual_token_count'] = visual_count
    return text, inputs, info
