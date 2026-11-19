"""Versioned readout interventions; imports are light until tensor use is requested."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import re

FORMATS = ('base', 'strict_letters', 'strict_digits', 'prefix_digits')
MAPPINGS = ('identity', 'cyclic')
PARSER_VERSION = 'explicit_outer_label_v3_2'


def format_variant(row, format_name, mapping):
    if format_name not in FORMATS or mapping not in MAPPINGS:
        raise ValueError('Unknown readout format or mapping')
    # Reuse the already validated suffix locator without changing old protocols.
    from scripts.prepare_readout_controls import option_suffix
    prefix, matches = option_suffix(row)
    original = list(row['option_labels'])
    labels = ([f'({i+1})' for i in range(len(original))]
              if format_name.endswith('digits') else original)
    order = list(range(len(labels)))
    if mapping == 'cyclic':
        order = order[1:] + order[:1]
    choices = [matches[i].group(3) for i in order]
    prompt = prefix + ''.join(label + matches[i].group(2) + choice + matches[i].group(4)
                              for i, (label, choice) in enumerate(zip(labels, choices)))
    if format_name != 'base':
        prompt += '\nAnswer with exactly one option label in parentheses. Do not include any explanation.'
    assistant_prefix = 'Answer: (' if format_name == 'prefix_digits' else ''
    forward = {labels[i]: original[j] for i, j in enumerate(order)}
    inverse = {v: k for k, v in forward.items()}
    return {'format': format_name, 'mapping': mapping, 'prompt': prompt,
            'assistant_prefix': assistant_prefix, 'labels': labels,
            'continuations': [label[1:] for label in labels] if assistant_prefix else labels,
            'label_to_original': forward, 'original_to_label': inverse,
            'answer_label': inverse[row['answer_label']],
            'original_answer_label': row['answer_label'], 'display_choices': choices}


def make_inputs(processor, row, variant, device='mps'):
    from PIL import Image
    images = []
    try:
        for path in row['images']:
            with Image.open(path) as image:
                images.append(image.convert('RGB'))
        messages = [{'role': 'user', 'content': [{'type': 'text', 'text': variant['prompt']},
                    *[{'type': 'image'} for _ in images]]}]
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompt += variant['assistant_prefix']
        inputs = processor(text=[prompt], images=images, return_tensors='pt').to(device)
        return inputs, prompt
    finally:
        for image in images:
            image.close()


def tokenize_variant(tokenizer, variant, prompt_text):
    from deepstack.scoring import TokenizedOptions, tokenize_options
    encoded = tokenize_options(tokenizer, variant['continuations'], prompt_text=prompt_text)
    return TokenizedOptions(tuple(variant['labels']), encoded.token_ids, encoded.prompt_sha256)


@contextmanager
def fp32_output_head(model):
    """Project BF16 hidden states with FP32 arithmetic; this is not full FP32 inference.

    The original stored weights are not changed. The temporary FP32 copy and
    forward replacement are released on exit, including exceptional exits.
    """
    import torch.nn.functional as F
    head = model.lm_head
    if getattr(head, '_v3_fp32_owner', False):
        raise RuntimeError('Nested output-head replacement')
    had_local = 'forward' in head.__dict__
    saved = head.__dict__.get('forward')
    weight = head.weight.detach().float()
    bias = head.bias.detach().float() if getattr(head, 'bias', None) is not None else None
    head._v3_fp32_owner = True
    def forward(hidden):
        return F.linear(hidden.float(), weight, bias)
    head.forward = forward
    try:
        yield
    finally:
        if had_local:
            head.forward = saved
        else:
            del head.forward
        del head._v3_fp32_owner


def parse_answer(text, labels, assistant_prefix=''):
    """Parse explicit outer labels only; never reinterpret Point/Box content.

    Exact compliance is recorded separately from an explicit label accompanied
    by prose. Multiple explicit legal labels are ambiguous. Generated content
    is retained for review; this parser never repairs label-content conflicts.
    """
    clean = (assistant_prefix + text).strip().replace('**', '').replace('__', '')
    symbols = {label[1:-1]: label for label in labels}
    token = r'([1-5])' if all(k.isdigit() for k in symbols) else r'([A-E])'
    # Compliance concerns the actual continuation, not our supplied prefix or
    # prose that can still carry a parseable answer. Bare digits remain parsed.
    expected = [label[1:] for label in labels] if assistant_prefix.endswith('(') else labels
    exact_compliance = text.strip() in expected
    exact = re.fullmatch(r'(?:Answer\s*:\s*)?(?:\(' + token + r'\)|' + token + r')[.!]?', clean, re.I)
    found = []
    if exact:
        found.append((exact.group(1) or exact.group(2)).upper())
    initial = re.match(r'^\(' + token + r'\)(?=\s|[.!,:;]|$)', clean, re.I)
    if initial:
        found.append(initial.group(1).upper())
    explicit = re.finditer(r'\b(?:final\s+answer|(?:the\s+)?(?:correct\s+)?answer)\s*(?:is\s*:?|:)\s*(?:\(' + token + r'\)|' + token + r'\b)', clean, re.I)
    found.extend((m.group(1) or m.group(2)).upper() for m in explicit)
    mentioned = {m.upper() for m in re.findall(r'\(' + token + r'\)', clean)} | set(found)
    if len(mentioned) > 1:
        return {'status': 'ambiguous', 'label': None, 'exact_label_only': False}
    if not found:
        return {'status': 'missing', 'label': None, 'exact_label_only': False}
    if found[0] not in symbols:
        return {'status': 'invalid_label', 'label': None, 'exact_label_only': False}
    return {'status': 'parsed', 'label': symbols[found[0]], 'exact_label_only': exact_compliance}
