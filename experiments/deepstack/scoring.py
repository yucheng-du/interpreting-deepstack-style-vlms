"""Complete option-label conditional log probabilities for batch-one prompts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import inspect
import math
from typing import Any, Mapping, Sequence

import torch


@dataclass(frozen=True)
class TokenizedOptions:
    labels: tuple[str, ...]
    token_ids: tuple[tuple[int, ...], ...]
    prompt_sha256: str

    @property
    def all_single_token(self) -> bool:
        return all(len(ids) == 1 for ids in self.token_ids)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class OptionScores:
    scores: dict[str, float]
    prediction: str
    correct_margin: float | None
    forward_count: int
    method: str
    token_ids: dict[str, list[int]]

    def to_dict(self) -> dict:
        return asdict(self)


def tokenize_options(tokenizer: Any, labels: Sequence[str], *, prompt_text: str) -> TokenizedOptions:
    """Tokenize whole labels in context and verify a common prompt boundary.

    ``prompt_text`` must be exactly the text passed to the processor, including
    the chat generation prefix. Image-token expansion can remain processor
    owned. If BPE merges across the text/label boundary, choose an appropriate
    common answer prefix instead of silently scoring a different string.
    No stripping, implicit spaces, EOS or label truncation is performed.
    """
    labels = tuple(labels)
    if len(labels) < 2 or len(set(labels)) != len(labels) or any(not label for label in labels):
        raise ValueError("Need at least two distinct nonempty labels")
    prefix = list(tokenizer.encode(prompt_text, add_special_tokens=False))
    if not prefix:
        raise ValueError("The common prompt must contain at least one token")
    encoded = []
    for label in labels:
        full = list(tokenizer.encode(prompt_text + label, add_special_tokens=False))
        if full[:len(prefix)] != prefix:
            raise ValueError(f"Tokenization crosses the prompt/label boundary for {label!r}")
        suffix = tuple(int(token_id) for token_id in full[len(prefix):])
        if not suffix:
            raise ValueError(f"Label {label!r} has no continuation tokens")
        encoded.append(suffix)
    if len(set(encoded)) != len(encoded):
        raise ValueError("Distinct option labels produced identical token sequences")
    return TokenizedOptions(labels, tuple(encoded), hashlib.sha256(prompt_text.encode()).hexdigest())


def _forward_kwargs(model, inputs, n_logits):
    result = dict(inputs)
    result["use_cache"] = False
    # Reduce a potentially large [prompt, vocabulary] allocation on Qwen3-VL.
    # Generic models without this explicit argument still work below.
    if "logits_to_keep" in inspect.signature(model.forward).parameters:
        result["logits_to_keep"] = n_logits
    return result


def _continuation_inputs(inputs: Mapping[str, Any], prefix_tokens: tuple[int, ...]) -> dict[str, Any]:
    result = dict(inputs)
    if not prefix_tokens:
        return result
    input_ids = inputs["input_ids"]
    continuation = input_ids.new_tensor([prefix_tokens])
    result["input_ids"] = torch.cat((input_ids, continuation), dim=-1)
    mask = inputs.get("attention_mask")
    if mask is not None:
        result["attention_mask"] = torch.cat((mask, mask.new_ones((1, len(prefix_tokens)))), dim=-1)
    for name in ("token_type_ids", "mm_token_type_ids"):
        if name in inputs:
            values = inputs[name]
            result[name] = torch.cat((values, values.new_zeros((1, len(prefix_tokens)))), dim=-1)
    return result


def _logits(output):
    logits = output.logits if hasattr(output, "logits") else output["logits"]
    if logits.ndim != 3 or logits.shape[0] != 1:
        raise ValueError("Expected batch-one causal logits [1, sequence, vocabulary]")
    return logits


def score_options(
    model: Any,
    inputs: Mapping[str, Any],
    options: TokenizedOptions,
    *,
    correct_label: str | None = None,
) -> OptionScores:
    """Sum log P(label token | prompt, previous label tokens), no length norm.

    Scores every legal label in full. All single-token labels share one forward.
    Otherwise each label uses a fresh teacher-forced forward of prompt plus
    label[:-1]. Vision inputs are held identical; no KV cache is reused.
    ``inputs`` must be a batch-one, unpadded processor output. Explicit position
    IDs and cache state are rejected so multimodal positions are recomputed.
    Call ``model.eval()`` before scoring. No generation protocol is implied.
    """
    if getattr(model, "training", False):
        raise ValueError("Call model.eval() before deterministic option scoring")
    input_ids = inputs.get("input_ids")
    if input_ids is None or input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] < 1:
        raise ValueError("Expected nonempty input_ids of shape [1, prompt_tokens]")
    for name in ("past_key_values", "cache_position", "position_ids", "inputs_embeds", "labels", "logits_to_keep"):
        if inputs.get(name) is not None:
            raise ValueError(f"Unsupported scoring input {name!r}; supply an uncached processor prompt")
    mask = inputs.get("attention_mask")
    if mask is not None and (mask.shape != input_ids.shape or not bool(torch.all(mask == 1).item())):
        raise ValueError("Option scoring currently requires an unpadded batch-one prompt")
    if correct_label is not None and correct_label not in options.labels:
        raise ValueError("correct_label is not a legal option")
    scores = {}
    with torch.inference_mode():
        if options.all_single_token:
            output = model(**_forward_kwargs(model, inputs, 1))
            logits = _logits(output)[0, -1].float()
            log_normalizer = torch.logsumexp(logits, dim=-1)
            targets = torch.tensor([ids[0] for ids in options.token_ids], device=logits.device)
            values = (logits[targets] - log_normalizer).detach().cpu().tolist()
            scores.update(zip(options.labels, values))
            forward_count = 1
            method = "single_token_shared_forward"
        else:
            for label, ids in zip(options.labels, options.token_ids):
                label_inputs = _continuation_inputs(inputs, ids[:-1])
                output = model(**_forward_kwargs(model, label_inputs, len(ids)))
                logits = _logits(output)[0, -len(ids):].float()
                if logits.shape[0] != len(ids):
                    raise ValueError("Model did not return enough continuation logits")
                targets = torch.tensor(ids, device=logits.device)
                selected = logits.gather(-1, targets[:, None]).squeeze(-1)
                value = (selected - torch.logsumexp(logits, dim=-1)).sum()
                scores[label] = float(value.item())
            forward_count = len(options.labels)
            method = "complete_label_teacher_forcing"
    if not all(math.isfinite(value) for value in scores.values()):
        raise FloatingPointError("Non-finite option score")
    # Ties use the first option in the declared legal-option order.
    prediction = max(options.labels, key=scores.__getitem__)
    margin = None
    if correct_label is not None:
        margin = scores[correct_label] - max(value for label, value in scores.items() if label != correct_label)
    return OptionScores(scores, prediction, margin, forward_count, method, {
        label: list(ids) for label, ids in zip(options.labels, options.token_ids)
    })
