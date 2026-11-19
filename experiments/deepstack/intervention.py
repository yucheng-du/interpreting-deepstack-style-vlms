"""Reversible interventions on Qwen3-VL's three post-merger visual paths.

This module never loads a model. The adapter is for eager, single-threaded inference only.
"""

from __future__ import annotations

import inspect
from types import MethodType
from typing import Any

import torch


ALL_MASKS = ("111", "110", "101", "011", "100", "010", "001", "000")
_MISSING = object()


def validate_mask(mask: str) -> tuple[bool, bool, bool]:
    if not isinstance(mask, str) or len(mask) != 3 or set(mask) - {"0", "1"}:
        raise ValueError("mask must be a three-character binary string, e.g. '101'")
    return tuple(bit == "1" for bit in mask)


def mask_features(features: list[torch.Tensor], mask: str) -> list[torch.Tensor]:
    """Keep all three ordered slots; never mutate the caller's feature tensors."""
    keep = validate_mask(mask)
    if len(features) != 3:
        raise ValueError(f"Expected exactly three DeepStack features, got {len(features)}")
    return [feature if enabled else torch.zeros_like(feature) for feature, enabled in zip(features, keep)]


class DeepStackIntervention:
    """Temporarily mask post-merger features at the text-model entry.

    ``model`` is a Qwen3VLForConditionalGeneration or its inner Qwen3VLModel.
    ``audit=True`` records norms and checks visual-only addition, requiring
    device synchronization. Use it for initial validation, then disable it.
    ``records`` contains one entry per text forward with visual features.
    Each record includes three entries in ``paths`` and actual injection order.
    The full-image main merger and the already constructed input embeddings
    are never intercepted. No language KV cache may cross mask conditions.
    """

    def __init__(self, model: Any, mask: str = "111", *, audit: bool = False):
        self.mask = mask
        self.keep = validate_mask(mask)
        self.audit = audit
        base = getattr(model, "model", model)
        self.text_model = getattr(base, "language_model", None)
        self.visual = getattr(base, "visual", None)
        if self.text_model is None or self.visual is None:
            raise TypeError("Expected a Qwen3-VL model with language_model and visual")
        self.encoder_layers = list(self.visual.deepstack_visual_indexes)
        if len(self.encoder_layers) != 3 or len(self.visual.deepstack_merger_list) != 3:
            raise ValueError("This experiment requires exactly three supplemental visual paths")
        # Official vision forward appends in traversal order and indexes mergers
        # using config order. Reject unusual ordering instead of mislabelling it.
        if self.encoder_layers != sorted(set(self.encoder_layers)):
            raise ValueError("deepstack_visual_indexes must be unique and increasing")
        if len(self.text_model.layers) < 3:
            raise ValueError("Need at least three text decoder layers")
        if not callable(getattr(self.text_model, "_deepstack_process", None)):
            raise TypeError("Unsupported Qwen3-VL implementation: no _deepstack_process")
        self.records: list[dict[str, Any]] = []
        self._active = False
        self._frame = None
        self._handles = []

    def __enter__(self) -> "DeepStackIntervention":
        text = self.text_model
        if self._active or getattr(text, "_deepstack_intervention_owner", None) is not None:
            raise RuntimeError("Nested or concurrent DeepStack interventions are unsupported")
        self._original_forward = text.forward
        self._original_process = text._deepstack_process
        self._signature = inspect.signature(self._original_forward)
        if "deepstack_visual_embeds" not in self._signature.parameters:
            raise TypeError("Text forward does not expose deepstack_visual_embeds")
        self._saved_forward = text.__dict__.get("forward", _MISSING)
        self._saved_process = text.__dict__.get("_deepstack_process", _MISSING)
        self._active = True
        text._deepstack_intervention_owner = self
        try:
            for idx, layer in enumerate(text.layers):
                self._handles.append(layer.register_forward_hook(self._decoder_hook(idx)))

            def forward(_module, *args, **kwargs):
                return self._forward(*args, **kwargs)

            def process(_module, hidden_states, visual_pos_masks, visual_embeds):
                return self._process(hidden_states, visual_pos_masks, visual_embeds)

            text.forward = MethodType(forward, text)
            text._deepstack_process = MethodType(process, text)
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if not self._active:
            return False
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        for name, saved in (("forward", self._saved_forward), ("_deepstack_process", self._saved_process)):
            if saved is _MISSING:
                delattr(self.text_model, name)
            else:
                setattr(self.text_model, name, saved)
        delattr(self.text_model, "_deepstack_intervention_owner")
        self._active = False
        self._frame = None
        return False

    def _decoder_hook(self, idx):
        def record_decoder(_module, _args, _output):
            if self._frame is not None:
                self._frame["last_decoder"] = idx
        return record_decoder

    def _forward(self, *args, **kwargs):
        if self._frame is not None:
            raise RuntimeError("Reentrant text forwards are unsupported")
        bound = self._signature.bind_partial(*args, **kwargs)
        original_features = bound.arguments.get("deepstack_visual_embeds")
        if original_features is None:
            return self._original_forward(*args, **kwargs)
        masked = mask_features(original_features, self.mask)
        visual_mask = bound.arguments.get("visual_pos_masks")
        if visual_mask is None or visual_mask.dtype != torch.bool or visual_mask.ndim != 2:
            raise ValueError("Expected a boolean visual_pos_masks tensor of shape [batch, sequence]")
        frame = {
            "mask": self.mask,
            "audit": self.audit,
            "visual_mask_shape": list(visual_mask.shape),
            "main_visual_input_intercepted": False,
            "paths": [],
            "injection_order": [],
            "completed": False,
        }
        if self.audit:
            frame["visual_positions"] = visual_mask.nonzero(as_tuple=False).detach().cpu().tolist()
            frame["visual_token_count"] = int(visual_mask.sum().item())
        for idx, feature in enumerate(original_features):
            if feature.ndim != 2:
                raise ValueError("Expected each post-merger feature to have shape [visual_tokens, hidden]")
            path = {
                "path": idx,
                "encoder_layer": int(self.encoder_layers[idx]),
                "decoder_after": idx,
                "enabled": self.keep[idx],
                "shape": list(feature.shape),
                "dtype": str(feature.dtype),
                "device": str(feature.device),
            }
            if self.audit:
                if feature.shape[0] != frame["visual_token_count"]:
                    raise ValueError("Post-merger feature count does not match visual mask")
                path["original_norm"] = float(feature.detach().float().norm().item())
                path["applied_norm"] = path["original_norm"] if self.keep[idx] else 0.0
            frame["paths"].append(path)
        # Preserve the caller's argument style. BoundArguments.args would move
        # keyword arguments into positional slots; HF's check_model_inputs
        # decorator then injects use_cache again as a keyword, causing a
        # duplicate argument (and potentially replacing explicit False).
        call_args = args
        call_kwargs = dict(kwargs)
        if "deepstack_visual_embeds" in call_kwargs:
            call_kwargs["deepstack_visual_embeds"] = masked
        else:
            positional_names = [
                name for name, parameter in self._signature.parameters.items()
                if parameter.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            ]
            feature_position = positional_names.index("deepstack_visual_embeds")
            replaced_args = list(args)
            replaced_args[feature_position] = masked
            call_args = tuple(replaced_args)
        self.records.append(frame)
        self._frame = frame
        self._expected_features = masked
        try:
            result = self._original_forward(*call_args, **call_kwargs)
            if frame["injection_order"] != [0, 1, 2]:
                raise RuntimeError(f"Unexpected injection order: {frame['injection_order']}")
            frame["completed"] = True
            return result
        finally:
            frame.pop("last_decoder", None)
            self._frame = None
            self._expected_features = None

    def _process(self, hidden_states, visual_pos_masks, visual_embeds):
        frame = self._frame
        if frame is None:
            raise RuntimeError("DeepStack injection occurred outside the intercepted text forward")
        idx = len(frame["injection_order"])
        if idx >= 3 or frame.get("last_decoder") != idx:
            raise RuntimeError(f"Unexpected DeepStack injection after decoder {frame.get('last_decoder')}")
        if visual_embeds is not self._expected_features[idx]:
            raise RuntimeError("DeepStack feature order or identity changed inside text forward")
        before = hidden_states.clone() if self.audit else None
        result = self._original_process(hidden_states, visual_pos_masks, visual_embeds)
        if self.audit:
            local_mask = visual_pos_masks.to(before.device)
            expected = before[local_mask] + visual_embeds.to(before.device, before.dtype)
            nonvisual_unchanged = bool(torch.equal(result[~local_mask], before[~local_mask]))
            visual_addition_exact = bool(torch.equal(result[local_mask], expected))
            frame["paths"][idx].update(
                nonvisual_unchanged=nonvisual_unchanged,
                visual_addition_exact=visual_addition_exact,
            )
            if not nonvisual_unchanged or not visual_addition_exact:
                raise RuntimeError("Official DeepStack addition failed visual-only semantic audit")
        frame["injection_order"].append(idx)
        return result
