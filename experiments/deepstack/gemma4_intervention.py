"""Mask only image-position context PLE in Gemma4, after its official RMSNorm.

No model is loaded here.
Supported use is eager, single-threaded, image-only, uncached inference.
"""

from __future__ import annotations

import inspect
from types import MethodType
from typing import Any, Sequence

import torch


ALL_MASKS = ("111", "110", "101", "011", "100", "010", "001", "000")
REFERENCE_TRANSFORMERS_REVISION = "96331a9f93b72697f160a958d2883d4b49a56739"
_MISSING = object()


def validate_mask(mask: str) -> tuple[bool, bool, bool]:
    if not isinstance(mask, str) or len(mask) != 3 or set(mask) - {"0", "1"}:
        raise ValueError("mask must be a three-character binary string")
    return tuple(bit == "1" for bit in mask)


def layer_partition(num_layers: int, groups: Sequence[Sequence[int]] | None = None) -> tuple[tuple[int, ...], ...]:
    """Three nonempty contiguous groups, covering decoder depth exactly once."""
    if not isinstance(num_layers, int) or num_layers < 3:
        raise ValueError("Need at least three decoder layers")
    if groups is None:
        if num_layers % 3:
            raise ValueError("Non-divisible depth requires an explicit layer_groups partition")
        width = num_layers // 3
        return tuple(tuple(range(i * width, (i + 1) * width)) for i in range(3))
    result = tuple(tuple(group) for group in groups)
    if len(result) != 3 or any(not group for group in result):
        raise ValueError("layer_groups must contain three nonempty groups")
    if any(type(layer) is not int for group in result for layer in group):
        raise ValueError("layer_groups must contain integer decoder indices")
    if tuple(layer for group in result for layer in group) != tuple(range(num_layers)):
        raise ValueError("layer_groups must be contiguous, ordered, and cover all decoder layers")
    return result


class Gemma4PLEIntervention:
    """Preserve identity PLE and main image inputs while masking context PLE.

    ``visual_token_mask`` is the original processor prompt's boolean [1, seq]
    image-placeholder mask (``input_ids == model.config.image_token_id``).
    Teacher-forced text suffixes are permitted: appended positions are false,
    and the entire mask is checked against each forward's actual image IDs.

    42 layers default to [0..13], [14..27], [28..41]. These bits name decoder
    depth segments, not three encoder sources. ``self_patch=True`` zeros the
    selected context cells and restores their original values before mixing.
    ``audit=True`` enables synchronizing tensor checks and norm diagnostics.
    It verifies PLE inputs, not downstream nonvisual hidden states: attention
    can legitimately propagate the intervention to text representations.
    """

    def __init__(
        self, model: Any, mask: str = "111", *, visual_token_mask: torch.Tensor,
        audit: bool = False, self_patch: bool = False,
        layer_groups: Sequence[Sequence[int]] | None = None,
    ):
        self.mask = mask
        self.keep = validate_mask(mask)
        self.audit = audit
        self.self_patch = self_patch
        self.base = getattr(model, "model", model)
        self.text_model = getattr(self.base, "language_model", None)
        if self.text_model is None or not callable(getattr(self.base, "get_placeholder_mask", None)):
            raise TypeError("Expected Gemma4ForConditionalGeneration or Gemma4Model")
        text = self.text_model
        required = ("project_per_layer_inputs", "per_layer_projection_norm", "per_layer_model_projection",
                    "per_layer_model_projection_scale", "per_layer_input_scale", "layers")
        if any(not hasattr(text, name) for name in required) or not getattr(text, "hidden_size_per_layer_input", 0):
            raise TypeError("Unsupported Gemma4 implementation or checkpoint without PLE")
        self.num_layers = text.config.num_hidden_layers
        if len(text.layers) != self.num_layers:
            raise ValueError("Configured decoder depth and actual layers disagree")
        self.layer_groups = layer_partition(self.num_layers, layer_groups)
        if not isinstance(visual_token_mask, torch.Tensor) or visual_token_mask.dtype != torch.bool:
            raise TypeError("visual_token_mask must be a boolean tensor")
        if visual_token_mask.ndim != 2 or visual_token_mask.shape[0] != 1 or not visual_token_mask.shape[1]:
            raise ValueError("Expected visual_token_mask of shape [1, prompt_tokens]")
        self.visual_token_mask = visual_token_mask.detach().clone()
        self.records: list[dict[str, Any]] = []
        self._active = False
        self._frame = None
        self._projecting = False
        self._handles = []

    def __enter__(self) -> "Gemma4PLEIntervention":
        if self._active or getattr(self.text_model, "_gemma4_ple_owner", None) is not None:
            raise RuntimeError("Nested or concurrent Gemma4 PLE interventions are unsupported")
        if self.base.training or self.text_model.training:
            raise ValueError("Call model.eval() before inference")
        self._original_forward = self.base.forward
        self._original_project = self.text_model.project_per_layer_inputs
        self._forward_signature = inspect.signature(self._original_forward)
        if "input_ids" not in self._forward_signature.parameters:
            raise TypeError("Unsupported multimodal forward signature")
        self._saved_forward = self.base.__dict__.get("forward", _MISSING)
        self._saved_project = self.text_model.__dict__.get("project_per_layer_inputs", _MISSING)
        self.text_model._gemma4_ple_owner = self
        self._active = True
        try:
            self._handles.append(self.text_model.per_layer_projection_norm.register_forward_hook(self._norm_hook))
            for idx, layer in enumerate(self.text_model.layers):
                self._handles.append(layer.register_forward_pre_hook(self._decoder_hook(idx), with_kwargs=True))

            def forward(_module, *args, **kwargs):
                return self._forward(*args, **kwargs)

            def project(_module, inputs_embeds, per_layer_inputs=None):
                return self._project(inputs_embeds, per_layer_inputs)

            self.base.forward = MethodType(forward, self.base)
            self.text_model.project_per_layer_inputs = MethodType(project, self.text_model)
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
        for module, name, saved in ((self.base, "forward", self._saved_forward),
                                    (self.text_model, "project_per_layer_inputs", self._saved_project)):
            if saved is _MISSING:
                if name in module.__dict__:
                    delattr(module, name)
            else:
                setattr(module, name, saved)
        delattr(self.text_model, "_gemma4_ple_owner")
        self._active = False
        self._frame = None
        self._projecting = False
        return False

    def _forward(self, *args, **kwargs):
        if self._frame is not None:
            raise RuntimeError("Reentrant multimodal forwards are unsupported")
        values = self._forward_signature.bind_partial(*args, **kwargs).arguments
        # HF wrappers may forward arbitrary keyword arguments through **kwargs.
        values = {**values.get("kwargs", {}), **values}
        ids = values.get("input_ids")
        if ids is None or ids.ndim != 2 or ids.shape[0] != 1:
            raise ValueError("Gemma4 PLE adapter requires batch-one input_ids")
        if values.get("use_cache") is not False:
            raise ValueError("Pass use_cache=False; cross-condition KV reuse is unsupported")
        for name in ("inputs_embeds", "per_layer_inputs", "past_key_values", "shared_kv_states",
                     "pixel_values_videos", "input_features"):
            if values.get(name) is not None:
                raise ValueError(f"Unsupported adapter input {name!r}")
        image_mask, video_mask, audio_mask = self.base.get_placeholder_mask(ids, None)
        if bool(video_mask.any().item()) or bool(audio_mask.any().item()):
            raise ValueError("Only image/text inputs are supported")
        prefix = self.visual_token_mask.to(ids.device)
        suffix_length = ids.shape[1] - prefix.shape[1]
        if suffix_length < 0:
            raise ValueError("Forward is shorter than the supplied visual prompt mask")
        expected = torch.cat((prefix, prefix.new_zeros((1, suffix_length))), dim=1)
        if image_mask.dtype != torch.bool or image_mask.shape != ids.shape or not torch.equal(image_mask, expected):
            raise ValueError("visual_token_mask disagrees with actual image placeholders")
        frame = {"mask": self.mask, "self_patch": self.self_patch, "audit": self.audit,
                 "layer_groups": [list(group) for group in self.layer_groups],
                 "visual_mask_shape": list(image_mask.shape), "decoder_order": [],
                 "project_calls": 0, "norm_calls": 0, "completed": False,
                 "target": "image_context_PLE_after_projection_RMSNorm_before_identity_mix",
                 "main_visual_input_intercepted": False}
        if self.audit:
            frame["visual_positions"] = image_mask.nonzero(as_tuple=False).detach().cpu().tolist()
            frame["visual_token_count"] = int(image_mask.sum().item())
        self.records.append(frame)
        self._frame = frame
        self._image_mask = image_mask
        self._combined = None
        self._main_input = None
        try:
            # Preserve args/kwargs verbatim; reconstituting BoundArguments.args
            # conflicts with HF decorators that inject default keyword values.
            result = self._original_forward(*args, **kwargs)
            if frame["project_calls"] != 1 or frame["norm_calls"] != 1:
                raise RuntimeError("Expected exactly one context PLE projection and RMSNorm")
            if frame["decoder_order"] != list(range(self.num_layers)):
                raise RuntimeError("Unexpected PLE decoder order")
            frame["completed"] = True
            return result
        finally:
            self._frame = None
            self._projecting = False
            self._image_mask = self._combined = self._main_input = None
            self._context_original = self._context_applied = None

    def _project(self, inputs_embeds, per_layer_inputs):
        frame = self._frame
        if frame is None or self._projecting or frame["project_calls"]:
            raise RuntimeError("Unexpected/reentrant context PLE projection")
        if per_layer_inputs is None:
            raise ValueError("Expected the official token-identity PLE component")
        expected_shape = (*inputs_embeds.shape[:2], self.num_layers, self.text_model.hidden_size_per_layer_input)
        if tuple(per_layer_inputs.shape) != expected_shape or inputs_embeds.shape[:2] != self._image_mask.shape:
            raise ValueError("PLE identity/input/mask shapes disagree")
        frame["project_calls"] += 1
        self._main_input = inputs_embeds
        input_before = inputs_embeds.clone() if self.audit else None
        identity_before = per_layer_inputs.clone() if self.audit else None
        self._projecting = True
        try:
            result = self._original_project(inputs_embeds, per_layer_inputs)
        finally:
            self._projecting = False
        if frame["norm_calls"] != 1:
            raise RuntimeError("Official context projection did not call RMSNorm exactly once")
        if tuple(result.shape) != expected_shape:
            raise RuntimeError("Unexpected combined PLE shape")
        if self.audit:
            scale = self.text_model.per_layer_input_scale
            original = (self._context_original + identity_before) * scale
            expected = (self._context_applied + identity_before) * scale
            image_mask = self._image_mask.to(result.device)
            checks = {
                "main_input_unchanged": bool(torch.equal(inputs_embeds, input_before)),
                "identity_unchanged": bool(torch.equal(per_layer_inputs, identity_before)),
                "official_mix_exact": bool(torch.equal(result, expected)),
                "nonvisual_ple_unchanged": bool(torch.equal(result[~image_mask], original[~image_mask])),
            }
            if self.mask == "111" or self.self_patch:
                checks["all_on_or_self_patch_exact"] = bool(torch.equal(result, original))
            frame.update(checks)
            frame["identity_norm"] = float(identity_before.float().norm().item())
            frame["mix_scale"] = float(scale)
            if not all(checks.values()):
                raise RuntimeError(f"Gemma4 PLE semantic audit failed: {checks}")
        self._combined = result
        return result

    def _norm_hook(self, _module, _args, output):
        if self._frame is None or not self._projecting:
            raise RuntimeError("PLE projection RMSNorm called outside intercepted projection")
        frame = self._frame
        frame["norm_calls"] += 1
        expected_shape = (*self._image_mask.shape, self.num_layers, self.text_model.hidden_size_per_layer_input)
        if frame["norm_calls"] != 1 or tuple(output.shape) != expected_shape:
            raise RuntimeError("Unexpected context RMSNorm count or output shape")
        enabled = [self.keep[group_idx] for group_idx, group in enumerate(self.layer_groups) for _ in group]
        off = torch.tensor([not flag for flag in enabled], dtype=torch.bool, device=output.device)
        selected = self._image_mask.to(output.device)[:, :, None, None] & off[None, None, :, None]
        applied = output if all(self.keep) else output.masked_fill(selected, 0)
        if self.self_patch:
            applied = torch.where(selected, output, applied)
        if self.audit:
            image_mask = self._image_mask.to(output.device)
            frame["context_shape"] = list(output.shape)
            frame["context_dtype"] = str(output.dtype)
            frame["context_device"] = str(output.device)
            frame["groups"] = [
                {"segment": idx, "decoder_layers": list(group), "enabled": self.keep[idx],
                 "original_visual_context_norm": float(output[image_mask][:, list(group)].float().norm().item()),
                 "applied_visual_context_norm": float(applied[image_mask][:, list(group)].float().norm().item())}
                for idx, group in enumerate(self.layer_groups)
            ]
            unchanged = torch.equal(applied[~image_mask], output[~image_mask])
            unselected = ~selected.expand_as(output)
            if not unchanged or not torch.equal(applied[unselected], output[unselected]):
                raise RuntimeError("Context masking changed unselected cells")
            self._context_original = output
            self._context_applied = applied
        return applied

    def _decoder_hook(self, idx):
        def hook(_module, args, kwargs):
            if self._frame is None or self._combined is None:
                raise RuntimeError("Decoder called before context PLE projection")
            if len(self._frame["decoder_order"]) != idx:
                raise RuntimeError("Unexpected decoder ordering")
            hidden = args[0] if args else kwargs.get("hidden_states")
            ple = args[1] if len(args) > 1 else kwargs.get("per_layer_input")
            if ple is None or tuple(ple.shape) != tuple(self._combined[:, :, idx, :].shape):
                raise RuntimeError("Decoder PLE input has an unexpected shape")
            if self.audit:
                if not torch.equal(ple, self._combined[:, :, idx, :]):
                    raise RuntimeError("Decoder received the wrong PLE slice")
                if idx == 0:
                    exact = torch.equal(hidden, self._main_input)
                    self._frame["main_input_reaches_decoder_exact"] = bool(exact)
                    if not exact:
                        raise RuntimeError("Main image/text embedding changed before decoder zero")
            self._frame["decoder_order"].append(idx)
        return hook
