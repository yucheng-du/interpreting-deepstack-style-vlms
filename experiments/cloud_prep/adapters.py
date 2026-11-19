"""Small cloud entry points around the existing, independently defined adapters.

Import and config inspection are pure Python. Check the CUDA intervention on each new checkpoint.
"""
from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Any, Mapping, Sequence


MASKS = ("111", "110", "101", "011", "100", "010", "001", "000")


def _get(value: Any, key: str, default: Any = None) -> Any:
    return value.get(key, default) if isinstance(value, Mapping) else getattr(value, key, default)


def _partition(depth: int, groups: Sequence[Sequence[int]]) -> tuple[tuple[int, ...], ...]:
    groups = tuple(tuple(group) for group in groups)
    if len(groups) != 3 or any(not group for group in groups):
        raise ValueError("PLE needs three nonempty groups; edit intervention.layer_groups")
    if any(type(index) is not int for group in groups for index in group):
        raise ValueError("PLE layer indices must be integers")
    if tuple(index for group in groups for index in group) != tuple(range(depth)):
        raise ValueError("PLE groups must be contiguous and cover decoder depth exactly once")
    return groups


def gemma_kv_layout(text_config: Any) -> dict[str, Any]:
    """Mirror the official per-type producer selection using config, without torch.

    In Gemma4TextDecoderLayer PLE is applied *after* this layer's attention.
    Thus a PLE change at layer i cannot change that same layer's stored KV.
    This is a static reachability description, not an empirical effect estimate.
    """
    depth = _get(text_config, "num_hidden_layers")
    shared = _get(text_config, "num_kv_shared_layers", 0)
    types = list(_get(text_config, "layer_types", []))
    if type(depth) is not int or type(shared) is not int or not 0 <= shared < depth:
        raise ValueError("Invalid Gemma decoder/shared-KV depth")
    if len(types) != depth or any(not isinstance(kind, str) for kind in types):
        raise ValueError("Need the actual config.layer_types for every decoder layer")
    boundary = depth - shared
    has_ple = bool(_get(text_config, "hidden_size_per_layer_input", 0))
    producers = {kind: max(i for i, item in enumerate(types[:boundary]) if item == kind)
                 for kind in dict.fromkeys(types[:boundary])}
    if any(kind not in producers for kind in types[boundary:]):
        raise ValueError("A shared-KV consumer type has no preceding producer")
    return {
        "decoder_layers": depth,
        "num_kv_shared_layers": shared,
        "first_shared_layer": boundary if shared else None,
        "shared_kv_producers": producers if shared else {},
        "shared_kv_consumers": [
            {"layer": i, "attention_type": types[i], "producer": producers[types[i]]}
            for i in range(boundary, depth)
        ],
        "all_kv_writing_layers": list(range(boundary)),
        "ple_applied_after_attention": has_ple,
        "image_ple_layers_without_later_kv_write": list(range(boundary - 1, depth)) if has_ple else [],
        "evidence_level": "config_and_reference_source; checkpoint forward still needs verification",
    }


def default_ple_groups(text_config: Any) -> tuple[tuple[int, ...], ...]:
    """Editable starting partitions, preserving the existing E4B depth groups."""
    layout = gemma_kv_layout(text_config)
    depth = layout["decoder_layers"]
    if depth == 42 and layout["first_shared_layer"] == 24:
        return _partition(depth, [range(0, 14), range(14, 28), range(28, 42)])
    # E2B: 35 layers; producer layers 13 (sliding) / 14 (full). The last
    # group starts at 14, because PLE is added after attention at layer 14.
    # Derive from the actual config so a revised checkpoint isn't mislabelled.
    if depth == 35 and layout["first_shared_layer"] is not None:
        last_producer = max(layout["shared_kv_producers"].values())
        split = last_producer // 2
        return _partition(depth, [range(0, split), range(split, last_producer), range(last_producer, depth)])
    # Unrecognised configurations require an explicit exploratory partition.
    raise ValueError("Specify intervention.layer_groups for this Gemma configuration")


def architecture_from_config(model_config: Any, intervention: Mapping[str, Any]) -> dict[str, Any]:
    """Return JSON-compatible architecture metadata, with no model imports."""
    kind = intervention.get("kind")
    text = _get(model_config, "text_config", model_config)
    vision = _get(model_config, "vision_config", {})
    summary = {"kind": kind, "model_type": _get(model_config, "model_type"),
               "decoder_layers": _get(text, "num_hidden_layers"),
               "generation_scope": "native_unmodified_model_only",
               "score_use_cache": False}
    if kind == "qwen_deepstack":
        if _get(model_config, "model_type") != "qwen3_vl":
            raise ValueError("qwen_deepstack requires a Qwen3-VL checkpoint")
        sources = list(_get(vision, "deepstack_visual_indexes", []))
        if len(sources) != 3 or sources != sorted(set(sources)):
            raise ValueError("Expected three ordered, distinct Qwen visual sources")
        summary.update(
            encoder_sources=sources, decoder_after=[0, 1, 2],
            target="supplemental_post_merger_visual_features",
            main_visual_input_preserved=True,
        )
    elif kind in {"gemma_ple", "native"}:
        expected_type = "gemma4_unified" if kind == "native" else "gemma4"
        if _get(model_config, "model_type") != expected_type:
            raise ValueError(f"This Gemma adapter entry requires model_type={expected_type}")
        ple_width = _get(text, "hidden_size_per_layer_input", 0)
        layout = gemma_kv_layout(text)
        summary.update(ple_width=ple_width, kv_layout=layout,
                       vision_model_type=_get(vision, "model_type"))
        if kind == "native":
            if ple_width or layout["num_kv_shared_layers"]:
                raise ValueError("Native boundary entry is for a checkpoint without PLE/shared KV")
            if _get(vision, "model_type") != "gemma4_unified_vision":
                raise ValueError("Expected the encoder-free Gemma4 unified vision configuration")
            summary.update(target="native_encoder_free_baseline", encoder_free=True,
                           intervention_applied=False)
        else:
            if not ple_width:
                raise ValueError("This checkpoint has no PLE; choose the native adapter")
            supplied = intervention.get("layer_groups")
            groups = default_ple_groups(text) if supplied is None else _partition(layout["decoder_layers"], supplied)
            writers = layout["all_kv_writing_layers"]
            summary.update(
                target="image_context_PLE_after_projection_RMSNorm_before_identity_mix",
                main_visual_input_preserved=True, token_identity_ple_preserved=True,
                layer_groups=[list(group) for group in groups],
                group_semantics="editable_exploratory_decoder_groups; not encoder-depth sources",
                groups=[{
                    "group": index,
                    "layers": list(group),
                    "layers_with_later_kv_write": [layer for layer in group if any(w > layer for w in writers)],
                    "layers_without_later_kv_write": [layer for layer in group if not any(w > layer for w in writers)],
                    "shared_producers_reachable_by_some_group_layer": {
                        key: value for key, value in layout["shared_kv_producers"].items()
                        if any(layer < value for layer in group)
                    },
                } for index, group in enumerate(groups)],
                reachability_scope="image-position PLE to subsequent text via later attention KV writes; static, pending CUDA check",
            )
    else:
        raise ValueError(f"Unknown intervention kind: {kind!r}")
    return summary


class _Context(AbstractContextManager):
    """Uniform records accessor; preserve the existing intervention's records."""
    def __init__(self, inner: Any = None, **metadata: Any):
        self.inner = inner
        self.metadata = metadata
        self._records: list[dict[str, Any]] = []

    @property
    def records(self):
        records = self.inner.records if self.inner is not None else self._records
        for record in records:
            record.update(self.metadata)
        return records

    def __enter__(self):
        if self.inner is not None:
            self.inner.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return self.inner.__exit__(exc_type, exc_value, traceback) if self.inner is not None else False


class CloudAdapter:
    def __init__(self, model: Any, config: Mapping[str, Any]):
        self.model = model
        self.intervention = config["intervention"]
        self.kind = self.intervention["kind"]
        self.base = getattr(model, "model", model)
        self.architecture_summary = architecture_from_config(model.config, self.intervention)
        self.full_condition = "native" if self.kind == "native" else "111"
        allowed = ("native",) if self.kind == "native" else MASKS
        self.conditions = tuple(config.get("conditions", allowed))
        if not self.conditions or len(set(self.conditions)) != len(self.conditions) or any(c not in allowed for c in self.conditions):
            raise ValueError(f"Conditions must be a nonempty unique subset of {allowed}")
        self._check_runtime_structure()

    def _check_runtime_structure(self):
        text = getattr(self.base, "language_model", None)
        if text is None:
            raise TypeError("Expected a multimodal model with model.language_model")
        layers = getattr(text, "layers", ())
        if len(layers) != self.architecture_summary["decoder_layers"]:
            raise ValueError("Actual decoder depth differs from checkpoint config")
        if self.kind == "qwen_deepstack":
            visual = getattr(self.base, "visual", None)
            sources = list(getattr(visual, "deepstack_visual_indexes", ()))
            if sources != self.architecture_summary["encoder_sources"] or len(getattr(visual, "deepstack_merger_list", ())) != 3:
                raise ValueError("Actual Qwen visual paths differ from checkpoint config")
            if not callable(getattr(text, "_deepstack_process", None)):
                raise TypeError("This Qwen implementation has no _deepstack_process")
        elif self.kind == "gemma_ple":
            required = ("project_per_layer_inputs", "per_layer_projection_norm", "per_layer_model_projection",
                        "per_layer_model_projection_scale", "per_layer_input_scale")
            if any(not hasattr(text, name) for name in required):
                raise TypeError("This Gemma implementation does not expose the PLE intervention entry")
            if getattr(text, "hidden_size_per_layer_input", None) != self.architecture_summary["ple_width"]:
                raise ValueError("Actual PLE width differs from checkpoint config")
            layout = self.architecture_summary["kv_layout"]
            boundary = layout["first_shared_layer"] or len(layers)
            producers = set(layout["shared_kv_producers"].values())
            flags = []
            for index, layer in enumerate(layers):
                attention = getattr(layer, "self_attn", None)
                sharing = getattr(attention, "is_kv_shared_layer", None)
                stores = getattr(attention, "store_full_length_kv", None)
                if sharing is not None and sharing != (index >= boundary):
                    raise ValueError(f"Actual KV sharing flag disagrees at layer {index}")
                if layout["num_kv_shared_layers"] and stores is not None and stores != (index in producers):
                    raise ValueError(f"Actual shared-KV producer disagrees at layer {index}")
                flags.append({"layer": index, "is_kv_shared_layer": sharing, "store_full_length_kv": stores})
            self.architecture_summary["runtime_kv_flags"] = flags
            self.architecture_summary["runtime_kv_flags_available"] = all(
                record["is_kv_shared_layer"] is not None and record["store_full_length_kv"] is not None for record in flags)

    def context(self, condition: str, inputs: Mapping[str, Any], *, audit: bool = False, self_patch: bool = False):
        """Wrap a scoring forward; callers must pass use_cache=False each time.

        Native generation uses model.generate outside this context. Full and
        self-patch controls may be requested even if omitted from a run subset.
        """
        allowed = ("native",) if self.kind == "native" else MASKS
        if condition not in allowed:
            raise ValueError(f"Unsupported condition {condition!r} for {self.kind}")
        metadata = {"cloud_condition": condition, "cloud_self_patch": self_patch}
        if self.kind == "native":
            if self_patch:
                raise ValueError("The native encoder-free baseline has no intervention to self-patch")
            return _Context(**metadata)
        if self.kind == "gemma_ple":
            from deepstack.gemma4_intervention import Gemma4PLEIntervention
            ids = inputs["input_ids"]
            image_id = _get(self.model.config, "image_token_id")
            if image_id is None:
                raise ValueError("Missing Gemma image_token_id")
            inner = Gemma4PLEIntervention(
                self.model, condition, visual_token_mask=ids == image_id, audit=audit,
                self_patch=self_patch, layer_groups=self.architecture_summary["layer_groups"])
        elif self_patch:
            from deepstack.path_controls import PathControlIntervention
            grid = inputs.get("image_grid_thw")
            if grid is None:
                raise ValueError("Qwen self-patch requires processor image_grid_thw")
            grid = grid.tolist() if hasattr(grid, "tolist") else grid
            merge = _get(_get(self.model.config, "vision_config"), "spatial_merge_size")
            if type(merge) is not int or merge < 1:
                raise ValueError("Need actual Qwen vision spatial_merge_size")
            counts = []
            for row in grid:
                if len(row) != 3 or any(type(value) is not int or value < 1 for value in row):
                    raise ValueError("Invalid Qwen image grid")
                cells = row[0] * row[1] * row[2]
                if cells % (merge * merge):
                    raise ValueError("Qwen grid is not divisible by the spatial merge area")
                counts.append(cells // (merge * merge))
            inner = PathControlIntervention(
                self.model, {"name": "selfrestore", "kind": "selfrestore", "alpha": [1., 1., 1.]},
                counts, seed=0, item_id="self_patch", audit=audit)
        else:
            from deepstack.intervention import DeepStackIntervention
            inner = DeepStackIntervention(self.model, condition, audit=audit)
        return _Context(inner, **metadata)


def build_adapter(model: Any, config: Mapping[str, Any]) -> CloudAdapter:
    """Build from the run's intervention/conditions fields; never load weights."""
    return CloudAdapter(model, config)
