"""Fresh deterministic generation across Transformers 4.x and 5.8.

Transformers 5.8 removed use_model_defaults; forwarding that legacy argument
through **kwargs leaves an unused model kwarg. Detect the real preparation
signature rather than interpreting **kwargs as support for this parameter.
No model/tensor libraries are imported here.
"""
from __future__ import annotations

import inspect

GREEDY_KWARGS = {"do_sample": False, "num_beams": 1, "num_return_sequences": 1, "use_cache": True}
_FORBIDDEN_STATE = ("past_key_values", "cache_position", "position_ids", "inputs_embeds", "labels", "logits_to_keep")


def greedy_kwargs(model):
    result = dict(GREEDY_KWARGS)
    if "use_model_defaults" in inspect.signature(model._prepare_generation_config).parameters:
        result["use_model_defaults"] = False
    return result


def resolve_greedy_config(model, config):
    """Record actual resolution and ensure generation-only parameters are consumed."""
    requested = greedy_kwargs(model)
    resolved, model_kwargs = model._prepare_generation_config(config, **requested)
    if any(getattr(resolved, key, None) != value for key, value in GREEDY_KWARGS.items()):
        raise RuntimeError("Effective generation settings are not the explicit greedy contract")
    if set(requested) & set(model_kwargs):
        raise RuntimeError("Generation-control arguments leaked into model kwargs")
    mode = resolved.get_generation_mode()
    if getattr(mode, "value", mode) != "greedy_search":
        raise RuntimeError("Effective generation mode is not greedy_search")
    # The common runner consumes a sequence tensor, not a GenerateOutput.
    resolved.return_dict_in_generate = False
    resolved.output_scores = False
    resolved.output_logits = False
    return resolved


def native_greedy(model, inputs, config, *, stopping_criteria=None):
    """Call the unmodified model.generate with fresh language state per condition."""
    if any(inputs.get(key) is not None for key in _FORBIDDEN_STATE):
        raise ValueError("Generation requires fresh processor inputs, without scorer or KV state")
    reserved = {*GREEDY_KWARGS, "use_model_defaults", "generation_config", "stopping_criteria"}
    if reserved & set(inputs):
        raise ValueError("Processor inputs contain generation-control arguments")
    kwargs = {key: value for key, value in inputs.items() if key not in _FORBIDDEN_STATE}
    return model.generate(**kwargs, generation_config=config, stopping_criteria=stopping_criteria,
                          **greedy_kwargs(model))
