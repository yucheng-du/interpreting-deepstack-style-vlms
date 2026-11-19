"""Explicit native generation shared by Transformers 4.57 and 5.17.

This module does not import torch or Transformers. Both supported resolvers give
explicit generation kwargs highest priority. Passing the same kwargs (without
a second GenerationConfig or the removed use_model_defaults argument) to the
preflight and generate avoids the 4.x default-value refill and the 5.x unknown
model-kwarg error.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from cloud_prep.common import generation_kwargs


@dataclass
class NativeGeneration:
    explicit_kwargs: dict[str, Any]
    resolved_config: Any
    mode: str

    def generate(self, model, inputs):
        return model.generate(**inputs, **self.explicit_kwargs)


def prepare_native_generation(model, settings):
    kwargs = generation_kwargs(settings)
    resolved, unused = model._prepare_generation_config(None, **kwargs)
    # Both resolvers intentionally also return these recognized generation
    # controls as model kwargs when enabled. Everything else must be consumed.
    unrecognized = set(unused) - {'output_attentions', 'output_hidden_states'}
    if unrecognized:
        raise ValueError(f'Unrecognized native generation settings: {sorted(unrecognized)}')
    changed = [key for key, value in kwargs.items() if getattr(resolved, key, None) != value]
    if changed:
        raise RuntimeError(f'Effective generation settings differ from explicit kwargs: {changed}')
    mode = resolved.get_generation_mode().value
    if not kwargs['do_sample'] and kwargs['num_beams'] == 1 and mode != 'greedy_search':
        raise RuntimeError(f'Expected explicit greedy_search, got {mode}')
    return NativeGeneration(kwargs, resolved, mode)
