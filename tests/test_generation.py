"""Generation API regressions without torch, model construction or CUDA.

The stubs model the two supported generation APIs.
"""
from __future__ import annotations

import copy
from types import MethodType, SimpleNamespace
import unittest

from cloud_prep.generation import prepare_native_generation


class Config:
    defaults = dict(do_sample=False, num_beams=1, num_return_sequences=1,
                    use_cache=True, max_new_tokens=512, temperature=1.0, top_p=1.0,
                    top_k=0, repetition_penalty=1.0, return_dict_in_generate=False,
                    output_scores=False, output_logits=False, output_attentions=False,
                    output_hidden_states=False, cache_implementation=None, eos_token_id=9)

    def __init__(self, **kwargs):
        self.__dict__.update(self.defaults, **kwargs)
        self._from_model_config = False

    def update(self, defaults_only=False, allow_custom_entries=False, **kwargs):
        unused = {}
        for key, value in kwargs.items():
            if hasattr(self, key) or allow_custom_entries:
                if not defaults_only or getattr(self, key, None) is None:
                    setattr(self, key, value)
            else:
                unused[key] = value
        return unused

    def to_dict(self):
        return dict(vars(self))

    def _get_default_generation_params(self):
        return dict(self.defaults)

    def get_generation_mode(self):
        return SimpleNamespace(value='sample' if self.do_sample else 'greedy_search')


class ModernModel:
    def __init__(self):
        self.generation_config = Config(do_sample=True, top_p=0.8)
        self.config = SimpleNamespace(_get_generation_parameters=lambda: {})
        self.calls = []

    def _prepare_generation_config(self, generation_config, **kwargs):
        resolved = copy.deepcopy(generation_config or self.generation_config)
        return resolved, resolved.update(**kwargs)

    def generate(self, input_ids, generation_config=None, **kwargs):
        resolved, unused = self._prepare_generation_config(generation_config, **kwargs)
        unused = {key: value for key, value in unused.items()
                  if key not in ('output_attentions', 'output_hidden_states')}
        if unused:
            raise ValueError(f'Unknown model kwargs: {sorted(unused)}')
        self.calls.append((generation_config, kwargs, resolved))
        return [input_ids, resolved.do_sample]


class LegacyModel(ModernModel):
    def _prepare_generation_config(self, generation_config, use_model_defaults=None, **kwargs):
        resolved = copy.deepcopy(generation_config or self.generation_config)
        # 4.57 refills global-default values in a supplied config from checkpoint
        # defaults, unless explicitly disabled. kwargs still have final priority.
        if generation_config is not None and use_model_defaults is not False:
            if not resolved.do_sample:
                resolved.do_sample = self.generation_config.do_sample
        return resolved, resolved.update(**kwargs)


class GenerationTests(unittest.TestCase):
    def test_same_explicit_kwargs_reach_preflight_and_generate_on_both_apis(self):
        for cls in (LegacyModel, ModernModel):
            with self.subTest(api=cls.__name__):
                model = cls()
                settings = {'enabled': True, 'max_new_tokens': 37, 'do_sample': False,
                            'top_p': 1.0, 'top_k': 0}
                before = copy.deepcopy(settings)
                prepared = prepare_native_generation(model, settings)
                self.assertEqual(prepared.mode, 'greedy_search')
                self.assertEqual(prepared.resolved_config.eos_token_id, 9)
                self.assertEqual(prepared.generate(model, {'input_ids': [1, 2]}), [[1, 2], False])
                supplied_config, kwargs, actual = model.calls[-1]
                self.assertIsNone(supplied_config)
                self.assertEqual(kwargs, prepared.explicit_kwargs)
                self.assertEqual(actual.max_new_tokens, 37)
                self.assertEqual(actual.top_p, 1.0)
                self.assertEqual(settings, before)
                self.assertTrue(model.generation_config.do_sample)

    def test_old_supplied_config_can_refill_sampling_in_457(self):
        model = LegacyModel()
        self.assertTrue(model.generate([1], generation_config=Config(do_sample=False))[-1])
        prepared = prepare_native_generation(model, {'do_sample': False})
        self.assertFalse(prepared.generate(model, {'input_ids': [1]})[-1])

    def test_removed_use_model_defaults_is_rejected_in_517(self):
        model = ModernModel()
        with self.assertRaisesRegex(ValueError, 'use_model_defaults'):
            model.generate([1], use_model_defaults=False, do_sample=False)
        prepared = prepare_native_generation(model, {'do_sample': False})
        self.assertFalse(prepared.generate(model, {'input_ids': [1]})[-1])
        with self.assertRaisesRegex(ValueError, 'use_model_defaults'):
            prepare_native_generation(model, {'use_model_defaults': False})

    def test_unknown_settings_fail_before_any_generation(self):
        model = ModernModel()
        with self.assertRaisesRegex(ValueError, 'max_new_token'):
            prepare_native_generation(model, {'max_new_token': 37})
        self.assertEqual(model.calls, [])

    def test_recognized_output_controls_may_also_be_forward_kwargs(self):
        model = ModernModel()
        resolve = model._prepare_generation_config
        def forward_controls(*args, **kwargs):
            config, unused = resolve(*args, **kwargs)
            unused.update(output_attentions=config.output_attentions,
                          output_hidden_states=config.output_hidden_states)
            return config, unused
        model._prepare_generation_config = forward_controls
        prepared = prepare_native_generation(model, {'output_attentions': True})
        self.assertTrue(prepared.resolved_config.output_attentions)
        self.assertFalse(prepared.generate(model, {'input_ids': [1]})[-1])

    def test_effective_configuration_drift_fails(self):
        model = ModernModel()
        resolve = model._prepare_generation_config
        def drift(*args, **kwargs):
            config, unused = resolve(*args, **kwargs)
            config.do_sample = True
            return config, unused
        model._prepare_generation_config = drift
        with self.assertRaisesRegex(RuntimeError, 'do_sample'):
            prepare_native_generation(model, {'do_sample': False})

    def test_unexpected_effective_mode_fails(self):
        model = ModernModel()
        model.generation_config.get_generation_mode = lambda: SimpleNamespace(value='contrastive_search')
        with self.assertRaisesRegex(RuntimeError, 'contrastive_search'):
            prepare_native_generation(model, {'do_sample': False})

    def test_explicit_sampling_remains_an_exploratory_override(self):
        prepared = prepare_native_generation(ModernModel(), {'do_sample': True})
        self.assertEqual(prepared.mode, 'sample')

if __name__ == '__main__':
    unittest.main()
