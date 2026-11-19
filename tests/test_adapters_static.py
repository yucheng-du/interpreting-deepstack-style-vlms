"""Pure config/fake-interface tests: no torch, transformers, or model execution."""
from __future__ import annotations

import json
import sys
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

from cloud_prep.adapters import architecture_from_config, build_adapter, default_ple_groups, gemma_kv_layout


def gemma_config(depth=35, shared=20, period=5, ple=256):
    return {
        "model_type": "gemma4" if ple else "gemma4_unified", "image_token_id": 255,
        "text_config": {"num_hidden_layers": depth, "num_kv_shared_layers": shared,
                        "hidden_size_per_layer_input": ple,
                        "layer_types": ["full_attention" if (i + 1) % period == 0 else "sliding_attention" for i in range(depth)]},
        "vision_config": {"model_type": "gemma4_vision" if ple else "gemma4_unified_vision"},
    }


def fake_gemma(config):
    layout = gemma_kv_layout(config["text_config"])
    boundary = layout["first_shared_layer"] or layout["decoder_layers"]
    producers = set(layout["shared_kv_producers"].values())
    text = NS(
        layers=[NS(self_attn=NS(is_kv_shared_layer=i >= boundary, store_full_length_kv=i in producers))
                for i in range(layout["decoder_layers"])],
        hidden_size_per_layer_input=config["text_config"]["hidden_size_per_layer_input"],
        project_per_layer_inputs=lambda *args: None, per_layer_projection_norm=object(),
        per_layer_model_projection=object(), per_layer_model_projection_scale=1,
        per_layer_input_scale=1,
    )
    return NS(config=config, model=NS(language_model=text))


class AdapterStaticTests(unittest.TestCase):
    def test_config_and_fake_model_require_no_tensor_imports(self):
        # This remains meaningful even when a wider test harness imported torch.
        import builtins
        original = builtins.__import__
        def prohibit(name, *args, **kwargs):
            if name.split(".")[0] in {"torch", "transformers"}:
                raise AssertionError(f"Unexpected runtime import: {name}")
            return original(name, *args, **kwargs)
        with patch("builtins.__import__", side_effect=prohibit):
            adapter = build_adapter(fake_gemma(gemma_config()), {"intervention": {"kind": "gemma_ple"}})
            json.dumps(adapter.architecture_summary)

    def test_e2b_boundary_accounts_for_post_attention_ple(self):
        config = gemma_config()
        summary = architecture_from_config(config, {"kind": "gemma_ple"})
        self.assertEqual(summary["kv_layout"]["first_shared_layer"], 15)
        self.assertEqual(summary["kv_layout"]["shared_kv_producers"], {"sliding_attention": 13, "full_attention": 14})
        self.assertEqual(summary["layer_groups"], [list(range(7)), list(range(7, 14)), list(range(14, 35))])
        self.assertEqual(summary["groups"][2]["layers_with_later_kv_write"], [])
        self.assertEqual(summary["groups"][1]["shared_producers_reachable_by_some_group_layer"],
                         {"sliding_attention": 13, "full_attention": 14})

    def test_e4b_preserves_existing_groups_and_different_boundary(self):
        summary = architecture_from_config(gemma_config(42, 18, 6), {"kind": "gemma_ple"})
        self.assertEqual(summary["layer_groups"], [list(range(14)), list(range(14, 28)), list(range(28, 42))])
        self.assertEqual(summary["kv_layout"]["shared_kv_producers"], {"sliding_attention": 22, "full_attention": 23})
        self.assertEqual(summary["groups"][1]["layers_without_later_kv_write"], list(range(23, 28)))

    def test_partition_is_editable_without_forcing_divisibility(self):
        groups = [list(range(11)), list(range(11, 23)), list(range(23, 35))]
        result = architecture_from_config(gemma_config(), {"kind": "gemma_ple", "layer_groups": groups})
        self.assertEqual(result["layer_groups"], groups)
        with self.assertRaises(ValueError):
            architecture_from_config(gemma_config(), {"kind": "gemma_ple", "layer_groups": [[0], [1], [2]]})

    def test_actual_layer_types_determine_producers(self):
        config = gemma_config()
        config["text_config"]["layer_types"][13:15] = ["full_attention", "sliding_attention"]
        layout = gemma_kv_layout(config["text_config"])
        self.assertEqual(layout["shared_kv_producers"], {"sliding_attention": 14, "full_attention": 13})

    def test_native_12b_has_no_synthetic_masks(self):
        config = gemma_config(48, 0, 6, 0)
        adapter = build_adapter(fake_gemma(config), {"intervention": {"kind": "native"}})
        self.assertEqual(adapter.conditions, ("native",))
        self.assertTrue(adapter.architecture_summary["encoder_free"])
        self.assertFalse(adapter.architecture_summary["kv_layout"]["ple_applied_after_attention"])
        with adapter.context("native", {}, audit=True) as context:
            self.assertEqual(context.records, [])
        with self.assertRaises(ValueError):
            adapter.context("000", {})
        with self.assertRaises(ValueError):
            adapter.context("native", {}, self_patch=True)
        with self.assertRaises(ValueError):
            architecture_from_config(config, {"kind": "gemma_ple"})

    def test_runtime_kv_mismatch_cannot_be_silently_labelled(self):
        model = fake_gemma(gemma_config())
        model.model.language_model.layers[14].self_attn.store_full_length_kv = False
        with self.assertRaisesRegex(ValueError, "producer"):
            build_adapter(model, {"intervention": {"kind": "gemma_ple"}})

    def test_qwen_masks_and_existing_selfrestore_factory(self):
        config = {"model_type": "qwen3_vl", "text_config": {"num_hidden_layers": 32},
                  "vision_config": {"deepstack_visual_indexes": [8, 16, 24], "spatial_merge_size": 2}}
        model = NS(config=config, model=NS(
            language_model=NS(layers=[object()] * 32, _deepstack_process=lambda: None),
            visual=NS(deepstack_visual_indexes=[8, 16, 24], deepstack_merger_list=[object()] * 3)))
        adapter = build_adapter(model, {"intervention": {"kind": "qwen_deepstack"}, "conditions": ["111", "000"]})
        fake = NS(records=[{"completed": True}])
        with patch.dict(sys.modules, {"deepstack.path_controls": NS(PathControlIntervention=lambda *a, **kw: (a, kw))}):
            context = adapter.context("000", {"image_grid_thw": [[1, 8, 8], [1, 4, 8]]}, self_patch=True)
            args, kwargs = context.inner
            self.assertEqual(args[2], [16, 8])
            self.assertEqual(args[1]["kind"], "selfrestore")
        with patch.dict(sys.modules, {"deepstack.intervention": NS(DeepStackIntervention=lambda *a, **kw: fake)}):
            context = adapter.context("101", {})
            self.assertEqual(context.records[0]["cloud_condition"], "101")

    def test_unknown_and_duplicate_conditions_rejected(self):
        for conditions in (["native"], ["111", "111"], []):
            with self.assertRaises(ValueError):
                build_adapter(fake_gemma(gemma_config()), {"intervention": {"kind": "gemma_ple"}, "conditions": conditions})


if __name__ == "__main__":
    unittest.main()
