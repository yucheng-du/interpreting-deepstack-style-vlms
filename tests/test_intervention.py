"""Intervention tests on small synthetic tensors."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from deepstack.intervention import ALL_MASKS, DeepStackIntervention, mask_features, validate_mask


class AddLayer(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def forward(self, hidden):
        return hidden + self.value


class TinyText(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([AddLayer(1), AddLayer(2), AddLayer(3), AddLayer(4)])

    def _deepstack_process(self, hidden_states, visual_pos_masks, visual_embeds):
        local = hidden_states[visual_pos_masks].clone() + visual_embeds.to(hidden_states)
        hidden_states[visual_pos_masks] = local
        return hidden_states

    def forward(self, inputs_embeds=None, visual_pos_masks=None, deepstack_visual_embeds=None):
        hidden = inputs_embeds
        for idx, layer in enumerate(self.layers):
            hidden = layer(hidden)
            if deepstack_visual_embeds is not None and idx < len(deepstack_visual_embeds):
                hidden = self._deepstack_process(hidden, visual_pos_masks, deepstack_visual_embeds[idx])
        return hidden


def setup_case():
    text = TinyText()
    visual = SimpleNamespace(deepstack_visual_indexes=[5, 11, 17], deepstack_merger_list=[None] * 3)
    model = SimpleNamespace(language_model=text, visual=visual)
    main = torch.arange(12, dtype=torch.float32).reshape(1, 4, 3)
    visual_mask = torch.tensor([[False, True, True, False]])
    features = [torch.full((2, 3), float(value)) for value in [2, 8, 32]]
    return model, main, visual_mask, features


@pytest.mark.parametrize("mask", ALL_MASKS)
def test_eight_masks_keep_main_feature_order_and_nonvisual_positions(mask):
    model, main, visual_mask, features = setup_case()
    saved_main = main.clone()
    saved_features = [feature.clone() for feature in features]
    with DeepStackIntervention(model, mask, audit=True) as adapter:
        output = model.language_model(inputs_embeds=main, visual_pos_masks=visual_mask, deepstack_visual_embeds=features)
    expected = main + 10
    expected[visual_mask] += sum(feature for bit, feature in zip(mask, features) if bit == "1") if "1" in mask else 0
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    torch.testing.assert_close(main, saved_main, rtol=0, atol=0)
    for feature, saved in zip(features, saved_features):
        torch.testing.assert_close(feature, saved, rtol=0, atol=0)
    record = adapter.records[0]
    assert record["injection_order"] == [0, 1, 2]
    assert record["visual_positions"] == [[0, 1], [0, 2]]
    assert [path["encoder_layer"] for path in record["paths"]] == [5, 11, 17]
    assert all(path["nonvisual_unchanged"] and path["visual_addition_exact"] for path in record["paths"])
    assert record["completed"]
    assert "forward" not in model.language_model.__dict__
    assert "_deepstack_process" not in model.language_model.__dict__


def test_all_on_exactly_matches_original_and_no_feature_mutation():
    model, main, visual_mask, features = setup_case()
    original = model.language_model(main, visual_mask, features)
    with DeepStackIntervention(model, "111") as adapter:
        all_on = model.language_model(main, visual_mask, features)
    assert torch.equal(original, all_on)
    assert "original_norm" not in adapter.records[0]["paths"][0]
    masked = mask_features(features, "101")
    assert len(masked) == 3 and masked[0] is features[0] and masked[2] is features[2]
    assert masked[1] is not features[1] and torch.count_nonzero(masked[1]) == 0


def test_restore_after_exception_and_reject_nested_context():
    model, *_ = setup_case()
    with pytest.raises(RuntimeError, match="stop"):
        with DeepStackIntervention(model):
            with pytest.raises(RuntimeError, match="Nested"):
                with DeepStackIntervention(model):
                    pass
            raise RuntimeError("stop")
    assert "forward" not in model.language_model.__dict__
    assert "_deepstack_intervention_owner" not in model.language_model.__dict__
    assert all(not layer._forward_hooks for layer in model.language_model.layers)


def test_no_visual_features_is_transparent():
    model, main, *_ = setup_case()
    with DeepStackIntervention(model, "000") as adapter:
        output = model.language_model(inputs_embeds=main)
    assert torch.equal(output, main + 10)
    assert adapter.records == []


def test_unknown_configuration_and_wrong_feature_count_fail_closed():
    model, main, visual_mask, features = setup_case()
    with DeepStackIntervention(model) as adapter:
        with pytest.raises(ValueError, match="exactly three"):
            model.language_model(main, visual_mask, features[:2])
    assert adapter.records == []
    model.visual.deepstack_visual_indexes = [11, 5, 17]
    with pytest.raises(ValueError, match="increasing"):
        DeepStackIntervention(model)


@pytest.mark.parametrize("mask", ["", "11", "1111", "1a1", 111, None])
def test_invalid_masks(mask):
    with pytest.raises(ValueError):
        validate_mask(mask)
