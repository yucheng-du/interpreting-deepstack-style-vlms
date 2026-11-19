"""Analytic checks with a fixed Markov transition matrix."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from deepstack.scoring import score_options, tokenize_options


class CharacterTokenizer:
    vocabulary = {"P": 0, "(": 1, "A": 2, "B": 3, ")": 4, " ": 5}

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return [self.vocabulary[char] for char in text]


class MarkovModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.table = torch.tensor([
            [0., 2., 4., 1., 0., 0.],
            [0., 0., 3., 1., 0., 0.],
            [0., 0., 0., 0., -5., 0.],
            [0., 0., 0., 0., 5., 0.],
            [0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0.],
        ])
        self.calls = []
        self.eval()

    def forward(self, input_ids, attention_mask=None, pixel_values=None, logits_to_keep=0, use_cache=None):
        assert use_cache is False
        self.calls.append({"ids": input_ids.clone(), "pixels": pixel_values, "keep": logits_to_keep})
        logits = self.table[input_ids]
        if logits_to_keep:
            logits = logits[:, -logits_to_keep:]
        return SimpleNamespace(logits=logits)


def test_single_token_labels_use_one_forward_and_correct_margin():
    tokenizer = CharacterTokenizer()
    model = MarkovModel()
    inputs = {"input_ids": torch.tensor([[0]]), "attention_mask": torch.ones((1, 1), dtype=torch.long)}
    options = tokenize_options(tokenizer, ["A", "B"], prompt_text="P")
    result = score_options(model, inputs, options, correct_label="B")
    expected = torch.log_softmax(model.table[0], dim=-1)
    assert result.scores["A"] == pytest.approx(expected[2].item())
    assert result.scores["B"] == pytest.approx(expected[3].item())
    assert result.correct_margin == pytest.approx(-3.)
    assert result.prediction == "A" and result.forward_count == len(model.calls) == 1
    assert model.calls[0]["keep"] == 1


def test_full_multitoken_label_scores_including_last_token_reverse_prediction():
    tokenizer = CharacterTokenizer()
    model = MarkovModel()
    pixels = torch.zeros((1, 3))
    inputs = {"input_ids": torch.tensor([[0]]), "attention_mask": torch.ones((1, 1), dtype=torch.long), "pixel_values": pixels}
    options = tokenize_options(tokenizer, ["(A)", "(B)"], prompt_text="P")
    result = score_options(model, inputs, options, correct_label="(B)")
    logp = torch.log_softmax(model.table, dim=-1)
    expected_a = logp[0, 1] + logp[1, 2] + logp[2, 4]
    expected_b = logp[0, 1] + logp[1, 3] + logp[3, 4]
    assert result.scores["(A)"] == pytest.approx(expected_a.item())
    assert result.scores["(B)"] == pytest.approx(expected_b.item())
    assert result.prediction == "(B)"  # First/second token alone would choose A.
    assert result.correct_margin == pytest.approx((expected_b - expected_a).item())
    assert result.forward_count == 2
    assert [call["ids"].tolist() for call in model.calls] == [[[0, 1, 2]], [[0, 1, 3]]]
    assert all(call["pixels"] is pixels and call["keep"] == 3 for call in model.calls)
    assert inputs["input_ids"].tolist() == [[0]] and inputs["attention_mask"].tolist() == [[1]]


def test_mixed_length_labels_are_complete_and_unnormalized():
    tokenizer = CharacterTokenizer()
    model = MarkovModel()
    options = tokenize_options(tokenizer, ["A", "(B)"], prompt_text="P")
    result = score_options(model, {"input_ids": torch.tensor([[0]])}, options)
    logp = torch.log_softmax(model.table, dim=-1)
    assert result.scores["A"] == pytest.approx(logp[0, 2].item())
    assert result.scores["(B)"] == pytest.approx((logp[0, 1] + logp[1, 3] + logp[3, 4]).item())
    assert result.correct_margin is None


def test_token_boundary_changes_are_rejected():
    class BoundaryTokenizer(CharacterTokenizer):
        def encode(self, text, add_special_tokens=False):
            return [42] if text == "PA" else super().encode(text, add_special_tokens=add_special_tokens)
    with pytest.raises(ValueError, match="boundary"):
        tokenize_options(BoundaryTokenizer(), ["A", "B"], prompt_text="P")


@pytest.mark.parametrize("extra", [{"past_key_values": object()}, {"position_ids": torch.tensor([[0]])}, {"attention_mask": torch.zeros((1, 1))}])
def test_cached_positions_or_padding_rejected(extra):
    options = tokenize_options(CharacterTokenizer(), ["A", "B"], prompt_text="P")
    with pytest.raises(ValueError):
        score_options(MarkovModel(), {"input_ids": torch.tensor([[0]]), **extra}, options)


def test_training_mode_rejected():
    options = tokenize_options(CharacterTokenizer(), ["A", "B"], prompt_text="P")
    with pytest.raises(ValueError, match="eval"):
        score_options(MarkovModel().train(), {"input_ids": torch.tensor([[0]])}, options)
