from __future__ import annotations

import math

import pytest
import torch

from logit_lens_debugger import LensMode, LogitLensEngine


def _final_layer_lens_logits(adapter, mode: LensMode, position: int) -> torch.Tensor:
    engine = LogitLensEngine(adapter, mode)
    input_ids = adapter.tokenize("the capital of japan is")
    hidden = adapter.get_hidden_states(input_ids, [position])[-1]
    return engine.lens_logits(hidden)[0]


def test_final_norm_last_layer_reproduces_model_logits(qwen_adapter):
    """The defining sanity check: final-norm lens at the last layer == the model's own output."""
    adapter = qwen_adapter
    input_ids = adapter.tokenize("the capital of japan is")
    position = input_ids.shape[1] - 1
    with torch.inference_mode():
        expected = adapter.model(input_ids=input_ids, use_cache=False).logits[0, position].float()
    lens = _final_layer_lens_logits(adapter, LensMode.FINAL_NORM, position)
    torch.testing.assert_close(lens, expected, atol=1e-4, rtol=1e-4)


def test_raw_mode_differs_from_final_norm(qwen_adapter):
    position = 4
    raw = _final_layer_lens_logits(qwen_adapter, LensMode.RAW, position)
    normed = _final_layer_lens_logits(qwen_adapter, LensMode.FINAL_NORM, position)
    assert not torch.allclose(raw, normed)
    # raw == lm_head(hidden) directly
    input_ids = qwen_adapter.tokenize("the capital of japan is")
    hidden = qwen_adapter.get_hidden_states(input_ids, [position])[-1]
    with torch.inference_mode():
        expected = qwen_adapter.model.lm_head(hidden)[0].float()
    torch.testing.assert_close(raw, expected)


def test_run_structure_and_ordering(qwen_adapter):
    engine = LogitLensEngine(qwen_adapter)
    input_ids = qwen_adapter.tokenize("the capital of japan is")
    results = engine.run(input_ids, positions=[4], top_k=5)
    assert len(results) == 1
    layers = results[0]
    assert [layer.layer_index for layer in layers] == list(range(qwen_adapter.num_layers + 1))
    vocab = qwen_adapter.model.config.vocab_size
    for layer in layers:
        assert len(layer.top_tokens) == 5
        assert [t.rank for t in layer.top_tokens] == [1, 2, 3, 4, 5]
        logits = [t.logit for t in layer.top_tokens]
        probs = [t.probability for t in layer.top_tokens]
        assert logits == sorted(logits, reverse=True)
        assert probs == sorted(probs, reverse=True)
        assert all(0.0 <= p <= 1.0 for p in probs)
        assert sum(probs) <= 1.0 + 1e-6
        assert all(0 <= t.token_id < vocab for t in layer.top_tokens)
        assert all(isinstance(t.token, str) for t in layer.top_tokens)
        assert layer.entropy is not None and 0.0 <= layer.entropy <= math.log(vocab) + 1e-6
        assert layer.target is None


def test_top_k_is_clamped_to_vocab(qwen_adapter):
    engine = LogitLensEngine(qwen_adapter)
    vocab = qwen_adapter.model.config.vocab_size
    layers = engine.run(qwen_adapter.tokenize("the capital"), positions=[0], top_k=vocab + 50)[0]
    assert all(len(layer.top_tokens) == vocab for layer in layers)
    assert all(abs(sum(t.probability for t in layer.top_tokens) - 1.0) < 1e-4 for layer in layers)


def test_probabilities_match_softmax_of_logits(qwen_adapter):
    engine = LogitLensEngine(qwen_adapter)
    input_ids = qwen_adapter.tokenize("the capital of japan is")
    layers = engine.run(input_ids, positions=[2], top_k=3)[0]
    full = engine.lens_logits(qwen_adapter.get_hidden_states(input_ids, [2])[1])[0]
    probs = torch.softmax(full, dim=-1)
    for pred in layers[1].top_tokens:
        assert pred.probability == pytest.approx(probs[pred.token_id].item(), rel=1e-4, abs=1e-6)
        assert pred.logit == pytest.approx(full[pred.token_id].item(), rel=1e-4, abs=1e-5)


def test_target_token_tracking(qwen_adapter):
    engine = LogitLensEngine(qwen_adapter)
    input_ids = qwen_adapter.tokenize("the capital of japan is")
    layers = engine.run(input_ids, positions=[4], top_k=3, target_token_id=6)[0]
    for layer in layers:
        assert layer.target is not None
        assert layer.target.token_id == 6
        assert layer.target.token == "tokyo"
        assert layer.target.rank >= 1
        in_topk = next((t for t in layer.top_tokens if t.token_id == 6), None)
        if in_topk is not None:
            assert in_topk.rank == layer.target.rank
            assert in_topk.probability == pytest.approx(layer.target.probability)
        else:
            assert layer.target.rank > 3

    # Tracking the layer's own top-1 must give rank 1.
    top1_id = layers[-1].top1.token_id
    tracked = engine.run(input_ids, positions=[4], top_k=3, target_token_id=top1_id)[0][-1]
    assert tracked.target.rank == 1
    assert tracked.target.probability == pytest.approx(tracked.top1.probability)


def test_multiple_positions_match_single_runs(qwen_adapter):
    engine = LogitLensEngine(qwen_adapter)
    input_ids = qwen_adapter.tokenize("the capital of japan is")
    multi = engine.run(input_ids, positions=[1, 3], top_k=4)
    single_1 = engine.run(input_ids, positions=[1], top_k=4)[0]
    single_3 = engine.run(input_ids, positions=[3], top_k=4)[0]
    assert len(multi) == 2
    for got, expected in zip(multi[0], single_1, strict=True):
        assert [t.token_id for t in got.top_tokens] == [t.token_id for t in expected.top_tokens]
    for got, expected in zip(multi[1], single_3, strict=True):
        assert [t.token_id for t in got.top_tokens] == [t.token_id for t in expected.top_tokens]


def test_run_validates_arguments(qwen_adapter):
    engine = LogitLensEngine(qwen_adapter)
    input_ids = qwen_adapter.tokenize("the capital")
    with pytest.raises(ValueError, match="top_k"):
        engine.run(input_ids, positions=[0], top_k=0)
    with pytest.raises(ValueError, match="positions"):
        engine.run(input_ids, positions=[], top_k=3)


def test_lens_mode_accepts_strings(qwen_adapter):
    assert LogitLensEngine(qwen_adapter, "raw").lens_mode is LensMode.RAW
    assert LogitLensEngine(qwen_adapter, "final-norm").lens_mode is LensMode.FINAL_NORM
    with pytest.raises(ValueError):
        LogitLensEngine(qwen_adapter, "layer-norm")
