from __future__ import annotations

import json

import pytest

from logit_lens_debugger import ComparisonResult, LensMode, LogitLensAnalyzer, LogitLensResult
from logit_lens_debugger.analyzer import normalize_index


def test_normalize_index():
    assert normalize_index(-1, 5) == 4
    assert normalize_index(0, 5) == 0
    assert normalize_index(4, 5) == 4
    with pytest.raises(IndexError):
        normalize_index(5, 5)
    with pytest.raises(IndexError):
        normalize_index(-6, 5)
    with pytest.raises(ValueError):
        normalize_index(0, 0)


def test_analyzer_requires_model_or_adapter():
    with pytest.raises(ValueError, match="model_name or an adapter"):
        LogitLensAnalyzer()


def test_analyze_basic_fields(analyzer):
    result = analyzer.analyze("the capital of japan is tokyo", token_index=4, top_k=5)
    assert isinstance(result, LogitLensResult)
    assert result.model_name == "tiny-qwen"
    assert result.lens_mode is LensMode.FINAL_NORM
    assert result.input_tokens == ["the", "capital", "of", "japan", "is", "tokyo"]
    assert result.input_token_ids == [1, 2, 3, 4, 5, 6]
    assert result.token_index == 4
    assert result.analyzed_token == "is"
    assert result.top_k == 5
    assert result.num_layers == 3
    assert len(result.layers) == 4
    assert len(result.trajectory()) == 4
    # The following token in the text is tracked automatically.
    assert result.target_token == "tokyo"
    assert result.target_token_id == 6
    assert all(layer.target is not None and layer.target.token_id == 6 for layer in result.layers)
    assert result.metadata["device"] == "cpu"


def test_analyze_negative_index_and_no_target_at_end(analyzer):
    result = analyzer.analyze("the capital of japan is", token_index=-1, top_k=3)
    assert result.token_index == 4
    assert result.target_token is None
    assert result.target_token_id is None
    assert all(layer.target is None for layer in result.layers)


def test_analyze_explicit_target(analyzer):
    by_str = analyzer.analyze("the capital of japan is", top_k=3, target_token="osaka")
    assert by_str.target_token_id == 7
    by_id = analyzer.analyze("the capital of japan is", top_k=3, target_token=7)
    assert by_id.target_token == "osaka"
    ranks_by_str = [layer.target.rank for layer in by_str.layers]
    ranks_by_id = [layer.target.rank for layer in by_id.layers]
    assert ranks_by_str == ranks_by_id


def test_analyze_multi_piece_target_warns(analyzer):
    with pytest.warns(UserWarning, match="tokenizes to 2 pieces"):
        result = analyzer.analyze("the capital of japan is", top_k=3, target_token="tokyo osaka")
    assert result.target_token_id == 6


def test_analyze_out_of_range_index(analyzer):
    with pytest.raises(IndexError):
        analyzer.analyze("the capital", token_index=7)


def test_lens_mode_switch(analyzer):
    normed = analyzer.analyze("the capital of japan is", top_k=3)
    analyzer.lens_mode = "raw"
    raw = analyzer.analyze("the capital of japan is", top_k=3)
    assert raw.lens_mode is LensMode.RAW
    assert normed.layers[-1].top1.logit != raw.layers[-1].top1.logit
    analyzer.lens_mode = LensMode.FINAL_NORM


def test_result_json_round_trip(sample_result, tmp_path):
    payload = json.loads(sample_result.to_json())
    assert payload["lens_mode"] == "final-norm"
    assert payload["layers"][0]["top_tokens"][0].keys() == {
        "token",
        "token_id",
        "logit",
        "probability",
        "rank",
    }
    restored = LogitLensResult.from_dict(payload)
    assert restored == sample_result

    path = sample_result.save_json(tmp_path / "nested" / "result.json")
    assert LogitLensResult.load_json(path) == sample_result


def test_compare_identical_texts(analyzer):
    comparison = analyzer.compare("the capital of japan is", "the capital of japan is", top_k=3)
    assert isinstance(comparison, ComparisonResult)
    assert comparison.top1_agreement == [True] * 4
    assert comparison.first_divergence_layer is None
    assert comparison.last_divergence_layer is None
    assert comparison.divergence_layers == []


def test_compare_different_texts(analyzer):
    comparison = analyzer.compare("the capital of japan is", "sales were 150 yen", top_k=3)
    assert len(comparison.top1_agreement) == 4
    assert comparison.result_a.text != comparison.result_b.text
    diverged = comparison.divergence_layers
    if diverged:
        assert comparison.first_divergence_layer == diverged[0]
        assert comparison.last_divergence_layer == diverged[-1]
    payload = json.loads(comparison.to_json())
    assert set(payload) == {
        "result_a",
        "result_b",
        "top1_agreement",
        "first_divergence_layer",
        "last_divergence_layer",
        "layers",
        "alignment",
        "kind",
        "metadata",
    }
    assert comparison.kind == "prompt"
    assert comparison.alignment == [(i, i) for i in range(len(comparison.top1_agreement))]
    assert len(comparison.layers) == len(comparison.top1_agreement)


def test_compare_identical_texts_kl_zero_everywhere(analyzer):
    comparison = analyzer.compare("the capital of japan is", "the capital of japan is", top_k=3)
    assert all(lc.topk_overlap == pytest.approx(1.0) for lc in comparison.layers)
    assert all(lc.kl_a_b == pytest.approx(0.0, abs=1e-5) for lc in comparison.layers)
    assert all(lc.kl_b_a == pytest.approx(0.0, abs=1e-5) for lc in comparison.layers)
    assert all(lc.js == pytest.approx(0.0, abs=1e-5) for lc in comparison.layers)


def test_compare_keep_log_probs_false_has_no_kl(analyzer):
    comparison = analyzer.compare(
        "the capital of japan is", "sales were 150 yen", top_k=3, keep_log_probs=False
    )
    assert all(lc.kl_a_b is None for lc in comparison.layers)
    assert all(lc.kl_b_a is None for lc in comparison.layers)
    assert all(lc.js is None for lc in comparison.layers)
    # top1/overlap/target fields are still computed without distributions.
    assert all(isinstance(lc.top1_same, bool) for lc in comparison.layers)


def test_compare_token_index_b_uses_different_position(analyzer):
    comparison = analyzer.compare(
        "the capital of japan is",
        "the capital of japan is tokyo",
        token_index=-1,
        token_index_b=-2,
        top_k=3,
    )
    assert comparison.result_a.token_index == 4
    assert comparison.result_b.token_index == 4
    assert comparison.result_a.analyzed_token == comparison.result_b.analyzed_token == "is"


def test_compare_common_target_token_populates_both_ranks(analyzer):
    comparison = analyzer.compare(
        "the capital of japan is", "sales were 150 yen", top_k=3, target_token="tokyo"
    )
    assert comparison.result_a.target_token_id == 6
    assert comparison.result_b.target_token_id == 6
    for lc in comparison.layers:
        assert lc.target_rank_a is not None
        assert lc.target_rank_b is not None


def test_analyze_equals_analyze_detailed_result(analyzer):
    result = analyzer.analyze("the capital of japan is tokyo", token_index=4, top_k=5)
    detailed_result, log_probs = analyzer._analyze_detailed(
        "the capital of japan is tokyo", token_index=4, top_k=5
    )
    assert result == detailed_result
    assert log_probs is None
