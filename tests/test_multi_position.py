"""Tests for LogitLensAnalyzer.analyze_positions / analyze_generation / analyze_diff
and MultiPositionResult."""

from __future__ import annotations

import dataclasses
import json

import pytest

from logit_lens_debugger.types import MultiPositionResult


def _assert_layers_equivalent(layers_a, layers_b):
    """Same layer/token-id/rank structure, floats equal up to batched-matmul rounding.

    Running N positions together in one forward vs. one position alone hits PyTorch's
    CPU matmul kernel with a different batch size, which can shift float32 results by
    ~1e-7 in the last bits (a well-known non-associativity of batched BLAS calls, not a
    bug in this codebase -- see ``test_multiple_positions_match_single_runs`` in
    ``test_engine.py``, which compares token ids for the same reason). Token ids, ranks
    and layer indices must match exactly; probabilities/logits/entropy only approximately.
    """
    assert len(layers_a) == len(layers_b)
    for la, lb in zip(layers_a, layers_b, strict=True):
        assert la.layer_index == lb.layer_index
        assert [t.token_id for t in la.top_tokens] == [t.token_id for t in lb.top_tokens]
        for ta, tb in zip(la.top_tokens, lb.top_tokens, strict=True):
            assert ta.rank == tb.rank
            assert ta.logit == pytest.approx(tb.logit, rel=1e-3, abs=1e-4)
            assert ta.probability == pytest.approx(tb.probability, rel=1e-3, abs=1e-4)
        assert la.entropy == pytest.approx(lb.entropy, rel=1e-3, abs=1e-4)
        if la.target is None or lb.target is None:
            assert la.target is None and lb.target is None
        else:
            assert la.target.token_id == lb.target.token_id
            assert la.target.rank == lb.target.rank
            assert la.target.logit == pytest.approx(lb.target.logit, rel=1e-3, abs=1e-4)
            assert la.target.probability == pytest.approx(lb.target.probability, rel=1e-3, abs=1e-4)


def test_analyze_positions_matches_single_analyze(analyzer):
    text = "the capital of japan is tokyo"
    result = analyzer.analyze_positions(text, [2, 4], top_k=5)

    assert [p.token_index for p in result.positions] == [2, 4]
    for pos_result in result.positions:
        direct = analyzer.analyze(text, token_index=pos_result.token_index, top_k=5)
        _assert_layers_equivalent(pos_result.layers, direct.layers)
        assert pos_result.target_token == direct.target_token
        assert pos_result.target_token_id == direct.target_token_id

    to_results = result.to_results()
    assert len(to_results) == 2
    for single, pos_result in zip(to_results, result.positions, strict=True):
        direct = analyzer.analyze(text, token_index=pos_result.token_index, top_k=5)
        _assert_layers_equivalent(single.layers, direct.layers)
        assert single.target_token == direct.target_token
        assert single.input_tokens == direct.input_tokens
        assert single.text == direct.text


def test_analyze_positions_dedupes_and_sorts(analyzer):
    text = "the capital of japan is tokyo"
    result = analyzer.analyze_positions(text, [4, 2, 2, 4])
    assert [p.token_index for p in result.positions] == [2, 4]


def test_analyze_positions_empty_raises(analyzer):
    with pytest.raises(ValueError, match="positions"):
        analyzer.analyze_positions("the capital", [])


def test_analyze_positions_targets_length_mismatch_raises(analyzer):
    with pytest.raises(ValueError, match="targets"):
        analyzer.analyze_positions("the capital of japan is", [1, 2], targets=["osaka"])


def test_analyze_positions_explicit_targets(analyzer):
    result = analyzer.analyze_positions(
        "the capital of japan is", [1, 4], targets=["osaka", 7], top_k=3
    )
    assert [p.target_token_id for p in result.positions] == [7, 7]


def test_analyze_positions_chunking_matches_single_call(analyzer):
    text = "the capital of japan is tokyo"
    positions = [0, 1, 2, 3, 4, 5]
    whole = analyzer.analyze_positions(text, positions, top_k=4)
    chunked = analyzer.analyze_positions(text, positions, top_k=4, chunk_size=1)
    chunked_2 = analyzer.analyze_positions(text, positions, top_k=4, chunk_size=2)

    assert [p.token_index for p in whole.positions] == [p.token_index for p in chunked.positions]
    assert [p.token_index for p in whole.positions] == [p.token_index for p in chunked_2.positions]
    for one, two, three in zip(
        whole.positions, chunked.positions, chunked_2.positions, strict=True
    ):
        _assert_layers_equivalent(one.layers, two.layers)
        _assert_layers_equivalent(one.layers, three.layers)
        assert one.target_token_id == two.target_token_id == three.target_token_id


def test_multi_position_result_properties(analyzer):
    result = analyzer.analyze_positions("the capital of japan is tokyo", [2, 4], prompt_length=3)
    assert result.num_layers == analyzer.adapter.num_layers
    assert result.output_tokens == result.input_tokens[3:]


def test_multi_position_result_num_layers_empty_raises(analyzer):
    result = analyzer.analyze_positions("the capital of japan is tokyo", [2])
    empty = dataclasses.replace(result, positions=[])
    with pytest.raises(ValueError):
        _ = empty.num_layers


def test_analyze_generation_basic(analyzer):
    result = analyzer.analyze_generation("the capital of japan is", "tokyo osaka")
    assert result.prompt_length == 5
    assert [p.token_index for p in result.positions] == [4, 5]
    assert [p.target_token for p in result.positions] == ["tokyo", "osaka"]
    assert result.output_tokens == ["tokyo", "osaka"]


def test_analyze_generation_output_as_ids_matches_text(analyzer):
    by_text = analyzer.analyze_generation("the capital of japan is", "tokyo osaka")
    by_ids = analyzer.analyze_generation("the capital of japan is", [6, 7])
    assert dataclasses.replace(by_text, metadata={}) == dataclasses.replace(by_ids, metadata={})


def test_analyze_generation_explicit_positions(analyzer):
    result = analyzer.analyze_generation("the capital of japan is", "tokyo osaka", positions=[1])
    assert [p.token_index for p in result.positions] == [5]
    assert result.positions[0].target_token == "osaka"


def test_analyze_diff_single_span(analyzer):
    result = analyzer.analyze_diff("sales were 150 yen", "sales were 1500 yen")
    assert result.prompt_length == 4
    assert len(result.diff_spans) == 1
    assert len(result.positions) == 1
    pos = result.positions[0]
    assert pos.target_token == "1500"
    assert pos.diff_span is not None
    assert pos.diff_span.source_text == "150"
    assert pos.diff_span.output_text == "1500"


def test_analyze_diff_explicit_prompt_shifts_position(analyzer):
    default_result = analyzer.analyze_diff("sales were 150 yen", "sales were 1500 yen")
    explicit_result = analyzer.analyze_diff(
        "sales were 150 yen", "sales were 1500 yen", prompt="the capital"
    )
    assert explicit_result.prompt_length == 2
    assert explicit_result.positions[0].token_index != default_result.positions[0].token_index
    assert explicit_result.positions[0].target_token == "1500"


def test_analyze_diff_identical_text_skips_engine(analyzer, monkeypatch):
    def _fail(*args, **kwargs):
        raise AssertionError("engine.run must not be called for identical source/output")

    monkeypatch.setattr(analyzer.engine, "run", _fail)
    result = analyzer.analyze_diff("sales were 150 yen", "sales were 150 yen")
    assert result.positions == []
    assert result.diff_spans == []


def test_multi_position_result_json_round_trip(analyzer, tmp_path):
    result = analyzer.analyze_diff("sales were 150 yen", "sales were 1500 yen")
    payload = json.loads(result.to_json())
    restored = MultiPositionResult.from_dict(payload)
    assert restored == result

    path = result.save_json(tmp_path / "nested" / "diff.json")
    assert MultiPositionResult.load_json(path) == result
