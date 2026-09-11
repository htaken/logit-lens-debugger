"""Tests for logit_lens_debugger.anomaly -- pure functions, no model required."""

from __future__ import annotations

import dataclasses
import json
import math

import pytest

from conftest import make_multi_position_result, make_result, to_position_result
from logit_lens_debugger import (
    DEFAULT_ANOMALY_WEIGHTS,
    AnomalyConfig,
    AnomalyReport,
    detect_anomalies,
    score_position,
)
from logit_lens_debugger.types import DiffSpan, PositionResult


def _score(position, context, *, config=None, vocab_size=None, source_tokens=None, is_entity=False):
    return score_position(
        position,
        top_k=3,
        prompt_length=0,
        vocab_size=vocab_size,
        source_tokens=source_tokens,
        is_entity=is_entity,
        config=config or AnomalyConfig(),
        context_result=context,
    )


def test_lower_final_probability_gives_higher_score():
    confident = make_result([0, 0, 0, 0], target_ranks=[1, 1, 1, 1], top_k=3)
    unsure = make_result([0, 0, 0, 0], target_ranks=[1, 1, 1, 50], top_k=3)

    confident_anomaly = _score(to_position_result(confident), confident)
    unsure_anomaly = _score(to_position_result(unsure), unsure)

    assert unsure_anomaly.signals["low_final_prob"] > confident_anomaly.signals["low_final_prob"]
    assert unsure_anomaly.score > confident_anomaly.score
    assert 0.0 <= confident_anomaly.score <= 1.0
    assert 0.0 <= unsure_anomaly.score <= 1.0


def test_late_emergence_first_topk_layer_and_never_in_topk():
    emerges_late = make_result([0, 0, 0, 0, 0], target_ranks=[10, 10, 3, 3, 3], top_k=3)
    anomaly = _score(to_position_result(emerges_late), emerges_late)
    assert anomaly.signals["late_emergence"] == pytest.approx(2 / 4)

    never_in_topk = make_result([0, 0, 0, 0, 0], target_ranks=[10, 10, 10, 10, 10], top_k=3)
    never_anomaly = _score(to_position_result(never_in_topk), never_in_topk)
    assert never_anomaly.signals["late_emergence"] == 1.0


def test_final_rank_saturation():
    rank1 = make_result([0, 0], target_ranks=[1, 1], top_k=3)
    rank1024 = make_result([0, 0], target_ranks=[1, 1024], top_k=3)

    assert _score(to_position_result(rank1), rank1).signals["final_rank"] == 0.0
    assert _score(to_position_result(rank1024), rank1024).signals["final_rank"] == pytest.approx(
        1.0
    )


def test_flip_at_end_variants():
    full_flip = make_result([0, 1, 2, 3])
    half_flip = make_result([0, 0, 1, 1])
    no_flip = make_result([5, 5, 5, 5])

    assert _score(to_position_result(full_flip), full_flip).signals["flip_at_end"] == 1.0
    assert _score(to_position_result(half_flip), half_flip).signals["flip_at_end"] == 0.5
    assert _score(to_position_result(no_flip), no_flip).signals["flip_at_end"] == 0.0


def test_final_entropy_omitted_without_vocab_size_present_otherwise():
    context = make_result([0, 1, 2])
    layers = list(context.layers)
    layers[-1] = dataclasses.replace(layers[-1], entropy=1.5)
    context = dataclasses.replace(context, layers=layers)
    position = to_position_result(context)

    omitted = _score(position, context, vocab_size=None)
    assert "final_entropy" not in omitted.signals

    present = _score(position, context, vocab_size=20)
    assert present.signals["final_entropy"] == pytest.approx(1.5 / math.log(20))
    assert 0.0 <= present.signals["final_entropy"] <= 1.0


def test_numeric_signal():
    context = make_result([0, 1])
    numeric = _score(to_position_result(context, target_token="150"), context)
    alpha = _score(to_position_result(context, target_token="tokyo"), context)
    assert numeric.signals["numeric"] == 1.0
    assert alpha.signals["numeric"] == 0.0


def test_novel_signal():
    context = make_result([0, 1])
    source_tokens = ["the", "capital", "of", "japan", "is", "tokyo"]

    novel = _score(
        to_position_result(context, target_token="osaka"), context, source_tokens=source_tokens
    )
    known = _score(
        to_position_result(context, target_token="tokyo"), context, source_tokens=source_tokens
    )
    no_source = _score(to_position_result(context, target_token="osaka"), context)

    assert novel.signals["novel"] == 1.0
    assert known.signals["novel"] == 0.0
    assert "novel" not in no_source.signals


def test_single_kana_token_is_not_novel():
    context = make_result([0, 1])
    anomaly = _score(
        to_position_result(context, target_token="は"), context, source_tokens=["まったく別の単語"]
    )
    assert anomaly.signals["novel"] == 0.0


def test_entity_like_signal():
    context = make_result([0, 1])
    english_source = ["mr", "smith", "said", "hello"]

    capitalized = _score(
        to_position_result(context, target_token="Tanaka"), context, source_tokens=english_source
    )
    kanji = _score(
        to_position_result(context, target_token="日本"),
        context,
        source_tokens=["まったく別の単語"],
    )
    lowercase = _score(
        to_position_result(context, target_token="foobar"), context, source_tokens=english_source
    )

    assert capitalized.signals["novel"] == 1.0
    assert capitalized.signals["entity_like"] == 1.0
    assert kanji.signals["novel"] == 1.0
    assert kanji.signals["entity_like"] == 1.0
    assert lowercase.signals["novel"] == 1.0
    assert lowercase.signals["entity_like"] == 0.0


def test_entity_signal_via_entity_spans():
    context_a = dataclasses.replace(make_result([0, 1]), token_index=0)
    context_b = dataclasses.replace(make_result([0, 1]), token_index=1)
    pos_a = PositionResult(token_index=0, layers=context_a.layers, target_token=None)
    pos_b = PositionResult(token_index=1, layers=context_b.layers, target_token=None)

    result = make_multi_position_result([pos_a, pos_b], prompt_length=0, top_k=3)
    report = detect_anomalies(result, entity_spans=[(2, 3)])

    by_index = {p.token_index: p for p in report.positions}
    assert by_index[0].signals["entity"] == 0.0
    assert by_index[1].signals["entity"] == 1.0


def test_zero_weight_removes_contribution():
    context = make_result([0, 1])
    config = AnomalyConfig(weights={**DEFAULT_ANOMALY_WEIGHTS, "numeric": 0.0})

    numeric = _score(to_position_result(context, target_token="150"), context, config=config)
    alpha = _score(to_position_result(context, target_token="abc"), context, config=config)

    assert numeric.signals["numeric"] == 1.0
    assert alpha.signals["numeric"] == 0.0
    assert numeric.score == pytest.approx(alpha.score)


def test_equal_weights_all_signals_one_gives_score_one():
    context = make_result([0, 1], target_ranks=[5, 1024], top_k=3)
    config = AnomalyConfig(weights={"final_rank": 1.0, "flip_at_end": 1.0, "entity": 1.0})
    anomaly = _score(to_position_result(context), context, config=config, is_entity=True)

    assert anomaly.signals["final_rank"] == pytest.approx(1.0)
    assert anomaly.signals["flip_at_end"] == 1.0
    assert anomaly.signals["entity"] == 1.0
    assert anomaly.score == pytest.approx(1.0)


def test_score_normalizes_over_available_signals_only():
    context = make_result([0, 1, 2], target_ranks=[5, 3, 2], top_k=3)
    position = PositionResult(token_index=0, layers=context.layers, target_token=None)
    config = AnomalyConfig()
    anomaly = _score(position, context, config=config)

    assert set(anomaly.signals) == {
        "low_final_prob",
        "final_rank",
        "late_emergence",
        "instability",
        "flip_at_end",
        "entity",
    }
    numerator = sum(config.weights[name] * value for name, value in anomaly.signals.items())
    denominator = sum(config.weights[name] for name in anomaly.signals)
    assert anomaly.score == pytest.approx(numerator / denominator)
    assert 0.0 <= anomaly.score <= 1.0


def test_instability_none_when_compute_instability_raises():
    context = make_result([0, 0], target_ranks=[1, 1], top_k=3)  # only 2 layers total
    config = AnomalyConfig(from_layer=5)  # out of range -> compute_instability raises ValueError
    anomaly = _score(to_position_result(context), context, config=config)

    assert anomaly.instability is None
    assert "instability" not in anomaly.signals
    assert "late_emergence" not in anomaly.signals
    assert "low_final_prob" in anomaly.signals
    assert "final_rank" in anomaly.signals


def test_flags_follow_default_weight_order_and_threshold():
    context = make_result([0, 1], target_ranks=[5, 1], top_k=3)
    position = PositionResult(token_index=0, layers=context.layers, target_token=None)
    anomaly = _score(position, context, is_entity=True)

    assert anomaly.flags == ["late_emergence", "flip_at_end", "entity"]
    assert list(DEFAULT_ANOMALY_WEIGHTS).index("late_emergence") < list(
        DEFAULT_ANOMALY_WEIGHTS
    ).index("flip_at_end")


def _build_scored_report() -> AnomalyReport:
    low = make_result([0, 0], target_ranks=[1, 1], top_k=3)
    high_a = make_result([0, 1], target_ranks=[1, 50], top_k=3)
    high_b = make_result([0, 1], target_ranks=[1, 50], top_k=3)

    pos_low = PositionResult(token_index=5, layers=low.layers, target_token=None)
    pos_high_a = PositionResult(token_index=2, layers=high_a.layers, target_token=None)
    pos_high_b = PositionResult(token_index=1, layers=high_b.layers, target_token=None)

    result = make_multi_position_result([pos_low, pos_high_a, pos_high_b], prompt_length=0, top_k=3)
    return detect_anomalies(result)


def test_report_sorted_by_score_desc_then_token_index():
    report = _build_scored_report()
    assert [p.token_index for p in report.positions] == [1, 2, 5]
    scores = [p.score for p in report.positions]
    assert scores[0] == pytest.approx(scores[1])
    assert scores[1] > scores[2]


def test_flagged_and_top_helpers():
    report = _build_scored_report()
    assert [p.token_index for p in report.flagged(0.5)] == [1, 2]
    assert report.flagged(0.99) == []
    assert report.flagged() == report.flagged(report.config.score_threshold)
    assert [p.token_index for p in report.top(1)] == [1]
    assert report.top(10) == report.positions


def test_detect_anomalies_empty_positions_gives_empty_report():
    result = make_multi_position_result([], prompt_length=0, top_k=3)
    report = detect_anomalies(result)
    assert report.positions == []


def test_anomaly_config_json_round_trip():
    config = AnomalyConfig(
        weights={"numeric": 1.0},
        flag_threshold=0.3,
        score_threshold=0.4,
        from_layer=0,
        late_fraction=0.5,
        rank_saturation_log2=8.0,
    )
    restored = AnomalyConfig.from_dict(json.loads(json.dumps(config.to_dict())))
    assert restored == config


def test_anomaly_report_json_round_trip(tmp_path):
    span = DiffSpan(
        kind="replace", source_range=(0, 1), output_range=(0, 1), source_text="a", output_text="b"
    )
    context = make_result([0, 1, 2], target_ranks=[5, 3, 1], top_k=3)
    position = PositionResult(
        token_index=3, layers=context.layers, target_token="b", target_token_id=99, diff_span=span
    )
    result = make_multi_position_result(
        [position], prompt_length=3, top_k=3, metadata={"vocab_size": 30}
    )
    report = detect_anomalies(result, source_tokens=["a"])

    payload = json.loads(report.to_json())
    restored = AnomalyReport.from_dict(payload)
    assert restored == report

    path = report.save_json(tmp_path / "nested" / "report.json")
    assert AnomalyReport.load_json(path) == report
