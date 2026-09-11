"""Tests for LogitLensAnalyzer.detect_anomalies (toy model, no anomaly-signal math)."""

from __future__ import annotations

import pytest

from conftest import VOCAB


def test_detect_anomalies_full_scan_novel_signal(analyzer):
    report = analyzer.detect_anomalies(
        "the capital of japan is", "tokyo osaka", source="the capital of japan is tokyo"
    )
    assert len(report.positions) == 2
    by_token = {p.token: p for p in report.positions}
    assert by_token["osaka"].signals["novel"] == 1.0
    assert by_token["tokyo"].signals["novel"] == 0.0


def test_detect_anomalies_only_diff(analyzer):
    report = analyzer.detect_anomalies(
        "sales were 150 yen",
        "sales were 1500 yen",
        source="sales were 150 yen",
        only_diff=True,
    )
    assert len(report.positions) == 1
    anomaly = report.positions[0]
    assert anomaly.token == "1500"
    assert anomaly.signals["in_diff"] == 1.0
    assert anomaly.signals["numeric"] == 1.0


def test_detect_anomalies_only_diff_without_source_raises(analyzer):
    with pytest.raises(ValueError, match="only_diff"):
        analyzer.detect_anomalies("the capital of japan is", "tokyo", only_diff=True)


def test_detect_anomalies_full_scan_attaches_diff_span(analyzer):
    report = analyzer.detect_anomalies(
        "sales were 150 yen", "sales were 1500 yen", source="sales were 150 yen"
    )
    by_token = {p.token: p for p in report.positions}
    assert by_token["1500"].diff_span is not None
    assert by_token["1500"].diff_span.source_text == "150"
    assert by_token["1500"].signals["in_diff"] == 1.0
    assert by_token["yen"].diff_span is None
    assert by_token["yen"].signals["in_diff"] == 0.0


def test_detect_anomalies_chunk_size_matches_full(analyzer):
    whole = analyzer.detect_anomalies("the capital of japan is", "tokyo osaka", chunk_size=None)
    chunked = analyzer.detect_anomalies("the capital of japan is", "tokyo osaka", chunk_size=1)

    whole_by_index = {p.token_index: p for p in whole.positions}
    chunked_by_index = {p.token_index: p for p in chunked.positions}
    assert set(whole_by_index) == set(chunked_by_index)
    for idx in whole_by_index:
        assert whole_by_index[idx].signals.keys() == chunked_by_index[idx].signals.keys()
        assert whole_by_index[idx].score == pytest.approx(
            chunked_by_index[idx].score, rel=1e-3, abs=1e-4
        )


def test_detect_anomalies_entity_spans(analyzer):
    report = analyzer.detect_anomalies(
        "the capital of japan is", "tokyo osaka", entity_spans=[(1, 2)]
    )
    by_token = {p.token: p for p in report.positions}
    assert by_token["tokyo"].signals["entity"] == 0.0
    assert by_token["osaka"].signals["entity"] == 1.0


def test_detect_anomalies_metadata_vocab_size(analyzer):
    report = analyzer.detect_anomalies("the capital of japan is", "tokyo osaka")
    assert report.metadata["vocab_size"] == len(VOCAB)
