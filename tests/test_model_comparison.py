"""Tests for model-vs-model comparison: ``compare_models`` / ``compare_model_names``.

Uses the tiny Qwen (3 layers) / Llama (2 layers) fixtures from ``conftest.py``, which
share one ``ToyTokenizer`` instance (so ``vocab_identical`` is True between them), plus
``llama_adapter_other_vocab`` (a distinct ``ToyTokenizer`` over a reversed vocabulary,
same size, for the "vocabularies differ" path).
"""

from __future__ import annotations

import json

import pytest

from logit_lens_debugger import (
    ComparisonResult,
    LogitLensAnalyzer,
    align_layers,
    compare_model_names,
    compare_models,
)

TEXT = "the capital of japan is"


@pytest.fixture
def qwen_analyzer(qwen_adapter) -> LogitLensAnalyzer:
    return LogitLensAnalyzer(adapter=qwen_adapter)


@pytest.fixture
def llama_analyzer(llama_adapter) -> LogitLensAnalyzer:
    return LogitLensAnalyzer(adapter=llama_adapter)


@pytest.fixture
def llama_analyzer_other_vocab(llama_adapter_other_vocab) -> LogitLensAnalyzer:
    return LogitLensAnalyzer(adapter=llama_adapter_other_vocab)


# --------------------------------------------------------------------- compare_models


def test_compare_models_same_vocab(qwen_analyzer, llama_analyzer):
    cmp = compare_models(qwen_analyzer, llama_analyzer, TEXT)
    assert cmp.kind == "model"
    assert cmp.alignment == align_layers(3, 2)
    assert cmp.metadata["same_vocab"] is True
    assert all(lc.kl_a_b is not None for lc in cmp.layers)
    assert all(lc.kl_b_a is not None for lc in cmp.layers)
    assert all(lc.js is not None for lc in cmp.layers)


def test_compare_models_different_vocab(qwen_analyzer, llama_analyzer_other_vocab):
    cmp = compare_models(qwen_analyzer, llama_analyzer_other_vocab, TEXT)
    assert cmp.kind == "model"
    assert cmp.metadata["same_vocab"] is False
    assert all(lc.kl_a_b is None for lc in cmp.layers)
    assert all(lc.kl_b_a is None for lc in cmp.layers)
    assert all(lc.js is None for lc in cmp.layers)
    # topk_overlap/top1_same are still computed, by decoded text.
    assert all(0.0 <= lc.topk_overlap <= 1.0 for lc in cmp.layers)


def test_compare_models_char_index_sets_both_token_indices(qwen_analyzer, llama_analyzer):
    char_index = 4  # inside "capital"
    expected_a = qwen_analyzer.adapter.char_to_token(TEXT, char_index)
    expected_b = llama_analyzer.adapter.char_to_token(TEXT, char_index)

    cmp = compare_models(qwen_analyzer, llama_analyzer, TEXT, char_index=char_index)

    assert cmp.result_a.token_index == expected_a
    assert cmp.result_b.token_index == expected_b
    assert cmp.metadata["char_index"] == char_index
    assert cmp.metadata["token_index_a"] == expected_a
    assert cmp.metadata["token_index_b"] == expected_b


def test_compare_models_target_token_populates_both_ranks(qwen_analyzer, llama_analyzer):
    cmp = compare_models(qwen_analyzer, llama_analyzer, TEXT, target_token="tokyo")
    assert all(lc.target_rank_a is not None for lc in cmp.layers)
    assert all(lc.target_rank_b is not None for lc in cmp.layers)


def test_compare_models_int_target_token_rejected(qwen_analyzer, llama_analyzer):
    with pytest.raises(TypeError):
        compare_models(qwen_analyzer, llama_analyzer, TEXT, target_token=6)


def test_compare_models_keep_log_probs_false_disables_kl(qwen_analyzer, llama_analyzer):
    cmp = compare_models(qwen_analyzer, llama_analyzer, TEXT, keep_log_probs=False)
    assert cmp.metadata["same_vocab"] is True  # still same vocab...
    assert all(lc.kl_a_b is None for lc in cmp.layers)  # ...but no distributions were kept
    assert all(lc.kl_b_a is None for lc in cmp.layers)
    assert all(lc.js is None for lc in cmp.layers)


# ----------------------------------------------------------------- compare_model_names


@pytest.mark.parametrize("sequential", [True, False])
def test_compare_model_names_factory_order_and_layers(qwen_adapter, llama_adapter, sequential):
    calls: list[str] = []

    def factory(name: str) -> LogitLensAnalyzer:
        calls.append(name)
        adapter = {"a": qwen_adapter, "b": llama_adapter}[name]
        return LogitLensAnalyzer(adapter=adapter)

    cmp = compare_model_names("a", "b", TEXT, factory=factory, sequential=sequential)

    assert calls == ["a", "b"]
    assert cmp.kind == "model"
    assert cmp.alignment == align_layers(3, 2)
    assert len(cmp.layers) == len(align_layers(3, 2))


# --------------------------------------------------------------------------- JSON


def test_comparison_result_model_json_round_trip(qwen_analyzer, llama_analyzer):
    cmp = compare_models(qwen_analyzer, llama_analyzer, TEXT)
    payload = json.loads(cmp.to_json())
    restored = ComparisonResult.from_dict(payload)

    assert restored.kind == "model"
    assert restored.alignment == cmp.alignment
    assert restored.alignment == align_layers(3, 2)
    assert restored.alignment != [(i, i) for i in range(len(restored.alignment))]
    assert restored.layers == cmp.layers
