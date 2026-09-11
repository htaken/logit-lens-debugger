from __future__ import annotations

import json
import math

import pytest
import torch

from logit_lens_debugger import ComparisonResult, LayerComparison, LogitLensAnalyzer
from logit_lens_debugger.comparison import (
    align_layers,
    compare_layer,
    compare_results,
    js_divergence,
    kl_divergence,
    topk_overlap,
    vocab_identical,
)
from logit_lens_debugger.types import LayerResult, TokenPrediction


def _pred(
    token_id: int, token: str | None = None, rank: int = 1, prob: float = 0.5
) -> TokenPrediction:
    return TokenPrediction(
        token=token if token is not None else f"t{token_id}",
        token_id=token_id,
        logit=0.0,
        probability=prob,
        rank=rank,
    )


def _layer(
    layer_index: int, top_ids: list[int], target: TokenPrediction | None = None
) -> LayerResult:
    top_tokens = [
        _pred(tid, rank=i + 1, prob=max(0.01, 1.0 - 0.1 * i)) for i, tid in enumerate(top_ids)
    ]
    return LayerResult(layer_index=layer_index, top_tokens=top_tokens, target=target, entropy=0.0)


# --------------------------------------------------------------------------- kl / js


def test_kl_divergence_self_is_zero():
    log_p = torch.log_softmax(torch.tensor([1.0, 2.0, -1.0, 0.5]), dim=-1)
    assert kl_divergence(log_p, log_p) == pytest.approx(0.0, abs=1e-6)


def test_kl_divergence_matches_hand_computation():
    log_p = torch.log_softmax(torch.tensor([1.0, 2.0, 0.0]), dim=-1)
    log_q = torch.log_softmax(torch.tensor([0.5, 0.1, 2.0]), dim=-1)
    expected = sum(
        math.exp(p.item()) * (p.item() - q.item()) for p, q in zip(log_p, log_q, strict=True)
    )
    assert kl_divergence(log_p, log_q) == pytest.approx(max(0.0, expected), rel=1e-5)


def test_kl_divergence_non_negative():
    torch.manual_seed(0)
    for _ in range(5):
        log_p = torch.log_softmax(torch.randn(20), dim=-1)
        log_q = torch.log_softmax(torch.randn(20), dim=-1)
        assert kl_divergence(log_p, log_q) >= 0.0


def test_js_divergence_symmetric_and_bounded():
    torch.manual_seed(1)
    log_p = torch.log_softmax(torch.randn(15), dim=-1)
    log_q = torch.log_softmax(torch.randn(15), dim=-1)
    js_pq = js_divergence(log_p, log_q)
    js_qp = js_divergence(log_q, log_p)
    assert js_pq == pytest.approx(js_qp, rel=1e-6)
    assert 0.0 <= js_pq <= math.log(2) + 1e-6


# --------------------------------------------------------------------- topk_overlap


def test_topk_overlap_by_id():
    a = _layer(0, [1, 2, 3])
    b = _layer(0, [2, 3, 4])
    assert topk_overlap(a, b, compare_by="token_id") == pytest.approx(2 / 4)


def test_topk_overlap_by_text_ignores_ids():
    a = LayerResult(layer_index=0, top_tokens=[_pred(1, token="x"), _pred(2, token="y")])
    b = LayerResult(layer_index=0, top_tokens=[_pred(101, token="x"), _pred(202, token="y")])
    assert topk_overlap(a, b, compare_by="token_id") == 0.0
    assert topk_overlap(a, b, compare_by="text") == 1.0


def test_topk_overlap_both_empty_is_zero():
    empty = LayerResult(layer_index=0, top_tokens=[])
    assert topk_overlap(empty, empty) == 0.0


def test_topk_overlap_rejects_bad_compare_by():
    a = _layer(0, [1])
    with pytest.raises(ValueError, match="compare_by"):
        topk_overlap(a, a, compare_by="nonsense")


# ----------------------------------------------------------------------- compare_layer


def test_compare_layer_kl_none_when_tensor_missing():
    a = _layer(1, [1, 2])
    b = _layer(1, [1, 3])
    log_q = torch.log_softmax(torch.randn(10), dim=-1)
    lc = compare_layer(a, b, None, log_q)
    assert lc.kl_a_b is None
    assert lc.kl_b_a is None
    assert lc.js is None
    assert lc.top1_same is True  # both top-1 are token_id 1


def test_compare_layer_kl_none_when_lengths_differ():
    a = _layer(1, [1, 2])
    b = _layer(1, [1, 3])
    log_p = torch.log_softmax(torch.randn(10), dim=-1)
    short = torch.log_softmax(torch.randn(5), dim=-1)
    lc = compare_layer(a, b, log_p, short)
    assert lc.kl_a_b is None
    assert lc.kl_b_a is None
    assert lc.js is None


def test_compare_layer_kl_present_when_both_given():
    a = _layer(1, [1, 2])
    b = _layer(1, [3, 4])
    log_p = torch.log_softmax(torch.randn(10), dim=-1)
    log_q = torch.log_softmax(torch.randn(10), dim=-1)
    lc = compare_layer(a, b, log_p, log_q)
    assert lc.kl_a_b is not None
    assert lc.kl_b_a is not None
    assert lc.js is not None
    assert lc.top1_same is False


def test_compare_layer_target_deltas():
    target_a = _pred(99, rank=2, prob=0.4)
    target_b = _pred(99, rank=8, prob=0.05)
    a = _layer(0, [1, 2], target=target_a)
    b = _layer(0, [1, 2], target=target_b)
    lc = compare_layer(a, b, None, None)
    assert lc.target_rank_a == 2
    assert lc.target_rank_b == 8
    assert lc.target_rank_log_delta == pytest.approx(math.log2(8) - math.log2(2))
    assert lc.target_logprob_delta == pytest.approx(math.log(0.05) - math.log(0.4))


def test_compare_layer_target_delta_none_without_both_targets():
    a = _layer(0, [1, 2], target=_pred(99, rank=2, prob=0.4))
    b = _layer(0, [1, 2], target=None)
    lc = compare_layer(a, b, None, None)
    assert lc.target_rank_b is None
    assert lc.target_rank_log_delta is None
    assert lc.target_logprob_delta is None


# --------------------------------------------------------------------- compare_results


def test_compare_results_identical(analyzer):
    result_a, log_probs_a = analyzer._analyze_detailed(
        "the capital of japan is", top_k=3, keep_log_probs=True
    )
    result_b, log_probs_b = analyzer._analyze_detailed(
        "the capital of japan is", top_k=3, keep_log_probs=True
    )
    cmp = compare_results(result_a, result_b, log_probs_a[:, 0, :], log_probs_b[:, 0, :])

    assert all(cmp.top1_agreement)
    assert cmp.first_divergence_layer is None
    assert cmp.last_divergence_layer is None
    assert cmp.kind == "prompt"
    assert cmp.alignment == [(i, i) for i in range(len(result_a.layers))]
    assert all(lc.topk_overlap == pytest.approx(1.0) for lc in cmp.layers)
    assert all(lc.kl_a_b == pytest.approx(0.0, abs=1e-5) for lc in cmp.layers)
    assert all(lc.kl_b_a == pytest.approx(0.0, abs=1e-5) for lc in cmp.layers)
    assert all(lc.js == pytest.approx(0.0, abs=1e-5) for lc in cmp.layers)


def test_compare_results_mismatched_layers_without_alignment_raises(analyzer, llama_adapter):
    llama_analyzer = LogitLensAnalyzer(adapter=llama_adapter)
    result_a = analyzer.analyze("the capital of japan is", top_k=3)
    result_b = llama_analyzer.analyze("the capital of japan is", top_k=3)
    assert len(result_a.layers) != len(result_b.layers)
    with pytest.raises(ValueError, match="alignment"):
        compare_results(result_a, result_b)


def test_compare_results_explicit_alignment(analyzer, llama_adapter):
    llama_analyzer = LogitLensAnalyzer(adapter=llama_adapter)
    result_a = analyzer.analyze("the capital of japan is", top_k=3)  # 4 layers: 0..3
    result_b = llama_analyzer.analyze("the capital of japan is", top_k=3)  # 3 layers: 0..2
    alignment = [(0, 0), (1, 1), (2, 1), (3, 2)]

    cmp = compare_results(result_a, result_b, alignment=alignment, compare_by="text", kind="model")
    assert cmp.alignment == alignment
    assert cmp.kind == "model"
    assert [lc.layer_a for lc in cmp.layers] == [0, 1, 2, 3]
    assert [lc.layer_b for lc in cmp.layers] == [0, 1, 1, 2]
    assert len(cmp.top1_agreement) == 4


# ------------------------------------------------------- first_layer_* helpers


def _comparison_with_layers(layers: list[LayerComparison], sample_result) -> ComparisonResult:
    return ComparisonResult(
        result_a=sample_result,
        result_b=sample_result,
        top1_agreement=[lc.top1_same for lc in layers],
        first_divergence_layer=next((lc.layer_a for lc in layers if not lc.top1_same), None),
        last_divergence_layer=next(
            (lc.layer_a for lc in reversed(layers) if not lc.top1_same), None
        ),
        layers=layers,
        alignment=[(lc.layer_a, lc.layer_b) for lc in layers],
    )


def test_first_layer_kl_above_each_direction(sample_result):
    layers = [
        LayerComparison(
            layer_a=0,
            layer_b=0,
            top1_same=True,
            topk_overlap=1.0,
            kl_a_b=0.01,
            kl_b_a=0.02,
            js=0.005,
            target_rank_a=1,
            target_rank_b=1,
            target_prob_a=0.9,
            target_prob_b=0.9,
            target_rank_log_delta=0.0,
            target_logprob_delta=0.0,
        ),
        LayerComparison(
            layer_a=1,
            layer_b=1,
            top1_same=False,
            topk_overlap=0.5,
            kl_a_b=0.2,
            kl_b_a=0.05,
            js=0.15,
            target_rank_a=1,
            target_rank_b=3,
            target_prob_a=0.8,
            target_prob_b=0.3,
            target_rank_log_delta=math.log2(3),
            target_logprob_delta=math.log(0.3) - math.log(0.8),
        ),
        LayerComparison(
            layer_a=2,
            layer_b=2,
            top1_same=True,
            topk_overlap=1.0,
            kl_a_b=None,
            kl_b_a=None,
            js=None,
            target_rank_a=3,
            target_rank_b=3,
            target_prob_a=0.3,
            target_prob_b=0.3,
            target_rank_log_delta=0.0,
            target_logprob_delta=0.0,
        ),
    ]
    cmp = _comparison_with_layers(layers, sample_result)

    assert cmp.first_layer_kl_above(0.1, direction="a_b") == 1
    assert cmp.first_layer_kl_above(0.1, direction="b_a") is None
    assert cmp.first_layer_kl_above(0.1, direction="js") == 1
    assert cmp.first_layer_kl_above(0.01, direction="max") == 0
    assert cmp.first_layer_kl_above(999.0, direction="js") is None
    with pytest.raises(ValueError, match="direction"):
        cmp.first_layer_kl_above(0.1, direction="bogus")

    assert cmp.first_layer_target_rank_differs() == 1


def test_first_layer_kl_above_no_distributions(sample_result):
    layers = [
        LayerComparison(
            layer_a=i,
            layer_b=i,
            top1_same=True,
            topk_overlap=1.0,
            kl_a_b=None,
            kl_b_a=None,
            js=None,
            target_rank_a=None,
            target_rank_b=None,
            target_prob_a=None,
            target_prob_b=None,
            target_rank_log_delta=None,
            target_logprob_delta=None,
        )
        for i in range(3)
    ]
    cmp = _comparison_with_layers(layers, sample_result)
    assert cmp.first_layer_kl_above(0.0, direction="js") is None
    assert cmp.first_layer_target_rank_differs() is None


# --------------------------------------------------------------------------- JSON


def test_comparison_result_json_round_trip(analyzer):
    cmp = analyzer.compare("the capital of japan is", "sales were 150 yen", top_k=3)
    payload = json.loads(cmp.to_json())
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
    restored = ComparisonResult.from_dict(payload)
    assert restored.kind == cmp.kind
    assert restored.alignment == cmp.alignment
    assert restored.layers == cmp.layers
    assert restored.top1_agreement == cmp.top1_agreement


def test_comparison_result_save_and_load_json(analyzer, tmp_path):
    cmp = analyzer.compare("the capital of japan is", "sales were 150 yen", top_k=3)
    path = cmp.save_json(tmp_path / "nested" / "cmp.json")
    loaded = ComparisonResult.load_json(path)
    assert loaded.layers == cmp.layers
    assert loaded.alignment == cmp.alignment


def test_comparison_result_from_dict_legacy(sample_result):
    legacy = {
        "result_a": sample_result.to_dict(),
        "result_b": sample_result.to_dict(),
        "top1_agreement": [True, True],
        "first_divergence_layer": None,
        "last_divergence_layer": None,
    }
    restored = ComparisonResult.from_dict(legacy)
    assert restored.layers == []
    assert restored.alignment == []
    assert restored.kind == "prompt"
    assert restored.metadata == {}


# --------------------------------------------------------------------------- align_layers


def test_align_layers_more_a_than_b():
    assert align_layers(3, 2) == [(0, 0), (1, 1), (2, 1), (3, 2)]


def test_align_layers_more_b_than_a():
    aligned = align_layers(2, 3)
    assert aligned[0] == (0, 0)
    assert aligned[-1] == (2, 3)
    layer_bs = [b for _, b in aligned]
    assert layer_bs == sorted(layer_bs)  # monotone non-decreasing


def test_align_layers_equal_counts_is_identity():
    assert align_layers(4, 4) == [(i, i) for i in range(5)]


def test_align_layers_zero_a():
    assert align_layers(0, 5) == [(0, 0)]


@pytest.mark.parametrize(("num_a", "num_b"), [(1, 1), (3, 7), (7, 3), (5, 1), (1, 5), (10, 10)])
def test_align_layers_endpoints_and_monotone(num_a, num_b):
    aligned = align_layers(num_a, num_b)
    assert len(aligned) == num_a + 1
    assert aligned[0] == (0, 0)
    assert aligned[-1] == (num_a, num_b)
    assert [a for a, _ in aligned] == list(range(num_a + 1))
    layer_bs = [b for _, b in aligned]
    assert layer_bs == sorted(layer_bs)


# ------------------------------------------------------------------------ vocab_identical


def test_vocab_identical_same_tokenizer_object(qwen_adapter, llama_adapter):
    assert vocab_identical(qwen_adapter, llama_adapter) is True


def test_vocab_identical_false_for_different_vocab(qwen_adapter, llama_adapter_other_vocab):
    assert vocab_identical(qwen_adapter, llama_adapter_other_vocab) is False
