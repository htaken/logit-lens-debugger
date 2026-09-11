from __future__ import annotations

import math
import statistics

import pytest

from conftest import make_result
from logit_lens_debugger import InstabilityMetrics, compute_instability
from logit_lens_debugger.types import LayerResult, LensMode, LogitLensResult, TokenPrediction


def test_oscillating_sequence():
    result = make_result([150, 1500, 150, 1500])
    metrics = compute_instability(result, from_layer=0)
    assert metrics.top1_switches == 3
    assert metrics.oscillations == 2
    assert metrics.distinct_top1 == 2
    assert metrics.settle_layer == 3  # last (and only) layer index in the final run
    assert metrics.instability_score == 1.0  # clipped


def test_settles_early():
    result = make_result([7, 1, 1, 1, 1])
    metrics = compute_instability(result)  # default from_layer=1
    assert metrics.num_evaluated_layers == 4
    assert metrics.top1_switches == 0
    assert metrics.oscillations == 0
    assert metrics.settle_layer == 1
    assert metrics.instability_score == 0.0


def test_from_layer_changes_switch_count():
    result = make_result([7, 1, 1, 1, 1])
    including_layer0 = compute_instability(result, from_layer=0)
    excluding_layer0 = compute_instability(result, from_layer=1)
    assert including_layer0.top1_switches == 1  # 7 -> 1 counted
    assert excluding_layer0.top1_switches == 0


def test_from_layer_out_of_range_raises():
    result = make_result([7, 1, 1, 1, 1])
    with pytest.raises(ValueError):
        compute_instability(result, from_layer=-1)
    with pytest.raises(ValueError):
        compute_instability(result, from_layer=len(result.layers))


def test_late_fraction_out_of_range_raises():
    result = make_result([7, 1, 1, 1, 1])
    with pytest.raises(ValueError):
        compute_instability(result, late_fraction=1.5)
    with pytest.raises(ValueError):
        compute_instability(result, late_fraction=-0.1)


def test_late_switch_at_final_layer():
    # 25 layers (layer_index 0..24, i.e. a 24-decoder-block model) with a single
    # switch right at the last layer.
    top1_ids = [0] * 24 + [1]
    result = make_result(top1_ids)
    metrics = compute_instability(result, from_layer=1, late_fraction=1 / 3)
    assert result.num_layers == 24
    # late_start_layer = max(1, 24 - round(1/3 * 24) + 1) = max(1, 17) = 17
    assert metrics.late_start_layer == 17
    assert metrics.top1_switches == 1
    assert metrics.late_switches == 1


def test_target_metrics():
    ranks = [5, 3, 1, 1]
    result = make_result([10, 10, 10, 10], target_ranks=ranks, top_k=3)
    metrics = compute_instability(result, from_layer=0)
    assert metrics.target_first_top1_layer == 2
    assert metrics.target_first_topk_layer == 1  # top_k=3, first rank <= 3 is index 1
    assert metrics.target_final_rank == 1
    assert metrics.target_final_prob == pytest.approx(1.0)
    log_ranks = [math.log2(r) for r in ranks]
    assert metrics.target_rank_log_mean == pytest.approx(statistics.fmean(log_ranks))
    assert metrics.target_rank_log_std == pytest.approx(statistics.pstdev(log_ranks))


def test_no_target_gives_none_fields():
    result = make_result([1, 2, 3])
    metrics = compute_instability(result, from_layer=0)
    assert metrics.target_first_top1_layer is None
    assert metrics.target_first_topk_layer is None
    assert metrics.target_rank_log_mean is None
    assert metrics.target_rank_log_std is None
    assert metrics.target_final_rank is None
    assert metrics.target_final_prob is None


def test_target_none_on_some_layers_gives_none_fields():
    # target_token_id is set, but one evaluated layer has no target prediction.
    layers = [
        LayerResult(
            layer_index=0,
            top_tokens=[TokenPrediction(token="a", token_id=1, logit=1.0, probability=0.9, rank=1)],
            target=TokenPrediction(token="t", token_id=9, logit=0.1, probability=0.1, rank=5),
        ),
        LayerResult(
            layer_index=1,
            top_tokens=[TokenPrediction(token="a", token_id=1, logit=1.0, probability=0.9, rank=1)],
            target=None,
        ),
    ]
    result = LogitLensResult(
        model_name="synthetic",
        lens_mode=LensMode.FINAL_NORM,
        text="synthetic",
        input_tokens=["synthetic"],
        input_token_ids=[0],
        token_index=0,
        top_k=3,
        layers=layers,
        target_token="t",
        target_token_id=9,
    )
    metrics = compute_instability(result, from_layer=0)
    assert metrics.target_final_rank is None
    assert metrics.target_rank_log_mean is None


def test_to_dict_from_dict_round_trip():
    result = make_result([150, 1500, 150, 1500], target_ranks=[5, 3, 1, 1])
    metrics = compute_instability(result, from_layer=0)
    restored = InstabilityMetrics.from_dict(metrics.to_dict())
    assert restored == metrics


def test_result_instability_method_matches_function():
    result = make_result([150, 1500, 150, 1500], target_ranks=[5, 3, 1, 1])
    assert result.instability() == compute_instability(result)
    assert result.instability(from_layer=0) == compute_instability(result, from_layer=0)


def test_sample_result_invariants(sample_result):
    metrics = compute_instability(sample_result)
    assert 0.0 <= metrics.instability_score <= 1.0
    assert metrics.from_layer <= metrics.settle_layer <= sample_result.num_layers
