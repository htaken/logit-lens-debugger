"""Layer instability metrics: how much the top-1 prediction moves across layers.

Pure stdlib + :mod:`.types` -- no torch/transformers import -- so a result loaded
from JSON can be scored without reloading the model.
"""

from __future__ import annotations

import math
import statistics

from .types import InstabilityMetrics, LogitLensResult


def compute_instability(
    result: LogitLensResult,
    from_layer: int = 1,
    late_fraction: float = 1 / 3,
) -> InstabilityMetrics:
    """Summarize how much the top-1 token switches across ``result.layers[from_layer:]``.

    Layer 0 is excluded by default: it is the embedding output, where the input
    token itself is trivially top-1 and carries no signal. The resulting score is a
    heuristic for prioritizing which positions to inspect further, not evidence of a
    cause -- early/mid layers of small models are unstable by nature.
    """
    if from_layer < 0 or from_layer >= len(result.layers):
        raise ValueError(f"from_layer {from_layer} is out of range for {len(result.layers)} layers")
    if not 0 <= late_fraction <= 1:
        raise ValueError(f"late_fraction must be within [0, 1], got {late_fraction}")

    evaluated = result.layers[from_layer:]
    num_evaluated_layers = len(evaluated)
    top1_ids = [layer.top1.token_id for layer in evaluated]
    final_id = top1_ids[-1]

    late_start_layer = max(
        from_layer, result.num_layers - round(late_fraction * result.num_layers) + 1
    )

    top1_switches = 0
    oscillations = 0
    late_switches = 0
    seen = {top1_ids[0]}
    for i in range(1, num_evaluated_layers):
        if top1_ids[i] != top1_ids[i - 1]:
            top1_switches += 1
            if top1_ids[i] in seen:
                oscillations += 1
            if evaluated[i].layer_index >= late_start_layer:
                late_switches += 1
        seen.add(top1_ids[i])

    settle_layer = evaluated[-1].layer_index
    for i in range(num_evaluated_layers - 1, -1, -1):
        if top1_ids[i] != final_id:
            break
        settle_layer = evaluated[i].layer_index

    instability_score = min(
        1.0, (top1_switches + 0.5 * oscillations) / max(1, num_evaluated_layers - 1)
    )

    target_first_top1_layer = None
    target_first_topk_layer = None
    target_rank_log_mean = None
    target_rank_log_std = None
    target_final_rank = None
    target_final_prob = None
    has_target = result.target_token_id is not None and all(
        layer.target is not None for layer in evaluated
    )
    if has_target:
        log_ranks = [math.log2(layer.target.rank) for layer in evaluated]
        target_rank_log_mean = statistics.fmean(log_ranks)
        target_rank_log_std = statistics.pstdev(log_ranks)
        target_final_rank = evaluated[-1].target.rank
        target_final_prob = evaluated[-1].target.probability
        for layer in evaluated:
            if layer.target.rank == 1:
                target_first_top1_layer = layer.layer_index
                break
        for layer in evaluated:
            if layer.target.rank <= result.top_k:
                target_first_topk_layer = layer.layer_index
                break

    return InstabilityMetrics(
        from_layer=from_layer,
        late_fraction=late_fraction,
        num_evaluated_layers=num_evaluated_layers,
        top1_switches=top1_switches,
        oscillations=oscillations,
        distinct_top1=len(set(top1_ids)),
        settle_layer=settle_layer,
        late_start_layer=late_start_layer,
        late_switches=late_switches,
        instability_score=instability_score,
        target_first_top1_layer=target_first_top1_layer,
        target_first_topk_layer=target_first_topk_layer,
        target_rank_log_mean=target_rank_log_mean,
        target_rank_log_std=target_rank_log_std,
        target_final_rank=target_final_rank,
        target_final_prob=target_final_prob,
    )
