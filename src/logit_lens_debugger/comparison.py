"""Layer-wise comparison between two lens runs: KL/JS divergence, top-k overlap, target rank deltas.

Besides ``engine.py``/``analyzer.py``/``adapters/``, this is the only module that
imports torch: KL/JS need the full-vocabulary log-probability tensors produced by
:meth:`~.engine.LogitLensEngine.run_detailed` (``keep_log_probs=True``). Everything
else here works from :class:`~.types.LogitLensResult`/:class:`~.types.LayerResult`
alone, so a comparison without distributions (e.g. two results loaded from JSON) still
gives ``top1_same``/``topk_overlap``/target-rank fields, just with ``kl_*``/``js`` as
``None``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch

from .adapters.base import ModelAdapter
from .types import ComparisonResult, LayerComparison, LayerResult, LogitLensResult


def kl_divergence(log_p: torch.Tensor, log_q: torch.Tensor) -> float:
    """KL(P || Q) in nats, from log-probability vectors. Clamped to be >= 0."""
    value = torch.sum(torch.exp(log_p) * (log_p - log_q)).item()
    return max(0.0, value)


def js_divergence(log_p: torch.Tensor, log_q: torch.Tensor) -> float:
    """Jensen-Shannon divergence in nats, symmetric and in ``[0, ln 2]``."""
    log_m = torch.logaddexp(log_p, log_q) - math.log(2.0)
    return 0.5 * kl_divergence(log_p, log_m) + 0.5 * kl_divergence(log_q, log_m)


def topk_overlap(a: LayerResult, b: LayerResult, compare_by: str = "token_id") -> float:
    """Jaccard index of the two layers' top-k sets, by ``token_id`` or decoded ``text``.

    ``0.0`` if both top-k sets are empty (rather than an undefined ``0/0``).
    """
    if compare_by == "token_id":
        set_a = {t.token_id for t in a.top_tokens}
        set_b = {t.token_id for t in b.top_tokens}
    elif compare_by == "text":
        set_a = {t.token for t in a.top_tokens}
        set_b = {t.token for t in b.top_tokens}
    else:
        raise ValueError(f"compare_by must be 'token_id' or 'text', got {compare_by!r}")
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)


def compare_layer(
    a: LayerResult,
    b: LayerResult,
    log_probs_a: torch.Tensor | None,
    log_probs_b: torch.Tensor | None,
    compare_by: str = "token_id",
) -> LayerComparison:
    """Compare one (aligned) pair of layers between two lens runs.

    ``log_probs_a``/``log_probs_b`` are full-vocabulary log-probabilities for this
    single layer/position (``[V]``); KL/JS are computed only when both are given and
    have the same length, otherwise they are ``None``. ``compare_by`` selects whether
    ``top1_same`` and ``topk_overlap`` compare token ids or decoded token strings (the
    latter is needed once vocabularies differ, e.g. model comparison).
    """
    if compare_by == "token_id":
        top1_same = a.top1.token_id == b.top1.token_id
    elif compare_by == "text":
        top1_same = a.top1.token == b.top1.token
    else:
        raise ValueError(f"compare_by must be 'token_id' or 'text', got {compare_by!r}")

    overlap = topk_overlap(a, b, compare_by=compare_by)

    kl_a_b = kl_b_a = js = None
    if (
        log_probs_a is not None
        and log_probs_b is not None
        and log_probs_a.shape[-1] == log_probs_b.shape[-1]
    ):
        kl_a_b = kl_divergence(log_probs_a, log_probs_b)
        kl_b_a = kl_divergence(log_probs_b, log_probs_a)
        js = js_divergence(log_probs_a, log_probs_b)

    target_rank_a = a.target.rank if a.target is not None else None
    target_rank_b = b.target.rank if b.target is not None else None
    target_prob_a = a.target.probability if a.target is not None else None
    target_prob_b = b.target.probability if b.target is not None else None

    target_rank_log_delta = None
    if target_rank_a is not None and target_rank_b is not None:
        target_rank_log_delta = math.log2(target_rank_b) - math.log2(target_rank_a)

    target_logprob_delta = None
    if target_prob_a is not None and target_prob_b is not None:
        target_logprob_delta = math.log(max(target_prob_b, 1e-12)) - math.log(
            max(target_prob_a, 1e-12)
        )

    return LayerComparison(
        layer_a=a.layer_index,
        layer_b=b.layer_index,
        top1_same=top1_same,
        topk_overlap=overlap,
        kl_a_b=kl_a_b,
        kl_b_a=kl_b_a,
        js=js,
        target_rank_a=target_rank_a,
        target_rank_b=target_rank_b,
        target_prob_a=target_prob_a,
        target_prob_b=target_prob_b,
        target_rank_log_delta=target_rank_log_delta,
        target_logprob_delta=target_logprob_delta,
    )


def compare_results(
    result_a: LogitLensResult,
    result_b: LogitLensResult,
    log_probs_a: torch.Tensor | None = None,
    log_probs_b: torch.Tensor | None = None,
    alignment: Sequence[tuple[int, int]] | None = None,
    compare_by: str = "token_id",
    kind: str = "prompt",
    metadata: dict[str, Any] | None = None,
) -> ComparisonResult:
    """Build a layer-wise :class:`~.types.ComparisonResult` between two lens runs.

    ``log_probs_a``/``log_probs_b`` are ``[num_layers + 1, V]`` full-vocabulary
    log-probabilities for the single compared position (already sliced from
    :class:`~.engine.EngineOutput.log_probs`), or ``None`` to skip KL/JS entirely.
    ``alignment`` is a ``(layer_a, layer_b)`` pair per entry of the resulting
    ``.layers``; ``None`` means identity, which requires both results to have the
    same number of layers (raises :class:`ValueError` otherwise -- pass an explicit
    ``alignment`` for layer counts that differ, e.g. once comparing two models).

    ``top1_agreement``/``first_divergence_layer``/``last_divergence_layer`` are
    filled from each :class:`~.types.LayerComparison.top1_same`, reproducing the
    pre-existing "prompt" comparison behaviour exactly.
    """
    if alignment is None:
        if len(result_a.layers) != len(result_b.layers):
            raise ValueError(
                f"result_a has {len(result_a.layers)} layers, result_b has "
                f"{len(result_b.layers)} layers; pass an explicit alignment"
            )
        alignment = [(i, i) for i in range(len(result_a.layers))]
    alignment = list(alignment)

    layers_by_index_a = {layer.layer_index: layer for layer in result_a.layers}
    layers_by_index_b = {layer.layer_index: layer for layer in result_b.layers}

    comparisons: list[LayerComparison] = []
    for layer_a_idx, layer_b_idx in alignment:
        la = layers_by_index_a[layer_a_idx]
        lb = layers_by_index_b[layer_b_idx]
        lp_a = log_probs_a[layer_a_idx] if log_probs_a is not None else None
        lp_b = log_probs_b[layer_b_idx] if log_probs_b is not None else None
        comparisons.append(compare_layer(la, lb, lp_a, lp_b, compare_by=compare_by))

    agreement = [lc.top1_same for lc in comparisons]
    diverged = [i for i, same in enumerate(agreement) if not same]

    return ComparisonResult(
        result_a=result_a,
        result_b=result_b,
        top1_agreement=agreement,
        first_divergence_layer=diverged[0] if diverged else None,
        last_divergence_layer=diverged[-1] if diverged else None,
        layers=comparisons,
        alignment=alignment,
        kind=kind,
        metadata=dict(metadata or {}),
    )


def align_layers(num_layers_a: int, num_layers_b: int) -> list[tuple[int, int]]:
    """Nearest relative-depth alignment between two models' decoder-block counts.

    One entry per ``layer_a`` in ``0..num_layers_a`` (inclusive; ``num_layers`` counts
    decoder blocks, layer 0 is the embedding output), with
    ``layer_b = round(layer_a * num_layers_b / num_layers_a)``. Endpoints always map to
    endpoints and the mapping is monotone non-decreasing; no interpolation is done, only
    a nearest-neighbour lookup by relative depth. ``num_layers_a == 0`` returns
    ``[(0, 0)]`` (nothing to divide by).
    """
    if num_layers_a == 0:
        return [(0, 0)]
    return [
        (layer_a, round(layer_a * num_layers_b / num_layers_a))
        for layer_a in range(num_layers_a + 1)
    ]


def vocab_identical(adapter_a: ModelAdapter, adapter_b: ModelAdapter) -> bool:
    """Whether two adapters expose the same id -> token vocabulary.

    Fast path: both expose a ``tokenizer`` attribute and it is the very same object
    (e.g. two adapters built from the same loaded tokenizer). Otherwise both
    :meth:`~.adapters.base.ModelAdapter.vocab_fingerprint` must be non-``None`` and
    equal -- a cheap, comparison-agnostic way to decide this without needing both
    adapters alive at the same time (see :func:`~.analyzer.compare_model_names`'s
    ``sequential=True``).
    """
    tokenizer_a = getattr(adapter_a, "tokenizer", None)
    tokenizer_b = getattr(adapter_b, "tokenizer", None)
    if tokenizer_a is not None and tokenizer_b is not None and tokenizer_a is tokenizer_b:
        return True
    fingerprint_a = adapter_a.vocab_fingerprint()
    fingerprint_b = adapter_b.vocab_fingerprint()
    return (
        fingerprint_a is not None and fingerprint_b is not None and fingerprint_a == fingerprint_b
    )
