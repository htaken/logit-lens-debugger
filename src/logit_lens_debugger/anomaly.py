"""Automatic anomaly detection: score generated tokens by how much they deserve a look.

Pure stdlib + :mod:`.types`, :mod:`.metrics`, :mod:`.diff` -- no torch/transformers
import -- so a :class:`~.types.MultiPositionResult` loaded from JSON can be scored
without reloading the model.

Each signal is a heuristic in ``[0, 1]``; see ``docs/design-next-features.md`` §4 and
the README's anomaly detection section for the full table. A signal that cannot be
evaluated for a position (e.g. no target tracked, or no ``source_tokens`` given) is
omitted from :attr:`~.types.PositionAnomaly.signals` rather than scored as 0, and the
composite score is a weighted average over only the available signals. The score is a
prioritization heuristic for what to inspect first, not evidence of a cause --
intermediate-layer predictions are especially unstable in small models, and the
"novel"/"entity_like" signals are character-class heuristics, not real NER.
"""

from __future__ import annotations

import math
import unicodedata
from collections.abc import Sequence

from .diff import token_class
from .metrics import compute_instability
from .types import (
    DEFAULT_ANOMALY_WEIGHTS,
    AnomalyConfig,
    AnomalyReport,
    LogitLensResult,
    MultiPositionResult,
    PositionAnomaly,
    PositionResult,
)

#: Token classes (see ``diff.token_class``) counted as "content" for the ``novel``
#: signal. A bare "kana" token only counts if it is at least 2 characters long --
#: single-kana particles (e.g. "は", "が") are function words, not content.
_CONTENT_CLASSES = frozenset({"digit", "alpha", "kanji", "mixed"})


def _is_content_token(token: str) -> bool:
    stripped = token.strip()
    cls = token_class(token)
    if cls == "kana":
        return len(stripped) >= 2
    return cls in _CONTENT_CLASSES


def _is_katakana_char(char: str) -> bool:
    try:
        name = unicodedata.name(char)
    except ValueError:
        return False
    return "KATAKANA" in name  # also matches U+30FC, the katakana prolonged-sound mark


def _is_all_katakana(text: str) -> bool:
    return bool(text) and all(_is_katakana_char(c) for c in text)


def score_position(
    position: PositionResult,
    *,
    top_k: int,
    prompt_length: int,
    vocab_size: int | None,
    source_tokens: Sequence[str] | None,
    is_entity: bool,
    config: AnomalyConfig,
    context_result: LogitLensResult,
) -> PositionAnomaly:
    """Score one analyzed position.

    ``context_result`` is this position's full per-layer trajectory (one entry of
    ``MultiPositionResult.to_results()``, needed to run
    :func:`~.metrics.compute_instability`); ``top_k`` is the lens run's top-k
    (informational, matches ``context_result.top_k``).
    """
    signals: dict[str, float] = {}
    layers = context_result.layers
    num_layers = context_result.num_layers  # decoder blocks, len(layers) - 1
    final = layers[-1]

    has_target_everywhere = all(layer.target is not None for layer in layers)
    if has_target_everywhere:
        target = final.target
        signals["low_final_prob"] = 1.0 - target.probability
        rank = target.rank
        signals["final_rank"] = (
            0.0 if rank == 1 else min(1.0, math.log2(rank) / config.rank_saturation_log2)
        )

    instability = None
    try:
        computed = compute_instability(context_result, config.from_layer, config.late_fraction)
    except ValueError:
        computed = None
    if computed is not None:
        instability = computed
        signals["instability"] = computed.instability_score
        if has_target_everywhere:
            if computed.target_first_topk_layer is None:
                signals["late_emergence"] = 1.0
            else:
                signals["late_emergence"] = computed.target_first_topk_layer / num_layers

    if vocab_size is not None and final.entropy is not None:
        signals["final_entropy"] = min(1.0, max(0.0, final.entropy / math.log(vocab_size)))

    top1_final = layers[-1].top1.token_id
    top1_prev = layers[-2].top1.token_id
    if top1_final != top1_prev:
        signals["flip_at_end"] = 1.0
    elif num_layers >= 2 and top1_final != layers[-3].top1.token_id:
        signals["flip_at_end"] = 0.5
    else:
        signals["flip_at_end"] = 0.0

    tok = position.target_token
    novel = None
    if tok is not None:
        signals["in_diff"] = 1.0 if position.diff_span is not None else 0.0
        signals["numeric"] = 1.0 if token_class(tok) == "digit" else 0.0
        if source_tokens is not None:
            normalized_source = {t.strip().lower() for t in source_tokens}
            novel = (
                1.0
                if _is_content_token(tok) and tok.strip().lower() not in normalized_source
                else 0.0
            )
            signals["novel"] = novel

    if novel is not None:
        stripped = tok.strip()
        entity_like = novel == 1.0 and (
            stripped[:1].isupper() or token_class(tok) == "kanji" or _is_all_katakana(stripped)
        )
        signals["entity_like"] = 1.0 if entity_like else 0.0

    signals["entity"] = 1.0 if is_entity else 0.0

    numerator = 0.0
    denominator = 0.0
    for name, value in signals.items():
        weight = config.weights.get(name, 0.0)
        if weight > 0:
            numerator += weight * value
            denominator += weight
    score = numerator / denominator if denominator > 0 else 0.0

    flags = [
        name
        for name in DEFAULT_ANOMALY_WEIGHTS
        if name in signals and signals[name] >= config.flag_threshold
    ]

    output_index = position.token_index - prompt_length + 1
    if output_index < 0:
        output_index = None

    return PositionAnomaly(
        token_index=position.token_index,
        output_index=output_index,
        token=tok,
        token_id=position.target_token_id,
        score=score,
        signals=signals,
        flags=flags,
        diff_span=position.diff_span,
        instability=instability,
    )


def detect_anomalies(
    result: MultiPositionResult,
    source_tokens: Sequence[str] | None = None,
    config: AnomalyConfig | None = None,
    entity_spans: Sequence[tuple[int, int]] | None = None,
) -> AnomalyReport:
    """Score every position of ``result`` and return a sorted :class:`AnomalyReport`.

    ``source_tokens`` (decoded source tokens) enables the ``novel``/``entity_like``
    text signals. ``entity_spans`` are ``[start, end)`` *output token index* ranges
    from an external NER pass: a position is treated as an entity when its
    ``output_index`` falls in one of them (see the ``entity`` signal) -- NER itself
    is out of scope for this package.
    """
    config = config or AnomalyConfig()
    if not result.positions:
        return AnomalyReport(
            model_name=result.model_name,
            lens_mode=result.lens_mode,
            text=result.text,
            prompt_length=result.prompt_length,
            config=config,
            positions=[],
            metadata=dict(result.metadata),
        )

    vocab_size = result.metadata.get("vocab_size")
    spans = entity_spans or []

    anomalies = []
    for context_result, position in zip(result.to_results(), result.positions, strict=True):
        output_index = position.token_index - result.prompt_length + 1
        is_entity = any(start <= output_index < end for start, end in spans)
        anomalies.append(
            score_position(
                position,
                top_k=result.top_k,
                prompt_length=result.prompt_length,
                vocab_size=vocab_size,
                source_tokens=source_tokens,
                is_entity=is_entity,
                config=config,
                context_result=context_result,
            )
        )

    anomalies.sort(key=lambda a: (-a.score, a.token_index))

    return AnomalyReport(
        model_name=result.model_name,
        lens_mode=result.lens_mode,
        text=result.text,
        prompt_length=result.prompt_length,
        config=config,
        positions=anomalies,
        metadata=dict(result.metadata),
    )
