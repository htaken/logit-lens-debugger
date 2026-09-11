"""Result types shared by the engine, analyzer, CLI and visualization layers.

Everything here is a plain dataclass so results can be serialized to JSON and
re-loaded without importing torch or transformers.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any


class LensMode(StrEnum):
    """Which projection variant the lens applies to intermediate hidden states.

    RAW:        hidden_state -> lm_head
    FINAL_NORM: hidden_state -> final RMSNorm -> lm_head   (default)

    In FINAL_NORM mode the last layer reproduces the model's real output
    distribution exactly, which makes it the natural reference point.
    """

    RAW = "raw"
    FINAL_NORM = "final-norm"


@dataclass(frozen=True)
class TokenPrediction:
    """One vocabulary entry scored at one layer for one token position."""

    token: str
    token_id: int
    logit: float
    probability: float
    rank: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TokenPrediction:
        return cls(**data)


@dataclass
class LayerResult:
    """Lens output for one layer at one token position.

    ``layer_index`` 0 is the embedding output (input to the first decoder
    block); ``layer_index`` i is the residual stream after decoder block i.
    """

    layer_index: int
    top_tokens: list[TokenPrediction]
    target: TokenPrediction | None = None
    entropy: float | None = None

    @property
    def top1(self) -> TokenPrediction:
        return self.top_tokens[0]

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer_index": self.layer_index,
            "top_tokens": [t.to_dict() for t in self.top_tokens],
            "target": self.target.to_dict() if self.target is not None else None,
            "entropy": self.entropy,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LayerResult:
        target = data.get("target")
        return cls(
            layer_index=data["layer_index"],
            top_tokens=[TokenPrediction.from_dict(t) for t in data["top_tokens"]],
            target=TokenPrediction.from_dict(target) if target is not None else None,
            entropy=data.get("entropy"),
        )


@dataclass
class LogitLensResult:
    """Full lens trajectory for one token position of one input text.

    ``token_index`` is the (non-negative) position whose residual stream was
    projected. Predictions are for the token that *follows* that position.
    """

    model_name: str
    lens_mode: LensMode
    text: str
    input_tokens: list[str]
    input_token_ids: list[int]
    token_index: int
    top_k: int
    layers: list[LayerResult]
    target_token: str | None = None
    target_token_id: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def num_layers(self) -> int:
        """Number of decoder blocks (``len(layers) - 1``; layer 0 is the embedding output)."""
        return len(self.layers) - 1

    @property
    def analyzed_token(self) -> str:
        return self.input_tokens[self.token_index]

    def trajectory(self) -> list[TokenPrediction]:
        """Top-1 prediction at each layer, in layer order."""
        return [layer.top1 for layer in self.layers]

    def instability(self, **kwargs: Any) -> InstabilityMetrics:
        """Compute layer instability metrics for this result.

        See :func:`logit_lens_debugger.metrics.compute_instability`. Imported lazily
        so this module stays importable without pulling in ``metrics.py``.
        """
        from .metrics import compute_instability

        return compute_instability(self, **kwargs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "lens_mode": self.lens_mode.value,
            "text": self.text,
            "input_tokens": list(self.input_tokens),
            "input_token_ids": list(self.input_token_ids),
            "token_index": self.token_index,
            "top_k": self.top_k,
            "target_token": self.target_token,
            "target_token_id": self.target_token_id,
            "metadata": dict(self.metadata),
            "layers": [layer.to_dict() for layer in self.layers],
        }

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    def save_json(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding="utf-8")
        return path

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LogitLensResult:
        return cls(
            model_name=data["model_name"],
            lens_mode=LensMode(data["lens_mode"]),
            text=data["text"],
            input_tokens=list(data["input_tokens"]),
            input_token_ids=list(data["input_token_ids"]),
            token_index=data["token_index"],
            top_k=data["top_k"],
            layers=[LayerResult.from_dict(layer) for layer in data["layers"]],
            target_token=data.get("target_token"),
            target_token_id=data.get("target_token_id"),
            metadata=dict(data.get("metadata", {})),
        )

    @classmethod
    def load_json(cls, path: str | Path) -> LogitLensResult:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass(frozen=True)
class DiffSpan:
    """One contiguous span where a source and output token sequence differ.

    Ranges are ``[start, end)`` token indices into the respective sequence.
    ``kind`` is ``"replace"`` (both ranges non-empty), ``"insert"`` (source empty)
    or ``"delete"`` (output empty). See :mod:`.diff`.
    """

    kind: str
    source_range: tuple[int, int]
    output_range: tuple[int, int]
    source_text: str
    output_text: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "source_range": list(self.source_range),
            "output_range": list(self.output_range),
            "source_text": self.source_text,
            "output_text": self.output_text,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DiffSpan:
        return cls(
            kind=data["kind"],
            source_range=tuple(data["source_range"]),
            output_range=tuple(data["output_range"]),
            source_text=data["source_text"],
            output_text=data["output_text"],
        )


@dataclass
class PositionResult:
    """Lens trajectory for one token position, without the shared context fields.

    Used inside :class:`MultiPositionResult`, which carries ``input_tokens`` once
    for every analyzed position instead of duplicating it per position.
    """

    token_index: int
    layers: list[LayerResult]
    target_token: str | None = None
    target_token_id: int | None = None
    diff_span: DiffSpan | None = None

    def trajectory(self) -> list[TokenPrediction]:
        """Top-1 prediction at each layer, in layer order."""
        return [layer.top1 for layer in self.layers]

    def to_dict(self) -> dict[str, Any]:
        return {
            "token_index": self.token_index,
            "layers": [layer.to_dict() for layer in self.layers],
            "target_token": self.target_token,
            "target_token_id": self.target_token_id,
            "diff_span": self.diff_span.to_dict() if self.diff_span is not None else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PositionResult:
        diff_span = data.get("diff_span")
        return cls(
            token_index=data["token_index"],
            layers=[LayerResult.from_dict(layer) for layer in data["layers"]],
            target_token=data.get("target_token"),
            target_token_id=data.get("target_token_id"),
            diff_span=DiffSpan.from_dict(diff_span) if diff_span is not None else None,
        )


@dataclass
class MultiPositionResult:
    """Several analyzed positions of one context (one shared forward pass).

    ``prompt_length`` is the index of the first output token in
    ``input_token_ids`` (0 if the context has no prompt prefix).
    """

    model_name: str
    lens_mode: LensMode
    text: str
    input_tokens: list[str]
    input_token_ids: list[int]
    prompt_length: int
    top_k: int
    positions: list[PositionResult]
    diff_spans: list[DiffSpan] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def num_layers(self) -> int:
        if not self.positions:
            raise ValueError("num_layers is undefined: positions is empty")
        return len(self.positions[0].layers) - 1

    @property
    def output_tokens(self) -> list[str]:
        return self.input_tokens[self.prompt_length :]

    def to_results(self) -> list[LogitLensResult]:
        """Expand each position into a standalone :class:`LogitLensResult`.

        Lets the existing per-result visualizations and metrics be reused without
        a ``MultiPositionResult``-specific code path.
        """
        results = []
        for pos in self.positions:
            metadata = {
                **self.metadata,
                "prompt_length": self.prompt_length,
                "diff_span": pos.diff_span.to_dict() if pos.diff_span is not None else None,
            }
            results.append(
                LogitLensResult(
                    model_name=self.model_name,
                    lens_mode=self.lens_mode,
                    text=self.text,
                    input_tokens=list(self.input_tokens),
                    input_token_ids=list(self.input_token_ids),
                    token_index=pos.token_index,
                    top_k=self.top_k,
                    layers=pos.layers,
                    target_token=pos.target_token,
                    target_token_id=pos.target_token_id,
                    metadata=metadata,
                )
            )
        return results

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "lens_mode": self.lens_mode.value,
            "text": self.text,
            "input_tokens": list(self.input_tokens),
            "input_token_ids": list(self.input_token_ids),
            "prompt_length": self.prompt_length,
            "top_k": self.top_k,
            "positions": [p.to_dict() for p in self.positions],
            "diff_spans": [s.to_dict() for s in self.diff_spans],
            "metadata": dict(self.metadata),
        }

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    def save_json(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding="utf-8")
        return path

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MultiPositionResult:
        return cls(
            model_name=data["model_name"],
            lens_mode=LensMode(data["lens_mode"]),
            text=data["text"],
            input_tokens=list(data["input_tokens"]),
            input_token_ids=list(data["input_token_ids"]),
            prompt_length=data["prompt_length"],
            top_k=data["top_k"],
            positions=[PositionResult.from_dict(p) for p in data["positions"]],
            diff_spans=[DiffSpan.from_dict(s) for s in data.get("diff_spans", [])],
            metadata=dict(data.get("metadata", {})),
        )

    @classmethod
    def load_json(cls, path: str | Path) -> MultiPositionResult:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass(frozen=True)
class InstabilityMetrics:
    """How much the top-1 trajectory of a :class:`LogitLensResult` moves across layers.

    Produced by :func:`logit_lens_debugger.metrics.compute_instability`. Not part of
    ``LogitLensResult.to_dict()``: these are derived values, always recomputable from
    the layers, so they are never stored as ground truth.
    """

    from_layer: int
    late_fraction: float
    num_evaluated_layers: int
    top1_switches: int
    oscillations: int
    distinct_top1: int
    settle_layer: int
    late_start_layer: int
    late_switches: int
    instability_score: float
    target_first_top1_layer: int | None
    target_first_topk_layer: int | None
    target_rank_log_mean: float | None
    target_rank_log_std: float | None
    target_final_rank: int | None
    target_final_prob: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> InstabilityMetrics:
        return cls(**data)


#: Default signal weights for :func:`logit_lens_debugger.anomaly.detect_anomalies`:
#: 60% lens-based signals (derived from the layer trajectory), 40% text-based
#: signals (derived from the generated token and the source/diff). See
#: ``docs/design-next-features.md`` §4 and the README's anomaly detection section.
DEFAULT_ANOMALY_WEIGHTS: dict[str, float] = {
    "low_final_prob": 0.15,
    "final_rank": 0.10,
    "final_entropy": 0.05,
    "late_emergence": 0.15,
    "instability": 0.10,
    "flip_at_end": 0.05,
    "in_diff": 0.15,
    "numeric": 0.05,
    "novel": 0.10,
    "entity_like": 0.05,
    "entity": 0.05,
}


@dataclass(frozen=True)
class AnomalyConfig:
    """Weights and thresholds for :func:`logit_lens_debugger.anomaly.detect_anomalies`.

    ``weights`` maps signal name -> weight; a signal with weight 0 (or absent
    from this mapping) never contributes to the score. ``flag_threshold`` gates
    :attr:`PositionAnomaly.flags`; ``score_threshold`` is the default cutoff for
    :meth:`AnomalyReport.flagged`. ``from_layer`` / ``late_fraction`` are forwarded
    to :func:`~logit_lens_debugger.metrics.compute_instability`.
    """

    weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_ANOMALY_WEIGHTS))
    flag_threshold: float = 0.5
    score_threshold: float = 0.5
    from_layer: int = 1
    late_fraction: float = 1 / 3
    rank_saturation_log2: float = 10.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "weights": dict(self.weights),
            "flag_threshold": self.flag_threshold,
            "score_threshold": self.score_threshold,
            "from_layer": self.from_layer,
            "late_fraction": self.late_fraction,
            "rank_saturation_log2": self.rank_saturation_log2,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AnomalyConfig:
        return cls(
            weights=dict(data["weights"]) if "weights" in data else dict(DEFAULT_ANOMALY_WEIGHTS),
            flag_threshold=data.get("flag_threshold", 0.5),
            score_threshold=data.get("score_threshold", 0.5),
            from_layer=data.get("from_layer", 1),
            late_fraction=data.get("late_fraction", 1 / 3),
            rank_saturation_log2=data.get("rank_saturation_log2", 10.0),
        )


@dataclass
class PositionAnomaly:
    """Anomaly score for one analyzed position, produced by :func:`.anomaly.score_position`.

    ``signals`` only contains the signals that could be evaluated for this
    position (see the module docstring of :mod:`.anomaly`); a missing signal
    never contributed to ``score``, it was not scored as 0.
    """

    token_index: int
    output_index: int | None
    token: str | None
    token_id: int | None
    score: float
    signals: dict[str, float]
    flags: list[str]
    diff_span: DiffSpan | None
    instability: InstabilityMetrics | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "token_index": self.token_index,
            "output_index": self.output_index,
            "token": self.token,
            "token_id": self.token_id,
            "score": self.score,
            "signals": dict(self.signals),
            "flags": list(self.flags),
            "diff_span": self.diff_span.to_dict() if self.diff_span is not None else None,
            "instability": (self.instability.to_dict() if self.instability is not None else None),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PositionAnomaly:
        diff_span = data.get("diff_span")
        instability = data.get("instability")
        return cls(
            token_index=data["token_index"],
            output_index=data.get("output_index"),
            token=data.get("token"),
            token_id=data.get("token_id"),
            score=data["score"],
            signals=dict(data.get("signals", {})),
            flags=list(data.get("flags", [])),
            diff_span=DiffSpan.from_dict(diff_span) if diff_span is not None else None,
            instability=(
                InstabilityMetrics.from_dict(instability) if instability is not None else None
            ),
        )


@dataclass
class AnomalyReport:
    """Scored positions for one context, produced by :func:`.anomaly.detect_anomalies`.

    ``positions`` is sorted by score descending, then ``token_index`` ascending.
    """

    model_name: str
    lens_mode: LensMode
    text: str
    prompt_length: int
    config: AnomalyConfig
    positions: list[PositionAnomaly]
    metadata: dict[str, Any] = field(default_factory=dict)

    def flagged(self, threshold: float | None = None) -> list[PositionAnomaly]:
        """Positions scoring at or above ``threshold`` (default: ``config.score_threshold``)."""
        cutoff = self.config.score_threshold if threshold is None else threshold
        return [p for p in self.positions if p.score >= cutoff]

    def top(self, n: int) -> list[PositionAnomaly]:
        """The ``n`` highest-scoring positions (``positions`` is already sorted)."""
        return self.positions[:n]

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "lens_mode": self.lens_mode.value,
            "text": self.text,
            "prompt_length": self.prompt_length,
            "config": self.config.to_dict(),
            "positions": [p.to_dict() for p in self.positions],
            "metadata": dict(self.metadata),
        }

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    def save_json(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding="utf-8")
        return path

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AnomalyReport:
        return cls(
            model_name=data["model_name"],
            lens_mode=LensMode(data["lens_mode"]),
            text=data["text"],
            prompt_length=data["prompt_length"],
            config=AnomalyConfig.from_dict(data["config"]),
            positions=[PositionAnomaly.from_dict(p) for p in data["positions"]],
            metadata=dict(data.get("metadata", {})),
        )

    @classmethod
    def load_json(cls, path: str | Path) -> AnomalyReport:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass
class ComparisonResult:
    """Layer-wise comparison of two lens runs (e.g. prompt A vs prompt B)."""

    result_a: LogitLensResult
    result_b: LogitLensResult
    top1_agreement: list[bool]
    first_divergence_layer: int | None
    last_divergence_layer: int | None

    @property
    def divergence_layers(self) -> list[int]:
        return [i for i, same in enumerate(self.top1_agreement) if not same]

    def to_dict(self) -> dict[str, Any]:
        return {
            "result_a": self.result_a.to_dict(),
            "result_b": self.result_b.to_dict(),
            "top1_agreement": list(self.top1_agreement),
            "first_divergence_layer": self.first_divergence_layer,
            "last_divergence_layer": self.last_divergence_layer,
        }

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    def save_json(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding="utf-8")
        return path
