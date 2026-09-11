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
