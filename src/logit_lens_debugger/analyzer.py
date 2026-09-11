"""High-level API: ``LogitLensAnalyzer.analyze()`` and ``.compare()``."""

from __future__ import annotations

import warnings
from typing import Any

import torch

from .adapters.base import ModelAdapter
from .adapters.registry import load_adapter
from .engine import LogitLensEngine
from .types import ComparisonResult, LensMode, LogitLensResult


def normalize_index(index: int, length: int) -> int:
    """Turn a possibly-negative token index into a valid non-negative one."""
    if length <= 0:
        raise ValueError("Input produced no tokens")
    resolved = index + length if index < 0 else index
    if not 0 <= resolved < length:
        raise IndexError(f"token_index {index} is out of range for {length} tokens")
    return resolved


class LogitLensAnalyzer:
    """Run a logit lens over a Hugging Face causal LM.

    Example::

        analyzer = LogitLensAnalyzer("Qwen/Qwen2.5-0.5B-Instruct", device="cuda")
        result = analyzer.analyze("The capital of Japan is", token_index=-1, top_k=10)
        result.layers[10].top_tokens[0].token
    """

    def __init__(
        self,
        model_name: str | None = None,
        device: str | torch.device | None = None,
        dtype: str | torch.dtype | None = None,
        lens_mode: LensMode | str = LensMode.FINAL_NORM,
        adapter: ModelAdapter | None = None,
        **model_kwargs: Any,
    ):
        if adapter is None:
            if model_name is None:
                raise ValueError("Provide either model_name or an adapter")
            adapter = load_adapter(model_name, device=device, dtype=dtype, **model_kwargs)
        self.adapter = adapter
        self.engine = LogitLensEngine(adapter, lens_mode)

    @property
    def lens_mode(self) -> LensMode:
        return self.engine.lens_mode

    @lens_mode.setter
    def lens_mode(self, value: LensMode | str) -> None:
        self.engine.lens_mode = LensMode(value)

    @property
    def model_name(self) -> str:
        return self.adapter.model_name

    # ------------------------------------------------------------------ analyze

    def analyze(
        self,
        text: str,
        token_index: int = -1,
        top_k: int = 10,
        target_token: str | int | None = None,
    ) -> LogitLensResult:
        """Project the residual stream at ``token_index`` through every layer.

        Predictions are for the token *following* ``token_index``. If
        ``target_token`` is omitted and a following token exists in ``text``,
        that token becomes the target so its rank/probability per layer is
        reported. To ask why the model produced the token at position ``j``,
        analyze ``token_index=j - 1``.
        """
        input_ids = self.adapter.tokenize(text)
        ids: list[int] = input_ids[0].tolist()
        position = normalize_index(token_index, len(ids))

        if target_token is not None:
            target_id = self._resolve_target(target_token)
        elif position + 1 < len(ids):
            target_id = ids[position + 1]
        else:
            target_id = None

        layers = self.engine.run(input_ids, [position], top_k=top_k, target_token_id=target_id)[0]
        return LogitLensResult(
            model_name=self.model_name,
            lens_mode=self.lens_mode,
            text=text,
            input_tokens=[self.adapter.decode_token(i) for i in ids],
            input_token_ids=ids,
            token_index=position,
            top_k=top_k,
            layers=layers,
            target_token=self.adapter.decode_token(target_id) if target_id is not None else None,
            target_token_id=target_id,
            metadata={"device": str(self.adapter.device)},
        )

    def _resolve_target(self, target_token: str | int) -> int:
        if isinstance(target_token, int):
            return target_token
        ids = self.adapter.encode(target_token)
        if not ids:
            raise ValueError(f"target_token {target_token!r} produced no tokens")
        if len(ids) > 1:
            pieces = [self.adapter.decode_token(i) for i in ids]
            warnings.warn(
                f"target_token {target_token!r} tokenizes to {len(ids)} pieces {pieces}; "
                f"tracking only the first piece {pieces[0]!r}",
                stacklevel=3,
            )
        return ids[0]

    # ------------------------------------------------------------------ compare

    def compare(
        self,
        text_a: str,
        text_b: str,
        token_index: int = -1,
        top_k: int = 10,
        target_token: str | int | None = None,
    ) -> ComparisonResult:
        """Run the lens on two texts at the same (relative) position and locate divergence.

        Divergence is measured on the top-1 token per layer. This is the seed
        for the planned prompt-comparison feature.
        """
        result_a = self.analyze(
            text_a, token_index=token_index, top_k=top_k, target_token=target_token
        )
        result_b = self.analyze(
            text_b, token_index=token_index, top_k=top_k, target_token=target_token
        )
        agreement = [
            la.top1.token_id == lb.top1.token_id
            for la, lb in zip(result_a.layers, result_b.layers, strict=True)
        ]
        diverged = [i for i, same in enumerate(agreement) if not same]
        return ComparisonResult(
            result_a=result_a,
            result_b=result_b,
            top1_agreement=agreement,
            first_divergence_layer=diverged[0] if diverged else None,
            last_divergence_layer=diverged[-1] if diverged else None,
        )
