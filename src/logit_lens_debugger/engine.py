"""Logit Lens engine: hidden state -> (final norm) -> lm_head -> softmax -> top-k.

The engine is model-agnostic; everything model-specific goes through a
``ModelAdapter``. Memory stays bounded because only the requested token
positions are projected to vocabulary logits and each layer's logits are
discarded right after the top-k has been extracted.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from .adapters.base import ModelAdapter
from .types import LayerResult, LensMode, TokenPrediction


@dataclass
class EngineOutput:
    """Result of :meth:`LogitLensEngine.run_detailed`."""

    layers: list[list[LayerResult]]
    log_probs: torch.Tensor | None  # [num_layers + 1, P, V] CPU float32, or None


class LogitLensEngine:
    def __init__(self, adapter: ModelAdapter, lens_mode: LensMode | str = LensMode.FINAL_NORM):
        self.adapter = adapter
        self.lens_mode = LensMode(lens_mode)

    def lens_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        """Project residual-stream vectors ``[..., hidden]`` to float32 logits ``[..., vocab]``."""
        if self.lens_mode is LensMode.FINAL_NORM:
            hidden = self.adapter.apply_final_norm(hidden)
        return self.adapter.project_to_vocab(hidden).float()

    @staticmethod
    def _normalize_targets(
        target_token_id: int | Sequence[int | None] | None, num_positions: int
    ) -> list[int | None]:
        """Broadcast a single target to every position, or validate a per-position list."""
        if target_token_id is None:
            return [None] * num_positions
        if isinstance(target_token_id, int):
            return [target_token_id] * num_positions
        targets = list(target_token_id)
        if len(targets) != num_positions:
            raise ValueError(
                f"target_token_id has {len(targets)} entries, expected {num_positions} "
                "(one per position)"
            )
        return targets

    def run(
        self,
        input_ids: torch.Tensor,
        positions: Sequence[int],
        top_k: int = 10,
        target_token_id: int | Sequence[int | None] | None = None,
    ) -> list[list[LayerResult]]:
        """Compute per-layer top-k predictions for each requested position.

        Thin wrapper around :meth:`run_detailed` that returns only ``.layers``
        (i.e. ``keep_log_probs=False``, the current memory behaviour).
        """
        return self.run_detailed(
            input_ids, positions, top_k=top_k, target_token_id=target_token_id
        ).layers

    @torch.inference_mode()
    def run_detailed(
        self,
        input_ids: torch.Tensor,
        positions: Sequence[int],
        top_k: int = 10,
        target_token_id: int | Sequence[int | None] | None = None,
        keep_log_probs: bool = False,
    ) -> EngineOutput:
        """Compute per-layer top-k predictions for each requested position.

        Returns ``output.layers[p][l]`` for position index ``p`` (in ``positions``
        order) and layer ``l`` in ``0..num_layers``. ``target_token_id`` is either
        a single id shared by every position, ``None`` (no target tracked for any
        position) or a sequence with one entry (an id or ``None``) per position.

        With ``keep_log_probs=True``, ``output.log_probs`` additionally holds the
        full-vocabulary log-probabilities for every layer and position (moved to
        CPU float32 right after each layer's logits are computed), shape
        ``[num_layers + 1, P, V]``; otherwise it is ``None``. This is the only way
        to keep a full-vocabulary distribution around (e.g. for KL/JS comparisons
        in :mod:`.comparison`), so it is opt-in and off by default -- the memory
        behaviour of plain ``run()``/``keep_log_probs=False`` is unchanged.
        """
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        positions = list(positions)
        if not positions:
            raise ValueError("positions must not be empty")

        targets = self._normalize_targets(target_token_id, len(positions))
        target_rows = [p for p, t in enumerate(targets) if t is not None]
        device = self.adapter.device
        target_row_index = (
            torch.tensor(target_rows, dtype=torch.long, device=device) if target_rows else None
        )
        target_ids_tensor = (
            torch.tensor([targets[p] for p in target_rows], dtype=torch.long, device=device)
            if target_rows
            else None
        )

        hidden_states = self.adapter.get_hidden_states(input_ids, positions)
        results: list[list[LayerResult]] = [[] for _ in positions]
        kept_log_probs: list[torch.Tensor] = []

        for layer_index, hidden in enumerate(hidden_states):
            logits = self.lens_logits(hidden)  # [P, V]
            log_probs = torch.log_softmax(logits, dim=-1)
            k = min(top_k, logits.shape[-1])
            top_logits, top_ids = torch.topk(logits, k, dim=-1)
            top_probs = log_probs.gather(-1, top_ids).exp()
            entropy = -(log_probs.exp() * log_probs).sum(dim=-1)

            target_cpu: dict[int, tuple[int, float, float]] = {}
            if target_row_index is not None:
                sel_logits = logits.index_select(0, target_row_index)  # [T, V]
                sel_log_probs = log_probs.index_select(0, target_row_index)
                target_logit = sel_logits.gather(-1, target_ids_tensor.unsqueeze(-1)).squeeze(-1)
                target_rank = (sel_logits > target_logit.unsqueeze(-1)).sum(dim=-1) + 1
                target_prob = (
                    sel_log_probs.gather(-1, target_ids_tensor.unsqueeze(-1)).squeeze(-1).exp()
                )
                for row, rank, logit, prob in zip(
                    target_rows,
                    target_rank.cpu().tolist(),
                    target_logit.cpu().tolist(),
                    target_prob.cpu().tolist(),
                    strict=True,
                ):
                    target_cpu[row] = (rank, logit, prob)

            if keep_log_probs:
                kept_log_probs.append(log_probs.detach().to("cpu", torch.float32))

            # Move the small per-position summaries to CPU, then free the full logits.
            top_logits_cpu = top_logits.cpu().tolist()
            top_ids_cpu = top_ids.cpu().tolist()
            top_probs_cpu = top_probs.cpu().tolist()
            entropy_cpu = entropy.cpu().tolist()
            del logits, log_probs, top_logits, top_ids, top_probs, entropy

            for p in range(len(positions)):
                top_tokens = [
                    TokenPrediction(
                        token=self.adapter.decode_token(token_id),
                        token_id=int(token_id),
                        logit=float(logit),
                        probability=float(prob),
                        rank=rank,
                    )
                    for rank, (token_id, logit, prob) in enumerate(
                        zip(top_ids_cpu[p], top_logits_cpu[p], top_probs_cpu[p], strict=True),
                        start=1,
                    )
                ]
                target = None
                if p in target_cpu:
                    rank, logit, prob = target_cpu[p]
                    target_id = targets[p]
                    target = TokenPrediction(
                        token=self.adapter.decode_token(target_id),
                        token_id=int(target_id),
                        logit=float(logit),
                        probability=float(prob),
                        rank=int(rank),
                    )
                results[p].append(
                    LayerResult(
                        layer_index=layer_index,
                        top_tokens=top_tokens,
                        target=target,
                        entropy=float(entropy_cpu[p]),
                    )
                )

        log_probs_out = torch.stack(kept_log_probs, dim=0) if keep_log_probs else None
        return EngineOutput(layers=results, log_probs=log_probs_out)
