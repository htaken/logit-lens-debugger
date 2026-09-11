"""Model adapter interface plus the shared Hugging Face decoder-only implementation.

The adapter is the only place that knows how a given model family exposes its
embedding layer, decoder blocks, final normalization and unembedding matrix.
The engine, analyzer and visualization code talk to ``ModelAdapter`` only.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

import torch
from torch import nn

_DTYPE_ALIASES: dict[str, torch.dtype] = {
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "fp16": torch.float16,
    "float16": torch.float16,
    "half": torch.float16,
    "fp32": torch.float32,
    "float32": torch.float32,
    "float": torch.float32,
}


def resolve_device(device: str | torch.device | None) -> torch.device:
    """Pick an explicit device, else the best available accelerator."""
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_dtype(dtype: str | torch.dtype | None, device: torch.device) -> torch.dtype | str:
    """Resolve a dtype spec.

    ``None`` -> bf16 on accelerators, fp32 on CPU. ``"auto"`` passes through to
    ``from_pretrained`` so the checkpoint's own dtype is used.
    """
    if isinstance(dtype, torch.dtype):
        return dtype
    if dtype is None:
        return torch.float32 if device.type == "cpu" else torch.bfloat16
    if dtype == "auto":
        return "auto"
    try:
        return _DTYPE_ALIASES[dtype.lower()]
    except KeyError as exc:
        raise ValueError(
            f"Unknown dtype {dtype!r}; expected one of {sorted(_DTYPE_ALIASES)} or 'auto'"
        ) from exc


class ModelAdapter(ABC):
    """Model-family specific access to the pieces a logit lens needs."""

    model_name: str
    device: torch.device

    @property
    @abstractmethod
    def num_layers(self) -> int:
        """Number of decoder blocks."""

    @abstractmethod
    def tokenize(self, text: str) -> torch.Tensor:
        """Return ``input_ids`` of shape ``[1, seq_len]`` on ``self.device``."""

    @abstractmethod
    def get_hidden_states(
        self, input_ids: torch.Tensor, positions: Sequence[int] | None = None
    ) -> list[torch.Tensor]:
        """Run the decoder and return ``num_layers + 1`` residual-stream tensors.

        Entry 0 is the embedding output; entry i is the output of decoder block
        i *before* the final normalization. If ``positions`` is given each
        tensor has shape ``[len(positions), hidden]``, else ``[seq_len, hidden]``.
        """

    @abstractmethod
    def apply_final_norm(self, hidden: torch.Tensor) -> torch.Tensor:
        """Apply the model's final normalization (e.g. RMSNorm) to hidden states."""

    @abstractmethod
    def project_to_vocab(self, hidden: torch.Tensor) -> torch.Tensor:
        """Project hidden states ``[..., hidden]`` to vocabulary logits ``[..., vocab]``."""

    @abstractmethod
    def decode_token(self, token_id: int) -> str:
        """Human-readable text for a single token id."""

    @abstractmethod
    def encode(self, text: str) -> list[int]:
        """Token ids for ``text`` without special tokens (used to resolve target tokens)."""


class HFCausalLMAdapter(ModelAdapter):
    """Shared implementation for Hugging Face decoder-only models.

    Assumes the standard layout used by Qwen2/Qwen3/Llama-style models::

        model.<base_model_prefix>.embed_tokens
        model.<base_model_prefix>.layers[i]
        model.<base_model_prefix>.norm
        model.lm_head

    Hidden states are captured with forward hooks rather than
    ``output_hidden_states=True`` because recent Transformers versions
    overwrite the last entry of that tuple with the *post*-norm
    ``last_hidden_state``, which would corrupt both lens modes for the final
    layer. Hooks also let us slice to the requested positions immediately,
    so we never keep ``[layers, seq, hidden]`` around.
    """

    supported_model_types: tuple[str, ...] = ()

    def __init__(self, model: nn.Module, tokenizer: Any, model_name: str | None = None):
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.model_name = model_name or getattr(
            model.config, "_name_or_path", model.__class__.__name__
        )
        first_param = next(model.parameters())
        self.device = first_param.device
        self.dtype = first_param.dtype

    # ------------------------------------------------------------------ loading

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        device: str | torch.device | None = None,
        dtype: str | torch.dtype | None = None,
        **model_kwargs: Any,
    ) -> HFCausalLMAdapter:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        resolved_device = resolve_device(device)
        resolved_dtype = resolve_dtype(dtype, resolved_device)
        tokenizer_kwargs = {
            k: model_kwargs[k]
            for k in ("revision", "trust_remote_code", "token")
            if k in model_kwargs
        }
        tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=resolved_dtype, **model_kwargs
        )
        model = model.to(resolved_device).eval()
        return cls(model, tokenizer, model_name=model_name)

    # ------------------------------------------------------- structure access

    def _decoder(self) -> nn.Module:
        prefix = getattr(self.model, "base_model_prefix", "model")
        return getattr(self.model, prefix)

    def _layers(self) -> nn.ModuleList:
        return self._decoder().layers

    def _embedding(self) -> nn.Module:
        return self._decoder().embed_tokens

    def _final_norm_module(self) -> nn.Module:
        return self._decoder().norm

    def _lm_head(self) -> nn.Module:
        return self.model.lm_head

    # ------------------------------------------------------------ interface

    @property
    def num_layers(self) -> int:
        return len(self._layers())

    def tokenize(self, text: str) -> torch.Tensor:
        encoded = self.tokenizer(text, return_tensors="pt")
        return encoded["input_ids"].to(self.device)

    @torch.inference_mode()
    def get_hidden_states(
        self, input_ids: torch.Tensor, positions: Sequence[int] | None = None
    ) -> list[torch.Tensor]:
        if input_ids.dim() != 2 or input_ids.shape[0] != 1:
            raise ValueError(
                f"input_ids must have shape [1, seq_len], got {tuple(input_ids.shape)}"
            )
        input_ids = input_ids.to(self.device)
        position_index = (
            torch.as_tensor(list(positions), dtype=torch.long, device=self.device)
            if positions is not None
            else None
        )
        captured: dict[int, torch.Tensor] = {}

        def make_hook(index: int):
            def hook(_module: nn.Module, _args: tuple, output: Any) -> None:
                hidden = output[0] if isinstance(output, tuple) else output
                if position_index is not None:
                    hidden = hidden.index_select(1, position_index)
                captured[index] = hidden[0].detach()

            return hook

        handles = [self._embedding().register_forward_hook(make_hook(0))]
        for index, layer in enumerate(self._layers(), start=1):
            handles.append(layer.register_forward_hook(make_hook(index)))
        try:
            self._decoder()(input_ids=input_ids, use_cache=False)
        finally:
            for handle in handles:
                handle.remove()

        expected = self.num_layers + 1
        if len(captured) != expected:
            raise RuntimeError(
                f"Captured {len(captured)} hidden states but expected {expected}; "
                "the model layout may not match this adapter."
            )
        return [captured[i] for i in range(expected)]

    @torch.inference_mode()
    def apply_final_norm(self, hidden: torch.Tensor) -> torch.Tensor:
        return self._final_norm_module()(hidden)

    @torch.inference_mode()
    def project_to_vocab(self, hidden: torch.Tensor) -> torch.Tensor:
        head = self._lm_head()
        return head(hidden.to(head.weight.dtype))

    def decode_token(self, token_id: int) -> str:
        return self.tokenizer.decode([int(token_id)])

    def encode(self, text: str) -> list[int]:
        return list(self.tokenizer.encode(text, add_special_tokens=False))
