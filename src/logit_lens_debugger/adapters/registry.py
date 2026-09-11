"""Maps Hugging Face ``config.model_type`` values to adapter classes."""

from __future__ import annotations

from typing import Any

import torch

from .base import HFCausalLMAdapter
from .llama import LlamaAdapter
from .qwen import QwenAdapter


class UnsupportedModelError(ValueError):
    """Raised when no adapter claims the model's ``model_type``."""


_ADAPTERS: list[type[HFCausalLMAdapter]] = [QwenAdapter, LlamaAdapter]


def register_adapter(adapter_cls: type[HFCausalLMAdapter]) -> type[HFCausalLMAdapter]:
    """Register an additional adapter class (usable as a decorator)."""
    if adapter_cls not in _ADAPTERS:
        _ADAPTERS.append(adapter_cls)
    return adapter_cls


def supported_model_types() -> list[str]:
    return sorted({t for cls in _ADAPTERS for t in cls.supported_model_types})


def adapter_class_for_model_type(model_type: str) -> type[HFCausalLMAdapter]:
    for cls in _ADAPTERS:
        if model_type in cls.supported_model_types:
            return cls
    raise UnsupportedModelError(
        f"No adapter registered for model_type={model_type!r}. Supported: {supported_model_types()}"
    )


def adapter_for_model(
    model: Any, tokenizer: Any, model_name: str | None = None
) -> HFCausalLMAdapter:
    """Wrap an already-instantiated Hugging Face model."""
    cls = adapter_class_for_model_type(model.config.model_type)
    return cls(model, tokenizer, model_name=model_name)


def load_adapter(
    model_name: str,
    device: str | torch.device | None = None,
    dtype: str | torch.dtype | None = None,
    **model_kwargs: Any,
) -> HFCausalLMAdapter:
    """Load ``model_name`` from the Hub / local path with the matching adapter."""
    from transformers import AutoConfig

    config_kwargs = {
        k: model_kwargs[k] for k in ("revision", "trust_remote_code", "token") if k in model_kwargs
    }
    config = AutoConfig.from_pretrained(model_name, **config_kwargs)
    cls = adapter_class_for_model_type(config.model_type)
    return cls.from_pretrained(model_name, device=device, dtype=dtype, **model_kwargs)
