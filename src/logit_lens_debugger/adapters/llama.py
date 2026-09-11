"""Adapter for Llama-architecture causal LMs (Llama 2/3, SmolLM2, ...)."""

from __future__ import annotations

from .base import HFCausalLMAdapter


class LlamaAdapter(HFCausalLMAdapter):
    """LlamaForCausalLM.

    Same layout as Qwen in Transformers: ``model.model.{embed_tokens, layers,
    norm}`` and ``model.lm_head``. Kept as a separate class so Llama-specific
    handling has an obvious home and so the registry can be explicit about
    which families have been verified.
    """

    supported_model_types = ("llama",)
