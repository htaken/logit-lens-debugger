"""Adapter for Qwen2 / Qwen2.5 / Qwen3 dense causal LMs."""

from __future__ import annotations

from .base import HFCausalLMAdapter


class QwenAdapter(HFCausalLMAdapter):
    """Qwen2ForCausalLM / Qwen3ForCausalLM.

    Layout (Transformers): ``model.model.embed_tokens``, ``model.model.layers``,
    ``model.model.norm`` (RMSNorm) and ``model.lm_head`` (weight tied to the
    embedding matrix for the small checkpoints). Nothing family-specific is
    needed beyond declaring the ``model_type`` values this adapter serves; this
    class is the place for Qwen-only quirks if they appear later.
    """

    supported_model_types = ("qwen2", "qwen3")
