"""Shared fixtures: tiny randomly-initialised Qwen2 / Llama models plus a toy tokenizer.

Everything runs on CPU in float32 and needs no network access.
"""

from __future__ import annotations

import pytest
import torch

from logit_lens_debugger import LogitLensAnalyzer
from logit_lens_debugger.adapters import adapter_for_model

VOCAB = [
    "<unk>",
    "the",
    "capital",
    "of",
    "japan",
    "is",
    "tokyo",
    "osaka",
    "kyoto",
    "sales",
    "were",
    "150",
    "1500",
    "yen",
] + [f"w{i}" for i in range(50)]


class ToyTokenizer:
    """Whitespace tokenizer exposing the subset of the HF tokenizer API the adapters use."""

    def __init__(self, vocab: list[str]):
        self.vocab = list(vocab)
        self.index = {tok: i for i, tok in enumerate(self.vocab)}

    def _ids(self, text: str) -> list[int]:
        return [self.index.get(word, 0) for word in text.split()]

    def __call__(self, text: str, return_tensors: str | None = None):
        return {"input_ids": torch.tensor([self._ids(text)], dtype=torch.long)}

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return self._ids(text)

    def decode(self, ids, **kwargs) -> str:
        return " ".join(self.vocab[int(i)] for i in ids)

    def convert_ids_to_tokens(self, ids) -> list[str]:
        return [self.vocab[int(i)] for i in ids]


@pytest.fixture(scope="session")
def toy_tokenizer() -> ToyTokenizer:
    return ToyTokenizer(VOCAB)


def _perturb_final_norm(model: torch.nn.Module) -> None:
    """Give the final RMSNorm a non-uniform weight so raw and final-norm lenses rank differently."""
    with torch.no_grad():
        weight = model.model.norm.weight
        weight.copy_(torch.linspace(0.2, 2.0, weight.numel()))


@pytest.fixture(scope="session")
def tiny_qwen_model():
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(0)
    config = Qwen2Config(
        vocab_size=len(VOCAB),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        tie_word_embeddings=False,
        rms_norm_eps=1e-6,
    )
    model = Qwen2ForCausalLM(config).eval()
    _perturb_final_norm(model)
    return model


@pytest.fixture(scope="session")
def tiny_llama_model():
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(1)
    config = LlamaConfig(
        vocab_size=len(VOCAB),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        tie_word_embeddings=False,
        rms_norm_eps=1e-5,
    )
    model = LlamaForCausalLM(config).eval()
    _perturb_final_norm(model)
    return model


@pytest.fixture
def qwen_adapter(tiny_qwen_model, toy_tokenizer):
    return adapter_for_model(tiny_qwen_model, toy_tokenizer, model_name="tiny-qwen")


@pytest.fixture
def llama_adapter(tiny_llama_model, toy_tokenizer):
    return adapter_for_model(tiny_llama_model, toy_tokenizer, model_name="tiny-llama")


@pytest.fixture
def analyzer(qwen_adapter) -> LogitLensAnalyzer:
    return LogitLensAnalyzer(adapter=qwen_adapter)


@pytest.fixture
def sample_result(analyzer):
    return analyzer.analyze("the capital of japan is tokyo", token_index=4, top_k=5)
