"""Shared fixtures: tiny randomly-initialised Qwen2 / Llama models plus a toy tokenizer.

Everything runs on CPU in float32 and needs no network access.
"""

from __future__ import annotations

import pytest
import torch

from logit_lens_debugger import LogitLensAnalyzer
from logit_lens_debugger.adapters import adapter_for_model
from logit_lens_debugger.types import (
    LayerResult,
    LensMode,
    LogitLensResult,
    MultiPositionResult,
    PositionResult,
    TokenPrediction,
)

VOCAB = (
    [
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
    ]
    + [f"w{i}" for i in range(50)]
    + [
        "1",
        "5",
        "0",
        "million",
        "yen2",
    ]
)


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


def make_result(
    top1_ids: list[int],
    target_ranks: list[int] | None = None,
    top_k: int = 3,
    target_token_id: int | None = None,
) -> LogitLensResult:
    """Build a synthetic ``LogitLensResult`` from a top-1 id sequence, for metrics tests.

    ``target_ranks[i]`` (if given) becomes ``layers[i].target.rank``; probability is a
    simple decreasing function of rank. Token strings are ``f"t{token_id}"``.
    """
    if target_ranks is not None and target_token_id is None:
        target_token_id = 9000  # arbitrary id, distinct from the dummy fill ids below

    layers = []
    for i, top1_id in enumerate(top1_ids):
        fill_ids = [top1_id] + [-(i * top_k + j) - 1 for j in range(top_k - 1)]
        top_tokens = [
            TokenPrediction(
                token=f"t{tok_id}",
                token_id=tok_id,
                logit=float(top_k - rank),
                probability=max(0.01, 1.0 - 0.1 * rank),
                rank=rank + 1,
            )
            for rank, tok_id in enumerate(fill_ids)
        ]
        target = None
        if target_ranks is not None:
            rank = target_ranks[i]
            target = TokenPrediction(
                token=f"t{target_token_id}",
                token_id=target_token_id,
                logit=1.0,
                probability=1.0 / rank,
                rank=rank,
            )
        layers.append(LayerResult(layer_index=i, top_tokens=top_tokens, target=target))

    return LogitLensResult(
        model_name="synthetic",
        lens_mode=LensMode.FINAL_NORM,
        text="synthetic",
        input_tokens=["synthetic"],
        input_token_ids=[0],
        token_index=0,
        top_k=top_k,
        layers=layers,
        target_token=f"t{target_token_id}" if target_token_id is not None else None,
        target_token_id=target_token_id,
    )


def to_position_result(
    result: LogitLensResult,
    target_token: str | None = None,
    diff_span=None,
) -> PositionResult:
    """Wrap a ``make_result``-built ``LogitLensResult`` into a matching ``PositionResult``,
    for ``anomaly.score_position(position, ..., context_result=result)`` tests.

    ``target_token`` overrides the generated-token text independently of the
    layer-tracked ``target_token_id`` -- the anomaly module never requires them to
    match, since ``PositionResult.target_token`` is only used for the text-based
    signals while ``layer.target`` drives the lens-based ones.
    """
    return PositionResult(
        token_index=result.token_index,
        layers=result.layers,
        target_token=target_token if target_token is not None else result.target_token,
        target_token_id=result.target_token_id,
        diff_span=diff_span,
    )


def make_multi_position_result(
    positions: list[PositionResult],
    prompt_length: int = 0,
    top_k: int = 3,
    metadata: dict | None = None,
) -> MultiPositionResult:
    """Wrap synthetic ``PositionResult``s (e.g. from ``to_position_result``) into a
    ``MultiPositionResult``, for ``anomaly.detect_anomalies`` tests.

    ``input_tokens`` / ``input_token_ids`` are unused by anomaly scoring, so they are
    left as placeholders rather than reconstructed from the positions.
    """
    return MultiPositionResult(
        model_name="synthetic",
        lens_mode=LensMode.FINAL_NORM,
        text="synthetic",
        input_tokens=["synthetic"],
        input_token_ids=[0],
        prompt_length=prompt_length,
        top_k=top_k,
        positions=positions,
        metadata=dict(metadata or {}),
    )
