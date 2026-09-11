from __future__ import annotations

import pytest
import torch

from conftest import VOCAB
from logit_lens_debugger.adapters import (
    LlamaAdapter,
    QwenAdapter,
    UnsupportedModelError,
    adapter_class_for_model_type,
    resolve_device,
    resolve_dtype,
    supported_model_types,
)


@pytest.mark.parametrize(
    ("model_type", "expected"),
    [("qwen2", QwenAdapter), ("qwen3", QwenAdapter), ("llama", LlamaAdapter)],
)
def test_registry_dispatch(model_type, expected):
    assert adapter_class_for_model_type(model_type) is expected


def test_registry_rejects_unknown_model_type():
    with pytest.raises(UnsupportedModelError, match="gpt2"):
        adapter_class_for_model_type("gpt2")
    assert {"qwen2", "qwen3", "llama"} <= set(supported_model_types())


def test_adapter_for_model_picks_family(qwen_adapter, llama_adapter):
    assert isinstance(qwen_adapter, QwenAdapter)
    assert isinstance(llama_adapter, LlamaAdapter)
    assert qwen_adapter.num_layers == 3
    assert llama_adapter.num_layers == 2


def test_vocab_size(qwen_adapter):
    assert qwen_adapter.vocab_size == len(VOCAB)


@pytest.mark.parametrize("adapter_fixture", ["qwen_adapter", "llama_adapter"])
def test_hidden_states_shapes(adapter_fixture, request):
    adapter = request.getfixturevalue(adapter_fixture)
    input_ids = adapter.tokenize("the capital of japan is")
    seq_len = input_ids.shape[1]

    full = adapter.get_hidden_states(input_ids)
    assert len(full) == adapter.num_layers + 1
    for hidden in full:
        assert hidden.shape == (seq_len, adapter.model.config.hidden_size)

    sliced = adapter.get_hidden_states(input_ids, positions=[1, seq_len - 1])
    assert len(sliced) == adapter.num_layers + 1
    for layer_full, layer_sliced in zip(full, sliced, strict=True):
        assert layer_sliced.shape == (2, adapter.model.config.hidden_size)
        torch.testing.assert_close(layer_sliced[0], layer_full[1])
        torch.testing.assert_close(layer_sliced[1], layer_full[-1])


def test_hidden_states_match_transformers_and_last_is_pre_norm(qwen_adapter):
    """Entries 0..N-1 equal HF's hidden_states; our last entry is *before* the final norm."""
    adapter = qwen_adapter
    input_ids = adapter.tokenize("the capital of japan is")
    ours = adapter.get_hidden_states(input_ids)

    with torch.inference_mode():
        hf = adapter.model(input_ids=input_ids, output_hidden_states=True, use_cache=False)
        decoder_out = adapter.model.model(input_ids=input_ids, use_cache=False)

    assert len(hf.hidden_states) == len(ours)
    for i in range(len(ours) - 1):
        torch.testing.assert_close(ours[i], hf.hidden_states[i][0])

    # The decoder's last_hidden_state is post-norm; ours must be the pre-norm residual stream.
    normed = adapter.apply_final_norm(ours[-1])
    torch.testing.assert_close(normed, decoder_out.last_hidden_state[0])
    assert not torch.allclose(ours[-1], decoder_out.last_hidden_state[0])


def test_hooks_are_removed_after_capture(qwen_adapter):
    """Our capture must leave the module hook tables exactly as it found them.

    (Transformers 5.x installs its own persistent output-capturing hooks the
    first time ``output_hidden_states=True`` is used, so compare before/after
    rather than asserting an empty table.)
    """
    adapter = qwen_adapter
    modules = [adapter._embedding(), *adapter._layers()]
    before = [dict(m._forward_hooks) for m in modules]
    adapter.get_hidden_states(adapter.tokenize("the capital"))
    after = [dict(m._forward_hooks) for m in modules]
    assert after == before


def test_hidden_states_rejects_batches(qwen_adapter):
    with pytest.raises(ValueError, match=r"\[1, seq_len\]"):
        qwen_adapter.get_hidden_states(torch.zeros(2, 4, dtype=torch.long))


def test_project_to_vocab_matches_lm_head(qwen_adapter):
    hidden = torch.randn(3, qwen_adapter.model.config.hidden_size)
    logits = qwen_adapter.project_to_vocab(hidden)
    assert logits.shape == (3, qwen_adapter.model.config.vocab_size)
    torch.testing.assert_close(logits, qwen_adapter.model.lm_head(hidden))


def test_tokenizer_helpers(qwen_adapter):
    assert qwen_adapter.decode_token(6) == "tokyo"
    assert qwen_adapter.encode("tokyo") == [6]


def test_decode_multiple_tokens(qwen_adapter):
    assert qwen_adapter.decode([1, 2]) == "the capital"


def test_resolve_device_and_dtype():
    assert resolve_device("cpu") == torch.device("cpu")
    assert resolve_device(None).type in {"cpu", "cuda", "mps"}
    cpu = torch.device("cpu")
    assert resolve_dtype(None, cpu) is torch.float32
    assert resolve_dtype(None, torch.device("cuda")) is torch.bfloat16
    assert resolve_dtype("fp16", cpu) is torch.float16
    assert resolve_dtype("BF16", cpu) is torch.bfloat16
    assert resolve_dtype(torch.float64, cpu) is torch.float64
    assert resolve_dtype("auto", cpu) == "auto"
    with pytest.raises(ValueError, match="Unknown dtype"):
        resolve_dtype("int4", cpu)
