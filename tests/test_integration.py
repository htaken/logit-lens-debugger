"""Opt-in end-to-end check against a real Qwen checkpoint.

Run with::

    LLD_RUN_INTEGRATION=1 uv run pytest -m integration -s

It downloads ``Qwen/Qwen2.5-0.5B`` (~1 GB) on first use. Override the checkpoint
with ``LLD_INTEGRATION_MODEL``.
"""

from __future__ import annotations

import os

import pytest

from logit_lens_debugger import LensMode, LogitLensAnalyzer

pytestmark = pytest.mark.integration

MODEL = os.environ.get("LLD_INTEGRATION_MODEL", "Qwen/Qwen2.5-0.5B")
PROMPT = "Q: What is the capital of Japan?\nA:"


@pytest.fixture(scope="module")
def real_analyzer():
    if not os.environ.get("LLD_RUN_INTEGRATION"):
        pytest.skip("set LLD_RUN_INTEGRATION=1 to run the real-model integration test")
    return LogitLensAnalyzer(MODEL)


def test_capital_of_japan(real_analyzer):
    """ " Tokyo" must lead the final layer and appear as top-1 somewhere in the trajectory.

    Note: the bare prompt "The capital of Japan is" is *not* used because the
    0.5B checkpoints genuinely continue it with a quiz-style blank (" ____").
    """
    result = real_analyzer.analyze(PROMPT, token_index=-1, top_k=5)
    assert len(result.layers) == real_analyzer.adapter.num_layers + 1
    final_tokens = [t.token.strip().lower() for t in result.layers[-1].top_tokens]
    assert final_tokens[0] == "tokyo", final_tokens
    top1_trajectory = [t.token.strip().lower() for t in result.trajectory()]
    assert "tokyo" in top1_trajectory, top1_trajectory


def test_target_tracking_on_generated_token(real_analyzer):
    """Explaining a produced token: analyze the position before it and track it."""
    result = real_analyzer.analyze(PROMPT + " Tokyo", token_index=-2, top_k=5)
    assert result.target_token.strip().lower() == "tokyo"
    assert result.layers[-1].target.rank == 1
    ranks = [layer.target.rank for layer in result.layers]
    assert min(ranks) == 1 and max(ranks) > 1  # not top-1 at every layer


def test_final_norm_matches_model_output(real_analyzer):
    import torch

    adapter = real_analyzer.adapter
    real_analyzer.lens_mode = LensMode.FINAL_NORM
    text = PROMPT
    result = real_analyzer.analyze(text, token_index=-1, top_k=1)
    input_ids = adapter.tokenize(text)
    with torch.inference_mode():
        logits = adapter.model(input_ids=input_ids, use_cache=False).logits[0, -1]
    assert result.layers[-1].top1.token_id == int(logits.argmax())
