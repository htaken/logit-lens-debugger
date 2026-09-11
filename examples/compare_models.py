"""Minimal example: compare two different models on the same prompt, layer by layer.

Run with::

    uv run python examples/compare_models.py

Downloads both models on first use (~1 GB + ~3 GB for Qwen2.5-0.5B / 1.5B). Models are
loaded sequentially (``sequential=True``, the default) so only one is resident in
memory at a time -- the 0.5B model is analyzed and released before the 1.5B model is
loaded, which matters on memory-constrained machines (e.g. an M2 16 GB Mac).
"""

from __future__ import annotations

from pathlib import Path

from logit_lens_debugger import compare_model_names
from logit_lens_debugger.visualization import format_comparison, format_model_comparison_header

MODEL_A = "Qwen/Qwen2.5-0.5B"
MODEL_B = "Qwen/Qwen2.5-1.5B"
PROMPT = "Q: What is the capital of Japan?\nA:"
OUT_DIR = Path("outputs")


def main() -> None:
    cmp = compare_model_names(
        MODEL_A,
        MODEL_B,
        PROMPT,
        token_index=-1,
        top_k=5,
        sequential=True,  # release model A before loading model B
    )

    print(format_model_comparison_header(cmp))
    print()
    print(format_comparison(cmp))

    # Both Qwen2.5 checkpoints share a tokenizer, so cmp.metadata["same_vocab"] is True
    # here and KL/JS are available; comparing e.g. Qwen vs Llama would fall back to
    # comparing decoded token text instead (cmp.metadata["same_vocab"] is False).
    print()
    print(f"same_vocab: {cmp.metadata['same_vocab']}")
    print(f"alignment (layer_a, layer_b) for the first few layers: {cmp.alignment[:5]}")

    OUT_DIR.mkdir(exist_ok=True)
    json_path = cmp.save_json(OUT_DIR / "compare_models.json")
    print()
    print(f"Saved {json_path}")


if __name__ == "__main__":
    main()
