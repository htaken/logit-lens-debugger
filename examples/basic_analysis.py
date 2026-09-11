"""Minimal end-to-end example: layer-wise predictions for the token after "is".

Run with::

    uv run python examples/basic_analysis.py [model_name]

Downloads the model on first use (~1 GB for Qwen2.5-0.5B-Instruct).
"""

from __future__ import annotations

import sys
from pathlib import Path

from logit_lens_debugger import LogitLensAnalyzer
from logit_lens_debugger.visualization import export_heatmap_html, format_trajectory

MODEL = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-0.5B"
OUT_DIR = Path("outputs")
PROMPT = "Q: What is the capital of Japan?\nA:"


def main() -> None:
    analyzer = LogitLensAnalyzer(MODEL, lens_mode="final-norm")  # device/dtype auto-selected

    # 1) Prompt-only: what does each layer predict as the next token?
    result = analyzer.analyze(PROMPT, token_index=-1, top_k=5)
    print(format_trajectory(result))
    print()
    last = result.layers[-1]
    print(f"Final layer top-{result.top_k}:")
    for pred in last.top_tokens:
        print(f"  {pred.rank}. {pred.token!r:<12} p={pred.probability:.3f} logit={pred.logit:.2f}")

    # 2) Prompt + produced text: track how the *actual* next token (" Tokyo") rises through the
    #    layers. token_index points at ":"; the token after it becomes the tracked target.
    tracked = analyzer.analyze(PROMPT + " Tokyo", token_index=-2, top_k=5)
    print()
    print(f"Tracking target {tracked.target_token!r} (id {tracked.target_token_id}):")
    for layer in tracked.layers:
        t = layer.target
        print(f"  layer {layer.layer_index:>2}: rank={t.rank:<6} p={t.probability:.4f}")

    # 3) Persist for later inspection / visualization without reloading the model.
    OUT_DIR.mkdir(exist_ok=True)
    json_path = tracked.save_json(OUT_DIR / "capital_of_japan.json")
    html_path = export_heatmap_html(tracked, OUT_DIR / "capital_of_japan.html")
    print()
    print(f"Saved {json_path} and {html_path}")


if __name__ == "__main__":
    main()
