"""Command-line interface: ``logit-lens analyze | compare | visualize``."""

from __future__ import annotations

from pathlib import Path

import typer

from .analyzer import LogitLensAnalyzer
from .types import ComparisonResult, LensMode, LogitLensResult
from .visualization.trajectory import format_layers, format_trajectory

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help=(
        "Logit Lens debugger for Hugging Face causal LMs. Projects each layer's residual "
        "stream through the unembedding to show layer-wise next-token predictions."
    ),
)


def _build_analyzer(
    model: str,
    device: str | None,
    dtype: str | None,
    lens_mode: LensMode,
    trust_remote_code: bool,
) -> LogitLensAnalyzer:
    """Factory kept separate so tests can substitute a tiny in-memory model."""
    return LogitLensAnalyzer(
        model,
        device=device,
        dtype=dtype,
        lens_mode=lens_mode,
        trust_remote_code=trust_remote_code,
    )


def _display(token: str) -> str:
    from .visualization._theme import display_token

    return display_token(token)


def _print_header(result: LogitLensResult) -> None:
    device = result.metadata.get("device", "?")
    typer.echo(f"Model: {result.model_name}  (device={device}, lens={result.lens_mode.value})")
    tokens = "  ".join(
        f"[{i}]{_display(tok)}" + ("  <-- analyzed" if i == result.token_index else "")
        for i, tok in enumerate(result.input_tokens)
    )
    typer.echo(f"Tokens ({len(result.input_tokens)}): {tokens}")
    target = _display(result.target_token) if result.target_token is not None else "none"
    typer.echo(
        f"Predicting the token after index {result.token_index} "
        f"{_display(result.analyzed_token)}; tracked target: {target}"
    )
    typer.echo("")


def _write_outputs(result: LogitLensResult, json_path: Path | None, html_path: Path | None) -> None:
    if json_path is not None:
        result.save_json(json_path)
        typer.echo(f"\nSaved JSON to {json_path}")
    if html_path is not None:
        from .visualization.heatmap import export_heatmap_html

        export_heatmap_html(result, html_path)
        typer.echo(f"Saved heatmap to {html_path}")


@app.command()
def analyze(
    text: str = typer.Option(..., "--text", "-t", help="Input text to analyze."),
    model: str = typer.Option(DEFAULT_MODEL, "--model", "-m", help="Hub id or local path."),
    token_index: int = typer.Option(
        -1,
        "--token-index",
        "-i",
        help=(
            "Position whose residual stream is projected; negative counts from the end. "
            "Predictions are for the token that follows this position."
        ),
    ),
    top_k: int = typer.Option(10, "--top-k", "-k", min=1, help="Candidates kept per layer."),
    lens_mode: LensMode = typer.Option(
        LensMode.FINAL_NORM,
        "--lens-mode",
        case_sensitive=False,
        help="raw: hidden -> lm_head. final-norm: hidden -> final norm -> lm_head.",
    ),
    device: str | None = typer.Option(None, "--device", help="cuda | mps | cpu (default: auto)."),
    dtype: str | None = typer.Option(
        None, "--dtype", help="bf16 | fp16 | fp32 | auto (default: bf16 on GPU/MPS, fp32 on CPU)."
    ),
    target_token: str | None = typer.Option(
        None,
        "--target-token",
        help=(
            "Track this token's rank/probability at every layer "
            "(default: the token following --token-index, when present in the text)."
        ),
    ),
    show: int | None = typer.Option(
        None, "--show", help="Tokens printed per layer (default: top-k)."
    ),
    json_path: Path | None = typer.Option(None, "--json", help="Write the full result as JSON."),
    html_path: Path | None = typer.Option(
        None, "--html", help="Write a Layer x Token heatmap (plotly)."
    ),
    trust_remote_code: bool = typer.Option(False, "--trust-remote-code"),
) -> None:
    """Show each layer's top-k next-token predictions for one token position."""
    analyzer = _build_analyzer(model, device, dtype, lens_mode, trust_remote_code)
    result = analyzer.analyze(text, token_index=token_index, top_k=top_k, target_token=target_token)

    _print_header(result)
    typer.echo(format_layers(result, show=show))
    typer.echo("")
    typer.echo(format_trajectory(result))
    _write_outputs(result, json_path, html_path)


@app.command()
def compare(
    text_a: str = typer.Option(..., "--text-a", help="First input text."),
    text_b: str = typer.Option(..., "--text-b", help="Second input text."),
    model: str = typer.Option(DEFAULT_MODEL, "--model", "-m"),
    token_index: int = typer.Option(-1, "--token-index", "-i", help="Applied to both texts."),
    top_k: int = typer.Option(10, "--top-k", "-k", min=1),
    lens_mode: LensMode = typer.Option(LensMode.FINAL_NORM, "--lens-mode", case_sensitive=False),
    device: str | None = typer.Option(None, "--device"),
    dtype: str | None = typer.Option(None, "--dtype"),
    json_path: Path | None = typer.Option(None, "--json", help="Write both results and the diff."),
    trust_remote_code: bool = typer.Option(False, "--trust-remote-code"),
) -> None:
    """Compare layer-wise top-1 predictions between two texts and report where they diverge."""
    analyzer = _build_analyzer(model, device, dtype, lens_mode, trust_remote_code)
    comparison: ComparisonResult = analyzer.compare(
        text_a, text_b, token_index=token_index, top_k=top_k
    )
    a, b = comparison.result_a, comparison.result_b
    typer.echo(f"Model: {a.model_name}  (lens={a.lens_mode.value})")
    typer.echo(f"A: {text_a!r}  -> analyzed {_display(a.analyzed_token)} at index {a.token_index}")
    typer.echo(f"B: {text_b!r}  -> analyzed {_display(b.analyzed_token)} at index {b.token_index}")
    typer.echo("")
    typer.echo(f"{'layer':>5}  {'A top-1':<26} {'B top-1':<26} same")
    for la, lb, same in zip(a.layers, b.layers, comparison.top1_agreement, strict=True):
        typer.echo(
            f"{la.layer_index:>5}  "
            f"{_display(la.top1.token) + f' ({la.top1.probability:.3f})':<26} "
            f"{_display(lb.top1.token) + f' ({lb.top1.probability:.3f})':<26} "
            f"{'yes' if same else 'NO'}"
        )
    typer.echo("")
    if comparison.first_divergence_layer is None:
        typer.echo("Top-1 predictions agree at every layer.")
    else:
        differing = len(comparison.divergence_layers)
        total = len(comparison.top1_agreement)
        typer.echo(
            f"Top-1 first diverges at layer {comparison.first_divergence_layer}, "
            f"last differs at layer {comparison.last_divergence_layer} "
            f"({differing} of {total} layers differ)."
        )
    if json_path is not None:
        comparison.save_json(json_path)
        typer.echo(f"Saved JSON to {json_path}")


@app.command()
def visualize(
    json_path: Path = typer.Argument(..., help="Result JSON written by `analyze --json`."),
    html_path: Path = typer.Option(..., "--html", help="Output HTML path."),
    kind: str = typer.Option("heatmap", "--kind", help="heatmap | trajectory"),
    max_tokens: int = typer.Option(30, "--max-tokens", help="Heatmap rows to keep."),
) -> None:
    """Render a saved result to HTML without reloading the model."""
    result = LogitLensResult.load_json(json_path)
    if kind == "heatmap":
        from .visualization.heatmap import export_heatmap_html

        export_heatmap_html(result, html_path, max_tokens=max_tokens)
    elif kind == "trajectory":
        from .visualization.trajectory import export_trajectory_html

        export_trajectory_html(result, html_path)
    else:
        raise typer.BadParameter("kind must be 'heatmap' or 'trajectory'")
    typer.echo(f"Saved {kind} to {html_path}")


if __name__ == "__main__":  # pragma: no cover
    app()
