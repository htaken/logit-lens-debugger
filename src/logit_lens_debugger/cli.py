"""Command-line interface: ``logit-lens analyze | compare | visualize``."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from .analyzer import LogitLensAnalyzer
from .types import AnomalyReport, ComparisonResult, LensMode, LogitLensResult
from .visualization.trajectory import (
    format_anomaly_report,
    format_diff_spans,
    format_instability,
    format_layers,
    format_position_summary,
    format_trajectory,
)

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


def _read_text_option(name: str, text: str | None, path: Path | None, required: bool) -> str | None:
    """Resolve a ``--<name>``/``--<name>-file`` option pair to the effective text.

    ``required``: exactly one of ``text``/``path`` must be given. Otherwise at most
    one may be given; if neither is, ``None`` is returned.
    """
    if required:
        if (text is None) == (path is None):
            raise typer.BadParameter(f"Provide exactly one of --{name} or --{name}-file.")
    elif text is not None and path is not None:
        raise typer.BadParameter(f"Provide at most one of --{name} or --{name}-file.")
    if text is not None:
        return text
    if path is not None:
        return path.read_text(encoding="utf-8")
    return None


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
    typer.echo("")
    typer.echo(format_instability(result.instability()))
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
def diff(
    source: str | None = typer.Option(None, "--source", help="Original/source text."),
    source_file: Path | None = typer.Option(
        None, "--source-file", help="Read source text from a file."
    ),
    output: str | None = typer.Option(None, "--output", help="Model output text."),
    output_file: Path | None = typer.Option(
        None, "--output-file", help="Read output text from a file."
    ),
    prompt: str | None = typer.Option(
        None, "--prompt", help="Text actually given to the model (default: --source)."
    ),
    prompt_file: Path | None = typer.Option(
        None, "--prompt-file", help="Read prompt text from a file."
    ),
    model: str = typer.Option(DEFAULT_MODEL, "--model", "-m"),
    top_k: int = typer.Option(10, "--top-k", "-k", min=1),
    lens_mode: LensMode = typer.Option(LensMode.FINAL_NORM, "--lens-mode", case_sensitive=False),
    device: str | None = typer.Option(None, "--device"),
    dtype: str | None = typer.Option(None, "--dtype"),
    context: int = typer.Option(
        0, "--context", help="Extra output tokens analyzed around each diff span."
    ),
    no_expand: bool = typer.Option(
        False, "--no-expand", help="Do not expand spans to whole same-class token runs."
    ),
    no_after_delete: bool = typer.Option(
        False,
        "--no-after-delete",
        help="Do not analyze the output token right after a deletion.",
    ),
    verbose: bool = typer.Option(
        False, "--verbose", help="Also print the per-layer top-k for each position."
    ),
    json_path: Path | None = typer.Option(
        None, "--json", help="Write the full MultiPositionResult as JSON."
    ),
    html_dir: Path | None = typer.Option(
        None, "--html-dir", help="Write one heatmap HTML per analyzed position into this directory."
    ),
    trust_remote_code: bool = typer.Option(False, "--trust-remote-code"),
) -> None:
    """Diff source text against model output and analyze only the flagged token positions."""
    source_text = _read_text_option("source", source, source_file, required=True)
    output_text = _read_text_option("output", output, output_file, required=True)
    prompt_text = _read_text_option("prompt", prompt, prompt_file, required=False)

    analyzer = _build_analyzer(model, device, dtype, lens_mode, trust_remote_code)
    result = analyzer.analyze_diff(
        source_text,
        output_text,
        prompt=prompt_text,
        top_k=top_k,
        context=context,
        expand_spans=not no_expand,
        include_after_delete=not no_after_delete,
    )

    typer.echo(
        f"Model: {result.model_name}  "
        f"(device={result.metadata.get('device', '?')}, lens={result.lens_mode.value})"
    )
    typer.echo(
        f"Context: {len(result.input_tokens)} tokens "
        f"(prompt {result.prompt_length}, output {len(result.output_tokens)})"
    )
    typer.echo("")

    if not result.diff_spans:
        typer.echo("No differences between source and output.")
    else:
        typer.echo(format_diff_spans(result))
        typer.echo("")
        typer.echo(format_position_summary(result))
        if verbose:
            for single in result.to_results():
                typer.echo("")
                typer.echo(format_layers(single))

    if json_path is not None:
        result.save_json(json_path)
        typer.echo(f"\nSaved JSON to {json_path}")
    if html_dir is not None and result.positions:
        from .visualization.heatmap import export_heatmap_html

        html_dir.mkdir(parents=True, exist_ok=True)
        for single in result.to_results():
            export_heatmap_html(single, html_dir / f"position_{single.token_index}.html")
        typer.echo(f"Saved heatmaps to {html_dir}")


@app.command()
def scan(
    prompt: str | None = typer.Option(None, "--prompt", help="Prompt text given to the model."),
    prompt_file: Path | None = typer.Option(
        None, "--prompt-file", help="Read prompt text from a file."
    ),
    output: str | None = typer.Option(None, "--output", help="Model output text."),
    output_file: Path | None = typer.Option(
        None, "--output-file", help="Read output text from a file."
    ),
    source: str | None = typer.Option(
        None, "--source", help="Original/source text (enables the novel/in_diff signals)."
    ),
    source_file: Path | None = typer.Option(
        None, "--source-file", help="Read source text from a file."
    ),
    model: str = typer.Option(DEFAULT_MODEL, "--model", "-m"),
    top_k: int = typer.Option(10, "--top-k", "-k", min=1),
    lens_mode: LensMode = typer.Option(LensMode.FINAL_NORM, "--lens-mode", case_sensitive=False),
    device: str | None = typer.Option(None, "--device"),
    dtype: str | None = typer.Option(None, "--dtype"),
    only_diff: bool = typer.Option(
        False, "--only-diff", help="Score only the positions the source/output diff flags."
    ),
    context: int = typer.Option(
        0, "--context", help="Extra output tokens scored around each diff span (with --only-diff)."
    ),
    top: int = typer.Option(20, "--top", help="Rows printed in the report table."),
    threshold: float = typer.Option(
        0.5, "--threshold", help="Score cutoff for the 'flagged' count in the report header."
    ),
    chunk_size: int | None = typer.Option(
        64, "--chunk-size", help="Positions scored per forward pass (memory/speed trade-off)."
    ),
    json_path: Path | None = typer.Option(
        None, "--json", help="Write the full AnomalyReport as JSON."
    ),
    trust_remote_code: bool = typer.Option(False, "--trust-remote-code"),
) -> None:
    """Score every (or, with --only-diff, every diff-flagged) output token for anomalies."""
    prompt_text = _read_text_option("prompt", prompt, prompt_file, required=True)
    output_text = _read_text_option("output", output, output_file, required=True)
    source_text = _read_text_option("source", source, source_file, required=False)
    if only_diff and source_text is None:
        raise typer.BadParameter("--only-diff requires --source or --source-file.")

    analyzer = _build_analyzer(model, device, dtype, lens_mode, trust_remote_code)
    report: AnomalyReport = analyzer.detect_anomalies(
        prompt_text,
        output_text,
        source=source_text,
        top_k=top_k,
        only_diff=only_diff,
        context=context,
        chunk_size=chunk_size,
    )

    prompt_length = len(analyzer.adapter.tokenize(prompt_text)[0])
    output_length = len(analyzer.adapter.encode(output_text))
    typer.echo(f"Model: {report.model_name}  (lens={report.lens_mode.value})")
    typer.echo(
        f"Context: {prompt_length + output_length} tokens "
        f"(prompt {prompt_length}, output {output_length})"
    )
    typer.echo("")
    typer.echo(format_anomaly_report(report, top, threshold))

    if json_path is not None:
        report.save_json(json_path)
        typer.echo(f"\nSaved JSON to {json_path}")


@app.command()
def metrics(
    result_json: Path = typer.Argument(..., help="Result JSON written by `analyze --json`."),
    from_layer: int = typer.Option(1, "--from-layer", help="First layer index to evaluate."),
    late_fraction: float = typer.Option(
        1 / 3, "--late-fraction", help="Fraction of final layers counted as 'late'."
    ),
    json_path: Path | None = typer.Option(None, "--json", help="Write metrics as JSON."),
) -> None:
    """Compute layer instability metrics from a saved result JSON, without loading the model."""
    result = LogitLensResult.load_json(result_json)
    computed = result.instability(from_layer=from_layer, late_fraction=late_fraction)
    typer.echo(format_instability(computed))
    if json_path is not None:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(computed.to_dict(), indent=2), encoding="utf-8")
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
