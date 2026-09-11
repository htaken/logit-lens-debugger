"""Top-1 prediction trajectory across layers, as text and as a Plotly line chart."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..types import (
    AnomalyReport,
    DiffSpan,
    InstabilityMetrics,
    LogitLensResult,
    MultiPositionResult,
)
from ._theme import CATEGORICAL, base_layout, display_token, layer_label, require_plotly


def format_instability(metrics: InstabilityMetrics) -> str:
    """Compact plain-text summary of layer instability (see ``metrics.compute_instability``)."""
    lines = [
        f"Instability from layer {metrics.from_layer} "
        f"({metrics.num_evaluated_layers} layers evaluated): score={metrics.instability_score:.2f}",
        f"  switches={metrics.top1_switches}  oscillations={metrics.oscillations}  "
        f"distinct={metrics.distinct_top1}  settle_layer={metrics.settle_layer}  "
        f"late_switches={metrics.late_switches} (>= layer {metrics.late_start_layer})",
    ]
    if metrics.target_final_rank is not None:
        lines.append(
            f"  target: first top-1 at layer {metrics.target_first_top1_layer}, "
            f"first top-k at layer {metrics.target_first_topk_layer}, "
            f"final rank={metrics.target_final_rank} p={metrics.target_final_prob:.3f}, "
            f"log2(rank) mean={metrics.target_rank_log_mean:.2f} "
            f"std={metrics.target_rank_log_std:.2f}"
        )
    return "\n".join(lines)


def format_trajectory(result: LogitLensResult) -> str:
    """Plain-text table: one line per layer with the top-1 token (and target rank if tracked)."""
    analyzed = display_token(result.analyzed_token)
    header = (
        f"Top-1 trajectory ({result.lens_mode.value} lens) for the token after "
        f"index {result.token_index} {analyzed}"
    )
    lines = [header]
    for layer in result.layers:
        top = layer.top1
        line = (
            f"  layer {layer.layer_index:>3}  {display_token(top.token):<24} "
            f"p={top.probability:.3f}"
        )
        if layer.target is not None:
            line += (
                f"   target {display_token(layer.target.token)}"
                f" rank={layer.target.rank:<5} p={layer.target.probability:.3f}"
            )
        lines.append(line)
    return "\n".join(lines)


def format_layers(result: LogitLensResult, show: int | None = None) -> str:
    """Plain-text listing of the top tokens for every layer (the CLI's main output)."""
    blocks = []
    for layer in result.layers:
        title = f"Layer {layer.layer_index}" + (" (embeddings)" if layer.layer_index == 0 else "")
        lines = [title]
        for pred in layer.top_tokens[: show or len(layer.top_tokens)]:
            lines.append(
                f"{pred.rank:>4}. {display_token(pred.token):<24} "
                f"p={pred.probability:.4f}  logit={pred.logit:.2f}"
            )
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def format_diff_spans(result: MultiPositionResult) -> str:
    """Plain-text listing of ``result.diff_spans``, one line per span (see ``diff.py``)."""
    positions_by_span: dict[DiffSpan, list[int]] = {}
    for pos in result.positions:
        if pos.diff_span is not None:
            positions_by_span.setdefault(pos.diff_span, []).append(pos.token_index)

    lines = [f"Diff spans ({len(result.diff_spans)}):"]
    for span in result.diff_spans:
        positions = positions_by_span.get(span, [])
        positions_str = " ".join(str(p) for p in positions) if positions else "-"
        lines.append(
            f"  [{span.kind}] source {display_token(span.source_text)} -> "
            f"output {display_token(span.output_text)}  "
            f"output tokens [{span.output_range[0]}, {span.output_range[1]}) "
            f"positions {positions_str}"
        )
    return "\n".join(lines)


def format_position_summary(result: MultiPositionResult) -> str:
    """Two-line-per-position summary: target rank/probability, settle layer, instability,
    and a compressed top-1 trajectory (only layers where the top-1 token changes)."""
    lines = []
    for single in result.to_results():
        metrics = single.instability()
        output_index = single.token_index - result.prompt_length + 1
        target = display_token(single.target_token) if single.target_token is not None else "none"
        final_target = single.layers[-1].target
        detail = (
            f" final p={final_target.probability:.3f} rank={final_target.rank}"
            if final_target is not None
            else ""
        )
        lines.append(
            f"Position {single.token_index} (output token {output_index} {target}): "
            f"target {target}{detail}; settle layer {metrics.settle_layer}, "
            f"instability {metrics.instability_score:.2f}"
        )
        trajectory_parts = []
        previous_id = None
        for layer in single.layers:
            top1 = layer.top1
            if top1.token_id != previous_id:
                trajectory_parts.append(f"layer {layer.layer_index} {display_token(top1.token)}")
                previous_id = top1.token_id
        lines.append("  " + " -> ".join(trajectory_parts))
    return "\n".join(lines)


def format_anomaly_report(
    report: AnomalyReport, top_n: int = 20, threshold: float | None = None
) -> str:
    """Header + fixed-width table of the top-scoring positions of an ``AnomalyReport``.

    ``threshold`` (default: ``report.config.score_threshold``) is only used for the
    "N with score >= T" header count; ``top_n`` controls how many rows are printed
    (the report itself is already sorted score desc, then ``token_index`` asc).
    """
    cutoff = report.config.score_threshold if threshold is None else threshold
    flagged = report.flagged(cutoff)
    lines = [
        f"Anomaly report: {len(report.positions)} positions scored, "
        f"{len(flagged)} with score >= {cutoff:.2f}",
        "",
        f"{'pos':>5} {'out#':>5} {'token':<16} {'score':>6} {'final_p':>8} {'settle':>6}  flags",
    ]
    for anomaly in report.top(top_n):
        out_idx = "-" if anomaly.output_index is None else str(anomaly.output_index)
        token = "-" if anomaly.token is None else display_token(anomaly.token)
        low_final_prob = anomaly.signals.get("low_final_prob")
        final_p = "-" if low_final_prob is None else f"{1.0 - low_final_prob:.3f}"
        settle = "-" if anomaly.instability is None else str(anomaly.instability.settle_layer)
        flags = ",".join(anomaly.flags) if anomaly.flags else "-"
        lines.append(
            f"{anomaly.token_index:>5} {out_idx:>5} {token:<16} "
            f"{anomaly.score:>6.3f} {final_p:>8} {settle:>6}  {flags}"
        )
    return "\n".join(lines)


def build_trajectory_figure(result: LogitLensResult, title: str | None = None) -> Any:
    """Line chart of top-1 probability per layer, plus the target token's probability if tracked.

    Layers where the top-1 token changes are direct-labeled with the new token.
    """
    go = require_plotly()
    x = [layer_label(layer.layer_index) for layer in result.layers]
    top1 = [layer.top1 for layer in result.layers]

    labels = []
    previous = None
    for pred in top1:
        labels.append(display_token(pred.token) if pred.token_id != previous else "")
        previous = pred.token_id

    traces = [
        go.Scatter(
            x=x,
            y=[p.probability for p in top1],
            mode="lines+markers+text",
            name="top-1 probability",
            text=labels,
            textposition="top center",
            textfont={"size": 11},
            line={"color": CATEGORICAL[0], "width": 2},
            marker={"size": 8, "color": CATEGORICAL[0]},
            customdata=[display_token(p.token) for p in top1],
            hovertemplate="layer %{x}<br>top-1 %{customdata}<br>p = %{y:.4f}<extra></extra>",
        )
    ]
    if result.target_token_id is not None and all(layer.target for layer in result.layers):
        target_label = display_token(result.target_token or "")
        traces.append(
            go.Scatter(
                x=x,
                y=[layer.target.probability for layer in result.layers],
                mode="lines+markers",
                name=f"target {target_label}",
                line={"color": CATEGORICAL[1], "width": 2, "dash": "dot"},
                marker={"size": 8, "color": CATEGORICAL[1]},
                customdata=[layer.target.rank for layer in result.layers],
                hovertemplate=(
                    f"layer %{{x}}<br>target {target_label}<br>p = %{{y:.4f}}<br>"
                    "rank %{customdata}<extra></extra>"
                ),
            )
        )

    subtitle = (
        f"{result.model_name} · lens={result.lens_mode.value} · token_index={result.token_index}"
    )
    layout = base_layout(title or "Logit lens: top-1 probability by layer", subtitle)
    layout["xaxis"].update({"title": {"text": "layer (0 = embedding output)"}, "type": "category"})
    layout["yaxis"].update({"title": {"text": "probability"}, "rangemode": "tozero"})
    layout["legend"] = {"orientation": "h", "y": -0.2}
    layout["height"] = 460
    return go.Figure(data=traces, layout=layout)


def export_trajectory_html(
    result: LogitLensResult,
    path: str | Path,
    include_plotlyjs: bool | str = True,
    **kwargs: Any,
) -> Path:
    fig = build_trajectory_figure(result, **kwargs)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(path), include_plotlyjs=include_plotlyjs, full_html=True)
    return path
