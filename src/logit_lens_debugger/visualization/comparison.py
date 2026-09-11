"""Layer-wise comparison rendering: a plain-text table and a Plotly divergence chart."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..types import ComparisonResult, LayerComparison
from ._theme import CATEGORICAL, base_layout, display_token, layer_label, require_plotly


def _fmt_metric(value: float | None, spec: str = ".4f") -> str:
    return "-" if value is None else format(value, spec)


def _fmt_target_rank(lc: LayerComparison) -> str:
    if lc.target_rank_a is None or lc.target_rank_b is None:
        return "-"
    return f"{lc.target_rank_a} -> {lc.target_rank_b}"


def format_comparison(cmp: ComparisonResult, kl_threshold: float = 0.1) -> str:
    """Plain-text table of ``cmp.layers``: top-1, overlap, KL/JS and target rank per layer.

    Followed by the existing top-1 divergence sentence and a
    ``JS first >= {kl_threshold} at layer X`` footer line (``never`` if it is never
    reached, ``no distributions`` if no layer has a JS value at all).
    """
    layers_by_index_a = {layer.layer_index: layer for layer in cmp.result_a.layers}
    layers_by_index_b = {layer.layer_index: layer for layer in cmp.result_b.layers}

    lines = [
        f"{'layer':>5}  {'A top-1 (p)':<26} {'B top-1 (p)':<26} {'same':<5} "
        f"{'overlap':>7} {'KL(A‖B)':>9} {'KL(B‖A)':>9} {'JS':>8}  target rank A->B"
    ]
    for lc in cmp.layers:
        la = layers_by_index_a[lc.layer_a]
        lb = layers_by_index_b[lc.layer_b]
        lines.append(
            f"{lc.layer_a:>5}  "
            f"{display_token(la.top1.token) + f' ({la.top1.probability:.3f})':<26} "
            f"{display_token(lb.top1.token) + f' ({lb.top1.probability:.3f})':<26} "
            f"{'yes' if lc.top1_same else 'NO':<5} "
            f"{lc.topk_overlap:>7.3f} "
            f"{_fmt_metric(lc.kl_a_b):>9} {_fmt_metric(lc.kl_b_a):>9} {_fmt_metric(lc.js):>8}  "
            f"{_fmt_target_rank(lc)}"
        )

    lines.append("")
    if cmp.first_divergence_layer is None:
        lines.append("Top-1 predictions agree at every layer.")
    else:
        differing = len(cmp.divergence_layers)
        total = len(cmp.top1_agreement)
        lines.append(
            f"Top-1 first diverges at layer {cmp.first_divergence_layer}, "
            f"last differs at layer {cmp.last_divergence_layer} "
            f"({differing} of {total} layers differ)."
        )

    if not any(lc.js is not None for lc in cmp.layers):
        lines.append(f"JS first >= {kl_threshold}: no distributions.")
    else:
        layer = cmp.first_layer_kl_above(kl_threshold, direction="js")
        lines.append(
            f"JS first >= {kl_threshold}: never."
            if layer is None
            else f"JS first >= {kl_threshold} at layer {layer}."
        )

    return "\n".join(lines)


def build_comparison_figure(cmp: ComparisonResult, title: str | None = None) -> Any:
    """Line chart of KL(A‖B) / KL(B‖A) / JS by layer, with markers where top-1 differs.

    Traces for a divergence measure are omitted entirely when it is ``None`` at every
    layer (e.g. ``keep_log_probs=False`` was used).
    """
    go = require_plotly()
    x = [layer_label(lc.layer_a) for lc in cmp.layers]

    traces = []
    series = [
        ("KL(A‖B)", [lc.kl_a_b for lc in cmp.layers], CATEGORICAL[0]),
        ("KL(B‖A)", [lc.kl_b_a for lc in cmp.layers], CATEGORICAL[1]),
        ("JS", [lc.js for lc in cmp.layers], CATEGORICAL[2]),
    ]
    for name, values, color in series:
        if any(v is not None for v in values):
            traces.append(
                go.Scatter(
                    x=x,
                    y=values,
                    mode="lines+markers",
                    name=name,
                    line={"color": color, "width": 2},
                    marker={"size": 7, "color": color},
                    connectgaps=False,
                )
            )

    diverging_x = [layer_label(lc.layer_a) for lc in cmp.layers if not lc.top1_same]
    if diverging_x:
        traces.append(
            go.Scatter(
                x=diverging_x,
                y=[1] * len(diverging_x),
                mode="markers",
                name="top-1 differs",
                marker={"symbol": "x", "size": 10, "color": CATEGORICAL[3]},
                yaxis="y2",
                hovertemplate="layer %{x}: top-1 differs<extra></extra>",
            )
        )

    subtitle = f"{cmp.result_a.model_name} · kind={cmp.kind}"
    layout = base_layout(title or "Prompt comparison: divergence by layer", subtitle)
    layout["xaxis"].update({"title": {"text": "layer"}, "type": "category"})
    layout["yaxis"].update({"title": {"text": "divergence (nats)"}, "rangemode": "tozero"})
    if diverging_x:
        layout["yaxis2"] = {
            "overlaying": "y",
            "side": "right",
            "range": [0, 2],
            "showticklabels": False,
        }
    layout["legend"] = {"orientation": "h", "y": -0.2}
    layout["height"] = 460
    return go.Figure(data=traces, layout=layout)


def export_comparison_html(
    cmp: ComparisonResult,
    path: str | Path,
    include_plotlyjs: bool | str = True,
    **kwargs: Any,
) -> Path:
    """Write the comparison figure to a standalone HTML file and return its path."""
    fig = build_comparison_figure(cmp, **kwargs)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(path), include_plotlyjs=include_plotlyjs, full_html=True)
    return path
