"""Layer x Token heatmap: how each candidate's probability evolves across layers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..types import LogitLensResult
from ._theme import (
    base_layout,
    display_token,
    layer_label,
    require_plotly,
    sequential_colorscale,
)


def _select_candidates(
    result: LogitLensResult, max_tokens: int | None
) -> tuple[list[int], dict[int, str]]:
    """Choose the token rows to draw.

    Rows are the union of every layer's top-k. If that exceeds ``max_tokens``
    the highest-peaking tokens are kept. Rows are ordered by the layer at which
    each token peaks so the picture reads left-to-right as a trajectory. The
    target token (if any) is always present.
    """
    peak_prob: dict[int, float] = {}
    peak_layer: dict[int, int] = {}
    labels: dict[int, str] = {}
    for layer in result.layers:
        for pred in layer.top_tokens:
            if pred.probability > peak_prob.get(pred.token_id, -1.0):
                peak_prob[pred.token_id] = pred.probability
                peak_layer[pred.token_id] = layer.layer_index
            labels.setdefault(pred.token_id, pred.token)

    target_id = result.target_token_id
    if target_id is not None:
        labels.setdefault(target_id, result.target_token or "")
        for layer in result.layers:
            if layer.target is not None and layer.target.probability > peak_prob.get(
                target_id, -1.0
            ):
                peak_prob[target_id] = layer.target.probability
                peak_layer[target_id] = layer.layer_index

    ordered = sorted(peak_prob, key=lambda tid: peak_prob[tid], reverse=True)
    if max_tokens is not None and len(ordered) > max_tokens:
        kept = ordered[:max_tokens]
        if target_id is not None and target_id not in kept:
            kept[-1] = target_id
        ordered = kept
    ordered.sort(key=lambda tid: (peak_layer[tid], -peak_prob[tid]))
    return ordered, labels


def _shorten(label: str, max_chars: int) -> str:
    return label if len(label) <= max_chars else label[: max_chars - 1] + "…"


def build_heatmap_figure(
    result: LogitLensResult,
    max_tokens: int | None = 30,
    title: str | None = None,
    max_label_chars: int = 24,
) -> Any:
    """Return a ``plotly.graph_objects.Figure`` (x = layer, y = token, color = probability).

    Cells are blank where a token fell outside that layer's top-k, except for
    the target token whose probability is known at every layer. Long token
    labels are shortened on the axis; the full token is shown on hover.
    """
    go = require_plotly()
    token_ids, labels = _select_candidates(result, max_tokens)
    layers = result.layers
    target_id = result.target_token_id

    z: list[list[float | None]] = []
    text: list[list[str]] = []
    custom: list[list[list[Any]]] = []
    for tid in token_ids:
        full_label = display_token(labels.get(tid, str(tid)))
        z_row: list[float | None] = []
        t_row: list[str] = []
        c_row: list[list[Any]] = []
        for layer in layers:
            hit = next((p for p in layer.top_tokens if p.token_id == tid), None)
            if hit is None and tid == target_id and layer.target is not None:
                hit = layer.target
            if hit is None:
                z_row.append(None)
                t_row.append("")
                c_row.append(["", "", full_label])
            else:
                z_row.append(hit.probability)
                t_row.append(str(hit.rank) if hit.rank <= 3 else "")
                c_row.append([hit.rank, f"{hit.logit:.2f}", full_label])
        z.append(z_row)
        text.append(t_row)
        custom.append(c_row)

    y_labels = []
    for tid in token_ids:
        label = _shorten(display_token(labels.get(tid, str(tid))), max_label_chars)
        if tid == target_id:
            label += "  ◀ target"
        y_labels.append(label)
    x_labels = [layer_label(layer.layer_index) for layer in layers]

    heatmap = go.Heatmap(
        z=z,
        x=x_labels,
        y=y_labels,
        text=text,
        customdata=custom,
        texttemplate="%{text}",
        textfont={"size": 11},
        colorscale=sequential_colorscale(),
        zmin=0.0,
        xgap=2,
        ygap=2,
        hoverongaps=False,
        colorbar={"title": {"text": "probability"}, "thickness": 12, "outlinewidth": 0},
        hovertemplate=(
            "token %{customdata[2]}<br>layer %{x}<br>p = %{z:.4f}<br>rank %{customdata[0]}"
            "<br>logit %{customdata[1]}<extra></extra>"
        ),
    )

    analyzed = display_token(result.analyzed_token)
    subtitle = (
        f"{result.model_name} · lens={result.lens_mode.value} · "
        f"residual stream at token_index={result.token_index} {analyzed}, "
        f"predicting the next token · top-{result.top_k} per layer; blank = outside top-k"
    )
    layout = base_layout(title or "Logit lens: probability by layer and candidate token", subtitle)
    layout["xaxis"].update({"title": {"text": "layer (0 = embedding output)"}, "type": "category"})
    layout["yaxis"].update({"autorange": "reversed", "type": "category", "tickfont": {"size": 12}})
    layout["height"] = max(380, 26 * len(token_ids) + 180)
    fig = go.Figure(data=[heatmap], layout=layout)
    return fig


def export_heatmap_html(
    result: LogitLensResult,
    path: str | Path,
    include_plotlyjs: bool | str = True,
    **kwargs: Any,
) -> Path:
    """Write the heatmap to a standalone HTML file and return its path."""
    fig = build_heatmap_figure(result, **kwargs)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(path), include_plotlyjs=include_plotlyjs, full_html=True)
    return path
