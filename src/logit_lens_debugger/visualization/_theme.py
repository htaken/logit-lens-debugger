"""Shared styling helpers for the Plotly-based visualizations."""

from __future__ import annotations

from typing import Any

# Single-hue sequential ramp (light -> dark) used for probability magnitude.
SEQUENTIAL_BLUE: list[str] = [
    "#cde2fb",
    "#b7d3f6",
    "#9ec5f4",
    "#86b6ef",
    "#6da7ec",
    "#5598e7",
    "#3987e5",
    "#2a78d6",
    "#256abf",
    "#1c5cab",
    "#184f95",
    "#104281",
    "#0d366b",
]
# Fixed categorical order for multi-series line charts.
CATEGORICAL: list[str] = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
SURFACE = "#fcfcfb"
INK = "#1f1f1e"
MUTED = "#6b6b68"
GRID = "#e6e5e1"
FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"


def require_plotly() -> Any:
    try:
        import plotly.graph_objects as go
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "Visualization requires plotly. Install with: pip install 'logit-lens-debugger[viz]'"
        ) from exc
    return go


def sequential_colorscale() -> list[list[Any]]:
    last = len(SEQUENTIAL_BLUE) - 1
    return [[i / last, color] for i, color in enumerate(SEQUENTIAL_BLUE)]


def display_token(token: str) -> str:
    """Make whitespace and control characters visible, wrapped in quotes."""
    shown = (
        token.replace("\\", "\\\\")
        .replace("\n", "\\n")
        .replace("\t", "\\t")
        .replace("\r", "\\r")
        .replace("'", "\\'")
    )
    return f"'{shown}'"


def layer_label(index: int) -> str:
    return "0 (emb)" if index == 0 else str(index)


def base_layout(title: str, subtitle: str | None = None) -> dict[str, Any]:
    title_text = (
        title
        if subtitle is None
        else (f"{title}<br><span style='font-size:12px;color:{MUTED}'>{subtitle}</span>")
    )
    return {
        "template": "plotly_white",
        "paper_bgcolor": SURFACE,
        "plot_bgcolor": SURFACE,
        "font": {"family": FONT, "color": INK, "size": 13},
        "title": {"text": title_text, "x": 0.0, "xanchor": "left"},
        "margin": {"l": 40, "r": 40, "t": 90, "b": 60},
        "xaxis": {"gridcolor": GRID, "zeroline": False},
        "yaxis": {"gridcolor": GRID, "zeroline": False},
    }
