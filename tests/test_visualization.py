from __future__ import annotations

import pytest

from logit_lens_debugger.visualization import (
    build_heatmap_figure,
    build_trajectory_figure,
    export_heatmap_html,
    export_trajectory_html,
    format_layers,
    format_trajectory,
)
from logit_lens_debugger.visualization._theme import display_token


def test_display_token_makes_whitespace_visible():
    assert display_token(" Tokyo") == "' Tokyo'"
    assert display_token("\n") == "'\\n'"
    assert display_token("it's") == "'it\\'s'"


def test_format_layers_and_trajectory(sample_result):
    text = format_layers(sample_result)
    assert "Layer 0 (embeddings)" in text
    assert "Layer 3" in text
    assert text.count("   1. ") == 4  # one top-1 line per layer

    limited = format_layers(sample_result, show=2)
    assert "   3. " not in limited

    trajectory = format_trajectory(sample_result)
    assert "Top-1 trajectory (final-norm lens)" in trajectory
    assert "layer   0" in trajectory
    assert "target 'tokyo'" in trajectory


def test_heatmap_figure_structure(sample_result):
    pytest.importorskip("plotly")
    fig = build_heatmap_figure(sample_result)
    heatmap = fig.data[0]
    assert list(heatmap.x) == ["0 (emb)", "1", "2", "3"]
    assert len(heatmap.z) == len(heatmap.y) >= 1
    assert all(len(row) == 4 for row in heatmap.z)
    # target row is present, labelled, and fully populated from per-layer target stats
    target_rows = [i for i, label in enumerate(heatmap.y) if "◀ target" in label]
    assert len(target_rows) == 1
    assert all(v is not None for v in heatmap.z[target_rows[0]])


def test_heatmap_respects_max_tokens(sample_result):
    pytest.importorskip("plotly")
    fig = build_heatmap_figure(sample_result, max_tokens=2)
    assert len(fig.data[0].y) == 2
    assert any("◀ target" in label for label in fig.data[0].y)


def test_export_html(sample_result, tmp_path):
    pytest.importorskip("plotly")
    heat = export_heatmap_html(
        sample_result, tmp_path / "out" / "heatmap.html", include_plotlyjs="cdn"
    )
    traj = export_trajectory_html(
        sample_result, tmp_path / "trajectory.html", include_plotlyjs="cdn"
    )
    for path in (heat, traj):
        assert path.exists()
        content = path.read_text(encoding="utf-8")
        assert "<html" in content.lower()
        assert "plotly" in content


def test_trajectory_figure_series(sample_result, analyzer):
    pytest.importorskip("plotly")
    fig = build_trajectory_figure(sample_result)
    assert len(fig.data) == 2  # top-1 + target
    assert fig.data[0].name == "top-1 probability"
    assert len(fig.data[0].y) == 4

    no_target = analyzer.analyze("the capital of japan is", token_index=-1, top_k=3)
    assert len(build_trajectory_figure(no_target).data) == 1
