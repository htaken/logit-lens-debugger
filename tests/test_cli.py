from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from logit_lens_debugger import LogitLensAnalyzer, cli

runner = CliRunner()


@pytest.fixture
def patched_cli(monkeypatch, qwen_adapter):
    calls: list[dict] = []

    def fake_build(model, device, dtype, lens_mode, trust_remote_code):
        calls.append(
            {
                "model": model,
                "device": device,
                "dtype": dtype,
                "lens_mode": lens_mode,
                "trust_remote_code": trust_remote_code,
            }
        )
        return LogitLensAnalyzer(adapter=qwen_adapter, lens_mode=lens_mode)

    monkeypatch.setattr(cli, "_build_analyzer", fake_build)
    return calls


def test_analyze_command(patched_cli, tmp_path):
    pytest.importorskip("plotly")
    json_path = tmp_path / "result.json"
    html_path = tmp_path / "heatmap.html"
    result = runner.invoke(
        cli.app,
        [
            "analyze",
            "--model",
            "tiny",
            "--text",
            "the capital of japan is tokyo",
            "--token-index",
            "4",
            "--top-k",
            "3",
            "--lens-mode",
            "raw",
            "--json",
            str(json_path),
            "--html",
            str(html_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert patched_cli[0]["model"] == "tiny"
    assert patched_cli[0]["lens_mode"].value == "raw"
    assert "Layer 0 (embeddings)" in result.output
    assert "Layer 3" in result.output
    assert "<-- analyzed" in result.output
    assert "Top-1 trajectory (raw lens)" in result.output
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["lens_mode"] == "raw"
    assert payload["token_index"] == 4
    assert len(payload["layers"]) == 4
    assert html_path.exists()


def test_analyze_show_limits_rows(patched_cli):
    result = runner.invoke(
        cli.app, ["analyze", "--text", "the capital of japan is", "--top-k", "5", "--show", "2"]
    )
    assert result.exit_code == 0, result.output
    assert "   3. " not in result.output


def test_analyze_out_of_range_index_fails(patched_cli):
    result = runner.invoke(cli.app, ["analyze", "--text", "the capital", "--token-index", "9"])
    assert result.exit_code != 0


def test_compare_command(patched_cli, tmp_path):
    json_path = tmp_path / "compare.json"
    result = runner.invoke(
        cli.app,
        [
            "compare",
            "--text-a",
            "the capital of japan is",
            "--text-b",
            "sales were 150 yen",
            "--top-k",
            "3",
            "--json",
            str(json_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "A top-1" in result.output and "B top-1" in result.output
    assert "diverges at layer" in result.output or "agree at every layer" in result.output
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert len(payload["top1_agreement"]) == 4


def test_visualize_from_json(sample_result, tmp_path):
    pytest.importorskip("plotly")
    json_path = sample_result.save_json(tmp_path / "r.json")
    for kind in ("heatmap", "trajectory"):
        html_path = tmp_path / f"{kind}.html"
        result = runner.invoke(
            cli.app, ["visualize", str(json_path), "--html", str(html_path), "--kind", kind]
        )
        assert result.exit_code == 0, result.output
        assert html_path.exists()
    bad = runner.invoke(cli.app, ["visualize", str(json_path), "--html", "x.html", "--kind", "pie"])
    assert bad.exit_code != 0
