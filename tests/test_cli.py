from __future__ import annotations

import json
from pathlib import Path

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
    assert "Instability from layer" in result.output
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
    assert "KL(A‖B)" in result.output and "KL(B‖A)" in result.output and "JS" in result.output
    assert "diverges at layer" in result.output or "agree at every layer" in result.output
    assert "JS first >=" in result.output
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert len(payload["top1_agreement"]) == 4
    assert len(payload["layers"]) == 4
    assert payload["kind"] == "prompt"


def test_compare_command_token_index_b(patched_cli):
    result = runner.invoke(
        cli.app,
        [
            "compare",
            "--text-a",
            "the capital of japan is",
            "--text-b",
            "the capital of japan is tokyo",
            "--token-index",
            "-1",
            "--token-index-b",
            "-2",
            "--top-k",
            "3",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "analyzed 'is' at index 4" in result.output


def test_compare_command_no_kl(patched_cli, tmp_path):
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
            "--no-kl",
            "--json",
            str(json_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "no distributions" in result.output
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert all(layer["kl_a_b"] is None for layer in payload["layers"])
    assert payload["metadata"]["keep_log_probs"] is False


def test_compare_command_html(patched_cli, tmp_path):
    pytest.importorskip("plotly")
    html_path = tmp_path / "compare.html"
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
            "--html",
            str(html_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert html_path.exists()


@pytest.fixture
def patched_cli_two_models(monkeypatch, qwen_adapter, llama_adapter):
    """Like ``patched_cli``, but dispatches on model name to one of two tiny models."""
    calls: list[dict] = []
    adapters = {"tiny-qwen": qwen_adapter, "tiny-llama": llama_adapter}

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
        return LogitLensAnalyzer(adapter=adapters[model], lens_mode=lens_mode)

    monkeypatch.setattr(cli, "_build_analyzer", fake_build)
    return calls


def test_compare_models_command(patched_cli_two_models, tmp_path):
    json_path = tmp_path / "cmp.json"
    result = runner.invoke(
        cli.app,
        [
            "compare-models",
            "--model-a",
            "tiny-qwen",
            "--model-b",
            "tiny-llama",
            "--text",
            "the capital of japan is",
            "--json",
            str(json_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert [c["model"] for c in patched_cli_two_models] == ["tiny-qwen", "tiny-llama"]
    assert "A: tiny-qwen" in result.output
    assert "B: tiny-llama" in result.output
    assert "vocab: identical" in result.output
    assert "A ly" in result.output and "B ly" in result.output
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["kind"] == "model"


def test_compare_models_command_token_index_and_char_index_conflict(patched_cli_two_models):
    result = runner.invoke(
        cli.app,
        [
            "compare-models",
            "--model-a",
            "tiny-qwen",
            "--model-b",
            "tiny-llama",
            "--text",
            "the capital of japan is",
            "--token-index",
            "-1",
            "--char-index",
            "3",
        ],
    )
    assert result.exit_code != 0


def test_compare_models_command_concurrent(patched_cli_two_models):
    result = runner.invoke(
        cli.app,
        [
            "compare-models",
            "--model-a",
            "tiny-qwen",
            "--model-b",
            "tiny-llama",
            "--text",
            "the capital of japan is",
            "--concurrent",
        ],
    )
    assert result.exit_code == 0, result.output
    assert [c["model"] for c in patched_cli_two_models] == ["tiny-qwen", "tiny-llama"]


def _save_two_results(tmp_path) -> tuple[Path, Path]:
    a_json = tmp_path / "a.json"
    b_json = tmp_path / "b.json"
    for model, path in (("tiny-qwen", a_json), ("tiny-llama", b_json)):
        result = runner.invoke(
            cli.app,
            ["analyze", "--model", model, "--text", "the capital of japan is", "--json", str(path)],
        )
        assert result.exit_code == 0, result.output
    return a_json, b_json


def test_compare_results_command(patched_cli_two_models, tmp_path):
    a_json, b_json = _save_two_results(tmp_path)
    out_json = tmp_path / "cmp.json"
    result = runner.invoke(
        cli.app,
        ["compare-results", str(a_json), str(b_json), "--json", str(out_json)],
    )
    assert result.exit_code == 0, result.output
    assert "no distributions" in result.output
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["kind"] == "model"


def test_compare_results_command_html(patched_cli_two_models, tmp_path):
    pytest.importorskip("plotly")
    a_json, b_json = _save_two_results(tmp_path)
    html_path = tmp_path / "cmp.html"
    result = runner.invoke(
        cli.app, ["compare-results", str(a_json), str(b_json), "--html", str(html_path)]
    )
    assert result.exit_code == 0, result.output
    assert html_path.exists()


def test_diff_command(patched_cli, tmp_path):
    json_path = tmp_path / "diff.json"
    result = runner.invoke(
        cli.app,
        [
            "diff",
            "--source",
            "sales were 150 yen",
            "--output",
            "sales were 1500 yen",
            "--top-k",
            "3",
            "--json",
            str(json_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Diff spans (1):" in result.output
    assert "[replace]" in result.output
    assert "Position " in result.output
    assert "settle layer" in result.output
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["prompt_length"] == 4
    assert len(payload["positions"]) == 1


def test_diff_command_no_differences(patched_cli):
    result = runner.invoke(
        cli.app,
        ["diff", "--source", "sales were 150 yen", "--output", "sales were 150 yen"],
    )
    assert result.exit_code == 0, result.output
    assert "No differences between source and output." in result.output


def test_diff_command_requires_source_or_source_file(patched_cli):
    result = runner.invoke(cli.app, ["diff", "--output", "sales were 150 yen"])
    assert result.exit_code != 0


def test_scan_command(patched_cli, tmp_path):
    json_path = tmp_path / "scan.json"
    result = runner.invoke(
        cli.app,
        [
            "scan",
            "--prompt",
            "the capital of japan is",
            "--output",
            "tokyo osaka",
            "--source",
            "the capital of japan is tokyo",
            "--top-k",
            "3",
            "--json",
            str(json_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Anomaly report:" in result.output
    assert "score" in result.output
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert "positions" in payload
    assert len(payload["positions"]) == 2


def test_scan_command_requires_output(patched_cli):
    result = runner.invoke(cli.app, ["scan", "--prompt", "the capital of japan is"])
    assert result.exit_code != 0


def test_metrics_command(sample_result, tmp_path):
    json_path = sample_result.save_json(tmp_path / "result.json")
    out_path = tmp_path / "metrics.json"
    result = runner.invoke(
        cli.app,
        ["metrics", str(json_path), "--from-layer", "1", "--json", str(out_path)],
    )
    assert result.exit_code == 0, result.output
    assert "Instability from layer 1" in result.output
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    expected = sample_result.instability(from_layer=1)
    assert payload == expected.to_dict()


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
