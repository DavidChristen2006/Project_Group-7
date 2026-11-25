"""Test portable packaging and default presentation without LSEG access."""

from __future__ import annotations

import ast
import pathlib
import zipfile

import pandas as pd
import pytest

from dashboard_visuals import (
    model_ranking_chart, strategy_tail_loss_chart,
    strategy_tradeoff_chart,
)
from project_paths import DATA_DIR, PROJECT_ROOT, data_path, output_dir


def test_portable_paths_do_not_use_working_directory(tmp_path, monkeypatch):
    """Resolve repository data under an unrelated process CWD."""
    monkeypatch.chdir(tmp_path)
    assert data_path("processed", "something.csv") == (
        PROJECT_ROOT / "data/processed/something.csv"
    )
    assert DATA_DIR.is_absolute()
    with pytest.raises(ValueError):
        data_path("..", "outside.csv")


def test_output_folder_is_created(tmp_path, monkeypatch):
    """An explicit output helper creates its directory without CWD use."""
    target = output_dir("refactor_test_only")
    try:
        assert target.is_dir()
    finally:
        target.rmdir()


def test_visualizations_preserve_input_data():
    """All Plotly charts use recorded data without retraining models."""
    frame = pd.DataFrame({
        "Strategy": ["No Hedge", "ETF 50%"],
        "Tail": [150.0, 75.0], "Average": [5.0, 1.0],
    })
    backup = frame.copy(deep=True)
    assert strategy_tail_loss_chart(frame, "Tail").data
    assert strategy_tradeoff_chart(frame, "Tail", "Average").data
    pd.testing.assert_frame_equal(frame, backup)
    with pytest.raises(ValueError):
        strategy_tradeoff_chart(frame, "absent", "Average")
    scores = pd.DataFrame({"Model": ["OLS", "HGB"], "MAE": [3.0, 1.0]})
    assert len(model_ranking_chart(scores, metric="MAE", title="Recorded").data) == 1


def test_no_private_config_in_distribution():
    """Never publish an actual LSEG session credential file."""
    assert not (PROJECT_ROOT / "lseg-data.config.json").exists()
    template = (PROJECT_ROOT / "lseg-data.config.example.json").read_text()
    assert "REPLACE_WITH" in template


def test_default_dashboard_exposes_payoffs_and_heatmap():
    """There is no Simple/Detailed gate hiding default graphs."""
    entry = (PROJECT_ROOT / "hedging_app.py").read_text()
    overview = (PROJECT_ROOT / "data_pipeline/simple_overview.py").read_text()
    assert "render_direct_options(result)" in entry
    assert "render_cross_charts(" in entry
    assert "make_joint_heatmap(" in entry
    assert "friendly_hedge_view" not in (entry + overview)
    assert '"Detailed research"' not in (entry + overview)
    assert "strategy_tradeoff_chart(" in overview


def test_python_modules_compile_and_are_documented():
    """Retained modules have syntactically valid module and function docs."""
    for path in PROJECT_ROOT.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        assert ast.get_docstring(tree), f"Missing module docstring: {path}"
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                assert ast.get_docstring(node), f"Missing function/class docs: {path}: {node.lineno}"


def test_legacy_archive_preserves_optional_sources():
    """Archived research remains recoverable without external services."""
    archive = PROJECT_ROOT / "research_archive/optional_legacy_research_sources.zip"
    with zipfile.ZipFile(archive) as z:
        assert len([name for name in z.namelist() if name.endswith(".py")]) > 80


def test_unhedged_baseline_without_etf_or_historical_data():
    """Preserve the No Hedge baseline when cross-market data are unavailable."""
    from types import SimpleNamespace

    from data_pipeline.simple_overview import build_strategy_overview

    snapshot = {
        "inputs": SimpleNamespace(spot=250.0, shares=100, ric="NVDA.O"),
        "meta": {"retrieved_at_new_york": "2026-11-23T10:00:00"},
    }
    table, ready = build_strategy_overview(snapshot)
    assert ready is False
    assert table["Strategy"].tolist() == ["No Hedge"]
    assert table["ETF shares"].tolist() == [0]


def test_legacy_archive_modules_compile():
    """Ensure archived scripts remain intact even if not directly installed."""
    archive = PROJECT_ROOT / "research_archive/optional_legacy_research_sources.zip"
    with zipfile.ZipFile(archive) as files:
        for name in files.namelist():
            if not name.endswith(".py"):
                continue
            tree = ast.parse(files.read(name).decode("utf-8"), filename=name)
            assert ast.get_docstring(tree), f"Missing archive docs in {name}"
