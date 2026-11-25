"""Offline regression checks for the compact calculator and saved research UI."""

from __future__ import annotations

import ast
from pathlib import Path

from project_paths import PROJECT_ROOT


def _code(path: str) -> str:
    """Read one source file relative to the installed repository."""
    return (PROJECT_ROOT / path).read_text(encoding="utf-8")


def test_one_position_form_no_currency_gate():
    """One live form must calculate stock and ETF without currency consent."""
    tree = ast.parse(_code("hedging_app.py"))
    main = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    calls = [
        node.func.attr for node in ast.walk(main)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    ]
    assert calls.count("form_submit_button") == 1
    assert "evaluate_cross_hedge" in [
        node.func.id for node in ast.walk(main)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    ]
    assert "same_currency" not in _code("hedging_app.py")
    assert "FX_DISCLAIMER" in _code("hedging_app.py")


def test_historical_overview_does_not_call_lseg_or_training():
    """The read-only historical landing page must load saved research only."""
    text = _code("data_pipeline/historical_research_overview.py")
    tree = ast.parse(text)
    assert ast.get_docstring(tree)
    assert "read_forecasts" in text
    assert "rank_validation_models" in text
    assert "open_session" not in text
    assert "train(" not in text


def test_bootstrap_runs_without_a_run_button():
    """The advanced synthetic simulation is cached and automatic."""
    text = _code("data_pipeline/block_bootstrap.py")
    assert '"Run paired moving-block bootstrap"' not in text
    assert 'saved_outcome = st.session_state.get("joint_bootstrap_result")' in text


def test_cli_does_not_require_same_currency():
    """A retired CLI argument may be accepted but cannot gate analysis."""
    text = _code("data_pipeline/estimate_etf_beta.py")
    assert "if not args.assume_same_currency:" not in text
    assert "FX conversion is omitted" in text
