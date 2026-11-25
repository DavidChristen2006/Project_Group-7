"""Concise, read-only overview of saved historical ML experiments.

This page only reads local research outputs. It never trains a model,
queries LSEG, changes model selection, or makes a current recommendation.
Full validation and decision audits remain on separate research pages.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from data_pipeline.historical_ml_view import (
    PROBABILITY_COLUMNS,
    REGRESSION_COLUMNS,
    read_forecasts,
)
from data_pipeline.ml_comparison_dashboard import (
    load_saved_metrics,
    rank_validation_models,
)


def _saved_forecast_files(project_root: Path) -> list[Path]:
    """Find available 63-session forecast tables under the repository root.

    Args:
        project_root: Directory containing the ``data/processed`` tree.

    Returns:
        The saved CSVs, most recently modified first. An empty list means
        no predictions are available, not that stock risk is zero.
    """
    folder = project_root / "data/processed/ml_inference/equity_63d_v1"
    if not folder.is_dir():
        return []
    return sorted(
        folder.glob("risk_forecasts_*.csv"),
        key=lambda path: (path.stat().st_mtime, path.name),
        reverse=True,
    )


def _render_forecast_summary(project_root: Path) -> None:
    """Summarize one security's stored 63-session historical risk forecast.

    Args:
        project_root: The repository directory containing saved forecasts.

    Notes:
        The chart presents predicted probabilities, not calibrated event
        frequencies or live risk forecasts.
    """
    st.subheader("1. Historical stock-risk snapshot")
    files = _saved_forecast_files(project_root)
    if not files:
        st.info("No saved stock-risk forecast CSVs were found.")
        return

    selected = files[0]
    try:
        frame = read_forecasts(selected)
    except (OSError, ValueError, KeyError) as exc:
        st.warning(f"Saved forecast cannot be read: {exc}")
        return

    universe = sorted(frame["Underlying RIC"].astype(str).tolist())
    default = universe.index("NVDA.OQ") if "NVDA.OQ" in universe else 0
    ric = st.selectbox(
        "Historical security",
        universe,
        index=default,
        key="overview_historical_ric",
    )
    record = frame.loc[frame["Underlying RIC"].eq(ric)].iloc[0]

    st.caption(
        f"Saved decision date: {frame['Decision Date'].iloc[0].date()} · "
        f"Securities: {len(frame):,} · Model horizon: 63 sessions"
    )
    first, second, third = st.columns(3)
    for container, (label, column) in zip(
        (first, second, third), REGRESSION_COLUMNS.items()
    ):
        container.metric(label, f"{float(record[column]):.1%}")

    values = [float(record[column]) * 100 for column in PROBABILITY_COLUMNS.values()]
    graph = go.Figure(go.Bar(
        x=list(PROBABILITY_COLUMNS),
        y=values,
        text=[f"{value:.1f}%" for value in values],
        textposition="outside",
        hovertemplate="%{x}: %{y:.1f}%<extra></extra>",
    ))
    graph.update_yaxes(title_text="Estimated probability (%)", range=[0, 100])
    graph.update_layout(height=320, margin=dict(l=10, r=10, t=15, b=15))
    st.plotly_chart(graph, use_container_width=True, key="historical_overview_loss_bars")
    st.caption(
        "Predicted loss-threshold probabilities are historical, not independently "
        "calibrated, and exclude hedging costs and option premium outcomes."
    )
    if "Membership Source" in frame and str(record["Membership Source"]) != "OBSERVED_SNAPSHOT_MATCH":
        st.warning("Historical index-membership evidence for this security is provisional.")


def _render_model_summary(project_root: Path) -> None:
    """Graph available validation leaders without using test data to select them.

    Args:
        project_root: Repository directory holding saved validation tables.

    Notes:
        Regression and classification target scales are not pooled;
        their distinct metrics are displayed separately.
    """
    st.subheader("2. Validation model comparison")
    folder = project_root / "data/processed/ml_baselines/equity_63d_suite_v1/full"
    try:
        metrics, _audit = load_saved_metrics(folder)
    except (FileNotFoundError, ValueError, OSError) as exc:
        st.info(f"Saved validation metrics unavailable: {exc}")
        return

    if metrics.empty:
        st.info("No recorded regression or classification validation metrics found.")
        return

    for task in ("Regression", "Classification"):
        subset = metrics.loc[metrics["Task"].eq(task)]
        if subset.empty:
            continue
        targets = sorted(subset["Target"].unique())
        target = st.selectbox(
            f"{task} outcome",
            targets,
            key=f"overview_{task.lower()}_target",
        )
        try:
            ranking, winner = rank_validation_models(metrics, target)
        except ValueError as exc:
            st.info(f"Cannot rank {task.lower()} metrics: {exc}")
            continue
        if ranking.empty:
            continue

        metric = (
            "MAE" if task == "Regression" and ranking["MAE"].notna().any()
            else "RMSE" if task == "Regression"
            else "Brier" if ranking["Brier"].notna().any()
            else "ROC AUC"
        )
        visible = ranking.loc[ranking[metric].notna(), ["Model", metric]].head(8)
        if visible.empty:
            continue
        if winner:
            st.write(f"**Recorded validation leader:** {winner} · Metric: {metric}")
        st.bar_chart(visible.set_index("Model")[[metric]])

    st.caption(
        "Model rankings use saved validation results only. Lower MAE, RMSE "
        "and Brier are preferable; higher ROC AUC is preferable. "
        "A prediction leader is not necessarily the best hedge policy."
    )


def render_historical_research_overview(project_root: str | Path) -> None:
    """Show a concise landing page for the separate historical research views.

    Args:
        project_root: Root of the downloaded repository; no user-entered
            absolute data paths are required.

    Notes:
        A missing dataset disables only its own section. Audit-grade
        experiments remain accessible from the research navigation menu.
    """
    root = Path(project_root).resolve()
    st.header("Historical ML Research — Overview")
    st.info(
        "This page reads saved historical forecasts and validation metrics. "
        "It does not retrain a model, obtain new prices or generate a live hedge."
    )
    _render_forecast_summary(root)
    st.divider()
    _render_model_summary(root)
    st.caption(
        "For walk-forward hedging outcomes, rule-policy comparisons and "
        "matched-date decision regret, select Hedging Performance or "
        "Decision Quality Audit from the sidebar."
    )
