"""Plotly charts for hedge and ML datasets; no live data requests.

These plotting functions only display already-computed results. They never
change risk estimates, model rankings or financial policy decisions.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def strategy_tail_loss_chart(table: pd.DataFrame, risk_col: str) -> go.Figure:
    """Visualize historical 95% tail-loss estimates by hedge strategy.

    Args:
        table: Previously calibrated per-strategy historical outcomes.
        risk_col: Column holding comparable tail-loss amounts.

    Returns:
        Plotly bar chart in the same financial units as the source data.
    """
    required = {"Strategy", risk_col}
    if not required.issubset(table):
        raise ValueError(f"Missing chart fields: {sorted(required-set(table))}")
    view = table.sort_values(risk_col, ascending=True).copy()
    fig = px.bar(view, x="Strategy", y=risk_col, title="Severe-tail loss by strategy")
    fig.update_layout(xaxis_title="", yaxis_title="Historical 95% CVaR loss")
    return fig


def strategy_tradeoff_chart(
    table: pd.DataFrame, risk_col: str, return_col: str,
) -> go.Figure:
    """Plot the historical return versus severe-loss trade-off.

    Args:
        table: Calculated strategy outcomes for matching observations.
        risk_col: Historical 95% tail-loss column.
        return_col: Mean historical P&L column.

    Returns:
        Interactive risk–return scatter plot, without invented scores.
    """
    required = {"Strategy", risk_col, return_col}
    if not required.issubset(table):
        raise ValueError(f"Missing chart fields: {sorted(required-set(table))}")
    fig = px.scatter(
        table, x=risk_col, y=return_col, hover_name="Strategy",
        text="Strategy", title="Risk–return trade-off",
    )
    fig.update_traces(textposition="top center", marker={"size": 11})
    fig.update_layout(xaxis_title="Historical 95% CVaR loss (lower is better)",
                      yaxis_title="Average scenario P&L (higher is better)")
    return fig


def model_ranking_chart(
    table: pd.DataFrame, *, metric: str, title: str,
    name_col: str = "Model",
) -> go.Figure:
    """Compare recorded ML results without changing their ordering.

    Args:
        table: Source evaluation results already selected and validated.
        metric: Recorded numeric metric to display.
        title: Chart title describing split and metric.
        name_col: Column identifying evaluated models.

    Returns:
        Interactive horizontal model comparison.
    """
    required = {name_col, metric}
    if not required.issubset(table):
        raise ValueError(f"Missing chart fields: {sorted(required-set(table))}")
    view = table[[name_col, metric]].copy()
    view[metric] = pd.to_numeric(view[metric], errors="coerce")
    view = view.dropna(subset=[metric]).sort_values(metric)
    fig = px.bar(view, y=name_col, x=metric, orientation="h", title=title)
    fig.update_layout(yaxis_title="", xaxis_title=metric)
    return fig
