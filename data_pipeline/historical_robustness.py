
"""Historical robustness audit of observed stock/ETF scenarios.

Uses existing horizon-matched historical scenario records.

Compares:
- All overlapping historical windows
- Three separate non-overlapping window selections
- Consecutive two-year historical periods
- Four downside/upside objectives

No new LSEG requests, model training, or option backtesting.

Non-overlapping phases are separate sensitivity checks.
Do not pool the phases as independent experiments.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data_pipeline.observed_joint_scenarios import (
    HISTORICAL_OBJECTIVES,
)


REQUIRED = {
    "Window end",
    "Strategy",
    "Stock return",
    "ETF return",
    "Total P&L",
    "Benefit vs No Hedge",
}


# =========================================================
# 1. VALIDATE HISTORICAL SCENARIO RECORDS
# =========================================================

def _checked_panel(
    details: pd.DataFrame,
) -> pd.DataFrame:

    """Handle the internal checked panel step.

    Args:
        details: Input to the _checked_panel operation.

    Returns:
        Computed result or status for the caller.
    """
    if (
        not isinstance(details, pd.DataFrame)
        or details.empty
    ):
        raise ValueError(
            "Observed scenario table is missing or empty"
        )

    missing = REQUIRED - set(details.columns)

    if missing:
        raise ValueError(
            f"Missing observed fields: {sorted(missing)}"
        )

    panel = details.copy()

    dates = pd.to_datetime(
        panel["Window end"],
        errors="coerce",
        utc=True,
    )

    if dates.isna().any():
        raise ValueError(
            "Invalid historical window dates"
        )

    panel["Window end"] = (
        dates.dt.tz_convert(None)
        .dt.normalize()
    )

    panel["Strategy"] = (
        panel["Strategy"]
        .astype("string")
        .str.strip()
    )

    if (
        panel["Strategy"].isna().any()
        or panel["Strategy"].eq("").any()
    ):
        raise ValueError(
            "Missing strategy name"
        )

    numeric = [
        "Stock return",
        "ETF return",
        "Total P&L",
        "Benefit vs No Hedge",
    ]

    for column in numeric:
        panel[column] = pd.to_numeric(
            panel[column],
            errors="coerce",
        )

    if not np.isfinite(
        panel[numeric].to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "Nonfinite historical returns or P&L"
        )

    if (
        panel[
            ["Stock return", "ETF return"]
        ] < -1
    ).any().any():
        raise ValueError(
            "Historical returns cannot fall below -100%"
        )

    if panel.duplicated(
        ["Window end", "Strategy"]
    ).any():
        raise ValueError(
            "Duplicate date/strategy outcome"
        )

    strategies = set(
        panel["Strategy"]
    )

    if "No Hedge" not in strategies:
        raise ValueError(
            "No Hedge baseline is missing"
        )

    count = len(strategies)

    # Every strategy must be evaluated
    # on exactly the same historical dates.
    if not (
        panel.groupby("Window end")["Strategy"]
        .nunique()
        .eq(count)
        .all()
    ):
        raise ValueError(
            "Strategies do not share identical dates"
        )

    # All strategies on one date must use
    # the same observed stock/ETF return pair.
    if (
        panel.groupby("Window end")[
            ["Stock return", "ETF return"]
        ]
        .nunique() > 1
    ).any().any():
        raise ValueError(
            "Stock/ETF returns differ across policies on a date"
        )

    baseline = (
        panel.loc[
            panel["Strategy"].eq("No Hedge"),
            ["Window end", "Total P&L"],
        ]
        .rename(
            columns={
                "Total P&L": "No-hedge P&L"
            }
        )
    )

    audit = panel.merge(
        baseline,
        on="Window end",
        validate="many_to_one",
    )

    calculated_benefit = (
        audit["Total P&L"]
        - audit["No-hedge P&L"]
    )

    if not np.allclose(
        calculated_benefit,
        audit["Benefit vs No Hedge"],
        atol=1e-5,
        rtol=1e-9,
    ):
        raise ValueError(
            "Recorded benefits do not match No Hedge P&L"
        )

    if not np.allclose(
        audit.loc[
            audit["Strategy"].eq("No Hedge"),
            "Benefit vs No Hedge",
        ],
        0.0,
        atol=1e-6,
    ):
        raise ValueError(
            "No Hedge must have zero benefit"
        )

    return (
        panel.sort_values(
            ["Window end", "Strategy"]
        )
        .reset_index(drop=True)
    )


# =========================================================
# 2. CALCULATE METRICS FOR ONE HISTORICAL SAMPLE
# =========================================================

def _segment_stats(
    panel: pd.DataFrame,
    *,
    label: str,
    kind: str,
    stock_notional: float,
    downside_weight: float,
) -> list[dict]:

    """Handle the internal segment stats step.

    Args:
        panel: Input to the _segment_stats operation.
        label: Input to the _segment_stats operation.
        kind: Input to the _segment_stats operation.
        stock_notional: Input to the _segment_stats operation.
        downside_weight: Input to the _segment_stats operation.

    Returns:
        Computed result or status for the caller.
    """
    rows = []

    for strategy, group in panel.groupby(
        "Strategy",
        sort=True,
    ):

        downside = group.loc[
            group["Stock return"] < 0
        ]

        upside = group.loc[
            group["Stock return"] > 0
        ]

        down_pnl = downside[
            "Total P&L"
        ]

        up_pnl = upside[
            "Total P&L"
        ]

        worst = (
            float(down_pnl.min())
            if len(downside)
            else np.nan
        )

        tail = (
            float(down_pnl.quantile(0.05))
            if len(downside)
            else np.nan
        )

        mean_down = (
            float(down_pnl.mean())
            if len(downside)
            else np.nan
        )

        mean_up = (
            float(up_pnl.mean())
            if len(upside)
            else np.nan
        )

        mean_benefit = (
            float(
                downside[
                    "Benefit vs No Hedge"
                ].mean()
            )
            if len(downside)
            else np.nan
        )

        improvement_frequency = (
            float(
                (
                    downside[
                        "Benefit vs No Hedge"
                    ] > 0
                ).mean()
            )
            if len(downside)
            else np.nan
        )

        # All scores are based on P&L normalized
        # by the same original stock notional.
        scores = {
            HISTORICAL_OBJECTIVES[0]: worst,

            HISTORICAL_OBJECTIVES[1]: tail,

            HISTORICAL_OBJECTIVES[2]: (
                downside_weight * mean_down
                + (1 - downside_weight) * mean_up
                if len(downside) and len(upside)
                else np.nan
            ),

            HISTORICAL_OBJECTIVES[3]: mean_up,
        }

        for objective, score in scores.items():

            rows.append({
                "Sample": label,
                "Sample type": kind,
                "Objective": objective,
                "Strategy": str(strategy),
                "Windows": int(len(group)),
                "Stock-down windows": int(
                    len(downside)
                ),
                "Score": float(
                    score / stock_notional
                ),
                "Worst stock-down P&L": worst,
                "Downside P&L p05": tail,
                "Mean downside benefit": mean_benefit,
                "Downside improvement frequency":
                    improvement_frequency,
                "Mean stock-up P&L": mean_up,
            })

    return rows


# =========================================================
# 3. COMPARE HISTORICAL SAMPLES
# =========================================================

def historical_robustness_tables(
    details: pd.DataFrame,
    *,
    horizon_sessions: int,
    stock_notional: float,
    downside_weight: float = 0.60,
    period_years: int = 2,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Return strategy metrics, winners and sample coverage.

    Disjoint samples select every H-th valid window end,
    where H is the hedge horizon in trading sessions.

    Since input window ends originate from the complete
    NYSE-session calibration, this is a conservative
    non-overlap selection when some dates are missing.
    """

    if (
        isinstance(horizon_sessions, bool)
        or not isinstance(horizon_sessions, int)
        or horizon_sessions < 1
    ):
        raise ValueError(
            "Horizon must be a positive integer number of sessions"
        )

    if (
        isinstance(period_years, bool)
        or not isinstance(period_years, int)
        or period_years < 1
    ):
        raise ValueError(
            "Period length must be a positive integer"
        )

    if (
        not np.isfinite(float(stock_notional))
        or stock_notional <= 0
    ):
        raise ValueError(
            "Stock notional must be positive and finite"
        )

    if (
        not np.isfinite(float(downside_weight))
        or not 0 <= downside_weight <= 1
    ):
        raise ValueError(
            "Downside weight must be between 0 and 1"
        )

    panel = _checked_panel(
        details
    )

    ends = pd.DatetimeIndex(
        panel["Window end"].unique()
    ).sort_values()

    samples = [
        (
            "All overlapping windows",
            "Overlapping",
            panel,
        )
    ]

    # -----------------------------------------------------
    # NON-OVERLAPPING SENSITIVITY SAMPLES
    # -----------------------------------------------------

    # Three distinct phase offsets.
    # Each phase contains non-overlapping windows.
    # The different phases are NOT independent samples.

    phases = sorted(set([
        0,
        horizon_sessions // 3,
        2 * horizon_sessions // 3,
    ]))

    for phase in phases:

        selected = ends[
            phase::horizon_sessions
        ]

        if len(selected):

            sample = panel.loc[
                panel["Window end"].isin(
                    selected
                )
            ].copy()

            samples.append((
                f"Disjoint phase +{phase}",
                "Disjoint",
                sample,
            ))

    # -----------------------------------------------------
    # TWO-YEAR HISTORICAL PERIODS
    # -----------------------------------------------------

    first_year = int(
        ends.min().year
    )

    last_year = int(
        ends.max().year
    )

    for start in range(
        first_year,
        last_year + 1,
        period_years,
    ):

        finish = (
            start + period_years - 1
        )

        period = panel.loc[
            panel["Window end"]
            .dt.year
            .between(start, finish)
        ].copy()

        if not period.empty:

            samples.append((
                f"Window ends {start}–{finish}",
                "End-year period",
                period,
            ))

    # -----------------------------------------------------
    # CALCULATE ALL STRATEGY SCORES
    # -----------------------------------------------------

    rows = []
    coverage = []

    for label, kind, sample in samples:

        number_of_dates = (
            sample["Window end"].nunique()
        )

        downside_dates = (
            sample.loc[
                sample["Stock return"] < 0,
                "Window end",
            ]
            .nunique()
        )

        coverage.append({
            "Sample": label,
            "Sample type": kind,
            "Windows": int(number_of_dates),
            "Stock-down windows": int(
                downside_dates
            ),
        })

        rows.extend(
            _segment_stats(
                sample,
                label=label,
                kind=kind,
                stock_notional=float(
                    stock_notional
                ),
                downside_weight=float(
                    downside_weight
                ),
            )
        )

    metrics = pd.DataFrame(
        rows
    )

    # -----------------------------------------------------
    # IDENTIFY THE TOP STRATEGY FOR EACH OBJECTIVE
    # -----------------------------------------------------

    winners = []

    for (label, objective), group in (
        metrics.groupby(
            ["Sample", "Objective"],
            sort=False,
        )
    ):

        evaluable = (
            group.loc[
                np.isfinite(group["Score"])
            ]
            .sort_values(
                ["Score", "Strategy"],
                ascending=[False, True],
                kind="stable",
            )
        )

        coverage_row = next(
            item
            for item in coverage
            if item["Sample"] == label
        )

        winner = (
            str(evaluable.iloc[0]["Strategy"])
            if len(evaluable)
            else "Insufficient observations"
        )

        top_score = (
            float(evaluable.iloc[0]["Score"])
            if len(evaluable)
            else np.nan
        )

        winners.append({
            "Sample": label,
            "Sample type":
                coverage_row["Sample type"],
            "Objective": objective,
            "Windows":
                coverage_row["Windows"],
            "Stock-down windows":
                coverage_row["Stock-down windows"],
            "Top strategy": winner,
            "Top score": top_score,
        })

    return (
        metrics,
        pd.DataFrame(winners),
        pd.DataFrame(coverage),
    )


# =========================================================
# 4. STREAMLIT ROBUSTNESS DASHBOARD
# =========================================================

def render_historical_robustness(
    details: pd.DataFrame,
    *,
    horizon_sessions: int,
    stock_notional: float,
) -> None:
    """Render robustness results below observed joint scenarios."""

    import streamlit as st

    st.subheader(
        "Historical robustness audit"
    )

    st.caption(
        "The same observed stock–ETF return pairs are "
        "evaluated across two-year historical periods "
        "and different non-overlapping window selections. "
        "Non-overlapping phases must not be pooled as "
        "independent samples. No new LSEG request or ML training."
    )

    objective = st.selectbox(
        "Robustness objective",
        HISTORICAL_OBJECTIVES,
        key="joint_robustness_objective",
    )

    weight = 0.60

    if objective == HISTORICAL_OBJECTIVES[2]:

        weight = (
            st.slider(
                "Robustness downside weight (%)",
                min_value=0,
                max_value=100,
                value=60,
                step=5,
                key="joint_robustness_downside_weight",
            )
            / 100
        )

    try:

        metrics, winners, coverage = (
            historical_robustness_tables(
                details,
                horizon_sessions=horizon_sessions,
                stock_notional=stock_notional,
                downside_weight=weight,
            )
        )

    except (
        ValueError,
        TypeError,
        KeyError,
    ) as exc:

        st.error(
            f"Cannot calculate robustness audit: {exc}"
        )

        return

    # -----------------------------------------------------
    # CHOOSE THE HISTORICAL SAMPLE
    # -----------------------------------------------------

    sample = st.selectbox(
        "Historical evaluation sample",
        coverage["Sample"].tolist(),
        key="joint_robustness_sample",
    )

    chosen = (
        metrics.loc[
            metrics["Sample"].eq(sample)
            & metrics["Objective"].eq(objective)
        ]
        .sort_values(
            ["Score", "Strategy"],
            ascending=[False, True],
            kind="stable",
        )
    )

    # -----------------------------------------------------
    # SHOW STRATEGY METRICS
    # -----------------------------------------------------

    st.dataframe(
        chosen[
            [
                "Strategy",
                "Windows",
                "Stock-down windows",
                "Score",
                "Worst stock-down P&L",
                "Downside P&L p05",
                "Mean downside benefit",
                "Downside improvement frequency",
                "Mean stock-up P&L",
            ]
        ]
        .style.format({
            "Score": "{:+.2%}",
            "Worst stock-down P&L": "{:,.2f}",
            "Downside P&L p05": "{:,.2f}",
            "Mean downside benefit": "{:+,.2f}",
            "Downside improvement frequency": "{:.1%}",
            "Mean stock-up P&L": "{:,.2f}",
        }),
        hide_index=True,
        use_container_width=True,
    )

    # -----------------------------------------------------
    # COMPARE WINNERS ACROSS ALL SAMPLES
    # -----------------------------------------------------

    relevant = winners.loc[
        winners["Objective"].eq(
            objective
        )
    ]

    st.write(
        "**Top-ranked strategy in each historical sample**"
    )

    st.dataframe(
        relevant[
            [
                "Sample",
                "Windows",
                "Stock-down windows",
                "Top strategy",
            ]
        ],
        hide_index=True,
        use_container_width=True,
    )

    # -----------------------------------------------------
    # DATA LIMITATIONS
    # -----------------------------------------------------

    selected_coverage = coverage.loc[
        coverage["Sample"].eq(sample)
    ].iloc[0]

    if (
        int(
            selected_coverage[
                "Stock-down windows"
            ]
        ) < 20
    ):

        st.warning(
            "Fewer than 20 stock-down observations "
            "in this selected sample. Tail quantiles "
            "and strategy rankings are highly unstable."
        )

    st.caption(
        "Two-year groups use window END dates; some "
        "63-session return windows begin in a previous "
        "period. The three disjoint phases are separate "
        "sensitivity checks, not three independent "
        "experiments. Financing, borrowing costs, "
        "dividends and implementation slippage are "
        "excluded. No statistical significance is claimed."
    )

    with st.expander(
        "Coverage for all historical samples"
    ):

        st.dataframe(
            coverage,
            hide_index=True,
            use_container_width=True,
        )
