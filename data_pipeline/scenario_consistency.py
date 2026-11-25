
"""Compare three stock/ETF share-hedge analyses without pooling scores.

Independent and linked stresses use hypothetical return grids.
Observed outcomes replay actual historical paired returns.

Two-year winners are descriptive historical evidence,
not independent validation trials.

No ML inference, retraining, or additional LSEG requests.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from math import isfinite

import pandas as pd

from data_pipeline.historical_robustness import (
    historical_robustness_tables,
)

from data_pipeline.observed_joint_scenarios import (
    HISTORICAL_OBJECTIVES,
    compare_observed_joint_scenarios,
)

from data_pipeline.risk_aware_recommendations import (
    Candidate,
    evaluate_candidates,
)


@dataclass(frozen=True)
class ConsistencyAudit:
    """Represent ConsistencyAudit within the hedge research workflow.
    """
    comparison: pd.DataFrame
    period_winners: pd.DataFrame
    phase_winners: pd.DataFrame
    independent_ranking: pd.DataFrame
    linked_ranking: pd.DataFrame
    observed_ranking: pd.DataFrame
    observed_windows: int
    eligible_periods: int


def compare_three_scenario_methods(
    candidates: list[Candidate],
    observed_windows: pd.DataFrame,
    *,
    evaluation_date: date,
    stock_spot: float,
    shares: int,
    etf_spot: float,
    fee_per_leg_contract: float,
    horizon_sessions: int,
    beta_stock_on_etf: float,
    r_squared: float,
    residual_width: float,
    independent_etf_width: float = 0.20,
    worst_stock_return: float = -0.30,
    minimum_period_stock_down: int = 20,
) -> ConsistencyAudit:
    """Compare share-only strategies using three risk models.

    Each method ranks by its own worst stock-down P&L.
    Different scenario sets must not have pooled scores.
    """

    if (
        isinstance(horizon_sessions, bool)
        or not isinstance(horizon_sessions, int)
        or horizon_sessions < 1
    ):
        raise ValueError(
            "Hedge horizon must be a positive integer"
        )

    if (
        isinstance(minimum_period_stock_down, bool)
        or not isinstance(minimum_period_stock_down, int)
        or minimum_period_stock_down < 1
    ):
        raise ValueError(
            "Minimum stock-down count must be positive"
        )

    if not isinstance(evaluation_date, date):
        raise ValueError(
            "Evaluation date must be a date"
        )

    for name, width in (
        ("residual_width", residual_width),
        ("independent_etf_width", independent_etf_width),
    ):

        if (
            not isfinite(float(width))
            or not 0 <= float(width) <= 1
        ):
            raise ValueError(
                f"{name} must be between 0 and 1"
            )

    # -----------------------------------------------------
    # SHARE-ONLY STRATEGIES
    # -----------------------------------------------------

    # Historical adjusted stock and ETF prices alone
    # cannot reconstruct option premiums or Greeks.

    share_only = [
        candidate
        for candidate in candidates
        if (
            candidate.expiry is None
            and not candidate.legs
        )
    ]

    names = [
        candidate.name
        for candidate in share_only
    ]

    if (
        names.count("No Hedge") != 1
        or len(names) != len(set(names))
    ):
        raise ValueError(
            "Unique share-only strategies and exactly "
            "one No Hedge baseline are required"
        )

    if (
        len(names) < 2
        or not any(
            candidate.etf_shares != 0
            for candidate in share_only
            if candidate.name != "No Hedge"
        )
    ):
        raise ValueError(
            "At least one nonzero ETF share hedge is required"
        )

    # -----------------------------------------------------
    # COMMON POSITION INPUTS
    # -----------------------------------------------------

    common = {
        "candidates": share_only,
        "evaluation_date": evaluation_date,
        "stock_spot": stock_spot,
        "shares": shares,
        "etf_spot": etf_spot,
        "fee_per_leg_contract": fee_per_leg_contract,
        "objective": "Capital protection",
        "worst_stock_return": worst_stock_return,
    }

    # -----------------------------------------------------
    # ANALYSIS 1: INDEPENDENT STRESS
    # -----------------------------------------------------

    independent, _ = evaluate_candidates(
        **common,
        stress_model="independent",
        max_abs_etf_return=independent_etf_width,
    )

    # -----------------------------------------------------
    # ANALYSIS 2: HISTORICALLY LINKED STRESS
    # -----------------------------------------------------

    linked, _ = evaluate_candidates(
        **common,
        stress_model="historical_conditional",
        stock_on_etf_beta=beta_stock_on_etf,
        stock_etf_r_squared=r_squared,
        max_abs_etf_return=residual_width,
    )

    # -----------------------------------------------------
    # ANALYSIS 3: OBSERVED HISTORICAL JOINT RETURNS
    # -----------------------------------------------------

    observed, observed_details = (
        compare_observed_joint_scenarios(
            share_only,
            observed_windows,
            stock_spot=stock_spot,
            shares=shares,
            etf_spot=etf_spot,
            objective=HISTORICAL_OBJECTIVES[0],
        )
    )

    # -----------------------------------------------------
    # VERIFY IDENTICAL STRATEGY COVERAGE
    # -----------------------------------------------------

    expected = set(names)

    for label, frame in (
        ("Independent", independent),
        ("Linked", linked),
        ("Observed", observed),
    ):

        if (
            set(frame["Strategy"]) != expected
            or frame["Strategy"].duplicated().any()
        ):
            raise ValueError(
                f"{label} comparison has inconsistent strategies"
            )

    # -----------------------------------------------------
    # HISTORICAL TWO-YEAR AND DISJOINT-PHASE EVIDENCE
    # -----------------------------------------------------

    _, winners, _ = historical_robustness_tables(
        observed_details,
        horizon_sessions=horizon_sessions,
        stock_notional=(
            float(stock_spot) * int(shares)
        ),
    )

    selected = winners.loc[
        winners["Objective"].eq(
            HISTORICAL_OBJECTIVES[0]
        )
    ].copy()

    # A minimum observation count is a reporting
    # safeguard, not a statistical significance test.

    periods = selected.loc[
        selected["Sample type"].eq(
            "End-year period"
        )
        & selected["Stock-down windows"].ge(
            minimum_period_stock_down
        )
        & selected["Top strategy"].ne(
            "Insufficient observations"
        )
    ].copy()

    phases = selected.loc[
        selected["Sample type"].eq(
            "Disjoint"
        )
    ].copy()

    period_counts = (
        periods["Top strategy"]
        .value_counts()
        .to_dict()
    )

    # -----------------------------------------------------
    # COMBINE RANKS WITHOUT COMBINING SCORES
    # -----------------------------------------------------

    independent_by_name = independent.set_index(
        "Strategy"
    )

    linked_by_name = linked.set_index(
        "Strategy"
    )

    observed_by_name = observed.set_index(
        "Strategy"
    )

    rows = []

    for name in names:

        independent_rank = int(
            independent_by_name.at[
                name,
                "Rank",
            ]
        )

        linked_rank = int(
            linked_by_name.at[
                name,
                "Rank",
            ]
        )

        observed_rank = int(
            observed_by_name.at[
                name,
                "Rank",
            ]
        )

        ranks = [
            independent_rank,
            linked_rank,
            observed_rank,
        ]

        rows.append({
            "Strategy": name,

            "Independent rank":
                independent_rank,

            "Independent worst P&L": float(
                independent_by_name.at[
                    name,
                    "Worst downside P&L",
                ]
            ),

            "Linked rank":
                linked_rank,

            "Linked worst P&L": float(
                linked_by_name.at[
                    name,
                    "Worst downside P&L",
                ]
            ),

            "Observed rank":
                observed_rank,

            "Observed worst P&L": float(
                observed_by_name.at[
                    name,
                    "Worst stock-down P&L",
                ]
            ),

            "Observed downside p05 P&L": float(
                observed_by_name.at[
                    name,
                    "Downside P&L 5th pct",
                ]
            ),

            "Rank range": (
                max(ranks) - min(ranks)
            ),

            "Top in all three": all(
                rank == 1
                for rank in ranks
            ),

            "Two-year period wins": int(
                period_counts.get(name, 0)
            ),
        })

    comparison = (
        pd.DataFrame(rows)
        .sort_values(
            ["Observed rank", "Strategy"],
            kind="stable",
        )
        .reset_index(drop=True)
    )

    return ConsistencyAudit(
        comparison=comparison,
        period_winners=periods,
        phase_winners=phases,
        independent_ranking=independent,
        linked_ranking=linked,
        observed_ranking=observed,
        observed_windows=len(observed_windows),
        eligible_periods=len(periods),
    )


def render_scenario_consistency(
    candidates: list[Candidate],
    observed_windows: pd.DataFrame,
    *,
    evaluation_date: date,
    stock_spot: float,
    shares: int,
    etf_spot: float,
    fee_per_leg_contract: float,
    horizon_sessions: int,
    beta_stock_on_etf: float,
    r_squared: float,
    residual_width: float,
) -> None:
    """Display three-method strategy comparison in Streamlit."""

    import streamlit as st

    st.subheader(
        "Cross-method hedge consistency"
    )

    st.caption(
        "Independent stress: stock down to -30%, "
        "ETF +/-20%. Linked stress: stock down to -30% "
        "with the calibrated ETF residual range. "
        "Observed analysis: actual historical paired "
        "horizon returns. These methods use different "
        "scenarios and their scores are not pooled."
    )

    try:

        audit = compare_three_scenario_methods(
            candidates,
            observed_windows,
            evaluation_date=evaluation_date,
            stock_spot=stock_spot,
            shares=shares,
            etf_spot=etf_spot,
            fee_per_leg_contract=fee_per_leg_contract,
            horizon_sessions=horizon_sessions,
            beta_stock_on_etf=beta_stock_on_etf,
            r_squared=r_squared,
            residual_width=residual_width,
        )

    except (
        ValueError,
        TypeError,
        KeyError,
        AttributeError,
    ) as exc:

        st.warning(
            f"Cross-method comparison unavailable: {exc}"
        )

        return

    st.write(
        f"**Historical rolling windows:** "
        f"{audit.observed_windows:,} | "
        f"**Eligible two-year periods:** "
        f"{audit.eligible_periods} | "
        f"**Linked residual width:** "
        f"±{residual_width:.0%}"
    )

    st.dataframe(
        audit.comparison.style.format({
            "Independent worst P&L": "{:,.2f}",
            "Linked worst P&L": "{:,.2f}",
            "Observed worst P&L": "{:,.2f}",
            "Observed downside p05 P&L": "{:,.2f}",
        }),
        hide_index=True,
        use_container_width=True,
    )

    if not audit.comparison[
        "Top in all three"
    ].any():

        st.info(
            "No share-only strategy ranks first "
            "under all three analyses. "
            "The preferred hedge is sensitive "
            "to the scenario assumptions."
        )

    else:

        top_names = audit.comparison.loc[
            audit.comparison["Top in all three"],
            "Strategy",
        ].tolist()

        st.info(
            "Top-ranked in all three analyses: "
            + ", ".join(top_names)
        )

    st.caption(
        "Worst P&L amounts across the three methods "
        "are not directly comparable because their "
        "scenario ranges differ. Two-year winners use "
        "overlapping historical windows and are not "
        "independent validation results. Options, ETF "
        "borrowing costs, dividends, financing, margin "
        "and execution slippage are excluded."
    )

    with st.expander(
        "Two-year period and disjoint-phase winners"
    ):

        st.write(
            "**Two-year historical periods**"
        )

        st.dataframe(
            audit.period_winners[
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

        st.write(
            "**Disjoint-phase sensitivity checks**"
        )

        st.dataframe(
            audit.phase_winners[
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

        st.caption(
            "The disjoint phases are alternative selections "
            "from the same historical record. They should "
            "not be pooled as independent samples."
        )
