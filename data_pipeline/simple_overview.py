
"""Simple overview for the Equity Hedge Explorer.

Uses existing stock snapshots, ETF hedge quantities and
previously calibrated historical stock/ETF returns.

No new LSEG requests, ML inference or trade execution.
"""

from __future__ import annotations

from math import isfinite

import numpy as np
import pandas as pd

from data_pipeline.tail_risk_analysis import (
    empirical_var_cvar,
)


GOALS = (
    "Reduce severe losses",
    "Preserve average returns",
    "Compare risk and return",
)

RISK = "95% historical tail loss"
AVERAGE = "Average historical scenario P&L"


def _positive(value: object, label: str) -> float:

    try:
        number = float(value)

    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid {label}"
        ) from exc

    if not isfinite(number) or number <= 0:
        raise ValueError(
            f"{label} must be positive and finite"
        )

    return number


def _matching_calibration(
    hedge_result: dict,
    cross_result: dict,
    cache: object,
) -> bool:
    """Confirm that cached research belongs to these inputs."""

    if (
        not isinstance(cache, dict)
        or not {"key", "windows"}.issubset(cache)
    ):
        return False

    key = cache["key"]

    if not isinstance(key, tuple) or len(key) != 7:
        return False

    meta = hedge_result["meta"]
    inputs = hedge_result["inputs"]
    report = cross_result.get("beta_report")

    if not isinstance(report, dict):
        return False

    try:
        return (
            key[0] == str(inputs.ric)
            and key[1] == str(cross_result["etf_ric"])
            and int(key[2])
            == int(meta["research_horizon_sessions"])
            and np.isclose(
                float(key[5]),
                float(cross_result["beta"]),
                rtol=0,
                atol=1e-12,
            )
            and np.isclose(
                float(key[6]),
                float(report["r_squared"]),
                rtol=0,
                atol=1e-12,
            )
        )

    except (
        ValueError,
        TypeError,
        KeyError,
        OverflowError,
    ):
        return False


def build_strategy_overview(
    hedge_result: dict,
    cross_result: dict | None = None,
    calibration_cache: dict | None = None,
    *,
    confidence: float = 0.95,
) -> tuple[pd.DataFrame, bool]:
    """Build a readable share-hedge comparison.

    Returns:
        table: strategy comparison
        ready: whether historical risk values are available

    Historical risk uses the same paired horizon returns
    as the existing research modules.

    Borrow costs, dividends and trading costs are excluded.
    """

    inputs = hedge_result["inputs"]
    meta = hedge_result["meta"]

    spot = _positive(
        inputs.spot,
        "Stock price",
    )

    if (
        isinstance(inputs.shares, bool)
        or not isinstance(inputs.shares, int)
        or inputs.shares < 1
    ):
        raise ValueError(
            "Shares must be a positive integer"
        )

    if (
        not 0 < float(confidence) < 1
        or not isfinite(float(confidence))
    ):
        raise ValueError(
            "Confidence must be between zero and one"
        )

    rows = [
        {
            "Strategy": "No Hedge",
            "ETF shares": 0,
            "Hedge fraction": 0.0,
        }
    ]

    cross_ok = (
        isinstance(cross_result, dict)
        and cross_result.get("snapshot")
        == meta.get("retrieved_at_new_york")
    )

    if not cross_ok:
        return pd.DataFrame(rows), False

    etf_spot = _positive(
        cross_result["etf_spot"],
        "ETF price",
    )

    selected = cross_result.get("selected")

    if not isinstance(selected, pd.DataFrame):
        raise ValueError(
            "ETF-share hedge quantities are missing"
        )

    required = {
        "Hedge Fraction",
        "ETF Position Shares (signed)",
    }

    if not required.issubset(selected.columns):
        raise ValueError(
            "Missing ETF share fields: "
            f"{sorted(required - set(selected))}"
        )

    # -----------------------------------------------------
    # BUILD AVAILABLE ETF SHARE STRATEGIES
    # -----------------------------------------------------

    for raw_fraction, group in selected.groupby(
        "Hedge Fraction",
        sort=True,
    ):

        fraction = float(raw_fraction)

        values = pd.to_numeric(
            group["ETF Position Shares (signed)"],
            errors="coerce",
        )

        if not isfinite(fraction) or not 0 < fraction <= 1:
            raise ValueError(
                "Invalid hedge fraction"
            )

        if (
            values.isna().any()
            or not np.isfinite(
                values.to_numpy(dtype=float)
            ).all()
            or values.nunique() != 1
        ):
            raise ValueError(
                "Inconsistent ETF-share quantity"
            )

        quantity = float(values.iloc[0])

        if not quantity.is_integer():
            raise ValueError(
                "ETF shares must be whole numbers"
            )

        if int(quantity) == 0:
            continue

        rows.append({
            "Strategy": (
                f"{cross_result['etf_ric']} "
                f"shares {fraction:.0%}"
            ),
            "ETF shares": int(quantity),
            "Hedge fraction": fraction,
        })

    table = pd.DataFrame(rows)

    if table["Strategy"].duplicated().any():
        raise ValueError(
            "Duplicate hedge strategy names"
        )

    # -----------------------------------------------------
    # CHECK AVAILABLE HISTORICAL CALIBRATION
    # -----------------------------------------------------

    if not _matching_calibration(
        hedge_result,
        cross_result,
        calibration_cache,
    ):
        return table, False

    windows = calibration_cache["windows"]

    required_returns = {
        "Stock return",
        "ETF return",
    }

    if (
        not isinstance(windows, pd.DataFrame)
        or windows.empty
        or not required_returns.issubset(windows)
    ):
        return table, False

    pairs = (
        windows[
            ["Stock return", "ETF return"]
        ]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
    )

    values = pairs.to_numpy(dtype=float)

    if (
        not np.isfinite(values).all()
        or np.any(values < -1)
    ):
        raise ValueError(
            "Invalid historical paired returns"
        )

    if windows.index.has_duplicates:
        raise ValueError(
            "Duplicate historical scenario dates"
        )

    # -----------------------------------------------------
    # REUSE OBSERVED HISTORICAL RETURN PAIRS
    # -----------------------------------------------------

    stock_pnl = (
        spot
        * inputs.shares
        * pairs["Stock return"].to_numpy(dtype=float)
    )

    etf_returns = (
        pairs["ETF return"].to_numpy(dtype=float)
    )

    for i, row in table.iterrows():

        pnl = (
            stock_pnl
            + int(row["ETF shares"])
            * etf_spot
            * etf_returns
        )

        _, cvar, _ = empirical_var_cvar(
            -pnl,
            confidence=confidence,
        )

        table.at[i, RISK] = cvar

        table.at[i, AVERAGE] = float(
            pnl.mean()
        )

    return table, True


def summarize_user_goal(
    table: pd.DataFrame,
    goal: str,
) -> tuple[list[str], str]:
    """Select a descriptive leader or a trade-off shortlist."""

    if goal not in GOALS:
        raise ValueError(
            "Unrecognized goal"
        )

    if not {
        "Strategy",
        RISK,
        AVERAGE,
    }.issubset(table):
        raise ValueError(
            "Historical risk results are unavailable"
        )

    if (
        table.empty
        or table[[RISK, AVERAGE]]
        .isna()
        .any()
        .any()
    ):
        raise ValueError(
            "Historical risk results contain missing values"
        )

    if goal == GOALS[0]:

        best = (
            table.sort_values(
                [RISK, "Strategy"],
                ascending=[True, True],
            )
            .iloc[0]
        )

        return (
            [str(best["Strategy"])],
            "Lowest historical 95% tail loss",
        )

    if goal == GOALS[1]:

        best = (
            table.sort_values(
                [AVERAGE, "Strategy"],
                ascending=[False, True],
            )
            .iloc[0]
        )

        return (
            [str(best["Strategy"])],
            "Highest average historical scenario P&L",
        )

    # -----------------------------------------------------
    # BALANCED: FIND NON-DOMINATED STRATEGIES
    # -----------------------------------------------------

    # Do not invent a combined risk/return score.
    #
    # A strategy is dominated if another strategy has:
    # - No higher severe-tail loss
    # - No lower mean P&L
    # - A strict improvement in at least one measure

    efficient = []

    for _, row in table.iterrows():

        dominated = (
            (table[RISK] <= row[RISK] + 1e-9)
            & (table[AVERAGE] >= row[AVERAGE] - 1e-9)
            & (
                (table[RISK] < row[RISK] - 1e-9)
                | (
                    table[AVERAGE]
                    > row[AVERAGE] + 1e-9
                )
            )
        ).any()

        if not dominated:
            efficient.append(
                str(row["Strategy"])
            )

    return (
        efficient,
        "Risk–return trade-off; no arbitrary combined score",
    )

def render_simple_overview(
    hedge_result: dict,
    cross_result: dict | None = None,
) -> bool:
    """Show a simple hedge comparison with one-click calibration.

    Returns True when Detailed research is selected.
    """

    import streamlit as st

    from data_pipeline.quick_calibration import (
        fetch_quick_calibration,
    )

    inputs = hedge_result["inputs"]
    meta = hedge_result["meta"]

    st.subheader(
        "Your hedge overview"
    )

    # -----------------------------------------------------
    # DISPLAY MODE AT THE TOP
    # -----------------------------------------------------

    view = st.radio(
        "Display mode",
        (
            "Simple overview",
            "Detailed research",
        ),
        horizontal=True,
        key="friendly_hedge_view",
    )

    # -----------------------------------------------------
    # EASY-TO-READ POSITION SUMMARY
    # -----------------------------------------------------

    first, second, third = st.columns(3)

    first.metric(
        "Current position value",
        f"{float(inputs.spot) * int(inputs.shares):,.2f}",
    )

    second.metric(
        "Shares",
        f"{int(inputs.shares):,}",
    )

    third.metric(
        "Hedge period",
        f"{int(meta['research_horizon_sessions'])} sessions",
    )

    st.caption(
        f"Stock: {meta.get('user_ticker', inputs.ric)} | "
        f"Target date: {meta['research_target_date']}"
    )

    if meta.get("used_close_fallback"):

        st.caption(
            "The stock reference uses the latest available "
            "closing price, not a synchronized live "
            "execution quote."
        )

    valid_cross = (
        isinstance(cross_result, dict)
        and cross_result.get("snapshot")
        == meta.get("retrieved_at_new_york")
    )

    # -----------------------------------------------------
    # WHAT MATTERS MOST TO THE USER?
    # -----------------------------------------------------

    goal = st.radio(
        "What matters most to you?",
        GOALS,
        horizontal=True,
        index=2,
        key="friendly_hedge_goal",
    )

    cache = st.session_state.get(
        "empirical_basis_result"
    )

    # -----------------------------------------------------
    # PREPARE AVAILABLE STRATEGIES
    # -----------------------------------------------------

    try:

        table, ready = build_strategy_overview(
            hedge_result,
            cross_result if valid_cross else None,
            cache,
        )

    except (
        ValueError,
        TypeError,
        KeyError,
        AttributeError,
    ) as exc:

        st.warning(
            f"Could not prepare the comparison: {exc}"
        )

        table = pd.DataFrame([
            {
                "Strategy": "No Hedge",
                "ETF shares": 0,
            }
        ])

        ready = False

    if not valid_cross:

        st.info(
            "Evaluate a comparison ETF above to see "
            "the available cross-hedge choices."
        )

    else:

        st.caption(
            f"Comparison ETF: {cross_result['etf_ric']} | "
            f"Reference price "
            f"{float(cross_result['etf_spot']):,.2f}"
        )

        # -------------------------------------------------
        # ONE-CLICK HISTORICAL CALIBRATION
        # -------------------------------------------------

        if not ready:

            st.write(
                "**Calculate historical risk and return** "
                "to compare the available ETF share hedges "
                "on the same recorded stock–ETF scenarios."
            )

            if st.button(
                "Calculate historical risk comparison",
                type="primary",
                key="friendly_calibrate",
            ):

                with st.spinner(
                    "Retrieving historical market prices..."
                ):

                    try:

                        new_cache = fetch_quick_calibration(
                            hedge_result,
                            cross_result,
                            lookback_years=8,
                        )

                        new_table, new_ready = (
                            build_strategy_overview(
                                hedge_result,
                                cross_result,
                                new_cache,
                            )
                        )

                        if not new_ready:

                            raise ValueError(
                                "Historical returns did not "
                                "match the selected position"
                            )

                        # Save the cache only after all
                        # calculations succeed.

                        st.session_state[
                            "empirical_basis_result"
                        ] = new_cache

                        table = new_table
                        ready = new_ready
                        cache = new_cache

                        st.success(
                            "Historical risk comparison is ready."
                        )

                    except Exception as exc:

                        st.error(
                            "Historical comparison is unavailable: "
                            f"{exc}. Existing ETF positions "
                            "are retained."
                        )

    # -----------------------------------------------------
    # HISTORICAL RISK / RETURN RESULTS
    # -----------------------------------------------------

    if ready:

        calibration = (
            cache.get("calibration")
            if isinstance(cache, dict)
            else None
        )

        count = (
            int(calibration.nonoverlapping_windows)
            if calibration is not None
            else None
        )

        if count is not None and count < 20:

            st.warning(
                "The history contains fewer than 20 "
                "complete non-overlapping hedge periods. "
                "Results are highly uncertain and should "
                "not decide a live trade."
            )

        selected, reason = summarize_user_goal(
            table,
            goal,
        )

        # -------------------------------------------------
        # PLAIN-LANGUAGE STRATEGY COMPARISON
        # -------------------------------------------------

        if goal == GOALS[2]:

            low_loss = table.sort_values(
                [RISK, "Strategy"],
                ascending=[True, True],
            ).iloc[0]["Strategy"]

            high_return = table.sort_values(
                [AVERAGE, "Strategy"],
                ascending=[False, True],
            ).iloc[0]["Strategy"]

            st.write(
                f"**For smaller severe losses:** {low_loss}.  "
                f"**For higher average scenario returns:** "
                f"{high_return}."
            )

            if len(selected) > 1:

                st.caption(
                    "Other available risk–return trade-offs: "
                    + ", ".join(selected)
                )

        else:

            st.write(
                "**Best in the selected historical measure:** "
                + selected[0]
            )

        st.caption(
            reason
            + ". This describes historical scenarios, "
            "not a validated live trading recommendation."
        )

        # -------------------------------------------------
        # TWO SIMPLE NUMBERS
        # -------------------------------------------------

        lowest = float(
            table[RISK].min()
        )

        baseline = float(
            table.loc[
                table["Strategy"].eq("No Hedge"),
                RISK,
            ].iloc[0]
        )

        left, right = st.columns(2)

        left.metric(
            "Smallest historical severe-tail loss",
            f"{lowest:,.0f}",
        )

        right.metric(
            "No Hedge severe-tail loss",
            f"{baseline:,.0f}",
        )

        # -------------------------------------------------
        # SHORT, READABLE STRATEGY TABLE
        # -------------------------------------------------

        if goal == GOALS[0]:

            order_by = [
                RISK,
                "Strategy",
            ]

            ascending = [
                True,
                True,
            ]

        elif goal == GOALS[1]:

            order_by = [
                AVERAGE,
                "Strategy",
            ]

            ascending = [
                False,
                True,
            ]

        else:

            order_by = [
                "Hedge fraction",
                "Strategy",
            ]

            ascending = [
                True,
                True,
            ]

        shown = (
            table.sort_values(
                order_by,
                ascending=ascending,
            )[
                [
                    "Strategy",
                    "ETF shares",
                    RISK,
                    AVERAGE,
                ]
            ]
        )

        st.dataframe(
            shown.style.format({
                RISK: "{:,.2f}",
                AVERAGE: "{:+,.2f}",
            }),
            hide_index=True,
            use_container_width=True,
        )

        st.caption(
            "Severe-tail loss is the average loss in "
            "the worst 5% of historical horizon outcomes "
            "(95% CVaR). The historical windows overlap. "
            "Short-ETF borrowing, distributions, financing, "
            "commissions and slippage are excluded. "
            "Figures assume compatible quote currencies."
        )

        if (
            isinstance(cache, dict)
            and "key" in cache
        ):

            st.caption(
                f"Historical price reference: "
                f"{cache['key'][4]}"
            )

    # -----------------------------------------------------
    # RESULTS NOT YET AVAILABLE
    # -----------------------------------------------------

    else:

        st.dataframe(
            table[
                [
                    "Strategy",
                    "ETF shares",
                ]
            ],
            hide_index=True,
            use_container_width=True,
        )

        if valid_cross:

            st.caption(
                "Hedge quantities are available, but "
                "historical loss estimates have not "
                "been calculated."
            )

    # -----------------------------------------------------
    # AVAILABLE HEDGE TYPES IN SIMPLE MODE
    # -----------------------------------------------------

    if view == "Simple overview":

        from data_pipeline.unified_hedge_choices import (
            render_unified_hedge_choices,
        )

        render_unified_hedge_choices(
            hedge_result=hedge_result,
            cross_result=(
                cross_result
                if valid_cross
                else None
            ),
        )

    return view == "Detailed research"

