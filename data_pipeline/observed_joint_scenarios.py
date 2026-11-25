
"""Research-only historical replay of paired stock/ETF horizon returns.

Input windows must contain observed, horizon-matched returns.
Only share-only hedges, including No Hedge, are evaluated.

No probability forecast or trading recommendation is produced.
"""

from __future__ import annotations

from math import isfinite

import numpy as np
import pandas as pd


HISTORICAL_OBJECTIVES = (
    "Capital protection — worst observed downside",
    "Capital protection — 5th percentile downside",
    "Balanced downside and upside",
    "Preserve upside",
)


def compare_observed_joint_scenarios(
    candidates: list,
    windows: pd.DataFrame,
    *,
    stock_spot: float,
    shares: int,
    etf_spot: float,
    objective: str = HISTORICAL_OBJECTIVES[0],
    downside_weight: float = 0.60,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Replay historical paired returns on current position sizes.

    Results describe observed historical scenarios.
    Rolling windows overlap and are not independent.
    """

    if objective not in HISTORICAL_OBJECTIVES:
        raise ValueError(
            "Unknown historical comparison objective"
        )

    for label, value in (
        ("Stock price", stock_spot),
        ("ETF price", etf_spot),
    ):
        if (
            not isfinite(float(value))
            or float(value) <= 0
        ):
            raise ValueError(
                f"{label} must be positive and finite"
            )

    if (
        isinstance(shares, bool)
        or not isinstance(shares, int)
        or shares <= 0
    ):
        raise ValueError(
            "Stock shares must be a positive integer"
        )

    if (
        not isfinite(float(downside_weight))
        or not 0 <= downside_weight <= 1
    ):
        raise ValueError(
            "Downside weight must lie between zero and one"
        )

    # -----------------------------------------------------
    # VALIDATE HISTORICAL WINDOWS
    # -----------------------------------------------------

    required = {
        "Stock return",
        "ETF return",
    }

    if (
        not isinstance(windows, pd.DataFrame)
        or not required.issubset(windows.columns)
    ):
        raise ValueError(
            "Horizon-matched paired stock/ETF "
            "returns are required"
        )

    if windows.empty:
        raise ValueError(
            "No historical paired windows"
        )

    if windows.index.has_duplicates:
        raise ValueError(
            "Duplicate historical window end dates"
        )

    returns = (
        windows[
            ["Stock return", "ETF return"]
        ]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
    )

    if (
        returns.isna().any().any()
        or not np.isfinite(
            returns.to_numpy(dtype=float)
        ).all()
    ):
        raise ValueError(
            "Historical returns must be finite, "
            "with no missing values"
        )

    if (returns < -1).any().any():
        raise ValueError(
            "Historical stock/ETF return cannot "
            "fall below -100%"
        )

    if not (returns["Stock return"] < 0).any():
        raise ValueError(
            "No observed stock-down windows available"
        )

    # -----------------------------------------------------
    # ELIGIBLE STRATEGIES
    # -----------------------------------------------------

    # Options are excluded: underlying closing prices
    # do not reconstruct historical option premiums,
    # implied volatility, or expiration outcomes.

    eligible = [
        candidate
        for candidate in candidates
        if (
            getattr(candidate, "expiry", None)
            is None
            and not getattr(candidate, "legs", ())
        )
    ]

    names = [
        str(candidate.name)
        for candidate in eligible
    ]

    if (
        len(names) != len(set(names))
        or names.count("No Hedge") != 1
    ):
        raise ValueError(
            "Unique share-only strategies and "
            "one No Hedge are required"
        )

    # -----------------------------------------------------
    # APPLY HISTORICAL RETURNS TO CURRENT POSITION
    # -----------------------------------------------------

    initial_stock_notional = (
        float(stock_spot) * shares
    )

    initial_etf_price = float(
        etf_spot
    )

    stock_move = (
        returns["Stock return"]
        .to_numpy(dtype=float)
    )

    etf_move = (
        returns["ETF return"]
        .to_numpy(dtype=float)
    )

    unhedged = (
        initial_stock_notional
        * stock_move
    )

    rows = []

    for candidate in eligible:

        qty = candidate.etf_shares

        if (
            isinstance(qty, bool)
            or not isinstance(qty, int)
        ):
            raise ValueError(
                "ETF shares must be an integer: "
                f"{candidate.name}"
            )

        total = (
            unhedged
            + qty * initial_etf_price * etf_move
        )

        rows.append(
            pd.DataFrame({
                "Window end": returns.index,
                "Strategy": candidate.name,
                "Stock return": stock_move,
                "ETF return": etf_move,
                "Total P&L": total,
                "Benefit vs No Hedge": (
                    total - unhedged
                ),
                "ETF shares (signed)": qty,
            })
        )

    details = pd.concat(
        rows,
        ignore_index=True,
    )

    # -----------------------------------------------------
    # MEASURE HISTORICAL DOWNSIDE AND UPSIDE OUTCOMES
    # -----------------------------------------------------

    summary_rows = []

    for candidate in eligible:

        group = details.loc[
            details["Strategy"].eq(
                candidate.name
            )
        ]

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

        # ---------------------------------------------
        # HISTORICAL OBJECTIVE SCORE
        # ---------------------------------------------

        if objective == HISTORICAL_OBJECTIVES[0]:

            score = (
                float(down_pnl.min())
                / initial_stock_notional
            )

        elif objective == HISTORICAL_OBJECTIVES[1]:

            score = (
                float(down_pnl.quantile(0.05))
                / initial_stock_notional
            )

        elif objective == HISTORICAL_OBJECTIVES[2]:

            if upside.empty:
                raise ValueError(
                    "No stock-up windows "
                    "for a balanced score"
                )

            score = (
                downside_weight
                * float(down_pnl.mean())
                + (1 - downside_weight)
                * float(up_pnl.mean())
            ) / initial_stock_notional

        else:

            if upside.empty:
                raise ValueError(
                    "No stock-up windows "
                    "for an upside score"
                )

            score = (
                float(up_pnl.mean())
                / initial_stock_notional
            )

        # ---------------------------------------------
        # RECORDED HISTORICAL PERFORMANCE
        # ---------------------------------------------

        summary_rows.append({
            "Strategy": candidate.name,
            "Objective score": score,
            "Observed windows": len(group),
            "Stock-down windows": len(downside),
            "Worst stock-down P&L": float(
                down_pnl.min()
            ),
            "Downside P&L 5th pct": float(
                down_pnl.quantile(0.05)
            ),
            "Mean downside benefit": float(
                downside[
                    "Benefit vs No Hedge"
                ].mean()
            ),
            "Downside improvement frequency": float(
                (
                    downside[
                        "Benefit vs No Hedge"
                    ] > 0
                ).mean()
            ),
            "Mean upside P&L": (
                float(up_pnl.mean())
                if not upside.empty
                else np.nan
            ),
            "ETF shares (signed)":
                candidate.etf_shares,
        })

    ranking = (
        pd.DataFrame(summary_rows)
        .sort_values(
            ["Objective score", "Strategy"],
            ascending=[False, True],
            kind="stable",
        )
        .reset_index(drop=True)
    )

    ranking.insert(
        0,
        "Rank",
        range(1, len(ranking) + 1),
    )

    return ranking, details


def render_observed_joint_comparison(
    hedge_result: dict,
    cross_result: dict | None,
) -> None:
    """Display historical stock/ETF share-hedge outcomes."""

    import streamlit as st

    from data_pipeline.risk_aware_recommendations import (
        build_candidates,
    )

    if not cross_result:
        return

    meta = hedge_result["meta"]

    if (
        cross_result.get("snapshot")
        != meta.get("retrieved_at_new_york")
    ):
        return

    st.subheader(
        "Observed historical joint-scenario comparison"
    )

    st.caption(
        "Replays historically observed paired stock–ETF "
        "horizon returns on current position sizes. "
        "This is descriptive overlapping-window research, "
        "not an executable historical backtest, "
        "ML forecast or guaranteed loss limit."
    )

    # -----------------------------------------------------
    # READ THE CALIBRATION ALREADY STORED IN STREAMLIT
    # -----------------------------------------------------

    saved = st.session_state.get(
        "empirical_basis_result"
    )

    if (
        not isinstance(saved, dict)
        or not {
            "key",
            "calibration",
            "windows",
        }.issubset(saved)
    ):
        st.info(
            "Run 'Calibrate from LSEG adjusted daily closes' "
            "above first."
        )
        return

    key = saved["key"]

    if (
        not isinstance(key, tuple)
        or len(key) != 7
    ):
        st.warning(
            "Unrecognized cached calibration metadata. "
            "Recalibrate."
        )
        return

    inputs = hedge_result["inputs"]

    report = cross_result.get(
        "beta_report"
    )

    r2 = (
        report.get("r_squared")
        if isinstance(report, dict)
        else None
    )

    # -----------------------------------------------------
    # PREVENT REUSE FOR ANOTHER STOCK, ETF OR HORIZON
    # -----------------------------------------------------

    try:

        compatible = (
            key[0] == str(inputs.ric)
            and key[1] == str(
                cross_result["etf_ric"]
            )
            and int(key[2]) == int(
                meta["research_horizon_sessions"]
            )
            and np.isclose(
                float(key[5]),
                float(cross_result["beta"]),
                rtol=0,
                atol=1e-12,
            )
            and r2 is not None
            and np.isclose(
                float(key[6]),
                float(r2),
                rtol=0,
                atol=1e-12,
            )
        )

    except (
        ValueError,
        TypeError,
        KeyError,
        IndexError,
    ):
        compatible = False

    if not compatible:

        st.warning(
            "Cached calibration belongs to different "
            "instruments, hedge horizon or beta estimate. "
            "Recalibrate before comparing."
        )

        return

    # -----------------------------------------------------
    # PREPARE CURRENT SHARE-HEDGE STRATEGIES
    # -----------------------------------------------------

    try:

        (
            candidates,
            target,
            etf_spot,
            notices,
        ) = build_candidates(
            hedge_result,
            cross_result,
        )

        if etf_spot is None:

            st.info(
                "No ETF price available for historical replay."
            )

            return

        objective = st.selectbox(
            "Historical outcome priority",
            HISTORICAL_OBJECTIVES,
            key="historical_joint_objective",
        )

        weight = 0.60

        if objective == "Balanced downside and upside":

            weight = (
                st.slider(
                    "Historical downside weight (%)",
                    min_value=0,
                    max_value=100,
                    value=60,
                    step=5,
                    key="historical_joint_downside_weight",
                )
                / 100.0
            )

        ranking, details = (
            compare_observed_joint_scenarios(
                candidates,
                saved["windows"],
                stock_spot=float(inputs.spot),
                shares=int(inputs.shares),
                etf_spot=float(etf_spot),
                objective=objective,
                downside_weight=weight,
            )
        )

    except (
        ValueError,
        TypeError,
        KeyError,
        AttributeError,
    ) as exc:

        st.error(
            f"Cannot evaluate observed joint scenarios: {exc}"
        )

        return

    # -----------------------------------------------------
    # SHOW HISTORICAL SAMPLE COVERAGE
    # -----------------------------------------------------

    calibration = saved[
        "calibration"
    ]

    st.write(
        f"**{len(saved['windows']):,}** overlapping "
        f"{int(calibration.horizon_sessions)}-session "
        "historical windows; "
        f"**{int(calibration.nonoverlapping_windows)}** "
        "complete non-overlapping windows "
        "in the coverage audit."
    )

    if calibration.nonoverlapping_windows < 40:

        st.warning(
            "Limited non-overlapping historical periods. "
            "Tail quantiles, extreme observations and "
            "strategy rankings are exploratory and "
            "sensitive to historical market regimes."
        )

    # -----------------------------------------------------
    # RANKED HISTORICAL SHARE-HEDGE RESULTS
    # -----------------------------------------------------

    st.dataframe(
        ranking.style.format({
            "Objective score": "{:+.2%}",
            "Worst stock-down P&L": "{:,.2f}",
            "Downside P&L 5th pct": "{:,.2f}",
            "Mean downside benefit": "{:+,.2f}",
            "Downside improvement frequency": "{:.1%}",
            "Mean upside P&L": "{:,.2f}",
        }),
        hide_index=True,
        use_container_width=True,
    )

    if len(ranking) == 1:

        st.info(
            "Only No Hedge is available for this "
            "historical comparison."
        )

    st.caption(
        "Option strategies are excluded because "
        "historical stock and ETF closing prices alone "
        "cannot reconstruct their premiums, implied "
        "volatility or expiration payoffs. "
        "ETF borrowing, dividends, financing and "
        "margin costs are also excluded."
    )

    with st.expander(
        "Inspect historical stock–ETF outcomes by window"
    ):

        st.dataframe(
            details,
            hide_index=True,
            use_container_width=True,
        )

    # -----------------------------------------------------
    # HISTORICAL ROBUSTNESS AUDIT
    # -----------------------------------------------------

    from data_pipeline.historical_robustness import (
        render_historical_robustness,
    )

    # -----------------------------------------------------
    # HISTORICAL CVaR / EXPECTED SHORTFALL
    # -----------------------------------------------------

    from data_pipeline.tail_risk_analysis import (
        render_historical_tail_risk,
    )

    render_historical_tail_risk(
        details,
        stock_notional=(
            float(inputs.spot)
            * int(inputs.shares)
        ),
        etf_spot=float(etf_spot),
        horizon_sessions=int(
            calibration.horizon_sessions
        ),
        nonoverlapping_windows=int(
            calibration.nonoverlapping_windows
        ),
    )

    render_historical_robustness(
        details,
        horizon_sessions=int(
            calibration.horizon_sessions
        ),
        stock_notional=(
            float(inputs.spot)
            * int(inputs.shares)
        ),
    )

    # -----------------------------------------------------
    # CROSS-METHOD HEDGE CONSISTENCY
    # -----------------------------------------------------

    from data_pipeline.historical_basis_calibration import (
        suggested_slider_width,
    )

    from data_pipeline.scenario_consistency import (
        render_scenario_consistency,
    )

    # Use the empirical width already computed from
    # the historical 5th/95th residual percentiles.
    #
    # Example:
    # 10.53% absolute residual -> 15% rounded width.

    empirical_width = (
        suggested_slider_width(calibration) / 100.0
    )

    render_scenario_consistency(
        candidates=candidates,
        observed_windows=saved["windows"],
        evaluation_date=target,
        stock_spot=float(inputs.spot),
        shares=int(inputs.shares),
        etf_spot=float(etf_spot),
        fee_per_leg_contract=float(
            inputs.fee_per_contract
        ),
        horizon_sessions=int(
            calibration.horizon_sessions
        ),
        beta_stock_on_etf=float(
            cross_result["beta"]
        ),
        r_squared=float(r2),
        residual_width=empirical_width,
    )


