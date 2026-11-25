
"""Empirical VaR and CVaR for historical stock-ETF scenarios.

Uses the existing observed historical joint-scenario records.

CVaR / Expected Shortfall is the fractionally weighted
average of the worst (1 - confidence) share of losses.

Historical rolling windows overlap and are dependent.
No new LSEG requests or ML training.
"""

from __future__ import annotations

from math import isfinite

import numpy as np
import pandas as pd

from data_pipeline.historical_robustness import (
    _checked_panel,
)


# =========================================================
# EMPIRICAL VALUE AT RISK AND EXPECTED SHORTFALL
# =========================================================

def empirical_var_cvar(
    losses: np.ndarray,
    confidence: float = 0.95,
) -> tuple[float, float, float]:
    """Return VaR, CVaR and tail-window weight.

    For four losses and confidence 62.5%,
    the tail contains 1.5 observation weights:

    CVaR = (largest loss + 0.5 * second largest) / 1.5
    """

    if (
        not isfinite(float(confidence))
        or not 0 < confidence < 1
    ):
        raise ValueError(
            "Confidence must be strictly between zero and one"
        )

    values = np.asarray(
        losses,
        dtype=float,
    )

    if (
        values.ndim != 1
        or values.size == 0
        or not np.isfinite(values).all()
    ):
        raise ValueError(
            "Loss observations must be a finite, "
            "nonempty 1D array"
        )

    n = int(values.size)

    tail_mass = (
        n * (1.0 - confidence)
    )

    descending = np.sort(
        values
    )[::-1]

    full = int(
        np.floor(tail_mass)
    )

    fractional = (
        tail_mass - full
    )

    weighted_sum = float(
        descending[:full].sum()
    )

    if fractional > 1e-12 and full < n:

        weighted_sum += (
            fractional
            * float(descending[full])
        )

    cvar = (
        weighted_sum / tail_mass
    )

    # Empirical observed-loss quantile.
    ascending = descending[::-1]

    index = max(
        0,
        min(
            n - 1,
            int(
                np.ceil(confidence * n)
            ) - 1,
        ),
    )

    var = float(
        ascending[index]
    )

    return (
        var,
        float(cvar),
        float(tail_mass),
    )


# =========================================================
# COMPARE HISTORICAL STRATEGIES
# =========================================================

def evaluate_historical_tail_risk(
    details: pd.DataFrame,
    *,
    stock_notional: float,
    etf_spot: float,
    horizon_sessions: int,
    confidence: float = 0.95,
    sample: str = "all",
    borrow_rate_annual: float = 0.0,
    dividend_yield_annual: float = 0.0,
    round_trip_fee_per_etf_share: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate risk on identical historical paired windows.

    Supports signed ETF short-share quantities.

    Illustrative ETF short cost:

        abs(ETF shares) * ETF spot
        * (annual borrow rate + annual dividend yield)
        * hedge sessions / 252

        + abs(ETF shares) * round-trip fee per ETF share

    These are user-specified assumptions,
    not verified execution costs.
    """

    if (
        not isfinite(float(stock_notional))
        or stock_notional <= 0
    ):
        raise ValueError(
            "Stock notional must be positive and finite"
        )

    if (
        not isfinite(float(etf_spot))
        or etf_spot <= 0
    ):
        raise ValueError(
            "ETF spot must be positive and finite"
        )

    if (
        isinstance(horizon_sessions, bool)
        or not isinstance(horizon_sessions, int)
        or horizon_sessions < 1
    ):
        raise ValueError(
            "Hedge horizon must be a positive integer"
        )

    if sample not in (
        "all",
        "stock_down",
    ):
        raise ValueError(
            "Sample must be all or stock_down"
        )

    for label, number in (
        ("Borrow rate", borrow_rate_annual),
        ("Dividend yield", dividend_yield_annual),
        (
            "Round-trip fee",
            round_trip_fee_per_etf_share,
        ),
    ):

        if (
            not isfinite(float(number))
            or float(number) < 0
        ):
            raise ValueError(
                f"{label} must be finite and nonnegative"
            )

    # Reuse the historical robustness validator:
    # same dates, same market returns and P&L audit.

    panel = _checked_panel(
        details
    )

    if "ETF shares (signed)" not in panel.columns:
        raise ValueError(
            "Missing recorded ETF share quantities"
        )

    # -----------------------------------------------------
    # VALIDATE ETF SHARE POSITIONS
    # -----------------------------------------------------

    group_shares = {}

    for name, group in panel.groupby(
        "Strategy",
        sort=True,
    ):

        raw = pd.to_numeric(
            group["ETF shares (signed)"],
            errors="coerce",
        )

        if (
            raw.isna().any()
            or not np.isfinite(raw).all()
            or raw.nunique() != 1
        ):
            raise ValueError(
                f"Invalid or inconsistent ETF shares for {name}"
            )

        quantity = float(
            raw.iloc[0]
        )

        if (
            not quantity.is_integer()
            or quantity > 0
        ):
            raise ValueError(
                "Only integer, short ETF share "
                f"positions supported: {name}"
            )

        if (
            name == "No Hedge"
            and quantity != 0
        ):
            raise ValueError(
                "No Hedge must have zero ETF shares"
            )

        group_shares[str(name)] = int(
            quantity
        )

    # -----------------------------------------------------
    # SELECT HISTORICAL SAMPLE
    # -----------------------------------------------------

    if sample == "stock_down":

        panel = panel.loc[
            panel["Stock return"] < 0
        ].copy()

        if panel.empty:
            raise ValueError(
                "No stock-down historical windows available"
            )

    # -----------------------------------------------------
    # HYPOTHETICAL COSTS
    # -----------------------------------------------------

    carry_rate = (
        float(borrow_rate_annual)
        + float(dividend_yield_annual)
    )

    annualization = (
        horizon_sessions / 252.0
    )

    cost_by_strategy = {
        name: (
            abs(shares)
            * float(etf_spot)
            * carry_rate
            * annualization

            + abs(shares)
            * float(
                round_trip_fee_per_etf_share
            )
        )
        for name, shares in group_shares.items()
    }

    panel["Hypothetical hedge costs"] = (
        panel["Strategy"]
        .map(cost_by_strategy)
        .astype(float)
    )

    panel["Net P&L"] = (
        panel["Total P&L"]
        - panel["Hypothetical hedge costs"]
    )

    panel["Net benefit vs No Hedge"] = (
        panel["Benefit vs No Hedge"]
        - panel["Hypothetical hedge costs"]
    )

    # Loss is the negative of the full-position P&L.
    panel["Loss"] = (
        -panel["Net P&L"]
    )

    # -----------------------------------------------------
    # VaR / CVaR PER STRATEGY
    # -----------------------------------------------------

    metrics = []

    for name, group in panel.groupby(
        "Strategy",
        sort=True,
    ):

        var, cvar, tail_weight = (
            empirical_var_cvar(
                group["Loss"].to_numpy(),
                confidence,
            )
        )

        worst = float(
            group["Loss"].max()
        )

        metrics.append({
            "Strategy": name,

            "Windows": len(group),

            "ETF shares (signed)":
                group_shares[name],

            "Hypothetical hedge costs":
                cost_by_strategy[name],

            "VaR loss": var,

            "CVaR loss": cvar,

            "CVaR / stock notional": (
                cvar / float(stock_notional)
            ),

            "Worst observed loss": worst,

            "Mean net P&L": float(
                group["Net P&L"].mean()
            ),

            "Mean net benefit vs No Hedge": float(
                group[
                    "Net benefit vs No Hedge"
                ].mean()
            ),

            "Tail window weight": tail_weight,
        })

    # Lower CVaR loss is preferred.
    ranking = (
        pd.DataFrame(metrics)
        .sort_values(
            ["CVaR loss", "Strategy"],
            ascending=[True, True],
            kind="stable",
        )
        .reset_index(drop=True)
    )

    ranking.insert(
        0,
        "Rank",
        range(1, len(ranking) + 1),
    )

    return ranking, panel


# =========================================================
# STREAMLIT INTERFACE
# =========================================================

def render_historical_tail_risk(
    details: pd.DataFrame,
    *,
    stock_notional: float,
    etf_spot: float,
    horizon_sessions: int,
    nonoverlapping_windows: int,
) -> None:
    """Display historical CVaR below observed joint scenarios."""

    import streamlit as st

    st.subheader(
        "Historical CVaR / Expected Shortfall"
    )

    st.caption(
        "Ranks share-only strategies by the average "
        "of the worst historical tail losses. "
        "CVaR is a risk measure, not a fourth "
        "scenario generator or a live probability "
        "forecast. Rolling windows overlap."
    )

    # -----------------------------------------------------
    # USER CONTROLS
    # -----------------------------------------------------

    left, right = st.columns(2)

    with left:

        confidence = st.selectbox(
            "Tail-risk confidence",
            [
                0.90,
                0.95,
                0.975,
                0.99,
            ],
            index=1,
            format_func=lambda value: (
                f"{value:.1%}"
            ),
            key="historical_es_confidence",
        )

    with right:

        sample_label = st.selectbox(
            "Historical sample",
            [
                "All paired windows",
                "Only stock-down windows",
            ],
            key="historical_es_sample",
        )

    sample = (
        "stock_down"
        if sample_label == "Only stock-down windows"
        else "all"
    )

    # -----------------------------------------------------
    # OPTIONAL COST SENSITIVITY
    # -----------------------------------------------------

    with st.expander(
        "Optional ETF short carrying/trading cost assumptions"
    ):

        st.caption(
            "These are researcher-entered assumptions, "
            "not verified execution quotes. "
            "Zero means gross P&L. "
            "Margin financing remains excluded."
        )

        c1, c2, c3 = st.columns(3)

        with c1:

            borrow_pct = st.number_input(
                "Short ETF annual borrow rate (%)",
                min_value=0.0,
                max_value=100.0,
                value=0.0,
                step=0.25,
                key="historical_es_borrow_rate",
            )

        with c2:

            dividend_pct = st.number_input(
                "ETF annual distribution yield (%)",
                min_value=0.0,
                max_value=100.0,
                value=0.0,
                step=0.25,
                key="historical_es_dividend_yield",
            )

        with c3:

            round_trip_fee = st.number_input(
                "Round-trip fee per ETF share "
                "(quote currency)",
                min_value=0.0,
                value=0.0,
                step=0.05,
                key="historical_es_round_trip_fee",
            )

    # -----------------------------------------------------
    # EVALUATE
    # -----------------------------------------------------

    try:

        ranking, audited = (
            evaluate_historical_tail_risk(
                details,
                stock_notional=stock_notional,
                etf_spot=etf_spot,
                horizon_sessions=horizon_sessions,
                confidence=confidence,
                sample=sample,
                borrow_rate_annual=(
                    float(borrow_pct) / 100.0
                ),
                dividend_yield_annual=(
                    float(dividend_pct) / 100.0
                ),
                round_trip_fee_per_etf_share=float(
                    round_trip_fee
                ),
            )
        )

    except (
        KeyError,
        TypeError,
        ValueError,
    ) as exc:

        st.error(
            f"Historical CVaR calculation unavailable: {exc}"
        )

        return

    # -----------------------------------------------------
    # SAMPLE AUDIT
    # -----------------------------------------------------

    n = int(
        ranking["Windows"].iloc[0]
    )

    tail_n = float(
        ranking["Tail window weight"].iloc[0]
    )

    st.write(
        f"**{n:,}** overlapping windows in the "
        f"selected sample; **{tail_n:.1f}** "
        "tail-window weights at "
        f"{confidence:.1%} confidence. "
        "Historical non-overlapping coverage audit: "
        f"**{nonoverlapping_windows}** complete windows "
        "across the full sample."
    )

    if (
        tail_n < 20
        or nonoverlapping_windows < 40
    ):

        st.warning(
            "Limited independent historical evidence. "
            "Tail observations overlap and rank "
            "differences must not be interpreted "
            "as statistically significant."
        )

    # -----------------------------------------------------
    # RESULTS
    # -----------------------------------------------------

    st.dataframe(
        ranking.style.format({
            "Hypothetical hedge costs": "{:,.2f}",
            "VaR loss": "{:,.2f}",
            "CVaR loss": "{:,.2f}",
            "CVaR / stock notional": "{:.2%}",
            "Worst observed loss": "{:,.2f}",
            "Mean net P&L": "{:+,.2f}",
            "Mean net benefit vs No Hedge": "{:+,.2f}",
            "Tail window weight": "{:.1f}",
        }),
        hide_index=True,
        use_container_width=True,
    )

    st.caption(
        "Positive loss values indicate losses; "
        "negative values indicate gains. "
        "Lower CVaR loss is better for the selected "
        "historical sample. Unlike maximum drawdown, "
        "these metrics use terminal P&L over the "
        "selected hedge horizon. Historical option "
        "costs and price paths are not inferred."
    )

    with st.expander(
        "Inspect audited after-cost historical tail scenarios"
    ):

        st.dataframe(
            audited,
            hide_index=True,
            use_container_width=True,
        )
