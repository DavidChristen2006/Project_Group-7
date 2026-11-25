
"""User-friendly inventory of eligible hedging strategies.

Reuses the existing candidate selection and payoff calculations.

Does not retrieve new LSEG quotes, invent missing prices,
or rank strategies across different evaluation dates.
"""

from __future__ import annotations

from datetime import date
from math import isfinite

import pandas as pd

from data_pipeline.risk_aware_recommendations import (
    build_candidates,
    opening_debit,
    scenario_pnl,
)


def _valid_return(
    value: object,
    label: str,
) -> float:
    """Handle the internal valid return step.

    Args:
        value: Input to the _valid_return operation.
        label: Input to the _valid_return operation.

    Returns:
        Computed result or status for the caller.
    """
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {label}") from exc

    if not isfinite(number) or number < -1:
        raise ValueError(
            f"{label} must be finite and at least -100%"
        )

    return number


def _trade_description(
    candidate: object,
    etf_name: str,
) -> str:
    """Express hedge quantities in plain language."""

    if candidate.legs:
        pieces = []

        for leg in candidate.legs:
            action = (
                "Buy" if leg.side == 1 else "Sell"
            )
            asset = (
                "stock"
                if leg.underlying == "stock"
                else "ETF"
            )

            pieces.append(
                f"{action} {leg.contracts} "
                f"{asset} {leg.right} "
                f"@ {leg.strike:g} "
                f"(x{leg.multiplier})"
            )

        return "; ".join(pieces)

    if candidate.etf_shares:
        side = (
            "Short"
            if candidate.etf_shares < 0
            else "Buy"
        )

        return (
            f"{side} "
            f"{abs(candidate.etf_shares):,} "
            f"{etf_name} shares"
        )

    return "Keep existing stock position"


def build_unified_hedge_choices(
    hedge_result: dict,
    cross_result: dict | None = None,
    *,
    stock_return: float = -0.20,
    etf_return: float = -0.10,
) -> tuple[pd.DataFrame, list[str]]:
    """Inventory and evaluate only eligible hedge candidates.

    Option candidates are evaluated at their own expiration.
    Share-only candidates are evaluated at the hedge target.

    This is not a unified strategy ranking.
    """

    stock_return = _valid_return(
        stock_return,
        "Stock return",
    )
    etf_return = _valid_return(
        etf_return,
        "ETF return",
    )

    candidates, target, etf_spot, notices = (
        build_candidates(
            hedge_result,
            cross_result,
        )
    )

    inputs = hedge_result["inputs"]

    stock_spot = float(inputs.spot)
    shares = int(inputs.shares)
    fee = float(inputs.fee_per_contract)

    if (
        not isfinite(stock_spot)
        or stock_spot <= 0
        or shares <= 0
    ):
        raise ValueError(
            "Invalid stock position"
        )

    if not isfinite(fee) or fee < 0:
        raise ValueError(
            "Invalid opening fee"
        )

    etf_name = str(
        (cross_result or {}).get(
            "etf_ric",
            "ETF",
        )
    )

    baseline_pnl = (
        shares
        * stock_spot
        * stock_return
    )

    records = []

    for candidate in candidates:
        has_options = bool(
            candidate.legs
        )

        is_share_hedge = bool(
            candidate.etf_shares
        )

        if has_options and all(
            leg.underlying == "stock"
            for leg in candidate.legs
        ):
            family = "Stock options"

        elif has_options:
            family = "ETF options"

        elif is_share_hedge:
            family = "ETF shares"

        else:
            family = "No Hedge"

        when = (
            candidate.expiry
            if has_options
            else target
        )

        if not isinstance(when, date):
            when = pd.Timestamp(
                when
            ).date()

        # Reuse the existing tested payoff engine.
        pnl = scenario_pnl(
            candidate,
            stock_spot=stock_spot,
            shares=shares,
            stock_return=stock_return,
            etf_spot=etf_spot,
            etf_return=etf_return,
            fee_per_leg_contract=fee,
        )

        debit = (
            opening_debit(candidate, fee)
            if has_options
            else None
        )

        if debit is not None and debit > 1e-9:
            cash_label = f"Pay {debit:,.2f}"

        elif debit is not None and debit < -1e-9:
            cash_label = (
                f"Receive {-debit:,.2f}"
            )

        elif has_options:
            cash_label = "0.00"

        elif family == "No Hedge":
            cash_label = "Not applicable"

        else:
            cash_label = (
                "Share trading/carry not priced"
            )

        records.append({
            "Strategy": str(candidate.name),
            "Type": family,
            "Position": _trade_description(
                candidate,
                etf_name,
            ),
            "Evaluation date": when.isoformat(),
            "Option premium + opening fees": cash_label,
            "Scenario P&L": float(pnl),
            "Benefit vs No Hedge": float(
                pnl - baseline_pnl
            ),
            "Has options": has_options,
            "Matches target date": (
                when == target
            ),
        })

    if (
        not records
        or sum(
            row["Strategy"] == "No Hedge"
            for row in records
        ) != 1
    ):
        raise ValueError(
            "Exactly one No Hedge reference is required"
        )

    if any(
        row["Has options"]
        and not row["Matches target date"]
        for row in records
    ):
        notices.append(
            "Some option contracts expire after the "
            "requested hedge date. Their P&L is shown "
            "at expiration, not at the requested date; "
            "do not rank it against target-date share hedges."
        )

    return (
        pd.DataFrame(records),
        list(notices),
    )


def render_unified_hedge_choices(
    hedge_result: dict,
    cross_result: dict | None = None,
) -> None:
    """Render available strategies with optional scenario controls."""

    import streamlit as st

    st.subheader(
        "Available hedging choices"
    )

    st.caption(
        "Strategies currently eligible for research. "
        "Availability does not mean that an order "
        "can be executed."
    )

    try:
        table, notices = (
            build_unified_hedge_choices(
                hedge_result,
                cross_result,
            )
        )

    except (
        ValueError,
        TypeError,
        KeyError,
        AttributeError,
    ) as exc:
        st.warning(
            f"Cannot show eligible hedge choices: {exc}"
        )
        return

    # -----------------------------------------------------
    # SIMPLE STRATEGY INVENTORY
    # -----------------------------------------------------

    columns = [
        "Strategy",
        "Type",
        "Position",
        "Evaluation date",
        "Option premium + opening fees",
    ]

    st.dataframe(
        table[columns],
        hide_index=True,
        use_container_width=True,
    )

    for notice in notices:
        st.info(notice)

    # -----------------------------------------------------
    # EXPLAIN UNAVAILABLE STRATEGY TYPES
    # -----------------------------------------------------

    meta = hedge_result["meta"]

    if meta.get("used_close_fallback"):
        st.caption(
            "Stock option strategies are excluded "
            "while the stock uses an unsynchronized "
            "closing-price reference."
        )

    elif not table["Type"].eq(
        "Stock options"
    ).any():
        st.caption(
            "No eligible direct stock-option "
            "structures in this snapshot."
        )

    if cross_result is None:
        st.caption(
            "Evaluate a comparison ETF above "
            "to add ETF hedge choices."
        )

    elif not table["Type"].eq(
        "ETF options"
    ).any():
        st.caption(
            "No eligible ETF-option structures "
            "in this snapshot. Missing quotes, "
            "contract checks or expiration dates "
            "may prevent selection."
        )

    if table["Has options"].any():
        st.caption(
            "Option premium values include opening fees "
            "only. They exclude later closing costs, "
            "early exercise, margin and financing. "
            "Contract identifiers are available "
            "in the contract tables above or the exported scenario CSV."
        )

    # -----------------------------------------------------
    # OPTIONAL INTERACTIVE PRICE SCENARIOS
    # -----------------------------------------------------

    with st.expander(
        "Explore a hypothetical stock and ETF price move"
    ):
        st.caption(
            "Change the price assumptions below. "
            "Share hedges are evaluated at the requested "
            "hedge date; options at their listed expiration. "
            "Do not rank results across different dates."
        )

        left, right = st.columns(2)

        with left:
            stock_pct = st.slider(
                "Stock price change (%)",
                min_value=-100,
                max_value=100,
                value=-20,
                step=5,
                key="unified_choices_stock_pct",
            )

        with right:
            etf_pct = st.slider(
                "ETF price change (%)",
                min_value=-100,
                max_value=100,
                value=-10,
                step=5,
                key="unified_choices_etf_pct",
            )

        try:
            stressed, _ = (
                build_unified_hedge_choices(
                    hedge_result,
                    cross_result,
                    stock_return=stock_pct / 100,
                    etf_return=etf_pct / 100,
                )
            )

        except (
            ValueError,
            TypeError,
            KeyError,
            AttributeError,
        ) as exc:
            st.warning(
                f"Cannot calculate scenario: {exc}"
            )
            return

        # Show each date separately so strategies
        # from different maturities are not ranked.
        for when, group in stressed.groupby(
            "Evaluation date",
            sort=True,
        ):
            st.write(
                f"**Evaluation date: {when}**"
            )

            st.dataframe(
                group[
                    [
                        "Strategy",
                        "Type",
                        "Scenario P&L",
                        "Benefit vs No Hedge",
                    ]
                ].style.format({
                    "Scenario P&L": "{:+,.2f}",
                    "Benefit vs No Hedge": "{:+,.2f}",
                }),
                hide_index=True,
                use_container_width=True,
            )

        st.caption(
            "These are hypothetical outcomes using "
            "the selected stock and ETF returns. "
            "They are not forecasts. ETF short carry "
            "and trading costs remain unpriced. "
            "No FX conversion is applied. If stock and ETF currencies "
            "differ, combined P&L and hedge sizing are not reliable."
        )
