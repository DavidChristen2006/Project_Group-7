
"""Research-only extensions to the existing hedge payoff engine.

Adds:
- Covered call
- Protective put spread
- ETF cross-hedge joint stress scenarios

No LSEG calls. No assumed relationship between future stock and ETF moves.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd

from data_pipeline.hedge_strategy_payoffs import (
    HedgeInputs,
    SHOCKS,
)


def select_lower_put(
    quoted_options: pd.DataFrame,
    *,
    long_put_strike: float,
    expiry,
    spot: float,
    max_spread_fraction: float = 0.10,
) -> Optional[pd.Series]:
    """Find a liquid lower-strike put to sell in a put spread."""

    required = {
        "RIC",
        "CallPutOption",
        "StrikePrice",
        "ExpiryDate",
        "BID",
        "ASK",
    }

    missing = required - set(quoted_options.columns)

    if missing:
        raise ValueError(
            f"Option table missing: {sorted(missing)}"
        )

    if not (math.isfinite(spot) and spot > 0):
        raise ValueError(
            "Spot must be positive and finite"
        )

    frame = quoted_options.copy()

    frame["expiry"] = (
        pd.to_datetime(
            frame["ExpiryDate"],
            errors="coerce",
            utc=True,
        )
        .dt.tz_convert(None)
        .dt.normalize()
    )

    target_expiry = pd.Timestamp(expiry).normalize()

    frame["option_type"] = (
        frame["CallPutOption"]
        .astype("string")
        .str.strip()
        .str.lower()
        .replace({"p": "put"})
    )

    for column in (
        "StrikePrice",
        "BID",
        "ASK",
    ):
        frame[column] = pd.to_numeric(
            frame[column],
            errors="coerce",
        )

    frame = frame.loc[
        frame["option_type"].eq("put")
        & frame["expiry"].eq(target_expiry)
        & frame["StrikePrice"].gt(0)
        & frame["StrikePrice"].lt(long_put_strike)
        & frame["BID"].gt(0)
        & frame["ASK"].ge(frame["BID"])
    ].copy()

    if frame.empty:
        return None

    frame["spread_fraction"] = (
        (frame["ASK"] - frame["BID"])
        / ((frame["ASK"] + frame["BID"]) / 2)
    )

    frame = frame.loc[
        frame["spread_fraction"].le(
            max_spread_fraction
        )
    ].copy()

    if frame.empty:
        return None

    # Prefer a short put near 80% of spot,
    # with a secondary preference for tight spreads.
    frame["selection_score"] = (
        (frame["StrikePrice"] / spot - 0.80).abs()
        + 0.2 * frame["spread_fraction"]
    )

    return (
        frame.sort_values(
            ["selection_score", "RIC"]
        )
        .iloc[0]
    )


def extra_option_payoffs(
    inputs: HedgeInputs,
    *,
    lower_put_strike: float | None = None,
    lower_put_bid: float | None = None,
) -> pd.DataFrame:
    """Additional expiration P&Ls on the existing stock position.

    Covered call:
        Stock + short call at BID.

    Protective put spread:
        Stock + higher-strike long put at ASK
              - lower-strike short put at BID.
    """

    if (
        (lower_put_strike is None)
        != (lower_put_bid is None)
    ):
        raise ValueError(
            "Provide both lower-put strike and bid, or neither"
        )

    if lower_put_strike is not None:
        valid = (
            math.isfinite(lower_put_strike)
            and 0 < lower_put_strike < inputs.put_strike
            and math.isfinite(lower_put_bid)
            and lower_put_bid >= 0
        )

        if not valid:
            raise ValueError(
                "Invalid lower-put strike or bid"
            )

    covered = inputs.covered_shares

    fee_per_leg = (
        inputs.contracts
        * inputs.fee_per_contract
    )

    rows = []

    for shock in SHOCKS:

        terminal = max(
            0.0,
            inputs.spot * (1 + shock),
        )

        stock_pnl = (
            inputs.shares
            * (terminal - inputs.spot)
        )

        call_payoff = (
            covered
            * max(
                terminal - inputs.call_strike,
                0.0,
            )
        )

        covered_call_pnl = (
            stock_pnl
            - call_payoff
            + covered * inputs.call_bid
            - fee_per_leg
        )

        cases = [
            (
                "covered_call",
                covered_call_pnl,
            )
        ]

        if lower_put_strike is not None:

            long_put_payoff = (
                covered
                * max(
                    inputs.put_strike - terminal,
                    0.0,
                )
            )

            short_put_payoff = (
                covered
                * max(
                    lower_put_strike - terminal,
                    0.0,
                )
            )

            put_spread_pnl = (
                stock_pnl
                + long_put_payoff
                - short_put_payoff
                - covered * inputs.put_ask
                + covered * lower_put_bid
                - 2 * fee_per_leg
            )

            cases.append(
                (
                    "protective_put_spread",
                    put_spread_pnl,
                )
            )

        for name, pnl in cases:

            rows.append({
                "Stock Price Change": shock,
                "Expiration Stock Price": terminal,
                "Strategy": name,
                "Total P&L": pnl,
                "P&L / Initial Stock Value": (
                    pnl
                    / (inputs.shares * inputs.spot)
                ),
            })

    return pd.DataFrame(rows)


def etf_cross_hedge_stresses(
    *,
    stock_spot: float,
    shares: int,
    etf_spot: float,
    beta: float,
    hedge_fractions=(
        0.25,
        0.50,
        0.75,
        1.00,
    ),
    stock_shocks=(
        -0.30,
        -0.20,
        -0.10,
        0.00,
        0.10,
        0.20,
    ),
    etf_shocks=(
        -0.20,
        -0.10,
        0.00,
        0.10,
        0.20,
    ),
) -> pd.DataFrame:
    """Joint stock/ETF scenarios with explicit basis risk.

    ETF position:
        -round(stock_notional * beta * hedge_fraction / etf_spot)

    A negative ETF position means short ETF shares.
    A positive position means long ETF shares.

    Beta must be supplied externally; it is not estimated here.
    No financing, margin, borrowing, distributions or trading costs.
    """

    if (
        not isinstance(shares, int)
        or shares <= 0
    ):
        raise ValueError(
            "Shares must be a positive integer"
        )

    if not all(
        math.isfinite(value)
        for value in (
            stock_spot,
            etf_spot,
            beta,
        )
    ):
        raise ValueError(
            "All prices and beta must be finite"
        )

    if stock_spot <= 0 or etf_spot <= 0:
        raise ValueError(
            "Prices must be positive"
        )

    if any(
        not 0 <= float(fraction) <= 1
        for fraction in hedge_fractions
    ):
        raise ValueError(
            "Hedge fractions must be between 0 and 1"
        )

    if any(
        not math.isfinite(float(shock))
        or float(shock) < -1
        for shock in (
            *stock_shocks,
            *etf_shocks,
        )
    ):
        raise ValueError(
            "Invalid stock or ETF shock"
        )

    stock_notional = shares * stock_spot

    rows = []

    for fraction in hedge_fractions:

        etf_position = -round(
            stock_notional
            * beta
            * fraction
            / etf_spot
        )

        for stock_return in stock_shocks:
            for etf_return in etf_shocks:

                stock_pnl = (
                    stock_notional
                    * stock_return
                )

                etf_pnl = (
                    etf_position
                    * etf_spot
                    * etf_return
                )

                rows.append({
                    "Hedge Fraction": float(fraction),
                    "ETF Position Shares (signed)": (
                        etf_position
                    ),
                    "Stock Shock": float(stock_return),
                    "ETF Shock": float(etf_return),
                    "Stock P&L": stock_pnl,
                    "ETF P&L": etf_pnl,
                    "Total P&L": (
                        stock_pnl + etf_pnl
                    ),
                    "Unhedged P&L": stock_pnl,
                })

    return pd.DataFrame(rows)
