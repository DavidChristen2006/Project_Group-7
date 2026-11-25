
"""Continuous expiration payoff charts for direct and ETF cross hedges.

Uses existing contract selections and payoff assumptions.
No LSEG calls, new Greek estimates or historical ML forecasts.
"""

from __future__ import annotations

from math import isfinite
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go


def _axis_values(
    upper_pct: float,
    exact_pct=(),
    points: int = 401,
) -> np.ndarray:
    """Create a chart domain starting at exactly -100%."""

    if (
        not isfinite(float(upper_pct))
        or upper_pct <= -100
    ):
        raise ValueError(
            "Upper chart limit must be finite and exceed -100%"
        )

    if points < 2:
        raise ValueError("At least two points required")

    bounds = np.linspace(
        -100.0,
        float(upper_pct),
        points,
    )

    # Include option strikes so the chart reproduces
    # the exact piecewise-linear payoff turning points.
    knots = [
        float(value)
        for value in exact_pct
        if -100 <= float(value) <= upper_pct
    ]

    return np.unique(
        np.concatenate(
            (
                bounds,
                np.asarray([
                    -100.0,
                    *knots,
                    float(upper_pct),
                ]),
            )
        )
    )


def direct_expiration_curves(
    inputs: Any,
    *,
    upper_pct: float,
    lower_put_strike: float | None = None,
    lower_put_bid: float | None = None,
) -> pd.DataFrame:
    """Continuous expiration P&L for direct stock hedges."""

    if (
        (lower_put_strike is None)
        != (lower_put_bid is None)
    ):
        raise ValueError(
            "Provide both lower put strike and bid"
        )

    if lower_put_strike is not None:
        if not (
            isfinite(float(lower_put_strike))
            and 0 < lower_put_strike < inputs.put_strike
            and isfinite(float(lower_put_bid))
            and lower_put_bid >= 0
        ):
            raise ValueError("Invalid lower put")

    knots = [
        0.0,
        100 * (inputs.put_strike / inputs.spot - 1),
        100 * (inputs.call_strike / inputs.spot - 1),
    ]

    if lower_put_strike is not None:
        knots.append(
            100 * (lower_put_strike / inputs.spot - 1)
        )

    x_pct = _axis_values(
        upper_pct,
        knots,
    )

    terminal = (
        inputs.spot * (1 + x_pct / 100)
    )

    if not np.isfinite(terminal).all():
        raise ValueError(
            "Chart range too large for finite calculation"
        )

    shares = inputs.shares
    covered = inputs.covered_shares

    fee_one_leg = (
        inputs.contracts * inputs.fee_per_contract
    )

    stock_pnl = (
        shares * (terminal - inputs.spot)
    )

    put_payoff = (
        covered
        * np.maximum(
            inputs.put_strike - terminal,
            0,
        )
    )

    call_liability = (
        covered
        * np.maximum(
            terminal - inputs.call_strike,
            0,
        )
    )

    result = pd.DataFrame({
        "Underlying move (%)": x_pct,

        "Unhedged": stock_pnl,

        "Protective put": (
            stock_pnl
            + put_payoff
            - covered * inputs.put_ask
            - fee_one_leg
        ),

        "Collar": (
            stock_pnl
            + put_payoff
            - call_liability
            + covered * (
                inputs.call_bid - inputs.put_ask
            )
            - 2 * fee_one_leg
        ),

        "Covered call": (
            stock_pnl
            - call_liability
            + covered * inputs.call_bid
            - fee_one_leg
        ),
    })

    if lower_put_strike is not None:

        short_put_payoff = (
            covered
            * np.maximum(
                lower_put_strike - terminal,
                0,
            )
        )

        result["Protective put spread"] = (
            stock_pnl
            + put_payoff
            - short_put_payoff
            + covered * (
                lower_put_bid - inputs.put_ask
            )
            - 2 * fee_one_leg
        )

    return result


def cross_hedge_expiration_curves(
    *,
    stock_spot: float,
    stock_shares: int,
    stock_move_pct: float,
    etf_spot: float,
    upper_etf_pct: float,
    hedge_fraction: float,
    share_stresses: pd.DataFrame,
    option_structures: dict[str, Any] | None = None,
    option_stresses: pd.DataFrame | None = None,
    fee_per_contract: float = 0.0,
) -> pd.DataFrame:
    """Vary ETF price while holding the stock shock fixed.

    Reuses existing ETF share positions and option contract
    counts. Does not size the hedge again.
    """

    parameters = (
        stock_spot,
        stock_move_pct,
        etf_spot,
        hedge_fraction,
        fee_per_contract,
    )

    if not all(
        isfinite(float(value))
        for value in parameters
    ):
        raise ValueError("Nonfinite hedge input")

    if (
        stock_spot <= 0
        or etf_spot <= 0
        or not isinstance(stock_shares, int)
        or stock_shares <= 0
        or stock_move_pct < -100
        or not 0 < hedge_fraction <= 1
        or fee_per_contract < 0
    ):
        raise ValueError("Invalid hedge exposure")

    matching_shares = share_stresses.loc[
        np.isclose(
            share_stresses["Hedge Fraction"],
            hedge_fraction,
        )
    ]

    if matching_shares.empty:
        raise ValueError(
            "Requested ETF share hedge fraction unavailable"
        )

    positions = pd.to_numeric(
        matching_shares[
            "ETF Position Shares (signed)"
        ],
        errors="raise",
    ).unique()

    if (
        len(positions) != 1
        or not np.isfinite(positions[0])
        or float(positions[0]) != int(positions[0])
    ):
        raise ValueError(
            "Ambiguous ETF share hedge position"
        )

    etf_position = int(positions[0])

    # Include each ETF option strike in the chart domain.
    knots = [0.0]

    if option_structures:
        for structure in option_structures.values():
            for leg in structure.legs:
                knots.append(
                    100 * (leg.strike / etf_spot - 1)
                )

    x_pct = _axis_values(
        upper_etf_pct,
        knots,
    )

    etf_terminal = (
        etf_spot * (1 + x_pct / 100)
    )

    if not np.isfinite(etf_terminal).all():
        raise ValueError(
            "ETF chart range too large"
        )

    stock_pnl = (
        stock_shares
        * stock_spot
        * stock_move_pct / 100
    )

    result = pd.DataFrame({
        "ETF move (%)": x_pct,

        "Unhedged stock": np.full_like(
            x_pct,
            stock_pnl,
        ),

        "ETF share cross hedge": (
            stock_pnl
            + etf_position
            * etf_spot
            * (x_pct / 100)
        ),
    })

    if option_structures:

        if (
            option_stresses is None
            or option_stresses.empty
        ):
            raise ValueError(
                "Missing previously calculated option positions"
            )

        for name, structure in option_structures.items():

            rows = option_stresses.loc[
                option_stresses["Strategy"].eq(name)
                & np.isclose(
                    option_stresses["Hedge Fraction"],
                    hedge_fraction,
                )
            ]

            if rows.empty:
                raise ValueError(
                    f"Missing contract count: {name}"
                )

            counts = pd.to_numeric(
                rows["Contracts Per Leg"],
                errors="raise",
            ).unique()

            if (
                len(counts) != 1
                or not np.isfinite(counts[0])
                or int(counts[0]) != counts[0]
                or counts[0] < 0
            ):
                raise ValueError(
                    f"Ambiguous contract count: {name}"
                )

            contracts = int(counts[0])

            premium_cashflow = (
                sum(
                    -leg.side * leg.premium
                    for leg in structure.legs
                )
                * structure.multiplier
                * contracts
            )

            intrinsic = np.zeros(
                len(x_pct)
            )

            for leg in structure.legs:

                if leg.right == "put":
                    payoff = np.maximum(
                        leg.strike - etf_terminal,
                        0,
                    )

                elif leg.right == "call":
                    payoff = np.maximum(
                        etf_terminal - leg.strike,
                        0,
                    )

                else:
                    raise ValueError(
                        f"Unknown option type: {leg.right}"
                    )

                intrinsic += leg.side * payoff

            fees = (
                contracts
                * len(structure.legs)
                * fee_per_contract
            )

            option_pnl = (
                intrinsic
                * structure.multiplier
                * contracts
                + premium_cashflow
                - fees
            )

            result[
                name.replace("_", " ").title()
            ] = stock_pnl + option_pnl

    if not np.isfinite(
        result.select_dtypes(
            include="number"
        ).to_numpy()
    ).all():
        raise ValueError(
            "Nonfinite chart payoff"
        )

    return result


def payoff_figure(
    data: pd.DataFrame,
    x_column: str,
    upper_pct: float,
) -> go.Figure:
    """Interactive chart whose x-axis cannot go below -100%."""

    if x_column not in data or data.empty:
        raise ValueError("Missing chart data")

    fig = go.Figure()

    for name in data.columns:

        if name == x_column:
            continue

        fig.add_trace(
            go.Scatter(
                x=data[x_column],
                y=data[name],
                name=name,
                mode="lines",
                hovertemplate=(
                    "%{x:,.1f}% | %{y:,.2f}"
                    "<extra>%{fullData.name}</extra>"
                ),
            )
        )

    fig.update_xaxes(
        title=x_column,
        range=[-100, float(upper_pct)],
        minallowed=-100,
        zeroline=True,
    )

    fig.update_yaxes(
        title="Expiration P&L (quote currency)",
        zeroline=True,
    )

    fig.update_layout(
        hovermode="x unified",
        height=510,
        margin=dict(
            l=15,
            r=15,
            t=15,
            b=20,
        ),
        legend=dict(
            orientation="h",
            y=-0.24,
        ),
    )

    return fig
