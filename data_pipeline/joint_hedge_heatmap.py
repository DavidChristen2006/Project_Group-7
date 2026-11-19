
"""Joint stock/ETF expiration payoff heatmap.

Uses previously calculated ETF share positions and
ETF option contract counts. No new LSEG calls.
"""

from __future__ import annotations

from math import isfinite

import numpy as np
import pandas as pd
import plotly.graph_objects as go


def _position_for_fraction(
    frame: pd.DataFrame,
    fraction: float,
    field: str,
    strategy: str | None = None,
) -> int:

    if frame is None or frame.empty:
        raise ValueError(
            f"Missing position data for {field}"
        )

    required = {"Hedge Fraction", field}

    if strategy is not None:
        required.add("Strategy")

    missing = required - set(frame.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    matches = frame.loc[
        np.isclose(
            pd.to_numeric(
                frame["Hedge Fraction"],
                errors="raise",
            ),
            fraction,
        )
    ]

    if strategy is not None:
        matches = matches.loc[
            matches["Strategy"].eq(strategy)
        ]

    if matches.empty:
        raise ValueError(
            f"No saved position for "
            f"{strategy or 'ETF shares'} "
            f"at {fraction:.0%}"
        )

    quantities = pd.to_numeric(
        matches[field],
        errors="raise",
    ).unique()

    if (
        len(quantities) != 1
        or not isfinite(float(quantities[0]))
        or float(quantities[0]) != int(quantities[0])
    ):
        raise ValueError(
            f"Inconsistent position quantity: "
            f"{strategy or 'ETF shares'}"
        )

    return int(quantities[0])


def build_joint_payoff_surface(
    *,
    stock_spot: float,
    stock_shares: int,
    etf_spot: float,
    hedge_fraction: float,
    share_stresses: pd.DataFrame,
    option_structures: dict | None = None,
    option_stresses: pd.DataFrame | None = None,
    fee_per_contract: float = 0.0,
    stock_max_pct: float = 100.0,
    etf_max_pct: float = 100.0,
    grid_points: int = 101,
) -> dict:
    """Calculate joint stock/ETF expiration P&L."""

    values = (
        stock_spot,
        etf_spot,
        hedge_fraction,
        fee_per_contract,
        stock_max_pct,
        etf_max_pct,
    )

    if not all(
        isfinite(float(value))
        for value in values
    ):
        raise ValueError(
            "All numerical inputs must be finite"
        )

    if (
        stock_spot <= 0
        or etf_spot <= 0
        or type(stock_shares) is not int
        or stock_shares <= 0
        or not 0 < hedge_fraction <= 1
        or fee_per_contract < 0
        or stock_max_pct < 0
        or etf_max_pct < 0
        or grid_points < 11
        or grid_points > 501
    ):
        raise ValueError(
            "Invalid position or chart limits"
        )

    stock_moves = np.linspace(
        -100.0,
        float(stock_max_pct),
        grid_points,
    )

    etf_moves = np.linspace(
        -100.0,
        float(etf_max_pct),
        grid_points,
    )

    stock_pnl = (
        stock_spot
        * stock_shares
        * stock_moves[:, None]
        / 100.0
    )

    etf_terminal = (
        etf_spot
        * (1.0 + etf_moves / 100.0)
    )

    unhedged = np.broadcast_to(
        stock_pnl,
        (grid_points, grid_points),
    ).copy()

    etf_shares = _position_for_fraction(
        share_stresses,
        hedge_fraction,
        "ETF Position Shares (signed)",
    )

    share_hedge = (
        unhedged
        + etf_shares
        * etf_spot
        * etf_moves[None, :]
        / 100.0
    )

    payoffs = {
        "Unhedged stock": unhedged,
        "Short/long ETF shares": share_hedge,
    }

    quantities = {
        "Short/long ETF shares": etf_shares,
    }

    for name, structure in (
        option_structures or {}
    ).items():

        contracts = _position_for_fraction(
            option_stresses,
            hedge_fraction,
            "Contracts Per Leg",
            name,
        )

        if not structure.legs:
            raise ValueError(
                f"Empty option structure: {name}"
            )

        if len({
            leg.expiry
            for leg in structure.legs
        }) != 1:
            raise ValueError(
                f"Mismatched expiries in {name}"
            )

        option_intrinsic = np.zeros_like(
            etf_terminal
        )

        premium_cashflow = 0.0

        for leg in structure.legs:

            if leg.right == "put":
                intrinsic = np.maximum(
                    leg.strike - etf_terminal,
                    0.0,
                )

            elif leg.right == "call":
                intrinsic = np.maximum(
                    etf_terminal - leg.strike,
                    0.0,
                )

            else:
                raise ValueError(
                    f"Invalid option type: {leg.right}"
                )

            option_intrinsic += (
                leg.side * intrinsic
            )

            premium_cashflow += (
                -leg.side * leg.premium
            )

        option_pnl = (
            contracts
            * structure.multiplier
            * (
                option_intrinsic
                + premium_cashflow
            )
            - contracts
            * len(structure.legs)
            * fee_per_contract
        )

        label = name.replace(
            "_", " "
        ).title()

        payoffs[label] = (
            unhedged + option_pnl[None, :]
        )

        quantities[label] = contracts

    if not all(
        np.isfinite(values).all()
        for values in payoffs.values()
    ):
        raise ValueError(
            "Nonfinite P&L in joint surface"
        )

    return {
        "stock_moves": stock_moves,
        "etf_moves": etf_moves,
        "payoffs": payoffs,
        "positions": quantities,
    }


def make_joint_heatmap(
    surface: dict,
    strategy: str,
    *,
    mode: str = "Total P&L",
    stock_cursor: float = -20.0,
    etf_cursor: float = -10.0,
) -> go.Figure:
    """Display total P&L or benefit versus no hedge."""

    if strategy not in surface["payoffs"]:
        raise ValueError(
            f"Unknown strategy: {strategy}"
        )

    if mode not in {
        "Total P&L",
        "Benefit versus unhedged",
    }:
        raise ValueError(
            "Unknown display mode"
        )

    z = surface["payoffs"][
        strategy
    ].copy()

    if mode == "Benefit versus unhedged":
        z -= surface["payoffs"][
            "Unhedged stock"
        ]

    fig = go.Figure(
        go.Heatmap(
            x=surface["etf_moves"],
            y=surface["stock_moves"],
            z=z,
            colorscale="RdBu",
            zmid=0,
            colorbar={
                "title": "P&L",
            },
            hovertemplate=(
                "ETF: %{x:.1f}%"
                "<br>Stock: %{y:.1f}%"
                "<br>Value: %{z:,.2f}"
                "<extra></extra>"
            ),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[etf_cursor],
            y=[stock_cursor],
            mode="markers",
            marker={
                "symbol": "x",
                "size": 13,
                "color": "black",
                "line": {
                    "width": 2,
                },
            },
            name="Selected scenario",
            hoverinfo="skip",
        )
    )

    fig.update_layout(
        height=570,
        margin={
            "l": 10,
            "r": 10,
            "t": 25,
            "b": 10,
        },
    )

    fig.update_xaxes(
        title="ETF return (%)",
        range=[
            -100,
            float(
                surface["etf_moves"][-1]
            ),
        ],
        minallowed=-100,
    )

    fig.update_yaxes(
        title="Stock return (%)",
        range=[
            -100,
            float(
                surface["stock_moves"][-1]
            ),
        ],
        minallowed=-100,
    )

    return fig


def selected_joint_scenario(
    surface: dict,
    stock_move_pct: float,
    etf_move_pct: float,
) -> pd.DataFrame:
    """Read P&L at the nearest heatmap grid point."""

    stock_index = int(
        np.argmin(
            abs(
                surface["stock_moves"]
                - stock_move_pct
            )
        )
    )

    etf_index = int(
        np.argmin(
            abs(
                surface["etf_moves"]
                - etf_move_pct
            )
        )
    )

    baseline = float(
        surface["payoffs"]["Unhedged stock"][
            stock_index,
            etf_index,
        ]
    )

    records = []

    for name, values in surface[
        "payoffs"
    ].items():

        pnl = float(
            values[
                stock_index,
                etf_index,
            ]
        )

        records.append({
            "Strategy": name,
            "Total P&L": pnl,
            "Benefit vs unhedged": (
                pnl - baseline
            ),
        })

    return pd.DataFrame(records)
