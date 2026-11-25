
"""Risk limits for ETF option cross-hedging strategies.

Computes expiration payoff bounds for the ETF option
positions only, including quoted premiums and opening fees.

Does not assume the stock and ETF move together.
"""

from __future__ import annotations

import math

import pandas as pd


def summarize_etf_option_risks(
    structures: dict,
    selected_options: pd.DataFrame,
    fee_per_contract: float = 0.0,
) -> pd.DataFrame:
    """Return option-only expiration risk by strategy and hedge fraction."""

    if (
        not math.isfinite(fee_per_contract)
        or fee_per_contract < 0
    ):
        raise ValueError("Invalid option transaction fee")

    required = {
        "Strategy",
        "Hedge Fraction",
        "Contracts Per Leg",
    }

    if required - set(selected_options.columns):
        raise ValueError(
            "Missing strategy or contract-count columns"
        )

    records = []

    for _, row in selected_options.iterrows():

        name = str(row["Strategy"])

        if name not in structures:
            raise ValueError(
                f"Unknown option structure: {name}"
            )

        structure = structures[name]
        legs = structure.legs

        contracts_value = float(
            row["Contracts Per Leg"]
        )

        if (
            not math.isfinite(contracts_value)
            or contracts_value < 0
            or not contracts_value.is_integer()
        ):
            raise ValueError(
                f"Invalid contract quantity: {name}"
            )

        contracts = int(contracts_value)
        multiplier = structure.multiplier

        premium_cashflow = (
            sum(
                -leg.side * leg.premium
                for leg in legs
            )
            * multiplier
            * contracts
        )

        fees = (
            contracts
            * len(legs)
            * fee_per_contract
        )

        # Expiration option P&L is piecewise linear.
        # For these bounded bearish structures, extrema
        # occur at zero, a strike, or above all strikes.
        strikes = sorted(
            {float(leg.strike) for leg in legs}
        )

        prices = [
            0.0,
            *strikes,
            max(strikes) * 2.0,
        ]

        payoffs = []

        for terminal in prices:

            intrinsic = 0.0

            for leg in legs:

                if leg.right == "put":
                    value = max(
                        leg.strike - terminal,
                        0.0,
                    )

                elif leg.right == "call":
                    value = max(
                        terminal - leg.strike,
                        0.0,
                    )

                else:
                    raise ValueError(
                        f"Unknown option type: {leg.right}"
                    )

                intrinsic += (
                    leg.side
                    * value
                )

            pnl = (
                intrinsic * multiplier * contracts
                + premium_cashflow
                - fees
            )

            payoffs.append(pnl)

        records.append({
            "Strategy": name,
            "Hedge Fraction": float(
                row["Hedge Fraction"]
            ),
            "Contracts Per Leg": contracts,
            "Opening Premium Cashflow": premium_cashflow,
            "Opening Fees": fees,
            "Maximum Option Profit": max(payoffs),
            "Maximum Option Loss": min(payoffs),
            "Option Capital at Risk": max(
                0.0,
                -min(payoffs),
            ),
        })

    return (
        pd.DataFrame(records)
        .sort_values(
            ["Strategy", "Hedge Fraction"]
        )
        .reset_index(drop=True)
    )
