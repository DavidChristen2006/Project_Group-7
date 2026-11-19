
import numpy as np
import pandas as pd


def build_stock_option_outcomes(
    option_leg_df,
    shares=1000,
    option_multiplier=100
):
    if shares <= 0 or option_multiplier <= 0:
        raise ValueError(
            "Shares and option multiplier must be positive."
        )

    contracts = shares / option_multiplier

    if not float(contracts).is_integer():
        raise ValueError(
            "Position must contain a whole number "
            "of option contracts."
        )

    result = option_leg_df.copy()

    if result.empty:
        raise ValueError("Option-leg dataset is empty.")

    result["Stock P&L"] = shares * (
        result["Stock Exit Price"]
        - result["Stock Entry Price"]
    )

    result["Option P&L"] = (
        result["Option P&L Per Share"]
        * contracts
        * option_multiplier
    )

    result["Hedged P&L"] = (
        result["Stock P&L"]
        + result["Option P&L"]
    )

    initial_value = (
        shares * result["Stock Entry Price"]
    )

    result["Unhedged Return"] = (
        result["Stock P&L"] / initial_value
    )

    result["Hedged Return"] = (
        result["Hedged P&L"] / initial_value
    )

    result["Hedge Improvement"] = (
        result["Hedged Return"]
        - result["Unhedged Return"]
    )

    if not np.isfinite(
        result[
            [
                "Stock P&L",
                "Option P&L",
                "Hedged P&L",
                "Hedged Return"
            ]
        ].to_numpy(dtype=float)
    ).all():
        raise ValueError("Invalid portfolio outcomes.")

    return result
