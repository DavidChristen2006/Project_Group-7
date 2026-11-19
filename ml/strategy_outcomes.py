
import numpy as np
import pandas as pd


JOIN_COLUMNS = [
    "Underlying RIC",
    "Decision Date",
    "Outcome Date",
    "Hedge Horizon Days"
]


def build_historical_strategy_outcomes(
    strategy_name,
    option_legs,
    shares=1000,
    option_multiplier=100
):
    if not option_legs:
        raise ValueError("At least one option leg is required.")

    if shares <= 0 or option_multiplier <= 0:
        raise ValueError("Invalid position size.")

    contracts = shares / option_multiplier

    if not float(contracts).is_integer():
        raise ValueError(
            "Whole option contracts are required."
        )

    combined = None

    for leg_name, leg_df in option_legs.items():
        if leg_df.empty:
            raise ValueError(
                f"No observations for {leg_name}."
            )

        if leg_df[JOIN_COLUMNS].duplicated().any():
            raise ValueError(
                f"Duplicate outcomes for {leg_name}."
            )

        current = leg_df[
            JOIN_COLUMNS + [
                "Option RIC",
                "Side",
                "Option P&L Per Share",
                "Stock Entry Price",
                "Stock Exit Price"
            ]
        ].copy()

        current = current.rename(
            columns={
                "Option RIC": f"{leg_name} RIC",
                "Side": f"{leg_name} Side",
                "Option P&L Per Share": f"{leg_name} P&L",
                "Stock Entry Price": f"{leg_name} Stock Entry",
                "Stock Exit Price": f"{leg_name} Stock Exit"
            }
        )

        if combined is None:
            combined = current
        else:
            combined = combined.merge(
                current,
                on=JOIN_COLUMNS,
                how="inner",
                validate="one_to_one"
            )

    if combined.empty:
        raise ValueError(
            "No overlapping historical strategy dates."
        )

    entry_columns = [
        f"{name} Stock Entry"
        for name in option_legs
    ]

    exit_columns = [
        f"{name} Stock Exit"
        for name in option_legs
    ]

    if not np.allclose(
        combined[entry_columns].to_numpy(dtype=float),
        combined[[entry_columns[0]]].to_numpy(dtype=float)
    ):
        raise ValueError("Stock entry prices do not match.")

    if not np.allclose(
        combined[exit_columns].to_numpy(dtype=float),
        combined[[exit_columns[0]]].to_numpy(dtype=float)
    ):
        raise ValueError("Stock exit prices do not match.")

    combined["Stock Entry Price"] = combined[
        entry_columns[0]
    ]

    combined["Stock Exit Price"] = combined[
        exit_columns[0]
    ]

    pnl_columns = [
        f"{name} P&L"
        for name in option_legs
    ]

    combined["Stock P&L"] = shares * (
        combined["Stock Exit Price"]
        - combined["Stock Entry Price"]
    )

    combined["Option P&L"] = (
        combined[pnl_columns].sum(axis=1)
        * contracts
        * option_multiplier
    )

    combined["Hedged P&L"] = (
        combined["Stock P&L"]
        + combined["Option P&L"]
    )

    initial_position_value = (
        shares * combined["Stock Entry Price"]
    )

    combined["Unhedged Return"] = (
        combined["Stock P&L"] / initial_position_value
    )

    combined["Hedged Return"] = (
        combined["Hedged P&L"] / initial_position_value
    )

    combined["Hedge Improvement"] = (
        combined["Hedged Return"]
        - combined["Unhedged Return"]
    )

    combined["Strategy"] = strategy_name

    if not np.isfinite(
        combined[
            [
                "Stock P&L",
                "Option P&L",
                "Hedged P&L",
                "Hedged Return"
            ]
        ].to_numpy(dtype=float)
    ).all():
        raise ValueError("Non-finite strategy outcomes.")

    return combined.sort_values(
        ["Decision Date", "Hedge Horizon Days"]
    ).reset_index(drop=True)
