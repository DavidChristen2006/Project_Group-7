
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TransactionCostConfig:
    # Per contract, for each opening or closing transaction.
    option_fee_per_contract_per_side: float = 0.65

    # Additional fixed charge per option order.
    option_fixed_fee_per_order: float = 0.0

    # Additional slippage beyond the observed BID/ASK.
    # Price units per underlying share, per transaction.
    option_slippage_per_share_per_side: float = 0.0

    # Optional commissions on the underlying stock.
    stock_fee_per_share_per_side: float = 0.0

    # Set True only when stock is actually bought or sold.
    stock_traded_at_entry: bool = False
    stock_traded_at_exit: bool = False

    def __post_init__(self):
        values = [
            self.option_fee_per_contract_per_side,
            self.option_fixed_fee_per_order,
            self.option_slippage_per_share_per_side,
            self.stock_fee_per_share_per_side
        ]

        if any(not np.isfinite(v) or v < 0 for v in values):
            raise ValueError(
                "Transaction costs must be finite and nonnegative."
            )


def calculate_strategy_transaction_costs(
    selection,
    strategy_name,
    config,
    shares=1000,
    option_multiplier=100
):
    if shares <= 0 or option_multiplier <= 0:
        raise ValueError("Invalid position size.")

    if shares % option_multiplier != 0:
        raise ValueError(
            "Shares must be divisible by option multiplier."
        )

    legs = selection["Strategies"].get(strategy_name)

    if not legs:
        raise ValueError(
            f"No selected option legs for {strategy_name}"
        )

    contracts_per_leg = shares // option_multiplier
    number_of_legs = len(legs)

    # One opening and one closing trade per option leg.
    option_transactions = 2 * number_of_legs

    option_commissions = (
        option_transactions
        * contracts_per_leg
        * config.option_fee_per_contract_per_side
    )

    option_fixed_fees = (
        option_transactions
        * config.option_fixed_fee_per_order
    )

    option_slippage = (
        option_transactions
        * contracts_per_leg
        * option_multiplier
        * config.option_slippage_per_share_per_side
    )

    stock_transactions = (
        int(config.stock_traded_at_entry)
        + int(config.stock_traded_at_exit)
    )

    stock_commissions = (
        stock_transactions
        * shares
        * config.stock_fee_per_share_per_side
    )

    total_cost = (
        option_commissions
        + option_fixed_fees
        + option_slippage
        + stock_commissions
    )

    return {
        "Option Contracts Per Leg": contracts_per_leg,
        "Number of Option Legs": number_of_legs,
        "Option Commissions": float(option_commissions),
        "Option Fixed Fees": float(option_fixed_fees),
        "Option Slippage": float(option_slippage),
        "Stock Commissions": float(stock_commissions),
        "Total Transaction Costs": float(total_cost)
    }


def apply_transaction_costs_to_path(
    path_df,
    summary,
    selection,
    strategy_name,
    config,
    shares=1000,
    option_multiplier=100,
    portfolio_value_column="Portfolio Value"
):
    if path_df.empty:
        raise ValueError("Portfolio path is empty.")

    if portfolio_value_column not in path_df.columns:
        raise ValueError(
            f"Missing '{portfolio_value_column}'. "
            f"Available columns: {path_df.columns.tolist()}"
        )

    costs = calculate_strategy_transaction_costs(
        selection=selection,
        strategy_name=strategy_name,
        config=config,
        shares=shares,
        option_multiplier=option_multiplier
    )

    
    # Recover the initial portfolio value from the
    # gross backtest, using the same price basis.

    initial_value = (
        float(path_df["Portfolio Value"].iloc[-1])
        - float(summary["Final P&L"])
    )

    if not np.isfinite(initial_value) or initial_value <= 0:
        raise ValueError(
            "Invalid initial portfolio value."
        )

    # Validate consistency across the full strategy path.
    if "Portfolio P&L" in path_df.columns:
        implied_initial_values = (
            pd.to_numeric(
                path_df["Portfolio Value"],
                errors="coerce",
            )
            - pd.to_numeric(
                path_df["Portfolio P&L"],
                errors="coerce",
            )
        ).to_numpy(dtype=float)

        if not np.allclose(
            implied_initial_values,
            initial_value,
            rtol=1e-10,
            atol=1e-6,
        ):
            raise ValueError(
                "Inconsistent initial portfolio value "
                "in strategy path."
            )


    total_cost = costs["Total Transaction Costs"]

    gross_values = pd.to_numeric(
        path_df[portfolio_value_column],
        errors="coerce"
    ).to_numpy(dtype=float)

    if not np.isfinite(gross_values).all():
        raise ValueError("Invalid portfolio values.")

    # Each date represents immediate liquidation of all legs.
    # Charge the full entry + hypothetical exit cost.
    net_values = gross_values - total_cost

    running_peak = np.maximum.accumulate(
        np.concatenate(([initial_value], net_values))
    )[1:]

    drawdowns = net_values / running_peak - 1.0

    net_path = path_df.copy()

    net_path["Gross Portfolio Value"] = gross_values
    net_path["Net Portfolio Value"] = net_values
    net_path["Net Portfolio P&L"] = (
        net_values - initial_value
    )
    net_path["Net Drawdown"] = drawdowns

    net_summary = dict(summary)

    net_summary["Gross Final Return"] = summary[
        "Final Return"
    ]
    net_summary["Gross Maximum Drawdown"] = summary[
        "Maximum Drawdown"
    ]
    net_summary["Gross Final P&L"] = summary[
        "Final P&L"
    ]

    net_summary["Final P&L"] = float(
        net_values[-1] - initial_value
    )
    expected_net_pnl = (
        float(summary["Final P&L"])
        - float(total_cost)
    )

    if not np.isclose(
        net_summary["Final P&L"],
        expected_net_pnl,
        rtol=1e-10,
        atol=1e-6,
    ):
        raise ValueError(
            "Transaction-cost P&L reconciliation failed."
        )
    net_summary["Final Return"] = float(
        net_summary["Final P&L"] / initial_value
    )
    net_summary["Maximum Drawdown"] = float(
        drawdowns.min()
    )

    net_summary.update(costs)

    return net_path, net_summary
