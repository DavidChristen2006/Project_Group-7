
import argparse

import numpy as np
import pandas as pd

from ml.candidate_walk_forward_comparison import (
    load_dataset,
)

from ml.candidate_uncertainty_audit import (
    collect_decisions,
)

from ml.strategy_paths import (
    build_historical_strategy_path,
)

from ml.transaction_costs import (
    TransactionCostConfig,
    apply_transaction_costs_to_path,
)

from pipeline.historical_candidate_backtest import (
    _selection_for_candidate,
    _enrich_selection_from_cache,
)


AUDIT_DATES = [
    "2024-05-01",
    "2024-06-03",
    "2024-11-01",
]

MODELS = [
    "Random Forest",
    "XGBoost",
]

OPTION_MULTIPLIER = 100

PNL_TOLERANCE = 1e-4
RETURN_TOLERANCE = 1e-8


def get_selected_candidates(dataset, dates, models):
    records = []

    for model in models:
        # Reconstruct the same walk-forward
        # recommendations as the previous audit.
        decisions = collect_decisions(
            dataset,
            model,
            pd.Timestamp(min(dates)),
            pd.Timestamp(max(dates))
            + pd.Timedelta(days=1),
        )

        decisions = decisions.loc[
            decisions["Decision Date"].isin(
                pd.to_datetime(dates)
            )
        ].copy()

        if len(decisions) != len(dates):
            raise ValueError(
                f"Missing decisions for {model}"
            )

        for _, decision in decisions.iterrows():
            records.append({
                "Model": model,
                "Decision Date": decision[
                    "Decision Date"
                ],
                "Decision": decision["Decision"],
                "Candidate Key": decision[
                    "Candidate Key"
                ],
                "Predicted Utility (pp)": decision[
                    "Predicted Utility (pp)"
                ],
                "Actual Utility (pp)": decision[
                    "Actual Utility (pp)"
                ],
            })

    return pd.DataFrame(records)


def attribute_candidate(
    row,
    cost_config,
    quote_cache,
):
    selection = _selection_for_candidate(row)

    problems = _enrich_selection_from_cache(
        selection,
        quote_cache,
    )

    if problems:
        raise ValueError(
            "Quote validation failed: "
            + "; ".join(problems)
        )

    strategy = row["Strategy"]
    shares = int(row["Shares"])

    gross_path, gross_summary = (
        build_historical_strategy_path(
            selection=selection,
            strategy_name=strategy,
            shares=shares,
            option_multiplier=OPTION_MULTIPLIER,
        )
    )

    net_path, net_summary = (
        apply_transaction_costs_to_path(
            path_df=gross_path,
            summary=gross_summary,
            selection=selection,
            strategy_name=strategy,
            config=cost_config,
            shares=shares,
            option_multiplier=OPTION_MULTIPLIER,
            portfolio_value_column="Portfolio Value",
        )
    )

    if len(net_path) != int(
        row["Hedge Horizon Days"]
    ) + 1:
        raise ValueError(
            "Unexpected portfolio path length."
        )

    last = net_path.iloc[-1]

    stock_pnl = float(
        last["Stock P&L"]
    )

    leg_names = [
        "Long Put",
        "Short Call",
        "Short Put",
    ]

    leg_pnls = {}

    for name in leg_names:
        column = f"{name} P&L"

        leg_pnls[name] = (
            float(last[column])
            if column in net_path.columns
            else 0.0
        )

    option_pnl = sum(
        leg_pnls.values()
    )

    gross_pnl = stock_pnl + option_pnl

    transaction_costs = float(
        net_summary["Total Transaction Costs"]
    )

    reconstructed_net_pnl = (
        gross_pnl - transaction_costs
    )

    original_net_pnl = float(
        row["Final P&L"]
    )

    recomputed_net_pnl = float(
        net_summary["Final P&L"]
    )

    if not np.isclose(
        reconstructed_net_pnl,
        recomputed_net_pnl,
        atol=PNL_TOLERANCE,
        rtol=0,
    ):
        raise ValueError(
            "Component P&L does not reconcile "
            "with the reconstructed backtest."
        )

    if not np.isclose(
        reconstructed_net_pnl,
        original_net_pnl,
        atol=PNL_TOLERANCE,
        rtol=0,
    ):
        raise ValueError(
            "Reconstructed P&L does not reconcile "
            "with saved historical results."
        )

    if not np.isclose(
        net_summary["Final Return"],
        row["Final Return"],
        atol=RETURN_TOLERANCE,
        rtol=0,
    ):
        raise ValueError(
            "Final return does not reconcile."
        )

    if not np.isclose(
        net_summary["Maximum Drawdown"],
        row["Maximum Drawdown"],
        atol=RETURN_TOLERANCE,
        rtol=0,
    ):
        raise ValueError(
            "Maximum drawdown does not reconcile."
        )

    if not np.isclose(
        transaction_costs,
        row["Total Transaction Costs"],
        atol=PNL_TOLERANCE,
        rtol=0,
    ):
        raise ValueError(
            "Transaction costs do not reconcile."
        )

    initial_value = (
        shares * float(row["Spot Price"])
    )

    def as_pp(amount):
        return amount / initial_value * 100

    return {
        "Candidate Key": row["Candidate Key"],
        "Strategy": strategy,
        "Expiry": row["Expiry"],
        "Shares": shares,
        "Initial Value": initial_value,
        "Stock P&L": stock_pnl,
        "Long Put P&L": leg_pnls["Long Put"],
        "Short Call P&L": leg_pnls["Short Call"],
        "Short Put P&L": leg_pnls["Short Put"],
        "Total Option P&L": option_pnl,
        "Transaction Costs": transaction_costs,
        "Net Portfolio P&L": reconstructed_net_pnl,
        "Stock Return (pp)": as_pp(stock_pnl),
        "Long Put (pp)": as_pp(
            leg_pnls["Long Put"]
        ),
        "Short Call (pp)": as_pp(
            leg_pnls["Short Call"]
        ),
        "Short Put (pp)": as_pp(
            leg_pnls["Short Put"]
        ),
        "Option P&L (pp)": as_pp(option_pnl),
        "Transaction Costs (pp)": as_pp(
            transaction_costs
        ),
        "Net Return (pp)": as_pp(
            reconstructed_net_pnl
        ),
        "Hedge Improvement (pp)": as_pp(
            option_pnl - transaction_costs
        ),
        "Reconciliation Error": (
            reconstructed_net_pnl
            - original_net_pnl
        ),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dates",
        default=",".join(AUDIT_DATES),
    )

    args = parser.parse_args()

    dates = [
        value.strip()
        for value in args.dates.split(",")
    ]

    if not dates:
        raise ValueError(
            "At least one date is required."
        )

    dates = sorted(set(
        pd.Timestamp(date).strftime("%Y-%m-%d")
        for date in dates
    ))

    dataset = load_dataset()

    choices = get_selected_candidates(
        dataset,
        dates,
        MODELS,
    )

    historical = dataset.set_index(
        "Candidate Key",
        verify_integrity=True,
    )

    # The same contract can be selected by
    # multiple ML models. Reconstruct it once.
    quote_cache = {}
    cost_config = TransactionCostConfig()
    attribution_cache = {}
    records = []

    for _, choice in choices.iterrows():
        key = choice["Candidate Key"]

        if pd.isna(key):
            print(
                f"{choice['Model']} "
                f"{choice['Decision Date']}: No Hedge"
            )
            continue

        if key not in historical.index:
            raise KeyError(
                f"Candidate not found: {key}"
            )

        if key not in attribution_cache:
            row = historical.loc[key].copy()
            row["Candidate Key"] = key

            attribution_cache[key] = (
                attribute_candidate(
                    row,
                    cost_config,
                    quote_cache,
                )
            )

        attribution = attribution_cache[key]

        record = {
            "Decision Date": choice["Decision Date"],
            "Model": choice["Model"],
            "Predicted Utility (pp)": choice[
                "Predicted Utility (pp)"
            ],
            "Actual Utility (pp)": choice[
                "Actual Utility (pp)"
            ],
            **attribution,
        }

        records.append(record)

    if not records:
        raise ValueError(
            "No selected hedges to attribute."
        )

    results = pd.DataFrame(records)

    print("\n--- OPTION LEG P&L ATTRIBUTION ---")

    display_columns = [
        "Decision Date",
        "Model",
        "Strategy",
        "Stock Return (pp)",
        "Long Put (pp)",
        "Short Call (pp)",
        "Short Put (pp)",
        "Transaction Costs (pp)",
        "Net Return (pp)",
        "Hedge Improvement (pp)",
    ]

    print(
        results[display_columns]
        .round(4)
        .to_string(index=False)
    )

    print("\n--- P&L IN USD ---")

    money_columns = [
        "Decision Date",
        "Model",
        "Initial Value",
        "Stock P&L",
        "Long Put P&L",
        "Short Call P&L",
        "Short Put P&L",
        "Transaction Costs",
        "Net Portfolio P&L",
    ]

    print(
        results[money_columns]
        .round(2)
        .to_string(index=False)
    )

    print("\n--- RECONCILIATION ---")

    max_error = (
        results["Reconciliation Error"]
        .abs()
        .max()
    )

    print(
        "Maximum absolute P&L difference:",
        f"{max_error:.8f}"
    )

    print(
        "Unique contracts reconstructed:",
        len(attribution_cache)
    )

    print("All selected candidates reconciled.")
    print("Original historical datasets unchanged.")
    print("No LSEG requests made.")


if __name__ == "__main__":
    main()
