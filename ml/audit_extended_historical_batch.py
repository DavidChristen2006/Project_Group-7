"""
this code is an audit-script, which answers 
1. if all generated candidates were backtested, 
2. which candidates are incomplete and why, 
3. are reported P&L, return, and hedge-improvement values mathematically consistant 
4. and how did each strategy perform on each decision date

the code only reads files and changes nothing 
"""

from pathlib import Path

import numpy as np
import pandas as pd

# configuration
# Run_Name is a folder name of one specific run,
# stock = Apple, date range = 2015-01-02 to 2015-06-01, horizon = H21 etc.
# to audit a different run, just change this valaue  

RUN_NAME = (
    "AAPL_O_20150102_20150601_H21_S1000_K500_"
    "TC00a56a790ea1_EXP6500-12500_STEP5000_"
    "CFG7d3bc7d1e923"
)

"""locate 3 files in the data folder"""
ROOT = Path("data/processed")

# all candidates (hedging strategies) that were originally generated
INVENTORY_PATH = (
    ROOT / "candidate_inventories" / RUN_NAME
    / "historical_candidates.parquet"
)

# completed backtest results
RESULT_PATH = (
    ROOT / "candidate_backtests" / RUN_NAME
    / "candidate_ml_results.parquet"
)

# status information for each candidate 
COVERAGE_PATH = (
    ROOT / "candidate_backtests" / RUN_NAME
    / "candidate_coverage.parquet"
)


def main():
    """Load three parquet files and perform four tests about the backtest"""

# load three parquet files 
    inventory = pd.read_parquet(INVENTORY_PATH)
    results = pd.read_parquet(RESULT_PATH)
    coverage = pd.read_parquet(COVERAGE_PATH)

# standardize the date column in all three tables 
# converts the date values to pandas timestamps and removes time components with normalize()
    for frame in (inventory, results, coverage):
        frame["Decision Date"] = pd.to_datetime(
            frame["Decision Date"]
        ).dt.normalize()

# check 1: where all candidates backtested?

    print("\n--- MONTHLY COVERAGE ---")

# number of candidates per decision date: generated vs. completed 
    generated = inventory.groupby(
        "Decision Date"
    ).size().rename("Generated")

    completed = results.groupby(
        "Decision Date"
    ).size().rename("Completed")
    
# creates the DataFrame 'monthly' and put both counts side by side 
# if no candidate is completed on a date fillna(0) handles it
    monthly = pd.concat(
        [generated, completed],
        axis=1,
    ).fillna(0).astype(int)

# calculates the imcomplete candidates and 
# adds the column to the DataFrame
    monthly["Incomplete"] = (
        monthly["Generated"]
        - monthly["Completed"]
    )

# calculates the completion rate (share of completed ones) and 
# adds the column to the DataFrame
    monthly["Completion Rate"] = (
        monthly["Completed"]
        / monthly["Generated"]
    )

    print(
        monthly.to_string(
            float_format=lambda x: f"{x:.2%}"
        )
    )

    print("\n--- INCOMPLETE CANDIDATES ---")

    incomplete = coverage.loc[
        coverage["Status"] != "Complete"
    ].copy()

    print("Total incomplete:", len(incomplete))

    if not incomplete.empty:
        columns = [
            "Decision Date",
            "Strategy",
            "Candidate ID",
            "Status",
            "Reason",
        ]

        available = [
            col for col in columns
            if col in incomplete.columns
        ]

        print(
            incomplete[available].to_string(
                index=False
            )
        )

    print("\n--- ACCOUNTING VALIDATION ---")

    expected_pnl = (
        results["Gross Final P&L"]
        - results["Total Transaction Costs"]
    )

    pnl_errors = (
        results["Final P&L"]
        - expected_pnl
    ).abs()

    expected_returns = (
        results["Final P&L"]
        / (
            results["Spot Price"]
            * results["Shares"]
        )
    )

    expected_improvement = (
        results["Final Return"]
        - results["Unhedged Final Return"]
    )

    assert np.allclose(
        results["Final P&L"],
        expected_pnl,
        rtol=1e-10,
        atol=1e-6,
    ), "P&L reconciliation failed."

    assert np.allclose(
        results["Final Return"],
        expected_returns,
        rtol=1e-10,
        atol=1e-8,
    ), "Return reconciliation failed."

    assert np.allclose(
        results["Hedge Improvement"],
        expected_improvement,
        rtol=1e-10,
        atol=1e-8,
    ), "Hedge improvement reconciliation failed."

    print(
        "Maximum P&L discrepancy:",
        f"{pnl_errors.max():.10f}",
    )
    print("P&L reconciliation: PASS")
    print("Return reconciliation: PASS")
    print("Hedge improvement reconciliation: PASS")

    print("\n--- STRATEGY PERFORMANCE BY MONTH ---")

    strategy_monthly = (
        results.groupby(
            ["Decision Date", "Strategy"]
        )
        .agg(
            Candidates=("Candidate ID", "size"),
            Mean_Improvement=(
                "Hedge Improvement",
                "mean",
            ),
            Mean_Drawdown=(
                "Maximum Drawdown",
                "mean",
            ),
        )
    )

    print(
        strategy_monthly.to_string(
            float_format=lambda x: f"{x:.4f}"
        )
    )

    print("\nAudit complete.")
    print("No files modified.")


if __name__ == "__main__":
    main()
