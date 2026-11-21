"""
This code is an audit-script, which answers 
1. if all generated candidates were backtested, 
2. which candidates are incomplete, 
3. are reported P&L, return, and hedge-improvement values mathematically consistent 
4. and how did each strategy perform on each decision date

the code only reads files and changes nothing 
"""

from pathlib import Path

import numpy as np
import pandas as pd

# configuration
# Run_Name is a folder name of one specific run,
# stock = Apple, date range = 2015-01-02 to 2015-06-01, horizon = H21 etc.
# to audit a different run, just change this value  

RUN_NAME = (
    "AAPL_O_20150102_20150601_H21_S1000_K500_"
    "TC00a56a790ea1_EXP6500-12500_STEP5000_"
    "CFG7d3bc7d1e923"
)

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
    """Run the four audit checks on one backtest run and print the results.

    Reads the candidate inventory, the backtest results and the coverage
    table of the run defined by RUN_NAME, then:
      1. compares generated vs. completed candidates per decision date,
      2. lists all candidates whose status is not "Complete",
      3. recomputes P&L, return and hedge improvement and compares them
         with the stored values,
      4. summarizes hedge improvement and drawdown per strategy and date.

    Raises:
        AssertionError: if one of the accounting reconciliations fails.

    Returns:
        None. Results are printed; no files are modified.
    """

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

    # Check 1: where all candidates backtested?

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

    # calculates the incomplete candidates and 
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

    # prints the completion rate as a percentage value 
    print(
        monthly.to_string(
            float_format=lambda x: f"{x:.2%}"
        )
    )

    # Check 2: Which candidates are incomplete?

    print("\n--- INCOMPLETE CANDIDATES ---")

    # selects the rows in the DataFrame, where status is not "Complete" 
    # print how many are incomplete together with the decision date, strategy, candidate ID
    # status and reason 
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

    # Check 3: is the accounting consistent? 

    print("\n--- ACCOUNTING VALIDATION ---")

    # calculate expectations and compare with the stored value 
    # net p&l (P&L minus transaction fees) 
    expected_pnl = (
        results["Gross Final P&L"]
        - results["Total Transaction Costs"]
    )

    pnl_errors = (
        results["Final P&L"]
        - expected_pnl
    ).abs()

    # expected returns (P&L divided by the invested capital)
    expected_returns = (
        results["Final P&L"]
        / (
            results["Spot Price"]
            * results["Shares"]
        )
    )

    # expected improvement (return with hedging minus return without hedging)
    expected_improvement = (
        results["Final Return"]
        - results["Unhedged Final Return"]
    )

    # np.allclose accepts tiny differences
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

    # We only reach this point if all three assertions passed.
    print(
        "Maximum P&L discrepancy:",
        f"{pnl_errors.max():.10f}",
    )
    print("P&L reconciliation: PASS")
    print("Return reconciliation: PASS")
    print("Hedge improvement reconciliation: PASS")

    # Check 4: Strategy decision per decision date
    print("\n--- STRATEGY PERFORMANCE BY MONTH ---")

    # Group by (decision date, strategy) and compute three metrics per group:
    # Candidates       = number of candidates
    # Mean_Improvement = average hedge improvement
    #                    (positive = the hedge helped)
    # Mean_Drawdown    = average maximum drop from the peak
    #                    (more negative = riskier)

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
