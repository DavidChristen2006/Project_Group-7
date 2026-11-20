
from pathlib import Path

import numpy as np
import pandas as pd

""""define the variable run_name with"""
RUN_NAME = (
    "AAPL_O_20150102_20150601_H21_S1000_K500_"
    "TC00a56a790ea1_EXP6500-12500_STEP5000_"
    "CFG7d3bc7d1e923"
)

ROOT = Path("data/processed")

INVENTORY_PATH = (
    ROOT / "candidate_inventories" / RUN_NAME
    / "historical_candidates.parquet"
)

RESULT_PATH = (
    ROOT / "candidate_backtests" / RUN_NAME
    / "candidate_ml_results.parquet"
)

COVERAGE_PATH = (
    ROOT / "candidate_backtests" / RUN_NAME
    / "candidate_coverage.parquet"
)


def main():
    inventory = pd.read_parquet(INVENTORY_PATH)
    results = pd.read_parquet(RESULT_PATH)
    coverage = pd.read_parquet(COVERAGE_PATH)

    for frame in (inventory, results, coverage):
        frame["Decision Date"] = pd.to_datetime(
            frame["Decision Date"]
        ).dt.normalize()

    print("\n--- MONTHLY COVERAGE ---")

    generated = inventory.groupby(
        "Decision Date"
    ).size().rename("Generated")

    completed = results.groupby(
        "Decision Date"
    ).size().rename("Completed")

    monthly = pd.concat(
        [generated, completed],
        axis=1,
    ).fillna(0).astype(int)

    monthly["Incomplete"] = (
        monthly["Generated"]
        - monthly["Completed"]
    )

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
