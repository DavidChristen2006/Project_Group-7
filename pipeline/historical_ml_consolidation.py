
import argparse
import json
import re

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed"

FEATURE_COLUMNS = [
    "Strategy",
    "Hedge Horizon Days",
    "Spot Price",
    "Shares",
    "Contracts",
    "Option Legs",
    "Long Put Strike Pct",
    "Short Call Strike Pct",
    "Short Put Strike Pct",
    "Long Put Ask",
    "Short Call Bid",
    "Short Put Bid",
    "Net Premium Per Share",
    "Net Premium Pct",
    "Net Premium Total",
    "Days to Expiry",
    "Days After Exit",
]

TARGET_COLUMNS = [
    "Final Return",
    "Maximum Drawdown",
    "Final P&L",
    "Unhedged Final Return",
    "Unhedged Maximum Drawdown",
    "Hedge Improvement",
    "Drawdown Improvement",
    "Observations",
    "Gross Final Return",
    "Gross Maximum Drawdown",
    "Gross Final P&L",
    "Total Transaction Costs",
    "Option Commissions",
    "Option Fixed Fees",
    "Option Slippage",
    "Stock Commissions",
]

IDENTIFIER_COLUMNS = [
    "Underlying RIC",
    "Decision Date",
    "Outcome Date",
    "Hedge Horizon Days",
    "Candidate ID",
    "Strategy",
    "Expiry",
]

OPTION_IDENTIFIER_COLUMNS = [
    "Long Put RIC",
    "Short Call RIC",
    "Short Put RIC",
]

DATE_COLUMNS = [
    "Decision Date",
    "Outcome Date",
    "Expiry",
]

PRIMARY_TARGETS = [
    "Hedge Improvement",
    "Maximum Drawdown",
    "Drawdown Improvement",
]


def require_columns(frame, columns, description):
    missing = [
        column for column in columns
        if column not in frame.columns
    ]

    if missing:
        raise ValueError(
            f"{description} is missing columns: {missing}"
        )



def normalize_dates(frame):
    """
    Normalize all date columns to midnight
    using a consistent nanosecond precision.
    """
    frame = frame.copy()

    for column in DATE_COLUMNS:
        if column in frame.columns:
            frame[column] = (
                pd.to_datetime(
                    frame[column],
                    errors="raise",
                )
                .astype("datetime64[ns]")
                .dt.normalize()
            )

    return frame



def candidate_ids(frame):
    return frame["Candidate ID"].astype("string")


def read_run(run_name):
    inventory_path = (
        PROCESSED_ROOT
        / "candidate_inventories"
        / run_name
        / "historical_candidates.parquet"
    )

    backtest_dir = (
        PROCESSED_ROOT
        / "candidate_backtests"
        / run_name
    )

    coverage_path = (
        backtest_dir / "candidate_coverage.parquet"
    )

    results_path = (
        backtest_dir / "candidate_ml_results.parquet"
    )

    for path in [
        inventory_path,
        coverage_path,
        results_path,
    ]:
        if not path.is_file():
            raise FileNotFoundError(path)

    inventory = normalize_dates(
        pd.read_parquet(inventory_path)
    )

    coverage = pd.read_parquet(coverage_path)
    results = normalize_dates(
        pd.read_parquet(results_path)
    )

    require_columns(
        inventory,
        IDENTIFIER_COLUMNS
        + FEATURE_COLUMNS
        + OPTION_IDENTIFIER_COLUMNS,
        f"{run_name}: inventory",
    )

    require_columns(
        coverage,
        ["Candidate ID", "Status"],
        f"{run_name}: coverage",
    )

    require_columns(
        results,
        IDENTIFIER_COLUMNS + TARGET_COLUMNS,
        f"{run_name}: results",
    )

    if inventory.empty:
        raise ValueError(
            f"{run_name}: empty candidate inventory"
        )

    # Normalize IDs without converting large unsigned
    # 64-bit hashes to floating-point values.
    for frame in [inventory, coverage, results]:
        frame["Candidate ID"] = candidate_ids(frame)

        if frame["Candidate ID"].isna().any():
            raise ValueError(
                f"{run_name}: missing candidate IDs"
            )

        if not frame["Candidate ID"].is_unique:
            raise ValueError(
                f"{run_name}: duplicate candidate IDs"
            )

    inventory_ids = set(inventory["Candidate ID"])
    coverage_ids = set(coverage["Candidate ID"])
    result_ids = set(results["Candidate ID"])

    if inventory_ids != coverage_ids:
        raise ValueError(
            f"{run_name}: inventory/coverage IDs differ"
        )

    valid_statuses = {"Complete", "Incomplete"}

    if not set(coverage["Status"]).issubset(
        valid_statuses
    ):
        raise ValueError(
            f"{run_name}: unexpected coverage status"
        )

    complete_ids = set(
        coverage.loc[
            coverage["Status"].eq("Complete"),
            "Candidate ID",
        ]
    )

    if result_ids != complete_ids:
        raise ValueError(
            f"{run_name}: completed IDs and results differ"
        )

    # Confirm that all results have the same core
    # contract metadata as the candidate inventory.
    common_metadata = [
        "Underlying RIC",
        "Decision Date",
        "Outcome Date",
        "Hedge Horizon Days",
        "Strategy",
        "Expiry",
    ]

    reference = inventory[
        ["Candidate ID"] + common_metadata
    ].merge(
        results[
            ["Candidate ID"] + common_metadata
        ],
        on="Candidate ID",
        how="inner",
        validate="one_to_one",
        suffixes=("_inventory", "_result"),
    )


    for column in common_metadata:
        inventory_values = reference[
            f"{column}_inventory"
        ].reset_index(drop=True)

        result_values = reference[
            f"{column}_result"
        ].reset_index(drop=True)

        # Compare values without requiring
        # the Series names to match.
        inventory_values.name = None
        result_values.name = None

        if not inventory_values.equals(result_values):
            raise ValueError(
                f"{run_name}: inconsistent {column}"
            )

    # Preserve all entry-time candidate details,
    # but add ONLY the explicitly approved
    # realized outcome columns.
    dataset = inventory.merge(
        coverage[["Candidate ID", "Status"]],
        on="Candidate ID",
        how="left",
        validate="one_to_one",
    )

    dataset = dataset.rename(
        columns={"Status": "Coverage Status"}
    )

    dataset = dataset.merge(
        results[["Candidate ID"] + TARGET_COLUMNS],
        on="Candidate ID",
        how="left",
        validate="one_to_one",
    )

    dataset["Run Name"] = run_name

    # Candidate IDs are unique within each run;
    # use a composite key across the whole dataset.
    dataset["Candidate Key"] = (
        dataset["Run Name"]
        + "::"
        + dataset["Candidate ID"]
    )

    invalid_timeline = (
        dataset["Decision Date"].isna()
        | dataset["Outcome Date"].isna()
        | dataset["Expiry"].isna()
        | (
            dataset["Outcome Date"]
            <= dataset["Decision Date"]
        )
        | (
            dataset["Expiry"]
            <= dataset["Outcome Date"]
        )
    )

    if invalid_timeline.any():
        raise ValueError(
            f"{run_name}: invalid decision/outcome/"
            "expiry chronology"
        )

    actual_days_to_expiry = (
        dataset["Expiry"]
        - dataset["Decision Date"]
    ).dt.days

    actual_days_after_exit = (
        dataset["Expiry"]
        - dataset["Outcome Date"]
    ).dt.days

    if not (
        actual_days_to_expiry
        == dataset["Days to Expiry"]
    ).all():
        raise ValueError(
            f"{run_name}: Days to Expiry mismatch"
        )

    if not (
        actual_days_after_exit
        == dataset["Days After Exit"]
    ).all():
        raise ValueError(
            f"{run_name}: Days After Exit mismatch"
        )

    completed = dataset[
        "Coverage Status"
    ].eq("Complete")

    for column in PRIMARY_TARGETS:
        values = pd.to_numeric(
            dataset.loc[completed, column],
            errors="coerce",
        )

        if values.isna().any():
            raise ValueError(
                f"{run_name}: invalid completed target "
                f"{column}"
            )

    return dataset


def find_runs(ric, start_date, end_date, horizon):
    root = PROCESSED_ROOT / "candidate_inventories"

    if not root.is_dir():
        raise FileNotFoundError(root)

    ric_token = re.sub(
        r"[^A-Za-z0-9]+",
        "_",
        ric,
    ).strip("_").upper()

    pattern = re.compile(
        rf"^{re.escape(ric_token)}_"
        rf"(\d{{8}})_\1_H{horizon}_"
        rf".*_EXP6500-12500_STEP5000_CFG[0-9a-f]{{12}}$"
    )

    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()

    selected = []

    for directory in root.iterdir():
        if not directory.is_dir():
            continue

        match = pattern.fullmatch(directory.name)

        if match is None:
            continue

        decision_date = pd.Timestamp(
            match.group(1)
        )

        if start <= decision_date <= end:
            selected.append(
                (decision_date, directory.name)
            )

    selected.sort()

    dates = [
        decision_date
        for decision_date, _ in selected
    ]

    if len(dates) != len(set(dates)):
        raise ValueError(
            "Multiple experiment configurations found "
            "for a decision date. Select one "
            "configuration before consolidating."
        )

    return [name for _, name in selected]


def consolidate(ric, start_date, end_date, horizon):
    runs = find_runs(
        ric=ric,
        start_date=start_date,
        end_date=end_date,
        horizon=horizon,
    )

    if not runs:
        raise ValueError(
            "No matching historical candidate runs."
        )

    print("\n--- CONSOLIDATING HISTORICAL RUNS ---")

    frames = []

    for run_name in runs:
        frame = read_run(run_name)
        frames.append(frame)

        completed = frame[
            "Coverage Status"
        ].eq("Complete").sum()

        print(
            f"{run_name.split('_')[2]}: "
            f"{len(frame)} candidates, "
            f"{completed} complete"
        )

    dataset = pd.concat(
        frames,
        ignore_index=True,
    )

    dataset = dataset.sort_values(
        ["Decision Date", "Candidate Key"]
    ).reset_index(drop=True)

    if not dataset["Candidate Key"].is_unique:
        raise ValueError(
            "Duplicate composite candidate keys."
        )

    completed = dataset.loc[
        dataset["Coverage Status"].eq("Complete")
    ].copy()

    print("\n--- CONSOLIDATED DATASET ---")
    print("Decision dates:", dataset["Decision Date"].nunique())
    print("Total candidates:", len(dataset))
    print("Complete candidates:", len(completed))
    print("Incomplete candidates:", len(dataset) - len(completed))

    print("\nCandidate counts by decision date:")
    print(
        dataset.groupby("Decision Date")
        ["Coverage Status"]
        .value_counts()
        .unstack(fill_value=0)
        .to_string()
    )

    return dataset, completed, runs


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ric", default="AAPL.O"
    )
    parser.add_argument(
        "--start", default="2022-01-01"
    )
    parser.add_argument(
        "--end", default="2022-12-31"
    )
    parser.add_argument(
        "--horizon", type=int, default=21
    )
    parser.add_argument(
        "--execute", action="store_true"
    )

    args = parser.parse_args()

    if args.horizon <= 0:
        raise ValueError("Horizon must be positive.")

    if pd.Timestamp(args.end) < pd.Timestamp(args.start):
        raise ValueError("End date precedes start date.")

    dataset, completed, runs = consolidate(
        ric=args.ric,
        start_date=args.start,
        end_date=args.end,
        horizon=args.horizon,
    )

    if not args.execute:
        print("\nDry run complete. No files created.")
        return

    ric_token = re.sub(
        r"[^A-Za-z0-9]+",
        "_",
        args.ric,
    ).strip("_").upper()

    output_dir = (
        PROCESSED_ROOT
        / "ml_datasets"
        / (
            f"{ric_token}_"
            f"{pd.Timestamp(args.start):%Y%m%d}_"
            f"{pd.Timestamp(args.end):%Y%m%d}_"
            f"H{args.horizon}"
        )
    )

    # Never overwrite a previous consolidated
    # dataset without explicit manual review.
    output_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    dataset.to_parquet(
        output_dir / "all_candidates.parquet",
        index=False,
    )

    completed.to_parquet(
        output_dir / "completed_candidates.parquet",
        index=False,
    )

    schema = {
        "underlying": args.ric,
        "start_date": args.start,
        "end_date": args.end,
        "hedge_horizon": args.horizon,
        "run_names": runs,
        "total_candidates": len(dataset),
        "completed_candidates": len(completed),
        "incomplete_candidates": (
            len(dataset) - len(completed)
        ),
        "decision_time_feature_columns": FEATURE_COLUMNS,
        "realized_outcome_columns": TARGET_COLUMNS,
        "primary_target_columns": PRIMARY_TARGETS,
        "identifier_columns": (
            IDENTIFIER_COLUMNS
            + OPTION_IDENTIFIER_COLUMNS
            + ["Run Name", "Candidate Key"]
        ),
        "coverage_column": "Coverage Status",
        "note": (
            "Only decision_time_feature_columns are "
            "eligible as predictors. Dates and RICs "
            "are audit metadata. Outcomes are future "
            "information. Split chronologically "
            "before fitting preprocessing or models."
        ),
    }

    with open(
        output_dir / "dataset_schema.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            schema,
            file,
            indent=4,
        )

    print("\nDataset saved to:")
    print(output_dir)


if __name__ == "__main__":
    main()
