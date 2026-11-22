"""Consolidate several historical AAPL backtest runs into one dataset.

The script loads the feature files of three historical runs, checks that
they are compatible and free of duplicates and combines them into one
parquet file with all columns of the runs.

Checks before the runs are combined:
    - exactly one run folder exists for each prefix in RUN_PREFIXES
    - all runs use the same transaction-cost configuration
    - all runs have the same columns

Checks on the combined data:
    - the identification columns have no missing values
    - no observation (same underlying, decision date, horizon and
      strategy) occurs twice; duplicates are an error and are not
      merged, so the runs must not overlap
    - every outcome date lies after its decision date

Run from project root:
    py -m ml.<module_name>

Reads:  historical_ml_features.parquet in each run folder under
        RUNS_DIR.
Writes: aapl_historical_hedging_2022_2023_full.parquet in OUTPUT_DIR.

GenAI usage: Claude (Anthropic) was used to explain the code and to
draft, review and correct comments and docstrings. The comments were
checked and adapted by the author(s).
"""

from pathlib import Path

import pandas as pd


# two folders above this file
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# folder that contains all historical run folders
RUNS_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "historical_runs"
)

# folder where the consolidated dataset is written
OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ml_datasets"
)


# The run folders to combine. Each prefix contains the underlying, the
# range of decision dates and the run settings (horizons 21, 63 and 126
# days, ...) and ends with "_TC", so it matches the folder whatever
# its transaction-cost tag is.
RUN_PREFIXES = [
    "AAPL_O_20220103_20221201_H21-63-126_S1000_K500_TC",
    "AAPL_O_20230403_20230601_H21-63-126_S1000_K500_TC",
    "AAPL_O_20230703_20231201_H21-63-126_S1000_K500_TC",
]


# columns that together identify one observation
KEY_COLUMNS = [
    "Underlying RIC",
    "Decision Date",
    "Hedge Horizon Days",
    "Strategy",
]


def find_run(prefix):
    """Find the run folder that belongs to a prefix.

    A folder matches if its name starts with the prefix and it contains
    historical_ml_features.parquet.

    Args:
        prefix: Start of the run folder name, one of RUN_PREFIXES.

    Returns:
        Path to the run folder.

    Raises:
        ValueError: if no folder or more than one folder matches. The
            run must be unambiguous so that the correct data is used.
    """

    matches = [
        path
        for path in RUNS_DIR.glob(f"{prefix}*")
        if path.is_dir()
        and (path / "historical_ml_features.parquet").exists()
    ]

    if len(matches) != 1:
        raise ValueError(
            f"Expected one run for {prefix}, "
            f"found {len(matches)}."
        )

    return matches[0]


def consolidate_historical_datasets():
    """Load, validate and combine the historical runs into one dataset.

    Returns:
        DataFrame with all observations of the runs, sorted by
        KEY_COLUMNS and with a fresh index. The same data is saved as a
        parquet file.

    Raises:
        ValueError: if a run is missing or ambiguous, required columns
            are missing, transaction-cost configurations or column sets
            differ between the runs, identification values are missing,
            duplicate observations exist, or an outcome date is not
            after its decision date.

    Side effects:
        Creates OUTPUT_DIR if needed, writes the parquet file and
        prints the number of observations per run, a summary and the
        coverage by horizon.
    """

    frames = []
    cost_tags = set()
    schemas = []

    for prefix in RUN_PREFIXES:
        run_dir = find_run(prefix)

        # the transaction-cost tag is the part of the folder name after
        # the last "_TC"
        cost_tag = run_dir.name.rsplit("_TC", 1)[-1]

        cost_tags.add(cost_tag)

        frame = pd.read_parquet(
            run_dir / "historical_ml_features.parquet"
        )

        # identification columns plus a few backtest results that every
        # run must contain
        required = KEY_COLUMNS + [
            "Outcome Date",
            "Final Return",
            "Maximum Drawdown",
            "Total Transaction Costs",
        ]

        missing = [
            column
            for column in required
            if column not in frame.columns
        ]

        if missing:
            raise ValueError(
                f"{run_dir.name}: missing columns {missing}"
            )

        # remember the columns of each run to compare them later
        schemas.append(set(frame.columns))

        print(
            f"Loaded {len(frame)} observations "
            f"from {run_dir.name}"
        )

        frames.append(frame)

    # results with different transaction costs must not be mixed
    if len(cost_tags) != 1:
        raise ValueError(
            "Incompatible transaction-cost configurations."
        )

    # all runs must have the same columns, otherwise the combined table
    # would contain gaps
    if any(schema != schemas[0] for schema in schemas[1:]):
        raise ValueError(
            "Historical ML feature schemas differ. "
            "Review the generating pipelines."
        )

    combined = pd.concat(
        frames,
        ignore_index=True
    )

    combined["Decision Date"] = pd.to_datetime(
        combined["Decision Date"]
    )

    combined["Outcome Date"] = pd.to_datetime(
        combined["Outcome Date"]
    )

    if combined[KEY_COLUMNS].isna().any().any():
        raise ValueError(
            "Missing strategy identification values."
        )

    # keep=False marks all copies of a duplicate, so that the error
    # message shows every affected row
    if combined[KEY_COLUMNS].duplicated().any():
        duplicates = combined.loc[
            combined[KEY_COLUMNS].duplicated(
                keep=False
            ),
            KEY_COLUMNS
        ]

        raise ValueError(
            "Duplicate strategy observations:\n"
            f"{duplicates.to_string(index=False)}"
        )

    # an outcome can only be known after the decision was made; a
    # missing outcome date also fails this check
    if not (
        combined["Outcome Date"]
        > combined["Decision Date"]
    ).all():
        raise ValueError(
            "Found outcomes on or before decision dates."
        )

    combined = combined.sort_values(
        KEY_COLUMNS
    ).reset_index(drop=True)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        OUTPUT_DIR
        / "aapl_historical_hedging_2022_2023_full.parquet"
    )

    combined.to_parquet(
        output_path,
        index=False
    )

    print("\n--- CONSOLIDATED DATASET ---")
    print(f"Observations: {len(combined)}")
    print(
        "Decision dates: "
        f"{combined['Decision Date'].nunique()}"
    )

    # number of observations per hedge horizon (21, 63 and 126 days)
    print("\nCoverage by horizon:")
    print(
        combined.groupby(
            "Hedge Horizon Days"
        )["Strategy"].count()
    )

    print(f"\nSaved to: {output_path}")

    return combined


if __name__ == "__main__":
    consolidate_historical_datasets()