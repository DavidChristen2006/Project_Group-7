
import argparse
from pathlib import Path

import pandas as pd

# the project dictionary is calculated from the script's location
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# defines a dictionary which contains historical backtest runs
RUNS_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "historical_runs"
)

# defines a dictionary where the final combined dataset will be saved
DATASETS_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ml_datasets"
)

# required unique columns which identify one observation 
UNIQUE_COLUMNS = [
    "Underlying RIC",
    "Decision Date",
    "Hedge Horizon Days",
    "Strategy",
]


def discover_historical_runs(ric, cost_tag=None):
    """The function searches RUNS_DIR for matching run folders"""
    
    # raises an Error if the historical-runs dictionary does not exist 
    if not RUNS_DIR.is_dir():
        raise FileNotFoundError(
            f"Historical runs directory not found: {RUNS_DIR}"
        )

    # convert the RIC into a folder prefix (ex. APPL.0 becomes APPL_0_)
    prefix = ric.replace(".", "_") + "_"
    paths = []

    for folder in sorted(RUNS_DIR.iterdir()):
        # considers only folders with this prefix and skips files that 
        # are not dictionaries, folders unrelated to the requested RIC, 
        # legacy folders without _TC in their name, or runs with a 
        # different transaction-cost tag
        if not folder.is_dir():
            continue

        if not folder.name.startswith(prefix):
            continue

        if "_TC" not in folder.name:
            print(f"Skipping legacy run: {folder.name}")
            continue

        actual_tag = folder.name.rsplit("_TC", 1)[-1]

        if cost_tag is not None and actual_tag != cost_tag:
            continue

        # for each valid folder, it looks for 
        # 'historical_ml_features.parquet' 
        path = folder / "historical_ml_features.parquet"

        if path.is_file():
            paths.append(path)

    return paths


def build_dataset(
    ric,
    start_date,
    end_date,
    cost_tag=None,
):
    
    """Converts the requested boundaries into normalized timestamps"""
    
    # remove the time components, leaving only dates 
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()

    # invalid dates are rejected 
    if end < start:
        raise ValueError(
            "End date must not precede start date."
        )

    # searches for matching runs 
    paths = discover_historical_runs(
        ric,
        cost_tag=cost_tag,
    )

    # raises an error if no matching runs are found 
    if not paths:
        raise ValueError(
            f"No historical runs found for {ric}."
        )

    candidates = []

    # every run requires two columns
    # UNIQUE_COLUMNS and Outcome Date 
    required = set(
        UNIQUE_COLUMNS + ["Outcome Date"]
    )

    for path in paths:
        df = pd.read_parquet(path).copy()

        if not required.issubset(df.columns):
            raise ValueError(
                f"Missing required columns in {path}"
            )

        df["Decision Date"] = pd.to_datetime(
            df["Decision Date"]
        ).dt.normalize()

        df["Outcome Date"] = pd.to_datetime(
            df["Outcome Date"]
        ).dt.normalize()

        # keep only observations that 
        # match the requested RIC, have a decision date on or after start
        # and have a decision date on or before end
        df = df.loc[
            (df["Underlying RIC"] == ric)
            & (df["Decision Date"] >= start)
            & (df["Decision Date"] <= end)
        ].copy()

        if df.empty:
            continue

        # rejects duplicates inside each individual run 
        if df.duplicated(UNIQUE_COLUMNS).any():
            raise ValueError(
                f"Duplicate observations within {path}"
            )

        tag = path.parent.name.rsplit("_TC", 1)[-1]

        # for each valid run, the script stores
        # the file path, the filtered DataFrame, the transaction-cost tag
        # and the set of column names 
        candidates.append({
            "path": path,
            "data": df,
            "tag": tag,
            "schema": set(df.columns),
        })

    if not candidates:
        raise ValueError(
            "No observations in the requested period."
        )

    # ensures that a transaction-cost configuration was selected 
    tags = {item["tag"] for item in candidates}

    # if runs with different transaction-cost tags are found, the script
    # raises an error 
    if len(tags) != 1:
        raise ValueError(
            "Multiple transaction-cost configurations found. "
            "Specify --cost-tag."
        )

    # Prioritize runs with more observations.
    # This puts the finalized 2022 run before
    # its smaller development runs.
    candidates.sort(
        key=lambda item: (
            -len(item["data"]),
            str(item["path"]),
        )
    )

    reference_schema = candidates[0]["schema"]

    # create a set for observation keys that have already been
    # included
    frames = []
    selected_keys = set()

    for item in candidates:
        df = item["data"]
        path = item["path"]

        # convert unique-column values into a MultiIndex
        keys = pd.MultiIndex.from_frame(
            df[UNIQUE_COLUMNS]
        ).tolist()

        # identifies key not already selected 
        # remove overlapping observations while preserving
        # observations unique to the run 
        new_mask = [
            key not in selected_keys
            for key in keys
        ]

        # keeps only those new observations 
        new_df = df.loc[new_mask].copy()

        # if the run completely overlapps, the run is skipped
        if new_df.empty:
            print(
                f"Skipping overlapping run: "
                f"{path.parent.name}"
            )
            continue

        # compare the current run's columns with the reference run
        # only checked if new observations are found 
        if item["schema"] != reference_schema:
            raise ValueError(
                "Incompatible schema contains additional "
                "observations and cannot be silently skipped: "
                f"{path.parent.name}"
            )

        # combine the selected frames 
        # each retained DataFrame is added to a list 
        frames.append(new_df)

        new_keys = pd.MultiIndex.from_frame(
            new_df[UNIQUE_COLUMNS]
        ).tolist()

        selected_keys.update(new_keys)

        print(
            f"Loaded {len(new_df)} observations "
            f"from {path.parent.name}"
        )

    # all frames are combined and ignore_index=True creates
    # a fresh row index from 0 to n-1
    combined = pd.concat(
        frames,
        ignore_index=True
    )

    # checks for remaining duplicates 
    # safety check after combining the runs 
    if combined.duplicated(UNIQUE_COLUMNS).any():
        raise ValueError(
            "Duplicate observations remain."
        )

    # every outcome date must be after its decision date, otherwise
    # the script rejects the dataset as temporally invalid 
    if (
        combined["Outcome Date"]
        <= combined["Decision Date"]
    ).any():
        raise ValueError(
            "Invalid outcome dates."
        )

    # sort the final dataset
    # the combined dataset is sorted by the underlying RIC, the decision
    # date, the Hedge horizon and the strategy
    combined = combined.sort_values(
        UNIQUE_COLUMNS
    ).reset_index(drop=True)

    # builds the output path
    DATASETS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # replaces dots in the RIC with underscores 
    safe_ric = ric.replace(".", "_")
    tag = next(iter(tags))

    # the output filename follows the defined pattern
    # which is <RIC>_<start_date>_<end_date>_TC<tag>.parquet 
    # ex. AAPL_O_20220101_20231231_TC00a56a790ea1.parquet
    output_path = (
        DATASETS_DIR
        / (
            f"{safe_ric}_{start:%Y%m%d}_"
            f"{end:%Y%m%d}_TC{tag}.parquet"
        )
    )

    # the final dataset is saved and reported 
    combined.to_parquet(
        output_path,
        index=False
    )

    # prints the RIC, the number of observations, the number of 
    # unique decision dates, the transaction-cost tag and 
    # the output file location 
    print("\n--- AUTOMATIC DATASET ---")
    print(f"Underlying: {ric}")
    print(f"Observations: {len(combined)}")
    print(
        "Decision dates: "
        f"{combined['Decision Date'].nunique()}"
    )
    print(f"Cost tag: {tag}")
    print(f"Saved to: {output_path}")

    # returns the path to the saved dataset 
    return output_path


def main():
    
    # four arguments are defined
    # the undelying RIC
    # the earliest decision date
    # the latest decision date
    # optional transaction-cost configuration (not required)
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ric",
        required=True,
    )

    parser.add_argument(
        "--start",
        required=True,
    )

    parser.add_argument(
        "--end",
        required=True,
    )

    parser.add_argument(
        "--cost-tag",
        default=None,
    )

    args = parser.parse_args()

    build_dataset(
        ric=args.ric,
        start_date=args.start,
        end_date=args.end,
        cost_tag=args.cost_tag,
    )


if __name__ == "__main__":
    main()
