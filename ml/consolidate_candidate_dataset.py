"""Build the ML dataset for AAPL hedging candidates from backtest results.

The script collects the completed candidate backtests of all matching
experiments, validates them, removes candidates that occur in several
experiments and splits the result into features, targets and metadata
for the ML models.

Procedure:
    1. Find all experiment folders with the required settings.
    2. Load their results and check that all required columns exist.
    3. Validate the data (underlying, horizon, targets, dates) and remove
       duplicate candidates. Duplicates with conflicting values are an
       error and are never resolved silently.
    4. Build the model features from information known at the decision
       date (entry prices, strikes, premiums, strategy). Outcomes are
       never used as features.
    5. Save the dataset, features, targets, metadata and a manifest.

Features, targets and metadata have the same row order: row i of every
file describes the same candidate. Later scripts rely on this.

Run from project root:
    py -m ml.consolidate_candidate_cataset

Reads:  candidate_ml_results.parquet of every matching experiment in
        ROOT / "candidate_backtests".
Writes: candidate_dataset.parquet, features.parquet, targets.parquet,
        metadata.parquet and dataset_manifest.json in OUTPUT.

GenAI usage: Claude (Anthropic) was used to explain the code and to
draft, review and correct comments and docstrings. The comments were
checked and adapted by the author(s).
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd


ROOT = Path("data/processed")

# all dataset files of this script are written here
OUTPUT = ROOT / "ml_datasets" / "aapl_h21"

# columns that together identify one position; two rows with the same
# key are the same candidate
KEY_COLUMNS = [
    "Underlying RIC",
    "Decision Date",
    "Hedge Horizon Days",
    "Strategy",
    "Expiry",
    "Long Put RIC",
    "Short Call RIC",
    "Short Put RIC",
]

# numeric characteristics of the position at entry, used as model inputs
NUMERIC_FEATURES = [
    "Spot Price",
    "Shares",
    "Contracts",
    "Long Put Strike Pct",
    "Short Call Strike Pct",
    "Short Put Strike Pct",
    "Long Put Ask",
    "Short Call Bid",
    "Short Put Bid",
    "Net Premium Pct",
    "Option Legs",
    "Days to Expiry",
]

# values the models learn to predict: return benefit and risk benefit
# of the hedge
TARGETS = [
    "Hedge Improvement",
    "Drawdown Improvement",
]

# backtest outcomes; they are only known after the outcome date and
# must not be used as features
OUTCOME_COLUMNS = [
    "Final Return",
    "Maximum Drawdown",
    "Final P&L",
    "Unhedged Final Return",
    "Unhedged Maximum Drawdown",
    "Hedge Improvement",
    "Drawdown Improvement",
    "Gross Final Return",
    "Gross Maximum Drawdown",
    "Gross Final P&L",
    "Total Transaction Costs",
    "Option Commissions",
    "Option Fixed Fees",
    "Option Slippage",
    "Stock Commissions",
    "Observations",
]

# columns that are converted to dates (if they exist)
DATE_COLUMNS = [
    "Decision Date",
    "Outcome Date",
    "Expiry",
    "Option Expiry",
]

# every result file must contain these columns; dict.fromkeys removes
# duplicates and keeps the order
REQUIRED_COLUMNS = list(dict.fromkeys(
    KEY_COLUMNS
    + NUMERIC_FEATURES
    + TARGETS
    + ["Candidate ID", "Outcome Date"]
))


def find_experiments() -> list[Path]:
    """Find the result files of all experiments with the required settings.

    An experiment folder qualifies if its name starts with "AAPL_O_" and
    contains the run settings "_H21_S1000_" and
    "_EXP6500-12500_STEP5000_CFG". Experiments with other settings
    are ignored.

    Returns:
        Sorted list of paths to candidate_ml_results.parquet. The list
        can be empty.
    """

    base = ROOT / "candidate_backtests"

    paths = sorted(
        base.glob("*/candidate_ml_results.parquet")
    )

    eligible = []

    for path in paths:
        name = path.parent.name

        if not name.startswith("AAPL_O_"):
            continue

        if "_H21_S1000_" not in name:
            continue

        if "_EXP6500-12500_STEP5000_CFG" not in name:
            continue

        eligible.append(path)

    return eligible


def load_experiments(
    paths: list[Path],
) -> pd.DataFrame:
    """Load and combine the results of several experiments.

    Checks that all required columns exist, converts the date columns
    and adds the experiment name as "Source Experiment".

    Args:
        paths: Paths to candidate_ml_results.parquet files, as returned
            by find_experiments().

    Returns:
        DataFrame with the rows of all experiments, in the order of the
        paths. Candidates that occur in several experiments appear
        several times.

    Raises:
        ValueError: if no experiment was found or a result file lacks
            required columns.

    Side effects:
        Prints the number of rows of each experiment.
    """

    frames = []

    for path in paths:
        df = pd.read_parquet(path).copy()

        missing = set(REQUIRED_COLUMNS) - set(df.columns)

        if missing:
            raise ValueError(
                f"{path.parent.name}: missing columns "
                f"{sorted(missing)}"
            )

        for column in DATE_COLUMNS:
            if column in df.columns:
                df[column] = pd.to_datetime(df[column])

        # remove the time of day so that dates compare equal
        df["Decision Date"] = (
            df["Decision Date"].dt.normalize()
        )

        df["Expiry"] = df["Expiry"].dt.normalize()

        df["Candidate ID"] = (
            df["Candidate ID"].astype(str)
        )

        # remember where each row came from
        df["Source Experiment"] = path.parent.name

        frames.append(df)

        print(
            f"{path.parent.name}: {len(df)} rows"
        )

    if not frames:
        raise ValueError(
            "No eligible historical experiments found."
        )

    return pd.concat(
        frames,
        ignore_index=True,
        sort=False,
    )


def validate_and_deduplicate(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Validate the combined results and remove duplicate candidates.

    A duplicate is the same position (same KEY_COLUMNS) on the same
    decision date that occurs in more than one experiment. Before
    duplicates are removed, they must agree in all entry
    characteristics, targets and outcomes. The first copy, sorted by
    decision date and experiment name, is kept.

    Args:
        df: Combined results as returned by load_experiments().

    Returns:
        DataFrame with one row per unique candidate and a fresh index.

    Raises:
        ValueError: if the underlying or the horizon is unexpected,
            identifiers or targets are missing or not finite, an
            outcome date precedes its decision date, or duplicates
            contain conflicting values.

    Side effects:
        Prints the number of rows that belong to duplicate groups.
    """

    if not df["Underlying RIC"].eq("AAPL.O").all():
        raise ValueError("Unexpected underlying RIC.")

    if not df["Hedge Horizon Days"].eq(21).all():
        raise ValueError("Unexpected hedge horizon.")

    # the first five key columns must always be present; the option
    # RICs may be missing because not every strategy uses every leg
    if df[KEY_COLUMNS[:5]].isna().any().any():
        raise ValueError("Missing required candidate identifiers.")

    if df[TARGETS].isna().any().any():
        raise ValueError("Missing target values.")

    if not np.isfinite(
        df[TARGETS].to_numpy(dtype=float)
    ).all():
        raise ValueError("Non-finite target values.")

    if (
        df["Outcome Date"] < df["Decision Date"]
    ).any():
        raise ValueError("Outcome date precedes decision date.")

    # A duplicate represents the same position on
    # the same decision date and horizon.
    #
    # Missing short legs are represented consistently
    # with a sentinel, only for duplicate detection.

    duplicate_key = df[KEY_COLUMNS].copy()

    for column in [
        "Long Put RIC",
        "Short Call RIC",
        "Short Put RIC",
    ]:
        duplicate_key[column] = (
            duplicate_key[column].fillna("<NONE>")
        )

    # one number per row, computed from the key columns; rows with the
    # same number belong to the same candidate
    group_id = pd.util.hash_pandas_object(
        duplicate_key,
        index=False,
    )

    df = df.copy()
    df["_group_id"] = group_id.to_numpy()

    # keep=False marks all copies, not only the later ones
    duplicated = df.duplicated(
        "_group_id",
        keep=False,
    )

    print(
        "Rows belonging to duplicate groups:",
        int(duplicated.sum()),
    )

    # Never silently choose between conflicting
    # historical outcomes or entry characteristics.

    # columns in which all copies of a candidate must agree
    compare_columns = list(dict.fromkeys(
        NUMERIC_FEATURES
        + TARGETS
        + [
            "Outcome Date",
            "Final Return",
            "Maximum Drawdown",
            "Unhedged Final Return",
            "Unhedged Maximum Drawdown",
        ]
    ))

    for _, group in df.loc[duplicated].groupby(
        "_group_id"
    ):
        for column in compare_columns:
            values = group[column]

            # numbers may differ by rounding only (allclose); NaN in
            # the same place counts as equal
            if pd.api.types.is_numeric_dtype(values):
                arr = pd.to_numeric(
                    values, errors="coerce"
                ).to_numpy(dtype=float)

                if not np.allclose(
                    arr,
                    arr[0],
                    equal_nan=True,
                    rtol=1e-10,
                    atol=1e-8,
                ):
                    raise ValueError(
                        "Conflicting duplicate candidate: "
                        f"{column}; sources: "
                        f"{group['Source Experiment'].tolist()}"
                    )
            # other types (e.g. dates) must be identical
            else:
                if values.nunique(dropna=False) > 1:
                    raise ValueError(
                        "Conflicting duplicate candidate: "
                        f"{column}"
                    )

    # all copies agree, so it does not matter which one is kept
    df = (
        df.sort_values(
            ["Decision Date", "Source Experiment"]
        )
        .drop_duplicates("_group_id")
        .drop(columns="_group_id")
        .reset_index(drop=True)
    )

    return df


def build_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Build the model features from the entry characteristics.

    Starts with NUMERIC_FEATURES and adds the moneyness of each option
    (strike divided by spot price), 0/1 flags for the presence of a
    short call and a short put, and one 0/1 column per strategy. Only
    information known at the decision date is used.

    Args:
        df: Unique candidates as returned by validate_and_deduplicate().
            It must contain the columns "Long Put Strike",
            "Short Call Strike" and "Short Put Strike".

    Returns:
        DataFrame with the same row order as df: the 12 numeric
        features, 3 moneyness columns, 2 flag columns and one column
        per strategy. Missing values are kept (e.g. the strike of a
        leg that a strategy does not have).
    """

    features = df[NUMERIC_FEATURES].copy()

    # moneyness = strike relative to the current price, e.g. 1.05 for
    # a strike 5% above the price
    features["Put Moneyness"] = (
        df["Long Put Strike"] / df["Spot Price"]
    )

    features["Call Moneyness"] = (
        df["Short Call Strike"] / df["Spot Price"]
    )

    features["Short Put Moneyness"] = (
        df["Short Put Strike"] / df["Spot Price"]
    )

    # 1 if the strategy contains this leg, otherwise 0
    features["Has Short Call"] = (
        df["Short Call RIC"].notna().astype(int)
    )

    features["Has Short Put"] = (
        df["Short Put RIC"].notna().astype(int)
    )

    # one-hot encoding of the strategy
    strategy_features = pd.get_dummies(
        df["Strategy"],
        prefix="Strategy",
        dtype=int,
    )

    features = pd.concat(
        [features, strategy_features],
        axis=1,
    )

    return features


def main() -> None:
    """Build the candidate dataset and save all output files.

    Finds and loads the experiments, validates and deduplicates the
    candidates, builds features, targets and metadata and writes them
    together with a manifest to OUTPUT.

    Raises:
        ValueError: if no experiment is found or the data validation
            fails (raised by the called functions).

    Side effects:
        Creates OUTPUT if needed, writes four parquet files and
        dataset_manifest.json and prints a summary with the yearly
        coverage.
    """

    paths = find_experiments()

    print("\n--- ELIGIBLE EXPERIMENTS ---")
    print("Experiment folders:", len(paths))

    dataset = load_experiments(paths)

    print("\nRaw rows:", len(dataset))

    dataset = validate_and_deduplicate(dataset)

    print("Unique candidates:", len(dataset))
    print(
        "Unique decision dates:",
        dataset["Decision Date"].nunique(),
    )

    features = build_features(dataset)

    # Retain missing feature values. Imputation
    # must be learned on training data only.
    labels = dataset[TARGETS].copy()

    # identifiers and source information; not used as model input
    metadata_columns = list(dict.fromkeys(
        KEY_COLUMNS
        + [
            "Candidate ID",
            "Outcome Date",
            "Source Experiment",
        ]
    ))

    metadata = dataset[metadata_columns].copy()

    OUTPUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    # the files share the row order of dataset
    dataset.to_parquet(
        OUTPUT / "candidate_dataset.parquet",
        index=False,
    )

    features.to_parquet(
        OUTPUT / "features.parquet",
        index=False,
    )

    labels.to_parquet(
        OUTPUT / "targets.parquet",
        index=False,
    )

    metadata.to_parquet(
        OUTPUT / "metadata.parquet",
        index=False,
    )

    # summary of the dataset for documentation and reproducibility
    manifest = {
        "underlying": "AAPL.O",
        "hedge_horizon_days": 21,
        "experiment_count": len(paths),
        "candidate_count": len(dataset),
        "decision_date_count": int(
            dataset["Decision Date"].nunique()
        ),
        "first_decision_date": str(
            dataset["Decision Date"].min().date()
        ),
        "last_decision_date": str(
            dataset["Decision Date"].max().date()
        ),
        "feature_columns": features.columns.tolist(),
        "target_columns": TARGETS,
        "source_experiments": [
            path.parent.name for path in paths
        ],
    }

    with (
        OUTPUT / "dataset_manifest.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(
            manifest,
            file,
            indent=2,
        )

    print("\n--- DATASET CREATED ---")
    print("Output:", OUTPUT)
    print("Candidates:", len(dataset))
    print("Features:", features.shape[1])
    print("Targets:", labels.shape[1])
    print(
        "Decision dates:",
        dataset["Decision Date"].nunique(),
    )

    # candidates and decision dates per year
    print("\n--- YEARLY COVERAGE ---")
    print(
        dataset.groupby(
            dataset["Decision Date"].dt.year
        ).agg(
            Candidates=("Candidate ID", "size"),
            Decision_Dates=(
                "Decision Date", "nunique"
            ),
        ).to_string()
    )


if __name__ == "__main__":
    main()