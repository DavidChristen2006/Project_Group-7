
from pathlib import Path
import json

import numpy as np
import pandas as pd


ROOT = Path("data/processed")
OUTPUT = ROOT / "ml_datasets" / "aapl_h21"

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

TARGETS = [
    "Hedge Improvement",
    "Drawdown Improvement",
]

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

DATE_COLUMNS = [
    "Decision Date",
    "Outcome Date",
    "Expiry",
    "Option Expiry",
]

REQUIRED_COLUMNS = list(dict.fromkeys(
    KEY_COLUMNS
    + NUMERIC_FEATURES
    + TARGETS
    + ["Candidate ID", "Outcome Date"]
))


def find_experiments() -> list[Path]:
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

        df["Decision Date"] = (
            df["Decision Date"].dt.normalize()
        )

        df["Expiry"] = df["Expiry"].dt.normalize()

        df["Candidate ID"] = (
            df["Candidate ID"].astype(str)
        )

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

    if not df["Underlying RIC"].eq("AAPL.O").all():
        raise ValueError("Unexpected underlying RIC.")

    if not df["Hedge Horizon Days"].eq(21).all():
        raise ValueError("Unexpected hedge horizon.")

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

    group_id = pd.util.hash_pandas_object(
        duplicate_key,
        index=False,
    )

    df = df.copy()
    df["_group_id"] = group_id.to_numpy()

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
            else:
                if values.nunique(dropna=False) > 1:
                    raise ValueError(
                        "Conflicting duplicate candidate: "
                        f"{column}"
                    )

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

    features = df[NUMERIC_FEATURES].copy()

    features["Put Moneyness"] = (
        df["Long Put Strike"] / df["Spot Price"]
    )

    features["Call Moneyness"] = (
        df["Short Call Strike"] / df["Spot Price"]
    )

    features["Short Put Moneyness"] = (
        df["Short Put Strike"] / df["Spot Price"]
    )

    features["Has Short Call"] = (
        df["Short Call RIC"].notna().astype(int)
    )

    features["Has Short Put"] = (
        df["Short Put RIC"].notna().astype(int)
    )

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
