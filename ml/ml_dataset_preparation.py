
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ml_datasets"
    / "aapl_historical_hedging_2022_2023.parquet"
)

TEST_START = pd.Timestamp("2023-07-03")

TARGET_COLUMNS = [
    "Hedge Improvement",
    "Maximum Drawdown",
]

NUMERIC_FEATURES = [
    "Hedge Horizon Days",
    "Days to Expiry",
    "Volatility_20D",
    "Volatility_60D",
    "Momentum_20D",
    "Momentum_60D",
    "Historical_Drawdown_20D",
    "Long_Put_Strike_Pct",
    "Long_Put_Spread_Pct",
    "Short_Call_Strike_Pct",
    "Short_Call_Spread_Pct",
    "Short_Put_Strike_Pct",
    "Short_Put_Spread_Pct",
    "Net Premium Pct",
    "Has_Short_Call",
    "Has_Short_Put",
]

CATEGORICAL_FEATURES = [
    "Strategy",
]

FEATURE_COLUMNS = (
    NUMERIC_FEATURES
    + CATEGORICAL_FEATURES
)

OPTIONAL_LEG_FEATURES = [
    "Short_Call_Strike_Pct",
    "Short_Call_Spread_Pct",
    "Short_Put_Strike_Pct",
    "Short_Put_Spread_Pct",
]



def load_ml_dataset(dataset_path=None):
    if dataset_path is None:
        dataset_path = DATA_PATH

    dataset_path = Path(dataset_path)

    if not dataset_path.is_file():
        raise FileNotFoundError(
            f"ML dataset not found: {dataset_path}"
        )

    df = pd.read_parquet(dataset_path).copy()


    required = set(
        FEATURE_COLUMNS
        + TARGET_COLUMNS
        + [
            "Decision Date",
            "Outcome Date",
            "Short_Call_RIC",
            "Short_Put_RIC",
        ]
    ) - {
        "Has_Short_Call",
        "Has_Short_Put",
    }

    missing = sorted(
        required - set(df.columns)
    )

    if missing:
        raise ValueError(
            f"Missing dataset columns: {missing}"
        )

    df["Decision Date"] = pd.to_datetime(
        df["Decision Date"]
    ).dt.normalize()

    df["Outcome Date"] = pd.to_datetime(
        df["Outcome Date"]
    ).dt.normalize()

    df["Has_Short_Call"] = (
        df["Short_Call_RIC"].notna().astype(int)
    )

    df["Has_Short_Put"] = (
        df["Short_Put_RIC"].notna().astype(int)
    )

    # Check the expected option structure.
    expected_call = (
        df["Strategy"] == "Collar"
    ).astype(int)

    expected_put = (
        df["Strategy"] == "Put Spread"
    ).astype(int)

    if not df["Has_Short_Call"].equals(expected_call):
        raise ValueError(
            "Unexpected Short Call structure."
        )

    if not df["Has_Short_Put"].equals(expected_put):
        raise ValueError(
            "Unexpected Short Put structure."
        )

    # Only structurally absent option legs
    # receive a zero representation.
    for column in OPTIONAL_LEG_FEATURES:
        presence_column = (
            "Has_Short_Call"
            if column.startswith("Short_Call")
            else "Has_Short_Put"
        )

        absent = df[presence_column] == 0

        df.loc[absent, column] = 0.0

        if df.loc[~absent, column].isna().any():
            raise ValueError(
                f"Missing active-leg values: {column}"
            )

    for column in NUMERIC_FEATURES + TARGET_COLUMNS:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    numeric_values = df[
        NUMERIC_FEATURES + TARGET_COLUMNS
    ].to_numpy(dtype=float)

    if not np.isfinite(numeric_values).all():
        raise ValueError(
            "Non-finite ML features or targets found."
        )

    if (
        df["Outcome Date"]
        <= df["Decision Date"]
    ).any():
        raise ValueError(
            "Invalid decision/outcome dates."
        )

    df = df.sort_values(
        [
            "Decision Date",
            "Hedge Horizon Days",
            "Strategy",
        ]
    ).reset_index(drop=True)

    return df


def temporal_train_test_split(
    df,
    test_start=TEST_START
):
    cutoff = pd.Timestamp(test_start).normalize()

    train = df.loc[
        (df["Decision Date"] < cutoff)
        & (df["Outcome Date"] < cutoff)
    ].copy()

    test = df.loc[
        df["Decision Date"] >= cutoff
    ].copy()

    excluded = df.loc[
        (df["Decision Date"] < cutoff)
        & (df["Outcome Date"] >= cutoff)
    ].copy()

    if train.empty or test.empty:
        raise ValueError(
            "Training or test dataset is empty."
        )

    assert (
        train["Outcome Date"] < cutoff
    ).all()

    assert (
        test["Decision Date"] >= cutoff
    ).all()

    return train, test, excluded


if __name__ == "__main__":
    dataset = load_ml_dataset()

    train, test, excluded = (
        temporal_train_test_split(dataset)
    )

    print("\n--- ML DATASET PREPARATION ---")

    print(f"Total observations: {len(dataset)}")
    print(f"Numeric features: {len(NUMERIC_FEATURES)}")
    print(
        f"Categorical features: "
        f"{len(CATEGORICAL_FEATURES)}"
    )

    print(f"Training observations: {len(train)}")
    print(f"Test observations: {len(test)}")
    print(f"Excluded observations: {len(excluded)}")

    print("\nTraining strategy distribution:")
    print(train["Strategy"].value_counts())

    print("\nTest strategy distribution:")
    print(test["Strategy"].value_counts())

    print("\nML dataset preparation passed.")
