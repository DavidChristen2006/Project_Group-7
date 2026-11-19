
import argparse
import json

from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.historical_data_pipeline import (
    load_price_history,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_INPUT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ml_datasets"
    / "AAPL_O_20220101_20241231_H21"
    / "completed_candidates.parquet"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ml_feature_datasets"
)

MARKET_FEATURES = [
    "Momentum_21D",
    "Momentum_63D",
    "Realized_Volatility_21D",
    "Realized_Volatility_63D",
    "Historical_Drawdown_63D",
    "Distance_From_High_63D",
]


def load_equity_prices(ric):
    prices = load_price_history(ric).copy()

    if "Close" not in prices.columns:
        raise ValueError(
            "Historical equity data has no Close column."
        )

    if "RIC" in prices.columns:
        identifiers = (
            prices["RIC"].dropna().astype(str).unique()
        )

        if (
            len(identifiers) != 1
            or identifiers[0] != ric
        ):
            raise ValueError(
                f"Unexpected RIC values: {identifiers}"
            )

    prices.index = pd.to_datetime(
        prices.index,
        errors="raise",
    )

    if prices.index.tz is not None:
        raise ValueError(
            "Expected timezone-naive trading dates."
        )

    prices.index = prices.index.normalize()
    prices = prices.sort_index()

    if prices.index.has_duplicates:
        raise ValueError(
            "Duplicate trading dates."
        )

    close = pd.to_numeric(
        prices["Close"],
        errors="coerce",
    ).astype(float)

    if (
        close.isna().any()
        or not np.isfinite(close.to_numpy()).all()
        or close.le(0).any()
    ):
        raise ValueError(
            "Invalid historical closing prices."
        )

    close.name = "Close"

    return close


def calculate_market_features(close):
    """
    All rolling features use prices available
    strictly BEFORE the decision date.
    """

    daily_returns = close.pct_change(
        fill_method=None
    )

    # The previous session's close is the
    # latest price allowed in the features.
    previous_close = close.shift(1)

    features = pd.DataFrame(
        index=close.index
    )

    features["Momentum_21D"] = (
        close.shift(1) / close.shift(22) - 1
    )

    features["Momentum_63D"] = (
        close.shift(1) / close.shift(64) - 1
    )

    past_returns = daily_returns.shift(1)

    features["Realized_Volatility_21D"] = (
        past_returns
        .rolling(21, min_periods=21)
        .std(ddof=1)
        * np.sqrt(252)
    )

    features["Realized_Volatility_63D"] = (
        past_returns
        .rolling(63, min_periods=63)
        .std(ddof=1)
        * np.sqrt(252)
    )

    def window_max_drawdown(values):
        running_peak = np.maximum.accumulate(
            values
        )

        drawdowns = values / running_peak - 1

        return float(np.min(drawdowns))

    features["Historical_Drawdown_63D"] = (
        previous_close
        .rolling(63, min_periods=63)
        .apply(
            window_max_drawdown,
            raw=True,
        )
    )

    trailing_high = previous_close.rolling(
        63,
        min_periods=63,
    ).max()

    features["Distance_From_High_63D"] = (
        previous_close / trailing_high - 1
    )

    # Audit metadata, not model predictors.
    features["Feature_Cutoff_Date"] = (
        pd.Series(close.index, index=close.index)
        .shift(1)
    )

    features["Previous_Close"] = previous_close

    return features


def enrich_dataset(dataset, close):
    required = [
        "Underlying RIC",
        "Decision Date",
        "Candidate Key",
        "Spot Price",
        "Hedge Improvement",
        "Drawdown Improvement",
    ]

    missing = [
        column for column in required
        if column not in dataset.columns
    ]

    if missing:
        raise ValueError(
            f"Missing dataset columns: {missing}"
        )

    dataset = dataset.copy()

    dataset["Decision Date"] = (
        pd.to_datetime(
            dataset["Decision Date"],
            errors="raise",
        )
        .astype("datetime64[ns]")
        .dt.normalize()
    )

    if not dataset["Candidate Key"].is_unique:
        raise ValueError(
            "Duplicate candidate keys."
        )

    if dataset["Decision Date"].isna().any():
        raise ValueError(
            "Missing decision dates."
        )

    market = calculate_market_features(close)

    market.index.name = "Decision Date"
    market = market.reset_index()

    enriched = dataset.merge(
        market,
        on="Decision Date",
        how="left",
        validate="many_to_one",
        sort=False,
    )

    if len(enriched) != len(dataset):
        raise ValueError(
            "Candidate count changed during merge."
        )

    if not enriched[
        "Candidate Key"
    ].is_unique:
        raise ValueError(
            "Duplicate keys after market feature merge."
        )

    missing_features = (
        enriched[MARKET_FEATURES]
        .isna()
        .sum()
    )

    if missing_features.any():
        raise ValueError(
            "Missing historical market features:\n"
            + missing_features.to_string()
        )

    if not np.isfinite(
        enriched[MARKET_FEATURES]
        .to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "Non-finite market features."
        )

    invalid_cutoff = (
        enriched["Feature_Cutoff_Date"].isna()
        | (
            enriched["Feature_Cutoff_Date"]
            >= enriched["Decision Date"]
        )
    )

    if invalid_cutoff.any():
        raise ValueError(
            "A market feature uses an invalid cutoff date."
        )

    return enriched.sort_values(
        ["Decision Date", "Candidate Key"]
    ).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ric",
        default="AAPL.O",
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
    )

    parser.add_argument(
        "--execute",
        action="store_true",
    )

    args = parser.parse_args()

    if not args.input.is_file():
        raise FileNotFoundError(
            args.input
        )

    dataset = pd.read_parquet(
        args.input
    )

    if dataset.empty:
        raise ValueError(
            "Empty candidate dataset."
        )

    if not dataset[
        "Underlying RIC"
    ].eq(args.ric).all():
        raise ValueError(
            "Candidate dataset contains another RIC."
        )

    close = load_equity_prices(
        args.ric
    )

    print("\n--- EQUITY HISTORY ---")
    print("RIC:", args.ric)
    print("Price observations:", len(close))
    print("First date:", close.index.min().date())
    print("Last date:", close.index.max().date())

    # Detect unusual jumps that may require
    # split/adjustment investigation.
    daily_returns = close.pct_change(
        fill_method=None
    )

    unusual = daily_returns.loc[
        daily_returns.abs() > 0.30
    ]

    print("\nDaily price changes above 30%:")
    print(len(unusual))

    if not unusual.empty:
        print(
            (unusual * 100)
            .round(2)
            .to_string()
        )

    enriched = enrich_dataset(
        dataset,
        close,
    )

    print("\n--- FEATURE VALIDATION ---")
    print("Candidate rows:", len(enriched))
    print(
        "Decision dates:",
        enriched["Decision Date"].nunique(),
    )

    print(
        "Market features:",
        len(MARKET_FEATURES),
    )

    print("\n--- FEATURE SUMMARY ---")
    print(
        enriched[MARKET_FEATURES]
        .describe()
        .round(4)
        .to_string()
    )

    # Separate audit of historical price alignment.
    audit = enriched[
        [
            "Decision Date",
            "Feature_Cutoff_Date",
            "Spot Price",
            "Previous_Close",
        ]
    ].drop_duplicates(
        subset=["Decision Date"]
    ).copy()

    audit["Spot_vs_Previous_Close_Pct"] = (
        (
            audit["Spot Price"]
            / audit["Previous_Close"]
        ) - 1
    ) * 100

    print("\n--- PRICE ALIGNMENT AUDIT ---")
    print(
        audit.round(3)
        .to_string(index=False)
    )

    if not args.execute:
        print(
            "\nDry run complete. No files created."
        )
        return

    output_dir = (
        OUTPUT_ROOT
        / args.input.parent.name
    )

    if output_dir.exists():
        raise FileExistsError(
            f"Output already exists: {output_dir}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    output_file = (
        output_dir
        / "completed_candidates_with_market_features.parquet"
    )

    enriched.to_parquet(
        output_file,
        index=False,
    )

    manifest = {
        "ric": args.ric,
        "source_dataset": str(args.input),
        "price_source": "load_price_history",
        "candidate_rows": int(len(enriched)),
        "decision_dates": int(
            enriched["Decision Date"].nunique()
        ),
        "market_features": MARKET_FEATURES,
        "feature_timing": (
            "All features use the previous "
            "trading session or earlier."
        ),
        "audit_columns": [
            "Feature_Cutoff_Date",
            "Previous_Close",
        ],
        "adjustment_convention": (
            "Unverified; review price audit."
        ),
    }

    with open(
        output_dir / "feature_manifest.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            manifest,
            file,
            indent=4,
        )

    print("\nEnriched dataset saved:")
    print(output_file)

    print("\nOriginal dataset unchanged.")


if __name__ == "__main__":
    main()
