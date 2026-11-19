"""Build historical market-state features for AAPL hedging ML.

Run from project root:
    py -m ml.build_market_features

Required packages:
    lseg-data, numpy, pandas, pyarrow
"""

from pathlib import Path

import lseg.data as ld
import numpy as np
import pandas as pd


ROOT = Path("data/processed/ml_datasets/aapl_h21")

RIC = "AAPL.O"

MARKET_COLUMNS = [
    "Return 5D",
    "Return 21D",
    "Return 63D",
    "Volatility 21D",
    "Volatility 63D",
    "Drawdown 63D",
    "Distance SMA 21",
    "Distance SMA 63",
]


def normalize_history(history):
    if history is None or history.empty:
        raise ValueError(
            "LSEG returned no historical prices."
        )

    frame = history.copy()

    if isinstance(frame.columns, pd.MultiIndex):
        for level in range(frame.columns.nlevels):
            if RIC in frame.columns.get_level_values(level):
                frame = frame.xs(
                    RIC,
                    axis=1,
                    level=level,
                )
                break

    if "TRDPRC_1" in frame.columns:
        series = frame["TRDPRC_1"]
    elif frame.shape[1] == 1:
        series = frame.iloc[:, 0]
    else:
        raise ValueError(
            "Cannot identify the historical price field: "
            f"{frame.columns.tolist()}"
        )

    if isinstance(series, pd.DataFrame):
        if series.shape[1] != 1:
            raise ValueError(
                "Multiple historical price columns found."
            )
        series = series.iloc[:, 0]

    prices = pd.to_numeric(
        series,
        errors="coerce",
    )

    prices.index = pd.to_datetime(
        prices.index
    ).normalize()

    prices = prices.sort_index()
    prices.name = "Historical Price"

    if prices.index.has_duplicates:
        raise ValueError(
            "Duplicate dates returned within an LSEG request."
        )

    prices = prices.dropna().astype(float)

    if prices.empty or (prices <= 0).any():
        raise ValueError(
            "Invalid historical stock prices."
        )

    return prices


def download_historical_prices():
    chunks = []

    print("\n--- LSEG HISTORICAL PRICE DOWNLOAD ---")

    ld.open_session()

    try:
        for year in range(2014, 2025):
            # Fetch overlapping calendar windows.
            # This also helps with start-date behavior.
            start = f"{year - 1}-12-20"
            end = f"{year + 1}-01-10"

            print(
                f"Downloading {year}: {start} to {end}"
            )

            history = ld.get_history(
                universe=RIC,
                fields=["TRDPRC_1"],
                interval="daily",
                start=start,
                end=end,
            )

            prices = normalize_history(history)

            print(
                f"Retrieved {len(prices)} observations."
            )

            chunks.append(prices)

    finally:
        ld.close_session()

    combined = pd.concat(chunks).sort_index()

    # Overlapping requests may contain the same date.
    # Their historical values must be consistent.
    duplicates = combined[
        combined.index.duplicated(keep=False)
    ]

    for date, values in duplicates.groupby(level=0):
        arr = values.to_numpy(dtype=float)

        if not np.allclose(
            arr,
            arr[0],
            rtol=1e-8,
            atol=1e-6,
        ):
            raise ValueError(
                "Inconsistent overlapping LSEG prices "
                f"on {date.date()}: {arr}"
            )

    combined = combined[
        ~combined.index.duplicated(keep="first")
    ]

    if combined.index.has_duplicates:
        raise ValueError(
            "Duplicate historical dates remain."
        )

    print("\nTotal price observations:", len(combined))

    print(
        "First price date:",
        combined.index.min().date(),
    )

    print(
        "Last price date:",
        combined.index.max().date(),
    )

    return combined


def calculate_market_features(prices):
    daily_returns = prices.pct_change(
        fill_method=None
    )

    features = pd.DataFrame(
        index=prices.index
    )

    features["Return 5D"] = (
        prices / prices.shift(5) - 1
    )

    features["Return 21D"] = (
        prices / prices.shift(21) - 1
    )

    features["Return 63D"] = (
        prices / prices.shift(63) - 1
    )

    features["Volatility 21D"] = (
        daily_returns.rolling(
            window=21,
            min_periods=21,
        ).std() * np.sqrt(252)
    )

    features["Volatility 63D"] = (
        daily_returns.rolling(
            window=63,
            min_periods=63,
        ).std() * np.sqrt(252)
    )

    rolling_high = prices.rolling(
        window=63,
        min_periods=63,
    ).max()

    features["Drawdown 63D"] = (
        prices / rolling_high - 1
    )

    sma_21 = prices.rolling(
        window=21,
        min_periods=21,
    ).mean()

    sma_63 = prices.rolling(
        window=63,
        min_periods=63,
    ).mean()

    features["Distance SMA 21"] = (
        prices / sma_21 - 1
    )

    features["Distance SMA 63"] = (
        prices / sma_63 - 1
    )

    return features[MARKET_COLUMNS]


def validate_split_adjustments(prices):
    print("\n--- CORPORATE ACTION CHECK ---")

    window = prices.loc[
        "2020-08-24":"2020-09-04"
    ]

    print(window.to_string())

    if pd.Timestamp("2020-08-28") not in window.index:
        raise ValueError(
            "Missing pre-split price on 2020-08-28."
        )

    if pd.Timestamp("2020-08-31") not in window.index:
        raise ValueError(
            "Missing post-split price on 2020-08-31."
        )

    before = float(
        window.loc["2020-08-28"]
    )

    after = float(
        window.loc["2020-08-31"]
    )

    split_window_return = (
        after / before - 1
    )

    print(
        "\n2020-08-28 to 2020-08-31 return:",
        f"{split_window_return:.2%}",
    )

    # A raw 4-for-1 split would create an apparent
    # decline of approximately 75%.
    if abs(split_window_return) > 0.50:
        raise ValueError(
            "Possible unadjusted corporate action "
            "in historical stock prices."
        )

    print("2020 split continuity: PASS")


def validate_decision_dates(
    prices,
    market_features,
    decision_dates,
):
    print("\n--- DECISION DATE VALIDATION ---")

    dates = pd.DatetimeIndex(
        pd.to_datetime(decision_dates)
    ).normalize()

    missing = dates.difference(prices.index)

    if len(missing) > 0:
        print("Missing decision dates:")
        for date in missing:
            print(date.date())

        raise ValueError(
            "Historical price data is missing "
            "required decision dates."
        )

    selected = market_features.loc[dates].copy()

    selected.index.name = "Decision Date"

    if selected.isna().any().any():
        invalid = selected[
            selected.isna().any(axis=1)
        ]

        print("\nInsufficient history:")
        print(invalid.to_string())

        raise ValueError(
            "Some decision dates have missing "
            "market-state features."
        )

    if not np.isfinite(
        selected.to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "Nonfinite market-state features found."
        )

    print(
        "Validated decision dates:",
        len(selected),
    )

    print("Feature columns:", len(selected.columns))

    print("Decision date validation: PASS")

    return selected


def build_augmented_dataset(
    selected_features,
    metadata,
    original_features,
):
    dates = pd.to_datetime(
        metadata["Decision Date"]
    ).dt.normalize()

    # Reindex to candidate order.
    # Multiple candidates on the same date
    # receive identical market-state features.
    market_rows = selected_features.reindex(
        pd.DatetimeIndex(dates)
    )

    market_rows = market_rows.reset_index(
        drop=True
    )

    if len(market_rows) != len(original_features):
        raise ValueError(
            "Feature row counts do not match."
        )

    if set(MARKET_COLUMNS).intersection(
        original_features.columns
    ):
        raise ValueError(
            "Market features already exist "
            "in the original feature table."
        )

    augmented = pd.concat(
        [
            original_features.reset_index(drop=True),
            market_rows,
        ],
        axis=1,
    )

    if augmented.columns.duplicated().any():
        raise ValueError(
            "Duplicate feature columns found."
        )

    if augmented[MARKET_COLUMNS].isna().any().any():
        raise ValueError(
            "Market-feature join introduced missing data."
        )

    return augmented


def main():
    metadata = pd.read_parquet(
        ROOT / "metadata.parquet"
    )

    original_features = pd.read_parquet(
        ROOT / "features.parquet"
    )

    if len(metadata) != len(original_features):
        raise ValueError(
            "Metadata and features have different row counts."
        )

    dates = pd.to_datetime(
        metadata["Decision Date"]
    ).dt.normalize()

    decision_dates = pd.DatetimeIndex(
        dates.unique()
    ).sort_values()

    print("\n--- DATASET ---")
    print("Candidates:", len(metadata))
    print("Decision dates:", len(decision_dates))
    print(
        "Existing features:",
        original_features.shape[1],
    )

    prices = download_historical_prices()

    validate_split_adjustments(prices)

    historical_features = calculate_market_features(
        prices
    )

    selected = validate_decision_dates(
        prices,
        historical_features,
        decision_dates,
    )

    augmented = build_augmented_dataset(
        selected,
        metadata,
        original_features,
    )

    print("\n--- FINAL FEATURE DATASET ---")

    print(
        "Candidate rows:",
        len(augmented),
    )

    print(
        "Original features:",
        original_features.shape[1],
    )

    print(
        "New market features:",
        len(MARKET_COLUMNS),
    )

    print(
        "Total features:",
        augmented.shape[1],
    )

    print("\n--- MARKET FEATURE SUMMARY ---")

    print(
        selected.describe().to_string()
    )

    # Save only after all checks have passed.
    prices.to_frame().to_parquet(
        ROOT / "historical_market_prices.parquet"
    )

    selected.to_parquet(
        ROOT / "market_features_by_date.parquet"
    )

    augmented.to_parquet(
        ROOT / "features_with_market.parquet",
        index=False,
    )

    print("\n--- FILES SAVED ---")

    print(
        ROOT / "historical_market_prices.parquet"
    )

    print(
        ROOT / "market_features_by_date.parquet"
    )

    print(
        ROOT / "features_with_market.parquet"
    )

    print("\nMarket-feature preparation complete.")


if __name__ == "__main__":
    main()
