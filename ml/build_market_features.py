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

# the data is stored under data/processed/ml_datasets/aapl_h21, 
# historical prices are requested for AAPL
ROOT = Path("data/processed/ml_datasets/aapl_h21")

# AAPL.O is LSEG's RIC identifier for AAPL
RIC = "AAPL.O"

# declares the eight market features (5 days, 21 days etc.)
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
    """
    converts LSEG's response into a clean pandas series
    """
    
    # if no historical price data is provided the script is caused
    # to stop
    if history is None or history.empty:
        raise ValueError(
            "LSEG returned no historical prices."
        )

    frame = history.copy()

    # searches for the column levels for AAPL.O and extracts that section
    if isinstance(frame.columns, pd.MultiIndex):
        for level in range(frame.columns.nlevels):
            if RIC in frame.columns.get_level_values(level):
                frame = frame.xs(
                    RIC,
                    axis=1,
                    level=level,
                )
                break

    # it looks for 'TRDPRC_1' (LSEG's historical trading-price field) 
    # if the tables has only 1 column, this one is chosen for the 
    # price series
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

    # converts prices to numbers and replaces invalid values with NaN
    prices = pd.to_numeric(
        series,
        errors="coerce",
    )

    # the index is converted to dates and normalize() removes times 
    prices.index = pd.to_datetime(
        prices.index
    ).normalize()

    # prices are stored chronologically 
    prices = prices.sort_index()
    prices.name = "Historical Price"

    # duplicate dates are rejected 
    if prices.index.has_duplicates:
        raise ValueError(
            "Duplicate dates returned within an LSEG request."
        )

    prices = prices.dropna().astype(float)

    # missing values or zero and negative prices are rejected
    if prices.empty or (prices <= 0).any():
        raise ValueError(
            "Invalid historical stock prices."
        )

    # the resulting series contains the date as its index and the AAPL price
    # as its values 
    return prices


def download_historical_prices():
    """
    downloads prices in overlapping yearly windows 
    """
    chunks = []

    print("\n--- LSEG HISTORICAL PRICE DOWNLOAD ---")

    # open the LSEG session and establishes the connection
    ld.open_session()

    try:
        for year in range(2014, 2025):
            # Fetch overlapping calendar windows from 2013-12-20 until
            # 2026-01-10
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

            # normalizes the result 
            prices = normalize_history(history)

            print(
                f"Retrieved {len(prices)} observations."
            )

            # the normalized results are added to chunks 
            chunks.append(prices)

    # ensures the session closes even if downloading fails 
    finally:
        ld.close_session()

    # concatenate all downloads Series and sort them by date 
    combined = pd.concat(chunks).sort_index()

    # Overlapping requests may contain the same date.
    # Their historical values must be consistent.
    duplicates = combined[
        combined.index.duplicated(keep=False)
    ]

    # checks for every duplicate date whether all downloaded price values
    # are approximately equal (np.allclose(...))
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

    # if overlapping values agree, the first copy of each date is retained 
    combined = combined[
        ~combined.index.duplicated(keep="first")
    ]

    if combined.index.has_duplicates:
        raise ValueError(
            "Duplicate historical dates remain."
        )

    # print the total observations
    print("\nTotal price observations:", len(combined))

    # print first available date 
    print(
        "First price date:",
        combined.index.min().date(),
    )

    # print last available date 
    print(
        "Last price date:",
        combined.index.max().date(),
    )

    # returns the combined price series 
    return combined


def calculate_market_features(prices):
    """
    calculates features using the historical prices 
    """
    # calculate daily returns from one trading date to the next
    daily_returns = prices.pct_change(
        fill_method=None
    )

    features = pd.DataFrame(
        index=prices.index
    )

    # compare today's price to the price from 5, 21, 63 observations
    # earlier 
    # 5-trading-day-return
    features["Return 5D"] = (
        prices / prices.shift(5) - 1
    )

    #21-trading-day-return
    features["Return 21D"] = (
        prices / prices.shift(21) - 1
    )

    # 63-trading-day-return 
    features["Return 63D"] = (
        prices / prices.shift(63) - 1
    )

    # calculate volatility (rolling standard deviation) over 21 observations
    # and multiplies by sqrt(252) to annualize the volatility 
    features["Volatility 21D"] = (
        daily_returns.rolling(
            window=21,
            min_periods=21,
        ).std() * np.sqrt(252)
    )

    # calculates volatility over 63 observations and multiplies by
    # sqrt(252) to annualize the volatility 
    features["Volatility 63D"] = (
        daily_returns.rolling(
            window=63,
            min_periods=63,
        ).std() * np.sqrt(252)
    )

    # finds the highest price in the previous 63 observations 
    rolling_high = prices.rolling(
        window=63,
        min_periods=63,
    ).max()

    # calculates the drawdown 
    # 0.00 means the price is at its recent maximum
    # -0.10 means the price is 10% below its recent maximum
    # -0.25 means the price is 25% below its recent maximum  
    features["Drawdown 63D"] = (
        prices / rolling_high - 1
    )

    # calculates the simple moving averages over 21 days
    sma_21 = prices.rolling(
        window=21,
        min_periods=21,
    ).mean()

    # calculates the simple moving averages over 63 days 
    sma_63 = prices.rolling(
        window=63,
        min_periods=63,
    ).mean()

    # express the current price relative to each average and adds the 
    # value to the features-DataFrame 
    # positive means the price is above the average
    # negative means the price is below the average 
    features["Distance SMA 21"] = (
        prices / sma_21 - 1
    )

    features["Distance SMA 63"] = (
        prices / sma_63 - 1
    )

    # returns the market DataFrame 
    return features[MARKET_COLUMNS]


def validate_split_adjustments(prices):
    """chekcs whether AAPL prices around the 2020 split are 
    adjusted correctly"""
    print("\n--- CORPORATE ACTION CHECK ---")

    # looks at 2020-08-24 through 2020-09-04
    # because a raw AAPL 4-for-1 split makes an unadjusted price 
    # appear to drop sharply
    window = prices.loc[
        "2020-08-24":"2020-09-04"
    ]

    print(window.to_string())
    # raises an error if the pre-split price is missing
    if pd.Timestamp("2020-08-28") not in window.index:
        raise ValueError(
            "Missing pre-split price on 2020-08-28."
        )

    # raises an error if the post-split price is missing 
    if pd.Timestamp("2020-08-31") not in window.index:
        raise ValueError(
            "Missing post-split price on 2020-08-31."
        )

    # before is the price at 2020-08-28
    before = float(
        window.loc["2020-08-28"]
    )

    # after is the price at 2020-08-31
    after = float(
        window.loc["2020-08-31"]
    )

    # calculates the absolute change over the split
    split_window_return = (
        after / before - 1
    )

    # prints the absolute change in percent 
    print(
        "\n2020-08-28 to 2020-08-31 return:",
        f"{split_window_return:.2%}",
    )

    # A raw 4-for-1 split would create an apparent
    # decline of approximately 75%.
    # if the absolute change exceeds 50%, it raises an error 
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
    """ensures that every candidate decision date has complete 
    historical features"""
    print("\n--- DECISION DATE VALIDATION ---")

    # converts the supplied dates into a normalized DatetimeIndex
    dates = pd.DatetimeIndex(
        pd.to_datetime(decision_dates)
    ).normalize()

    # if a decision date is absent from the price date, the script 
    # prints it and stops
    missing = dates.difference(prices.index)

    if len(missing) > 0:
        print("Missing decision dates:")
        for date in missing:
            print(date.date())

        raise ValueError(
            "Historical price data is missing "
            "required decision dates."
        )

    # creates a table with one row per unique decision date 
    # and one column per market feature 
    selected = market_features.loc[dates].copy()

    selected.index.name = "Decision Date"

    # checks for NaN values (fewer than 5, 21 or 63 observations 
    # or fewer than enough rolling data), reports the affected 
    # dates and raises an error 
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

    # rejects values that are
    # infinite, undefined or otherwise not finite numbers 
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

    # returns a validated table indexed by decision date 
    return selected


def build_augmented_dataset(
    selected_features,
    metadata,
    original_features,
):
    """joins the date-level features onto the original candidate table """
    
    # normalize candidate decision dates 
    # ensures the join keys match the market-feature index 
    dates = pd.to_datetime(
        metadata["Decision Date"]
    ).dt.normalize()

    # Reindex to candidate order.
    # Multiple candidates on the same date
    # receive identical market-state features.
    market_rows = selected_features.reindex(
        pd.DatetimeIndex(dates)
    )

    # it removes the date inde and creates a row index matching the
    # original candidate table 
    market_rows = market_rows.reset_index(
        drop=True
    )

    # verifies if the market table have exactly as many rows as the original
    # feature table 
    if len(market_rows) != len(original_features):
        raise ValueError(
            "Feature row counts do not match."
        )

    # prevents silently replacing or duplicating existing columns
    if set(MARKET_COLUMNS).intersection(
        original_features.columns
    ):
        raise ValueError(
            "Market features already exist "
            "in the original feature table."
        )

    # this horizonally joins the original cnadidate features and
    # newly calculated market features 
    augmented = pd.concat(
        [
            original_features.reset_index(drop=True),
            market_rows,
        ],
        axis=1,
    )

    # checks for duplicate column names and missing market values 
    if augmented.columns.duplicated().any():
        raise ValueError(
            "Duplicate feature columns found."
        )

    if augmented[MARKET_COLUMNS].isna().any().any():
        raise ValueError(
            "Market-feature join introduced missing data."
        )

    # returns the augmented DataFrame
    return augmented


def main():
    """executes the full pipeline 
    """
    
    # reads .parquet files   
    metadata = pd.read_parquet(
        ROOT / "metadata.parquet"
    )

    original_features = pd.read_parquet(
        ROOT / "features.parquet"
    )

    # requires both tables to contain the same number of rows
    if len(metadata) != len(original_features):
        raise ValueError(
            "Metadata and features have different row counts."
        )

    # normalize the decision date 
    dates = pd.to_datetime(
        metadata["Decision Date"]
    ).dt.normalize()

    # duplicate candidate dates are removed because one set of market 
    # features is needed per date 
    decision_dates = pd.DatetimeIndex(
        dates.unique()
    ).sort_values()

    # prints candidate count, unique decision date count and
    # number of original feature columns 
    print("\n--- DATASET ---")
    print("Candidates:", len(metadata))
    print("Decision dates:", len(decision_dates))
    print(
        "Existing features:",
        original_features.shape[1],
    )

    # download historical prices
    prices = download_historical_prices()

    # checks for the 2020 stock split
    validate_split_adjustments(prices)

    # calculate market features 
    historical_features = calculate_market_features(
        prices
    )

    # validate each decision date 
    selected = validate_decision_dates(
        prices,
        historical_features,
        decision_dates,
    )

    # market features are attached to the original rows 
    augmented = build_augmented_dataset(
        selected,
        metadata,
        original_features,
    )

    # print a summary
    print("\n--- FINAL FEATURE DATASET ---")

    # final candidate row count 
    print(
        "Candidate rows:",
        len(augmented),
    )

    # number of original features
    print(
        "Original features:",
        original_features.shape[1],
    )

    # number of newly added features 
    print(
        "New market features:",
        len(MARKET_COLUMNS),
    )

    # total feature count 
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

    # print the saved files
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
