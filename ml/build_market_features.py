"""Build historical market-state features for AAPL hedging ML.

Downloads daily AAPL prices from LSEG, computes eight market-state
features per decision date (returns, volatility, drawdown, distance to
moving averages), validates them and appends them to the candidate
feature table. All features use only prices up to the decision date,
so no future information leaks into the model.

Run from project root:
    py -m ml.build_market_features

Reads:  metadata.parquet, features.parquet (in ROOT)
Writes: historical_market_prices.parquet, market_features_by_date.parquet,
        features_with_market.parquet (in ROOT)

Required packages:
    lseg-data, numpy, pandas, pyarrow

GenAI usage: Claude (Anthropic) was used to explain the code and to
draft, review and correct comments and docstrings. The comments were
checked and adapted by the author(s).
"""

from pathlib import Path

import lseg.data as ld
import numpy as np
import pandas as pd

# the data is stored under data/processed/ml_datasets/aapl_h21, 
ROOT = Path("data/processed/ml_datasets/aapl_h21")

# historical prices are requested for AAPL
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
    """Convert an LSEG history response into a clean price series.

    Selects the AAPL.O price field (TRDPRC_1, or the only column),
    converts values to numbers, normalizes the dates, sorts them and
    drops missing values.

    Args:
        history: DataFrame returned by ld.get_history(). Its columns may
            be a MultiIndex containing the RIC.

    Returns:
        pandas Series named "Historical Price" with a sorted,
        duplicate-free DatetimeIndex (no time of day) and float values.

    Raises:
        ValueError: if the response is empty, the price column cannot be
            identified, dates are duplicated, or no valid positive
            prices remain.
    """
    
    # raises an error if LSEG returned no data
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
    # if the table has only 1 column, this one is chosen for the 
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

    # missing values are dropped
    prices = prices.dropna().astype(float)

    # an empty series or 
    # prices ≤ 0 raise an error
    if prices.empty or (prices <= 0).any():
        raise ValueError(
            "Invalid historical stock prices."
        )

    # the resulting series contains the date as its index and the AAPL price
    # as its values 
    return prices


def download_historical_prices():
    """Download daily AAPL prices from LSEG in overlapping yearly windows.

    For each year from 2014 to 2024, one window from 20 December of the
    previous year to 10 January of the next year is requested, so the
    data covers 2013-12-20 to 2025-01-10. Days returned by two windows
    must have consistent prices (checked with np.allclose); the first
    copy is kept.

    Returns:
        pandas Series of daily prices with a sorted, duplicate-free
        DatetimeIndex, as produced by normalize_history().

    Raises:
        ValueError: if overlapping windows contain inconsistent prices
            or duplicate dates remain.

    Side effects:
        Opens and closes an LSEG session (also if the download fails)
        and prints progress and a short summary.
    """
    chunks = []

    print("\n--- LSEG HISTORICAL PRICE DOWNLOAD ---")

    # open the LSEG session and establishes the connection
    ld.open_session()

    try:
        for year in range(2014, 2025):
            # Fetch overlapping calendar windows from 2013-12-20 until
            # 2025-01-10
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
    """Compute the eight market-state features for every trading day.

    Features: 5/21/63-day returns, 21/63-day volatility (standard
    deviation of daily returns, annualized with sqrt(252)), drawdown
    relative to the 63-day high (including today) and the distance to
    the 21/63-day simple moving averages. Only past and current prices
    are used, so there is no look-ahead. The first days lack enough
    history and contain NaN.

    Args:
        prices: pandas Series of daily prices with a sorted
            DatetimeIndex.

    Returns:
        DataFrame indexed by date with the columns of MARKET_COLUMNS.
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

    # 21-trading-day-return
    features["Return 21D"] = (
        prices / prices.shift(21) - 1
    )

    # 63-trading-day-return 
    features["Return 63D"] = (
        prices / prices.shift(63) - 1
    )

    # calculate volatility (rolling standard deviation) over 21 observations
    # of the daily returns and multiplies by sqrt(252) to annualize 
    # the volatility 
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

    # highest price of the last 63 observations, including today 
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
    """Check that AAPL prices contain no unadjusted jump at the 2020 split.

    Apple split its stock 4-for-1, with 2020-08-31 as the first
    split-adjusted trading day. Unadjusted prices would show an apparent
    drop of about 75% between 2020-08-28 and 2020-08-31. The function
    compares these two prices and fails if the return exceeds 50% in
    either direction. This is a plausibility check, not proof that the
    adjustment is exact.

    Args:
        prices: pandas Series of daily AAPL prices with a sorted
            DatetimeIndex.

    Raises:
        ValueError: if the price on 2020-08-28 or 2020-08-31 is missing,
            or if the return between them suggests an unadjusted split.

    Side effects:
        Prints the price window and the split-window return.
    """
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

    # calculates the relative change (return) over the split
    split_window_return = (
        after / before - 1
    )

    # prints the return in percent 
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
    """Ensure every decision date has a complete set of valid features.

    Args:
        prices: pandas Series of daily prices (DatetimeIndex).
        market_features: DataFrame of features indexed by date, as
            returned by calculate_market_features().
        decision_dates: Iterable of unique decision dates.

    Returns:
        DataFrame with one row per decision date (index named
        "Decision Date") and one column per market feature.

    Raises:
        ValueError: if a decision date is missing in the price data, a
            feature is NaN (too little history), or a feature is not a
            finite number.

    Side effects:
        Prints the affected dates on failure and a validation summary.
    """
    print("\n--- DECISION DATE VALIDATION ---")

    # converts the supplied dates into a normalized DatetimeIndex
    dates = pd.DatetimeIndex(
        pd.to_datetime(decision_dates)
    ).normalize()

    # if a decision date is absent from the price dates, the script 
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

    # checks for NaN values (NaN appears when fewer 
    # than 5/21/63 prior prices exist), reports the affected 
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
    """Attach the date-level market features to each candidate row.

    Candidates sharing a decision date receive identical market
    features. The join is positional: row i of metadata must belong to
    row i of original_features.

    Args:
        selected_features: DataFrame of market features indexed by
            decision date.
        metadata: DataFrame with a "Decision Date" column, one row per
            candidate.
        original_features: DataFrame of the existing candidate
            features, in the same row order as metadata.

    Returns:
        DataFrame with the original feature columns followed by the
        market feature columns.

    Raises:
        ValueError: if row counts differ, market columns already exist,
            column names are duplicated, or the join leaves missing
            values.
    """
    
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

    # it removes the index and creates a row index matching the
    # original candidate table 
    market_rows = market_rows.reset_index(
        drop=True
    )

    # verifies if the market table has exactly as many rows as the original
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

    # this horizontally joins the original candidate features and
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
    """Run the full pipeline and save the augmented feature dataset.

    Loads the candidate tables, downloads and validates AAPL prices,
    computes and validates the market features for all decision dates,
    appends them to the candidate features and prints a summary. Files
    are written only after all checks have passed.

    Raises:
        ValueError: if metadata and features have different row counts
            or any validation step fails.

    Side effects:
        Writes historical_market_prices.parquet,
        market_features_by_date.parquet and
        features_with_market.parquet to ROOT.
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
