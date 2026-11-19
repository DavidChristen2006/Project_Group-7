
import numpy as np
import pandas as pd


AAPL_SPLITS = (
    (pd.Timestamp("2020-08-31"), 4.0),
)

SUPPORTED_START = pd.Timestamp("2014-06-09")


def get_aapl_historical_split_factor(date):
    """
    Convert a current-split-adjusted AAPL stock price
    to its historical trading-price scale.

    Supported from 2014-06-09 onward.

    Assumes the input series is already adjusted for
    Apple's 2014 and 2020 stock splits.
    """
    date = pd.Timestamp(str(date)[:10]).normalize()

    if date < SUPPORTED_START:
        raise ValueError(
            "Dates before 2014-06-09 are not supported."
        )

    factor = 1.0

    for split_date, split_ratio in AAPL_SPLITS:
        if date < split_date:
            factor *= split_ratio

    return factor


def normalize_historical_stock_prices(
    prices,
    underlying_ric,
):
    """
    Convert split-adjusted stock prices to the
    historical trading-price scale.

    Parameters
    ----------
    prices : pandas.Series
        Series indexed by trading dates.

    underlying_ric : str
        LSEG underlying RIC.

    Returns
    -------
    pandas.Series
        Converted prices on the historical trading
        share basis.
    """
    if underlying_ric != "AAPL.O":
        raise NotImplementedError(
            "Historical split normalization is "
            "currently implemented only for AAPL.O."
        )

    if not isinstance(prices, pd.Series):
        raise TypeError(
            "prices must be a pandas Series."
        )

    result = prices.copy()

    result.index = pd.DatetimeIndex(
        pd.to_datetime(result.index)
    ).normalize()

    result = pd.to_numeric(
        result,
        errors="raise",
    ).astype(float)

    if result.empty:
        return result

    if (
        not np.isfinite(result.to_numpy()).all()
        or (result <= 0).any()
    ):
        raise ValueError(
            "Historical prices must be finite "
            "and positive."
        )

    factors = pd.Series(
        [
            get_aapl_historical_split_factor(date)
            for date in result.index
        ],
        index=result.index,
        dtype=float,
    )

    return result * factors
