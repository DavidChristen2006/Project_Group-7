
"""Shared historical basis calibration for the hedge overview.

Reuses the existing LSEG adjusted-close retrieval and
horizon-matched historical basis calibration.

Does not change or re-estimate the 252-session beta.
"""

from __future__ import annotations

from datetime import date, timedelta
from math import isfinite

import pandas as pd


def make_calibration_key(
    hedge_result: dict,
    cross_result: dict,
    *,
    as_of: date,
    lookback_years: int = 8,
) -> tuple:
    """Create the same cache key as the detailed calibration."""

    if (
        isinstance(lookback_years, bool)
        or not isinstance(lookback_years, int)
        or not 3 <= lookback_years <= 12
    ):
        raise ValueError(
            "Historical lookback must be between 3 and 12 years"
        )

    if not isinstance(as_of, date):
        raise ValueError(
            "as_of must be a date"
        )

    inputs = hedge_result["inputs"]
    meta = hedge_result["meta"]

    if (
        cross_result.get("snapshot")
        != meta.get("retrieved_at_new_york")
    ):
        raise ValueError(
            "ETF result belongs to another stock snapshot"
        )

    stock_ric = str(inputs.ric).strip()

    etf_ric = str(
        cross_result["etf_ric"]
    ).strip()

    horizon = int(
        meta["research_horizon_sessions"]
    )

    beta = float(
        cross_result["beta"]
    )

    r_squared = float(
        cross_result["beta_report"]["r_squared"]
    )

    if (
        not stock_ric
        or not etf_ric
        or horizon <= 0
    ):
        raise ValueError(
            "Missing instrument or invalid hedge horizon"
        )

    if (
        not isfinite(beta)
        or abs(beta) <= 1e-8
    ):
        raise ValueError(
            "A finite nonzero stock-on-ETF beta is required"
        )

    if (
        not isfinite(r_squared)
        or not 0 <= r_squared <= 1
    ):
        raise ValueError(
            "R-squared must be between 0 and 1"
        )

    return (
        stock_ric,
        etf_ric,
        horizon,
        lookback_years,
        as_of.isoformat(),
        beta,
        r_squared,
    )


def make_calibration_cache(
    hedge_result: dict,
    cross_result: dict,
    *,
    as_of: date,
    stock_prices: pd.Series,
    etf_prices: pd.Series,
    lookback_years: int = 8,
) -> dict:
    """Build a cache compatible with all existing research panels."""

    from data_pipeline.historical_basis_calibration import (
        calibrate_basis_residuals,
    )

    key = make_calibration_key(
        hedge_result,
        cross_result,
        as_of=as_of,
        lookback_years=lookback_years,
    )

    calibration, windows = calibrate_basis_residuals(
        stock_prices,
        etf_prices,
        horizon_sessions=key[2],
        beta_stock_on_etf=key[5],
        r_squared=key[6],
        as_of=as_of,
    )

    return {
        "key": key,
        "calibration": calibration,
        "windows": windows,
        "stock_prices": stock_prices,
        "etf_prices": etf_prices,
    }


def fetch_quick_calibration(
    hedge_result: dict,
    cross_result: dict,
    *,
    lookback_years: int = 8,
) -> dict:
    """Retrieve paired prices and build a reusable historical calibration.

    Args:
        hedge_result: Validated stock snapshot with date and position.
        cross_result: Matching ETF analysis containing beta and R-squared.
        lookback_years: Number of years of daily adjusted-close history.

    Returns:
        A cache with calibration metrics, horizon windows, paired prices
        and a deterministic research-identity key.

    Notes:
        Called automatically by the one-button calculator; no trading
        action is placed. Uses and closes the default LSEG Platform Session.
    """

    import lseg.data as ld

    from data_pipeline.estimate_etf_beta import (
        last_completed_nyse_session,
    )

    from data_pipeline.historical_basis_calibration import (
        retrieve_adjusted_close_lseg,
    )

    as_of = pd.Timestamp(
        last_completed_nyse_session()
    ).date()

    key = make_calibration_key(
        hedge_result,
        cross_result,
        as_of=as_of,
        lookback_years=lookback_years,
    )

    start = as_of - timedelta(
        days=round(
            lookback_years * 365.25
        ) + 10
    )

    ld.open_session()

    try:
        stock_prices = retrieve_adjusted_close_lseg(
            key[0],
            start,
            as_of,
        )

        etf_prices = retrieve_adjusted_close_lseg(
            key[1],
            start,
            as_of,
        )

    finally:
        ld.close_session()

    return make_calibration_cache(
        hedge_result,
        cross_result,
        as_of=as_of,
        stock_prices=stock_prices,
        etf_prices=etf_prices,
        lookback_years=lookback_years,
    )
