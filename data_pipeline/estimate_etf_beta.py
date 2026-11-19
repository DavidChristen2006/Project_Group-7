
"""Estimate ETF cross-hedge beta from aligned historical LSEG daily prices.

Uses the default LSEG session via ld.open_session().
Missing historical prices are never forward-filled.
Research-only: not an executable hedge recommendation.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from data_pipeline.expanded_hedge_strategies import (
    etf_cross_hedge_stresses,
)


ROOT = Path(__file__).resolve().parents[1]

OUTPUT = (
    ROOT
    / "data/processed/hedge_comparisons/etf_beta_v1"
)

LOOKBACK = 252
MIN_RETURNS = 220
MAX_MISSING_PRICES = 8

ADJUSTMENTS = [
    "exchangeCorrection",
    "manualCorrection",
    "CCH",
    "CRE",
    "RPO",
    "RTS",
]


def last_completed_nyse_session(
    now: datetime | None = None,
) -> date:
    """Find the most recent completed US equity-market session."""

    import pandas_market_calendars as mcal

    if now is None:
        now = datetime.now(
            ZoneInfo("America/New_York")
        )

    if now.tzinfo is None:
        raise ValueError(
            "A timezone-aware datetime is required"
        )

    cal = mcal.get_calendar("NYSE")

    schedule = cal.schedule(
        start_date=(now - timedelta(days=20)).date(),
        end_date=now.date(),
    )

    now_utc = pd.Timestamp(
        now.astimezone(timezone.utc)
    )

    completed = schedule.loc[
        schedule["market_close"] <= now_utc
    ]

    if completed.empty:
        raise ValueError(
            "Cannot identify a completed NYSE session"
        )

    return completed.index[-1].date()


def nyse_lookback_sessions(
    as_of: date,
    sessions: int = LOOKBACK,
) -> pd.DatetimeIndex:
    """Return the previous sessions plus an entry price date."""

    import pandas_market_calendars as mcal

    if sessions < 30:
        raise ValueError(
            "At least 30 sessions are required"
        )

    cal = mcal.get_calendar("NYSE")

    valid = cal.valid_days(
        start_date=(
            as_of - timedelta(days=sessions * 3)
        ),
        end_date=as_of,
    ).tz_localize(None).normalize()

    if len(valid) < sessions + 1:
        raise ValueError(
            "Not enough NYSE sessions to estimate beta"
        )

    if valid[-1].date() != as_of:
        raise ValueError(
            f"{as_of} is not an NYSE trading session"
        )

    return valid[-(sessions + 1):]


def normalize_lseg_history(
    raw: pd.DataFrame,
    name: str,
) -> pd.Series:
    """Normalize one instrument's daily TRDPRC_1 series."""

    if (
        not isinstance(raw, pd.DataFrame)
        or raw.empty
    ):
        raise ValueError(
            f"{name}: LSEG returned no history"
        )

    if isinstance(raw.index, pd.MultiIndex):
        raise ValueError(
            f"{name}: unexpected MultiIndex history; "
            "request one RIC"
        )

    if "TRDPRC_1" not in raw.columns:
        raise ValueError(
            f"{name}: expected TRDPRC_1, "
            f"found {list(raw.columns)!r}"
        )

    dates = pd.DatetimeIndex(
        pd.to_datetime(
            raw.index,
            errors="raise",
        )
    )

    if dates.tz is not None:
        dates = dates.tz_localize(None)

    dates = dates.normalize()

    if dates.has_duplicates:
        raise ValueError(
            f"{name}: duplicate daily price dates"
        )

    values = pd.to_numeric(
        raw["TRDPRC_1"],
        errors="coerce",
    ).to_numpy(dtype=float)

    if (
        not np.isfinite(values).all()
        or np.any(values <= 0)
    ):
        raise ValueError(
            f"{name}: invalid or nonpositive daily prices"
        )

    return pd.Series(
        values,
        index=dates,
        name=name,
        dtype=float,
    ).sort_index()


def estimate_beta(
    stock_close: pd.Series,
    etf_close: pd.Series,
    session_dates: pd.DatetimeIndex,
    *,
    min_returns: int = MIN_RETURNS,
    max_missing_prices: int = MAX_MISSING_PRICES,
) -> dict:
    """Estimate historical stock beta relative to one ETF.

    Returns are computed AFTER both series are reindexed
    to the same market calendar.

    Missing observations are never filled.
    """

    dates = pd.DatetimeIndex(
        session_dates
    ).normalize()

    if (
        len(dates) < 31
        or dates.has_duplicates
        or not dates.is_monotonic_increasing
    ):
        raise ValueError(
            "Invalid ordered exchange-session dates"
        )

    stock = stock_close.reindex(dates)
    etf = etf_close.reindex(dates)

    def validate(
        series: pd.Series,
        label: str,
    ) -> int:

        numeric = pd.to_numeric(
            series,
            errors="coerce",
        )

        valid = numeric.notna()

        if (
            (numeric[valid] <= 0).any()
            or not np.isfinite(
                numeric[valid]
            ).all()
        ):
            raise ValueError(
                f"{label}: bad prices in selected window"
            )

        missing = int(
            (~valid).sum()
        )

        if missing > max_missing_prices:
            raise ValueError(
                f"{label}: {missing} missing sessions; "
                f"limit {max_missing_prices}"
            )

        if not bool(valid.iloc[-1]):
            raise ValueError(
                f"{label}: latest completed "
                "session price missing"
            )

        return missing

    stock_missing = validate(
        stock,
        "stock",
    )

    etf_missing = validate(
        etf,
        "ETF",
    )

    returns = pd.concat(
        [
            stock.pct_change(
                fill_method=None
            ).rename("stock"),

            etf.pct_change(
                fill_method=None
            ).rename("etf"),
        ],
        axis=1,
    ).dropna()

    if len(returns) < min_returns:
        raise ValueError(
            f"Only {len(returns)} aligned daily returns; "
            f"require {min_returns}"
        )

    x = returns["etf"].to_numpy(
        dtype=float
    )

    y = returns["stock"].to_numpy(
        dtype=float
    )

    etf_variance = float(
        np.var(x, ddof=1)
    )

    if etf_variance <= 1e-12:
        raise ValueError(
            "ETF return variance too small"
        )

    beta = float(
        np.cov(
            y,
            x,
            ddof=1,
        )[0, 1] / etf_variance
    )

    correlation = float(
        np.corrcoef(y, x)[0, 1]
    )

    if not (
        np.isfinite(beta)
        and np.isfinite(correlation)
    ):
        raise ValueError(
            "Invalid beta or correlation"
        )

    alpha = float(
        y.mean() - beta * x.mean()
    )

    residuals = (
        y - alpha - beta * x
    )

    residual_vol = float(
        np.std(
            residuals,
            ddof=1,
        ) * np.sqrt(252)
    )

    return {
        "as_of_session": str(
            dates[-1].date()
        ),

        "lookback_sessions": len(dates) - 1,

        "aligned_returns": len(returns),

        "stock_missing_price_sessions": (
            stock_missing
        ),

        "etf_missing_price_sessions": (
            etf_missing
        ),

        "beta_stock_on_etf": beta,

        "correlation": correlation,

        "r_squared": correlation**2,

        "annualized_residual_vol": residual_vol,

        "first_aligned_return": str(
            returns.index[0].date()
        ),

        "last_aligned_return": str(
            returns.index[-1].date()
        ),

        "price_field": "TRDPRC_1",

        "price_adjustments": ADJUSTMENTS,
    }


def retrieve_beta_lseg(
    stock_ric: str,
    etf_ric: str,
    as_of: date,
) -> dict:
    """Retrieve prices and estimate beta.

    The caller must already have opened an LSEG session.
    """

    import lseg.data as ld

    sessions = nyse_lookback_sessions(
        as_of
    )

    start = (
        sessions[0] - timedelta(days=12)
    ).date().isoformat()

    end = (
        as_of + timedelta(days=1)
    ).isoformat()

    prices = {}

    for name, ric in (
        ("stock", stock_ric),
        ("ETF", etf_ric),
    ):

        raw = ld.get_history(
            universe=ric,
            fields=["TRDPRC_1"],
            interval="daily",
            start=start,
            end=end,
            adjustments=ADJUSTMENTS,
        )

        prices[name] = normalize_lseg_history(
            raw,
            name,
        )

    return estimate_beta(
        prices["stock"],
        prices["ETF"],
        sessions,
    )


def run_live_analysis(
    stock_ric: str,
    etf_ric: str,
    shares: int,
) -> tuple[dict, pd.DataFrame]:
    """Estimate beta and produce ETF hedge stress scenarios."""

    import lseg.data as ld

    from lseg_api.equity import (
        get_current_equity_price,
    )

    as_of = last_completed_nyse_session()

    ld.open_session()

    try:

        stats = retrieve_beta_lseg(
            stock_ric,
            etf_ric,
            as_of,
        )

        stock_spot = float(
            get_current_equity_price(
                stock_ric
            )
        )

        etf_spot = float(
            get_current_equity_price(
                etf_ric
            )
        )

    finally:
        ld.close_session()

    if not all(
        math.isfinite(value) and value > 0
        for value in (
            stock_spot,
            etf_spot,
        )
    ):
        raise ValueError(
            "Invalid stock or ETF snapshot price"
        )

    scenarios = etf_cross_hedge_stresses(
        stock_spot=stock_spot,
        shares=shares,
        etf_spot=etf_spot,
        beta=stats["beta_stock_on_etf"],
    )

    stats.update({
        "stock_ric": stock_ric,
        "etf_ric": etf_ric,
        "shares": shares,

        "stock_spot_latest": stock_spot,
        "etf_spot_latest": etf_spot,

        "quotes_freshness_verified": False,
        "currency_parity_verified": False,
        "research_only": True,

        "limitations": [
            "TRDPRC_1 is a daily trade-price proxy.",
            "Adjustment coverage requires verification.",
            "Dividend and total-return accounting is incomplete.",
            "Spot quotes may differ in time from the beta date.",
            "Same quote currency is assumed, not verified.",
            "ETF shorting requires borrowing and margin.",
            "Beta can change in stressed markets.",
        ],
    })

    return stats, scenarios


def main() -> None:

    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--ric",
        default="NVDA.O",
    )

    parser.add_argument(
        "--etf-ric",
        required=True,
    )

    parser.add_argument(
        "--shares",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--assume-same-currency",
        action="store_true",
    )

    args = parser.parse_args()

    if args.shares <= 0:
        parser.error(
            "--shares must be positive"
        )

    if (
        args.ric.strip().upper()
        == args.etf_ric.strip().upper()
    ):
        parser.error(
            "Choose a different ETF RIC"
        )

    if not args.assume_same_currency:
        parser.error(
            "Confirm quote-currency assumption "
            "with --assume-same-currency"
        )

    stats, scenarios = run_live_analysis(
        args.ric,
        args.etf_ric,
        args.shares,
    )

    OUTPUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    stem = (
        f"{args.ric.replace('/', '_')}_"
        f"{args.etf_ric.replace('/', '_')}_"
        f"{stats['as_of_session']}"
    )

    scenarios.to_csv(
        OUTPUT / f"{stem}_stresses.csv",
        index=False,
    )

    (
        OUTPUT / f"{stem}_audit.json"
    ).write_text(
        json.dumps(
            stats,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\n--- AUTOMATIC ETF CROSS-HEDGE BETA ---"
    )

    fields = [
        "stock_ric",
        "etf_ric",
        "as_of_session",
        "aligned_returns",
        "stock_missing_price_sessions",
        "etf_missing_price_sessions",
        "beta_stock_on_etf",
        "correlation",
        "r_squared",
        "annualized_residual_vol",
        "stock_spot_latest",
        "etf_spot_latest",
    ]

    for field in fields:
        print(
            f"{field}: {stats[field]}"
        )

    print(
        "\nETF hedge positions "
        "(negative = short):"
    )

    print(
        scenarios.groupby(
            "Hedge Fraction"
        )[
            "ETF Position Shares (signed)"
        ].first().to_string()
    )

    if stats["r_squared"] < 0.10:
        print(
            "\nWARNING: Low historical explanatory power; "
            "substantial basis-risk concern."
        )

    print(
        "\nOutput:",
        OUTPUT,
    )

    print(
        "RESEARCH ONLY: quote freshness, currencies "
        "and ETF borrowing terms unverified."
    )


if __name__ == "__main__":
    main()
