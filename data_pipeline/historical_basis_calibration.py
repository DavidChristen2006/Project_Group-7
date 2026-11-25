
"""Historical stock–ETF basis-risk calibration.

Uses daily adjusted LSEG closing prices.

No missing prices are forward filled.
No option quotes or live trades are required.

The fixed 252-session beta estimate is reused;
it is not retrained or re-estimated here.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from math import ceil, isfinite

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BasisCalibration:
    """Represent BasisCalibration within the hedge research workflow.
    """
    horizon_sessions: int
    beta_stock_on_etf: float
    r_squared: float
    reverse_slope: float
    observations: int
    nonoverlapping_windows: int
    first_window_end: date
    last_window_end: date
    last_paired_close: date
    residual_p05: float
    residual_p25: float
    residual_p50: float
    residual_p75: float
    residual_p95: float
    symmetric_width: float


# =========================================================
# LSEG HISTORICAL PRICE NORMALIZATION
# =========================================================

def parse_adjusted_close(
    frame: pd.DataFrame,
    ric: str,
) -> pd.Series:
    """Convert one LSEG adjusted closing-price response into a series."""

    if frame is None or frame.empty:
        raise ValueError(
            f"No adjusted closing-price history for {ric}"
        )

    work = frame.copy()

    if "Instrument" in work.columns:

        work = work.loc[
            work["Instrument"]
            .astype(str)
            .str.strip()
            .str.upper()
            == ric.strip().upper()
        ].copy()

        if work.empty:
            raise ValueError(
                f"No response rows matched instrument {ric}"
            )

    date_columns = [
        col
        for col in work.columns
        if "date" in str(col).casefold()
        and col != "Instrument"
    ]

    price_columns = [
        col
        for col in work.columns
        if "close" in str(col).casefold()
        and "date" not in str(col).casefold()
        and col != "Instrument"
    ]

    if (
        len(date_columns) != 1
        or len(price_columns) != 1
    ):
        raise ValueError(
            "Cannot identify one date and adjusted "
            f"close column for {ric}; "
            f"columns={list(work.columns)}"
        )

    dates = pd.to_datetime(
        work[date_columns[0]],
        errors="coerce",
        utc=True,
    )

    prices = pd.to_numeric(
        work[price_columns[0]],
        errors="coerce",
    )

    valid = (
        dates.notna()
        & prices.notna()
    )

    dated = pd.DataFrame({
        "Date": (
            dates.loc[valid]
            .dt.tz_convert(None)
            .dt.normalize()
        ),
        "Close": (
            prices.loc[valid]
            .astype(float)
        ),
    })

    if dated.empty:
        raise ValueError(
            f"No valid dated adjusted closes for {ric}"
        )

    if (
        (dated["Close"] <= 0).any()
        or not np.isfinite(
            dated["Close"]
        ).all()
    ):
        raise ValueError(
            f"Invalid adjusted closing price for {ric}"
        )

    if dated["Date"].duplicated().any():
        raise ValueError(
            f"Duplicate adjusted close dates for {ric}"
        )

    result = (
        dated.set_index("Date")["Close"]
        .sort_index()
    )

    result.name = ric

    return result


def retrieve_adjusted_close_lseg(
    ric: str,
    start: date,
    end: date,
) -> pd.Series:
    """Fetch adjusted daily close prices.

    Caller must open and close the LSEG session.
    """

    import lseg.data as ld

    frame = ld.get_data(
        universe=[ric],
        fields=[
            "TR.ClosePrice.date",
            "TR.ClosePrice",
        ],
        parameters={
            "SDate": start.isoformat(),
            "EDate": end.isoformat(),
            "Frq": "D",
            "Adjusted": "1",
        },
    )

    return parse_adjusted_close(
        frame,
        ric,
    )


# =========================================================
# HORIZON-MATCHED RESIDUAL CALIBRATION
# =========================================================

def calibrate_basis_residuals(
    stock_prices: pd.Series,
    etf_prices: pd.Series,
    *,
    horizon_sessions: int,
    beta_stock_on_etf: float,
    r_squared: float,
    calendar_index: pd.DatetimeIndex | None = None,
    as_of: date | None = None,
) -> tuple[
    BasisCalibration,
    pd.DataFrame,
]:
    """Estimate ETF basis residuals over complete horizon windows.

    Residual =
        ETF horizon return
        - (R2 / stock-on-ETF beta) * stock horizon return

    Rolling windows overlap and are not independent.
    """

    if (
        not isinstance(horizon_sessions, int)
        or horizon_sessions < 1
    ):
        raise ValueError(
            "Horizon must be a positive number "
            "of NYSE trading sessions"
        )

    beta = float(
        beta_stock_on_etf
    )

    r2 = float(
        r_squared
    )

    if (
        not isfinite(beta)
        or abs(beta) <= 1e-8
    ):
        raise ValueError(
            "A nonzero finite historical beta is required"
        )

    if (
        not isfinite(r2)
        or not 0 <= r2 <= 1
    ):
        raise ValueError(
            "Historical R-squared must lie between 0 and 1"
        )

    slope = r2 / beta

    # -----------------------------------------------------
    # VALIDATE SERIES AND NORMALIZE OBSERVATION DATES
    # -----------------------------------------------------

    def prepare(
        series: pd.Series,
        name: str,
    ) -> pd.Series:

        """Perform the prepare operation for this research workflow.

        Args:
            series: Input to the prepare operation.
            name: Input to the prepare operation.

        Returns:
            Computed result or status for the caller.
        """
        values = pd.to_numeric(
            series,
            errors="coerce",
        ).copy()

        index = pd.to_datetime(
            values.index,
            errors="coerce",
            utc=True,
        )

        if index.isna().any():
            raise ValueError(
                f"Invalid {name} observation date"
            )

        values.index = (
            index.tz_convert(None)
            .normalize()
        )

        if values.index.duplicated().any():
            raise ValueError(
                f"Duplicate {name} dates"
            )

        observed = values.dropna()

        if (
            (observed <= 0).any()
            or not np.isfinite(observed).all()
        ):
            raise ValueError(
                f"Invalid {name} adjusted price"
            )

        return values.sort_index()

    stock = prepare(
        stock_prices,
        "stock",
    )

    etf = prepare(
        etf_prices,
        "ETF",
    )

    # -----------------------------------------------------
    # BUILD THE NYSE CALENDAR
    # -----------------------------------------------------

    if calendar_index is None:

        import pandas_market_calendars as mcal

        calendar = mcal.get_calendar(
            "NYSE"
        )

        start = min(
            stock.index.min(),
            etf.index.min(),
        )

        end = max(
            stock.index.max(),
            etf.index.max(),
        )

        if as_of is not None:
            end = min(
                end,
                pd.Timestamp(as_of),
            )

        calendar_index = (
            calendar.valid_days(
                start_date=start,
                end_date=end,
            )
            .tz_localize(None)
            .normalize()
        )

    calendar_index = (
        pd.DatetimeIndex(
            pd.to_datetime(
                calendar_index,
                utc=True,
            )
        )
        .tz_convert(None)
        .normalize()
        .sort_values()
        .unique()
    )

    if as_of is not None:

        calendar_index = calendar_index[
            calendar_index <= pd.Timestamp(as_of)
        ]

    if len(calendar_index) <= horizon_sessions:
        raise ValueError(
            "Not enough calendar sessions "
            "for the selected hedge horizon"
        )

    # -----------------------------------------------------
    # ALIGN STOCK AND ETF CLOSING PRICES
    # -----------------------------------------------------

    aligned = pd.DataFrame({
        "Stock close": stock.reindex(
            calendar_index
        ),
        "ETF close": etf.reindex(
            calendar_index
        ),
    })

    pair_ok = (
        aligned.notna()
        .all(axis=1)
    )

    if not pair_ok.any():
        raise ValueError(
            "No matching adjusted stock/ETF closes"
        )

    last_paired_close = (
        aligned.index[pair_ok][-1]
        .date()
    )

    # Every session in an H-session window must
    # have both stock and ETF closing prices.
    # No forward-filling is allowed.

    complete = (
        pair_ok.rolling(
            horizon_sessions + 1,
            min_periods=horizon_sessions + 1,
        )
        .sum()
        .eq(horizon_sessions + 1)
    )

    gross_returns = (
        aligned
        / aligned.shift(horizon_sessions)
        - 1.0
    )

    returns = (
        gross_returns.loc[complete]
        .dropna()
        .copy()
    )

    if returns.empty:
        raise ValueError(
            "No complete historical stock/ETF windows "
            "for this hedge horizon"
        )

    returns = returns.rename(
        columns={
            "Stock close": "Stock return",
            "ETF close": "ETF return",
        }
    )

    # -----------------------------------------------------
    # COMPUTE ETF RESIDUALS
    # -----------------------------------------------------

    returns["ETF linear center"] = (
        slope
        * returns["Stock return"]
    )

    returns["ETF residual"] = (
        returns["ETF return"]
        - returns["ETF linear center"]
    )

    # Count complete non-overlapping periods.
    # Their return intervals do not share daily returns.
    # This is a coverage audit, not an exact effective
    # statistical sample size.

    eligible_end_positions = np.flatnonzero(
        complete.to_numpy()
    )

    disjoint = 0
    last_end = -1

    for end_position in eligible_end_positions:

        if (
            end_position - horizon_sessions
            >= last_end
        ):

            disjoint += 1
            last_end = end_position

    quantiles = returns[
        "ETF residual"
    ].quantile([
        0.05,
        0.25,
        0.50,
        0.75,
        0.95,
    ])

    low = float(
        quantiles.loc[0.05]
    )

    high = float(
        quantiles.loc[0.95]
    )

    calibration = BasisCalibration(
        horizon_sessions=horizon_sessions,
        beta_stock_on_etf=beta,
        r_squared=r2,
        reverse_slope=slope,
        observations=len(returns),
        nonoverlapping_windows=disjoint,
        first_window_end=returns.index[0].date(),
        last_window_end=returns.index[-1].date(),
        last_paired_close=last_paired_close,
        residual_p05=low,
        residual_p25=float(quantiles.loc[0.25]),
        residual_p50=float(quantiles.loc[0.50]),
        residual_p75=float(quantiles.loc[0.75]),
        residual_p95=high,
        symmetric_width=max(
            abs(low),
            abs(high),
        ),
    )

    return calibration, returns


# =========================================================
# CONVERT EMPIRICAL WIDTH TO EXISTING STREAMLIT SLIDER
# =========================================================

def suggested_slider_width(
    calibration: BasisCalibration,
) -> int:
    """Round the empirical residual width up to a 5pp step."""

    width_points = (
        100.0
        * calibration.symmetric_width
    )

    if (
        not isfinite(width_points)
        or width_points < 0
    ):
        raise ValueError(
            "Invalid empirical residual width"
        )

    rounded = (
        5
        * ceil(
            (width_points - 1e-12)
            / 5
        )
    )

    if rounded > 100:
        raise ValueError(
            "Empirical residual width exceeds "
            "the slider's 100pp limit"
        )

    return int(rounded)


# =========================================================
# STREAMLIT INTEGRATION
# =========================================================

def render_historical_basis_calibration(
    hedge_result: dict,
    cross_result: dict | None,
) -> None:
    """Display empirical basis-risk calibration.

    This function must run BEFORE the existing
    render_risk_aware_recommendations() function
    when an ETF result is available.
    """

    import streamlit as st
    import lseg.data as ld

    from data_pipeline.estimate_etf_beta import (
        last_completed_nyse_session,
    )

    if not cross_result:
        return

    meta = hedge_result["meta"]

    if (
        cross_result.get("snapshot")
        != meta.get("retrieved_at_new_york")
    ):
        return

    report = cross_result.get(
        "beta_report"
    )

    if not isinstance(report, dict):
        return

    beta = float(
        cross_result["beta"]
    )

    r2 = float(
        report["r_squared"]
    )

    if (
        not isfinite(beta)
        or abs(beta) < 1e-8
        or not 0 <= r2 <= 1
    ):
        return

    stock_ric = str(
        hedge_result["inputs"].ric
    )

    etf_ric = str(
        cross_result["etf_ric"]
    )

    horizon = int(
        meta["research_horizon_sessions"]
    )

    as_of = pd.Timestamp(
        last_completed_nyse_session()
    ).date()

    st.subheader(
        "Empirical ETF basis-risk calibration"
    )

    st.caption(
        "Adjusted historical closing-price returns, "
        "not dividend-reinvested total returns. "
        "Rolling hedge-horizon windows overlap "
        "and are not independent. "
        "Historical percentiles are descriptive, "
        "not VaR confidence guarantees. "
        "The fixed 252-session beta is unchanged."
    )

    years = int(
        st.number_input(
            "Historical lookback for basis residuals (years)",
            min_value=3,
            max_value=12,
            value=8,
            step=1,
            key="empirical_basis_years",
        )
    )

    # Cache the result for this specific stock, ETF,
    # horizon, date, lookback and beta estimate.

    key = (
        stock_ric,
        etf_ric,
        horizon,
        years,
        as_of.isoformat(),
        beta,
        r2,
    )

    saved = st.session_state.get(
        "empirical_basis_result"
    )

    if (
        not isinstance(saved, dict)
        or saved.get("key") != key
    ):
        saved = None

    # -----------------------------------------------------
    # FETCH HISTORICAL PRICES ON USER REQUEST ONLY
    # -----------------------------------------------------

    if st.button(
        "Calibrate from LSEG adjusted daily closes",
        key="empirical_basis_fetch",
    ):

        start = (
            as_of
            - timedelta(
                days=round(years * 365.25) + 10
            )
        )

        try:

            ld.open_session()

            try:

                stock_prices = (
                    retrieve_adjusted_close_lseg(
                        stock_ric,
                        start,
                        as_of,
                    )
                )

                etf_prices = (
                    retrieve_adjusted_close_lseg(
                        etf_ric,
                        start,
                        as_of,
                    )
                )

            finally:

                ld.close_session()

            calibration, windows = (
                calibrate_basis_residuals(
                    stock_prices,
                    etf_prices,
                    horizon_sessions=horizon,
                    beta_stock_on_etf=beta,
                    r_squared=r2,
                    as_of=as_of,
                )
            )

            saved = {
                "key": key,
                "calibration": calibration,
                "windows": windows,
                "stock_prices": stock_prices,
                "etf_prices": etf_prices,
            }

            st.session_state[
                "empirical_basis_result"
            ] = saved

        except Exception as exc:

            st.session_state.pop(
                "empirical_basis_result",
                None,
            )

            st.error(
                "Historical basis calibration unavailable: "
                f"{exc}"
            )

            saved = None

    if saved is None:
        return

    calibration = saved[
        "calibration"
    ]

    windows = saved[
        "windows"
    ]

    # -----------------------------------------------------
    # DISPLAY SAMPLE COVERAGE AND RESIDUAL QUANTILES
    # -----------------------------------------------------

    st.write(
        f"**{calibration.observations:,}** overlapping "
        f"{horizon}-session returns; "
        f"**{calibration.nonoverlapping_windows}** "
        "complete non-overlapping windows "
        "(audit count)."
    )

    st.write(
        "Observed ETF residual quantiles: "
        f"5th **{calibration.residual_p05:+.2%}**, "
        f"median **{calibration.residual_p50:+.2%}**, "
        f"95th **{calibration.residual_p95:+.2%}**. "
        f"Last paired close: {calibration.last_paired_close}."
    )

    st.dataframe(
        pd.DataFrame([
            {
                "Horizon sessions":
                    calibration.horizon_sessions,
                "Rolling windows":
                    calibration.observations,
                "Non-overlap windows":
                    calibration.nonoverlapping_windows,
                "First window end":
                    calibration.first_window_end,
                "Last window end":
                    calibration.last_window_end,
                "Residual p05":
                    calibration.residual_p05,
                "Residual p50":
                    calibration.residual_p50,
                "Residual p95":
                    calibration.residual_p95,
            }
        ]),
        hide_index=True,
        use_container_width=True,
    )

    # -----------------------------------------------------
    # OBSERVED HISTORICAL STOCK–ETF RETURN PAIRS
    # -----------------------------------------------------

    scatter = (
        windows[
            ["Stock return", "ETF return"]
        ]
        .mul(100)
        .rename(
            columns={
                "Stock return": "Stock return (%)",
                "ETF return": "ETF return (%)",
            }
        )
    )

    st.scatter_chart(
        scatter,
        x="Stock return (%)",
        y="ETF return (%)",
    )

    # -----------------------------------------------------
    # MINIMUM DATA-COVERAGE SAFEGUARD
    # -----------------------------------------------------

    if calibration.nonoverlapping_windows < 20:

        st.warning(
            "Fewer than 20 complete non-overlapping "
            "hedge-horizon windows. The residual range "
            "is too sample-limited to apply automatically. "
            "The descriptive results remain available."
        )

        return

    # Guard against stale historical data.

    if calibration.last_paired_close < as_of:

        import pandas_market_calendars as mcal

        missed = len(
            mcal.get_calendar("NYSE").valid_days(
                start_date=(
                    calibration.last_paired_close
                    + timedelta(days=1)
                ),
                end_date=as_of,
            )
        )

        if missed > 5:

            st.warning(
                f"Historical adjusted prices lag "
                f"{missed} NYSE sessions. "
                "Refresh the data before applying "
                "the empirical range."
            )

            return

    try:

        width = suggested_slider_width(
            calibration
        )

    except ValueError as exc:

        st.warning(str(exc))
        return

    st.caption(
        "Symmetric residual width covering the observed "
        "5th–95th percentile range: "
        f"±{width} percentage points, "
        "rounded upward to the existing slider step. "
        "This is a conservative symmetric approximation "
        "to potentially asymmetric residual quantiles."
    )

    # -----------------------------------------------------
    # APPLY TO THE EXISTING RISK-AWARE STRATEGY COMPARISON
    # -----------------------------------------------------

    if st.button(
        f"Apply ±{width}pp to linked-scenario residual slider",
        key="empirical_basis_apply",
    ):

        st.session_state[
            "risk_aware_etf_residual_width"
        ] = width

        st.session_state[
            "risk_aware_stress_model"
        ] = (
            "Historically linked shocks + residual stress"
        )

        st.success(
            f"Historically linked mode selected, "
            f"with the ETF residual slider set to "
            f"±{width} percentage points."
        )
