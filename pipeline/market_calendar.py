
import pandas as pd
import pandas_market_calendars as mcal

from pipeline.expired_option_pipeline import third_friday


# Cboe calendar for US equity options.
OPTIONS_CALENDAR = mcal.get_calendar(
    "CBOE_Equity_Options"
)


def get_option_trading_days(start_date, end_date):
    sessions = OPTIONS_CALENDAR.valid_days(
        start_date=start_date,
        end_date=end_date
    )

    # Calendar dates for comparison with locally stored data.
    return pd.DatetimeIndex(
        sessions.tz_convert("UTC").tz_localize(None)
    ).normalize()


def is_option_trading_day(date):
    date = pd.Timestamp(date).normalize()

    sessions = get_option_trading_days(
        date,
        date
    )

    return date in sessions


def previous_option_trading_day(date):
    date = pd.Timestamp(date).normalize()

    sessions = get_option_trading_days(
        date - pd.Timedelta(days=14),
        date
    )

    eligible = sessions[sessions <= date]

    if len(eligible) == 0:
        raise ValueError(
            f"No option trading session before {date.date()}"
        )

    return eligible[-1]


def monthly_option_trading_expiry(year, month):
    standard_expiry = third_friday(year, month)

    return previous_option_trading_day(
        standard_expiry
    )


def next_monthly_expiry(exit_date):
    exit_date = pd.Timestamp(
        exit_date
    ).normalize()

    month_start = exit_date.replace(day=1)

    for offset in range(12):
        month = month_start + pd.DateOffset(
            months=offset
        )

        expiry = monthly_option_trading_expiry(
            month.year,
            month.month
        )

        if expiry > exit_date:
            return expiry

    raise ValueError(
        "No suitable monthly option expiry found."
    )
