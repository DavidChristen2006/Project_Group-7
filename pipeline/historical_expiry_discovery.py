
from dataclasses import dataclass

import pandas as pd

from pipeline.market_calendar import (
    next_monthly_expiry,
)


@dataclass(frozen=True)
class ExpiryConfig:
    min_days_after_exit: int = 1
    max_days_after_exit: int = 90
    max_expiries: int = 3

    def __post_init__(self):
        values = (
            self.min_days_after_exit,
            self.max_days_after_exit,
            self.max_expiries,
        )

        if any(
            not isinstance(value, int)
            or isinstance(value, bool)
            for value in values
        ):
            raise ValueError(
                "Expiry settings must be integers."
            )

        if self.min_days_after_exit < 1:
            raise ValueError(
                "Expiry must be after the hedge exit."
            )

        if (
            self.max_days_after_exit
            < self.min_days_after_exit
        ):
            raise ValueError(
                "Invalid expiry window."
            )

        if self.max_expiries < 1:
            raise ValueError(
                "max_expiries must be positive."
            )


def select_historical_expiries(
    available_expiries,
    decision_date,
    exit_date,
    config=None,
):
    """
    Select eligible expiries from supplied dates.

    The available_expiries argument must contain
    dates discovered or verified separately.

    This function does not query LSEG and does not
    establish historical contract availability.
    """

    if config is None:
        config = ExpiryConfig()

    decision = pd.Timestamp(
        decision_date
    ).normalize()

    exit_day = pd.Timestamp(
        exit_date
    ).normalize()

    if pd.isna(decision) or pd.isna(exit_day):
        raise ValueError(
            "Decision and exit dates are required."
        )

    if exit_day <= decision:
        raise ValueError(
            "Exit date must follow decision date."
        )

    rows = []

    for value in available_expiries:
        expiry = pd.Timestamp(value).normalize()

        if pd.isna(expiry):
            raise ValueError(
                "Invalid option expiration date."
            )

        days_after_exit = (
            expiry - exit_day
        ).days

        if not (
            config.min_days_after_exit
            <= days_after_exit
            <= config.max_days_after_exit
        ):
            continue

        rows.append({
            "Expiry": expiry,
            "Days After Exit": days_after_exit,
            "Days to Expiry": (
                expiry - decision
            ).days,
        })

    columns = [
        "Expiry",
        "Days After Exit",
        "Days to Expiry",
    ]

    if not rows:
        return pd.DataFrame(columns=columns)

    result = pd.DataFrame(rows)

    result = (
        result
        .drop_duplicates(subset=["Expiry"])
        .sort_values(
            ["Days After Exit", "Expiry"]
        )
        .head(config.max_expiries)
        .reset_index(drop=True)
    )

    return result[columns]


def generate_monthly_expiry_candidates(
    decision_date,
    exit_date,
    config=None,
):
    """
    Generate eligible monthly expiration dates.

    These are calendar candidates, not confirmed
    historical option contracts.

    Contract existence and liquidity must be
    verified by the LSEG discovery pipeline.
    """

    if config is None:
        config = ExpiryConfig(
            min_days_after_exit=1,
            max_days_after_exit=45,
            max_expiries=3,
        )

    decision = pd.Timestamp(
        decision_date
    ).normalize()

    exit_day = pd.Timestamp(
        exit_date
    ).normalize()

    if pd.isna(decision) or pd.isna(exit_day):
        raise ValueError("Invalid decision or exit date.")

    if exit_day <= decision:
        raise ValueError(
            "Exit date must follow decision date."
        )

    available_expiries = []

    cursor = exit_day

    for _ in range(config.max_expiries):
        expiry = pd.Timestamp(
            next_monthly_expiry(cursor)
        ).normalize()

        # Guard against unexpected calendar behavior.
        if expiry <= cursor:
            raise ValueError(
                "Monthly expiry function did not advance."
            )

        if (
            expiry - exit_day
        ).days > config.max_days_after_exit:
            break

        available_expiries.append(expiry)

        # Advance beyond the discovered expiry.
        cursor = expiry + pd.Timedelta(days=1)

    selected = select_historical_expiries(
        available_expiries=available_expiries,
        decision_date=decision,
        exit_date=exit_day,
        config=config,
    )

    return selected["Expiry"].tolist()

