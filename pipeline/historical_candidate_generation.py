"""pipeline/historical_candidate_generation.py: hedge research implementation.

Functions are documented at their point of definition.
Historical outputs are research estimates, not executable orders.
"""

from datetime import timedelta
from dataclasses import dataclass
from itertools import product

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "RIC",
    "Option Type",
    "Strike",
    "Expiry",
    "BID",
    "ASK",
}


@dataclass(frozen=True)
class CandidateConfig:
    """Represent CandidateConfig within the hedge research workflow.
    """
    max_spread_pct: float = 0.10
    max_expiry_gap_days: int = 45

    # Strikes expressed as fractions of stock price.
    long_put_range: tuple = (0.80, 1.05)
    short_call_range: tuple = (1.00, 1.25)
    short_put_range: tuple = (0.65, 0.95)

    def __post_init__(self):
        """Handle the internal post init step.
        """
        if not 0 < self.max_spread_pct <= 1:
            raise ValueError("Invalid spread limit.")

        if self.max_expiry_gap_days < 0:
            raise ValueError("Invalid expiry gap.")

        for lower, upper in (
            self.long_put_range,
            self.short_call_range,
            self.short_put_range,
        ):
            if not 0 < lower <= upper:
                raise ValueError("Invalid strike range.")


def prepare_option_quotes(
    options,
    spot_price,
    hedge_end_date,
    config,
):
    """Perform the prepare option quotes operation for this research workflow.

    Args:
        options: Input to the prepare_option_quotes operation.
        spot_price: Input to the prepare_option_quotes operation.
        hedge_end_date: Input to the prepare_option_quotes operation.
        config: Input to the prepare_option_quotes operation.

    Returns:
        Computed result or status for the caller.
    """
    missing = REQUIRED_COLUMNS - set(options.columns)

    if missing:
        raise ValueError(
            f"Missing option columns: {sorted(missing)}"
        )

    if not np.isfinite(spot_price) or spot_price <= 0:
        raise ValueError("Invalid stock price.")

    end_date = pd.Timestamp(
        hedge_end_date
    ).normalize()

    max_expiry_date = pd.Timestamp(
        end_date.date()
        + timedelta(days=config.max_expiry_gap_days)
    )

    if pd.isna(end_date):
        raise ValueError("Invalid hedge end date.")

    df = options.copy()

    df["Option Type"] = (
        df["Option Type"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    for column in ["Strike", "BID", "ASK"]:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df["Expiry"] = pd.to_datetime(
        df["Expiry"],
        errors="coerce",
    ).dt.normalize()

    df = df.loc[
        df["Option Type"].isin(["put", "call"])
        & df["RIC"].notna()
        & (df["Strike"] > 0)
        & (df["BID"] > 0)
        & (df["ASK"] >= df["BID"])
        & (df["Expiry"] > end_date)
        & (df["Expiry"] <= max_expiry_date)
    ].copy()

    # If the upstream data already identifies
    # contracts as illiquid, respect that flag.
    if "Liquid" in df.columns:
        df = df.loc[
            df["Liquid"].fillna(False).eq(True)
        ].copy()

    df["MID"] = (df["BID"] + df["ASK"]) / 2

    df["SPREAD_PCT"] = (
        (df["ASK"] - df["BID"]) / df["MID"]
    )

    df = df.loc[
        df["SPREAD_PCT"] <= config.max_spread_pct
    ].copy()

    df["Strike Pct"] = (
        df["Strike"] / spot_price
    )

    # Duplicate contracts must not enter
    # candidate generation with ambiguous prices.
    duplicates = df.duplicated(
        ["RIC", "Expiry", "Option Type", "Strike"]
    )

    if duplicates.any():
        raise ValueError(
            "Duplicate option quotes detected."
        )

    return df.sort_values(
        ["Expiry", "Option Type", "Strike", "RIC"]
    ).reset_index(drop=True)


def make_candidate(
    strategy,
    expiry,
    long_put,
    short_call=None,
    short_put=None,
):
    # Positive Net Premium = debit paid.
    # Buy at ASK and sell at BID.
    """Construct candidate from the supplied inputs.

    Args:
        strategy: Input to the make_candidate operation.
        expiry: Input to the make_candidate operation.
        long_put: Input to the make_candidate operation.
        short_call: Input to the make_candidate operation.
        short_put: Input to the make_candidate operation.

    Returns:
        Computed result or status for the caller.
    """
    debit = float(long_put["ASK"])

    if short_call is not None:
        debit -= float(short_call["BID"])

    if short_put is not None:
        debit -= float(short_put["BID"])

    return {
        "Strategy": strategy,
        "Expiry": expiry,
        "Long Put RIC": long_put["RIC"],
        "Long Put Strike": float(long_put["Strike"]),
        "Long Put Ask": float(long_put["ASK"]),
        "Short Call RIC": (
            short_call["RIC"]
            if short_call is not None else None
        ),
        "Short Call Strike": (
            float(short_call["Strike"])
            if short_call is not None else np.nan
        ),
        "Short Call Bid": (
            float(short_call["BID"])
            if short_call is not None else np.nan
        ),
        "Short Put RIC": (
            short_put["RIC"]
            if short_put is not None else None
        ),
        "Short Put Strike": (
            float(short_put["Strike"])
            if short_put is not None else np.nan
        ),
        "Short Put Bid": (
            float(short_put["BID"])
            if short_put is not None else np.nan
        ),
        "Net Premium Per Share": debit,
        "Option Legs": (
            1
            + int(short_call is not None)
            + int(short_put is not None)
        ),
    }


def generate_hedge_candidates(
    options,
    underlying_ric,
    spot_price,
    hedge_end_date,
    shares,
    option_multiplier=100,
    config=None,
):
    """Perform the generate hedge candidates operation for this research workflow.

    Args:
        options: Input to the generate_hedge_candidates operation.
        underlying_ric: Input to the generate_hedge_candidates operation.
        spot_price: Input to the generate_hedge_candidates operation.
        hedge_end_date: Input to the generate_hedge_candidates operation.
        shares: Input to the generate_hedge_candidates operation.
        option_multiplier: Input to the generate_hedge_candidates operation.
        config: Input to the generate_hedge_candidates operation.

    Returns:
        Computed result or status for the caller.
    """
    if config is None:
        config = CandidateConfig()

    if not isinstance(shares, (int, np.integer)):
        raise ValueError("Shares must be an integer.")

    if shares <= 0 or option_multiplier <= 0:
        raise ValueError("Invalid position size.")

    if shares % option_multiplier != 0:
        raise ValueError(
            "Shares must be divisible by multiplier."
        )

    if not underlying_ric:
        raise ValueError("Underlying RIC is required.")

    quotes = prepare_option_quotes(
        options,
        spot_price,
        hedge_end_date,
        config,
    )

    contracts = shares // option_multiplier
    records = []

    # Every option leg in a candidate shares
    # the same expiration date.
    for expiry, chain in quotes.groupby(
        "Expiry",
        sort=True,
    ):
        puts = chain.loc[
            chain["Option Type"] == "put"
        ]

        calls = chain.loc[
            chain["Option Type"] == "call"
        ]

        long_puts = puts.loc[
            puts["Strike Pct"].between(
                *config.long_put_range
            )
        ]

        short_calls = calls.loc[
            calls["Strike Pct"].between(
                *config.short_call_range
            )
        ]

        short_puts = puts.loc[
            puts["Strike Pct"].between(
                *config.short_put_range
            )
        ]

        long_put_records = long_puts.to_dict(
            "records"
        )

        short_call_records = short_calls.to_dict(
            "records"
        )

        short_put_records = short_puts.to_dict(
            "records"
        )

        # Protective Put
        for long_put in long_put_records:
            records.append(
                make_candidate(
                    "Protective Put",
                    expiry,
                    long_put,
                )
            )

        # Collar: long put and short call.
        for long_put, short_call in product(
            long_put_records,
            short_call_records,
        ):
            if (
                short_call["Strike"]
                <= long_put["Strike"]
            ):
                continue

            records.append(
                make_candidate(
                    "Collar",
                    expiry,
                    long_put,
                    short_call=short_call,
                )
            )

        # Put Spread: long higher-strike put
        # and short lower-strike put.
        for long_put, short_put in product(
            long_put_records,
            short_put_records,
        ):
            if (
                short_put["Strike"]
                >= long_put["Strike"]
            ):
                continue

            records.append(
                make_candidate(
                    "Put Spread",
                    expiry,
                    long_put,
                    short_put=short_put,
                )
            )

    result = pd.DataFrame(records)

    if result.empty:
        return result

    result.insert(
        0, "Underlying RIC", underlying_ric
    )

    result["Spot Price"] = float(spot_price)
    result["Shares"] = int(shares)
    result["Contracts"] = int(contracts)

    result["Long Put Strike Pct"] = (
        result["Long Put Strike"] / spot_price
    )

    result["Short Call Strike Pct"] = (
        result["Short Call Strike"] / spot_price
    )

    result["Short Put Strike Pct"] = (
        result["Short Put Strike"] / spot_price
    )

    result["Net Premium Pct"] = (
        result["Net Premium Per Share"]
        / spot_price
    )

    result["Net Premium Total"] = (
        result["Net Premium Per Share"]
        * shares
    )

    # Stable candidate identity across repeated runs.
    identity_columns = [
        "Underlying RIC",
        "Strategy",
        "Expiry",
        "Long Put RIC",
        "Short Call RIC",
        "Short Put RIC",
    ]

    result["Candidate ID"] = (
        pd.util.hash_pandas_object(
            result[identity_columns].fillna(""),
            index=False,
        )
        .astype(str)
    )

    return result.sort_values(
        ["Expiry", "Strategy", "Candidate ID"]
    ).reset_index(drop=True)
