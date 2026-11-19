
"""Horizon-aware selection of direct and ETF option structures.

Never invent contracts or shorten the requested hedge horizon.
Caller supplies LSEG pricing/selection functions and an open session.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Callable, Any

import numpy as np
import pandas as pd


@dataclass
class DirectOptionSelection:
    put: pd.Series
    call: pd.Series
    scored_candidates: pd.DataFrame
    expiry: date
    attempted_expiries: tuple[date, ...]


def _dated_chain(
    chain: pd.DataFrame,
    target_date: date,
) -> tuple[pd.DataFrame, list[date]]:
    """Return contracts at or after the target date."""

    if (
        chain is None
        or chain.empty
        or "ExpiryDate" not in chain.columns
    ):
        raise ValueError(
            "No option chain with an ExpiryDate column was supplied"
        )

    target = pd.Timestamp(target_date).date()
    work = chain.copy()

    work["_expiry_day"] = pd.to_datetime(
        work["ExpiryDate"],
        errors="coerce",
        utc=True,
    ).dt.date

    work = work.loc[
        work["_expiry_day"].notna()
    ].copy()

    if work.empty:
        raise ValueError(
            "All option expiration dates were missing or invalid"
        )

    observed = sorted(
        work["_expiry_day"].unique()
    )

    work = work.loc[
        work["_expiry_day"] >= target
    ].copy()

    if work.empty:
        raise ValueError(
            f"No option expiry on or after {target}; "
            f"available range {observed[0]} to {observed[-1]}"
        )

    return work, sorted(
        work["_expiry_day"].unique()
    )


def _no_success(
    kind: str,
    target_date: date,
    attempts: list[str],
) -> ValueError:

    reasons = " | ".join(
        attempts[-8:]
    )

    return ValueError(
        f"No eligible {kind} for target {target_date} "
        f"after checking {len(attempts)} expiry dates. "
        f"{reasons}"
    )


def select_nearest_direct_option_pair(
    *,
    chain: pd.DataFrame,
    spot: float,
    target_date: date,
    max_spread: float,
    assume_standard_contracts: bool,
    prepare_candidates: Callable[..., pd.DataFrame],
    add_option_market_data: Callable[..., pd.DataFrame],
    filter_liquid_options: Callable[..., pd.DataFrame],
    choose_pair: Callable[
        ...,
        tuple[pd.Series, pd.Series, pd.DataFrame],
    ],
    max_expiries: int = 12,
    multiplier: int = 100,
) -> DirectOptionSelection:
    """Try successive listed expirations for a direct-stock hedge."""

    if not np.isfinite(spot) or spot <= 0:
        raise ValueError(
            "Underlying spot must be positive and finite"
        )

    if (
        not np.isfinite(max_spread)
        or not 0 < max_spread <= 1
    ):
        raise ValueError(
            "max_spread must be between 0 and 1"
        )

    if max_expiries < 1:
        raise ValueError(
            "max_expiries must be positive"
        )

    dated, expiries = _dated_chain(
        chain,
        target_date,
    )

    attempts: list[str] = []
    attempted_dates: list[date] = []

    for expiry in expiries[:max_expiries]:

        attempted_dates.append(expiry)

        subset = dated.loc[
            dated["_expiry_day"] == expiry
        ].drop(
            columns="_expiry_day"
        )

        try:

            candidates = prepare_candidates(
                subset,
                spot=spot,
                target_expiry=pd.Timestamp(expiry),
            )

            if candidates is None or candidates.empty:
                raise ValueError(
                    "no eligible strike candidates"
                )

            quotes = add_option_market_data(
                candidates,
                batch_size=50,
            )

            if quotes is None or quotes.empty:
                raise ValueError(
                    "no bid/ask quotes"
                )

            if "QuoteSource" in quotes.columns:

                source = (
                    quotes["QuoteSource"]
                    .astype("string")
                    .str.lower()
                )

                quotes = quotes.loc[
                    ~source.str.contains(
                        "historical",
                        na=False,
                    )
                ].copy()

            if quotes.empty:
                raise ValueError(
                    "only historical fallback quotes"
                )

            liquid = filter_liquid_options(
                quotes,
                max_spread_pct=max_spread,
            )

            if liquid is None or liquid.empty:
                raise ValueError(
                    "no contracts passed liquidity filter"
                )

            put, call, scored = choose_pair(
                liquid,
                spot=spot,
                target_expiry=pd.Timestamp(expiry),
                multiplier=multiplier,
                assume_standard_contracts=(
                    assume_standard_contracts
                ),
            )

            if (
                put is None
                or call is None
                or scored is None
                or scored.empty
            ):
                raise ValueError(
                    "option pair not found"
                )

            put_expiry = pd.Timestamp(
                put["ExpiryDate"]
            ).date()

            call_expiry = pd.Timestamp(
                call["ExpiryDate"]
            ).date()

            if (
                put_expiry != expiry
                or call_expiry != expiry
            ):
                raise ValueError(
                    "selector returned mismatched expiration"
                )

            if expiry < target_date:
                raise ValueError(
                    "option expires before hedge target"
                )

            return DirectOptionSelection(
                put=put.copy(),
                call=call.copy(),
                scored_candidates=scored.copy(),
                expiry=expiry,
                attempted_expiries=tuple(
                    attempted_dates
                ),
            )

        except Exception as exc:

            attempts.append(
                f"{expiry}: {type(exc).__name__}: {exc}"
            )

    raise _no_success(
        "direct-stock option pair",
        target_date,
        attempts,
    )


def select_nearest_etf_option_structures(
    *,
    liquid: pd.DataFrame,
    etf_spot: float,
    target_date: date,
    select_structures: Callable[
        ...,
        dict[str, Any],
    ],
    multiplier: int = 100,
    assume_standard_contracts: bool = False,
    max_spread: float = 0.10,
    max_expiries: int = 12,
) -> dict[str, Any]:
    """Try successive ETF expirations until structures are eligible."""

    if (
        not np.isfinite(etf_spot)
        or etf_spot <= 0
    ):
        raise ValueError(
            "ETF spot must be positive and finite"
        )

    if max_expiries < 1:
        raise ValueError(
            "max_expiries must be positive"
        )

    dated, expiries = _dated_chain(
        liquid,
        target_date,
    )

    attempts: list[str] = []

    for expiry in expiries[:max_expiries]:

        group = dated.loc[
            dated["_expiry_day"] == expiry
        ].drop(
            columns="_expiry_day"
        )

        try:

            structures = select_structures(
                group,
                etf_spot=etf_spot,
                target_date=expiry,
                multiplier=multiplier,
                assume_standard_contracts=(
                    assume_standard_contracts
                ),
                max_days_after=1,
                max_spread=max_spread,
            )

            if not structures:
                raise ValueError(
                    "no complete structure"
                )

            for name, structure in structures.items():

                legs = getattr(
                    structure,
                    "legs",
                    None,
                )

                if not legs:
                    raise ValueError(
                        f"{name}: missing legs"
                    )

                if any(
                    pd.Timestamp(leg.expiry).date()
                    != expiry
                    for leg in legs
                ):
                    raise ValueError(
                        f"{name}: inconsistent expiration"
                    )

            return structures

        except Exception as exc:

            attempts.append(
                f"{expiry}: {type(exc).__name__}: {exc}"
            )

    raise _no_success(
        "ETF option structure",
        target_date,
        attempts,
    )
