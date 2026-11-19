
"""Discover ETF option contracts using LSEG Equity Derivative Quotes.

An LSEG session must already be open.

Only records with put/call type, strike, expiration,
and a valid option RIC are accepted.
"""

from __future__ import annotations

import math
import re
from datetime import date

import pandas as pd


# The ETF price RIC and option-search key may differ.


REQUIRED = (
    "RIC",
    "CallPutOption",
    "StrikePrice",
    "ExpiryDate",
)

SELECT_FIELDS = (
    "RIC,UnderlyingQuoteRIC,CallPutOption,"
    "StrikePrice,ExpiryDate,RCSAssetClass,"
    "IsChain,AssetState"
)


def option_search_keys(
    etf_ric: str,
) -> tuple[str, ...]:
    """Generate search identifiers for any LSEG instrument.

    Examples:
        ABC.O -> ("ABC.O", "ABC")
        XYZ.P -> ("XYZ.P", "XYZ")
        XYZ   -> ("XYZ",)

    No instrument-specific mappings are used.
    """

    ric = str(etf_ric).strip().upper()

    if not re.fullmatch(
        r"[A-Z0-9.^_-]{1,40}",
        ric,
    ):
        raise ValueError(
            f"Invalid LSEG RIC: {etf_ric!r}"
        )

    candidates = [ric]

    if "." in ric:
        ticker = ric.split(".", 1)[0]

        if ticker:
            candidates.append(ticker)

    return tuple(
        dict.fromkeys(candidates)
    )


def validate_contracts(
    raw: pd.DataFrame | None,
    target_date: date,
    spot: float,
    days_after: int = 35,
) -> pd.DataFrame:

    if raw is None or raw.empty:
        return pd.DataFrame(
            columns=list(REQUIRED)
        )

    if not set(REQUIRED).issubset(
        raw.columns
    ):
        return pd.DataFrame(
            columns=list(REQUIRED)
        )

    frame = raw.copy()

    frame["RIC"] = (
        frame["RIC"]
        .astype("string")
        .str.strip()
    )

    frame["CallPutOption"] = (
        frame["CallPutOption"]
        .astype("string")
        .str.strip()
        .str.lower()
        .replace({
            "p": "put",
            "c": "call",
        })
    )

    frame["StrikePrice"] = pd.to_numeric(
        frame["StrikePrice"],
        errors="coerce",
    )

    frame["ExpiryDate"] = (
        pd.to_datetime(
            frame["ExpiryDate"],
            errors="coerce",
            utc=True,
        )
        .dt.tz_convert(None)
        .dt.normalize()
    )

    start = pd.Timestamp(
        target_date
    ).normalize()

    end = (
        start
        + pd.DateOffset(
            days=days_after
        )
    )

    mask = (
        frame["RIC"].notna()
        & frame["RIC"].ne("").fillna(False)
        & frame["CallPutOption"].isin(
            ["put", "call"]
        )
        & frame["StrikePrice"].between(
            spot * 0.70,
            spot * 1.20,
        )
        & frame["ExpiryDate"].between(
            start,
            end,
        )
    )

    if "IsChain" in frame.columns:

        is_chain = (
            frame["IsChain"]
            .astype("string")
            .str.lower()
            .isin(
                ["true", "1", "yes"]
            )
        )

        mask &= ~is_chain

    if "RCSAssetClass" in frame.columns:

        category = (
            frame["RCSAssetClass"]
            .astype("string")
            .str.strip()
            .str.upper()
        )

        mask &= (
            category.isna()
            | category.eq("OPT")
        )

    return (
        frame.loc[mask]
        .drop_duplicates("RIC")
        .reset_index(drop=True)
    )


def discover_etf_option_chain(
    etf_ric: str,
    etf_spot: float,
    target_date: date,
    *,
    max_days_after: int = 90,
    max_results: int = 2000,
) -> pd.DataFrame:
    """Retrieve actual option-contract candidates, not chains."""

    import lseg.data as ld

    if (
        not math.isfinite(etf_spot)
        or etf_spot <= 0
    ):
        raise ValueError(
            "Invalid ETF spot price"
        )

    if not (
        1 <= max_days_after <= 365
        and 1 <= max_results <= 10000
    ):
        raise ValueError(
            "Invalid discovery limits"
        )

    start = pd.Timestamp(
        target_date
    ).normalize()

    end = (
        start
        + pd.DateOffset(
            days=max_days_after
        )
    )

    attempts = []
    found = []

    for key in option_search_keys(
        etf_ric
    ):

        search_filter = (
            f"UnderlyingQuoteRIC eq '{key}' "
            "and RCSAssetClass eq 'OPT' "
            "and IsChain eq false "
            "and AssetState eq 'AC' "
            f"and ExpiryDate ge {start.date().isoformat()} "
            f"and ExpiryDate le {end.date().isoformat()} "
            f"and StrikePrice ge {etf_spot * 0.70:.4f} "
            f"and StrikePrice le {etf_spot * 1.20:.4f}"
        )

        try:

            raw = ld.discovery.search(
                view=(
                    ld.discovery.Views
                    .EQUITY_DERIVATIVE_QUOTES
                ),
                filter=search_filter,
                select=SELECT_FIELDS,
                top=max_results,
            )

        except Exception as exc:

            attempts.append(
                f"{key}: {type(exc).__name__}: {exc}"
            )

            continue

        count = (
            len(raw)
            if isinstance(
                raw,
                pd.DataFrame,
            )
            else 0
        )

        columns = (
            list(raw.columns)
            if isinstance(
                raw,
                pd.DataFrame,
            )
            else []
        )

        contracts = validate_contracts(
            raw,
            target_date,
            etf_spot,
            max_days_after,
        )

        attempts.append(
            f"{key}: {count} records, "
            f"{len(contracts)} valid contracts; "
            f"fields={columns}"
        )

        if not contracts.empty:
            found.append(
                contracts
            )

        if count >= max_results:

            print(
                f"WARNING: {key} reached the "
                f"{max_results} record limit. "
                "Results may be incomplete."
            )

    if not found:

        raise ValueError(
            "No valid ETF option contracts found. "
            "Search details: "
            + " | ".join(attempts)
        )

    result = (
        pd.concat(
            found,
            ignore_index=True,
        )
        .drop_duplicates("RIC")
        .reset_index(drop=True)
    )

    print(
        "ETF option discovery:",
        " | ".join(attempts),
    )

    return result
