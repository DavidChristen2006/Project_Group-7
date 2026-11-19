
"""Resolve ordinary stock and ETF tickers to LSEG RICs.

Known mappings are used directly.
Unknown tickers are resolved through LSEG Discovery.
Ambiguous results require an explicit RIC.
"""

from __future__ import annotations

import argparse
import re

import pandas as pd


# These two mappings have already worked in this project.
PINNED_RICS = {
    "NVDA": "NVDA.O",
    "SPY": "SPY.P",
}

SYMBOL_PATTERN = re.compile(
    r"[A-Z0-9][A-Z0-9-]{0,14}\Z"
)

RICH_SELECT = (
    "RIC,TickerSymbol,DocumentTitle,ExchangeName,"
    "ExchangeCode,RCSExchangeCountryLeaf,AssetState,"
    "IsPrimaryRIC,RCSAssetCategoryLeaf"
)

BASIC_SELECT = (
    "RIC,TickerSymbol,DocumentTitle"
)


class TickerLookupError(ValueError):
    """A ticker could not be resolved unambiguously."""


def _is_true(value) -> bool:
    return str(value).strip().lower() in {
        "true",
        "1",
        "yes",
    }


def normalize_candidates(
    ticker: str,
    raw_frames: list[pd.DataFrame],
) -> pd.DataFrame:
    """Keep exact, active ticker matches without duplicate RICs."""

    columns = [
        "RIC",
        "TickerSymbol",
        "DocumentTitle",
        "ExchangeName",
        "ExchangeCode",
        "RCSExchangeCountryLeaf",
        "AssetState",
        "IsPrimaryRIC",
        "RCSAssetCategoryLeaf",
    ]

    parts = []

    for raw in raw_frames:

        if raw is None or raw.empty:
            continue

        if not {
            "RIC",
            "TickerSymbol",
        }.issubset(raw.columns):
            raise TickerLookupError(
                "LSEG search response omitted RIC "
                "or TickerSymbol columns."
            )

        piece = raw.copy()

        for column in columns:
            if column not in piece.columns:
                piece[column] = pd.NA

        parts.append(
            piece[columns]
        )

    if not parts:
        return pd.DataFrame(
            columns=columns
        )

    result = pd.concat(
        parts,
        ignore_index=True,
    )

    result["RIC"] = (
        result["RIC"]
        .astype("string")
        .str.strip()
    )

    result["TickerSymbol"] = (
        result["TickerSymbol"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    exact_match = (
        result["TickerSymbol"]
        .eq(ticker)
        .fillna(False)
    )

    has_ric = (
        result["RIC"].notna()
        & result["RIC"].ne("").fillna(False)
    )

    result = result.loc[
        exact_match & has_ric
    ].copy()

    # AC is LSEG's active-asset state.
    # If state metadata is unavailable,
    # retain the candidate for further review.
    state = (
        result["AssetState"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    active = (
        state.isna()
        | state.isin(
            ["AC", "ACTIVE"]
        )
    )

    result = result.loc[
        active
    ].copy()

    result["_primary"] = (
        result["IsPrimaryRIC"]
        .map(_is_true)
    )

    country = (
        result["RCSExchangeCountryLeaf"]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    result["_usa"] = country.isin(
        [
            "united states",
            "united states of america",
            "usa",
            "us",
        ]
    )

    # A security may appear in both Equity Quotes
    # and Fund Quotes. Keep only one row per RIC.
    result = (
        result.sort_values(
            ["_usa", "_primary", "RIC"],
            ascending=[False, False, True],
        )
        .drop_duplicates("RIC")
        .reset_index(drop=True)
    )

    return result


def choose_ric(
    ticker: str,
    candidates: pd.DataFrame,
) -> str:
    """Select a uniquely identifiable US instrument."""

    if candidates.empty:
        raise TickerLookupError(
            f"No exact active LSEG match for '{ticker}'. "
            "Enter a full LSEG RIC."
        )

    us_candidates = candidates.loc[
        candidates["_usa"]
    ]

    primary_us = us_candidates.loc[
        us_candidates["_primary"]
    ]

    if len(primary_us) == 1:
        return str(
            primary_us.iloc[0]["RIC"]
        )

    if len(us_candidates) == 1:
        return str(
            us_candidates.iloc[0]["RIC"]
        )

    # Metadata can be incomplete. Permit a single
    # otherwise unambiguous result when its exchange
    # country is not supplied.
    if len(candidates) == 1:

        only = candidates.iloc[0]

        if pd.isna(
            only["RCSExchangeCountryLeaf"]
        ):
            return str(
                only["RIC"]
            )

    # Ambiguous tickers must not be guessed.
    suggestions = []

    for _, row in candidates.head(12).iterrows():

        exchange = row["ExchangeName"]

        if pd.isna(exchange):
            exchange = "exchange unknown"

        suggestions.append(
            f"{row['RIC']} ({exchange})"
        )

    raise TickerLookupError(
        f"Ticker '{ticker}' has multiple or "
        "unconfirmed US listings. "
        "Enter a full RIC from these candidates: "
        + "; ".join(suggestions)
    )


def search_lseg_ticker(
    ticker: str,
) -> pd.DataFrame:
    """Search active equity and fund quote records in LSEG."""

    import lseg.data as ld

    frames = []
    errors = []
    successful_searches = 0

    ld.open_session()

    try:

        views = (
            ld.discovery.Views.EQUITY_QUOTES,
            ld.discovery.Views.FUND_QUOTES,
        )

        for view in views:

            # Some entitlements or views may not support
            # all metadata fields. Use a minimal fallback.
            for fields in (
                RICH_SELECT,
                BASIC_SELECT,
            ):

                try:
                    response = ld.discovery.search(
                        view=view,
                        filter=(
                            f"TickerSymbol eq '{ticker}'"
                        ),
                        select=fields,
                        top=100,
                    )

                    if (
                        response is not None
                        and not isinstance(
                            response,
                            pd.DataFrame,
                        )
                    ):
                        raise TypeError(
                            "Expected a pandas DataFrame "
                            "from LSEG Discovery."
                        )

                except Exception as exc:

                    errors.append(
                        f"{view}: "
                        f"{type(exc).__name__}: {exc}"
                    )

                    continue

                successful_searches += 1

                if response is not None:
                    frames.append(response)

                break

    finally:
        ld.close_session()

    if successful_searches == 0:
        raise TickerLookupError(
            "LSEG instrument discovery failed. "
            + " | ".join(errors[-2:])
        )

    if errors and not any(
        not frame.empty
        for frame in frames
    ):
        raise TickerLookupError(
            "No usable instrument matches and "
            "at least one discovery request failed. "
            + " | ".join(errors[-2:])
        )

    return normalize_candidates(
        ticker,
        frames,
    )


def resolve_lseg_ric(
    user_symbol: str,
) -> str:
    """Resolve a friendly ticker or accept an explicit RIC."""

    symbol = user_symbol.strip().upper()

    if not symbol:
        raise TickerLookupError(
            "Please enter a ticker symbol."
        )

    # Use confirmed project mappings.
    if symbol in PINNED_RICS:
        return PINNED_RICS[symbol]

    # Already specified as a full LSEG RIC.
    if "." in symbol or "^" in symbol:
        return symbol

    if not SYMBOL_PATTERN.fullmatch(
        symbol
    ):
        raise TickerLookupError(
            "Unsupported ticker format. "
            "Enter a full LSEG RIC if needed."
        )

    candidates = search_lseg_ticker(
        symbol
    )

    return choose_ric(
        symbol,
        candidates,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "symbols",
        nargs="+",
        help="Tickers or full LSEG RICs",
    )

    args = parser.parse_args()

    print(
        "\n--- LSEG TICKER RESOLUTION ---"
    )

    for symbol in args.symbols:

        try:
            ric = resolve_lseg_ric(
                symbol
            )

            print(
                f"{symbol.upper()} -> {ric}"
            )

        except TickerLookupError as exc:

            print(
                f"{symbol.upper()} -> ERROR: {exc}"
            )


if __name__ == "__main__":
    main()
