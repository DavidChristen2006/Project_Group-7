
"""Fill missing ETF ISINs using exchange-specific LSEG fund quotes.

Creates an enriched ETF universe without modifying the original.
Requires data_pipeline/verify_etf_rics.py.
"""

from pathlib import Path

import lseg.data as ld
import pandas as pd

from lseg.data.content import symbol_conversion

from data_pipeline.verify_etf_rics import (
    clean,
    ric_ticker,
    venue_matches,
)


ROOT = (Path(__file__).resolve().parents[1] / "data/processed/etf_universe")
SOURCE = ROOT / "etf_instruments.parquet"
OUTPUT = ROOT / "isin_bootstrap"

SEARCH_FIELDS = [
    "RIC,TickerSymbol,IssueISIN,ExchangeName,"
    "ExchangeCode,Currency,DocumentTitle",

    "RIC,TickerSymbol,IssueISIN,ExchangeName,"
    "Currency,DocumentTitle",

    "RIC,TickerSymbol,IssueISIN,ExchangeName,"
    "DocumentTitle",
]


def discover(ticker):
    """Find possible fund listings by exact ticker."""

    errors = []

    for fields in SEARCH_FIELDS:
        try:
            escaped = ticker.replace("'", "''")

            df = ld.discovery.search(
                view=ld.discovery.Views.FUND_QUOTES,
                filter=f"TickerSymbol eq '{escaped}'",
                select=fields,
                top=200,
            )

            if isinstance(df, pd.DataFrame):
                if len(df) >= 200:
                    print(
                        f"WARNING: {ticker} search reached "
                        "the 200-result limit."
                    )

                return df.copy(), ""

        except Exception as error:
            errors.append(
                f"{type(error).__name__}: {error}"
            )

    return pd.DataFrame(), " | ".join(errors)


def currency_lookup(rics):
    """Retrieve the actual quote currency of candidate RICs."""

    currencies = {}

    for offset in range(0, len(rics), 20):
        batch = rics[offset:offset + 20]

        try:
            df = ld.get_data(
                universe=batch,
                fields=["TR.PriceClose.Currency"],
            )

            columns = [
                column
                for column in df.columns
                if (
                    column != "Instrument"
                    and "currency" in str(column).lower()
                )
            ]

            if (
                "Instrument" not in df.columns
                or not columns
            ):
                continue

            for _, row in df.iterrows():
                ric = clean(row["Instrument"])

                currencies[ric] = clean(
                    row[columns[0]]
                ).upper()

        except Exception as error:
            print(
                "Currency lookup warning:",
                type(error).__name__,
                str(error),
            )

    return currencies


def isin_lookup(rics):
    """Verify RIC-to-ISIN mappings through Symbol Conversion."""

    result = {}

    for offset in range(0, len(rics), 20):
        batch = rics[offset:offset + 20]

        try:
            response = symbol_conversion.Definition(
                symbols=batch,
                from_symbol_type=(
                    symbol_conversion.SymbolTypes.RIC
                ),
                to_symbol_types=[
                    symbol_conversion.SymbolTypes.ISIN
                ],
            ).get_data()

            frame = response.data.df

            for ric in batch:
                if ric not in frame.index:
                    result[ric] = ""
                    continue

                values = (
                    frame.loc[[ric], "IssueISIN"]
                    .dropna()
                    .astype("string")
                    .str.upper()
                    .unique()
                    .tolist()
                )

                result[ric] = (
                    values[0]
                    if len(values) == 1
                    else ""
                )

        except Exception as error:
            print(
                "Conversion warning:",
                type(error).__name__,
                str(error),
            )

            # Isolate failures rather than discarding the batch.
            for ric in batch:
                try:
                    response = symbol_conversion.Definition(
                        symbols=[ric],
                        from_symbol_type=(
                            symbol_conversion.SymbolTypes.RIC
                        ),
                        to_symbol_types=[
                            symbol_conversion.SymbolTypes.ISIN
                        ],
                    ).get_data()

                    frame = response.data.df

                    result[ric] = (
                        clean(
                            frame.loc[ric, "IssueISIN"]
                        ).upper()
                        if (
                            ric in frame.index
                            and "IssueISIN" in frame.columns
                        )
                        else ""
                    )

                except Exception:
                    result[ric] = ""

    return result


def main():
    """Run the module entry point and its workflow.
    """
    original = pd.read_parquet(SOURCE).copy()

    required = [
        "Ticker",
        "Venue",
        "Currency",
        "ISIN",
        "LSEG RIC",
        "RIC Verified",
    ]

    missing = [
        column
        for column in required
        if column not in original.columns
    ]

    if missing:
        raise ValueError(
            f"ETF universe missing columns: {missing}"
        )

    if "Listing ID" not in original.columns:
        original["Listing ID"] = (
            original["Ticker"].astype("string")
            + "|"
            + original["Venue"].astype("string")
        )

    if (
        original["Listing ID"].isna().any()
        or original["Listing ID"].duplicated().any()
    ):
        raise ValueError(
            "ETF listing IDs must be present and unique."
        )

    decisions = []
    candidate_records = []
    search_cache = {}

    print("\n--- ETF MISSING-ISIN BOOTSTRAP ---")
    print("ETF listings:", len(original))
    print("Session: configured default LSEG session")

    ld.open_session()

    try:
        for _, listing in original.iterrows():
            ticker = clean(
                listing["Ticker"]
            ).upper()

            venue = clean(
                listing["Venue"]
            ).upper()

            currency = clean(
                listing["Currency"]
            ).upper()

            existing_isin = clean(
                listing["ISIN"]
            ).upper()

            listing_id = clean(
                listing["Listing ID"]
            )

            # Existing ISINs are retained.
            if existing_isin:
                decisions.append({
                    "Listing ID": listing_id,
                    "Resolved ISIN": existing_isin,
                    "ISIN Discovery Status": (
                        "SOURCE_ISIN_PRESENT"
                    ),
                })

                print(
                    f"{ticker}: existing ISIN retained"
                )

                continue

            # Avoid repeating searches for identical tickers.
            if ticker not in search_cache:
                search_cache[ticker] = discover(ticker)

            found, search_error = search_cache[ticker]

            print(
                f"\n{ticker} | {venue} | {currency}"
                f" | Search results: {len(found)}"
            )

            candidates = []

            for _, hit in found.iterrows():
                ric = clean(
                    hit.get("RIC")
                )

                search_ticker = clean(
                    hit.get("TickerSymbol")
                ).upper()

                isin = clean(
                    hit.get("IssueISIN")
                ).upper()

                exchange_name = clean(
                    hit.get("ExchangeName")
                )

                exchange_code = clean(
                    hit.get("ExchangeCode")
                )

                search_currency = clean(
                    hit.get("Currency")
                ).upper()

                ticker_match = (
                    search_ticker == ticker
                    if search_ticker
                    else ric_ticker(ric) == ticker
                )

                venue_match = venue_matches(
                    venue,
                    exchange_name,
                    exchange_code,
                )

                basic_match = bool(
                    ric
                    and isin
                    and ticker_match
                    and venue_match
                    and "^" not in ric
                )

                candidates.append({
                    "Listing ID": listing_id,
                    "Ticker": ticker,
                    "Venue": venue,
                    "Expected Currency": currency,
                    "RIC": ric,
                    "TickerSymbol": search_ticker,
                    "ISIN": isin,
                    "ExchangeName": exchange_name,
                    "ExchangeCode": exchange_code,
                    "Search Currency": search_currency,
                    "Ticker Match": bool(ticker_match),
                    "Venue Match": bool(venue_match),
                    "Eligible Basic": basic_match,
                })

            shortlist = [
                candidate
                for candidate in candidates
                if candidate["Eligible Basic"]
            ]

            rics = sorted({
                candidate["RIC"]
                for candidate in shortlist
            })

            currencies = (
                currency_lookup(rics)
                if rics else {}
            )

            converted_isins = (
                isin_lookup(rics)
                if rics else {}
            )

            for candidate in candidates:
                ric = candidate["RIC"]

                # Prefer the currency directly associated
                # with the quote. Fall back to Search.
                quote_currency = (
                    currencies.get(ric)
                    or candidate["Search Currency"]
                )

                roundtrip_isin = (
                    converted_isins.get(ric, "")
                )

                currency_match = bool(
                    quote_currency
                    and quote_currency == currency
                )

                isin_match = bool(
                    roundtrip_isin
                    and roundtrip_isin == candidate["ISIN"]
                )

                candidate["Quote Currency"] = (
                    quote_currency
                )

                candidate["Roundtrip ISIN"] = (
                    roundtrip_isin
                )

                candidate["Currency Match"] = (
                    currency_match
                )

                candidate["ISIN Match"] = isin_match

                candidate["Eligible Verified"] = bool(
                    candidate["Eligible Basic"]
                    and currency_match
                    and isin_match
                )

                candidate_records.append(candidate)

            verified = [
                candidate
                for candidate in candidates
                if candidate["Eligible Verified"]
            ]

            unique_isins = {
                candidate["ISIN"]
                for candidate in verified
            }

            if len(unique_isins) == 1:
                resolved = next(iter(unique_isins))
                status = "ISIN_RESOLVED"

            elif len(unique_isins) > 1:
                resolved = ""
                status = "REVIEW_MULTIPLE_ISINS"

            elif search_error and not candidates:
                resolved = ""
                status = "REVIEW_SEARCH_ERROR"

            elif not candidates:
                resolved = ""
                status = "REVIEW_NO_SEARCH_RESULTS"

            elif not shortlist:
                resolved = ""
                status = "REVIEW_NO_VENUE_TICKER_MATCH"

            else:
                resolved = ""
                status = "REVIEW_CURRENCY_OR_CONVERSION"

            decisions.append({
                "Listing ID": listing_id,
                "Resolved ISIN": resolved,
                "ISIN Discovery Status": status,
            })

            print(
                f"Result: {status}"
                f" | ISIN: {resolved or '-'}"
            )

    finally:
        ld.close_session()

    decision_df = pd.DataFrame(decisions)

    enriched = original.merge(
        decision_df,
        on="Listing ID",
        how="left",
        validate="one_to_one",
    )

    current_isin = (
        enriched["ISIN"]
        .astype("string")
        .str.strip()
    )

    needs_isin = (
        current_isin.isna()
        | current_isin.eq("")
    )

    enriched.loc[
        needs_isin, "ISIN"
    ] = (
        enriched.loc[
            needs_isin, "Resolved ISIN"
        ]
        .replace("", pd.NA)
    )

    OUTPUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    enriched.to_parquet(
        OUTPUT / "etf_instruments_with_isin.parquet",
        index=False,
    )

    enriched.to_csv(
        OUTPUT / "etf_instruments_with_isin.csv",
        index=False,
    )

    pd.DataFrame(candidate_records).to_csv(
        OUTPUT / "isin_candidates.csv",
        index=False,
    )

    enriched.loc[
        enriched["ISIN"].isna()
    ].to_csv(
        OUTPUT / "isin_review.csv",
        index=False,
    )

    print("\n--- ISIN DISCOVERY SUMMARY ---")

    print(
        enriched["ISIN Discovery Status"]
        .value_counts(dropna=False)
        .to_string()
    )

    print(
        "With ISIN:",
        int(enriched["ISIN"].notna().sum()),
        "/",
        len(enriched),
    )

    print("\nSaved in:", OUTPUT)
    print("Original ETF universe unchanged.")


if __name__ == "__main__":
    main()
