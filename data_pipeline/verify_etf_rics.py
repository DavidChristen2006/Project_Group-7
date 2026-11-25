
"""Discover and verify exchange-specific LSEG RICs for the ETF universe.

Requires configured LSEG session.
Original ETF universe is never overwritten.
"""

import argparse
from pathlib import Path

import lseg.data as ld
import pandas as pd

from lseg.data.content import symbol_conversion


ROOT = (Path(__file__).resolve().parents[1] / "data/processed/etf_universe")
SOURCE = ROOT / "isin_bootstrap" / "etf_instruments_with_isin.parquet"
OUTPUT = ROOT / "ric_verification"

SEARCH_FIELDS = [
    "RIC,IssueISIN,ExchangeName,ExchangeCode,AssetStateName,DocumentTitle",
    "RIC,IssueISIN,ExchangeName,DocumentTitle",
]

VENUE_WORDS = {
    "US": (
        "nyse", "new york stock exchange", "nasdaq",
        "arca", "cboe", "bats", "iex",
        "american stock exchange",
    ),
    "LSE": (
        "london stock exchange", "london",
    ),
    "SIX": (
        "six swiss exchange", "six swiss",
        "swiss exchange", "swx",
    ),
    "XETRA": (
        "xetra",
    ),
    "EURONEXT_AMSTERDAM": (
        "euronext amsterdam", "amsterdam",
    ),
}

VENUE_CODES = {
    "US": {
        "NYS", "NAS", "ASE", "PCQ", "PCX",
        "BTS", "BZX", "GSM", "NSM", "NMS",
    },
    "LSE": {"LSE", "LON"},
    "SIX": {"SWX", "SWZ"},
    "XETRA": {"ETR", "XET", "XETRA"},
    "EURONEXT_AMSTERDAM": {"AMS", "AEX"},
}

CANDIDATE_COLUMNS = [
    "Listing ID",
    "Ticker",
    "Venue",
    "Currency",
    "Expected ISIN",
    "Candidate RIC",
    "Search ISIN",
    "Exchange Name",
    "Exchange Code",
    "Asset State",
    "Document Title",
    "Quote Currency",
    "Roundtrip ISIN",
    "Ticker Match",
    "Venue Match",
    "Search ISIN Match",
    "Roundtrip ISIN Match",
    "Currency Match",
    "Verification Eligible",
]


def clean(value):
    """Perform the clean operation for this research workflow.

    Args:
        value: Input to the clean operation.

    Returns:
        Computed result or status for the caller.
    """
    if value is None or pd.isna(value):
        return ""

    result = str(value).strip()

    if result.lower() in {
        "nan", "none", "null", "<na>"
    }:
        return ""

    return result


def ric_ticker(ric):
    """Perform the ric ticker operation for this research workflow.

    Args:
        ric: Input to the ric_ticker operation.

    Returns:
        Computed result or status for the caller.
    """
    return (
        clean(ric)
        .split(".", 1)[0]
        .split("^", 1)[0]
        .upper()
    )


def venue_matches(venue, exchange_name, exchange_code):
    """Perform the venue matches operation for this research workflow.

    Args:
        venue: Input to the venue_matches operation.
        exchange_name: Input to the venue_matches operation.
        exchange_code: Input to the venue_matches operation.

    Returns:
        Computed result or status for the caller.
    """
    venue = clean(venue).upper()
    name = clean(exchange_name).lower()
    code = clean(exchange_code).upper()

    return (
        any(
            word in name
            for word in VENUE_WORDS.get(venue, ())
        )
        or code in VENUE_CODES.get(venue, set())
    )


def search_isin(isin):
    """Search LSEG fund quotes, then broader search."""

    views = [
        getattr(ld.discovery.Views, "FUND_QUOTES", None),
        getattr(ld.discovery.Views, "SEARCH_ALL", None),
    ]

    last_error = None

    for view in views:
        if view is None:
            continue

        for fields in SEARCH_FIELDS:
            try:
                df = ld.discovery.search(
                    view=view,
                    filter=f"IssueISIN eq '{isin}'",
                    select=fields,
                    top=200,
                )

                if isinstance(df, pd.DataFrame) and not df.empty:
                    if len(df) >= 200:
                        print(
                            f"WARNING: Search reached 200 "
                            f"rows for {isin}."
                        )

                    return df.copy()

            except Exception as error:
                last_error = error

    if last_error is not None:
        print(
            f"Search warning for {isin}: "
            f"{type(last_error).__name__}: {last_error}"
        )

    return pd.DataFrame()


def search_quotes(universe):
    """Find possible RICs for each intended ETF listing."""

    cache = {}
    records = []

    rows = universe.to_dict("records")

    for number, row in enumerate(rows, start=1):
        ticker = clean(row["Ticker"]).upper()
        venue = clean(row["Venue"]).upper()
        currency = clean(row["Currency"]).upper()
        isin = clean(row["ISIN"]).upper()
        listing_id = clean(row["Listing ID"])

        print(
            f"[{number}/{len(rows)}] "
            f"{ticker} | {venue} | {currency} | {isin}"
        )

        if not isin:
            continue

        # Same ISIN can occur across multiple listings.
        if isin not in cache:
            cache[isin] = search_isin(isin)

        df = cache[isin]

        for _, item in df.iterrows():
            ric = clean(item.get("RIC"))

            if not ric:
                continue

            records.append({
                "Listing ID": listing_id,
                "Ticker": ticker,
                "Venue": venue,
                "Currency": currency,
                "Expected ISIN": isin,
                "Candidate RIC": ric,
                "Search ISIN": clean(
                    item.get("IssueISIN")
                ).upper(),
                "Exchange Name": clean(
                    item.get("ExchangeName")
                ),
                "Exchange Code": clean(
                    item.get("ExchangeCode")
                ),
                "Asset State": clean(
                    item.get("AssetStateName")
                ),
                "Document Title": clean(
                    item.get("DocumentTitle")
                ),
            })

    return pd.DataFrame(
        records,
        columns=CANDIDATE_COLUMNS[:11],
    )


def conversion_lookup(rics):
    """Verify RIC -> ISIN through Symbol Conversion."""

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
                    .unique()
                    .tolist()
                )

                result[ric] = (
                    clean(values[0]).upper()
                    if len(values) == 1
                    else ""
                )

        except Exception as error:
            print(
                "Symbol Conversion warning:",
                type(error).__name__,
                str(error),
            )

            # Retry individually if batch conversion fails.
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
                        clean(frame["IssueISIN"].iloc[0]).upper()
                        if len(frame) == 1
                        else ""
                    )

                except Exception:
                    result[ric] = ""

    return result


def quote_currency_lookup(rics):
    """Retrieve the exchange quote currency."""

    result = {}

    for offset in range(0, len(rics), 20):
        batch = rics[offset:offset + 20]

        try:
            df = ld.get_data(
                universe=batch,
                fields=["TR.PriceClose.Currency"],
            )

            currency_columns = [
                column
                for column in df.columns
                if (
                    column != "Instrument"
                    and "currency" in str(column).lower()
                )
            ]

            if (
                "Instrument" not in df.columns
                or not currency_columns
            ):
                raise ValueError(
                    f"Currency field missing: "
                    f"{df.columns.tolist()}"
                )

            for _, row in df.iterrows():
                ric = clean(row["Instrument"])

                if ric:
                    result[ric] = clean(
                        row[currency_columns[0]]
                    )

        except Exception as error:
            print(
                "Currency lookup warning:",
                type(error).__name__,
                str(error),
            )

        for ric in batch:
            result.setdefault(ric, "")

    return result


def verify_candidates(candidates):
    """Evaluate ticker, venue, ISIN and currency matches."""

    if candidates.empty:
        return candidates.reindex(
            columns=CANDIDATE_COLUMNS
        )

    candidates = candidates.drop_duplicates(
        subset=[
            "Listing ID",
            "Candidate RIC",
            "Exchange Name",
            "Exchange Code",
        ]
    ).copy()

    candidates["Ticker Match"] = candidates.apply(
        lambda row: (
            ric_ticker(row["Candidate RIC"])
            == row["Ticker"]
        ),
        axis=1,
    )

    candidates["Venue Match"] = candidates.apply(
        lambda row: venue_matches(
            row["Venue"],
            row["Exchange Name"],
            row["Exchange Code"],
        ),
        axis=1,
    )

    candidates["Search ISIN Match"] = (
        candidates["Search ISIN"].str.upper()
        == candidates["Expected ISIN"].str.upper()
    )

    relevant = (
        candidates["Ticker Match"]
        & candidates["Venue Match"]
    )

    rics = sorted(
        candidates.loc[
            relevant, "Candidate RIC"
        ].unique().tolist()
    )

    print(
        f"\nChecking identifiers and currency "
        f"for {len(rics)} shortlisted RICs..."
    )

    isin_by_ric = conversion_lookup(rics)
    currency_by_ric = quote_currency_lookup(rics)

    candidates["Quote Currency"] = (
        candidates["Candidate RIC"]
        .map(currency_by_ric)
        .fillna("")
    )

    candidates["Roundtrip ISIN"] = (
        candidates["Candidate RIC"]
        .map(isin_by_ric)
        .fillna("")
    )

    candidates["Roundtrip ISIN Match"] = (
        candidates["Roundtrip ISIN"].str.upper()
        == candidates["Expected ISIN"].str.upper()
    )

    # Keep currency matching strict.
    # GBp and GBP are not interchangeable price units.
    candidates["Currency Match"] = (
        candidates["Quote Currency"].str.upper()
        == candidates["Currency"].str.upper()
    )

    candidates["Verification Eligible"] = (
        candidates["Ticker Match"]
        & candidates["Venue Match"]
        & candidates["Search ISIN Match"]
        & candidates["Roundtrip ISIN Match"]
        & candidates["Currency Match"]
        & ~candidates["Candidate RIC"].str.contains(
            r"\^",
            regex=True,
        )
    )

    return candidates[CANDIDATE_COLUMNS].copy()


def assemble(universe, candidates):
    """Choose only unambiguous, fully verified RICs."""

    decisions = []

    for _, row in universe.iterrows():
        listing_id = clean(row["Listing ID"])

        entries = candidates.loc[
            candidates["Listing ID"] == listing_id
        ]

        strict = entries.loc[
            entries["Verification Eligible"]
        ]

        basic = entries.loc[
            entries["Ticker Match"]
            & entries["Venue Match"]
            & entries["Search ISIN Match"]
        ]

        chosen = ""

        if not clean(row["ISIN"]):
            status = "REVIEW_MISSING_ISIN"

        elif len(strict) == 1:
            status = "VERIFIED"
            chosen = clean(
                strict.iloc[0]["Candidate RIC"]
            )

        elif len(strict) > 1:
            status = "REVIEW_AMBIGUOUS"

        elif len(basic) == 1:
            status = "REVIEW_ISIN_OR_CURRENCY"
            chosen = clean(
                basic.iloc[0]["Candidate RIC"]
            )

        elif len(basic) > 1:
            status = "REVIEW_AMBIGUOUS"

        else:
            status = "REVIEW_NO_LISTING_MATCH"

        decisions.append({
            "Listing ID": listing_id,
            "Proposed RIC": chosen,
            "RIC Verification Status": status,
            "RIC Candidate Count": len(entries),
        })

    decision = pd.DataFrame(decisions)

    updated = universe.merge(
        decision,
        on="Listing ID",
        how="left",
        validate="one_to_one",
    )

    updated["Original LSEG RIC"] = updated["LSEG RIC"]

    # Only verified RICs enter the authoritative column.
    updated["LSEG RIC"] = updated.apply(
        lambda row: (
            row["Proposed RIC"]
            if row["RIC Verification Status"] == "VERIFIED"
            else pd.NA
        ),
        axis=1,
    )

    updated["RIC Verified"] = (
        updated["RIC Verification Status"] == "VERIFIED"
    )

    # Prevent the same RIC from being assigned to two
    # different ETF listing records.
    counts = (
        updated.loc[
            updated["RIC Verified"], "LSEG RIC"
        ].value_counts()
    )

    duplicates = set(
        counts.loc[counts > 1].index
    )

    bad = updated["LSEG RIC"].isin(duplicates)

    updated.loc[
        bad, "RIC Verification Status"
    ] = "REVIEW_DUPLICATE_RIC"

    updated.loc[
        bad, "RIC Verified"
    ] = False

    updated.loc[
        bad, "LSEG RIC"
    ] = pd.NA

    return updated


def main():
    """Run the module entry point and its workflow.
    """
    parser = argparse.ArgumentParser()

    mode = parser.add_mutually_exclusive_group(
        required=True
    )

    mode.add_argument(
        "--limit",
        type=int,
        help="Pilot first N ETF listings",
    )

    mode.add_argument(
        "--all",
        action="store_true",
        help="Verify all ETF listings",
    )

    mode.add_argument(
        "--tickers",
        nargs="+",
        help="Pilot selected ETF tickers",
    )

    args = parser.parse_args()

    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive.")

    universe = pd.read_parquet(SOURCE)

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
        if column not in universe.columns
    ]

    if missing:
        raise ValueError(
            f"ETF universe missing columns: {missing}"
        )

    if "Listing ID" not in universe.columns:
        universe["Listing ID"] = (
            universe["Ticker"].astype("string")
            + "|"
            + universe["Venue"].astype("string")
        )

    if (
        universe["Listing ID"].isna().any()
        or universe["Listing ID"].duplicated().any()
    ):
        raise ValueError(
            "ETF listing IDs must be present and unique."
        )

    if args.limit is not None:
        universe = universe.head(args.limit).copy()

    if args.tickers:
        requested = {
            ticker.upper()
            for ticker in args.tickers
        }

        present = set(
            universe["Ticker"]
            .astype("string")
            .str.upper()
        )

        unknown = requested - present

        if unknown:
            raise ValueError(
                f"Unknown ETF tickers: {sorted(unknown)}"
            )

        universe = universe.loc[
            universe["Ticker"]
            .astype("string")
            .str.upper()
            .isin(requested)
        ].copy()

    destination = (
        OUTPUT
        if args.all
        else OUTPUT / "pilot"
    )

    destination.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("\n--- ETF RIC VERIFICATION ---")
    print("ETF listings:", len(universe))
    print("Output:", destination)
    print("Session: configured default LSEG session")

    ld.open_session()

    try:
        candidates = verify_candidates(
            search_quotes(universe)
        )

        verified = assemble(
            universe,
            candidates,
        )

    finally:
        ld.close_session()

    verified.to_parquet(
        destination / "etf_instruments_verified.parquet",
        index=False,
    )

    verified.to_csv(
        destination / "etf_instruments_verified.csv",
        index=False,
    )

    candidates.to_csv(
        destination / "etf_ric_candidates.csv",
        index=False,
    )

    verified.loc[
        ~verified["RIC Verified"]
    ].to_csv(
        destination / "etf_ric_review.csv",
        index=False,
    )

    print("\n--- ETF RIC VERIFICATION SUMMARY ---")

    print(
        verified["RIC Verification Status"]
        .value_counts(dropna=False)
        .to_string()
    )

    print(
        "Verified:",
        int(verified["RIC Verified"].sum()),
        "/",
        len(verified),
    )

    print("\nListing summary:")

    print(
        verified[
            [
                "Ticker",
                "Venue",
                "Currency",
                "ISIN",
                "Proposed RIC",
                "RIC Verification Status",
            ]
        ].to_string(index=False)
    )

    print("\nSaved in:", destination)

    print(
        "Original ETF universe and collection grid "
        "were not modified."
    )


if __name__ == "__main__":
    main()
