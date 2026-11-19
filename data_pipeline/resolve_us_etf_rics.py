
"""Resolve US ETF RIC ambiguity using verified primary exchanges.

No LSEG API requests required.
Existing verification output is not overwritten.
"""

from pathlib import Path

import pandas as pd


ROOT = Path("data/processed/etf_universe")
SOURCE = (
    ROOT / "ric_verification"
    / "etf_instruments_verified.parquet"
)
CANDIDATES = (
    ROOT / "ric_verification"
    / "etf_ric_candidates.csv"
)
OUTPUT = ROOT / "ric_verification" / "primary_listing"


# Primary listing exchanges established from issuer information.
PRIMARY_EXCHANGE = {
    "ACWI": "NASDAQ",
    "VXUS": "NASDAQ",
    "QQQ": "NASDAQ",
    "QQQM": "NASDAQ",

    "EEM": "NYSE_ARCA",
    "VWO": "NYSE_ARCA",
    "VT": "NYSE_ARCA",
    "EFA": "NYSE_ARCA",
    "VEA": "NYSE_ARCA",
    "XLE": "NYSE_ARCA",
    "XLF": "NYSE_ARCA",
    "XLK": "NYSE_ARCA",
    "DIA": "NYSE_ARCA",
    "IVV": "NYSE_ARCA",
    "SPY": "NYSE_ARCA",
    "VOO": "NYSE_ARCA",
    "IWM": "NYSE_ARCA",
    "ITOT": "NYSE_ARCA",
    "VTI": "NYSE_ARCA",
}


def clean(value):
    if value is None or pd.isna(value):
        return ""

    return str(value).strip()


def primary_quote_match(row, exchange):
    name = clean(row["Exchange Name"]).lower()
    code = clean(row["Exchange Code"]).upper()

    if exchange == "NYSE_ARCA":
        # Exclude the consolidated quote and other venues.
        return name == "nyse arca" and code == "PSE"

    if exchange == "NASDAQ":
        # Exclude consolidated quotes, Nasdaq Basic,
        # and secondary venues trading Nasdaq securities.
        return (
            name.startswith("nasdaq stock exchange")
            and "consolidated" not in name
            and "basic" not in name
        )

    return False


def eligible_candidates(frame, listing):
    checks = [
        "Ticker Match",
        "Venue Match",
        "Search ISIN Match",
        "Roundtrip ISIN Match",
        "Currency Match",
        "Verification Eligible",
    ]

    valid = pd.Series(True, index=frame.index)

    for column in checks:
        valid &= (
            frame[column]
            .astype("string")
            .str.strip()
            .str.lower()
            .eq("true")
        )

    valid &= (
        frame["Ticker"].str.upper()
        == clean(listing["Ticker"]).upper()
    )

    valid &= (
        frame["Expected ISIN"].str.upper()
        == clean(listing["ISIN"]).upper()
    )

    valid &= (
        frame["Search ISIN"].str.upper()
        == clean(listing["ISIN"]).upper()
    )

    valid &= (
        frame["Roundtrip ISIN"].str.upper()
        == clean(listing["ISIN"]).upper()
    )

    valid &= (
        frame["Quote Currency"].str.upper()
        == clean(listing["Currency"]).upper()
    )

    valid &= (
        frame["Asset State"].str.lower()
        == "active"
    )

    valid &= ~frame["Candidate RIC"].str.contains(
        "^",
        regex=False,
    )

    return frame.loc[valid].copy()


def main():
    universe = pd.read_parquet(SOURCE).copy()

    candidates = pd.read_csv(
        CANDIDATES,
        dtype="string",
        keep_default_na=False,
    ).fillna("")

    required_universe = {
        "Listing ID",
        "Ticker",
        "Venue",
        "Currency",
        "ISIN",
        "LSEG RIC",
        "Proposed RIC",
        "RIC Verification Status",
        "RIC Verified",
    }

    required_candidates = {
        "Listing ID",
        "Ticker",
        "Expected ISIN",
        "Candidate RIC",
        "Exchange Name",
        "Exchange Code",
        "Asset State",
        "Search ISIN",
        "Roundtrip ISIN",
        "Quote Currency",
        "Ticker Match",
        "Venue Match",
        "Search ISIN Match",
        "Roundtrip ISIN Match",
        "Currency Match",
        "Verification Eligible",
    }

    if required_universe - set(universe.columns):
        raise ValueError(
            "Missing ETF universe columns: "
            f"{sorted(required_universe - set(universe.columns))}"
        )

    if required_candidates - set(candidates.columns):
        raise ValueError(
            "Missing candidate columns: "
            f"{sorted(required_candidates - set(candidates.columns))}"
        )

    if (
        universe["Listing ID"].isna().any()
        or universe["Listing ID"].duplicated().any()
    ):
        raise ValueError("Invalid or duplicate ETF listing IDs.")

    universe["Primary Listing Exchange"] = pd.NA
    universe["RIC Selection Method"] = pd.NA
    universe["Consolidated RIC"] = pd.NA

    audit = []

    print("\n--- ETF PRIMARY LISTING RESOLUTION ---")
    print("ETF listings:", len(universe))

    for index, listing in universe.iterrows():
        ticker = clean(listing["Ticker"]).upper()
        listing_id = clean(listing["Listing ID"])
        previous_status = clean(
            listing["RIC Verification Status"]
        )

        # Keep the six listings already verified.
        if previous_status == "VERIFIED":
            universe.at[
                index, "RIC Selection Method"
            ] = "EXISTING_VERIFICATION"

            print(
                f"{ticker}: retained "
                f"{clean(listing['LSEG RIC'])}"
            )
            continue

        if clean(listing["Venue"]).upper() != "US":
            print(f"{ticker}: non-US listing left for review")
            continue

        exchange = PRIMARY_EXCHANGE.get(ticker)

        universe.at[
            index, "Primary Listing Exchange"
        ] = exchange

        if exchange is None:
            universe.at[
                index, "RIC Verification Status"
            ] = "REVIEW_UNKNOWN_PRIMARY_EXCHANGE"
            continue

        frame = candidates.loc[
            candidates["Listing ID"] == listing_id
        ].copy()

        valid = eligible_candidates(
            frame,
            listing,
        )

        # Record consolidated quotes separately for later
        # historical-price availability comparisons.
        consolidated = valid.loc[
            valid["Exchange Name"]
            .str.contains(
                "consolidated",
                case=False,
                regex=False,
            )
        ]

        consolidated_rics = (
            consolidated["Candidate RIC"]
            .drop_duplicates()
            .tolist()
        )

        if len(consolidated_rics) == 1:
            universe.at[
                index, "Consolidated RIC"
            ] = consolidated_rics[0]

        primary = valid.loc[
            valid.apply(
                lambda row: primary_quote_match(
                    row,
                    exchange,
                ),
                axis=1,
            )
        ].copy()

        primary = primary.drop_duplicates(
            subset=["Candidate RIC"]
        )

        choices = primary["Candidate RIC"].tolist()

        if len(choices) == 1:
            ric = choices[0]

            universe.at[index, "LSEG RIC"] = ric
            universe.at[index, "Proposed RIC"] = ric

            universe.at[
                index, "RIC Verified"
            ] = True

            universe.at[
                index, "RIC Verification Status"
            ] = "VERIFIED"

            universe.at[
                index, "RIC Selection Method"
            ] = "PRIMARY_EXCHANGE"

            status = "VERIFIED"

        elif len(choices) > 1:
            status = "REVIEW_MULTIPLE_PRIMARY_QUOTES"

        else:
            status = "REVIEW_NO_PRIMARY_QUOTE"

        if status != "VERIFIED":
            universe.at[index, "LSEG RIC"] = pd.NA
            universe.at[index, "RIC Verified"] = False
            universe.at[
                index, "RIC Verification Status"
            ] = status

        audit.append({
            "Ticker": ticker,
            "Listing ID": listing_id,
            "Expected Primary Exchange": exchange,
            "Eligible Candidates": len(valid),
            "Primary Candidates": len(choices),
            "Primary RIC Candidates": ", ".join(choices),
            "Selection Status": status,
        })

        print(
            f"{ticker}: {status}"
            f" | Primary RIC: {', '.join(choices) or '-'}"
            f" | Exchange: {exchange}"
        )

    # Verify that no single RIC was approved for multiple
    # ETF listing records.
    approved = universe.loc[
        universe["RIC Verified"] == True
    ].copy()

    duplicate_rics = (
        approved["LSEG RIC"]
        .dropna()
        .loc[
            lambda series: series.duplicated(keep=False)
        ]
        .unique()
        .tolist()
    )

    if duplicate_rics:
        affected = universe["LSEG RIC"].isin(
            duplicate_rics
        )

        universe.loc[
            affected, "RIC Verification Status"
        ] = "REVIEW_DUPLICATE_RIC"

        universe.loc[
            affected, "RIC Verified"
        ] = False

        universe.loc[
            affected, "LSEG RIC"
        ] = pd.NA

    OUTPUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    universe.to_parquet(
        OUTPUT / "etf_instruments_verified.parquet",
        index=False,
    )

    universe.to_csv(
        OUTPUT / "etf_instruments_verified.csv",
        index=False,
    )

    universe.loc[
        universe["RIC Verified"] != True
    ].to_csv(
        OUTPUT / "etf_ric_review.csv",
        index=False,
    )

    pd.DataFrame(audit).to_csv(
        OUTPUT / "primary_listing_audit.csv",
        index=False,
    )

    print("\n--- FINAL ETF RIC SUMMARY ---")

    print(
        universe["RIC Verification Status"]
        .value_counts(dropna=False)
        .to_string()
    )

    print(
        "\nVerified:",
        int(universe["RIC Verified"].sum()),
        "/",
        len(universe),
    )

    print("\nFinal listings:")

    print(
        universe[
            [
                "Ticker",
                "Venue",
                "Currency",
                "LSEG RIC",
                "RIC Verification Status",
            ]
        ].to_string(index=False)
    )

    print("\nSaved in:", OUTPUT)
    print("Previous ETF verification files unchanged.")


if __name__ == "__main__":
    main()
