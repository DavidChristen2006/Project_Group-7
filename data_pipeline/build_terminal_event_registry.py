
"""Create an independently sourced terminal-event evidence registry.

This script:
- Uses existing local corporate-action review data.
- Adds six events documented by official sources.
- Distinguishes known transaction terms from unresolved payoffs.
- Does not modify historical prices, features, or outcome labels.
- Refuses to overwrite previously generated output files.
"""

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

REVIEW_DIR = (
    ROOT
    / "data"
    / "processed"
    / "research_audit"
    / "corporate_action_review"
)

INPUT_FILE = (
    REVIEW_DIR
    / "reconciliation_review"
    / "terminal_corporate_action_review_queue.csv"
)

OUTPUT_DIR = REVIEW_DIR / "terminal_event_registry_v1"


# Independently documented historical events.
#
# These records establish event facts, NOT complete realized
# 63-day shareholder returns or option hedge payoffs.
#
# A missing consideration amount means UNKNOWN, never zero.

EVENTS = [
    {
        "Underlying RIC": "TWTR.N^J22",
        "Event Type": "CASH_MERGER",
        "Documented Event Date": "2022-10-27",
        "Cash Per Old Share": 54.20,
        "Successor Shares Per Old Share": None,
        "Successor Company": "",
        "Other Consideration": "",
        "Outcome Readiness": "CASH_TERMS_VERIFIED",
        "Source Type": "SEC_FORM_8K",
        "Source URL": (
            "https://www.sec.gov/Archives/edgar/data/"
            "1418091/000119312522272772/d411753d8k.htm"
        ),
        "Remaining Work": (
            "Verify settlement timing and the cash-holding "
            "convention for the remaining investment horizon."
        ),
    },
    {
        "Underlying RIC": "ABMD.OQ^L22",
        "Event Type": "CASH_MERGER_WITH_CVR",
        "Documented Event Date": "2022-12-22",
        "Cash Per Old Share": 380.00,
        "Successor Shares Per Old Share": None,
        "Successor Company": "Johnson & Johnson",
        "Other Consideration": (
            "One nontradeable contingent value right per share; "
            "potential future payments are not valued here."
        ),
        "Outcome Readiness": "CASH_VERIFIED_CVR_UNVALUED",
        "Source Type": "COMPANY_COMPLETION_RELEASE",
        "Source URL": (
            "https://www.jnj.com/media-center/"
            "press-releases/"
            "johnson-johnson-completes-acquisition-of-abiomed"
        ),
        "Remaining Work": (
            "Specify treatment of the unpriced CVR and "
            "post-acquisition cash balance."
        ),
    },
    {
        "Underlying RIC": "NFX.N^B19",
        "Event Type": "STOCK_FOR_STOCK_MERGER",
        "Documented Event Date": "2019-02-13",
        "Cash Per Old Share": None,
        "Successor Shares Per Old Share": 2.6719,
        "Successor Company": "Encana Corporation",
        "Other Consideration": (
            "Cash in lieu of fractional acquirer shares."
        ),
        "Outcome Readiness": "EXCHANGE_RATIO_VERIFIED",
        "Source Type": "SEC_FORM_8K",
        "Source URL": (
            "https://www.sec.gov/Archives/edgar/data/"
            "1157806/000119312519044440/d701379d8k.htm"
        ),
        "Remaining Work": (
            "Map the historical Encana security, obtain "
            "post-merger prices, and account for later "
            "successor corporate actions."
        ),
    },
    {
        "Underlying RIC": "SIVB.OQ^C23",
        "Event Type": "BANK_RECEIVERSHIP",
        "Documented Event Date": "2023-03-10",
        "Cash Per Old Share": None,
        "Successor Shares Per Old Share": None,
        "Successor Company": "",
        "Other Consideration": (
            "FDIC protection of depositors did not extend "
            "to shareholders."
        ),
        "Outcome Readiness": "EQUITY_RECOVERY_UNVERIFIED",
        "Source Type": "FDIC_RELEASE",
        "Source URL": (
            "https://www.fdic.gov/news/"
            "press-releases/2023/pr23019.html"
        ),
        "Remaining Work": (
            "Establish appropriate equity recovery and "
            "trading/valuation treatment. Do not assume "
            "that an unknown payoff equals zero."
        ),
    },
    {
        "Underlying RIC": "SBNY.OQ^C23",
        "Event Type": "BANK_RECEIVERSHIP",
        "Documented Event Date": "2023-03-12",
        "Cash Per Old Share": None,
        "Successor Shares Per Old Share": None,
        "Successor Company": "",
        "Other Consideration": (
            "FDIC depositor protection; shareholders "
            "were not protected."
        ),
        "Outcome Readiness": "EQUITY_RECOVERY_UNVERIFIED",
        "Source Type": "FDIC_JOINT_STATEMENT",
        "Source URL": (
            "https://www.fdic.gov/news/"
            "press-releases/2023/pr23017.html"
        ),
        "Remaining Work": (
            "Verify subsequent shareholder trading, "
            "recoveries, and appropriate horizon valuation."
        ),
    },
    {
        "Underlying RIC": "FRC.N^E23",
        "Event Type": "BANK_RECEIVERSHIP",
        "Documented Event Date": "2023-05-01",
        "Cash Per Old Share": None,
        "Successor Shares Per Old Share": None,
        "Successor Company": "",
        "Other Consideration": (
            "JPMorgan Chase acquired deposits and "
            "substantially all assets, not automatically "
            "the old common shares."
        ),
        "Outcome Readiness": "EQUITY_RECOVERY_UNVERIFIED",
        "Source Type": "FDIC_RELEASE",
        "Source URL": (
            "https://www.fdic.gov/news/"
            "press-releases/2023/pr23034.html"
        ),
        "Remaining Work": (
            "Verify common-share trading and recovery "
            "treatment after receivership."
        ),
    },
]


def main() -> None:
    """Run the module entry point and its workflow.
    """
    if not INPUT_FILE.is_file():
        raise FileNotFoundError(INPUT_FILE)

    queue = pd.read_csv(
        INPUT_FILE,
        dtype="string",
    )

    required = {
        "Underlying RIC",
        "Last Valid Quote Date",
        "Affected Decision Windows",
    }

    missing = required - set(queue.columns)

    if missing:
        raise ValueError(
            f"Review queue missing columns: {sorted(missing)}"
        )

    if queue["Underlying RIC"].isna().any():
        raise ValueError("Null RICs in review queue")

    if queue["Underlying RIC"].duplicated().any():
        raise ValueError("Duplicate RICs in review queue")

    events = pd.DataFrame(EVENTS)

    if events["Underlying RIC"].duplicated().any():
        raise ValueError("Duplicate source-verified events")

    unknown_rics = (
        set(events["Underlying RIC"])
        - set(queue["Underlying RIC"])
    )

    if unknown_rics:
        raise ValueError(
            f"Event RICs absent from queue: {sorted(unknown_rics)}"
        )

    event_dates = pd.to_datetime(
        events["Documented Event Date"],
        errors="raise",
    )

    last_quote_dates = pd.to_datetime(
        queue["Last Valid Quote Date"],
        errors="coerce",
    )

    if last_quote_dates.isna().any():
        raise ValueError("Invalid last-quote dates")

    merged = queue.merge(
        events,
        on="Underlying RIC",
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    merged["External Evidence Status"] = (
        merged["_merge"]
        .map({
            "both": "SOURCE_DOCUMENTED_EVENT",
            "left_only": "NOT_YET_SOURCE_DOCUMENTED",
        })
        .astype("string")
    )

    merged = merged.drop(columns=["_merge"])

    merged["Documented Event Date"] = pd.to_datetime(
        merged["Documented Event Date"],
        errors="coerce",
    )

    merged["Last Valid Quote Date"] = pd.to_datetime(
        merged["Last Valid Quote Date"],
        errors="coerce",
    )

    merged["Days After Last Quote"] = (
        merged["Documented Event Date"]
        - merged["Last Valid Quote Date"]
    ).dt.days

    merged["Needs Date Reconciliation"] = (
        merged["Documented Event Date"].notna()
        & (
            (merged["Days After Last Quote"] < 0)
            | (merged["Days After Last Quote"] > 10)
        )
    )

    merged["Affected Decision Windows"] = pd.to_numeric(
        merged["Affected Decision Windows"],
        errors="raise",
    )

    merged = merged.sort_values(
        ["Affected Decision Windows", "Underlying RIC"],
        ascending=[False, True],
    )

    # Explicitly separate evidence from approved outcome use.
    #
    # Even source-documented events remain excluded from
    # automatic target reconstruction until their full
    # horizon-specific economic treatment is implemented.
    merged["Terminal Outcome Reconstructed"] = False
    merged["Approved For Model Target Repair"] = False

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    event_file = (
        OUTPUT_DIR / "source_documented_events.csv"
    )

    review_file = (
        OUTPUT_DIR / "terminal_event_review_panel.csv"
    )

    for path in (event_file, review_file):
        if path.exists():
            raise FileExistsError(
                f"Refusing to overwrite: {path}"
            )

    events.to_csv(
        event_file,
        index=False,
        encoding="utf-8-sig",
    )

    merged.to_csv(
        review_file,
        index=False,
        encoding="utf-8-sig",
    )

    documented = merged.loc[
        merged["External Evidence Status"].eq(
            "SOURCE_DOCUMENTED_EVENT"
        )
    ]

    print("\n--- TERMINAL EVENT EVIDENCE REGISTRY ---")

    print("Total review RICs:", len(merged))
    print("Source-documented event RICs:", len(documented))

    print(
        "Not yet source-documented:",
        len(merged) - len(documented),
    )

    print(
        "Date discrepancies requiring inspection:",
        int(merged["Needs Date Reconciliation"].sum()),
    )

    print("\nDocumented events:")

    display_columns = [
        "Underlying RIC",
        "Event Type",
        "Documented Event Date",
        "Outcome Readiness",
        "Days After Last Quote",
    ]

    print(
        documented[display_columns].to_string(
            index=False
        )
    )

    print("\nSource event registry:", event_file)
    print("Review panel:", review_file)

    print(
        "\nNo stock outcomes were reconstructed. "
        "No original research datasets were modified."
    )


if __name__ == "__main__":
    main()
