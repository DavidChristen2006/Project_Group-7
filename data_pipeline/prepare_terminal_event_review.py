
"""Prepare an event-level corporate-action research worklist.

No LSEG calls. No price or training-label modifications.
"""

from pathlib import Path
import hashlib
import re

import numpy as np
import pandas as pd


ROOT = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "processed"
)

AUDIT_FILE = (
    ROOT / "research_audit/forward_63d_quality"
    / "forward_cases_for_review.csv"
)

EVENT_FILE = (
    ROOT / "historical_universe"
    / "sp500_membership_events.parquet"
)

PRICE_ROOT = ROOT / "market_history_v2/series"

OUTPUT = (
    ROOT / "research_audit"
    / "corporate_action_review"
)

PRICE_PERIOD = "2013-09-01_2025-04-30.parquet"


def price_path(ric: str) -> Path:
    safe = re.sub(
        r"[^A-Za-z0-9_.-]", "_", ric
    )

    digest = hashlib.sha256(
        ric.encode("utf-8")
    ).hexdigest()[:8]

    return (
        PRICE_ROOT
        / f"{safe}_{digest}"
        / "adjusted"
        / PRICE_PERIOD
    )


def load_last_quote(ric: str):
    path = price_path(ric)

    if not path.is_file():
        raise FileNotFoundError(path)

    frame = pd.read_parquet(
        path,
        columns=["Date", "Close"],
    )

    frame["Date"] = pd.to_datetime(
        frame["Date"],
        errors="raise",
    ).dt.normalize()

    frame["Close"] = pd.to_numeric(
        frame["Close"],
        errors="coerce",
    )

    good = frame.loc[
        np.isfinite(frame["Close"])
        & frame["Close"].gt(0)
    ].sort_values("Date")

    if good.empty:
        return pd.NaT, np.nan

    last = good.iloc[-1]

    return (
        last["Date"],
        float(last["Close"]),
    )


def main():
    if not AUDIT_FILE.is_file():
        raise FileNotFoundError(AUDIT_FILE)

    if not EVENT_FILE.is_file():
        raise FileNotFoundError(EVENT_FILE)

    cases = pd.read_csv(
        AUDIT_FILE
    )

    required = {
        "Decision Date",
        "Underlying RIC",
        "Review Category",
        "target_exit_date",
        "target_first_missing_stock_date",
        "Last Valid Historical Quote",
    }

    missing = required - set(cases.columns)

    if missing:
        raise ValueError(
            f"Missing audit columns: {sorted(missing)}"
        )

    for column in (
        "Decision Date",
        "target_exit_date",
        "target_first_missing_stock_date",
        "Last Valid Historical Quote",
    ):
        cases[column] = pd.to_datetime(
            cases[column],
            errors="coerce",
        ).dt.normalize()

    if cases[
        ["Decision Date", "Underlying RIC"]
    ].isna().any().any():
        raise ValueError(
            "Missing decision-date/RIC keys"
        )

    if cases.duplicated(
        ["Decision Date", "Underlying RIC"]
    ).any():
        raise ValueError(
            "Duplicate audit cases"
        )

    terminal = cases.loc[
        cases["Review Category"].eq(
            "POSSIBLE_TERMINAL_HISTORY"
        )
    ].copy()

    if terminal.empty:
        raise ValueError(
            "No possible terminal histories found"
        )

    events = pd.read_parquet(
        EVENT_FILE
    )

    needed_events = {
        "Change Date",
        "RIC",
        "Change",
    }

    if not needed_events.issubset(
        events.columns
    ):
        raise ValueError(
            "Membership events have unexpected schema"
        )

    events["Change Date"] = pd.to_datetime(
        events["Change Date"],
        errors="raise",
    ).dt.normalize()

    leavers = events.loc[
        events["Change"].eq("Leaver")
    ].copy()

    grouped = (
        terminal.groupby(
            "Underlying RIC",
            as_index=False,
        )
        .agg(
            Affected_Decisions=(
                "Decision Date", "nunique"
            ),
            First_Affected_Decision=(
                "Decision Date", "min"
            ),
            Last_Affected_Decision=(
                "Decision Date", "max"
            ),
            First_Missing_Quote_Date=(
                "target_first_missing_stock_date",
                "min",
            ),
            Last_Exit_Date_Affected=(
                "target_exit_date",
                "max",
            ),
            Audit_Last_Valid_Quote=(
                "Last Valid Historical Quote",
                "max",
            ),
        )
    )

    records = []

    for row in grouped.itertuples(index=False):
        ric = str(row[0])

        last_quote_date, last_close = (
            load_last_quote(ric)
        )

        matching = leavers.loc[
            leavers["RIC"].eq(ric)
        ].copy()

        nearest_leaver = pd.NaT

        days_from_leaver = pd.NA

        if (
            not matching.empty
            and pd.notna(last_quote_date)
        ):
            matching["Distance"] = (
                matching["Change Date"]
                - last_quote_date
            ).dt.days.abs()

            nearest = matching.sort_values(
                ["Distance", "Change Date"]
            ).iloc[0]

            if nearest["Distance"] <= 90:
                nearest_leaver = (
                    nearest["Change Date"]
                )

                days_from_leaver = int(
                    (
                        nearest_leaver
                        - last_quote_date
                    ).days
                )

        audit_date_matches = (
            pd.notna(row.Audit_Last_Valid_Quote)
            and last_quote_date
            == row.Audit_Last_Valid_Quote
        )

        first_missing = (
            row.First_Missing_Quote_Date
        )

        internal_gap_possible = bool(
            pd.notna(first_missing)
            and pd.notna(last_quote_date)
            and first_missing <= last_quote_date
        )

        records.append({
            "Underlying RIC": ric,
            "Affected Decision Windows": int(
                row.Affected_Decisions
            ),
            "First Affected Decision": (
                row.First_Affected_Decision
            ),
            "Last Affected Decision": (
                row.Last_Affected_Decision
            ),
            "First Missing Quote Date": (
                first_missing
            ),
            "Last Valid Quote Date": (
                last_quote_date
            ),
            "Last Adjusted Close": last_close,
            "Last Affected Target Exit": (
                row.Last_Exit_Date_Affected
            ),
            "Audit Last Quote Matches": bool(
                audit_date_matches
            ),
            "Internal Gap Also Possible": (
                internal_gap_possible
            ),
            "Nearby Index Leaver Date": (
                nearest_leaver
            ),
            "Days from Last Quote to Leaver": (
                days_from_leaver
            ),
            "Corporate Action Type": "UNKNOWN",
            "Verified Effective Date": pd.NaT,
            "Cash Consideration Per Share": np.nan,
            "Successor RIC": "",
            "Successor Shares Per Old Share": np.nan,
            "Other Distribution or Consideration": "",
            "Option Contract Adjustment Verified": False,
            "Evidence Source": "",
            "Review Status": "PENDING",
            "Reviewer Notes": "",
        })

    worklist = pd.DataFrame(records)

    worklist = worklist.sort_values(
        [
            "Affected Decision Windows",
            "Underlying RIC",
        ],
        ascending=[False, True],
    ).reset_index(drop=True)

    other_cases = cases.loc[
        ~cases["Review Category"].eq(
            "POSSIBLE_TERMINAL_HISTORY"
        )
    ].copy()

    OUTPUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Generated worklist only: retain a separate
    # researcher-maintained file for manual annotations.
    worklist.to_csv(
        OUTPUT / "terminal_event_worklist_generated.csv",
        index=False,
    )

    other_cases.to_csv(
        OUTPUT / "nonterminal_and_entry_cases.csv",
        index=False,
    )

    print("\n--- CORPORATE-ACTION REVIEW WORKLIST ---")

    print(
        "Terminal-looking decision windows:",
        len(terminal),
    )

    print(
        "Unique terminal-looking RICs:",
        len(worklist),
    )

    print(
        "With nearby index-leaver events:",
        int(
            worklist[
                "Nearby Index Leaver Date"
            ].notna().sum()
        ),
    )

    print(
        "Audit last-quote dates matching source:",
        int(
            worklist[
                "Audit Last Quote Matches"
            ].sum()
        ),
    )

    print(
        "Additional internal-gap possibility:",
        int(
            worklist[
                "Internal Gap Also Possible"
            ].sum()
        ),
    )

    print(
        "Other incomplete windows:",
        len(other_cases),
    )

    print("\nTop 10 by affected decision windows:")

    print(
        worklist[
            [
                "Underlying RIC",
                "Affected Decision Windows",
                "Last Valid Quote Date",
                "Nearby Index Leaver Date",
                "Corporate Action Type",
            ]
        ].head(10).to_string(index=False)
    )

    print("\nOutput:", OUTPUT)

    print(
        "All corporate-action classifications remain "
        "unverified until external evidence is added."
    )


if __name__ == "__main__":
    main()
