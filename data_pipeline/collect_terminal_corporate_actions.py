
"""Collect LSEG event evidence for discontinued equity histories.

Uses the default LSEG Platform Session.
Does not modify prices, membership, or training targets.
"""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

import lseg.data as ld
import pandas as pd


ROOT = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "processed"
)

WORKLIST_FILE = (
    ROOT
    / "research_audit/corporate_action_review"
    / "terminal_event_worklist_generated.csv"
)

OUTPUT_ROOT = (
    ROOT
    / "research_audit/corporate_action_review"
    / "lseg_event_enrichment"
)

CAPITAL_CHANGE_FIELDS = [
    "TR.CACorpActDesc",
    "TR.CAAnnouncementDate",
    "TR.CAEffectiveDate(CAEventType=All)",
    "TR.CACorpActEventType",
]

GENERAL_EVENT_FIELDS = [
    "TR.EventStartDate",
    "TR.EventType",
    "TR.EventTitle",
    "TR.EventEndDate",
]


def load_worklist() -> pd.DataFrame:
    if not WORKLIST_FILE.is_file():
        raise FileNotFoundError(WORKLIST_FILE)

    data = pd.read_csv(WORKLIST_FILE)

    required = {
        "Underlying RIC",
        "Last Valid Quote Date",
        "Nearby Index Leaver Date",
        "Affected Decision Windows",
    }

    missing = required - set(data.columns)

    if missing:
        raise ValueError(
            f"Missing worklist fields: {sorted(missing)}"
        )

    data["Underlying RIC"] = (
        data["Underlying RIC"]
        .astype("string")
        .str.strip()
    )

    if (
        data["Underlying RIC"].isna().any()
        or data["Underlying RIC"].eq("").any()
        or data["Underlying RIC"].duplicated().any()
    ):
        raise ValueError(
            "Worklist contains blank or duplicated RICs"
        )

    for column in (
        "Last Valid Quote Date",
        "Nearby Index Leaver Date",
    ):
        data[column] = pd.to_datetime(
            data[column],
            errors="coerce",
        ).dt.normalize()

    if data["Last Valid Quote Date"].isna().any():
        raise ValueError(
            "Last trading dates must be available"
        )

    data["Affected Decision Windows"] = (
        pd.to_numeric(
            data["Affected Decision Windows"],
            errors="raise",
        )
    )

    return data.sort_values(
        [
            "Affected Decision Windows",
            "Underlying RIC",
        ],
        ascending=[False, True],
    ).reset_index(drop=True)


def query_lseg(
    ric: str,
    family: str,
    start: str,
    end: str,
) -> pd.DataFrame:

    if family == "CAPITAL_CHANGE":
        return ld.get_data(
            universe=[ric],
            fields=CAPITAL_CHANGE_FIELDS,
            parameters={
                "SDate": start,
                "EDate": end,
            },
        )

    if family == "GENERAL_EVENT":
        return ld.get_data(
            universe=[ric],
            fields=GENERAL_EVENT_FIELDS,
            parameters={
                "SDate": start,
                "EDate": end,
                "EventType": "ALL",
            },
        )

    raise ValueError(
        f"Unknown request family: {family}"
    )


def safe_filename(ric: str) -> str:
    safe = re.sub(
        r"[^A-Za-z0-9_.-]",
        "_",
        ric,
    )

    digest = hashlib.sha256(
        ric.encode("utf-8")
    ).hexdigest()[:8]

    return f"{safe}_{digest}"


def process_query(
    ric: str,
    family: str,
    start: str,
    end: str,
    raw_folder: Path,
) -> dict:

    summary = {
        "Underlying RIC": ric,
        "Query Family": family,
        "Query Start": start,
        "Query End": end,
        "Request Status": "NOT_RUN",
        "Returned Rows": 0,
        "Rows With Nonempty Fields": 0,
        "Returned Columns": "",
        "Raw File": "",
        "Error": "",
    }

    try:
        result = query_lseg(
            ric=ric,
            family=family,
            start=start,
            end=end,
        )

        if result is None:
            summary["Request Status"] = "NO_RESPONSE"
            return summary

        if not isinstance(result, pd.DataFrame):
            raise TypeError(
                f"Expected DataFrame; received "
                f"{type(result).__name__}"
            )

        frame = result.copy().reset_index(
            drop=True
        )

        frame.columns = [
            str(column)
            for column in frame.columns
        ]

        summary["Returned Rows"] = len(frame)

        summary["Returned Columns"] = " | ".join(
            frame.columns
        )

        if frame.empty:
            summary["Request Status"] = "NO_ROWS"
            return summary

        # A response containing only an Instrument
        # identifier is not evidence of an event.
        detail_columns = [
            col
            for col in frame.columns
            if col.lower() not in {
                "instrument",
                "ric",
                "error",
            }
        ]

        meaningful = (
            int(
                frame[detail_columns]
                .notna()
                .any(axis=1)
                .sum()
            )
            if detail_columns
            else 0
        )

        summary[
            "Rows With Nonempty Fields"
        ] = meaningful

        summary["Request Status"] = (
            "RETURNED_EVENT_FIELDS"
            if meaningful > 0
            else "PLACEHOLDER_ONLY"
        )

        # Preserve the original API output and
        # explicitly identify the query provenance.
        frame.insert(
            0,
            "Requested RIC",
            ric,
        )

        frame.insert(
            1,
            "Query Family",
            family,
        )

        frame.insert(
            2,
            "Query Start",
            start,
        )

        frame.insert(
            3,
            "Query End",
            end,
        )

        filename = (
            f"{safe_filename(ric)}"
            f"_{family.lower()}.csv"
        )

        destination = raw_folder / filename

        frame.to_csv(
            destination,
            index=False,
            encoding="utf-8-sig",
        )

        summary["Raw File"] = str(
            destination
        )

    except Exception as exc:
        summary["Request Status"] = "ERROR"
        summary["Error"] = (
            f"{type(exc).__name__}: {exc}"
        )[:1000]

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    selection = parser.add_mutually_exclusive_group()

    selection.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Number of priority RICs to query",
    )

    selection.add_argument(
        "--all",
        action="store_true",
        help="Query the entire terminal-event worklist",
    )

    args = parser.parse_args()

    worklist = load_worklist()

    if args.all:
        selected = worklist.copy()
        run_name = "full"
    else:
        if args.limit <= 0:
            parser.error(
                "--limit must be positive"
            )

        selected = worklist.head(
            args.limit
        ).copy()

        run_name = (
            f"pilot_{len(selected)}"
        )

    output_folder = (
        OUTPUT_ROOT / run_name
    )

    raw_folder = output_folder / "raw"

    raw_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    summaries = []

    print(
        "\n--- LSEG CORPORATE-ACTION EVIDENCE COLLECTION ---"
    )

    print(
        "Worklist RICs:",
        len(worklist),
    )

    print(
        "RICs selected:",
        len(selected),
    )

    print(
        "Queries per RIC: 2"
    )

    print(
        "Price labels will not be modified."
    )

    ld.open_session()

    try:
        for i, (_, row) in enumerate(
            selected.iterrows(),
            start=1,
        ):
            ric = str(
                row["Underlying RIC"]
            )

            last_quote = pd.Timestamp(
                row["Last Valid Quote Date"]
            )

            # Include earlier announcements and
            # events shortly after the last quote.
            start = (
                last_quote
                - pd.Timedelta(days=365)
            ).date().isoformat()

            end = (
                last_quote
                + pd.Timedelta(days=90)
            ).date().isoformat()

            print(
                f"\n[{i}/{len(selected)}] "
                f"{ric} ({start} to {end})",
                flush=True,
            )

            for family in (
                "CAPITAL_CHANGE",
                "GENERAL_EVENT",
            ):
                summary = process_query(
                    ric=ric,
                    family=family,
                    start=start,
                    end=end,
                    raw_folder=raw_folder,
                )

                summary[
                    "Last Valid Quote Date"
                ] = last_quote

                summary[
                    "Nearby Index Leaver Date"
                ] = row[
                    "Nearby Index Leaver Date"
                ]

                summary[
                    "Affected Decision Windows"
                ] = int(
                    row[
                        "Affected Decision Windows"
                    ]
                )

                summaries.append(summary)

                print(
                    f"  {family}: "
                    f"{summary['Request Status']}, "
                    f"rows={summary['Returned Rows']}",
                    flush=True,
                )

                if summary["Error"]:
                    print(
                        f"    {summary['Error']}",
                        flush=True,
                    )

            # Persist progress after each RIC.
            pd.DataFrame(
                summaries
            ).to_csv(
                output_folder
                / "lseg_request_summary.csv",
                index=False,
                encoding="utf-8-sig",
            )

    finally:
        ld.close_session()

    report = pd.DataFrame(
        summaries
    )

    print(
        "\n--- LSEG EVENT COLLECTION SUMMARY ---"
    )

    print(
        "RICs queried:",
        report[
            "Underlying RIC"
        ].nunique()
    )

    print(
        "Total requests:",
        len(report),
    )

    print("\nRequest statuses:")

    print(
        report["Request Status"]
        .value_counts()
        .to_string()
    )

    print(
        "\nRICs with at least one nonempty "
        "event response:",
        report.loc[
            report[
                "Request Status"
            ].eq("RETURNED_EVENT_FIELDS"),
            "Underlying RIC",
        ].nunique(),
    )

    print(
        "Requests with errors:",
        int(
            report[
                "Request Status"
            ].eq("ERROR").sum()
        ),
    )

    print(
        "\nResults:",
        output_folder,
    )

    print(
        "STATUS: UNVERIFIED SOURCE EVIDENCE. "
        "No corporate-action payoff was inferred."
    )


if __name__ == "__main__":
    main()
