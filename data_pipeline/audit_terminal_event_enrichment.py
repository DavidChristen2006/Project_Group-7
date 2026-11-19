
"""Audit existing LSEG corporate-action responses without modifying outcomes.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1] / "data" / "processed"

BASE = ROOT / "research_audit" / "corporate_action_review"

WORKLIST = BASE / "terminal_event_worklist_generated.csv"

SUMMARY = (
    BASE
    / "lseg_event_enrichment"
    / "full"
    / "lseg_request_summary.csv"
)

OUTPUT = BASE / "reconciliation_review"

IDENTIFIER_FIELDS = {
    "instrument",
    "ric",
    "requested ric",
    "query family",
    "query start",
    "query end",
    "error",
}

EMPTY_VALUES = {"", "NAN", "<NA>", "NONE", "NULL", "NAT"}


def load_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Missing input: {path}")

    return pd.read_csv(
        path,
        dtype="string",
        keep_default_na=False,
    )


def nonempty(value: object) -> bool:
    return str(value).strip().upper() not in EMPTY_VALUES


def main() -> None:
    work = load_csv(WORKLIST)
    requests = load_csv(SUMMARY)

    required_work = {
        "Underlying RIC",
        "Last Valid Quote Date",
        "Nearby Index Leaver Date",
        "Affected Decision Windows",
    }

    required_requests = {
        "Underlying RIC",
        "Query Family",
        "Request Status",
        "Raw File",
    }

    missing_work = required_work - set(work.columns)
    missing_requests = required_requests - set(requests.columns)

    if missing_work:
        raise ValueError(
            f"Worklist missing: {sorted(missing_work)}"
        )

    if missing_requests:
        raise ValueError(
            f"Request summary missing: {sorted(missing_requests)}"
        )

    if work["Underlying RIC"].duplicated().any():
        raise ValueError("Worklist contains duplicated RICs")

    if requests.duplicated(
        ["Underlying RIC", "Query Family"]
    ).any():
        raise ValueError(
            "Request summary contains duplicated RIC/query pairs"
        )

    work_ric = set(work["Underlying RIC"])
    request_ric = set(requests["Underlying RIC"])

    if work_ric != request_ric:
        raise ValueError(
            f"RIC mismatch: "
            f"missing={sorted(work_ric - request_ric)}, "
            f"extra={sorted(request_ric - work_ric)}"
        )

    expected_families = {
        "CAPITAL_CHANGE",
        "GENERAL_EVENT",
    }

    if set(requests["Query Family"]) != expected_families:
        raise ValueError("Unexpected query families")

    if len(requests) != 2 * len(work):
        raise ValueError(
            "Expected exactly two requests per RIC"
        )

    statuses = requests.pivot(
        index="Underlying RIC",
        columns="Query Family",
        values="Request Status",
    ).reset_index()

    statuses = statuses.rename(
        columns={
            "CAPITAL_CHANGE": "Capital Change Request Status",
            "GENERAL_EVENT": "General Event Request Status",
        }
    )

    queue = work.merge(
        statuses,
        on="Underlying RIC",
        how="left",
        validate="one_to_one",
    )

    field_records = []
    details_by_ric = {}
    missing_raw = []

    returned = requests.loc[
        requests["Request Status"].eq(
            "RETURNED_EVENT_FIELDS"
        )
    ]

    for _, req in returned.iterrows():
        ric = str(req["Underlying RIC"])
        path_string = str(req["Raw File"]).strip()

        if not path_string:
            missing_raw.append(f"{ric}: blank Raw File")
            continue

        raw_path = Path(path_string)

        if not raw_path.is_file():
            missing_raw.append(f"{ric}: {raw_path}")
            continue

        raw = load_csv(raw_path)

        for record_number, (_, source_row) in enumerate(
            raw.iterrows(),
            start=1,
        ):
            for field, value in source_row.items():
                if (
                    str(field).strip().lower()
                    in IDENTIFIER_FIELDS
                ):
                    continue

                if not nonempty(value):
                    continue

                field_records.append(
                    {
                        "Underlying RIC": ric,
                        "Query Family": str(req["Query Family"]),
                        "Response Row": str(record_number),
                        "Source Field": str(field),
                        "Original Value": str(value),
                        "Raw File": str(raw_path),
                    }
                )

                details_by_ric.setdefault(
                    ric, []
                ).append(
                    f"{field}={value}"
                )

    queue["Returned Event Field Values"] = (
        queue["Underlying RIC"].map(
            lambda ric: " | ".join(
                details_by_ric.get(str(ric), [])
            )
        )
    )

    queue["Evidence Review Status"] = (
        queue["Underlying RIC"].map(
            lambda ric: (
                "UNVERIFIED_FIELDS_RETURNED"
                if details_by_ric.get(str(ric))
                else "NO_USABLE_LSEG_EVENT_FIELDS"
            )
        )
    )

    queue["Review Priority"] = pd.to_numeric(
        queue["Affected Decision Windows"],
        errors="coerce",
    )

    if queue["Review Priority"].isna().any():
        raise ValueError(
            "Invalid Affected Decision Windows values"
        )

    # Reserved for independently verified transaction facts.
    # No values are inferred automatically.
    for column in (
        "Verified Corporate Action Type",
        "Verified Effective Date",
        "Cash Per Old Share",
        "Acquirer Shares Per Old Share",
        "Acquirer RIC",
        "Additional Consideration / CVR",
        "Primary Source URL",
        "Verification Notes",
    ):
        queue[column] = ""

    queue = queue.sort_values(
        ["Review Priority", "Underlying RIC"],
        ascending=[False, True],
    ).drop(columns="Review Priority")

    OUTPUT.mkdir(parents=True, exist_ok=True)

    queue_file = (
        OUTPUT
        / "terminal_corporate_action_review_queue.csv"
    )

    fields_file = (
        OUTPUT
        / "lseg_returned_event_fields.csv"
    )

    queue.to_csv(
        queue_file,
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(
        field_records,
        columns=[
            "Underlying RIC",
            "Query Family",
            "Response Row",
            "Source Field",
            "Original Value",
            "Raw File",
        ],
    ).to_csv(
        fields_file,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "\n--- TERMINAL CORPORATE-ACTION EVIDENCE AUDIT ---"
    )

    print("RICs:", len(queue))
    print("LSEG requests:", len(requests))
    print("Requests with event fields:", len(returned))
    print(
        "RICs with parsed nonempty fields:",
        len(details_by_ric),
    )
    print("Raw files unavailable:", len(missing_raw))

    print("\nQuery request statuses:")

    print(
        requests.groupby(
            ["Query Family", "Request Status"]
        ).size().to_string()
    )

    print("\nRICs with returned fields:")

    for ric in sorted(details_by_ric):
        print(f"  {ric}:")

        for detail in details_by_ric[ric]:
            print(f"    {detail[:350]}")

    for error in missing_raw:
        print("MISSING_RAW_FILE:", error)

    print("\nQueue:", queue_file)
    print("Original returned fields:", fields_file)

    print(
        "No research outcomes or corporate-action "
        "labels were changed."
    )


if __name__ == "__main__":
    main()
