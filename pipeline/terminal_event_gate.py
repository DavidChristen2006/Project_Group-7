
"""Quarantine unresolved terminal corporate-action windows.

Uses the existing registry. No LSEG calls or price modifications.
"""

from functools import lru_cache
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

REGISTRY = (
    "data/processed/research_audit/"
    "corporate_action_review/"
    "terminal_event_registry_v1/"
    "terminal_event_review_panel.csv"
)


class TerminalEventWindowError(ValueError):
    pass


def _date(value):
    result = pd.Timestamp(value)

    if pd.isna(result) or result.tzinfo is not None:
        raise ValueError("Invalid event/window date")

    return result.normalize()


@lru_cache(maxsize=8)
def _load_registry(path_string):

    path = Path(path_string)

    if not path.is_file():
        raise FileNotFoundError(
            f"Corporate-action registry missing: {path}"
        )

    frame = pd.read_csv(
        path,
        dtype="string",
    )

    required = {
        "Underlying RIC",
        "First Missing Quote Date",
        "Documented Event Date",
        "Event Type",
    }

    missing = required - set(frame.columns)

    if missing:
        raise ValueError(
            f"Registry missing columns: {sorted(missing)}"
        )

    if (
        frame.empty
        or frame["Underlying RIC"].isna().any()
        or frame["Underlying RIC"].duplicated().any()
    ):
        raise ValueError(
            "Invalid corporate-action registry"
        )

    result = {}

    for _, row in frame.iterrows():

        ric = str(row["Underlying RIC"])

        documented = row["Documented Event Date"]

        is_documented = pd.notna(documented)

        event_date = (
            documented
            if is_documented
            else row["First Missing Quote Date"]
        )

        if pd.isna(event_date):
            raise ValueError(
                f"Missing event/review date: {ric}"
            )

        event_type = (
            str(row["Event Type"])
            if is_documented
            else "SUSPECTED_TRADING_CESSATION"
        )

        result[ric] = (
            _date(event_date),
            event_type,
            is_documented,
        )

    return result


def require_no_unresolved_terminal_event(
    ric,
    decision_date,
    outcome_date,
    *,
    project_root=None,
):

    start = _date(decision_date)
    end = _date(outcome_date)

    if start >= end:
        raise ValueError(
            "Decision must precede outcome"
        )

    root = (
        Path(project_root)
        if project_root is not None
        else ROOT
    )

    registry_path = (
        root / REGISTRY
    ).resolve()

    registry = _load_registry(
        str(registry_path)
    )

    if ric not in registry:
        return

    event_date, event_type, documented = (
        registry[ric]
    )

    if start <= event_date <= end:

        status = (
            "DOCUMENTED"
            if documented
            else "UNVERIFIED"
        )

        raise TerminalEventWindowError(
            f"Corporate-action quarantine: {ric}, "
            f"{start.date()} to {end.date()}, "
            f"event/review date {event_date.date()}, "
            f"{event_type} ({status}). "
            "Stock and option settlement "
            "have not been verified."
        )


if __name__ == "__main__":

    events = _load_registry(
        str((ROOT / REGISTRY).resolve())
    )

    assert len(events) >= 6

    for ric, start, end in [
        ("TWTR.N^J22", "2022-09-01", "2022-11-30"),
        ("ALXN.OQ^G21", "2021-06-01", "2021-08-31"),
        ("FRC.N^E23", "2023-03-01", "2023-06-01"),
    ]:

        try:
            require_no_unresolved_terminal_event(
                ric, start, end
            )

        except TerminalEventWindowError:
            pass

        else:
            raise AssertionError(
                f"Expected quarantine: {ric}"
            )

    require_no_unresolved_terminal_event(
        "JPM.N",
        "2023-01-03",
        "2023-04-03",
    )

    print(
        "TERMINAL EVENT GATE: PASS",
        f"({len(events)} reviewed securities)",
    )
