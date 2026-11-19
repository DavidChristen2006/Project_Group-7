
"""Audit historical equity price availability at each research decision date.

No LSEG requests. No original files are modified.
No forward-filling, future returns, or ML labels are created.
"""

from pathlib import Path
import hashlib
import re

import numpy as np
import pandas as pd


ROOT = Path("data/processed")

MEMBERSHIP_FILE = (
    ROOT
    / "historical_universe/equity_collection_candidate"
    / "equity_membership_grid_candidate.parquet"
)

HISTORY_ROOT = ROOT / "market_history_v2/series"

OUTPUT = ROOT / "research_audit/equity_decision_coverage"

PERIOD = "2013-09-01_2025-04-30"
ADJUSTMENT_MODE = "adjusted"

MIN_HISTORY_PRICES = 253
HISTORY_WINDOW_DAYS = 400
MAX_STALENESS_DAYS = 7
LARGE_DAILY_MOVE = 0.50


def history_path(ric: str) -> Path:
    safe = re.sub(
        r"[^A-Za-z0-9_.-]", "_", ric
    )

    digest = hashlib.sha256(
        ric.encode("utf-8")
    ).hexdigest()[:8]

    return (
        HISTORY_ROOT
        / f"{safe}_{digest}"
        / ADJUSTMENT_MODE
        / f"{PERIOD}.parquet"
    )


def load_history(ric: str) -> pd.DataFrame:
    path = history_path(ric)

    if not path.is_file():
        raise FileNotFoundError(path)

    df = pd.read_parquet(path)

    required = {
        "Date",
        "Close",
        "RIC",
        "Adjustment Mode",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"{ric}: missing columns {sorted(missing)}"
        )

    if not df["RIC"].astype("string").eq(ric).all():
        raise ValueError(
            f"{ric}: incorrect RIC metadata"
        )

    if not df["Adjustment Mode"].astype(
        "string"
    ).eq(ADJUSTMENT_MODE).all():
        raise ValueError(
            f"{ric}: incorrect adjustment mode"
        )

    df = df.copy()

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="raise",
    ).dt.normalize()

    if (
        df["Date"].isna().any()
        or df["Date"].duplicated().any()
    ):
        raise ValueError(
            f"{ric}: invalid or duplicate dates"
        )

    df["Close"] = pd.to_numeric(
        df["Close"],
        errors="coerce",
    )

    return (
        df.sort_values("Date")
        .reset_index(drop=True)
    )


def audit_ric(
    ric: str,
    members: pd.DataFrame,
):
    prices = load_history(ric)

    closes = prices["Close"].to_numpy(
        dtype=float
    )

    valid = (
        np.isfinite(closes)
        & (closes > 0)
    )

    good = (
        prices.loc[
            valid, ["Date", "Close"]
        ]
        .reset_index(drop=True)
    )

    decisions = members[
        "Decision Date"
    ].to_numpy(dtype="datetime64[ns]")

    dates = good[
        "Date"
    ].to_numpy(dtype="datetime64[ns]")

    # Strictly prior to the decision date.
    # No same-day closing price is used.
    positions = np.searchsorted(
        dates,
        decisions,
        side="left",
    )

    available = positions > 0

    prior_dates = np.full(
        len(members),
        np.datetime64("NaT", "ns"),
        dtype="datetime64[ns]",
    )

    prior_prices = np.full(
        len(members),
        np.nan,
    )

    if available.any():
        previous = positions[available] - 1

        prior_dates[available] = dates[previous]

        prior_prices[available] = (
            good["Close"]
            .to_numpy(dtype=float)[previous]
        )

    window_starts = (
        decisions
        - np.timedelta64(
            HISTORY_WINDOW_DAYS, "D"
        )
    )

    start_positions = np.searchsorted(
        dates,
        window_starts,
        side="left",
    )

    trailing_counts = (
        positions - start_positions
    )

    result = (
        members.copy()
        .reset_index(drop=True)
    )

    result["Last Pre-Decision Price Date"] = (
        pd.to_datetime(prior_dates)
    )

    result["Last Pre-Decision Close"] = (
        prior_prices
    )

    result["Days Since Last Close"] = (
        result["Decision Date"]
        - result["Last Pre-Decision Price Date"]
    ).dt.days

    result["Valid Closes in Prior 400 Days"] = (
        trailing_counts
    )

    result["Has Prior Price"] = available

    result["Recent Prior Price"] = (
        result["Has Prior Price"]
        & result["Days Since Last Close"].le(
            MAX_STALENESS_DAYS
        )
    )

    result["253-Price Feature Candidate"] = (
        result["Recent Prior Price"]
        & result[
            "Valid Closes in Prior 400 Days"
        ].ge(MIN_HISTORY_PRICES)
    )

    # Diagnose invalid or missing closes.
    invalid = prices.loc[
        ~valid, ["Date", "Close"]
    ].copy()

    invalid.insert(
        0, "RIC", ric
    )

    if "Close Field" in prices.columns:
        invalid["Close Field"] = (
            prices.loc[
                ~valid, "Close Field"
            ].to_numpy()
        )

    # No automatic filling of missing prices.
    clean_close = prices["Close"].where(
        valid
    )

    returns = clean_close.pct_change(
        fill_method=None
    )

    unusual_mask = returns.abs().ge(
        LARGE_DAILY_MOVE
    )

    unusual = prices.loc[
        unusual_mask,
        ["Date", "Close"],
    ].copy()

    unusual.insert(
        0, "RIC", ric
    )

    unusual["Daily Adjusted Price Return"] = (
        returns.loc[
            unusual_mask
        ].to_numpy()
    )

    summary = {
        "RIC": ric,
        "History Rows": len(prices),
        "Positive Close Rows": int(valid.sum()),
        "Invalid Close Rows": int((~valid).sum()),
        "First Positive Close": (
            good["Date"].min()
            if len(good)
            else pd.NaT
        ),
        "Last Positive Close": (
            good["Date"].max()
            if len(good)
            else pd.NaT
        ),
        "Membership Decision Dates": len(members),
        "Dates With Prior Price": int(
            result["Has Prior Price"].sum()
        ),
        "Dates With Recent Price": int(
            result["Recent Prior Price"].sum()
        ),
        "Feature Candidate Dates": int(
            result[
                "253-Price Feature Candidate"
            ].sum()
        ),
        "Large Daily Moves": len(unusual),
    }

    return (
        result,
        summary,
        invalid,
        unusual,
    )


def main():
    panel = pd.read_parquet(
        MEMBERSHIP_FILE
    )

    required = {
        "Decision Date",
        "Underlying RIC",
        "Candidate Index Member",
        "Membership Source",
        "Same-Day Event Timing Unverified",
    }

    missing = required - set(panel.columns)

    if missing:
        raise ValueError(
            f"Missing membership columns: "
            f"{sorted(missing)}"
        )

    panel = panel.copy()

    panel["Decision Date"] = pd.to_datetime(
        panel["Decision Date"],
        errors="raise",
    ).dt.normalize()

    if panel["Decision Date"].isna().any():
        raise ValueError(
            "Missing decision dates"
        )

    panel["Underlying RIC"] = (
        panel["Underlying RIC"]
        .astype("string")
        .str.strip()
    )

    if (
        panel["Underlying RIC"].isna().any()
        or panel["Underlying RIC"].eq("").any()
    ):
        raise ValueError(
            "Missing equity RICs"
        )

    key = [
        "Decision Date",
        "Underlying RIC",
    ]

    if panel.duplicated(key).any():
        raise ValueError(
            "Duplicate membership-date keys"
        )

    if not pd.api.types.is_bool_dtype(
        panel["Candidate Index Member"]
    ):
        raise ValueError(
            "Candidate Index Member must be boolean"
        )

    members = panel.loc[
        panel["Candidate Index Member"].eq(True)
    ].copy()

    members = members.sort_values(
        ["Underlying RIC", "Decision Date"]
    )

    groups = list(
        members.groupby(
            "Underlying RIC",
            sort=True,
        )
    )

    audited_rows = []
    summaries = []
    invalid_rows = []
    unusual_rows = []
    missing_files = []

    print(
        "\n--- POINT-IN-TIME EQUITY PRICE AUDIT ---"
    )

    print(
        "Membership-date rows:",
        len(members),
    )

    print(
        "Equity RICs:",
        len(groups),
    )

    print(
        "Price convention: strictly before "
        "decision date"
    )

    for i, (ric, part) in enumerate(
        groups, 1
    ):
        ric = str(ric)

        try:
            (
                audited,
                summary,
                invalid,
                unusual,
            ) = audit_ric(
                ric, part
            )

        except FileNotFoundError as error:
            missing_files.append(
                (ric, str(error))
            )

            print(
                f"Missing history file: {ric}"
            )

            continue

        audited_rows.append(audited)
        summaries.append(summary)

        if not invalid.empty:
            invalid_rows.append(invalid)

        if not unusual.empty:
            unusual_rows.append(unusual)

        if (
            i % 100 == 0
            or i == len(groups)
        ):
            print(
                f"Audited {i}/{len(groups)} RICs"
            )

    if missing_files:
        raise RuntimeError(
            f"{len(missing_files)} history files "
            "are missing. Example: "
            f"{missing_files[:3]}"
        )

    decisions = pd.concat(
        audited_rows,
        ignore_index=True,
    )

    stats = pd.DataFrame(
        summaries
    ).sort_values("RIC")

    invalid_closes = (
        pd.concat(
            invalid_rows,
            ignore_index=True,
        )
        if invalid_rows
        else pd.DataFrame(
            columns=["RIC", "Date", "Close"]
        )
    )

    unusual_moves = (
        pd.concat(
            unusual_rows,
            ignore_index=True,
        )
        if unusual_rows
        else pd.DataFrame(
            columns=[
                "RIC",
                "Date",
                "Close",
                "Daily Adjusted Price Return",
            ]
        )
    )

    if len(decisions) != len(members):
        raise ValueError(
            "Membership-date rows were lost"
        )

    if decisions.duplicated(key).any():
        raise ValueError(
            "Duplicate audited keys"
        )

    by_date = (
        decisions.groupby(
            "Decision Date",
            sort=True,
        )
        .agg(
            Members=(
                "Underlying RIC",
                "size",
            ),
            Have_Prior_Price=(
                "Has Prior Price",
                "sum",
            ),
            Have_Recent_Price=(
                "Recent Prior Price",
                "sum",
            ),
            Feature_Candidates=(
                "253-Price Feature Candidate",
                "sum",
            ),
            Same_Day_Event_Flags=(
                "Same-Day Event Timing Unverified",
                "sum",
            ),
        )
        .reset_index()
    )

    by_date["Feature Candidate Share"] = (
        by_date["Feature_Candidates"]
        / by_date["Members"]
    )

    stats["Unavailable Feature Dates"] = (
        stats["Membership Decision Dates"]
        - stats["Feature Candidate Dates"]
    )

    OUTPUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    decisions.to_parquet(
        OUTPUT
        / "equity_member_decision_price_audit.parquet",
        index=False,
    )

    stats.to_csv(
        OUTPUT
        / "equity_ric_price_coverage.csv",
        index=False,
    )

    by_date.to_csv(
        OUTPUT
        / "equity_decision_date_coverage.csv",
        index=False,
    )

    invalid_closes.to_csv(
        OUTPUT
        / "equity_invalid_close_observations.csv",
        index=False,
    )

    unusual_moves.to_csv(
        OUTPUT
        / "equity_large_adjusted_price_moves.csv",
        index=False,
    )

    print("\n--- SUMMARY ---")

    print(
        "RIC files audited:",
        len(stats),
    )

    print(
        "Member-date rows:",
        len(decisions),
    )

    print(
        "With a prior price:",
        int(
            decisions[
                "Has Prior Price"
            ].sum()
        ),
    )

    print(
        "With a recent prior price:",
        int(
            decisions[
                "Recent Prior Price"
            ].sum()
        ),
    )

    print(
        "253-price feature candidates:",
        int(
            decisions[
                "253-Price Feature Candidate"
            ].sum()
        ),
    )

    print(
        "Invalid close observations:",
        len(invalid_closes),
    )

    print(
        "Large adjusted daily moves:",
        len(unusual_moves),
    )

    print(
        "\nWeakest five decision dates:"
    )

    print(
        by_date.nsmallest(
            5,
            "Feature Candidate Share",
        ).to_string(index=False)
    )

    print(
        "\nRICs with most unavailable feature dates:"
    )

    print(
        stats.nlargest(
            10,
            "Unavailable Feature Dates",
        )[
            [
                "RIC",
                "Membership Decision Dates",
                "Unavailable Feature Dates",
                "First Positive Close",
                "Last Positive Close",
            ]
        ].to_string(index=False)
    )

    print("\nReports:", OUTPUT)

    print(
        "No prices were forward-filled. "
        "No ML labels were created."
    )


if __name__ == "__main__":
    main()
