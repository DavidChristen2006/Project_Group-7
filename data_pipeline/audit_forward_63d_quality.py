
"""Validate 63-session research outcomes and isolate missing-path cases.

Reads only previously generated Parquet/CSV files.
Never modifies them.
"""

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "processed"
)

OUTCOMES = ROOT / "research_outcomes/equity_forward_63d_v1"

FEATURES = ROOT / "research_features/equity_features_v1"

OUTPUT = ROOT / "research_audit/forward_63d_quality"

KEY = ["Decision Date", "Underlying RIC"]


def bool_column(
    series: pd.Series,
    name: str,
) -> pd.Series:

    values = (
        series.astype("string")
        .str.lower()
        .str.strip()
    )

    mapped = values.map({
        "true": True,
        "false": False,
        "1": True,
        "0": False,
    })

    if mapped.isna().any():
        raise ValueError(
            f"Invalid or missing Boolean flag: {name}"
        )

    return mapped.astype(bool)


def dates(series: pd.Series) -> pd.Series:
    return pd.to_datetime(
        series,
        errors="raise",
    ).dt.normalize()


def assert_consistent(frame: pd.DataFrame) -> None:

    flags = [
        "Feature Candidate",
        "Complete Forward Outcome",
        "Price Feature And Outcome Candidate",
        "Conservative Observed-Snapshot Candidate",
        "Provisional Membership",
        "Same-Day Event Timing Unverified",
        "target_possible_trading_cessation",
        "target_extreme_return_review",
    ]

    for name in flags:
        frame[name] = bool_column(
            frame[name],
            name,
        )

    expected_complete = frame[
        "target_status"
    ].eq("COMPLETE_63D_PRICE_PATH")

    if not frame[
        "Complete Forward Outcome"
    ].equals(expected_complete):
        raise ValueError(
            "Complete outcome flag contradicts status"
        )

    expected_candidate = (
        frame["Feature Candidate"]
        & expected_complete
    )

    if not frame[
        "Price Feature And Outcome Candidate"
    ].equals(expected_candidate):
        raise ValueError(
            "Feature/outcome candidate flag is inconsistent"
        )

    conservative = (
        expected_candidate
        & frame["Membership Source"].eq(
            "OBSERVED_SNAPSHOT_MATCH"
        )
        & ~frame[
            "Same-Day Event Timing Unverified"
        ]
    )

    if not frame[
        "Conservative Observed-Snapshot Candidate"
    ].equals(conservative):
        raise ValueError(
            "Conservative candidate flag is inconsistent"
        )

    invalid_cessation = (
        frame["target_possible_trading_cessation"]
        & ~frame["target_status"].eq(
            "INCOMPLETE_FORWARD_PRICE_PATH"
        )
    )

    if invalid_cessation.any():
        raise ValueError(
            "Possible cessation flagged on another status"
        )

    invalid_extreme = (
        frame["target_extreme_return_review"]
        & ~expected_complete
    )

    if invalid_extreme.any():
        raise ValueError(
            "Extreme-return flag on incomplete outcome"
        )

    complete = frame.loc[
        expected_complete
    ].copy()

    if complete.empty:
        raise ValueError(
            "No complete forward outcomes"
        )

    numeric_cols = [
        "target_entry_adjusted_close",
        "target_exit_adjusted_close",
        "target_return_63d",
        "target_worst_return_from_entry_63d",
        "target_best_return_from_entry_63d",
        "target_max_drawdown_63d",
        "target_max_loss_from_entry_63d",
        "target_realized_vol_63d",
        "target_spy_return_63d",
        "target_excess_return_spy_63d",
    ]

    for name in numeric_cols:
        complete[name] = pd.to_numeric(
            complete[name],
            errors="coerce",
        )

    if not np.isfinite(
        complete[numeric_cols].to_numpy(
            dtype=float
        )
    ).all():
        raise ValueError(
            "Nonfinite values among complete targets"
        )

    tol = 1e-9

    entry = complete[
        "target_entry_adjusted_close"
    ]

    exit_ = complete[
        "target_exit_adjusted_close"
    ]

    final = complete[
        "target_return_63d"
    ]

    low = complete[
        "target_worst_return_from_entry_63d"
    ]

    high = complete[
        "target_best_return_from_entry_63d"
    ]

    dd = complete[
        "target_max_drawdown_63d"
    ]

    loss = complete[
        "target_max_loss_from_entry_63d"
    ]

    excess = complete[
        "target_excess_return_spy_63d"
    ]

    checks = {
        "Nonpositive entry or exit": (
            (entry <= 0) | (exit_ <= 0)
        ),
        "Incorrect final return": (
            final - (exit_ / entry - 1)
        ).abs() > tol,
        "Final outside path extrema": (
            (final < low - tol)
            | (final > high + tol)
        ),
        "Invalid best/worst returns": (
            (high < -tol) | (low > tol)
        ),
        "Invalid maximum drawdown": (
            (dd > tol)
            | (dd > low + tol)
        ),
        "Incorrect maximum entry loss": (
            loss - (-low).clip(lower=0)
        ).abs() > tol,
        "Incorrect excess return": (
            excess
            - (
                final
                - complete["target_spy_return_63d"]
            )
        ).abs() > tol,
    }

    failures = {
        name: int(mask.sum())
        for name, mask in checks.items()
        if mask.any()
    }

    if failures:
        raise ValueError(
            f"Forward-outcome checks failed: {failures}"
        )

    invalid_exit = (
        frame["target_exit_date"].notna()
        & (
            frame["target_exit_date"]
            <= frame["Decision Date"]
        )
    )

    if invalid_exit.any():
        raise ValueError(
            "Exit date is not after decision date"
        )

    if not frame[
        "target_entry_date"
    ].equals(frame["Decision Date"]):
        raise ValueError(
            "Target entry date differs from decision date"
        )


def main() -> None:

    source = (
        OUTCOMES
        / "point_in_time_equity_forward_outcomes.parquet"
    )

    feature_file = (
        FEATURES
        / "point_in_time_equity_features.parquet"
    )

    metadata_file = (
        OUTCOMES
        / "reports/history_calendar_coverage.csv"
    )

    moves_file = (
        OUTCOMES
        / "reports/large_moves_in_forward_windows.csv"
    )

    for path in (
        source,
        feature_file,
        metadata_file,
        moves_file,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    results = pd.read_parquet(source)

    for column in (
        "Decision Date",
        "target_entry_date",
        "target_exit_date",
        "target_first_missing_stock_date",
    ):
        results[column] = dates(
            results[column]
        )

    results["Underlying RIC"] = (
        results["Underlying RIC"]
        .astype("string")
        .str.strip()
    )

    if (
        results[KEY].isna().any().any()
        or results.duplicated(KEY).any()
    ):
        raise ValueError(
            "Outcome keys missing or duplicated"
        )

    features = pd.read_parquet(
        feature_file
    )

    features["Decision Date"] = dates(
        features["Decision Date"]
    )

    if (
        features[KEY].isna().any().any()
        or features.duplicated(KEY).any()
    ):
        raise ValueError(
            "Feature keys missing or duplicated"
        )

    if len(results) != len(features):
        raise ValueError(
            "Feature and outcome row counts differ"
        )

    key_check = results[KEY].merge(
        features[KEY],
        on=KEY,
        how="outer",
        indicator=True,
        validate="one_to_one",
    )

    if not key_check["_merge"].eq(
        "both"
    ).all():
        raise ValueError(
            "Feature and outcome keys differ"
        )

    assert_consistent(results)

    metadata = pd.read_csv(
        metadata_file
    )

    if metadata["RIC"].duplicated().any():
        raise ValueError(
            "Duplicate RICs in history metadata"
        )

    metadata["Last Valid Historical Quote"] = (
        pd.to_datetime(
            metadata[
                "Last Valid Historical Quote"
            ],
            errors="coerce",
        )
    )

    incomplete = results.loc[
        ~results["Complete Forward Outcome"]
    ].copy()

    incomplete = incomplete.merge(
        metadata[
            [
                "RIC",
                "Last Valid Historical Quote",
            ]
        ],
        left_on="Underlying RIC",
        right_on="RIC",
        how="left",
        validate="many_to_one",
    ).drop(columns="RIC")

    incomplete["Review Category"] = np.select(
        [
            incomplete["target_status"].eq(
                "MISSING_ENTRY_CLOSE"
            ),
            incomplete[
                "target_possible_trading_cessation"
            ],
            incomplete["target_status"].eq(
                "INCOMPLETE_FORWARD_PRICE_PATH"
            ),
        ],
        [
            "MISSING_ENTRY_PRICE",
            "POSSIBLE_TERMINAL_HISTORY",
            "NONTERMINAL_OR_INTERNAL_GAP",
        ],
        default="OTHER_INCOMPLETE_OUTCOME",
    )

    incomplete = (
        incomplete
        .sort_values(KEY)
        .reset_index(drop=True)
    )

    by_ric = (
        incomplete.groupby(
            ["Underlying RIC", "Review Category"],
            dropna=False,
            as_index=False,
        )
        .agg(
            Affected_Decisions=(
                "Decision Date",
                "size",
            ),
            First_Affected_Decision=(
                "Decision Date",
                "min",
            ),
            Last_Affected_Decision=(
                "Decision Date",
                "max",
            ),
            First_Missing_Price_Date=(
                "target_first_missing_stock_date",
                "min",
            ),
            Last_Valid_Quote=(
                "Last Valid Historical Quote",
                "max",
            ),
        )
        .sort_values(
            [
                "Affected_Decisions",
                "Underlying RIC",
            ],
            ascending=[False, True],
        )
    )

    moves = pd.read_csv(
        moves_file
    )

    if not moves.empty:
        moves["Extreme Move Date"] = pd.to_datetime(
            moves["Extreme Move Date"]
        )

        move_summary = (
            moves.groupby(
                ["RIC", "Extreme Move Date"],
                as_index=False,
            )
            .agg(
                Affected_Decision_Windows=(
                    "Decision Date",
                    "nunique",
                ),
                Daily_Adjusted_Return=(
                    "Daily Adjusted Return",
                    "first",
                ),
            )
            .sort_values(
                "Affected_Decision_Windows",
                ascending=False,
            )
        )

    else:
        move_summary = pd.DataFrame(
            columns=[
                "RIC",
                "Extreme Move Date",
                "Affected_Decision_Windows",
                "Daily_Adjusted_Return",
            ]
        )

    by_date = results.groupby(
        "Decision Date",
        as_index=False,
    ).agg(
        Members=(
            "Underlying RIC",
            "size",
        ),
        Complete_Outcomes=(
            "Complete Forward Outcome",
            "sum",
        ),
        Feature_And_Outcome=(
            "Price Feature And Outcome Candidate",
            "sum",
        ),
        Conservative_Subset=(
            "Conservative Observed-Snapshot Candidate",
            "sum",
        ),
        Possible_Terminal=(
            "target_possible_trading_cessation",
            "sum",
        ),
    )

    OUTPUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    incomplete.to_csv(
        OUTPUT / "forward_cases_for_review.csv",
        index=False,
    )

    by_ric.to_csv(
        OUTPUT / "forward_cases_by_ric.csv",
        index=False,
    )

    move_summary.to_csv(
        OUTPUT / "unique_extreme_forward_moves.csv",
        index=False,
    )

    by_date.to_csv(
        OUTPUT / "forward_coverage_by_date.csv",
        index=False,
    )

    print(
        "\n--- 63-DAY FORWARD-OUTCOME QUALITY AUDIT ---"
    )

    print(
        "Total membership-date rows:",
        len(results),
    )

    print(
        "Complete future paths:",
        int(
            results[
                "Complete Forward Outcome"
            ].sum()
        ),
    )

    print(
        "Feature + outcome candidates:",
        int(
            results[
                "Price Feature And Outcome Candidate"
            ].sum()
        ),
    )

    print(
        "Conservative candidates:",
        int(
            results[
                "Conservative Observed-Snapshot Candidate"
            ].sum()
        ),
    )

    print("\nIncomplete path categories:")

    print(
        incomplete[
            "Review Category"
        ].value_counts().to_string()
    )

    print(
        "Unique RICs needing path review:",
        incomplete["Underlying RIC"].nunique(),
    )

    print(
        "Unique RICs with possible terminal history:",
        incomplete.loc[
            incomplete["Review Category"].eq(
                "POSSIBLE_TERMINAL_HISTORY"
            ),
            "Underlying RIC",
        ].nunique(),
    )

    print(
        "Unique extreme move dates/RIC pairs:",
        len(move_summary),
    )

    print("\nMost affected securities:")

    print(
        by_ric.head(15).to_string(
            index=False
        )
    )

    print("\nData integrity checks: PASS")
    print("Audit reports:", OUTPUT)

    print(
        "No targets, source prices, or membership "
        "records were modified."
    )


if __name__ == "__main__":
    main()
