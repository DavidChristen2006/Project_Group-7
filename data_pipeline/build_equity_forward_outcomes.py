
"""Construct forward 63-trading-session stock-risk outcomes.

This is a research-label calculation, not realized trading P&L.
No LSEG requests, forward fills, or original-file modifications.
"""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd


HORIZON = 63
BENCHMARK = "SPY.P"
PRICE_PERIOD = "2013-09-01_2025-04-30.parquet"
EXTREME_RETURN = 0.50
BREACH_LEVELS = (0.05, 0.10, 0.15, 0.20)
KEY = ["Decision Date", "Underlying RIC"]


def require_boolean(series: pd.Series, name: str) -> pd.Series:
    """Perform the require boolean operation for this research workflow.

    Args:
        series: Input to the require_boolean operation.
        name: Input to the require_boolean operation.

    Returns:
        Computed result or status for the caller.
    """
    values = series.astype("string").str.strip().str.lower()
    parsed = values.map({
        "true": True,
        "false": False,
        "1": True,
        "0": False,
    })

    if parsed.isna().any():
        raise ValueError(
            f"{name} contains null or invalid Boolean values"
        )

    return parsed.astype(bool)


def price_file(root: Path, ric: str) -> Path:
    """Perform the price file operation for this research workflow.

    Args:
        root: Input to the price_file operation.
        ric: Input to the price_file operation.

    Returns:
        Computed result or status for the caller.
    """
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", ric)
    digest = hashlib.sha256(
        ric.encode("utf-8")
    ).hexdigest()[:8]

    return (
        root
        / f"{safe}_{digest}"
        / "adjusted"
        / PRICE_PERIOD
    )


def load_close(root: Path, ric: str) -> pd.Series:
    """Load close from stored research artifacts.

    Args:
        root: Input to the load_close operation.
        ric: Input to the load_close operation.

    Returns:
        Computed result or status for the caller.
    """
    path = price_file(root, ric)

    if not path.is_file():
        raise FileNotFoundError(
            f"Missing history for {ric}: {path}"
        )

    frame = pd.read_parquet(path)

    required = {
        "Date",
        "Close",
        "RIC",
        "Adjustment Mode",
    }

    missing = required - set(frame.columns)

    if missing:
        raise ValueError(
            f"{ric} missing columns: {sorted(missing)}"
        )

    if not frame["RIC"].astype("string").eq(ric).all():
        raise ValueError(
            f"RIC metadata mismatch for {ric}"
        )

    if not frame["Adjustment Mode"].astype(
        "string"
    ).eq("adjusted").all():
        raise ValueError(
            f"Adjustment metadata mismatch for {ric}"
        )

    dates = pd.to_datetime(
        frame["Date"],
        errors="raise",
    ).dt.normalize()

    if dates.isna().any() or dates.duplicated().any():
        raise ValueError(
            f"Invalid/duplicate historical dates: {ric}"
        )

    close = pd.to_numeric(
        frame["Close"],
        errors="coerce",
    ).to_numpy(dtype=float)

    close[
        ~np.isfinite(close) | (close <= 0)
    ] = np.nan

    result = pd.Series(
        close,
        index=pd.DatetimeIndex(dates),
        name="Close",
    )

    return result.sort_index()


def max_drawdown(prices: np.ndarray) -> float:
    """Perform the max drawdown operation for this research workflow.

    Args:
        prices: Input to the max_drawdown operation.

    Returns:
        Computed result or status for the caller.
    """
    running_peak = np.maximum.accumulate(prices)

    return float(
        np.min(prices / running_peak - 1.0)
    )


def outcome_values(
    stock: np.ndarray,
    market: np.ndarray,
) -> dict:

    """Perform the outcome values operation for this research workflow.

    Args:
        stock: Input to the outcome_values operation.
        market: Input to the outcome_values operation.

    Returns:
        Computed result or status for the caller.
    """
    stock_returns = (
        stock[1:] / stock[:-1] - 1.0
    )

    market_return = (
        market[-1] / market[0] - 1.0
    )

    path_from_entry = (
        stock / stock[0] - 1.0
    )

    worst = float(np.min(path_from_entry))
    best = float(np.max(path_from_entry))
    final = float(path_from_entry[-1])

    result = {
        "target_entry_adjusted_close": float(
            stock[0]
        ),
        "target_exit_adjusted_close": float(
            stock[-1]
        ),
        "target_return_63d": final,
        "target_worst_return_from_entry_63d": worst,
        "target_best_return_from_entry_63d": best,
        "target_max_drawdown_63d": max_drawdown(
            stock
        ),
        "target_max_loss_from_entry_63d": max(
            0.0, -worst
        ),
        "target_realized_vol_63d": float(
            np.std(stock_returns, ddof=1)
            * np.sqrt(252)
        ),
        "target_spy_return_63d": float(
            market_return
        ),
        "target_spy_max_drawdown_63d": (
            max_drawdown(market)
        ),
        "target_excess_return_spy_63d": (
            final - market_return
        ),
        "target_extreme_return_count_63d": int(
            np.count_nonzero(
                np.abs(stock_returns)
                >= EXTREME_RETURN
            )
        ),
    }

    for threshold in BREACH_LEVELS:
        pct = int(round(threshold * 100))

        result[
            f"target_breached_loss_{pct}pct_63d"
        ] = bool(
            worst <= -threshold
        )

    return result


def main() -> None:
    """Run the module entry point and its workflow.
    """
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=0,
    )

    args = parser.parse_args()

    if args.limit < 0:
        parser.error(
            "--limit must be nonnegative"
        )

    root = args.project_root.resolve()

    input_file = (
        root
        / "data/processed/research_features/"
        "equity_features_v1/"
        "point_in_time_equity_features.parquet"
    )

    series_root = (
        root
        / "data/processed/market_history_v2/series"
    )

    output_root = (
        root
        / "data/processed/research_outcomes/"
        "equity_forward_63d_v1"
    )

    if not input_file.is_file():
        raise FileNotFoundError(input_file)

    features = pd.read_parquet(input_file)

    needed = {
        "Decision Date",
        "Underlying RIC",
        "Feature Candidate",
        "Membership Source",
        "Provisional Membership",
        "Same-Day Event Timing Unverified",
    }

    missing = needed - set(features.columns)

    if missing:
        raise ValueError(
            f"Missing feature-panel fields: "
            f"{sorted(missing)}"
        )

    features["Decision Date"] = pd.to_datetime(
        features["Decision Date"],
        errors="raise",
    ).dt.normalize()

    features["Underlying RIC"] = (
        features["Underlying RIC"]
        .astype("string")
        .str.strip()
    )

    if (
        features["Decision Date"].isna().any()
        or features["Underlying RIC"].isna().any()
        or features["Underlying RIC"].eq("").any()
        or features.duplicated(KEY).any()
    ):
        raise ValueError(
            "Feature panel contains null or duplicate keys"
        )

    for flag in (
        "Feature Candidate",
        "Provisional Membership",
        "Same-Day Event Timing Unverified",
    ):
        features[flag] = require_boolean(
            features[flag],
            flag,
        )

    # Historical predictor prices must predate
    # each research decision.
    for column in (
        "Last Prior Equity Price Date",
        "Last Benchmark Trading Date",
    ):
        if column in features.columns:
            predictor_dates = pd.to_datetime(
                features[column],
                errors="coerce",
            )

            invalid_timing = (
                predictor_dates.notna()
                & (
                    predictor_dates
                    >= features["Decision Date"]
                )
            )

            if invalid_timing.any():
                raise ValueError(
                    f"Potential predictor lookahead: {column}"
                )

    if args.limit:
        selected = (
            features["Underlying RIC"]
            .drop_duplicates()
            .sort_values()
            .head(args.limit)
        )

        features = features.loc[
            features["Underlying RIC"].isin(
                selected
            )
        ].copy()

        output_root = (
            output_root
            / f"pilot_{len(selected)}_rics"
        )

    benchmark = load_close(
        series_root,
        BENCHMARK,
    )

    if benchmark.empty or benchmark.isna().any():
        raise ValueError(
            "SPY benchmark must have a complete "
            "trading-session calendar"
        )

    sessions = benchmark.index

    market_close = benchmark.to_numpy(
        dtype=float
    )

    records = []
    anomaly_rows = []
    history_metadata = []

    groups = features.groupby(
        "Underlying RIC",
        sort=True,
    )

    total_ric = features[
        "Underlying RIC"
    ].nunique()

    print(
        "\n--- 63-TRADING-DAY FORWARD "
        "STOCK-OUTCOME COLLECTION ---"
    )

    print("Rows:", len(features))
    print("Equity RICs:", total_ric)
    print("Benchmark:", BENCHMARK)
    print("Entry: decision-date adjusted close")
    print(
        "Exit: 63 benchmark trading sessions "
        "after entry"
    )

    for i, (ric, part) in enumerate(
        groups, start=1
    ):
        ric = str(ric)

        prices = load_close(
            series_root,
            ric,
        )

        aligned = (
            prices.reindex(sessions)
            .to_numpy(dtype=float)
        )

        valid_source = prices.dropna()

        last_valid_quote = (
            valid_source.index.max()
            if not valid_source.empty
            else pd.NaT
        )

        unexpected_dates = int(
            (~prices.index.isin(sessions)).sum()
        )

        history_metadata.append({
            "RIC": ric,
            "Last Valid Historical Quote": (
                last_valid_quote
            ),
            "Equity Dates Outside SPY Calendar": (
                unexpected_dates
            ),
        })

        positions = sessions.get_indexer(
            part["Decision Date"]
        )

        for (_, item), entry in zip(
            part.iterrows(),
            positions,
        ):
            date = pd.Timestamp(
                item["Decision Date"]
            )

            record = {
                "Decision Date": date,
                "Underlying RIC": ric,
                "Feature Candidate": bool(
                    item["Feature Candidate"]
                ),
                "Membership Source": str(
                    item["Membership Source"]
                ),
                "Provisional Membership": bool(
                    item["Provisional Membership"]
                ),
                "Same-Day Event Timing Unverified": bool(
                    item[
                        "Same-Day Event Timing Unverified"
                    ]
                ),
                "target_entry_date": date,
                "target_exit_date": pd.NaT,
                "target_status": "",
                "target_missing_stock_sessions_63d": (
                    pd.NA
                ),
                "target_first_missing_stock_date": (
                    pd.NaT
                ),
                "target_possible_trading_cessation": (
                    False
                ),
                "target_extreme_return_review": False,
            }

            if entry < 0:
                record["target_status"] = (
                    "NO_BENCHMARK_SESSION_ON_DECISION_DATE"
                )

            elif entry + HORIZON >= len(sessions):
                record["target_status"] = (
                    "INSUFFICIENT_FUTURE_MARKET_SESSIONS"
                )

            else:
                end = entry + HORIZON

                record["target_exit_date"] = (
                    sessions[end]
                )

                stock_window = aligned[
                    entry:end + 1
                ]

                market_window = market_close[
                    entry:end + 1
                ]

                valid = (
                    np.isfinite(stock_window)
                    & (stock_window > 0)
                )

                if not valid[0]:
                    record["target_status"] = (
                        "MISSING_ENTRY_CLOSE"
                    )

                elif not valid.all():
                    missing_positions = np.flatnonzero(
                        ~valid
                    )

                    record["target_status"] = (
                        "INCOMPLETE_FORWARD_PRICE_PATH"
                    )

                    record[
                        "target_missing_stock_sessions_63d"
                    ] = len(missing_positions)

                    record[
                        "target_first_missing_stock_date"
                    ] = sessions[
                        entry + int(missing_positions[0])
                    ]

                    record[
                        "target_possible_trading_cessation"
                    ] = bool(
                        pd.notna(last_valid_quote)
                        and last_valid_quote
                        < sessions[end]
                    )

                else:
                    record["target_status"] = (
                        "COMPLETE_63D_PRICE_PATH"
                    )

                    record[
                        "target_missing_stock_sessions_63d"
                    ] = 0

                    record.update(
                        outcome_values(
                            stock_window,
                            market_window,
                        )
                    )

                    extreme_count = record[
                        "target_extreme_return_count_63d"
                    ]

                    record[
                        "target_extreme_return_review"
                    ] = bool(
                        extreme_count > 0
                    )

                    if extreme_count:
                        daily_returns = (
                            stock_window[1:]
                            / stock_window[:-1]
                            - 1.0
                        )

                        extreme_positions = np.flatnonzero(
                            np.abs(daily_returns)
                            >= EXTREME_RETURN
                        )

                        for offset in extreme_positions:
                            anomaly_rows.append({
                                "Decision Date": date,
                                "RIC": ric,
                                "Extreme Move Date": sessions[
                                    entry + int(offset) + 1
                                ],
                                "Daily Adjusted Return": float(
                                    daily_returns[offset]
                                ),
                            })

            record["Complete Forward Outcome"] = bool(
                record["target_status"]
                == "COMPLETE_63D_PRICE_PATH"
            )

            record[
                "Price Feature And Outcome Candidate"
            ] = bool(
                record["Feature Candidate"]
                and record["Complete Forward Outcome"]
            )

            record[
                "Conservative Observed-Snapshot Candidate"
            ] = bool(
                record[
                    "Price Feature And Outcome Candidate"
                ]
                and record["Membership Source"]
                == "OBSERVED_SNAPSHOT_MATCH"
                and not record[
                    "Same-Day Event Timing Unverified"
                ]
            )

            records.append(record)

        if i % 100 == 0 or i == total_ric:
            print(
                f"Processed {i}/{total_ric} RICs",
                flush=True,
            )

    result = pd.DataFrame.from_records(
        records
    )

    result = result.sort_values(
        KEY
    ).reset_index(drop=True)

    if (
        len(result) != len(features)
        or result.duplicated(KEY).any()
    ):
        raise AssertionError(
            "Forward outcome keys do not match "
            "the feature panel"
        )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    reports = output_root / "reports"

    reports.mkdir(exist_ok=True)

    result.to_parquet(
        output_root
        / "point_in_time_equity_forward_outcomes.parquet",
        index=False,
    )

    result.groupby(
        "target_status",
        dropna=False,
    ).size().rename(
        "Rows"
    ).to_csv(
        reports / "forward_outcome_status.csv"
    )

    incomplete = result.loc[
        ~result["Complete Forward Outcome"]
    ]

    incomplete.to_csv(
        reports / "incomplete_forward_outcomes.csv",
        index=False,
    )

    pd.DataFrame(
        anomaly_rows,
        columns=[
            "Decision Date",
            "RIC",
            "Extreme Move Date",
            "Daily Adjusted Return",
        ],
    ).to_csv(
        reports / "large_moves_in_forward_windows.csv",
        index=False,
    )

    pd.DataFrame(
        history_metadata
    ).to_csv(
        reports / "history_calendar_coverage.csv",
        index=False,
    )

    by_date = result.groupby(
        "Decision Date",
        as_index=False,
    ).agg(
        Members=(
            "Underlying RIC",
            "size",
        ),
        Complete_Labels=(
            "Complete Forward Outcome",
            "sum",
        ),
        Feature_And_Outcome_Candidates=(
            "Price Feature And Outcome Candidate",
            "sum",
        ),
        Conservative_Candidates=(
            "Conservative Observed-Snapshot Candidate",
            "sum",
        ),
        Possible_Trading_Cessations=(
            "target_possible_trading_cessation",
            "sum",
        ),
        Outcome_Anomaly_Flags=(
            "target_extreme_return_review",
            "sum",
        ),
    )

    by_date.to_csv(
        reports / "coverage_by_decision_date.csv",
        index=False,
    )

    print("\n--- FORWARD OUTCOME SUMMARY ---")

    print(
        "Total member-date rows:",
        len(result),
    )

    print("\nOutcome status counts:")

    print(
        result["target_status"]
        .value_counts()
        .to_string()
    )

    print(
        "\nComplete future paths:",
        int(
            result["Complete Forward Outcome"].sum()
        ),
    )

    print(
        "Feature + outcome candidates:",
        int(
            result[
                "Price Feature And Outcome Candidate"
            ].sum()
        ),
    )

    print(
        "Conservative observed-snapshot candidates:",
        int(
            result[
                "Conservative Observed-Snapshot Candidate"
            ].sum()
        ),
    )

    print(
        "Possible trading cessation flags:",
        int(
            result[
                "target_possible_trading_cessation"
            ].sum()
        ),
    )

    print(
        "Forward-window extreme return flags:",
        int(
            result[
                "target_extreme_return_review"
            ].sum()
        ),
    )

    print("\nOutput:", output_root)

    print(
        "Research stock-price-path outcomes only; "
        "not actual hedge P&L."
    )


if __name__ == "__main__":
    main()
