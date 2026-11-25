
"""Build a point-in-time S&P 500 equity risk-feature panel from local adjusted prices.

No market-data API calls, forward fills, future labels, or cross-RIC substitutions.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


HORIZONS = (21, 63, 126, 252)
BENCHMARK_RIC = "SPY.P"
PRICE_WINDOW = "2013-09-01_2025-04-30.parquet"
MAX_PRICE_AGE_DAYS = 7
LARGE_MOVE_THRESHOLD = 0.50
ANNUALIZATION = 252


def parse_bool(values: pd.Series, name: str) -> pd.Series:
    """Perform the parse bool operation for this research workflow.

    Args:
        values: Input to the parse_bool operation.
        name: Input to the parse_bool operation.

    Returns:
        Computed result or status for the caller.
    """
    parsed = (
        values.astype("string")
        .str.strip()
        .str.lower()
        .map({"true": True, "false": False, "1": True, "0": False})
    )
    if parsed.isna().any():
        raise ValueError(f"{name}: contains missing or non-Boolean values")
    return parsed.astype(bool)


def load_membership(path: Path) -> pd.DataFrame:
    """Load membership from stored research artifacts.

    Args:
        path: Input to the load_membership operation.

    Returns:
        Computed result or status for the caller.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Membership grid not found: {path}")

    grid = pd.read_parquet(path)
    required = (
        "Decision Date", "Underlying RIC", "Candidate Index Member",
        "Membership Source", "Provisional Membership",
        "Same-Day Event Timing Unverified",
    )
    missing = [name for name in required if name not in grid.columns]
    if missing:
        raise ValueError(f"Missing membership columns: {missing}")

    grid["Decision Date"] = pd.to_datetime(
        grid["Decision Date"], errors="raise"
    ).dt.normalize()

    if grid["Decision Date"].isna().any():
        raise ValueError("Decision Date contains nulls")

    grid["Underlying RIC"] = (
        grid["Underlying RIC"].astype("string").str.strip()
    )

    if (
        grid["Underlying RIC"].isna().any()
        or grid["Underlying RIC"].eq("").any()
    ):
        raise ValueError("Underlying RIC contains blanks")

    if grid.duplicated(["Decision Date", "Underlying RIC"]).any():
        raise ValueError("Duplicate decision-date/RIC keys in membership grid")

    for name in (
        "Candidate Index Member",
        "Provisional Membership",
        "Same-Day Event Timing Unverified",
    ):
        grid[name] = parse_bool(grid[name], name)

    members = grid.loc[grid["Candidate Index Member"]].copy()

    if members.empty:
        raise ValueError("No positive candidate index members")

    return members.sort_values(
        ["Underlying RIC", "Decision Date"]
    ).reset_index(drop=True)


def index_price_files(series_root: Path) -> dict[str, Path]:
    """Perform the index price files operation for this research workflow.

    Args:
        series_root: Input to the index_price_files operation.

    Returns:
        Computed result or status for the caller.
    """
    if not series_root.is_dir():
        raise FileNotFoundError(
            f"Adjusted price root not found: {series_root}"
        )

    lookup = {}

    for path in series_root.glob(f"*/adjusted/{PRICE_WINDOW}"):
        folder = path.parent.parent.name
        suffix = folder.rsplit("_", 1)[-1]

        if len(suffix) != 8:
            continue

        if suffix in lookup:
            raise ValueError(
                f"Ambiguous adjusted price histories for hash {suffix}"
            )

        lookup[suffix] = path

    if not lookup:
        raise FileNotFoundError(
            f"No adjusted histories found for {PRICE_WINDOW} in {series_root}"
        )

    return lookup


def load_prices(
    ric: str,
    files_by_hash: dict[str, Path],
) -> pd.DataFrame:

    """Load prices from stored research artifacts.

    Args:
        ric: Input to the load_prices operation.
        files_by_hash: Input to the load_prices operation.

    Returns:
        Computed result or status for the caller.
    """
    suffix = hashlib.sha256(
        ric.encode("utf-8")
    ).hexdigest()[:8]

    path = files_by_hash.get(suffix)

    if path is None:
        raise FileNotFoundError(
            f"Missing adjusted history for {ric} (hash {suffix})"
        )

    frame = pd.read_parquet(path)

    needed = {"Date", "RIC", "Close"}

    if not needed.issubset(frame.columns):
        raise ValueError(
            f"Missing columns in {path}: "
            f"{sorted(needed - set(frame.columns))}"
        )

    if not frame["RIC"].astype(str).eq(ric).all():
        raise ValueError(
            f"RIC mismatch in {path} (expected {ric})"
        )

    if "Adjustment Mode" in frame.columns:
        if not (
            frame["Adjustment Mode"]
            .astype(str)
            .str.lower()
            .eq("adjusted")
            .all()
        ):
            raise ValueError(
                f"Not an adjusted-price series: {path}"
            )

    frame["Date"] = pd.to_datetime(
        frame["Date"], errors="raise"
    ).dt.normalize()

    if (
        frame["Date"].isna().any()
        or frame["Date"].duplicated().any()
    ):
        raise ValueError(
            f"Invalid or duplicate price dates for {ric}"
        )

    frame["Close"] = pd.to_numeric(
        frame["Close"], errors="coerce"
    )

    frame.loc[
        ~np.isfinite(frame["Close"])
        | (frame["Close"] <= 0),
        "Close",
    ] = np.nan

    return frame.sort_values("Date").set_index("Date")


def calculate_window(
    prices: np.ndarray,
    benchmark: np.ndarray,
    horizon: int,
) -> dict[str, float]:

    """Calculate window using the available data.

    Args:
        prices: Input to the calculate_window operation.
        benchmark: Input to the calculate_window operation.
        horizon: Input to the calculate_window operation.

    Returns:
        Computed result or status for the caller.
    """
    returns = prices[1:] / prices[:-1] - 1.0

    benchmark_returns = (
        benchmark[1:] / benchmark[:-1] - 1.0
    )

    prefix = f"{horizon}d"

    q05 = float(np.quantile(returns, 0.05))

    downside_returns = np.minimum(returns, 0.0)

    running_peak = np.maximum.accumulate(prices)

    features = {
        f"momentum_{prefix}": float(
            prices[-1] / prices[0] - 1.0
        ),
        f"realized_vol_{prefix}": float(
            np.std(returns, ddof=1)
            * np.sqrt(ANNUALIZATION)
        ),
        f"downside_dev_{prefix}": float(
            np.sqrt(np.mean(downside_returns ** 2))
            * np.sqrt(ANNUALIZATION)
        ),
        f"max_drawdown_{prefix}": float(
            np.min(prices / running_peak - 1.0)
        ),
        f"daily_var95_{prefix}": max(
            0.0, -q05
        ),
        f"daily_es95_{prefix}": max(
            0.0,
            -float(np.mean(returns[returns <= q05])),
        ),
    }

    if horizon in (63, 252):
        benchmark_var = float(
            np.var(benchmark_returns, ddof=1)
        )

        stock_var = float(
            np.var(returns, ddof=1)
        )

        features[f"beta_spy_{prefix}"] = (
            float(
                np.cov(
                    returns,
                    benchmark_returns,
                    ddof=1,
                )[0, 1] / benchmark_var
            )
            if benchmark_var > 1e-16
            else np.nan
        )

        features[f"corr_spy_{prefix}"] = (
            float(
                np.corrcoef(
                    returns,
                    benchmark_returns,
                )[0, 1]
            )
            if (
                benchmark_var > 1e-16
                and stock_var > 1e-16
            )
            else np.nan
        )

        features[f"excess_momentum_spy_{prefix}"] = float(
            prices[-1] / prices[0]
            - benchmark[-1] / benchmark[0]
        )

    return features


def build_panel(
    members: pd.DataFrame,
    files_by_hash: dict[str, Path],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    """Construct panel from the supplied inputs.

    Args:
        members: Input to the build_panel operation.
        files_by_hash: Input to the build_panel operation.

    Returns:
        Computed result or status for the caller.
    """
    spy = load_prices(
        BENCHMARK_RIC,
        files_by_hash,
    )

    if spy["Close"].isna().any():
        raise ValueError(
            "SPY.P benchmark has invalid closing prices; "
            "cannot establish trading calendar"
        )

    calendar = spy.index

    calendar_dates = calendar.to_numpy(
        dtype="datetime64[ns]"
    )

    benchmark_prices = spy["Close"].to_numpy(
        dtype=float
    )

    if len(calendar) <= max(HORIZONS):
        raise ValueError(
            "Insufficient SPY.P benchmark history"
        )

    records = []
    anomaly_records = []
    history_quality = []

    groups = members.groupby(
        "Underlying RIC",
        sort=True,
    )

    total = members["Underlying RIC"].nunique()

    for count, (ric, group) in enumerate(groups, start=1):

        source = load_prices(
            str(ric),
            files_by_hash,
        )

        source_price = source["Close"]

        aligned = source_price.reindex(calendar)

        prices = aligned.to_numpy(dtype=float)

        valid = (
            np.isfinite(prices)
            & (prices > 0)
        )

        last_seen = np.maximum.accumulate(
            np.where(
                valid,
                np.arange(len(prices)),
                -1,
            )
        )

        dates = group["Decision Date"].to_numpy(
            dtype="datetime64[ns]"
        )

        cutoffs = (
            np.searchsorted(
                calendar_dates,
                dates,
                side="left",
            ) - 1
        )

        source_field = (
            source["Close Source Field"]
            .reindex(calendar)
            .to_numpy()
            if "Close Source Field" in source.columns
            else None
        )

        # Calculate returns only between adjacent benchmark sessions.
        returns = np.full(len(prices), np.nan)

        both_valid = valid[1:] & valid[:-1]

        local_r = np.full(
            len(prices) - 1,
            np.nan,
        )

        local_r[both_valid] = (
            prices[1:][both_valid]
            / prices[:-1][both_valid]
            - 1.0
        )

        returns[1:] = local_r

        shock_positions = np.flatnonzero(
            np.isfinite(returns)
            & (
                np.abs(returns)
                >= LARGE_MOVE_THRESHOLD
            )
        )

        for pos in shock_positions:
            anomaly_records.append({
                "RIC": str(ric),
                "Date": calendar[pos],
                "Prior Adjusted Close": float(
                    prices[pos - 1]
                ),
                "Adjusted Close": float(
                    prices[pos]
                ),
                "Daily Return": float(
                    returns[pos]
                ),
                "Close Source Field": (
                    source_field[pos]
                    if source_field is not None
                    else pd.NA
                ),
            })

        history_quality.append({
            "RIC": str(ric),
            "Historical Rows": len(source),
            "Invalid Close Observations": int(
                source_price.isna().sum()
            ),
            "Large Adjusted Daily Moves": len(
                shock_positions
            ),
        })

        for row, cutoff, decision_date in zip(
            group.to_dict("records"),
            cutoffs,
            dates,
        ):
            item = dict(row)

            item["Last Benchmark Trading Date"] = (
                calendar[cutoff]
                if cutoff >= 0
                else pd.NaT
            )

            prev = (
                int(last_seen[cutoff])
                if cutoff >= 0
                else -1
            )

            last_date = (
                calendar[prev]
                if prev >= 0
                else pd.NaT
            )

            age_days = (
                int(
                    (
                        pd.Timestamp(decision_date)
                        - last_date
                    ).days
                )
                if prev >= 0
                else np.nan
            )

            item["Last Prior Equity Price Date"] = last_date

            item["Last Prior Adjusted Close"] = (
                float(prices[prev])
                if prev >= 0
                else np.nan
            )

            item["Prior Price Age Days"] = age_days

            item["Recent Prior Price"] = bool(
                prev >= 0
                and age_days <= MAX_PRICE_AGE_DAYS
            )

            item["Complete 253 Session History"] = False

            item["Extreme Move Count 252d"] = pd.NA

            item["Last Close Source Field"] = (
                source_field[prev]
                if (
                    prev >= 0
                    and source_field is not None
                )
                else pd.NA
            )

            for h in HORIZONS:

                if cutoff < h:
                    continue

                stock_window = prices[
                    cutoff - h:cutoff + 1
                ]

                if (
                    not np.isfinite(stock_window).all()
                    or (stock_window <= 0).any()
                ):
                    continue

                market_window = benchmark_prices[
                    cutoff - h:cutoff + 1
                ]

                item.update(
                    calculate_window(
                        stock_window,
                        market_window,
                        h,
                    )
                )

                if h == 252:
                    item["Complete 253 Session History"] = True

                    stock_returns = (
                        stock_window[1:]
                        / stock_window[:-1]
                        - 1.0
                    )

                    item["Extreme Move Count 252d"] = int(
                        np.count_nonzero(
                            np.abs(stock_returns)
                            >= LARGE_MOVE_THRESHOLD
                        )
                    )

            item["Feature Candidate"] = bool(
                item["Complete 253 Session History"]
                and item["Recent Prior Price"]
            )

            item["Verified Membership Feature Candidate"] = bool(
                item["Feature Candidate"]
                and not item["Provisional Membership"]
                and not item["Same-Day Event Timing Unverified"]
            )

            records.append(item)

        if count % 100 == 0 or count == total:
            print(
                f"Built features for {count}/{total} RICs",
                flush=True,
            )

    result = pd.DataFrame.from_records(records)

    result = result.sort_values(
        ["Decision Date", "Underlying RIC"]
    ).reset_index(drop=True)

    if result.duplicated(
        ["Decision Date", "Underlying RIC"]
    ).any():
        raise AssertionError(
            "Duplicate output feature keys"
        )

    if (
        result["Last Prior Equity Price Date"].notna()
        & (
            result["Last Prior Equity Price Date"]
            >= result["Decision Date"]
        )
    ).any():
        raise AssertionError(
            "Lookahead found: equity price on/after decision date"
        )

    if (
        result["Last Benchmark Trading Date"].notna()
        & (
            result["Last Benchmark Trading Date"]
            >= result["Decision Date"]
        )
    ).any():
        raise AssertionError(
            "Lookahead found: benchmark on/after decision date"
        )

    anomalies = pd.DataFrame(
        anomaly_records,
        columns=[
            "RIC",
            "Date",
            "Prior Adjusted Close",
            "Adjusted Close",
            "Daily Return",
            "Close Source Field",
        ],
    )

    return (
        result,
        anomalies,
        pd.DataFrame(history_quality),
    )


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
        help="Process first N sorted RICs; 0 means all",
    )

    args = parser.parse_args()

    if args.limit < 0:
        parser.error("--limit must be >= 0")

    root = args.project_root.resolve()

    grid_path = (
        root
        / "data/processed/historical_universe/"
        "equity_collection_candidate/"
        "equity_membership_grid_candidate.parquet"
    )

    series_root = (
        root
        / "data/processed/market_history_v2/series"
    )

    output_root = (
        root
        / "data/processed/research_features/equity_features_v1"
    )

    members = load_membership(grid_path)

    if args.limit:
        selected = (
            members["Underlying RIC"]
            .drop_duplicates()
            .sort_values()
            .head(args.limit)
        )

        members = members.loc[
            members["Underlying RIC"].isin(selected)
        ].copy()

        output_root = (
            output_root
            / f"pilot_{len(selected)}_rics"
        )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Membership-date rows:", len(members))
    print(
        "Equity RICs:",
        members["Underlying RIC"].nunique(),
    )
    print(
        "Using prices strictly before each decision date"
    )
    print("Benchmark:", BENCHMARK_RIC)

    paths = index_price_files(series_root)

    panel, anomalies, quality = build_panel(
        members,
        paths,
    )

    feature_path = (
        output_root
        / "point_in_time_equity_features.parquet"
    )

    panel.to_parquet(
        feature_path,
        index=False,
    )

    report_root = output_root / "reports"

    report_root.mkdir(exist_ok=True)

    by_date = panel.groupby(
        "Decision Date",
        as_index=False,
    ).agg(
        Members=("Underlying RIC", "size"),
        Prior_Prices=(
            "Last Prior Equity Price Date", "count"
        ),
        Recent_Prices=("Recent Prior Price", "sum"),
        Full_253=(
            "Complete 253 Session History", "sum"
        ),
        Feature_Candidates=(
            "Feature Candidate", "sum"
        ),
        Verified_Feature_Candidates=(
            "Verified Membership Feature Candidate", "sum"
        ),
        Provisional_Members=(
            "Provisional Membership", "sum"
        ),
        Same_Day_Event_Flags=(
            "Same-Day Event Timing Unverified", "sum"
        ),
    )

    by_date["Feature Candidate Share"] = (
        by_date["Feature_Candidates"]
        / by_date["Members"]
    )

    by_date.to_csv(
        report_root / "coverage_by_decision_date.csv",
        index=False,
    )

    anomalies.to_csv(
        report_root / "large_adjusted_daily_moves.csv",
        index=False,
    )

    quality.to_csv(
        report_root / "price_history_quality_by_ric.csv",
        index=False,
    )

    panel.loc[
        ~panel["Feature Candidate"],
        [
            "Decision Date",
            "Underlying RIC",
            "Prior Price Age Days",
            "Complete 253 Session History",
            "Recent Prior Price",
            "Provisional Membership",
            "Same-Day Event Timing Unverified",
        ],
    ].to_csv(
        report_root / "incomplete_feature_candidates.csv",
        index=False,
    )

    print("\n--- POINT-IN-TIME FEATURE SUMMARY ---")
    print("Rows:", len(panel))
    print(
        "Dates:",
        panel["Decision Date"].nunique(),
    )
    print(
        "Features with complete 253-session history:",
        int(panel["Complete 253 Session History"].sum()),
    )
    print(
        "Feature candidates:",
        int(panel["Feature Candidate"].sum()),
    )
    print(
        "Verified-membership feature candidates:",
        int(
            panel[
                "Verified Membership Feature Candidate"
            ].sum()
        ),
    )
    print(
        "Provisional membership rows:",
        int(panel["Provisional Membership"].sum()),
    )
    print(
        "Same-day event timing flags:",
        int(
            panel[
                "Same-Day Event Timing Unverified"
            ].sum()
        ),
    )
    print(
        "Invalid closes in history:",
        int(quality["Invalid Close Observations"].sum()),
    )
    print(
        "Large adjusted daily moves:",
        len(anomalies),
    )
    print("Feature panel:", feature_path)
    print("Reports:", report_root)
    print(
        "No missing prices were forward-filled. "
        "No future targets were calculated."
    )


if __name__ == "__main__":
    main()
