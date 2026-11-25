
"""Collect LSEG daily histories with audited close-field selection.

Uses ld.open_session() with the configured default Platform session.
Version 2 output does not overwrite market_history_v1.
"""

import argparse
import hashlib
import re
import time
from datetime import date, timedelta
from pathlib import Path

import lseg.data as ld
import pandas as pd


ROOT = (Path(__file__).resolve().parents[1] / "data/processed")

ETF_FILE = (
    ROOT
    / "etf_universe/ric_verification/primary_listing"
    / "etf_instruments_verified.parquet"
)

EQUITY_FILE = (
    ROOT
    / "ever_member_universe/instrument_mapping"
    / "instrument_mapping.parquet"
)

OUTPUT = ROOT / "market_history_v2"

FIELDS = [
    "OPEN_PRC",
    "HIGH_1",
    "LOW_1",
    "TRDPRC_1",
    "OFF_CLOSE",
    "ACVOL_UNS",
]

CLOSE_FIELDS = [
    "TRDPRC_1",
    "OFF_CLOSE",
]

ADJUSTMENTS = {
    "unadjusted": ["unadjusted"],
    "adjusted": [
        "exchangeCorrection",
        "manualCorrection",
        "CCH",
        "CRE",
        "RPO",
        "RTS",
    ],
}

COLUMNS = [
    "Date",
    "RIC",
    "Adjustment Mode",
    "Open",
    "High",
    "Low",
    "Close",
    "Close Field",
    "Volume",
]

WINDOW_YEARS = 3
OVERLAP_DAYS = 10
PAUSE_SECONDS = 0.3


def clean(value):
    """Perform the clean operation for this research workflow.

    Args:
        value: Input to the clean operation.

    Returns:
        Computed result or status for the caller.
    """
    if value is None or pd.isna(value):
        return ""

    return str(value).strip()


def unique_rics(items):
    """Perform the unique rics operation for this research workflow.

    Args:
        items: Input to the unique_rics operation.

    Returns:
        Computed result or status for the caller.
    """
    values = list(
        dict.fromkeys(clean(x) for x in items)
    )

    if not values or any(not x for x in values):
        raise ValueError("No valid RICs supplied.")

    return values


def select_rics(args):
    """Perform the select rics operation for this research workflow.

    Args:
        args: Input to the select_rics operation.

    Returns:
        Computed result or status for the caller.
    """
    if args.rics:
        return unique_rics(args.rics), "manual"

    if args.rics_file:
        path = Path(args.rics_file)

        if not path.is_file():
            raise FileNotFoundError(path)

        selected = unique_rics(
            path.read_text(
                encoding="utf-8"
            ).splitlines()
        )

        mapping = pd.read_parquet(EQUITY_FILE)

        mapped = set(
            mapping.loc[
                mapping["Mapping Status"].eq(
                    "IDENTIFIER_MAPPED"
                ),
                "Underlying RIC",
            ].astype(str)
        )

        unknown = sorted(
            set(selected) - mapped
        )

        if unknown:
            raise ValueError(
                f"Unmapped RICs in file: "
                f"{unknown[:10]}"
            )

        return selected, "equity_member_manifest"


    if args.all_etfs or args.etf_tickers:
        df = pd.read_parquet(ETF_FILE)

        required = {
            "Ticker",
            "LSEG RIC",
            "RIC Verified",
        }

        if not required.issubset(df.columns):
            raise ValueError(
                f"Missing ETF columns: "
                f"{required - set(df.columns)}"
            )

        if args.etf_tickers:
            selected = {
                x.upper()
                for x in args.etf_tickers
            }

            existing = set(
                df["Ticker"].astype(str).str.upper()
            )

            if selected - existing:
                raise ValueError(
                    f"Unknown ETF tickers: "
                    f"{sorted(selected - existing)}"
                )

            df = df.loc[
                df["Ticker"]
                .astype(str)
                .str.upper()
                .isin(selected)
            ]

        if not df["RIC Verified"].fillna(False).astype(bool).all():
            raise ValueError(
                "Selected ETFs include unverified RICs."
            )

        return unique_rics(df["LSEG RIC"]), "etf"

    df = pd.read_parquet(EQUITY_FILE)

    if args.equity_rics:
        selected = set(args.equity_rics)

        existing = set(
            df["Underlying RIC"].astype(str)
        )

        if selected - existing:
            raise ValueError(
                f"Unknown equity RICs: "
                f"{sorted(selected - existing)}"
            )

        df = df.loc[
            df["Underlying RIC"].isin(selected)
        ]

    if not df["Mapping Status"].eq(
        "IDENTIFIER_MAPPED"
    ).all():
        raise ValueError(
            "Selected equities are not all mapped."
        )

    return unique_rics(df["Underlying RIC"]), "equity"


def windows(start, end):
    """Split requests into consecutive three-year windows."""

    cursor = start

    while cursor <= end:
        following = (
            pd.Timestamp(cursor)
            + pd.DateOffset(years=WINDOW_YEARS)
        ).date()

        stop = min(
            following - timedelta(days=1),
            end,
        )

        yield cursor, stop

        cursor = stop + timedelta(days=1)


def flat_columns(df):
    """Perform the flat columns operation for this research workflow.

    Args:
        df: Input to the flat_columns operation.

    Returns:
        Computed result or status for the caller.
    """
    df = df.copy()

    if isinstance(df.columns, pd.MultiIndex):
        renamed = []

        for parts in df.columns:
            matching = [
                str(part)
                for part in parts
                if str(part) in FIELDS
            ]

            renamed.append(
                matching[0]
                if len(matching) == 1
                else "|".join(map(str, parts))
            )

        df.columns = renamed

    if df.columns.duplicated().any():
        raise ValueError(
            f"Duplicate LSEG columns: "
            f"{df.columns.tolist()}"
        )

    return df


def normalize_history(frame, ric, mode):
    """Perform the normalize history operation for this research workflow.

    Args:
        frame: Input to the normalize_history operation.
        ric: Input to the normalize_history operation.
        mode: Input to the normalize_history operation.

    Returns:
        Computed result or status for the caller.
    """
    if frame is None or frame.empty:
        return pd.DataFrame(columns=COLUMNS)

    df = flat_columns(frame)

    if isinstance(df.index, pd.MultiIndex):
        date_values = None

        for level in range(df.index.nlevels):
            values = df.index.get_level_values(level)

            if pd.api.types.is_datetime64_any_dtype(values):
                date_values = pd.DatetimeIndex(values)
                break

        if date_values is None:
            raise ValueError(
                f"No datetime index for {ric}."
            )

    else:
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError(
                f"Unexpected historical index for {ric}: "
                f"{type(df.index)}"
            )

        date_values = df.index

    if date_values.tz is not None:
        date_values = date_values.tz_localize(None)

    dates = date_values.normalize()

    if dates.isna().any():
        raise ValueError(
            f"Invalid historical dates for {ric}."
        )

    numeric = {}

    for field in FIELDS:
        if field in df.columns:
            numeric[field] = pd.to_numeric(
                df[field],
                errors="coerce",
            )
        else:
            numeric[field] = pd.Series(
                float("nan"),
                index=df.index,
            )

    # Prefer the field with more valid closing prices.
    # If counts are equal, TRDPRC_1 has priority.
    available = {
        field: int(numeric[field].gt(0).sum())
        for field in CLOSE_FIELDS
    }

    close_field = max(
        CLOSE_FIELDS,
        key=lambda field: available[field],
    )

    if available[close_field] == 0:
        raise ValueError(
            f"No positive close in TRDPRC_1 or OFF_CLOSE "
            f"for {ric}; columns: {df.columns.tolist()}"
        )

    result = pd.DataFrame({
        "Date": dates,
        "RIC": ric,
        "Adjustment Mode": mode,
        "Open": numeric["OPEN_PRC"].to_numpy(),
        "High": numeric["HIGH_1"].to_numpy(),
        "Low": numeric["LOW_1"].to_numpy(),
        "Close": numeric[close_field].to_numpy(),
        "Close Field": close_field,
        "Volume": numeric["ACVOL_UNS"].to_numpy(),
    })

    if result["Date"].duplicated().any():
        raise ValueError(
            f"Duplicate daily dates for {ric}."
        )

    return (
        result
        .sort_values("Date")
        .reset_index(drop=True)
    )


def data_path(ric, mode, start, end, folder):
    """Perform the data path operation for this research workflow.

    Args:
        ric: Input to the data_path operation.
        mode: Input to the data_path operation.
        start: Input to the data_path operation.
        end: Input to the data_path operation.
        folder: Input to the data_path operation.

    Returns:
        Computed result or status for the caller.
    """
    safe = re.sub(
        r"[^A-Za-z0-9_.-]",
        "_",
        ric,
    )

    digest = hashlib.sha256(
        ric.encode("utf-8")
    ).hexdigest()[:8]

    return (
        OUTPUT
        / folder
        / f"{safe}_{digest}"
        / mode
        / f"{start}_{end}.parquet"
    )


def validate_cache(df, ric, mode, start, end):
    """Validate cache against the expected constraints.

    Args:
        df: Input to the validate_cache operation.
        ric: Input to the validate_cache operation.
        mode: Input to the validate_cache operation.
        start: Input to the validate_cache operation.
        end: Input to the validate_cache operation.

    Returns:
        Computed result or status for the caller.
    """
    if not set(COLUMNS).issubset(df.columns):
        return False

    if df.empty:
        return False

    dates = pd.to_datetime(
        df["Date"],
        errors="coerce",
    )

    return bool(
        df["RIC"].eq(ric).all()
        and df["Adjustment Mode"].eq(mode).all()
        and df["Close Field"].isin(CLOSE_FIELDS).all()
        and dates.notna().all()
        and dates.between(
            pd.Timestamp(start),
            pd.Timestamp(end),
        ).all()
        and not dates.duplicated().any()
        and pd.to_numeric(
            df["Close"],
            errors="coerce",
        ).gt(0).any()
    )


def get_window(
    ric,
    mode,
    start,
    end,
    refresh=False,
):
    """Get window from the current context.

    Args:
        ric: Input to the get_window operation.
        mode: Input to the get_window operation.
        start: Input to the get_window operation.
        end: Input to the get_window operation.
        refresh: Input to the get_window operation.

    Returns:
        Computed result or status for the caller.
    """
    cache = data_path(
        ric, mode, start, end, "cache"
    )

    if cache.exists() and not refresh:
        try:
            saved = pd.read_parquet(cache)

            if validate_cache(
                saved, ric, mode, start, end
            ):
                print("    Cached")
                return saved

        except Exception as error:
            print(f"    Invalid cache: {error}")

    # Retrieve additional preceding dates, then trim.
    request_start = (
        start - timedelta(days=OVERLAP_DAYS)
    )

    last_error = None

    for attempt in range(1, 3):
        try:
            time.sleep(PAUSE_SECONDS)

            kwargs = {
                "universe": ric,
                "interval": "daily",
                "start": str(request_start),
                "end": str(end),
                "count": 3000,
                "adjustments": ADJUSTMENTS[mode],
                "header_type": ld.HeaderType.NAME,
            }

            response = ld.get_history(
                fields=FIELDS,
                **kwargs,
            )

            try:
                series = normalize_history(
                    response,
                    ric,
                    mode,
                )

            except ValueError as error:
                if "No positive close" not in str(error):
                    raise

                print(
                    "    Close unavailable in selected fields; "
                    "trying full field set"
                )

                response = ld.get_history(
                    **kwargs
                )

                series = normalize_history(
                    response,
                    ric,
                    mode,
                )

            series = series.loc[
                series["Date"].between(
                    pd.Timestamp(start),
                    pd.Timestamp(end),
                )
            ].reset_index(drop=True)

            if series.empty:
                return series

            cache.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            temporary = cache.with_name(
                f"{cache.stem}.tmp.parquet"
            )

            series.to_parquet(
                temporary,
                index=False,
            )

            temporary.replace(cache)

            return series

        except Exception as error:
            last_error = error

            print(
                f"    Attempt {attempt}/2 failed: "
                f"{type(error).__name__}: {error}"
            )

            if attempt == 1:
                time.sleep(1)

    raise RuntimeError(
        f"{ric} {mode} {start} to {end}: "
        f"{last_error}"
    )


def collect_one(
    ric,
    mode,
    start,
    end,
    refresh=False,
):
    """Retrieve one from configured sources.

    Args:
        ric: Input to the collect_one operation.
        mode: Input to the collect_one operation.
        start: Input to the collect_one operation.
        end: Input to the collect_one operation.
        refresh: Input to the collect_one operation.

    Returns:
        Computed result or status for the caller.
    """
    parts = []
    errors = []

    for left, right in windows(start, end):
        print(
            f"  {mode:10s} {left} to {right}"
        )

        try:
            frame = get_window(
                ric,
                mode,
                left,
                right,
                refresh,
            )

            if not frame.empty:
                parts.append(frame)

        except Exception as error:
            errors.append(
                f"{left} to {right}: {error}"
            )

    if parts:
        combined = pd.concat(
            parts,
            ignore_index=True,
        )

        combined = (
            combined
            .sort_values("Date")
            .reset_index(drop=True)
        )

        if combined["Date"].duplicated().any():
            raise ValueError(
                f"Duplicate dates after merging {ric}."
            )

        path = data_path(
            ric,
            mode,
            start,
            end,
            "series",
        )

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        combined.to_parquet(
            path,
            index=False,
        )

    else:
        combined = pd.DataFrame(
            columns=COLUMNS
        )

    close = pd.to_numeric(
        combined["Close"],
        errors="coerce",
    )

    rows = len(combined)
    valid = int(close.gt(0).sum())

    if errors:
        status = "PARTIAL" if rows else "FAILED"

    elif not rows:
        status = "NO_DATA"

    elif valid < rows:
        status = "REVIEW_MISSING_CLOSE"

    else:
        status = "DOWNLOADED"

    used = sorted(
        combined["Close Field"]
        .dropna()
        .astype(str)
        .unique()
    )

    if len(used) > 1 and status == "DOWNLOADED":
        status = "REVIEW_MIXED_CLOSE_FIELDS"

    summary = {
        "RIC": ric,
        "Mode": mode,
        "Rows": rows,
        "Positive close": valid,
        "Close fields used": ", ".join(used),
        "First date": (
            str(combined["Date"].min().date())
            if rows else ""
        ),
        "Last date": (
            str(combined["Date"].max().date())
            if rows else ""
        ),
        "Failed windows": len(errors),
        "Status": status,
        "Errors": " | ".join(errors),
    }

    return (
        summary,
        set(
            pd.to_datetime(
                combined["Date"]
            ).dt.date
        ),
    )


def main():
    """Run the module entry point and its workflow.
    """
    parser = argparse.ArgumentParser()

    group = parser.add_mutually_exclusive_group(
        required=True
    )

    group.add_argument(
        "--rics",
        nargs="+",
        help="Exact LSEG RICs",
    )

    group.add_argument(
        "--etf-tickers",
        nargs="+",
        help="Verified ETF tickers",
    )

    group.add_argument(
        "--all-etfs",
        action="store_true",
    )

    group.add_argument(
        "--equity-rics",
        nargs="+",
        help="Mapped equity RICs",
    )

    group.add_argument(
        "--all-equities",
        action="store_true",
    )

    group.add_argument(
        "--rics-file",
        type=Path,
        help="Text file containing one RIC per line",
    )

    parser.add_argument(
        "--start",
        required=True,
        type=date.fromisoformat,
    )

    parser.add_argument(
        "--end",
        required=True,
        type=date.fromisoformat,
    )

    parser.add_argument(
        "--mode",
        choices=["both", "unadjusted", "adjusted"],
        default="both",
    )

    parser.add_argument(
        "--refresh",
        action="store_true",
    )

    args = parser.parse_args()

    if args.end < args.start:
        parser.error(
            "--end must not be earlier than --start."
        )

    rics, source = select_rics(args)

    modes = (
        list(ADJUSTMENTS)
        if args.mode == "both"
        else [args.mode]
    )

    print(
        "\n--- LSEG DAILY HISTORY COLLECTION (V2) ---"
    )

    print("RICs:", len(rics))
    print("Source:", source)
    print("Date range:", args.start, "to", args.end)
    print("Modes:", modes)
    print("Output:", OUTPUT)

    summaries = []
    audits = []

    ld.open_session()

    try:
        for i, ric in enumerate(rics, start=1):
            print(f"\n[{i}/{len(rics)}] {ric}")

            sets = {}

            for mode in modes:
                summary, dates = collect_one(
                    ric,
                    mode,
                    args.start,
                    args.end,
                    args.refresh,
                )

                summaries.append(summary)
                sets[mode] = dates

            if len(modes) == 2:
                only_unadjusted = sorted(
                    sets["unadjusted"]
                    - sets["adjusted"]
                )

                only_adjusted = sorted(
                    sets["adjusted"]
                    - sets["unadjusted"]
                )

                audits.append({
                    "RIC": ric,
                    "Dates only unadjusted": len(
                        only_unadjusted
                    ),
                    "Dates only adjusted": len(
                        only_adjusted
                    ),
                    "Example only unadjusted": ", ".join(
                        map(str, only_unadjusted[:10])
                    ),
                    "Example only adjusted": ", ".join(
                        map(str, only_adjusted[:10])
                    ),
                    "Date sets identical": not (
                        only_unadjusted
                        or only_adjusted
                    ),
                })

    finally:
        ld.close_session()

    signature = hashlib.sha256(
        "\n".join(
            sorted(rics)
        ).encode("utf-8")
    ).hexdigest()[:8]

    stem = (
        f"{source}_{args.start}_{args.end}_"
        f"{args.mode}_{signature}"
    )

    report_dir = OUTPUT / "reports"

    report_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    report = pd.DataFrame(summaries)

    report.to_csv(
        report_dir / f"{stem}_coverage.csv",
        index=False,
    )

    print("\n--- HISTORICAL COVERAGE SUMMARY ---")

    print(
        report[
            [
                "RIC",
                "Mode",
                "Rows",
                "Positive close",
                "Close fields used",
                "First date",
                "Last date",
                "Failed windows",
                "Status",
            ]
        ].to_string(index=False)
    )

    if audits:
        audit = pd.DataFrame(audits)

        audit.to_csv(
            report_dir / f"{stem}_mode_dates.csv",
            index=False,
        )

        print(
            "\n--- ADJUSTED VS UNADJUSTED DATE AUDIT ---"
        )

        print(
            audit.to_string(index=False)
        )

        print(
            "Date audit:",
            report_dir / f"{stem}_mode_dates.csv",
        )

    print(
        "Coverage report:",
        report_dir / f"{stem}_coverage.csv",
    )

    print(
        "No missing prices were forward-filled. "
        "No total-return or membership mask was applied."
    )


if __name__ == "__main__":
    main()
