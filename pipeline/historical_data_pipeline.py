
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import lseg.data as ld


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "raw" / "market_prices"

MARKET_UNIVERSE = {
    "AAPL.O": "equity",
    "NVDA.O": "equity",
    "SPY": "etf",
    "VOO": "etf"
}


def download_price_history(
    ric,
    start_date="2021-01-01",
    end_date="2026-11-03"
):
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()

    if start > end:
        raise ValueError("start_date must precede end_date.")

    batches = []
    year = start.year

    while year <= end.year:
        batch_start = max(
            start,
            pd.Timestamp(year=year, month=1, day=1)
        )
        batch_end = min(
            end,
            pd.Timestamp(year=year, month=12, day=31)
        )

        history = ld.get_history(
            universe=ric,
            fields=["TRDPRC_1"],
            interval="daily",
            start=batch_start.strftime("%Y-%m-%d"),
            end=batch_end.strftime("%Y-%m-%d")
        )

        if history is None or history.empty:
            print(f"{ric}: no observations in {year}")
            year += 1
            continue

        if isinstance(history.columns, pd.MultiIndex):
            raise ValueError(
                f"Unexpected column structure for {ric}: "
                f"{history.columns.tolist()}"
            )

        if "TRDPRC_1" not in history.columns:
            raise ValueError(
                f"{ric}: TRDPRC_1 missing. "
                f"Available: {history.columns.tolist()}"
            )

        batch = history[["TRDPRC_1"]].copy()
        batch = batch.rename(
            columns={"TRDPRC_1": "Close"}
        )

        batches.append(batch)
        print(f"{ric}: downloaded {len(batch)} rows for {year}")

        year += 1

    if not batches:
        raise ValueError(f"No historical prices for {ric}.")

    result = pd.concat(batches)

    result.index = pd.to_datetime(
        result.index
    ).normalize()

    result = result.sort_index()
    result = result[
        ~result.index.duplicated(keep="last")
    ]

    result["Close"] = pd.to_numeric(
        result["Close"],
        errors="coerce"
    ).astype("float64")

    result = result.dropna(subset=["Close"])

    if (result["Close"] <= 0).any():
        raise ValueError(f"{ric}: invalid closing prices.")

    result.index.name = "Date"
    result["RIC"] = ric

    result["DownloadedUTC"] = datetime.now(
        timezone.utc
    ).isoformat()

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    filename = ric.replace(".", "_") + ".parquet"
    filepath = DATA_DIR / filename

    # Preserve previously downloaded observations.
    if filepath.is_file():
        existing = pd.read_parquet(filepath)

        existing.index = pd.to_datetime(
            existing.index
        ).normalize()

        result = pd.concat(
            [existing, result],
            axis=0
        )

        result = result.sort_index(
            kind="stable"
        )

        # Newly downloaded observations take
        # priority when dates overlap.
        result = result.loc[
            ~result.index.duplicated(keep="last")
        ].copy()

        result["Close"] = pd.to_numeric(
            result["Close"],
            errors="coerce"
        )

        result = result.dropna(
            subset=["Close"]
        )

        if (result["Close"] <= 0).any():
            raise ValueError(
                f"{ric}: invalid cached prices."
            )

        result.index.name = "Date"

    result.to_parquet(
        filepath,
        index=True
    )


    print(
        f"Saved {ric}: {len(result)} rows "
        f"to {filepath}"
    )

    return result


def load_price_history(ric):
    filename = ric.replace(".", "_") + ".parquet"
    filepath = DATA_DIR / filename

    if not filepath.exists():
        raise FileNotFoundError(
            f"Local dataset not found: {filepath}"
        )

    result = pd.read_parquet(filepath)
    result.index = pd.to_datetime(result.index)

    return result


def download_market_universe(
    start_date="2021-01-01",
    end_date="2026-11-03"
):
    results = {}
    failures = {}

    for ric in MARKET_UNIVERSE:
        try:
            results[ric] = download_price_history(
                ric=ric,
                start_date=start_date,
                end_date=end_date
            )
        except Exception as exc:
            failures[ric] = str(exc)
            print(f"Download failed for {ric}: {exc}")

    print("\n--- LOCAL DATA DOWNLOAD SUMMARY ---")

    for ric, dataframe in results.items():
        print(
            f"{ric}: {len(dataframe)} observations, "
            f"{dataframe.index.min().date()} to "
            f"{dataframe.index.max().date()}"
        )

    if failures:
        print("\nFailed instruments:")
        for ric, error in failures.items():
            print(f"{ric}: {error}")

    return results


def ensure_price_history(
    ric,
    start_date,
    end_date,
):
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()

    if start > end:
        raise ValueError(
            "Start date must not exceed end date."
        )

    filename = (
        ric.replace(".", "_") + ".parquet"
    )

    filepath = DATA_DIR / filename

    if not filepath.is_file():
        print(
            f"\nNo local stock history for {ric}."
        )

        return download_price_history(
            ric=ric,
            start_date=start,
            end_date=end,
        )

    existing = load_price_history(ric)

    if existing.empty:
        return download_price_history(
            ric=ric,
            start_date=start,
            end_date=end,
        )

    first = pd.Timestamp(
        existing.index.min()
    ).normalize()

    last = pd.Timestamp(
        existing.index.max()
    ).normalize()

    # Allow calendar-day differences around
    # weekends and exchange holidays.
    tolerance = pd.Timedelta(days=7)

    if first > start + tolerance:
        print(
            f"\nExtending {ric} history backwards."
        )

        download_price_history(
            ric=ric,
            start_date=start,
            end_date=first - pd.Timedelta(days=1),
        )

    if last < end - tolerance:
        print(
            f"\nExtending {ric} history forwards."
        )

        download_price_history(
            ric=ric,
            start_date=last + pd.Timedelta(days=1),
            end_date=end,
        )

    result = load_price_history(ric)

    print("\n--- LOCAL STOCK HISTORY ---")
    print(f"Underlying: {ric}")
    print(f"Observations: {len(result)}")
    print(
        f"Available: "
        f"{result.index.min().date()} to "
        f"{result.index.max().date()}"
    )

    return result
