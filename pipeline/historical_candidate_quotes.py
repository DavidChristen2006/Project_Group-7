
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import lseg.data as ld

from pipeline.historical_data_pipeline import (
    load_price_history,
)
from pipeline.historical_ml_pipeline import (
    ensure_local_option_history,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent

LEG_COLUMNS = [
    "Long Put RIC",
    "Short Call RIC",
    "Short Put RIC",
]

REQUIRED_COLUMNS = [
    "Underlying RIC",
    "Decision Date",
    "Outcome Date",
    "Hedge Horizon Days",
    "Candidate ID",
    "Strategy",
    *LEG_COLUMNS,
]


def prepare_quote_requirements(
    inventory,
    max_candidates=None,
):
    missing = set(REQUIRED_COLUMNS) - set(
        inventory.columns
    )

    if missing:
        raise ValueError(
            f"Missing inventory columns: {sorted(missing)}"
        )

    if inventory.empty:
        raise ValueError("Candidate inventory is empty.")

    candidates = inventory.copy()

    for column in ["Decision Date", "Outcome Date"]:
        candidates[column] = pd.to_datetime(
            candidates[column]
        ).dt.normalize()

    key_columns = [
        "Underlying RIC",
        "Decision Date",
        "Hedge Horizon Days",
        "Candidate ID",
    ]

    if candidates.duplicated(key_columns).any():
        raise ValueError("Duplicate candidate keys.")

    candidates = candidates.sort_values(
        key_columns,
        kind="mergesort",
    ).reset_index(drop=True)

    if max_candidates is not None:
        if max_candidates <= 0:
            raise ValueError(
                "max_candidates must be positive."
            )

        candidates = candidates.head(
            max_candidates
        ).copy()

    stock_cache = {}
    requirements = {}

    for _, candidate in candidates.iterrows():
        underlying = candidate["Underlying RIC"]

        if underlying not in stock_cache:
            stock = load_price_history(
                underlying
            ).sort_index()

            stock.index = pd.to_datetime(
                stock.index
            ).normalize()

            if stock.index.has_duplicates:
                raise ValueError(
                    f"Duplicate stock dates: {underlying}"
                )

            stock_cache[underlying] = stock

        stock = stock_cache[underlying]

        start = candidate["Decision Date"]
        end = candidate["Outcome Date"]
        horizon = int(candidate["Hedge Horizon Days"])

        if (
            pd.isna(start)
            or pd.isna(end)
            or horizon <= 0
            or start >= end
            or float(candidate["Hedge Horizon Days"]) != horizon
        ):
            raise ValueError(
                "Invalid candidate dates or horizon."
            )

        required_dates = stock.loc[
            start:end
        ].index

        if (
            len(required_dates) != horizon + 1
            or required_dates[0] != start
            or required_dates[-1] != end
        ):
            raise ValueError(
                "Candidate has an incomplete "
                "stock trading-date window."
            )

        for column in LEG_COLUMNS:
            option_ric = candidate[column]

            if pd.isna(option_ric):
                continue

            option_ric = str(option_ric).strip()

            if not option_ric:
                continue

            if option_ric not in requirements:
                requirements[option_ric] = set()

            requirements[option_ric].update(
                required_dates
            )

    if not requirements:
        raise ValueError(
            "No option contracts found in inventory."
        )

    return {
        ric: pd.DatetimeIndex(
            sorted(dates)
        )
        for ric, dates in sorted(requirements.items())
    }


def check_quote_coverage(
    quotes,
    required_dates,
):
    required_dates = pd.DatetimeIndex(
        required_dates
    ).normalize()

    if not {"BID", "ASK"}.issubset(quotes.columns):
        raise ValueError(
            "Historical quotes require BID and ASK."
        )

    data = quotes.copy()

    data.index = pd.to_datetime(
        data.index
    ).normalize()

    if data.index.has_duplicates:
        raise ValueError(
            "Duplicate historical quote dates."
        )

    data = data.reindex(required_dates)

    bid = pd.to_numeric(
        data["BID"],
        errors="coerce",
    )

    ask = pd.to_numeric(
        data["ASK"],
        errors="coerce",
    )

    valid = (
        np.isfinite(bid)
        & np.isfinite(ask)
        & (bid >= 0)
        & (ask > 0)
        & (ask >= bid)
    )

    return int((~valid).sum())


def prepare_candidate_quote_histories(
    inventory,
    max_candidates=None,
    refresh_missing_quotes=False,
    downloader=None,
):
    if downloader is None:
        downloader = ensure_local_option_history

    requirements = prepare_quote_requirements(
        inventory=inventory,
        max_candidates=max_candidates,
    )

    records = []

    for option_ric, required_dates in requirements.items():
        print(
            f"\nChecking {option_ric}: "
            f"{len(required_dates)} required dates"
        )

        try:
            quotes = downloader(
                option_ric=option_ric,
                start_date=required_dates.min(),
                end_date=required_dates.max(),
                required_dates=required_dates,
                refresh=refresh_missing_quotes,
            )

            missing_count = check_quote_coverage(
                quotes,
                required_dates,
            )

            status = (
                "Complete"
                if missing_count == 0
                else "Incomplete"
            )

            error = ""

        except Exception as exc:
            missing_count = len(required_dates)
            status = "Download Failed"
            error = str(exc)

        records.append({
            "Option RIC": option_ric,
            "Start Date": required_dates.min(),
            "End Date": required_dates.max(),
            "Required Dates": len(required_dates),
            "Missing Dates": missing_count,
            "Status": status,
            "Error": error,
        })

        print(
            f"Status: {status} | "
            f"Missing: {missing_count}"
        )

    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare historical candidate quote histories."
    )

    parser.add_argument(
        "--inventory",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--max-candidates",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--refresh-missing-quotes",
        action="store_true",
    )

    parser.add_argument(
        "--download",
        action="store_true",
        help="Enable LSEG retrieval; otherwise show requirements.",
    )

    args = parser.parse_args()

    inventory_path = args.inventory

    if not inventory_path.is_absolute():
        inventory_path = (
            PROJECT_ROOT / inventory_path
        )

    if not inventory_path.is_file():
        parser.error(
            f"Inventory not found: {inventory_path}"
        )

    inventory = pd.read_parquet(
        inventory_path
    )

    if not args.download:
        requirements = prepare_quote_requirements(
            inventory,
            max_candidates=args.max_candidates,
        )

        print("\n--- QUOTE REQUIREMENTS ---")
        print(f"Unique contracts: {len(requirements)}")

        for ric, dates in requirements.items():
            print(
                f"{ric}: {len(dates)} trading dates "
                f"({dates.min().date()} to "
                f"{dates.max().date()})"
            )

        print(
            "\nNo LSEG downloads performed. "
            "Use --download to retrieve quote histories."
        )

        return


    # Open the configured default LSEG session.
    # Your configuration uses the Platform session.
    print("\nOpening LSEG Platform session...")

    try:
        ld.open_session()

        print("LSEG session opened successfully.")

        coverage = prepare_candidate_quote_histories(
            inventory=inventory,
            max_candidates=args.max_candidates,
            refresh_missing_quotes=args.refresh_missing_quotes,
        )

    finally:
        ld.close_session()
        print("\nLSEG session closed.")


    output_dir = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "candidate_quote_coverage"
        / inventory_path.parent.name
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir / "quote_coverage.parquet"
    )

    coverage.to_parquet(
        output_path,
        index=False,
    )

    print("\n--- QUOTE COVERAGE SUMMARY ---")

    print(
        coverage["Status"]
        .value_counts()
        .to_string()
    )

    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
