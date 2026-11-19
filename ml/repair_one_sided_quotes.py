
from datetime import datetime
from pathlib import Path
import shutil

import lseg.data as ld
import numpy as np
import pandas as pd

from pipeline.historical_option_pipeline import OPTIONS_DIR


RICS = [
    "AAPLF191515000.U^F15",
    "AAPLS171508500.U^G15",
]

TARGET_DATE = pd.Timestamp("2015-06-02")

BACKUP_DIR = Path(
    "data/backups/option_quote_repairs"
)


def get_option_filepath(ric):
    safe_ric = (
        ric.replace(".", "_")
        .replace("/", "_")
    )
    return OPTIONS_DIR / f"{safe_ric}.parquet"


def fetch_quotes(ric):
    history = ld.get_history(
        universe=ric,
        fields=["BID", "ASK"],
        interval="daily",
        start="2015-05-28",
        end="2015-06-05",
    )

    if history is None or history.empty:
        raise ValueError(f"No LSEG history for {ric}")

    history = history.copy()

    if isinstance(history.columns, pd.MultiIndex):
        for level in range(history.columns.nlevels):
            if ric in history.columns.get_level_values(level):
                history = history.xs(
                    ric,
                    axis=1,
                    level=level,
                )
                break

    if not {"BID", "ASK"}.issubset(history.columns):
        raise ValueError(
            f"Unexpected LSEG columns for {ric}"
        )

    history = history[["BID", "ASK"]].copy()

    history.index = pd.to_datetime(
        history.index
    ).normalize()

    history.index.name = "Date"

    if history.index.has_duplicates:
        raise ValueError(
            f"Duplicate downloaded dates for {ric}"
        )

    for column in ["BID", "ASK"]:
        history[column] = pd.to_numeric(
            history[column],
            errors="coerce",
        ).astype(float)

    history = history.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    bid = history["BID"]
    ask = history["ASK"]

    valid = (
        (bid.isna() | (bid >= 0))
        & (ask.isna() | (ask > 0))
        & (bid.notna() | ask.notna())
        & (bid.isna() | ask.isna() | (ask >= bid))
    )

    return history.loc[valid].sort_index()


def repair_contract(ric, downloaded):
    filepath = get_option_filepath(ric)

    if not filepath.exists():
        raise FileNotFoundError(filepath)

    existing = pd.read_parquet(filepath)

    existing.index = pd.to_datetime(
        existing.index
    ).normalize()

    existing.index.name = "Date"

    if existing.index.has_duplicates:
        raise ValueError(
            f"Duplicate cached dates for {ric}"
        )

    if TARGET_DATE in existing.index:
        print(f"{ric}: target date already cached.")
        return

    if TARGET_DATE not in downloaded.index:
        raise ValueError(
            f"{ric}: target date unavailable from LSEG"
        )

    target = downloaded.loc[[TARGET_DATE]].copy()

    if pd.isna(target["ASK"].iloc[0]):
        raise ValueError(
            f"{ric}: required exit ASK missing"
        )

    # Preserve the original cache schema.
    for column in existing.columns:
        if column not in target.columns:
            target[column] = np.nan

    if "RIC" in target.columns:
        target["RIC"] = ric

    if "MID" in target.columns:
        target["MID"] = (
            target["BID"] + target["ASK"]
        ) / 2

    if "SPREAD" in target.columns:
        target["SPREAD"] = (
            target["ASK"] - target["BID"]
        )

    if "SPREAD_PCT" in target.columns:
        target["SPREAD_PCT"] = (
            target["SPREAD"] / target["MID"]
        )

    target = target[existing.columns]

    combined = pd.concat(
        [existing, target],
    ).sort_index()

    if combined.index.has_duplicates:
        raise ValueError(
            f"Duplicate merged dates for {ric}"
        )

    # Back up before modifying any file.
    BACKUP_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    )

    backup = BACKUP_DIR / (
        f"{filepath.stem}_{timestamp}.parquet"
    )

    shutil.copy2(filepath, backup)

    # Write to a temporary file first.
    temporary = filepath.with_name(
        filepath.stem + ".repair_tmp.parquet"
    )

    try:
        combined.to_parquet(temporary)

        verification = pd.read_parquet(temporary)

        assert len(verification) == len(existing) + 1
        assert TARGET_DATE in verification.index

        temporary.replace(filepath)

    finally:
        if temporary.exists():
            temporary.unlink()

    print(f"\nRepaired: {ric}")
    print("Backup:", backup)
    print("Rows before:", len(existing))
    print("Rows after:", len(combined))
    print(
        "June 2 quote:",
        combined.loc[TARGET_DATE, ["BID", "ASK"]].to_dict(),
    )


def main():
    ld.open_session()

    try:
        # Fetch and validate both before changing either cache.
        downloaded = {
            ric: fetch_quotes(ric)
            for ric in RICS
        }

        for ric in RICS:
            repair_contract(
                ric,
                downloaded[ric],
            )

    finally:
        ld.close_session()

    print("\nQuote repair completed.")


if __name__ == "__main__":
    main()
