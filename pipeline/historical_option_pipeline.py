
from pathlib import Path

import numpy as np
import pandas as pd
import lseg.data as ld


PROJECT_ROOT = Path(__file__).resolve().parent.parent

OPTIONS_DIR = (
    PROJECT_ROOT / "data" / "raw" / "options"
)


def download_option_quote_history(
    option_ric,
    start_date,
    end_date
):
    history = ld.get_history(
        universe=option_ric,
        fields=["BID", "ASK"],
        interval="daily",
        start=start_date,
        end=end_date
    )

    if history is None or history.empty:
        raise ValueError(
            f"No historical BID/ASK available for {option_ric}"
        )

    history = history.copy()

    # Handle possible instrument-labelled columns.
    if isinstance(history.columns, pd.MultiIndex):
        found = False

        for level in range(history.columns.nlevels):
            if option_ric in history.columns.get_level_values(level):
                history = history.xs(
                    option_ric,
                    axis=1,
                    level=level
                )
                found = True
                break

        if not found or isinstance(history.columns, pd.MultiIndex):
            raise ValueError(
                f"Unexpected columns for {option_ric}: "
                f"{history.columns.tolist()}"
            )

    if not {"BID", "ASK"}.issubset(history.columns):
        raise ValueError(
            f"BID/ASK missing for {option_ric}: "
            f"{history.columns.tolist()}"
        )

    history = history[["BID", "ASK"]].copy()

    history.index = pd.to_datetime(
        history.index
    ).normalize()

    history.index.name = "Date"

    for column in ["BID", "ASK"]:
        history[column] = pd.to_numeric(
            history[column],
            errors="coerce"
        ).astype("float64")

    history = history.replace(
        [np.inf, -np.inf],
        np.nan
    )

    
    # Preserve valid one-sided quotations.
    # A short option can be liquidated at ASK even if BID
    # is unavailable. A long option requires BID to close.

    bid = history["BID"]
    ask = history["ASK"]

    # Non-missing prices must be economically valid.
    valid_bid = bid.isna() | (bid >= 0)
    valid_ask = ask.isna() | (ask > 0)

    # At least one executable quote must exist.
    has_quote = bid.notna() | ask.notna()

    # If both prices exist, reject crossed markets.
    valid_market = (
        bid.isna()
        | ask.isna()
        | (ask >= bid)
    )

    history = history.loc[
        valid_bid
        & valid_ask
        & has_quote
        & valid_market
    ].copy()


    history = history.sort_index()
    history = history[
        ~history.index.duplicated(keep="last")
    ]

    if history.empty:
        raise ValueError(
            f"No valid historical BID/ASK for {option_ric}"
        )

    history["MID"] = (
        history["BID"] + history["ASK"]
    ) / 2

    history["SPREAD"] = (
        history["ASK"] - history["BID"]
    )

    history["SPREAD_PCT"] = (
        history["SPREAD"] / history["MID"]
    )

    history["RIC"] = option_ric

    OPTIONS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    safe_ric = (
        option_ric
        .replace(".", "_")
        .replace("/", "_")
    )

    filepath = OPTIONS_DIR / f"{safe_ric}.parquet"

    history.to_parquet(filepath)

    print(
        f"{option_ric}: "
        f"{len(history)} historical observations saved"
    )

    return history


def load_option_quote_history(option_ric):
    safe_ric = (
        option_ric
        .replace(".", "_")
        .replace("/", "_")
    )

    filepath = OPTIONS_DIR / f"{safe_ric}.parquet"

    if not filepath.exists():
        raise FileNotFoundError(filepath)

    return pd.read_parquet(filepath)
