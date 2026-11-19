
import hashlib
import json
from pathlib import Path

import pandas as pd

from pipeline.expired_option_pipeline import (
    discover_expired_options
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent

CACHE_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "option_discovery_cache"
)


def discover_expired_options_cached(
    underlying_root,
    decision_date,
    expiry,
    strikes,
    option_types=("put", "call"),
    max_spread_pct=0.10,
    refresh=False
):
    decision_date = pd.Timestamp(
        decision_date
    ).normalize()

    expiry = pd.Timestamp(
        expiry
    ).normalize()

    # Normalize inputs to create a reproducible cache key.
    strikes = sorted(set(float(s) for s in strikes))
    option_types = tuple(
        sorted(set(str(t).lower() for t in option_types))
    )

    config = {
        "underlying_root": underlying_root.upper(),
        "decision_date": decision_date.strftime("%Y-%m-%d"),
        "expiry": expiry.strftime("%Y-%m-%d"),
        "strikes": strikes,
        "option_types": option_types,
        "max_spread_pct": float(max_spread_pct)
    }

    config_string = json.dumps(
        config,
        sort_keys=True
    )

    cache_key = hashlib.sha256(
        config_string.encode("utf-8")
    ).hexdigest()[:16]

    CACHE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    cache_file = (
        CACHE_DIR
        / f"{underlying_root.upper()}_{cache_key}.parquet"
    )

    if cache_file.exists() and not refresh:
        print(
            f"Using cached option discovery: "
            f"{decision_date.date()} / {expiry.date()}"
        )

        return pd.read_parquet(cache_file)

    print(
        f"Discovering options: "
        f"{decision_date.date()} / {expiry.date()}"
    )

    result = discover_expired_options(
        underlying_root=underlying_root,
        decision_date=decision_date,
        expiry=expiry,
        strikes=strikes,
        option_types=option_types,
        max_spread_pct=max_spread_pct
    )

    # Cache successful discoveries only.
    # Empty results can be retried in a later run.
    if not result.empty:
        result.to_parquet(
            cache_file,
            index=False
        )

        print(f"Discovery cached: {len(result)} contracts")

    return result
