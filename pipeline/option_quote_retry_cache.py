
import hashlib
import json

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

CACHE_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "option_quote_retry_cache"
)


def get_missing_quote_dates(quote_df, required_dates):
    required = pd.DatetimeIndex(
        pd.to_datetime(required_dates)
    ).normalize().unique().sort_values()

    quotes = quote_df.copy()

    quotes.index = pd.DatetimeIndex(
        pd.to_datetime(quotes.index)
    ).normalize()

    if quotes.index.has_duplicates:
        raise ValueError("Duplicate option quote dates.")

    prices = quotes.reindex(required)[
        ["BID", "ASK"]
    ].apply(
        pd.to_numeric,
        errors="coerce"
    )

    valid = (
        prices["BID"].notna()
        & prices["ASK"].notna()
        & (prices["BID"] >= 0)
        & (prices["ASK"] > 0)
        & (prices["ASK"] >= prices["BID"])
    )

    return [
        date.strftime("%Y-%m-%d")
        for date in required[~valid]
    ]


def _cache_file(option_ric, required_dates):
    required = pd.DatetimeIndex(
        pd.to_datetime(required_dates)
    ).normalize()

    payload = {
        "option_ric": option_ric,
        "required_dates": sorted(
            date.strftime("%Y-%m-%d")
            for date in required.unique()
        )
    }

    key = hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True
        ).encode("utf-8")
    ).hexdigest()[:20]

    return CACHE_DIR / f"{key}.json"


def should_skip_quote_download(
    option_ric,
    required_dates,
    existing_quotes,
    retry_after_days=7,
    refresh=False
):
    if refresh or existing_quotes is None:
        return False

    missing = get_missing_quote_dates(
        existing_quotes,
        required_dates
    )

    if not missing:
        return False

    cache_file = _cache_file(
        option_ric,
        required_dates
    )

    if not cache_file.exists():
        return False

    try:
        record = json.loads(
            cache_file.read_text(encoding="utf-8")
        )

        last_attempt = datetime.fromisoformat(
            record["last_attempt_utc"]
        )

        if last_attempt.tzinfo is None:
            return False

        elapsed = (
            datetime.now(timezone.utc)
            - last_attempt
        )

        if elapsed < timedelta(0):
            return False

        return (
            record.get("option_ric") == option_ric
            and record.get("missing_dates") == missing
            and elapsed < timedelta(
                days=retry_after_days
            )
        )

    except (
        OSError,
        ValueError,
        KeyError,
        TypeError
    ):
        return False


def record_quote_download_attempt(
    option_ric,
    required_dates,
    quote_df
):
    missing = get_missing_quote_dates(
        quote_df,
        required_dates
    )

    cache_file = _cache_file(
        option_ric,
        required_dates
    )

    if not missing:
        cache_file.unlink(missing_ok=True)
        return

    CACHE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    record = {
        "option_ric": option_ric,
        "last_attempt_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "missing_dates": missing
    }

    temporary_file = cache_file.with_suffix(".tmp")

    temporary_file.write_text(
        json.dumps(
            record,
            indent=2
        ),
        encoding="utf-8"
    )

    temporary_file.replace(cache_file)
