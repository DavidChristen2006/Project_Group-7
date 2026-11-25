"""lseg_api/option_history.py: hedge research implementation.

Functions are documented at their point of definition.
Historical outputs are research estimates, not executable orders.
"""

import pandas as pd
import lseg.data as ld
import time


def get_latest_option_quote(option_ric, count=10):
    """Get latest option quote from the current context.

    Args:
        option_ric: Input to the get_latest_option_quote operation.
        count: Input to the get_latest_option_quote operation.

    Returns:
        Computed result or status for the caller.
    """
    history_df = ld.get_history(
        universe=option_ric,
        fields=["BID", "ASK"],
        interval="daily",
        count=count
    )

    if history_df is None or history_df.empty:
        return None

    history_df = history_df.copy()

    history_df["BID"] = pd.to_numeric(
        history_df["BID"], errors="coerce"
    )

    history_df["ASK"] = pd.to_numeric(
        history_df["ASK"], errors="coerce"
    )

    valid_quotes = history_df[
        history_df["BID"].notna()
        & history_df["ASK"].notna()
        & (history_df["BID"] >= 0)
        & (history_df["ASK"] > 0)
        & (history_df["ASK"] >= history_df["BID"])
    ]

    if valid_quotes.empty:
        return None

    latest_date = valid_quotes.index.max()
    latest = valid_quotes.loc[latest_date]

    return {
        "RIC": option_ric,
        "Date": pd.Timestamp(latest_date).date(),
        "BID": float(latest["BID"]),
        "ASK": float(latest["ASK"]),
        "MID": float(
            (latest["BID"] + latest["ASK"]) / 2
        ),
        "SPREAD": float(
            latest["ASK"] - latest["BID"]
        )
    }

def fill_historical_option_quotes(
    options,
    batch_size=25,
    count=5,
    max_age_days=7
):
    """Perform the fill historical option quotes operation for this research workflow.

    Args:
        options: Input to the fill_historical_option_quotes operation.
        batch_size: Input to the fill_historical_option_quotes operation.
        count: Input to the fill_historical_option_quotes operation.
        max_age_days: Input to the fill_historical_option_quotes operation.

    Returns:
        Computed result or status for the caller.
    """
    result = options.copy()

    if result.empty:
        return result

    result["BID"] = pd.to_numeric(
        result["BID"], errors="coerce"
    )
    result["ASK"] = pd.to_numeric(
        result["ASK"], errors="coerce"
    )

    valid_current = (
        result["BID"].notna()
        & result["ASK"].notna()
        & (result["BID"] >= 0)
        & (result["ASK"] > 0)
        & (result["ASK"] >= result["BID"])
    )

    result["QuoteSource"] = "Current"
    result["QuoteDate"] = pd.NaT

    missing_rics = (
        result.loc[~valid_current, "RIC"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    if not missing_rics:
        return result

    historical_quotes = {}
    today = pd.Timestamp.now(tz="UTC").normalize()

    for i in range(0, len(missing_rics), batch_size):
        batch = missing_rics[i:i + batch_size]

        try:
            history = ld.get_history(
                universe=batch,
                fields=["BID", "ASK"],
                interval="daily",
                count=count
            )

            if history is None or history.empty:
                continue

            for ric in batch:
                if isinstance(history.columns, pd.MultiIndex):
                    try:
                        if ric in history.columns.get_level_values(0):
                            quotes = history.xs(
                                ric, axis=1, level=0
                            )
                        elif ric in history.columns.get_level_values(1):
                            quotes = history.xs(
                                ric, axis=1, level=1
                            )
                        else:
                            continue
                    except KeyError:
                        continue

                elif len(batch) == 1:
                    quotes = history.copy()

                else:
                    raise ValueError(
                        "Unexpected multi-RIC history format: "
                        f"{history.columns.tolist()}"
                    )

                if not {"BID", "ASK"}.issubset(quotes.columns):
                    continue

                quotes = quotes[["BID", "ASK"]].copy()
                quotes["BID"] = pd.to_numeric(
                    quotes["BID"], errors="coerce"
                )
                quotes["ASK"] = pd.to_numeric(
                    quotes["ASK"], errors="coerce"
                )

                quotes = quotes[
                    (quotes["BID"] >= 0)
                    & (quotes["ASK"] > 0)
                    & (quotes["ASK"] >= quotes["BID"])
                ].sort_index()

                if quotes.empty:
                    continue

                latest_date = pd.Timestamp(
                    quotes.index[-1]
                ).normalize()

                quote_date_utc = (
                    latest_date.tz_localize("UTC")
                    if latest_date.tzinfo is None
                    else latest_date.tz_convert("UTC")
                )

                age_days = (today - quote_date_utc).days

                if age_days < 0 or age_days > max_age_days:
                    continue

                latest = quotes.iloc[-1]

                historical_quotes[ric] = {
                    "BID": float(latest["BID"]),
                    "ASK": float(latest["ASK"]),
                    "Date": latest_date
                }

        except Exception as exc:
            print(
                f"Historical quote batch {i // batch_size + 1} "
                f"failed: {exc}"
            )

            print("Retrying contracts individually...")

            for ric in batch:
                try:
                    quote = get_latest_option_quote(
                        option_ric=ric,
                        count=count
                    )

                    if quote is None:
                        continue

                    quote_date = pd.Timestamp(
                        quote["Date"]
                    ).normalize()

                    quote_date_utc = quote_date.tz_localize("UTC")

                    age_days = (
                        today - quote_date_utc
                    ).days

                    if (
                        age_days < 0
                        or age_days > max_age_days
                    ):
                        continue

                    historical_quotes[ric] = {
                        "BID": quote["BID"],
                        "ASK": quote["ASK"],
                        "Date": quote_date
                    }

                except Exception as individual_exc:
                    print(
                        f"Skipping {ric}: "
                        f"{individual_exc}"
                    )

                time.sleep(0.35)

    updated = 0

    for idx in result.index[~valid_current]:
        ric = str(result.at[idx, "RIC"])
        quote = historical_quotes.get(ric)

        if quote is None:
            continue

        result.at[idx, "BID"] = quote["BID"]
        result.at[idx, "ASK"] = quote["ASK"]
        result.at[idx, "QuoteDate"] = quote["Date"]
        result.at[idx, "QuoteSource"] = "Historical"

        updated += 1

    print(
        f"Historical quote fallback: "
        f"{updated}/{len(missing_rics)} contracts updated"
    )

    return result
