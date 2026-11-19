import lseg.data as ld
import pandas as pd

from analytics.options_math import calculate_option_analytics
from lseg_api.pricing import get_delayed_streaming_snapshot


def get_option_chain(
    underlying_ric,
    minimum_strike=None,
    maximum_strike=None,
    max_results=2000
):
    filters = [
        "SearchAllCategoryv2 eq 'Options'",
        f"UnderlyingQuoteRIC eq '{underlying_ric}'"
    ]

    if minimum_strike is not None:
        filters.append(
            f"StrikePrice ge {float(minimum_strike):.6f}"
        )

    if maximum_strike is not None:
        filters.append(
            f"StrikePrice le {float(maximum_strike):.6f}"
        )

    filter_string = " and ".join(filters)

    options = ld.discovery.search(
        view=ld.discovery.Views.EQUITY_QUOTES,
        top=max_results,
        filter=filter_string,
        select=(
            "RIC, "
            "CallPutOption, "
            "StrikePrice, "
            "ExpiryDate, "
            "UnderlyingQuoteRIC, "
            "ExchangeName"
        ),
        order_by=(
            "ExpiryDate asc, "
            "StrikePrice asc"
        )
    )

    return options


def select_put_option(
    options,
    current_price,
    hedge_horizon_days,
    target_strike_pct=0.90
):
    data = options.copy()

    data = data[
        data["CallPutOption"] == "Put"
    ].copy()

    data["ExpiryDate"] = pd.to_datetime(
        data["ExpiryDate"]
    )

    target_expiry = (
        pd.Timestamp.today().normalize()
        + pd.Timedelta(
            days=round(
                hedge_horizon_days
                * 365
                / 252
            )
        )
    )

    data = data[
        data["ExpiryDate"]
        >= target_expiry
    ].copy()

    if data.empty:
        raise ValueError(
            "No put option was found with "
            "an expiry on or after the "
            "hedge horizon."
        )

    target_strike = (
        current_price
        * target_strike_pct
    )

    data["Expiry Distance"] = (
        data["ExpiryDate"]
        - target_expiry
    ).abs()

    data["Strike Distance"] = (
        data["StrikePrice"]
        - target_strike
    ).abs()

    data = data.sort_values(
        by=[
            "Expiry Distance",
            "Strike Distance"
        ]
    )

    return data.iloc[0]



def add_option_market_data(options, batch_size=50):
    """Retrieve BID, ASK and UTC quote clock together.

    QUOTIM is a UTC time of day, not a complete event datetime.
    QuoteRetrievedUTC records when the local snapshot returned.
    If QUOTIM is unsupported, BID/ASK retrieval still works.
    """
    from datetime import datetime, timezone

    option_rics = (
        options["RIC"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    parts = []

    for start in range(0, len(option_rics), batch_size):

        batch = option_rics[start:start + batch_size]

        try:
            try:
                raw = get_delayed_streaming_snapshot(
                    universe=batch,
                    fields=["BID", "ASK", "QUOTIM"],
                )

            except Exception as optional_error:
                print(
                    "QUOTIM unavailable; using BID/ASK only: "
                    f"{optional_error}"
                )

                raw = get_delayed_streaming_snapshot(
                    universe=batch,
                    fields=["BID", "ASK"],
                )

            retrieved_at = (
                datetime.now(timezone.utc).isoformat()
            )

            if raw is None or raw.empty:
                continue

            frame = raw.copy()

            if "Instrument" not in frame.columns:
                continue

            frame = frame.rename(
                columns={"Instrument": "RIC"}
            )

            frame["RIC"] = (
                frame["RIC"]
                .astype(str)
                .str.lstrip("/")
            )

            if not {
                "RIC",
                "BID",
                "ASK",
            }.issubset(frame.columns):
                continue

            if "QUOTIM" not in frame.columns:
                frame["QUOTIM"] = pd.NA

            frame["QuoteRetrievedUTC"] = retrieved_at

            parts.append(
                frame[
                    [
                        "RIC",
                        "BID",
                        "ASK",
                        "QUOTIM",
                        "QuoteRetrievedUTC",
                    ]
                ]
            )

        except Exception as exc:
            print(
                f"Option quote batch failed: {exc}"
            )

    if not parts:
        result = options.copy()

        for column in (
            "BID",
            "ASK",
            "QUOTIM",
            "QuoteRetrievedUTC",
        ):
            result[column] = pd.NA

        return result

    market_data = (
        pd.concat(
            parts,
            ignore_index=True,
        )
        .drop_duplicates(
            subset="RIC",
            keep="first",
        )
    )

    return options.merge(
        market_data,
        on="RIC",
        how="left",
    )


def add_option_analytics(
    options,
    current_price,
    risk_free_rate,
    dividend_yield
):
    results = []

    valuation_date = pd.Timestamp.today().normalize()

    for _, row in options.iterrows():
        option = row.copy()
        ric = option["RIC"]

        try:
            expiry_date = pd.Timestamp(
                option["ExpiryDate"]
            ).normalize()

            calendar_days = (
                expiry_date - valuation_date
            ).days

            if calendar_days <= 0:
                continue

            bid = float(option["BID"])
            ask = float(option["ASK"])

            if (
                not pd.notna(bid)
                or not pd.notna(ask)
                or bid <= 0
                or ask <= 0
                or ask < bid
            ):
                continue

            option_price = (bid + ask) / 2
            time_to_expiry = calendar_days / 365

            local_analytics = calculate_option_analytics(
                option_type=option["CallPutOption"],
                option_price=option_price,
                stock_price=float(current_price),
                strike=float(option["StrikePrice"]),
                time_to_expiry=time_to_expiry,
                risk_free_rate=float(risk_free_rate),
                dividend_yield=float(dividend_yield)
            )

            required_fields = [
                "Implied Volatility",
                "Delta",
                "Gamma",
                "Theta",
                "Vega"
            ]

            if not all(
                pd.notna(local_analytics[field])
                for field in required_fields
            ):
                continue

            for field in required_fields:
                option[field] = float(
                    local_analytics[field]
                )

            option["Analytics Source"] = "Local BSM"

            results.append(option)

        except Exception as error:
            print(
                f"Skipping {ric}: "
                f"local analytics failed: {error}"
            )

    if not results:
        return options.iloc[0:0].copy()

    return pd.DataFrame(results)

def filter_liquid_options(
    options,
    max_spread_pct=0.10
):
    data = options.copy()

    data = data[
        data["BID"].notna()
        & data["ASK"].notna()
        & (data["BID"] > 0)
        & (data["ASK"] > 0)
    ].copy()

    data["Mid Price"] = (
        data["BID"]
        + data["ASK"]
    ) / 2

    data["Bid-Ask Spread"] = (
        data["ASK"]
        - data["BID"]
    )

    data["Spread %"] = (
        data["Bid-Ask Spread"]
        / data["Mid Price"]
    )

    data = data[
        data["Spread %"]
        <= max_spread_pct
    ].copy()

    return data

def rank_put_options(
    options,
    current_price,
    hedge_horizon_days,
    target_strike_pct,
    target_delta,
    selection_mode="balanced"):
    data = options.copy()

    data = data[data["CallPutOption"] == "Put"]

    data["ExpiryDate"] = pd.to_datetime(data["ExpiryDate"])

    target_expiry = (
        pd.Timestamp.today().normalize()
        + pd.Timedelta(days=round(hedge_horizon_days * 365 / 252)))

    data = data[data["ExpiryDate"] >= target_expiry]

    if data.empty:
        raise ValueError("No valid put options remain after expiry filtering.")

    target_strike = current_price * target_strike_pct

    data["Expiry Distance Days"] = (data["ExpiryDate"] - target_expiry).dt.days

    data["Strike Distance %"] = (abs(data["StrikePrice"] - target_strike) / current_price)

    data["Delta Distance"] = (data["Delta"] - target_delta).abs()

    data["Premium %"] = (data["ASK"] / current_price)

    if selection_mode == "strike":
        data["Selection Score"] = (
            data["Strike Distance %"] * 10
            + data["Expiry Distance Days"] / 30
            + data["Spread %"] * 5
            + data["Premium %"] * 5)

    elif selection_mode == "delta":
        data["Selection Score"] = (
            data["Delta Distance"] * 10
            + data["Expiry Distance Days"] / 30
            + data["Spread %"] * 5
            + data["Premium %"] * 5)

    elif selection_mode == "balanced":
        data["Selection Score"] = (
            data["Strike Distance %"] * 10
            + data["Delta Distance"] * 2
            + data["Expiry Distance Days"] / 30
            + data["Spread %"] * 5
            + data["Premium %"] * 10)

    else:
        raise ValueError("selection_mode must be 'strike', 'delta', or 'balanced'.")

    return data.sort_values(by="Selection Score")

def rank_call_options(
    options,
    current_price,
    hedge_horizon_days,
    target_strike_pct=1.10,
    target_delta=0.25):
    data = options.copy()

    data = data[data["CallPutOption"] == "Call"]

    data["ExpiryDate"] = pd.to_datetime(data["ExpiryDate"])

    target_expiry = (pd.Timestamp.today().normalize() + pd.Timedelta(days=round(hedge_horizon_days * 365 / 252)))

    data = data[data["ExpiryDate"] >= target_expiry]

    if data.empty:
        raise ValueError("No valid call options remain after expiry filtering.")

    target_strike = current_price * target_strike_pct

    data["Expiry Distance Days"] = (data["ExpiryDate"] - target_expiry).dt.days

    data["Strike Distance %"] = ((data["StrikePrice"] - target_strike).abs() / current_price)

    data["Delta Distance"] = (data["Delta"] - target_delta).abs()

    data["Premium %"] = (data["BID"] / current_price)

    data["Selection Score"] = (
        data["Expiry Distance Days"] / 30
        + data["Strike Distance %"] * 10
        + data["Spread %"] * 5
        + data["Delta Distance"] * 2)

    data = data.sort_values(by="Selection Score")

    return data

def rank_call_options_same_expiry(
    options,
    current_price,
    expiry_date,
    target_strike_pct=1.10,
    target_delta=0.25,
    selection_mode="balanced"):
    data = options.copy()

    data = data[data["CallPutOption"] == "Call"]

    data["ExpiryDate"] = pd.to_datetime(data["ExpiryDate"])

    data = data[data["ExpiryDate"] == pd.to_datetime(expiry_date)]

    if data.empty:
        raise ValueError("No call options were found with the same expiry as the selected put.")

    target_strike = current_price * target_strike_pct

    data["Strike Distance %"] = (
        (data["StrikePrice"] - target_strike).abs()
        / current_price
    )

    data["Delta Distance"] = (
        data["Delta"] - target_delta
    ).abs()

    data["Premium %"] = (
        data["BID"] / current_price
    )

    if selection_mode == "strike":
        data["Selection Score"] = (
            data["Strike Distance %"] * 10
            + data["Spread %"] * 5
            + data["Premium %"] * 5
        )

    elif selection_mode == "delta":
        data["Selection Score"] = (
            data["Delta Distance"] * 10
            + data["Spread %"] * 5
            + data["Premium %"] * 5
        )

    elif selection_mode == "balanced":
        data["Selection Score"] = (
            data["Strike Distance %"] * 10
            + data["Delta Distance"] * 2
            + data["Spread %"] * 5
            + data["Premium %"] * 5
        )

    else:
        raise ValueError(
            "selection_mode must be "
            "'strike', 'delta', or 'balanced'."
        )

    return data.sort_values(
        by="Selection Score"
    )