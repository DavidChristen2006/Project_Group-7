
import numpy as np
import pandas as pd

from pipeline.historical_data_pipeline import (
    load_price_history
)
from pipeline.historical_option_pipeline import (
    load_option_quote_history
)


def build_historical_option_features(
    underlying_ric,
    option_contracts,
    volatility_window=20,
    momentum_window=20
):
    stock = load_price_history(underlying_ric).copy()

    stock = stock.sort_index()
    stock["Close"] = pd.to_numeric(
        stock["Close"],
        errors="coerce"
    ).astype("float64")

    # Features use only information through the decision date.
    stock["Daily Return"] = stock["Close"].pct_change(
        fill_method=None
    )

    stock["Historical Volatility"] = (
        stock["Daily Return"]
        .rolling(volatility_window)
        .std()
        * np.sqrt(252)
    )

    stock["Momentum"] = (
        stock["Close"].pct_change(
            periods=momentum_window,
            fill_method=None
        )
    )

    feature_frames = []

    for contract in option_contracts:
        option_ric = contract["RIC"]

        quotes = load_option_quote_history(
            option_ric
        ).copy()

        quotes = quotes.sort_index()

        features = quotes.join(
            stock[
                [
                    "Close",
                    "Daily Return",
                    "Historical Volatility",
                    "Momentum"
                ]
            ],
            how="inner"
        )

        features["Underlying RIC"] = underlying_ric
        features["Option RIC"] = option_ric

        features["Option Type"] = contract["Type"]
        features["Strike"] = float(contract["Strike"])

        expiry = pd.Timestamp(
            contract["Expiry"]
        ).normalize()

        features["Expiry"] = expiry

        features["Days to Expiry"] = (
            expiry - features.index
        ).days

        features = features[
            features["Days to Expiry"] > 0
        ].copy()

        features["Moneyness"] = (
            features["Close"] / features["Strike"]
        )

        features["Strike to Spot"] = (
            features["Strike"] / features["Close"]
        )

        features["Relative Spread"] = (
            features["SPREAD"] / features["MID"]
        )

        features["Decision Date"] = features.index

        feature_frames.append(
            features.reset_index(drop=True)
        )

    if not feature_frames:
        raise ValueError("No option contracts supplied.")

    dataset = pd.concat(
        feature_frames,
        ignore_index=True
    )

    dataset = dataset.sort_values(
        ["Decision Date", "Option RIC"]
    ).reset_index(drop=True)

    required_features = [
        "BID",
        "ASK",
        "Close",
        "Historical Volatility",
        "Momentum",
        "Moneyness",
        "Days to Expiry"
    ]

    dataset = dataset.dropna(
        subset=required_features
    )

    if dataset.empty:
        raise ValueError(
            "No complete historical feature observations."
        )

    return dataset
