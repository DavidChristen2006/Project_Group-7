
import numpy as np
import pandas as pd

from pipeline.historical_data_pipeline import (
    load_price_history
)


def build_ml_training_features(historical_df):
    if historical_df.empty:
        raise ValueError("Historical dataset is empty.")

    result = historical_df.copy()

    result["Decision Date"] = pd.to_datetime(
        result["Decision Date"]
    ).dt.normalize()

    result["Outcome Date"] = pd.to_datetime(
        result["Outcome Date"]
    ).dt.normalize()

    if (
        result[
            ["Underlying RIC", "Decision Date",
             "Hedge Horizon Days", "Strategy"]
        ].duplicated().any()
    ):
        raise ValueError("Duplicate historical observations.")

    feature_frames = []

    for ric, group in result.groupby("Underlying RIC"):
        stock = load_price_history(ric).copy()

        stock.index = pd.to_datetime(
            stock.index
        ).normalize()

        stock = stock.sort_index()

        stock["Close"] = pd.to_numeric(
            stock["Close"],
            errors="coerce"
        ).astype(float)

        returns = stock["Close"].pct_change(
            fill_method=None
        )

        stock["Volatility_20D"] = (
            returns.rolling(20).std() * np.sqrt(252)
        )

        stock["Volatility_60D"] = (
            returns.rolling(60).std() * np.sqrt(252)
        )

        stock["Momentum_20D"] = stock["Close"].pct_change(
            periods=20,
            fill_method=None
        )

        stock["Momentum_60D"] = stock["Close"].pct_change(
            periods=60,
            fill_method=None
        )

        rolling_peak = stock["Close"].rolling(
            window=20,
            min_periods=20
        ).max()

        stock["Historical_Drawdown_20D"] = (
            stock["Close"] / rolling_peak - 1.0
        )

        stock_features = stock[
            [
                "Volatility_20D",
                "Volatility_60D",
                "Momentum_20D",
                "Momentum_60D",
                "Historical_Drawdown_20D"
            ]
        ].copy()

        current = group.join(
            stock_features,
            on="Decision Date",
            how="left"
        )

        feature_frames.append(current)

    result = pd.concat(
        feature_frames,
        ignore_index=True
    )

    # Normalize option strikes by the historical stock price.
    leg_prefixes = [
        "Long_Put",
        "Short_Put",
        "Short_Call"
    ]

    for prefix in leg_prefixes:
        strike_column = f"{prefix}_Strike"

        if strike_column in result.columns:
            result[f"{prefix}_Strike_Pct"] = (
                pd.to_numeric(
                    result[strike_column],
                    errors="coerce"
                ) / result["Stock Price"]
            )

    required_columns = [
        "Volatility_20D",
        "Volatility_60D",
        "Momentum_20D",
        "Momentum_60D",
        "Historical_Drawdown_20D"
    ]

    if result[required_columns].isna().any().any():
        raise ValueError(
            "Historical features missing. "
            "Check stock data and lookback coverage."
        )

    if not (
        result["Outcome Date"] > result["Decision Date"]
    ).all():
        raise ValueError("Invalid outcome dates.")

    result = result.sort_values(
        ["Decision Date", "Hedge Horizon Days", "Strategy"]
    ).reset_index(drop=True)

    return result
