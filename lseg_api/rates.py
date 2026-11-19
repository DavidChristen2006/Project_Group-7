import lseg.data as ld
import pandas as pd


def get_rate_ric(
    currency,
    hedge_horizon_days
):
    currency = currency.upper()

    if hedge_horizon_days <= 21:
        tenor = "1M"

    elif hedge_horizon_days <= 63:
        tenor = "3M"

    elif hedge_horizon_days <= 126:
        tenor = "6M"

    else:
        tenor = "1Y"

    rate_rics = {
        "USD": {
            "1M": "US1MT=RR",
            "3M": "US3MT=RR",
            "6M": "US6MT=RR",
            "1Y": "US1YT=RR"
        },

        "CHF": {
            "1M": "CHF1MD=",
            "3M": "CHF3MD=",
            "6M": "CHF6MD=",
            "1Y": "CHF1YD="
        },

        "EUR": {
            "1M": "EUR1MD=",
            "3M": "EUR3MD=",
            "6M": "EUR6MD=",
            "1Y": "EUR1YD="
        },

        "GBP": {
            "1M": "GBP1MD=",
            "3M": "GBP3MD=",
            "6M": "GBP6MD=",
            "1Y": "GBP1YD="
        }
    }

    if currency not in rate_rics:
        raise ValueError(
            f"No risk-free-rate mapping "
            f"defined for currency {currency}."
        )

    return rate_rics[currency][tenor]


def get_risk_free_rate(
    currency,
    hedge_horizon_days
):
    ric = get_rate_ric(
        currency=currency,
        hedge_horizon_days=hedge_horizon_days
    )

    data = ld.get_data(
        universe=[ric],
        fields=["BID"],
        header_type=ld.HeaderType.NAME
    )

    if data.empty or data.shape[1] < 2:
        raise ValueError(
            f"No risk-free rate available "
            f"for {currency} using {ric}."
        )

    rate_pct = data.iloc[0, 1]

    if pd.isna(rate_pct):
        raise ValueError(
            f"Risk-free rate is missing "
            f"for {currency} using {ric}."
        )

    return float(rate_pct) / 100