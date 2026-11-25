"""lseg_api/equity.py: hedge research implementation.

Functions are documented at their point of definition.
Historical outputs are research estimates, not executable orders.
"""

import lseg.data as ld
import pandas as pd

from models.stock_metadata import StockMetadata
from lseg_api.pricing import get_delayed_streaming_snapshot


def get_current_equity_price(ric):
    """Get current equity price from the current context.

    Args:
        ric: Input to the get_current_equity_price operation.

    Returns:
        Computed result or status for the caller.
    """
    try:
        data = get_delayed_streaming_snapshot(
            universe=[ric],
            fields=["TRDPRC_1"]
        )

        if data.empty:
            return None

        price = data["TRDPRC_1"].iloc[0]

        if pd.isna(price):
            return None

        return float(price)

    except Exception as error:
        print(
            f"Current price unavailable for {ric}: {error}"
        )

        return None
    
def get_equity_history(
        ric: str,
        start_date: str,
        end_date: str,
        interval: str = "1d"
) -> pd.DataFrame:
    """Get equity history from the current context.

    Args:
        ric: Input to the get_equity_history operation.
        start_date: Input to the get_equity_history operation.
        end_date: Input to the get_equity_history operation.
        interval: Input to the get_equity_history operation.

    Returns:
        Computed result or status for the caller.
    """
    data = ld.get_history(
        universe=ric,
        fields=[
            "OPEN_PRC",
            "HIGH_1",
            "LOW_1",
            "TRDPRC_1",
            "ACVOL_UNS"],
        start=start_date,
        end=end_date,
        interval=interval)
    data = data.rename(columns={
    "OPEN_PRC": "Open",
    "HIGH_1": "High",
    "LOW_1": "Low",
    "TRDPRC_1": "Close",
    "ACVOL_UNS": "Volume"})
    data["Return"] = data["Close"].pct_change()
    data["Running_Max"] = data["Close"].cummax()
    data["Drawdown"] = data["Close"] / data["Running_Max"] - 1
    data["Volatility_20D"] = data["Return"].rolling(20).std()
    data["Volatility_20D_Annualized"] = (
    data["Volatility_20D"] * (252 ** 0.5))
    return data

def get_dividend_yield(
    ric,
    lookback_years=2
):
    """Get dividend yield from the current context.

    Args:
        ric: Input to the get_dividend_yield operation.
        lookback_years: Input to the get_dividend_yield operation.

    Returns:
        Computed result or status for the caller.
    """
    try:
        data = ld.get_data(
            universe=[ric],
            fields=[
                "TR.DividendYield"
            ]
        )

        if not data.empty:
            value = data.iloc[0, 1]

            if not pd.isna(value):
                return float(value) / 100

    except Exception as error:
        print(
            f"Current dividend yield unavailable "
            f"for {ric}: {error}"
        )

    end_date = (
        pd.Timestamp.today()
        .normalize()
    )

    start_date = (
        end_date
        - pd.DateOffset(
            years=lookback_years
        )
    )

    try:
        historical_data = ld.get_data(
            universe=[ric],
            fields=[
                "TR.DividendYield.date",
                "TR.DividendYield"
            ],
            parameters={
                "SDate":
                    start_date.strftime(
                        "%Y-%m-%d"
                    ),
                "EDate":
                    end_date.strftime(
                        "%Y-%m-%d"
                    ),
                "Frq": "D"
            }
        )

        if not historical_data.empty:

            yield_column = (
                historical_data.columns[-1]
            )

            valid_yields = (
                historical_data[
                    yield_column
                ]
                .dropna()
            )

            if not valid_yields.empty:
                return float(
                    valid_yields.iloc[-1]
                ) / 100

    except Exception as error:
        print(
            f"Historical dividend yield "
            f"unavailable for {ric}: "
            f"{error}"
        )

    print(
        f"No dividend yield available "
        f"for {ric}. Using 0.0."
    )

    return 0.0

def get_stock_metadata(ric):
    """Get stock metadata from the current context.

    Args:
        ric: Input to the get_stock_metadata operation.

    Returns:
        Computed result or status for the caller.
    """
    data = ld.get_data(
        universe=[ric],
        fields=[
            "TR.CommonName",
            "TR.HeadquartersCountry",
            "TR.ExchangeName",
            "TR.TRBCEconomicSector",
            "TR.TRBCIndustryGroup",
            "TR.TRBCIndustry"
        ],
        header_type=ld.HeaderType.NAME
    )

    if data.empty:
        raise ValueError(
            f"No stock metadata available for {ric}."
        )

    return StockMetadata(
        name=data["TR.CommonName"].iloc[0],
        country=data["TR.HeadquartersCountry"].iloc[0],
        exchange=data["TR.ExchangeName"].iloc[0],
        economic_sector=data["TR.TRBCEconomicSector"].iloc[0],
        industry_group=data["TR.TRBCIndustryGroup"].iloc[0],
        industry=data["TR.TRBCIndustry"].iloc[0],
        currency=None
    )

def get_quote_currency(ric):
    """Get quote currency from the current context.

    Args:
        ric: Input to the get_quote_currency operation.

    Returns:
        Computed result or status for the caller.
    """
    data = ld.get_data(
        universe=[ric],
        fields=[
            "TR.PriceClose.currency"
        ],
        header_type=ld.HeaderType.NAME
    )

    if data.empty or data.shape[1] < 2:
        raise ValueError(
            f"No currency data available for {ric}."
        )

    currency = data.iloc[0, 1]

    if pd.isna(currency):
        raise ValueError(
            f"Currency is missing for {ric}."
        )

    return str(currency)