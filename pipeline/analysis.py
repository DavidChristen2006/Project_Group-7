import pandas as pd

from models.market_context import MarketContext
from lseg_api.equity import get_equity_history, get_current_equity_price, get_dividend_yield, get_stock_metadata, get_quote_currency
from lseg_api.rates import get_risk_free_rate
from analytics.historical_risk import analyze_historical_risk
from analytics.ml_dataset import build_ml_rows

def analyze_position(position, history_years=5):
    metadata = get_stock_metadata(position.ric)
    metadata.currency = get_quote_currency(position.ric)

    end_date = pd.Timestamp.today().normalize()
    start_date = end_date - pd.DateOffset(years=history_years)

    history = get_equity_history(
        ric=position.ric,
        start_date=start_date,
        end_date=end_date
    )

    current_price = get_current_equity_price(
        position.ric
    )

    if pd.isna(current_price):
        valid_close_prices = (
            history["Close"]
            .dropna()
        )

        if valid_close_prices.empty:
            raise ValueError(
                f"No current or historical price "
                f"available for {position.ric}."
            )

        current_price = float(
            valid_close_prices.iloc[-1]
        )
    else:
        current_price = float(
            current_price
        )

    dividend_yield = get_dividend_yield(
        position.ric
    )

    risk_free_rate = get_risk_free_rate(
    metadata.currency,
    position.hedge_horizon_days)

    market = MarketContext(
        current_price=current_price,
        dividend_yield=dividend_yield,
        risk_free_rate=risk_free_rate
    )

    historical_risk = analyze_historical_risk(history=history, shares=position.shares)

    return {
        "position": position,
        "metadata": metadata,
        "market": market,
        "history": history,
        "historical_risk": historical_risk,
        "start_date": start_date,
        "end_date": end_date}