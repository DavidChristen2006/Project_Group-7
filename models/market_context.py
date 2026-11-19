from dataclasses import dataclass

@dataclass
class MarketContext:
    current_price: float
    dividend_yield: float
    risk_free_rate: float