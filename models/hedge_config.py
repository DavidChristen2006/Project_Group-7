from dataclasses import dataclass

@dataclass
class HedgeConfig:
    put_strike_pct: float
    put_delta: float
    call_strike_pct: float
    call_delta: float
    short_put_strike_pct: float
    cross_hedge_ratio: float