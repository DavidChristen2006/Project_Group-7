from dataclasses import dataclass

@dataclass
class Position:
    ric: str
    shares: int
    hedge_horizon_days: int

    def __post_init__(self):
        if self.shares <= 0:
            raise ValueError("shares must be greater than zero.")

        if self.hedge_horizon_days <= 0:
            raise ValueError("hedge_horizon_days must be greater than zero.")