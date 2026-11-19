
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


STRATEGIES = {
    "No Hedge",
    "Protective Put",
    "Collar",
    "Put Spread",
}

REQUIRED_COLUMNS = [
    "Underlying RIC",
    "Strategy",
    "Expiry",
    "Long Put Strike",
    "Short Call Strike",
    "Short Put Strike",
    "Predicted Improvement",
    "Predicted Drawdown",
]


@dataclass(frozen=True)
class HedgePosition:
    ric: str
    shares: int
    spot_price: float
    hedge_end_date: pd.Timestamp
    option_multiplier: int = 100

    def __post_init__(self):
        if not self.ric.strip():
            raise ValueError("An underlying RIC is required.")

        if self.shares <= 0:
            raise ValueError("Shares must be positive.")

        if not np.isfinite(self.spot_price):
            raise ValueError("Invalid stock price.")

        if self.spot_price <= 0:
            raise ValueError("Stock price must be positive.")

        if self.option_multiplier <= 0:
            raise ValueError("Invalid option multiplier.")

        if self.shares % self.option_multiplier != 0:
            raise ValueError(
                "Position is not exactly divisible by "
                "the option contract multiplier."
            )

        end_date = pd.Timestamp(
            self.hedge_end_date
        ).normalize()

        if pd.isna(end_date):
            raise ValueError("Invalid hedge end date.")

        object.__setattr__(
            self, "hedge_end_date", end_date
        )

    @property
    def contracts(self):
        return self.shares // self.option_multiplier


@dataclass(frozen=True)
class OptimizationConfig:
    max_drawdown: float = 0.10
    objective: str = "maximum_improvement"

    def __post_init__(self):
        if not np.isfinite(self.max_drawdown):
            raise ValueError("Invalid drawdown limit.")

        if not 0 < self.max_drawdown <= 1:
            raise ValueError(
                "Drawdown limit must be in (0, 1]."
            )

        if self.objective not in {
            "maximum_improvement",
            "minimum_drawdown",
        }:
            raise ValueError(
                "Unsupported optimization objective."
            )


def validate_candidate(row, position):
    strategy = row["Strategy"]

    if strategy not in STRATEGIES:
        raise ValueError(
            f"Unknown strategy: {strategy}"
        )

    if row["Underlying RIC"] != position.ric:
        raise ValueError(
            "Candidate belongs to another underlying."
        )

    long_put = row["Long Put Strike"]
    short_call = row["Short Call Strike"]
    short_put = row["Short Put Strike"]

    if strategy == "No Hedge":
        if any(
            pd.notna(strike)
            for strike in [long_put, short_call, short_put]
        ):
            raise ValueError(
                "No Hedge must have no option strikes."
            )

        if pd.notna(row["Expiry"]):
            raise ValueError(
                "No Hedge must have no option expiry."
            )

        if not np.isclose(
            row["Predicted Improvement"],
            0.0,
            atol=1e-12,
        ):
            raise ValueError(
                "No Hedge improvement must be zero."
            )

        return

    expiry = row["Expiry"]

    if pd.isna(expiry):
        raise ValueError(
            "An option strategy requires an expiry."
        )

    if expiry < position.hedge_end_date:
        raise ValueError(
            "Option expires before the hedge ends."
        )

    if pd.isna(long_put) or long_put <= 0:
        raise ValueError(
            "Every option hedge requires a long put."
        )

    if strategy == "Protective Put":
        if pd.notna(short_call) or pd.notna(short_put):
            raise ValueError(
                "Invalid Protective Put legs."
            )

    elif strategy == "Collar":
        if pd.isna(short_call):
            raise ValueError(
                "Collar requires a short call."
            )

        if short_call <= long_put:
            raise ValueError(
                "Short call strike must exceed "
                "the long put strike."
            )

        if pd.notna(short_put):
            raise ValueError(
                "Collar cannot contain a short put."
            )

    elif strategy == "Put Spread":
        if pd.isna(short_put):
            raise ValueError(
                "Put Spread requires a short put."
            )

        if short_put >= long_put:
            raise ValueError(
                "Short put strike must be below "
                "the long put strike."
            )

        if pd.notna(short_call):
            raise ValueError(
                "Put Spread cannot contain a short call."
            )


def optimize_hedge_candidates(
    candidates,
    position,
    config,
):
    missing = set(REQUIRED_COLUMNS) - set(
        candidates.columns
    )

    if missing:
        raise ValueError(
            f"Missing candidate columns: {sorted(missing)}"
        )

    if candidates.empty:
        raise ValueError(
            "No candidates were provided."
        )

    df = candidates.copy()

    df["Expiry"] = pd.to_datetime(
        df["Expiry"],
        errors="raise",
    ).dt.normalize()

    numeric_columns = [
        "Long Put Strike",
        "Short Call Strike",
        "Short Put Strike",
        "Predicted Improvement",
        "Predicted Drawdown",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="raise",
        )

    if not np.isfinite(
        df[
            [
                "Predicted Improvement",
                "Predicted Drawdown",
            ]
        ].to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "Predicted outcomes must be finite."
        )

    if (
        (df["Predicted Drawdown"] < -1)
        | (df["Predicted Drawdown"] > 0)
    ).any():
        raise ValueError(
            "Expected negative fractional drawdowns "
            "between -1 and 0."
        )

    for _, row in df.iterrows():
        validate_candidate(row, position)

    if (df["Strategy"] == "No Hedge").sum() != 1:
        raise ValueError(
            "Exactly one No Hedge candidate is required."
        )

    df["Eligible"] = (
        df["Predicted Drawdown"]
        >= -config.max_drawdown
    )

    feasible = df.loc[
        df["Eligible"]
    ].copy()

    if feasible.empty:
        return {
            "Status": "No Feasible Strategy",
            "Recommendation": None,
            "Candidates": df,
        }

    if config.objective == "maximum_improvement":
        ranking_columns = [
            "Predicted Improvement",
            "Predicted Drawdown",
        ]

    else:
        ranking_columns = [
            "Predicted Drawdown",
            "Predicted Improvement",
        ]

    feasible = feasible.sort_values(
        ranking_columns,
        ascending=[False, False],
        kind="mergesort",
    )

    best = feasible.iloc[0].copy()

    best["Contracts"] = (
        0
        if best["Strategy"] == "No Hedge"
        else position.contracts
    )

    return {
        "Status": "Feasible",
        "Recommendation": best,
        "Candidates": df,
    }
