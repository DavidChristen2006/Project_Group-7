
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class HedgeConstraints:
    # Minimum initial upside allowance for Collars.
    # Example: 0.05 means short call strike >= 105% of spot.
    min_call_otm_pct: float = 0.05

    # Maximum initial net debit relative to
    # the position value, before commissions.
    max_net_debit_pct: float = 0.02

    allowed_strategies: tuple = (
        "Protective Put",
        "Collar",
        "Put Spread",
    )


def filter_eligible_candidates(
    candidates: pd.DataFrame,
    constraints: HedgeConstraints,
) -> pd.DataFrame:
    """
    Filter hedge candidates using decision-time data only.

    Does not inspect future returns, drawdowns,
    hedge improvements or other realized outcomes.
    """
    if not np.isfinite(
        [
            constraints.min_call_otm_pct,
            constraints.max_net_debit_pct,
        ]
    ).all():
        raise ValueError(
            "Constraint values must be finite."
        )

    if constraints.min_call_otm_pct < 0:
        raise ValueError(
            "Minimum call OTM must be nonnegative."
        )

    if constraints.max_net_debit_pct < 0:
        raise ValueError(
            "Maximum net debit must be nonnegative."
        )

    required = [
        "Strategy",
        "Long Put Strike Pct",
        "Short Call Strike Pct",
        "Short Put Strike Pct",
        "Net Premium Pct",
    ]

    missing = [
        column for column in required
        if column not in candidates.columns
    ]

    if missing:
        raise ValueError(
            f"Missing candidate columns: {missing}"
        )

    eligible = candidates.loc[
        candidates["Strategy"].isin(
            constraints.allowed_strategies
        )
    ].copy()

    # Net Premium Pct is positive for a net
    # debit and negative for a net credit.
    # This is an initial premium constraint,
    # not a complete transaction-cost budget.
    eligible = eligible.loc[
        eligible["Net Premium Pct"]
        <= constraints.max_net_debit_pct
    ]

    # A Collar's short call limits upside.
    is_collar = eligible["Strategy"].eq(
        "Collar"
    )

    call_allowed = (
        eligible["Short Call Strike Pct"]
        >= 1.0 + constraints.min_call_otm_pct
    )

    eligible = eligible.loc[
        (~is_collar) | call_allowed
    ]

    # Ensure every retained candidate has
    # a valid long put below spot.
    eligible = eligible.loc[
        eligible["Long Put Strike Pct"].gt(0)
        & eligible["Long Put Strike Pct"].lt(1)
    ]

    # Validate the short put in a Put Spread.
    is_spread = eligible["Strategy"].eq(
        "Put Spread"
    )

    spread_allowed = (
        eligible["Short Put Strike Pct"].gt(0)
        & (
            eligible["Short Put Strike Pct"]
            < eligible["Long Put Strike Pct"]
        )
    )

    eligible = eligible.loc[
        (~is_spread) | spread_allowed
    ]

    return eligible.copy()
