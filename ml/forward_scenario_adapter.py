
"""Research-only forward scenario generation for equity hedges.

Stock scenarios:
    Moving-block bootstrap of historical log returns, using data
    strictly before the decision date.

Option exit values:
    European Black-Scholes approximation, with fixed implied
    volatility, risk-free rate, and dividend yield.

Important limitations:
    Not an American-option valuation model.
    No early exercise, assignment, margin calls, funding carry,
    or changing volatility surface.
    Exit bid/ask prices are modeled, not observed or guaranteed.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd
from scipy.special import ndtr


@dataclass(frozen=True)
class OptionLeg:
    ric: str
    option_type: str
    side: str
    strike: float
    contracts: int
    expiry_years: float
    implied_volatility: float
    bid: float
    ask: float
    exit_spread_pct: float


@dataclass(frozen=True)
class HedgeCandidate:
    candidate_id: str
    strategy: str
    legs: tuple[OptionLeg, ...]
    entry_fees_cash: float = 0.0
    exit_fees_cash: float = 0.0
    additional_margin_cash: float = 0.0


def european_option_mid(
    spot: np.ndarray,
    strike: float,
    remaining_years: float,
    volatility: float,
    rate: float,
    dividend_yield: float,
    option_type: str,
) -> np.ndarray:
    """Vectorized European option valuation approximation."""

    s = np.asarray(spot, dtype=float)

    parameters = [
        strike,
        remaining_years,
        volatility,
        rate,
        dividend_yield,
    ]

    if (
        s.ndim != 1
        or len(s) == 0
        or not np.isfinite(s).all()
        or (s <= 0).any()
        or not np.isfinite(parameters).all()
        or strike <= 0
        or remaining_years <= 0
        or volatility <= 0
        or option_type not in {"put", "call"}
    ):
        raise ValueError("Invalid option valuation inputs")

    vol_time = volatility * math.sqrt(
        remaining_years
    )

    d1 = (
        np.log(s / strike)
        + (
            rate
            - dividend_yield
            + volatility**2 / 2
        ) * remaining_years
    ) / vol_time

    d2 = d1 - vol_time

    discounted_spot = (
        s * math.exp(
            -dividend_yield * remaining_years
        )
    )

    discounted_strike = (
        strike * math.exp(
            -rate * remaining_years
        )
    )

    if option_type == "call":
        price = (
            discounted_spot * ndtr(d1)
            - discounted_strike * ndtr(d2)
        )
    else:
        price = (
            discounted_strike * ndtr(-d2)
            - discounted_spot * ndtr(-d1)
        )

    return np.maximum(price, 0.0)


def _check_candidate(
    candidate: HedgeCandidate,
    *,
    shares: int,
    multiplier: int,
    horizon_years: float,
) -> None:
    """Validate option structure and position coverage."""

    if not isinstance(candidate, HedgeCandidate):
        raise TypeError("Expected HedgeCandidate")

    if (
        not isinstance(candidate.candidate_id, str)
        or not candidate.candidate_id.strip()
    ):
        raise ValueError("Blank candidate_id")

    if candidate.candidate_id == "NO_HEDGE":
        raise ValueError(
            "NO_HEDGE candidate ID is reserved"
        )

    patterns = {
        "Protective Put": (
            ("put", "long"),
        ),
        "Collar": (
            ("put", "long"),
            ("call", "short"),
        ),
        "Put Spread": (
            ("put", "long"),
            ("put", "short"),
        ),
    }

    if candidate.strategy not in patterns:
        raise ValueError(
            f"Unsupported strategy: {candidate.strategy}"
        )

    actual = tuple(
        (leg.option_type, leg.side)
        for leg in candidate.legs
    )

    if actual != patterns[candidate.strategy]:
        raise ValueError(
            f"Incorrect option legs for {candidate.strategy}"
        )

    for amount in (
        candidate.entry_fees_cash,
        candidate.exit_fees_cash,
        candidate.additional_margin_cash,
    ):
        if not np.isfinite(amount) or amount < 0:
            raise ValueError(
                "Fees and additional margin must "
                "be nonnegative and finite"
            )

    for leg in candidate.legs:

        if (
            not isinstance(leg.ric, str)
            or not leg.ric.strip()
        ):
            raise ValueError("Option RIC missing")

        if (
            not isinstance(
                leg.contracts,
                (int, np.integer),
            )
            or isinstance(leg.contracts, bool)
        ):
            raise ValueError(
                "Option contracts must be an integer"
            )

        if leg.contracts * multiplier != shares:
            raise ValueError(
                "This prototype requires 100% "
                "share-equivalent hedge coverage"
            )

        values = [
            leg.strike,
            leg.expiry_years,
            leg.implied_volatility,
            leg.bid,
            leg.ask,
            leg.exit_spread_pct,
        ]

        if not np.isfinite(values).all():
            raise ValueError(
                "Nonfinite option parameters"
            )

        if (
            leg.strike <= 0
            or leg.expiry_years <= horizon_years
            or leg.implied_volatility <= 0
            or leg.bid < 0
            or leg.ask <= 0
            or leg.ask < leg.bid
            or not 0 <= leg.exit_spread_pct < 2
        ):
            raise ValueError(
                "Invalid quotes, volatility, spread, "
                "or expiry before hedge exit"
            )

    if len(candidate.legs) > 1:

        if not np.isclose(
            candidate.legs[0].expiry_years,
            candidate.legs[1].expiry_years,
            rtol=0.0,
            atol=1e-8,
        ):
            raise ValueError(
                "Two-leg strategy expiries must match"
            )

    if candidate.strategy == "Put Spread":

        if (
            candidate.legs[0].strike
            <= candidate.legs[1].strike
        ):
            raise ValueError(
                "Put Spread must buy a higher-strike put"
            )


def generate_full_horizon_scenarios(
    *,
    historical_close: pd.Series,
    decision_date: str | pd.Timestamp,
    spot: float,
    shares: int,
    hedge_horizon_days: int,
    candidates: list[HedgeCandidate],
    option_multiplier: int = 100,
    n_scenarios: int = 2000,
    block_length: int = 5,
    min_history_returns: int = 252,
    seed: int = 42,
    risk_free_rate: float = 0.03,
    dividend_yield: float = 0.0,
) -> pd.DataFrame:
    """Generate matched full-horizon terminal-return scenarios.

    Existing stock position is assumed to be held throughout.

    Returns include:
      - stock capital gain/loss
      - option entry BID/ASK
      - modeled option liquidation BID/ASK at horizon
      - explicit entry and exit fees

    Additional margin is a cash requirement, not a P&L cost.
    No funding interest or margin-call dynamics are modeled.
    """

    for name, value in (
        ("spot", spot),
        ("risk_free_rate", risk_free_rate),
        ("dividend_yield", dividend_yield),
    ):
        if not np.isfinite(value):
            raise ValueError(f"Nonfinite {name}")

    if spot <= 0:
        raise ValueError("Spot must be positive")

    if (
        not isinstance(shares, (int, np.integer))
        or isinstance(shares, bool)
        or shares <= 0
        or not isinstance(
            option_multiplier,
            (int, np.integer),
        )
        or option_multiplier <= 0
        or shares % option_multiplier != 0
    ):
        raise ValueError(
            "Shares must be positive and divisible "
            "by option multiplier"
        )

    integer_settings = (
        ("hedge_horizon_days", hedge_horizon_days, 1),
        ("n_scenarios", n_scenarios, 100),
        ("block_length", block_length, 1),
        ("min_history_returns", min_history_returns, 100),
    )

    for name, value, minimum in integer_settings:

        if (
            isinstance(value, bool)
            or not isinstance(
                value,
                (int, np.integer),
            )
            or value < minimum
        ):
            raise ValueError(
                f"{name} must be an integer >= {minimum}"
            )

    if (
        not isinstance(candidates, (list, tuple))
        or not candidates
    ):
        raise ValueError(
            "Supply at least one eligible hedge candidate"
        )

    ids = [c.candidate_id for c in candidates]

    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate candidate IDs")

    horizon_years = hedge_horizon_days / 252.0

    for candidate in candidates:

        _check_candidate(
            candidate,
            shares=shares,
            multiplier=option_multiplier,
            horizon_years=horizon_years,
        )

    if not isinstance(historical_close, pd.Series):
        raise TypeError(
            "historical_close must be a dated pandas Series"
        )

    if not isinstance(
        historical_close.index,
        pd.DatetimeIndex,
    ):
        raise ValueError(
            "historical_close requires DatetimeIndex"
        )

    if (
        historical_close.index.has_duplicates
        or not historical_close.index.is_monotonic_increasing
    ):
        raise ValueError(
            "Historical price index must be unique and increasing"
        )

    date = pd.Timestamp(
        decision_date
    ).normalize()

    if (
        historical_close.index.normalize() >= date
    ).any():
        raise ValueError(
            "Historical data contains decision-date "
            "or future prices"
        )

    closes = pd.to_numeric(
        historical_close,
        errors="coerce",
    ).to_numpy(dtype=float)

    if (
        not np.isfinite(closes).all()
        or (closes <= 0).any()
    ):
        raise ValueError(
            "Historical closing prices must "
            "be finite and positive"
        )

    if len(closes) - 1 < min_history_returns:
        raise ValueError(
            "Insufficient historical returns for bootstrap"
        )

    log_returns = np.diff(
        np.log(closes)
    )

    if block_length > len(log_returns):
        raise ValueError(
            "Block length exceeds historical returns"
        )

    # Sample consecutive blocks of historical returns.
    # Each simulated path contains hedge_horizon_days returns.
    rng = np.random.default_rng(seed)

    blocks = math.ceil(
        hedge_horizon_days / block_length
    )

    starts = rng.integers(
        0,
        len(log_returns) - block_length + 1,
        size=(n_scenarios, blocks),
    )

    indices = (
        starts[:, :, None]
        + np.arange(block_length)[None, None, :]
    )

    sampled = log_returns[indices].reshape(
        n_scenarios,
        -1,
    )[:, :hedge_horizon_days]

    terminal_spot = (
        spot * np.exp(
            sampled.sum(axis=1)
        )
    )

    if (
        not np.isfinite(terminal_spot).all()
        or (terminal_spot <= 0).any()
    ):
        raise ValueError(
            "Bootstrap generated invalid terminal prices"
        )

    stock_return = (
        terminal_spot / spot - 1.0
    )

    initial_stock_value = shares * spot

    scenario_ids = [
        f"S{i:06d}"
        for i in range(n_scenarios)
    ]

    # No Hedge has the actual simulated stock return.
    results = [
        pd.DataFrame({
            "Candidate ID": "NO_HEDGE",
            "Strategy": "No Hedge",
            "Scenario ID": scenario_ids,
            "Hedge Horizon Days": hedge_horizon_days,
            "Net Terminal Return": stock_return,
            "Upfront Cash Required": 0.0,
            "Underlying Terminal Price": terminal_spot,
        })
    ]

    for candidate in candidates:

        option_pnl = np.zeros(
            n_scenarios,
            dtype=float,
        )

        net_entry_debit = 0.0

        for leg in candidate.legs:

            sign = (
                1 if leg.side == "long"
                else -1
            )

            # Buy at ASK; sell at BID.
            entry_price = (
                leg.ask if sign == 1
                else leg.bid
            )

            # Revalue before expiry using remaining time.
            exit_mid = european_option_mid(
                terminal_spot,
                leg.strike,
                leg.expiry_years - horizon_years,
                leg.implied_volatility,
                risk_free_rate,
                dividend_yield,
                leg.option_type,
            )

            # Long closes at modeled BID.
            # Short closes at modeled ASK.
            exit_price = exit_mid * (
                1 - leg.exit_spread_pct / 2
                if sign == 1
                else 1 + leg.exit_spread_pct / 2
            )

            amount = (
                leg.contracts * option_multiplier
            )

            option_pnl += (
                sign
                * amount
                * (exit_price - entry_price)
            )

            net_entry_debit += (
                sign * amount * entry_price
            )

        total_fees = (
            candidate.entry_fees_cash
            + candidate.exit_fees_cash
        )

        net_return = (
            stock_return
            + (
                option_pnl - total_fees
            ) / initial_stock_value
        )

        # Incremental cash requirement beyond the
        # already-owned stock position.
        upfront_cash = (
            max(
                0.0,
                net_entry_debit
                + candidate.entry_fees_cash,
            )
            + candidate.additional_margin_cash
        )

        results.append(
            pd.DataFrame({
                "Candidate ID": candidate.candidate_id,
                "Strategy": candidate.strategy,
                "Scenario ID": scenario_ids,
                "Hedge Horizon Days": hedge_horizon_days,
                "Net Terminal Return": net_return,
                "Upfront Cash Required": upfront_cash,
                "Underlying Terminal Price": terminal_spot,
            })
        )

    return pd.concat(
        results,
        ignore_index=True,
    )
