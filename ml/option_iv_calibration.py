
"""Calibrate European Black-Scholes IV from historical option quotes.

This is a model-implied IV estimate, not a directly observed
LSEG implied-volatility field.

US equity options can contain early-exercise value.
If the observed midpoint cannot be reconciled with the
European model, fail explicitly instead of silently using
an unrelated volatility estimate.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from ml.forward_scenario_adapter import european_option_mid


def calibrate_entry_mid_iv(
    *,
    spot: float,
    strike: float,
    expiry_years: float,
    bid: float,
    ask: float,
    option_type: str,
    risk_free_rate: float,
    dividend_yield: float,
    minimum_volatility: float = 1e-5,
    maximum_volatility: float = 5.0,
) -> float:
    """Invert the entry quote midpoint to obtain annualized volatility."""

    inputs = (
        spot,
        strike,
        expiry_years,
        bid,
        ask,
        risk_free_rate,
        dividend_yield,
        minimum_volatility,
        maximum_volatility,
    )

    if not np.isfinite(inputs).all():
        raise ValueError(
            "IV calibration requires finite numeric inputs"
        )

    if (
        spot <= 0
        or strike <= 0
        or expiry_years <= 0
        or bid < 0
        or ask <= 0
        or ask < bid
        or option_type not in ("put", "call")
        or minimum_volatility <= 0
        or maximum_volatility <= minimum_volatility
    ):
        raise ValueError(
            "Invalid market quote or IV calibration parameters"
        )

    midpoint = (float(bid) + float(ask)) / 2.0

    def model_price(sigma: float) -> float:
        result = european_option_mid(
            np.array([float(spot)]),
            strike=float(strike),
            remaining_years=float(expiry_years),
            volatility=float(sigma),
            rate=float(risk_free_rate),
            dividend_yield=float(dividend_yield),
            option_type=option_type,
        )

        return float(result[0])

    lower = model_price(minimum_volatility)
    upper = model_price(maximum_volatility)

    tolerance = 1e-8 * max(
        1.0,
        spot,
        strike,
    )

    if midpoint <= lower + tolerance:
        raise ValueError(
            f"Option midpoint {midpoint:.6f} is at/below "
            f"the European model minimum {lower:.6f}. "
            "Check early-exercise premium, quotes, "
            "rates and dividends."
        )

    if midpoint >= upper - tolerance:
        raise ValueError(
            f"Option midpoint {midpoint:.6f} exceeds "
            f"the model value at maximum IV "
            f"({upper:.6f}). Check inputs and "
            "volatility assumptions."
        )

    implied_volatility = brentq(
        lambda sigma: model_price(sigma) - midpoint,
        minimum_volatility,
        maximum_volatility,
        xtol=1e-12,
        rtol=1e-12,
        maxiter=200,
    )

    return float(implied_volatility)
