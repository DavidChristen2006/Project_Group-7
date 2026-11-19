
"""Historical and scenario-based portfolio tail-risk metrics.

All returns are decimals: -0.05 means -5%.
VaR and CVaR are expressed as losses: +0.05 means 5% loss.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def empirical_tail_risk(
    returns,
    confidence=0.95,
    min_tail_count=5,
):
    """Equal-weight empirical VaR and CVaR.

    Uses fractional weighting at the tail boundary rather
    than rounding the number of tail observations upward.
    """

    if not np.isfinite(confidence) or not 0 < confidence < 1:
        raise ValueError(
            "confidence must be strictly between 0 and 1"
        )

    if not isinstance(min_tail_count, int) or min_tail_count < 1:
        raise ValueError(
            "min_tail_count must be a positive integer"
        )

    r = np.asarray(returns, dtype=float)

    if (
        r.ndim != 1
        or len(r) == 0
        or not np.isfinite(r).all()
    ):
        raise ValueError(
            "returns must be a nonempty finite 1D array"
        )

    # Positive loss means a negative investment return.
    losses = np.sort(-r)[::-1]

    n = len(losses)
    tail_mass = n * (1.0 - confidence)

    full = int(np.floor(tail_mass + 1e-12))
    fractional = max(0.0, tail_mass - full)

    weighted_loss = float(
        losses[:full].sum()
    )

    if fractional > 1e-12:
        weighted_loss += (
            fractional * float(losses[full])
        )

    cvar = weighted_loss / tail_mass

    var = float(
        np.quantile(
            losses,
            confidence,
            method="inverted_cdf",
        )
    )

    return {
        "Observations": n,
        "Confidence": float(confidence),
        "Tail Effective Observations": float(tail_mass),
        "VaR Loss": var,
        "CVaR Loss": float(cvar),
        "Low Tail Sample": bool(
            tail_mass < min_tail_count
        ),
    }


def realized_net_path_risk(
    net_path,
    net_summary,
    confidence=0.95,
):
    """Ex-post risk from an already cost-adjusted portfolio path.

    Includes the initial pre-execution wealth reference.
    The first return step may represent immediate execution
    costs rather than a full trading day.

    This is NOT a forecast of full-horizon CVaR.
    """

    column = "Net Portfolio Value"

    if column not in net_path.columns:
        raise ValueError(
            "Net Portfolio Value missing"
        )

    for key in (
        "Final P&L",
        "Final Return",
        "Maximum Drawdown",
    ):
        if key not in net_summary:
            raise ValueError(
                f"net_summary missing {key}"
            )

    values = pd.to_numeric(
        net_path[column],
        errors="coerce",
    ).to_numpy(dtype=float)

    if (
        len(values) == 0
        or not np.isfinite(values).all()
        or (values <= 0).any()
    ):
        raise ValueError(
            "Net Portfolio Value must be positive and finite"
        )

    # Recover original pre-execution capital from
    # the already net-of-costs final portfolio P&L.
    initial = float(
        values[-1] - net_summary["Final P&L"]
    )

    if not np.isfinite(initial) or initial <= 0:
        raise ValueError(
            "Invalid inferred initial portfolio value"
        )

    reconstructed_return = (
        values[-1] - initial
    ) / initial

    if not np.isclose(
        reconstructed_return,
        float(net_summary["Final Return"]),
        rtol=0,
        atol=1e-8,
    ):
        raise ValueError(
            "Net final return does not reconcile"
        )

    # Include pre-execution capital when calculating
    # drawdowns and the first change in wealth.
    wealth = np.concatenate((
        [initial],
        values,
    ))

    running_peaks = np.maximum.accumulate(
        wealth
    )

    max_drawdown = float(
        np.min(wealth / running_peaks - 1.0)
    )

    if not np.isclose(
        max_drawdown,
        float(net_summary["Maximum Drawdown"]),
        rtol=0,
        atol=1e-8,
    ):
        raise ValueError(
            "Net maximum drawdown does not reconcile"
        )

    # Net portfolio valuation-step returns.
    # Do not subtract transaction costs again.
    step_returns = (
        wealth[1:] / wealth[:-1] - 1.0
    )

    risk = empirical_tail_risk(
        step_returns,
        confidence=confidence,
    )

    return {
        **risk,
        "Initial Portfolio Value": initial,
        "Final Return": float(
            net_summary["Final Return"]
        ),
        "Maximum Drawdown": max_drawdown,
        "Return Type": (
            "realized net valuation-step changes "
            "(including entry step)"
        ),
        "Forecast CVaR": False,
    }


def horizon_scenario_risk(
    terminal_returns,
    confidence=0.95,
):
    """CVaR over equal-weight full-horizon scenarios.

    Each observation must be a completed terminal return
    for the same strategy and horizon, under a different
    scenario.

    Do not pass daily returns from one historical path.
    """

    result = empirical_tail_risk(
        terminal_returns,
        confidence=confidence,
    )

    return {
        **result,
        "Return Type": (
            "full-horizon scenario terminal returns"
        ),
        "Forecast CVaR": (
            "depends on scenario model validity"
        ),
    }
