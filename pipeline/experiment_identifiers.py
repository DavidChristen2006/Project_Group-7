
import hashlib
import json

from dataclasses import asdict
from datetime import date, datetime

import pandas as pd

from pipeline.run_identifiers import (
    transaction_cost_tag,
)


def build_historical_run_name(
    underlying_ric,
    decision_dates,
    hedge_horizons,
    shares,
    option_multiplier,
    max_spread_pct,
    max_strike_deviation_pct,
    transaction_cost_config,
    candidate_config=None,
    export_candidates=False,
    strike_discovery_mode="fixed",
    minimum_strike_pct=0.65,
    maximum_strike_pct=1.25,
    strike_step=5.0,
    expiry_mode="single",
    expiry_config=None,
):
    """
    Create one reproducible historical run name.

    Legacy fixed-strike runs without candidate export
    retain their original naming convention.

    New candidate-export runs receive a configuration
    fingerprint to distinguish different settings.
    """

    if not underlying_ric or not underlying_ric.strip():
        raise ValueError("Underlying RIC is required.")

    if strike_discovery_mode not in ("fixed", "expanded"):
        raise ValueError("Invalid discovery mode.")

    dates = sorted({
        pd.Timestamp(value).normalize()
        for value in decision_dates
    })

    horizons = sorted({
        int(value)
        for value in hedge_horizons
    })

    if not dates:
        raise ValueError("No decision dates provided.")

    if any(pd.isna(value) for value in dates):
        raise ValueError("Invalid decision date.")

    if not horizons or any(h <= 0 for h in horizons):
        raise ValueError("Invalid hedge horizons.")

    if (
        shares <= 0
        or option_multiplier <= 0
        or shares % option_multiplier != 0
    ):
        raise ValueError("Invalid position size.")

    if not 0 < max_spread_pct <= 1:
        raise ValueError("Invalid spread limit.")

    if not 0 < max_strike_deviation_pct < 1:
        raise ValueError("Invalid strike deviation.")

    if not (
        0 < minimum_strike_pct < maximum_strike_pct
    ):
        raise ValueError("Invalid strike range.")

    if not 0 < strike_step < float("inf"):
        raise ValueError("Invalid strike step.")

    cost_tag = transaction_cost_tag(
        transaction_cost_config
    )

    run_name = (
        f"{underlying_ric.replace('.', '_')}_"
        f"{dates[0]:%Y%m%d}_"
        f"{dates[-1]:%Y%m%d}_"
        f"H{'-'.join(map(str, horizons))}_"
        f"S{int(round(max_spread_pct * 10000))}_"
        f"K{int(round(max_strike_deviation_pct * 10000))}_"
        f"{cost_tag}"
    )

    if strike_discovery_mode == "expanded":
        run_name += (
            "_EXP"
            f"{int(round(minimum_strike_pct * 10000))}"
            f"-{int(round(maximum_strike_pct * 10000))}"
            f"_STEP{int(round(strike_step * 1000))}"
        )

    # Preserve historical legacy naming for
    # fixed-strike backtests without candidate export.
    if not export_candidates:
        return run_name

    if candidate_config is None:
        raise ValueError(
            "Candidate configuration is required "
            "when exporting candidates."
        )

    configuration = {
        "schema_version": 1,
        "underlying_ric": underlying_ric.upper(),
        "decision_dates": [
            value.strftime("%Y-%m-%d")
            for value in dates
        ],
        "hedge_horizons": horizons,
        "shares": int(shares),
        "option_multiplier": int(option_multiplier),
        "max_spread_pct": float(max_spread_pct),
        "max_strike_deviation_pct": float(
            max_strike_deviation_pct
        ),
        "transaction_costs": asdict(
            transaction_cost_config
        ),
        "candidate_config": asdict(
            candidate_config
        ),
        "strike_discovery_mode": strike_discovery_mode,
        "minimum_strike_pct": float(
            minimum_strike_pct
        ),
        "maximum_strike_pct": float(
            maximum_strike_pct
        ),
        "strike_step": float(strike_step),
        "fixed_strike_targets": {
            "long_put": 0.90,
            "short_call": 1.10,
            "short_put": 0.80,
        },
    }


    if expiry_mode not in ("single", "multiple"):
        raise ValueError("Invalid expiry mode.")

    # Preserve the existing single-expiry
    # configuration fingerprint.
    if expiry_mode == "multiple":
        if expiry_config is None:
            raise ValueError(
                "Expiry configuration required "
                "for multiple expiry mode."
            )

        configuration["expiry_mode"] = "multiple"

        configuration["expiry_config"] = asdict(
            expiry_config
        )

        # Explicitly distinguish the initial
        # monthly-expiry discovery implementation.
        configuration["expiry_calendar"] = (
            "monthly_v1"
        )


    serialized = json.dumps(
        configuration,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )

    fingerprint = hashlib.sha256(
        serialized.encode("utf-8")
    ).hexdigest()[:12]

    return f"{run_name}_CFG{fingerprint}"
