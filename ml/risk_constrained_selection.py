
"""Choose a hedge from matched, full-horizon net-return scenarios.

This module does NOT generate scenarios or estimate their validity.
No Hedge must have its own market-dependent return path.
Expected shortfall is a statistical estimate, not a loss guarantee.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml.path_risk_metrics import horizon_scenario_risk


REQUIRED_COLUMNS = (
    "Candidate ID",
    "Strategy",
    "Scenario ID",
    "Hedge Horizon Days",
    "Net Terminal Return",
    "Upfront Cash Required",
)


def select_risk_constrained_hedge(
    scenarios: pd.DataFrame,
    *,
    max_cvar_loss: float,
    max_upfront_cash: float,
    confidence: float = 0.95,
    min_effective_tail: int = 5,
) -> tuple[pd.DataFrame, dict | None]:
    """
    Rank hedges by expected net return, subject to constraints.

    max_cvar_loss:
        Decimal fraction of portfolio value; 0.10 means 10%.

    max_upfront_cash:
        Maximum additional cash commitment in the same currency
        as Upfront Cash Required.

    Each Scenario ID must describe the SAME market scenario
    across all candidates.

    All terminal returns must already include modeled costs.
    Do not deduct those costs again.
    """

    if not isinstance(scenarios, pd.DataFrame) or scenarios.empty:
        raise ValueError("scenarios must be a nonempty DataFrame")

    missing = set(REQUIRED_COLUMNS) - set(scenarios.columns)

    if missing:
        raise ValueError(
            f"Missing scenario columns: {sorted(missing)}"
        )

    if not np.isfinite(max_cvar_loss) or max_cvar_loss < 0:
        raise ValueError(
            "max_cvar_loss must be finite and nonnegative"
        )

    if not np.isfinite(max_upfront_cash) or max_upfront_cash < 0:
        raise ValueError(
            "max_upfront_cash must be finite and nonnegative"
        )

    if not np.isfinite(confidence) or not 0 < confidence < 1:
        raise ValueError(
            "confidence must lie strictly between 0 and 1"
        )

    if (
        isinstance(min_effective_tail, bool)
        or not isinstance(min_effective_tail, int)
        or min_effective_tail < 1
    ):
        raise ValueError(
            "min_effective_tail must be a positive integer"
        )

    df = scenarios.loc[:, REQUIRED_COLUMNS].copy()

    for column in (
        "Candidate ID",
        "Strategy",
        "Scenario ID",
    ):
        if df[column].isna().any():
            raise ValueError(f"Null {column}")

        df[column] = (
            df[column].astype(str).str.strip()
        )

        if df[column].eq("").any():
            raise ValueError(f"Blank {column}")

    for column in (
        "Hedge Horizon Days",
        "Net Terminal Return",
        "Upfront Cash Required",
    ):
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

        if not np.isfinite(
            df[column].to_numpy(dtype=float)
        ).all():
            raise ValueError(f"Invalid {column}")

    horizons = df["Hedge Horizon Days"].unique()

    if (
        len(horizons) != 1
        or horizons[0] <= 0
        or horizons[0] % 1 != 0
    ):
        raise ValueError(
            "All scenarios must share one positive integer horizon"
        )

    if (df["Upfront Cash Required"] < 0).any():
        raise ValueError(
            "Upfront Cash Required cannot be negative"
        )

    if df.duplicated([
        "Candidate ID",
        "Scenario ID",
    ]).any():
        raise ValueError(
            "Duplicate candidate/scenario pair"
        )

    counts = df.groupby(
        "Candidate ID",
        sort=False,
    )[
        ["Strategy", "Upfront Cash Required"]
    ].nunique()

    if (counts != 1).any().any():
        raise ValueError(
            "Strategy or cash requirement changes across scenarios"
        )

    no_hedge = df.loc[
        df["Strategy"].eq("No Hedge")
    ]

    if no_hedge["Candidate ID"].nunique() != 1:
        raise ValueError(
            "Supply exactly one No Hedge candidate"
        )

    if not np.allclose(
        no_hedge["Upfront Cash Required"],
        0,
        atol=1e-9,
    ):
        raise ValueError(
            "No Hedge must require zero incremental upfront cash"
        )

    reference_scenarios = set(
        no_hedge["Scenario ID"]
    )

    required_count = len(reference_scenarios)

    effective_tail = (
        required_count * (1 - confidence)
    )

    if effective_tail + 1e-12 < min_effective_tail:
        raise ValueError(
            f"Insufficient scenarios for CVaR: "
            f"{required_count} at confidence {confidence:.1%} "
            f"provides fewer than {min_effective_tail} "
            f"effective tail observations"
        )

    records = []

    for candidate_id, group in df.groupby(
        "Candidate ID",
        sort=True,
    ):
        if set(group["Scenario ID"]) != reference_scenarios:
            raise ValueError(
                f"Unmatched market scenarios for candidate "
                f"{candidate_id}"
            )

        terminal_returns = group[
            "Net Terminal Return"
        ].to_numpy(dtype=float)

        risk = horizon_scenario_risk(
            terminal_returns,
            confidence=confidence,
        )

        cash = float(
            group["Upfront Cash Required"].iloc[0]
        )

        cvar = float(
            risk["CVaR Loss"]
        )

        records.append({
            "Candidate ID": candidate_id,
            "Strategy": str(
                group["Strategy"].iloc[0]
            ),
            "Hedge Horizon Days": int(horizons[0]),
            "Scenarios": required_count,
            "Mean Net Terminal Return": float(
                np.mean(terminal_returns)
            ),
            "Probability of Negative Return": float(
                np.mean(terminal_returns < 0)
            ),
            "VaR Loss": float(risk["VaR Loss"]),
            "CVaR Loss": cvar,
            "Upfront Cash Required": cash,
            "Budget Feasible": bool(
                cash <= max_upfront_cash + 1e-9
            ),
            "CVaR Feasible": bool(
                cvar <= max_cvar_loss + 1e-12
            ),
        })

    report = pd.DataFrame(records)

    report["Feasible"] = (
        report["Budget Feasible"]
        & report["CVaR Feasible"]
    )

    report["Reason"] = np.select(
        [
            ~report["Budget Feasible"]
            & ~report["CVaR Feasible"],

            ~report["Budget Feasible"],

            ~report["CVaR Feasible"],
        ],
        [
            "Budget and CVaR",
            "Budget",
            "CVaR",
        ],
        default="Eligible",
    )

    # Conservative tie-break: prefer No Hedge if
    # expected return and risk are otherwise equal.
    report["Tie Priority"] = (
        report["Strategy"].ne("No Hedge").astype(int)
    )

    report = report.sort_values(
        [
            "Feasible",
            "Mean Net Terminal Return",
            "CVaR Loss",
            "Upfront Cash Required",
            "Tie Priority",
            "Candidate ID",
        ],
        ascending=[
            False,
            False,
            True,
            True,
            True,
            True,
        ],
        kind="stable",
    ).reset_index(drop=True)

    report = report.drop(
        columns=["Tie Priority"]
    )

    eligible = report.loc[
        report["Feasible"]
    ]

    selected = (
        None
        if eligible.empty
        else eligible.iloc[0].to_dict()
    )

    return report, selected
