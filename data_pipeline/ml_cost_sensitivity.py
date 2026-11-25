
"""Paired historical ML policy-outcome and cost-readiness audit.

Uses saved walk-forward recommendations.

Optionally supports a verified, per-decision policy cost ledger.
Never invents missing costs, changes outcome units, or treats
historical research results as live trading recommendations.

A net hindsight oracle cannot be reconstructed from selected
strategy costs alone: costs for all candidates are required.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


POLICY_KEY = [
    "Decision Date",
    "Hedge Horizon Days",
    "Target",
    "Model",
]

COST_KEY = POLICY_KEY + [
    "Selected Strategy",
]


def _prepare_records(
    records: pd.DataFrame,
    target: str,
    horizon: int,
) -> pd.DataFrame:
    """Validate and filter one historical target and horizon."""

    if target != "Hedge Improvement":
        raise ValueError(
            "This outcome comparison is restricted "
            "to Hedge Improvement"
        )

    if (
        isinstance(horizon, bool)
        or not isinstance(horizon, int)
        or horizon < 1
    ):
        raise ValueError(
            "Horizon must be a positive integer"
        )

    required = set(
        COST_KEY + [
            "Selected Actual",
            "Best Actual",
            "Regret",
            "Best Strategy",
        ]
    )

    if (
        not isinstance(records, pd.DataFrame)
        or not required.issubset(records.columns)
    ):
        raise ValueError(
            "Missing saved recommendation fields: "
            f"{sorted(required - set(getattr(records, 'columns', [])))}"
        )

    data = records.copy()

    data["Hedge Horizon Days"] = pd.to_numeric(
        data["Hedge Horizon Days"],
        errors="coerce",
    )

    data = data.loc[
        data["Target"].eq(target)
        & data["Hedge Horizon Days"].eq(horizon)
    ].copy()

    if data.empty:
        raise ValueError(
            "No recommendations for the target and horizon"
        )

    data["Decision Date"] = (
        pd.to_datetime(
            data["Decision Date"],
            errors="coerce",
            utc=True,
        )
        .dt.tz_convert(None)
        .dt.normalize()
    )

    if data["Decision Date"].isna().any():
        raise ValueError(
            "Invalid decision date"
        )

    for column in (
        "Model",
        "Selected Strategy",
        "Best Strategy",
    ):
        data[column] = (
            data[column]
            .astype("string")
            .str.strip()
        )

        if (
            data[column].isna().any()
            or data[column].eq("").any()
        ):
            raise ValueError(
                f"Missing {column}"
            )

    for column in (
        "Selected Actual",
        "Best Actual",
        "Regret",
    ):
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        )

        if not np.isfinite(
            data[column].to_numpy(dtype=float)
        ).all():
            raise ValueError(
                f"Nonfinite {column}"
            )

    if (data["Regret"] < -1e-8).any():
        raise ValueError(
            "Negative saved regret"
        )

    if data.duplicated(POLICY_KEY).any():
        raise ValueError(
            "Duplicate model decision"
        )

    groups = data.groupby(
        "Decision Date"
    )

    if (
        groups["Best Strategy"].nunique() != 1
    ).any():
        raise ValueError(
            "Inconsistent hindsight-best strategy"
        )

    differences = (
        groups["Best Actual"].max()
        - groups["Best Actual"].min()
    )

    if (differences > 1e-8).any():
        raise ValueError(
            "Inconsistent hindsight-best outcome"
        )

    return data


def compare_recorded_policy_outcomes(
    recommendations: pd.DataFrame,
    *,
    target: str,
    horizon: int,
    benchmark: str = "Historical Mean",
    verified_costs: pd.DataFrame | None = None,
    outcomes_exclude_costs: bool = False,
    cost_units: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare policies on identical historical decision dates.

    Positive difference:
        The model's recorded selected outcome is higher
        than the benchmark's outcome.

    Optional net comparison:
        Requires explicitly verified pre-cost outcomes
        and an exact, complete cost ledger in matching units.

    This does not retrain models or prove their original
    walk-forward training pipeline was leakage-free.
    """

    data = _prepare_records(
        recommendations,
        target,
        horizon,
    )

    if not benchmark.strip():
        raise ValueError(
            "Benchmark cannot be blank"
        )

    base = data.loc[
        data["Model"].eq(benchmark),
        [
            "Decision Date",
            "Selected Actual",
        ],
    ]

    if base.empty:
        raise ValueError(
            f"Missing benchmark: {benchmark}"
        )

    benchmark_dates = len(base)

    # -----------------------------------------------------
    # OPTIONAL VERIFIED COST LEDGER
    # -----------------------------------------------------

    if verified_costs is not None:

        if (
            not outcomes_exclude_costs
            or cost_units not in {
                "fractional_return",
                "percentage_points",
            }
        ):
            raise ValueError(
                "Verified pre-cost provenance and explicit "
                "matching return units are required"
            )

        required = set(
            COST_KEY + ["Cost"]
        )

        if (
            not isinstance(verified_costs, pd.DataFrame)
            or not required.issubset(
                verified_costs.columns
            )
        ):
            raise ValueError(
                "Missing verified cost ledger fields"
            )

        ledger = verified_costs.copy()

        ledger["Hedge Horizon Days"] = pd.to_numeric(
            ledger["Hedge Horizon Days"],
            errors="coerce",
        )

        ledger = ledger.loc[
            ledger["Target"].eq(target)
            & ledger["Hedge Horizon Days"].eq(horizon)
        ].copy()

        ledger["Decision Date"] = (
            pd.to_datetime(
                ledger["Decision Date"],
                errors="coerce",
                utc=True,
            )
            .dt.tz_convert(None)
            .dt.normalize()
        )

        for column in (
            "Model",
            "Selected Strategy",
        ):
            ledger[column] = (
                ledger[column]
                .astype("string")
                .str.strip()
            )

        ledger["Cost"] = pd.to_numeric(
            ledger["Cost"],
            errors="coerce",
        )

        if (
            ledger["Decision Date"].isna().any()
            or ledger[
                [
                    "Model",
                    "Selected Strategy",
                ]
            ].isna().any().any()
        ):
            raise ValueError(
                "Missing cost ledger keys"
            )

        if (
            not np.isfinite(
                ledger["Cost"].to_numpy(dtype=float)
            ).all()
            or (ledger["Cost"] < 0).any()
        ):
            raise ValueError(
                "Invalid cost ledger values"
            )

        if ledger.duplicated(COST_KEY).any():
            raise ValueError(
                "Duplicate cost ledger policy"
            )

        # An exact match is required. In particular,
        # a cost for the wrong strategy is NOT accepted.

        merged = data.merge(
            ledger[COST_KEY + ["Cost"]],
            on=COST_KEY,
            how="outer",
            indicator=True,
            validate="one_to_one",
        )

        if not merged["_merge"].eq("both").all():
            raise ValueError(
                "Cost ledger does not exactly cover "
                "saved decisions and strategies"
            )

        data = merged.drop(
            columns="_merge"
        )

        data["Evaluated outcome"] = (
            data["Selected Actual"]
            - data["Cost"]
        )

        metric = "Net selected-outcome difference"

    else:

        # Preserve the saved experimental outcome
        # when costs have not been verified.

        data["Cost"] = np.nan

        data["Evaluated outcome"] = (
            data["Selected Actual"]
        )

        metric = (
            "Recorded selected-outcome difference"
        )

    # -----------------------------------------------------
    # MATCH EACH MODEL WITH BENCHMARK ON THE SAME DATES
    # -----------------------------------------------------

    base = (
        data.loc[
            data["Model"].eq(benchmark),
            [
                "Decision Date",
                "Evaluated outcome",
            ],
        ]
        .rename(columns={
            "Evaluated outcome": "Benchmark outcome"
        })
    )

    paired = data.merge(
        base,
        on="Decision Date",
        how="inner",
        validate="many_to_one",
    )

    paired[metric] = (
        paired["Evaluated outcome"]
        - paired["Benchmark outcome"]
    )

    paired = (
        paired.sort_values(
            [
                "Model",
                "Decision Date",
            ],
            kind="stable",
        )
        .reset_index(drop=True)
    )

    # -----------------------------------------------------
    # MODEL-LEVEL SUMMARY
    # -----------------------------------------------------

    summary = (
        paired.groupby(
            "Model",
            as_index=False,
        )
        .agg(**{
            "Matched decisions": (
                "Decision Date",
                "nunique",
            ),
            "Mean selected outcome": (
                "Evaluated outcome",
                "mean",
            ),
            "Mean benchmark outcome on same dates": (
                "Benchmark outcome",
                "mean",
            ),
            "Mean difference": (
                metric,
                "mean",
            ),
            "Improved dates": (
                metric,
                lambda x: int(
                    (x > 1e-12).sum()
                ),
            ),
            "Worsened dates": (
                metric,
                lambda x: int(
                    (x < -1e-12).sum()
                ),
            ),
            "Tied dates": (
                metric,
                lambda x: int(
                    (x.abs() <= 1e-12).sum()
                ),
            ),
        })
    )

    summary["Benchmark date coverage"] = (
        summary["Matched decisions"]
        / benchmark_dates
    )

    summary["Evaluation"] = (
        "Verified net of recorded costs"
        if verified_costs is not None
        else "Stored outcomes; costs unverified"
    )

    return (
        summary.sort_values("Model").reset_index(drop=True),
        paired,
    )


def render_ml_cost_sensitivity(
    records: pd.DataFrame,
    *,
    target: str,
    horizon: int,
) -> None:
    """Optional Streamlit panel for the selected saved ML run."""

    import streamlit as st

    with st.expander(
        "Recorded outcomes and cost readiness",
        expanded=False,
    ):

        if target != "Hedge Improvement":

            st.info(
                "Only Hedge Improvement is eligible "
                "for this comparison. Drawdown and other "
                "risk targets are not investment returns."
            )

            return

        try:

            summary, _ = compare_recorded_policy_outcomes(
                records,
                target=target,
                horizon=int(horizon),
            )

        except (
            ValueError,
            TypeError,
            KeyError,
        ) as exc:

            st.warning(
                f"Outcome comparison unavailable: {exc}"
            )

            return

        st.caption(
            "Compares models with Historical Mean only "
            "on matching decision dates. Positive "
            "difference means a higher recorded selected "
            "outcome. Units are unchanged from the "
            "saved experiment."
        )

        st.dataframe(
            summary.style.format({
                "Mean selected outcome": "{:+.6f}",
                "Mean benchmark outcome on same dates":
                    "{:+.6f}",
                "Mean difference": "{:+.6f}",
                "Benchmark date coverage": "{:.1%}",
            }),
            hide_index=True,
            use_container_width=True,
        )

        if int(
            summary["Matched decisions"].max()
        ) < 20:

            st.warning(
                "Fewer than 20 matching historical "
                "decisions. These differences cannot "
                "establish reliable model superiority."
            )

        st.info(
            "Net-of-cost evidence is not yet available "
            "from saved recommendations. A cost comparison "
            "requires a verified policy/date-specific "
            "ledger in the SAME units, and confirmation "
            "that Selected Actual is pre-cost. "
            "Recomputing hindsight regret after costs "
            "also requires costs for every available "
            "strategy—not only the model-selected strategy. "
            "Historical Mean is not the same as No Hedge."
        )
