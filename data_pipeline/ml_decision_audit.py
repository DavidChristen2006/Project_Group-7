
"""Paired audit of saved ML hedge decisions.

Compares historical walk-forward ML models against
the recorded Historical Mean benchmark.

Only matching dates, targets and horizons are compared.
No live predictions, retraining or order execution.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from data_pipeline.ml_project_results import (
    find_walkforward_runs,
    read_walkforward_recommendations,
)


REQUIRED = {
    "Decision Date",
    "Hedge Horizon Days",
    "Target",
    "Model",
    "Selected Strategy",
    "Best Strategy",
    "Best Actual",
    "Regret",
}


def paired_regret_audit(
    records: pd.DataFrame,
    *,
    target: str,
    horizon: int,
    benchmark_model: str = "Historical Mean",
    minimum_decisions: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare each model against a benchmark on matching dates.

    Improvement = benchmark regret - model regret.

    Positive improvement means lower realized regret.
    Values retain the original experiment's stored units.
    """

    if (
        isinstance(horizon, bool)
        or not isinstance(horizon, int)
        or horizon < 1
    ):
        raise ValueError(
            "Hedge horizon must be a positive integer"
        )

    if (
        isinstance(minimum_decisions, bool)
        or not isinstance(minimum_decisions, int)
        or minimum_decisions < 1
    ):
        raise ValueError(
            "minimum_decisions must be a positive integer"
        )

    if not target.strip() or not benchmark_model.strip():
        raise ValueError(
            "Target and benchmark model must be provided"
        )

    if not isinstance(records, pd.DataFrame):
        raise ValueError(
            "Expected saved walk-forward DataFrame"
        )

    missing = REQUIRED - set(records.columns)

    if missing:
        raise ValueError(
            f"Missing walk-forward fields: {sorted(missing)}"
        )

    # -----------------------------------------------------
    # FILTER TO ONE TARGET AND ONE HEDGE HORIZON
    # -----------------------------------------------------

    frame = records.copy()

    frame["Hedge Horizon Days"] = pd.to_numeric(
        frame["Hedge Horizon Days"],
        errors="coerce",
    )

    frame = frame.loc[
        frame["Target"].eq(target)
        & frame["Hedge Horizon Days"].eq(horizon)
    ].copy()

    if frame.empty:
        raise ValueError(
            "No records for the selected target and horizon"
        )

    frame["Decision Date"] = pd.to_datetime(
        frame["Decision Date"],
        errors="coerce",
        utc=True,
    ).dt.tz_convert(None).dt.normalize()

    if frame["Decision Date"].isna().any():
        raise ValueError(
            "Invalid decision date"
        )

    # -----------------------------------------------------
    # VALIDATE THE SAVED EXPERIMENT
    # -----------------------------------------------------

    for label in (
        "Model",
        "Selected Strategy",
        "Best Strategy",
    ):
        frame[label] = (
            frame[label]
            .astype("string")
            .str.strip()
        )

        if (
            frame[label].isna().any()
            or frame[label].eq("").any()
        ):
            raise ValueError(
                f"Missing {label}"
            )

    for field in (
        "Regret",
        "Best Actual",
    ):
        frame[field] = pd.to_numeric(
            frame[field],
            errors="coerce",
        )

        if not np.isfinite(
            frame[field].to_numpy(dtype=float)
        ).all():
            raise ValueError(
                f"Nonfinite {field}"
            )

    if (frame["Regret"] < -1e-8).any():
        raise ValueError(
            "Negative realized regret is inconsistent"
        )

    if frame.duplicated(
        ["Decision Date", "Model"]
    ).any():
        raise ValueError(
            "Duplicate decision date/model records"
        )

    # All models evaluated on one date must have
    # the same hindsight-best strategy and outcome.

    by_date = frame.groupby(
        "Decision Date",
        sort=False,
    )

    if (
        by_date["Best Strategy"].nunique() > 1
    ).any():
        raise ValueError(
            "Inconsistent hindsight-best strategy "
            "on matching dates"
        )

    if (
        by_date["Best Actual"].agg(
            lambda values: values.max() - values.min()
        ) > 1e-8
    ).any():
        raise ValueError(
            "Inconsistent hindsight-best outcome "
            "on matching dates"
        )

    # -----------------------------------------------------
    # PREPARE THE RECORDED BENCHMARK
    # -----------------------------------------------------

    benchmark = (
        frame.loc[
            frame["Model"].eq(benchmark_model),
            [
                "Decision Date",
                "Regret",
                "Selected Strategy",
            ],
        ]
        .rename(columns={
            "Regret": "Benchmark regret",
            "Selected Strategy":
                "Benchmark selected strategy",
        })
    )

    if benchmark.empty:
        raise ValueError(
            f"Benchmark model {benchmark_model!r} "
            "not present in this run"
        )

    benchmark_count = len(benchmark)

    summaries = []
    paired_records = []

    # -----------------------------------------------------
    # EVALUATE EACH MODEL ON MATCHING DATES
    # -----------------------------------------------------

    for model_name, model_data in frame.groupby(
        "Model",
        sort=True,
    ):

        paired = model_data.merge(
            benchmark,
            on="Decision Date",
            validate="one_to_one",
        )

        if paired.empty:
            continue

        paired["Regret improvement"] = (
            paired["Benchmark regret"]
            - paired["Regret"]
        )

        paired["Model"] = str(model_name)

        improvement = paired[
            "Regret improvement"
        ]

        n = len(paired)

        summaries.append({
            "Model": str(model_name),

            "Paired decisions": n,

            "Benchmark decisions":
                benchmark_count,

            "Coverage":
                n / benchmark_count,

            "Mean regret": float(
                paired["Regret"].mean()
            ),

            "Benchmark mean regret on matched dates":
                float(
                    paired["Benchmark regret"].mean()
                ),

            "Mean regret improvement":
                float(improvement.mean()),

            "Median regret improvement":
                float(improvement.median()),

            "Improved dates": int(
                (improvement > 1e-12).sum()
            ),

            "Worsened dates": int(
                (improvement < -1e-12).sum()
            ),

            "Tied dates": int(
                (improvement.abs() <= 1e-12).sum()
            ),

            "Evidence": (
                "Benchmark"
                if str(model_name) == benchmark_model
                else (
                    "Limited historical decisions"
                    if n < minimum_decisions
                    else "Descriptive comparison only"
                )
            ),
        })

        paired_records.append(
            paired[
                [
                    "Decision Date",
                    "Model",
                    "Selected Strategy",
                    "Benchmark selected strategy",
                    "Regret",
                    "Benchmark regret",
                    "Regret improvement",
                ]
            ]
        )

    if not summaries:
        raise ValueError(
            "No matching benchmark/model decisions"
        )

    summary = (
        pd.DataFrame(summaries)
        .sort_values("Model")
        .reset_index(drop=True)
    )

    details = (
        pd.concat(
            paired_records,
            ignore_index=True,
        )
        .sort_values(
            ["Decision Date", "Model"]
        )
        .reset_index(drop=True)
    )

    return summary, details


def render_ml_decision_audit(
    project_root: str | Path,
) -> None:
    """User-friendly historical ML decision scorecard."""

    import streamlit as st

    root = Path(project_root)

    st.subheader(
        "ML hedge decisions — historical evidence"
    )

    st.caption(
        "Saved walk-forward decisions only. "
        "ML models are compared with Historical Mean "
        "on the same dates, prediction target and horizon. "
        "No retraining or live recommendations."
    )

    # -----------------------------------------------------
    # DISCOVER SAVED EXPERIMENTS
    # -----------------------------------------------------

    runs = find_walkforward_runs(root)

    if not runs:
        st.info(
            "No saved walk-forward recommendations "
            "found in data/processed/ml_evaluations."
        )
        return

    run = st.selectbox(
        "Historical experiment",
        runs,
        format_func=lambda path: (
            str(path.relative_to(root))
            if path.is_relative_to(root)
            else str(path)
        ),
        key="ml_decision_audit_run",
    )

    try:

        records = read_walkforward_recommendations(
            run
        )

        targets = sorted(
            records["Target"]
            .dropna()
            .astype(str)
            .unique()
        )

        if not targets:
            raise ValueError(
                "No recorded outcome targets"
            )

        target = st.selectbox(
            "Outcome target",
            targets,
            key="ml_decision_audit_target",
        )

        possible_horizons = pd.to_numeric(
            records.loc[
                records["Target"].eq(target),
                "Hedge Horizon Days",
            ],
            errors="coerce",
        ).dropna()

        horizons = sorted({
            int(value)
            for value in possible_horizons
            if (
                float(value).is_integer()
                and value > 0
            )
        })

        if not horizons:
            raise ValueError(
                "No valid recorded hedge horizons"
            )

        horizon = st.selectbox(
            "Recorded hedge horizon (as stored)",
            horizons,
            key="ml_decision_audit_horizon",
        )

        subset = records.loc[
            records["Target"].eq(target)
            & pd.to_numeric(
                records["Hedge Horizon Days"],
                errors="coerce",
            ).eq(horizon)
        ]

        models = set(
            subset["Model"].astype(str)
        )

        if "Historical Mean" not in models:
            st.warning(
                "The Historical Mean benchmark is absent. "
                "A paired comparison cannot be calculated."
            )
            return

        summary, details = paired_regret_audit(
            records,
            target=target,
            horizon=int(horizon),
        )

    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        ImportError,
    ) as exc:

        st.warning(
            f"Historical ML decision audit unavailable: {exc}"
        )
        return

    # -----------------------------------------------------
    # DISPLAY READABLE MODEL COMPARISON
    # -----------------------------------------------------

    ml_models = summary.loc[
        summary["Model"].ne("Historical Mean")
    ]

    if ml_models.empty:

        st.info(
            "Only the Historical Mean benchmark "
            "is recorded for this experiment."
        )

    else:

        display_columns = [
            "Model",
            "Paired decisions",
            "Coverage",
            "Mean regret",
            "Benchmark mean regret on matched dates",
            "Mean regret improvement",
            "Improved dates",
            "Worsened dates",
            "Tied dates",
            "Evidence",
        ]

        # A model's paired regret change is an interpretable visual score;
        # display the same recorded values as the table underneath.
        from dashboard_visuals import model_ranking_chart
        st.plotly_chart(
            model_ranking_chart(
                summary, metric="Mean regret improvement",
                title="Matched-date regret improvement vs Historical Mean",
            ),
            use_container_width=True,
            key="ml_decision_regret_chart",
        )

        st.dataframe(
            summary[
                display_columns
            ].style.format({
                "Coverage": "{:.0%}",
                "Mean regret": "{:.6f}",
                "Benchmark mean regret on matched dates":
                    "{:.6f}",
                "Mean regret improvement": "{:+.6f}",
            }),
            hide_index=True,
            use_container_width=True,
        )

        largest_sample = int(
            ml_models["Paired decisions"].max()
        )

        if largest_sample < 20:

            st.warning(
                f"Only up to {largest_sample} matching "
                "historical decision dates. "
                "There is insufficient evidence to "
                "declare a consistently superior model."
            )

        else:

            st.info(
                "Differences are descriptive, not proof "
                "of statistical superiority. "
                "Chronological holdouts and hedging costs "
                "still need to be assessed."
            )

    # -----------------------------------------------------
    # EXPLAIN HOW TO INTERPRET THE RESULTS
    # -----------------------------------------------------

    st.caption(
        "Positive regret improvement means the ML model "
        "made a better historical hedge choice than the "
        "Historical Mean benchmark on the same date. "
        "Units are retained exactly as saved by the "
        "original experiment. Historical Mean is not "
        "the same as No Hedge. These records cannot "
        "generate a current trading recommendation."
    )

    with st.expander(
        "Inspect paired historical decisions"
    ):

        st.dataframe(
            details,
            hide_index=True,
            use_container_width=True,
        )

    # -----------------------------------------------------
    # RECORDED OUTCOME AND COST-READINESS AUDIT
    # -----------------------------------------------------

    from data_pipeline.ml_cost_sensitivity import (
        render_ml_cost_sensitivity,
    )

    render_ml_cost_sensitivity(
        records,
        target=target,
        horizon=int(horizon),
    )

    # -----------------------------------------------------
    # SAVED WALK-FORWARD INTEGRITY AUDIT
    # -----------------------------------------------------

    from data_pipeline.ml_walkforward_integrity import (
        render_walkforward_integrity,
    )

    render_walkforward_integrity(
        run,
        target=target,
        horizon=int(horizon),
    )