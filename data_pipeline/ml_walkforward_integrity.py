
"""Audit saved walk-forward recommendations and predictions.

This checks consistency of saved experiment artifacts.
It does NOT certify that the original training was leakage-free.

No retraining, LSEG requests, or live recommendations.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


RECOMMENDATION_FIELDS = {
    "Decision Date",
    "Hedge Horizon Days",
    "Target",
    "Model",
    "Selected Strategy",
    "Best Strategy",
    "Selected Actual",
    "Best Actual",
    "Regret",
}

PREDICTION_FIELDS = {
    "Decision Date",
    "Outcome Date",
    "Hedge Horizon Days",
    "Target",
    "Model",
    "Strategy",
    "Actual",
}


def _filtered(
    frame: pd.DataFrame,
    required: set[str],
    target: str,
    horizon: int,
) -> pd.DataFrame:

    if not isinstance(frame, pd.DataFrame):
        raise ValueError("Expected a DataFrame")

    missing = required.difference(frame.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    data = frame.copy()

    h = pd.to_numeric(
        data["Hedge Horizon Days"],
        errors="coerce",
    )

    data = data.loc[
        data["Target"].eq(target)
        & h.eq(horizon)
    ].copy()

    if data.empty:
        raise ValueError(
            "No records for this target and horizon"
        )

    for col in ("Decision Date", "Outcome Date"):

        if col in data:

            data[col] = (
                pd.to_datetime(
                    data[col],
                    errors="coerce",
                    utc=True,
                )
                .dt.tz_convert(None)
                .dt.normalize()
            )

            if data[col].isna().any():
                raise ValueError(
                    f"Invalid {col}"
                )

    for col in (
        "Model",
        "Selected Strategy",
        "Best Strategy",
        "Strategy",
    ):

        if col in data:

            data[col] = (
                data[col]
                .astype("string")
                .str.strip()
            )

            if (
                data[col].isna().any()
                or data[col].eq("").any()
            ):
                raise ValueError(
                    f"Missing {col}"
                )

    for col in (
        "Actual",
        "Selected Actual",
        "Best Actual",
        "Regret",
    ):

        if col in data:

            data[col] = pd.to_numeric(
                data[col],
                errors="coerce",
            )

            if not np.isfinite(
                data[col].to_numpy(dtype=float)
            ).all():
                raise ValueError(
                    f"Invalid numeric {col}"
                )

    return data


def audit_walkforward_artifacts(
    recommendations: pd.DataFrame,
    predictions: pd.DataFrame,
    folds: pd.DataFrame | None = None,
    *,
    target: str = "Hedge Improvement",
    horizon: int = 21,
    tolerance: float = 1e-8,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    list[str],
]:
    """Reconcile recommendations with recorded predictions.

    A higher Actual is assumed better for Hedge Improvement.

    Returns:
        summary
        decision_details
        fold_summary
        warnings
    """

    if target != "Hedge Improvement":
        raise ValueError(
            "Only Hedge Improvement is supported"
        )

    if (
        isinstance(horizon, bool)
        or not isinstance(horizon, int)
        or horizon < 1
    ):
        raise ValueError(
            "Horizon must be a positive integer"
        )

    if (
        not np.isfinite(tolerance)
        or tolerance < 0
    ):
        raise ValueError(
            "Invalid comparison tolerance"
        )

    r = _filtered(
        recommendations,
        RECOMMENDATION_FIELDS,
        target,
        horizon,
    )

    p = _filtered(
        predictions,
        PREDICTION_FIELDS,
        target,
        horizon,
    )

    key = [
        "Decision Date",
        "Model",
    ]

    prediction_key = key + ["Strategy"]

    # -----------------------------------------------------
    # CHECK UNIQUE DECISIONS AND VALID OUTCOME DATES
    # -----------------------------------------------------

    if r.duplicated(key).any():
        raise ValueError(
            "Duplicate recommendation decision/model"
        )

    if p.duplicated(prediction_key).any():
        raise ValueError(
            "Duplicate prediction decision/model/strategy"
        )

    if (
        p["Outcome Date"]
        <= p["Decision Date"]
    ).any():
        raise ValueError(
            "Prediction outcome must be after decision date"
        )

    if (r["Regret"] < -tolerance).any():
        raise ValueError(
            "Negative stored regret"
        )

    warnings: list[str] = []

    # -----------------------------------------------------
    # AUDIT REALIZED VALUES ACROSS DIFFERENT MODELS
    # -----------------------------------------------------

    actual_ranges = (
        p.groupby(
            ["Decision Date", "Strategy"]
        )["Actual"]
        .agg(
            lambda values: (
                values.max() - values.min()
            )
        )
    )

    if (actual_ranges > tolerance).any():
        warnings.append(
            "Actual outcomes differ across models "
            "for the same decision and strategy."
        )

    candidate_sets = (
        p.groupby(key)["Strategy"]
        .agg(lambda values: frozenset(values))
    )

    for day, group in candidate_sets.groupby(level=0):

        if len(set(group)) != 1:
            warnings.append(
                "Different model candidate sets on "
                f"{pd.Timestamp(day).date()}."
            )

    # -----------------------------------------------------
    # RECOMPUTE HINDSIGHT-BEST OUTCOMES
    # -----------------------------------------------------

    best = (
        p.groupby(key)["Actual"]
        .max()
        .rename("Recomputed best actual")
        .reset_index()
    )

    chosen = (
        p[
            key + [
                "Strategy",
                "Actual",
                "Outcome Date",
            ]
        ]
        .rename(
            columns={
                "Strategy": "Selected Strategy",
                "Actual": "Predicted-row actual",
            }
        )
    )

    details = r.merge(
        chosen,
        on=key + ["Selected Strategy"],
        how="left",
        indicator="_candidate_match",
        validate="one_to_one",
    )

    details = details.merge(
        best,
        on=key,
        how="left",
        validate="many_to_one",
    )

    candidate_lookup = (
        p.set_index(prediction_key)["Actual"]
        .to_dict()
    )

    # -----------------------------------------------------
    # RECONCILE EVERY RECORDED DECISION
    # -----------------------------------------------------

    statuses = []

    for _, row in details.iterrows():

        issues = []

        if row["_candidate_match"] != "both":

            issues.append(
                "Selected strategy missing from predictions"
            )

        else:

            if abs(
                row["Selected Actual"]
                - row["Predicted-row actual"]
            ) > tolerance:

                issues.append(
                    "Selected actual mismatch"
                )

            if abs(
                row["Best Actual"]
                - row["Recomputed best actual"]
            ) > tolerance:

                issues.append(
                    "Best actual mismatch"
                )

            expected_regret = max(
                0.0,
                float(
                    row["Best Actual"]
                    - row["Selected Actual"]
                ),
            )

            if abs(
                row["Regret"] - expected_regret
            ) > tolerance:

                issues.append(
                    "Saved regret mismatch"
                )

            candidate_best = candidate_lookup.get(
                (
                    row["Decision Date"],
                    row["Model"],
                    row["Best Strategy"],
                )
            )

            if (
                candidate_best is None
                or abs(
                    candidate_best
                    - row["Recomputed best actual"]
                ) > tolerance
            ):
                issues.append(
                    "Best strategy not an optimal candidate"
                )

        statuses.append(
            "Matched"
            if not issues
            else "; ".join(issues)
        )

    details["Audit status"] = statuses

    details = details.drop(
        columns=["_candidate_match"]
    )

    # -----------------------------------------------------
    # MODEL-LEVEL RECONCILIATION
    # -----------------------------------------------------

    summary = (
        details.groupby(
            "Model",
            as_index=False,
        )
        .agg(**{
            "Decisions": (
                "Decision Date",
                "nunique",
            ),
            "Matched": (
                "Audit status",
                lambda values: int(
                    values.eq("Matched").sum()
                ),
            ),
            "Flagged": (
                "Audit status",
                lambda values: int(
                    values.ne("Matched").sum()
                ),
            ),
        })
    )

    summary["Record match rate"] = (
        summary["Matched"]
        / summary["Decisions"]
    )

    # -----------------------------------------------------
    # READ AGGREGATE TRAINING-FOLD EVIDENCE
    # -----------------------------------------------------

    coverage = pd.DataFrame()

    if folds is None:

        warnings.append(
            "Training-fold summary file unavailable: "
            "training chronology not assessable."
        )

    else:

        fold_fields = {
            "Test Date",
            "Training Observations",
            "Training Decision Dates",
            "Test Observations",
            "Unfinished Outcomes",
        }

        missing = fold_fields.difference(
            folds.columns
        )

        if missing:
            raise ValueError(
                "Missing fold summary fields: "
                f"{sorted(missing)}"
            )

        coverage = folds.copy()

        coverage["Test Date"] = (
            pd.to_datetime(
                coverage["Test Date"],
                errors="coerce",
                utc=True,
            )
            .dt.tz_convert(None)
            .dt.normalize()
        )

        if coverage["Test Date"].isna().any():
            raise ValueError(
                "Invalid fold test date"
            )

        for col in fold_fields - {"Test Date"}:

            coverage[col] = pd.to_numeric(
                coverage[col],
                errors="coerce",
            )

            if (
                not np.isfinite(
                    coverage[col]
                    .to_numpy(dtype=float)
                ).all()
                or (coverage[col] < 0).any()
            ):
                raise ValueError(
                    f"Invalid fold count: {col}"
                )

        recorded_dates = set(
            r["Decision Date"]
        )

        fold_dates = set(
            coverage["Test Date"]
        )

        if not recorded_dates.issubset(
            fold_dates
        ):
            warnings.append(
                "Some recommendation dates have no "
                "matching fold Test Date. Verify whether "
                "fold Test Date identifies each decision "
                "or a larger evaluation period."
            )

        warnings.append(
            "Fold summaries provide aggregate counts only. "
            "They cannot prove each training label was "
            "available before its test decision."
        )

    if details["Audit status"].ne(
        "Matched"
    ).any():

        warnings.append(
            "Saved recommendations and candidate "
            "prediction records disagree. "
            "Inspect the flagged decisions."
        )

    return (
        summary,
        details,
        coverage,
        list(dict.fromkeys(warnings)),
    )


def render_walkforward_integrity(
    run_dir: str | Path,
    *,
    target: str,
    horizon: int,
) -> None:
    """Show the saved-artifact audit in Historical ML Research."""

    import streamlit as st

    with st.expander(
        "Walk-forward integrity and leakage audit",
        expanded=False,
    ):

        run = Path(run_dir)

        prediction_path = (
            run / "walk_forward_predictions.parquet"
        )

        fold_path = (
            run / "walk_forward_folds.parquet"
        )

        try:

            if not prediction_path.is_file():
                raise FileNotFoundError(
                    f"Missing {prediction_path.name}"
                )

            from data_pipeline.ml_project_results import (
                read_walkforward_recommendations,
            )

            recommendations = (
                read_walkforward_recommendations(run)
            )

            predictions = pd.read_parquet(
                prediction_path
            )

            folds = (
                pd.read_parquet(fold_path)
                if fold_path.is_file()
                else None
            )

            (
                summary,
                details,
                coverage,
                warnings,
            ) = audit_walkforward_artifacts(
                recommendations,
                predictions,
                folds,
                target=target,
                horizon=int(horizon),
            )

        except (
            OSError,
            ValueError,
            KeyError,
            ImportError,
        ) as exc:

            st.warning(
                f"Walk-forward integrity audit unavailable: "
                f"{exc}"
            )

            return

        st.dataframe(
            summary.style.format({
                "Record match rate": "{:.1%}",
            }),
            hide_index=True,
            use_container_width=True,
        )

        # Explain excluded observations without incorrectly
        # presenting their existence as evidence of leakage.

        if (
            not coverage.empty
            and "Unfinished Outcomes" in coverage.columns
        ):

            excluded_count = int(
                pd.to_numeric(
                    coverage["Unfinished Outcomes"],
                    errors="raise",
                ).sum()
            )

            if excluded_count > 0:

                st.info(
                    f"{excluded_count:,} unfinished outcome "
                    "records were reported as excluded across "
                    "the historical training folds. "
                    "The inspected get_walk_forward_split() "
                    "function explicitly excludes them. "
                    "This is a count across folds, not "
                    "necessarily unique observations."
                )

        for message in warnings:
            st.warning(message)

        if not warnings:
            st.info(
                "Saved records agree on the checked fields. "
                "This does not establish leakage-free training."
            )

        with st.expander(
            "Inspect decision-level reconciliation"
        ):

            st.dataframe(
                details,
                hide_index=True,
                use_container_width=True,
            )

        if not coverage.empty:

            with st.expander(
                "Inspect recorded training-fold counts"
            ):

                st.dataframe(
                    coverage,
                    hide_index=True,
                    use_container_width=True,
                )

        st.caption(
            "This audit reconciles stored results and fold "
            "summary counts. It cannot determine the exact "
            "training-sample label availability dates. "
            "A full look-ahead audit requires the original "
            "training code and feature-generation pipeline."
        )
