
"""Compare historical hedge-selection policies on identical decision dates.

No Hedge = zero return improvement for the Hedge Improvement target only.

Rule-based outcomes are excluded unless their stored units are confirmed.

Exploratory research, not a live hedge recommendation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from data_pipeline.ml_project_results import (
    find_walkforward_runs,
    read_walkforward_recommendations,
)


def _clean_dates(
    frame: pd.DataFrame,
) -> pd.Series:

    """Handle the internal clean dates step.

    Args:
        frame: Input to the _clean_dates operation.

    Returns:
        Computed result or status for the caller.
    """
    dates = pd.to_datetime(
        frame["Decision Date"],
        errors="coerce",
        utc=True,
    )

    if dates.isna().any():
        raise ValueError(
            "Invalid or missing decision dates"
        )

    return (
        dates.dt.tz_convert(None)
        .dt.normalize()
    )


def _clean_values(
    frame: pd.DataFrame,
    column: str,
) -> pd.Series:

    """Handle the internal clean values step.

    Args:
        frame: Input to the _clean_values operation.
        column: Input to the _clean_values operation.

    Returns:
        Computed result or status for the caller.
    """
    values = pd.to_numeric(
        frame[column],
        errors="coerce",
    )

    if (
        values.isna().any()
        or not np.isfinite(
            values.to_numpy(dtype=float)
        ).all()
    ):
        raise ValueError(
            f"Invalid or missing values in {column}"
        )

    return values.astype(float)


def _policy_rows(
    frame: pd.DataFrame,
    *,
    policy_col: str,
    outcome_col: str,
    prefix: str,
    scale: float,
) -> pd.DataFrame:

    """Handle the internal policy rows step.

    Args:
        frame: Input to the _policy_rows operation.
        policy_col: Input to the _policy_rows operation.
        outcome_col: Input to the _policy_rows operation.
        prefix: Input to the _policy_rows operation.
        scale: Input to the _policy_rows operation.

    Returns:
        Computed result or status for the caller.
    """
    columns = [
        "Decision Date",
        "Policy",
        "Source",
        "Selected Strategy",
        "Return Improvement",
    ]

    if frame.empty:
        return pd.DataFrame(columns=columns)

    work = frame.copy()

    work["Decision Date"] = _clean_dates(
        work
    )

    work["Return Improvement"] = (
        _clean_values(work, outcome_col)
        * scale
    )

    names = (
        work[policy_col]
        .astype("string")
        .str.strip()
    )

    if (
        names.isna().any()
        or names.eq("").any()
    ):
        raise ValueError(
            f"Missing policy names in {policy_col}"
        )

    work["Policy"] = (
        prefix + names.astype(str)
    )

    if work.duplicated(
        ["Decision Date", "Policy"]
    ).any():
        raise ValueError(
            f"Duplicate date/policy rows "
            f"in {prefix} results"
        )

    if "Selected Strategy" not in work.columns:
        work["Selected Strategy"] = pd.NA

    work["Source"] = (
        "ML"
        if prefix == "ML: "
        else "Rule"
    )

    return work[columns]


def compare_historical_policies(
    recommendations: pd.DataFrame,
    *,
    target: str,
    horizon: int,
    rule_results: pd.DataFrame | None = None,
    rule_units: str = "unverified",
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Compare outcomes on dates shared by every policy.

    ML Selected Actual uses fractional return units
    in the saved Hedge Improvement research data.

    Rule outcomes are not combined until their units
    are explicitly specified.
    """

    if target.strip().casefold() != "hedge improvement":
        raise ValueError(
            "No Hedge = 0 is defined here only for "
            "the 'Hedge Improvement' return-improvement "
            "target; risk targets require another baseline."
        )

    if rule_units not in {
        "unverified",
        "fraction",
        "percentage_points",
    }:
        raise ValueError(
            "Unknown rule unit convention"
        )

    needed_ml = {
        "Decision Date",
        "Hedge Horizon Days",
        "Target",
        "Model",
        "Selected Actual",
    }

    if not needed_ml.issubset(
        recommendations.columns
    ):
        raise ValueError(
            "Missing ML fields: "
            f"{sorted(needed_ml - set(recommendations))}"
        )

    ml = recommendations.loc[
        recommendations["Target"]
        .astype(str)
        .str.strip()
        .eq(target)
        & pd.to_numeric(
            recommendations["Hedge Horizon Days"],
            errors="coerce",
        ).eq(horizon)
    ].copy()

    if ml.empty:
        raise ValueError(
            "No ML recommendations for "
            "the selected target/horizon"
        )

    ml_rows = _policy_rows(
        ml,
        policy_col="Model",
        outcome_col="Selected Actual",
        prefix="ML: ",
        scale=1.0,
    )

    # No Hedge provides zero return improvement
    # on each historical ML decision date.
    base_dates = (
        ml_rows["Decision Date"]
        .drop_duplicates()
        .sort_values()
    )

    no_hedge = pd.DataFrame({
        "Decision Date": base_dates.to_numpy(),
        "Policy": "No Hedge",
        "Source": "Baseline",
        "Selected Strategy": "No Hedge",
        "Return Improvement": 0.0,
    })

    parts = [
        ml_rows,
        no_hedge,
    ]

    # -----------------------------------------------------
    # OPTIONAL RULE-BASED BENCHMARK
    # -----------------------------------------------------

    if rule_units != "unverified":

        if rule_results is None:
            raise ValueError(
                "Rule units were set but no "
                "rule-based data was supplied"
            )

        needed_rule = {
            "Decision Date",
            "Horizon",
            "Target",
            "Approach",
            "Realized Outcome",
        }

        if not needed_rule.issubset(
            rule_results.columns
        ):
            raise ValueError(
                "Missing rule fields: "
                f"{sorted(needed_rule - set(rule_results))}"
            )

        rules = rule_results.loc[
            rule_results["Target"]
            .astype(str)
            .str.strip()
            .eq(target)
            & pd.to_numeric(
                rule_results["Horizon"],
                errors="coerce",
            ).eq(horizon)
        ].copy()

        if rules.empty:
            raise ValueError(
                "No rule-based records for "
                "the selected target/horizon"
            )

        scale = (
            0.01
            if rule_units == "percentage_points"
            else 1.0
        )

        parts.append(
            _policy_rows(
                rules,
                policy_col="Approach",
                outcome_col="Realized Outcome",
                prefix="Rule: ",
                scale=scale,
            )
        )

    # -----------------------------------------------------
    # MATCH IDENTICAL DECISION DATES
    # -----------------------------------------------------

    all_rows = pd.concat(
        parts,
        ignore_index=True,
    )

    coverage = (
        all_rows.groupby(
            "Policy",
            as_index=False,
        )
        .agg(
            **{
                "Available dates": (
                    "Decision Date",
                    "nunique",
                )
            }
        )
    )

    date_sets = [
        set(group["Decision Date"])
        for _, group in all_rows.groupby("Policy")
    ]

    common_dates = set.intersection(
        *date_sets
    )

    if not common_dates:
        raise ValueError(
            "No overlapping decision dates "
            "for all selected policies"
        )

    matched = (
        all_rows.loc[
            all_rows["Decision Date"].isin(
                common_dates
            )
        ]
        .copy()
        .sort_values([
            "Decision Date",
            "Policy",
        ])
        .reset_index(drop=True)
    )

    coverage["Matched dates"] = len(
        common_dates
    )

    coverage["Excluded dates"] = (
        coverage["Available dates"]
        - len(common_dates)
    )

    # -----------------------------------------------------
    # SUMMARY
    # -----------------------------------------------------

    summary = (
        matched.groupby(
            "Policy",
            as_index=False,
        )
        .agg(
            **{
                "Decision dates": (
                    "Decision Date",
                    "nunique",
                ),
                "Mean return improvement": (
                    "Return Improvement",
                    "mean",
                ),
                "Median return improvement": (
                    "Return Improvement",
                    "median",
                ),
                "Positive-return frequency": (
                    "Return Improvement",
                    lambda values: float(
                        (values > 0).mean()
                    ),
                ),
            }
        )
    )

    summary = (
        summary.sort_values(
            [
                "Mean return improvement",
                "Policy",
            ],
            ascending=[False, True],
            kind="stable",
        )
        .reset_index(drop=True)
    )

    return matched, summary, coverage


def render_policy_benchmarks(
    project_root: str | Path,
) -> None:
    """Compare ML, No Hedge, and same-run rule policies.

    Rule-based parquet Realized Outcome uses fractional returns.
    Only the generating script's CSV summary scales outcomes by 100.
    """

    import streamlit as st

    root = Path(project_root)

    st.subheader(
        "Historical policy benchmarks — matched decisions"
    )

    st.caption(
        "Historical, exploratory return comparisons. All policies "
        "are evaluated on matching decision dates and horizons. "
        "No live recommendations are produced."
    )

    runs = find_walkforward_runs(root)

    if not runs:
        st.warning(
            "No saved historical recommendation runs were found."
        )
        return

    preferred = next(
        (
            index
            for index, path in enumerate(runs)
            if path.name == (
                "AAPL_O_20220101_20241231_"
                "TC00a56a790ea1_TEST20240102"
            )
        ),
        0,
    )

    run = st.selectbox(
        "Historical experiment (one run only)",
        runs,
        index=preferred,
        format_func=lambda path: (
            path.relative_to(root).as_posix()
        ),
        key="policy_benchmark_run",
    )

    try:

        recommendations = read_walkforward_recommendations(
            run
        )

    except (
        ImportError,
        OSError,
        ValueError,
        KeyError,
    ) as exc:

        st.error(
            f"Unable to read saved recommendations: {exc}"
        )
        return

    target = "Hedge Improvement"

    subset = recommendations.loc[
        recommendations["Target"]
        .astype(str)
        .str.strip()
        .eq(target)
    ]

    if subset.empty:
        st.info(
            "This experiment has no Hedge Improvement target."
        )
        return

    horizons = sorted(
        pd.to_numeric(
            subset["Hedge Horizon Days"],
            errors="coerce",
        )
        .dropna()
        .unique()
        .tolist()
    )

    if not horizons:
        st.warning(
            "No valid hedge horizons were found in this run."
        )
        return

    horizon = st.selectbox(
        "Historical hedge horizon (recorded days)",
        horizons,
        key="policy_benchmark_horizon",
    )

    # -----------------------------------------------------
    # RULE-BASED BENCHMARK: VERIFIED FRACTIONAL RETURNS
    # -----------------------------------------------------

    rule_path = (
        run / "rule_based_benchmark.parquet"
    )

    rules = None
    rule_units = "unverified"

    if rule_path.is_file():

        try:

            candidate_rules = pd.read_parquet(
                rule_path
            )

            if candidate_rules.empty:

                st.warning(
                    "The rule-based benchmark file is empty."
                )

            else:

                rules = candidate_rules

                # Verified from:
                # ml/evaluate_rule_based_benchmark.py
                #
                # Individual outcomes are stored as
                # fractional returns.
                #
                # Only the separate summary CSV
                # multiplies values by 100.

                rule_units = "fraction"

                st.caption(
                    "Rule-based outcome units: fractional return "
                    "(0.01 = 1%). Verified from the generating "
                    "code; automatically converted to "
                    "percentage display."
                )

                with st.expander(
                    "Inspect original rule-based outcomes"
                ):

                    st.write(
                        str(rule_path.relative_to(root))
                    )

                    st.dataframe(
                        rules.head(20),
                        hide_index=True,
                        use_container_width=True,
                    )

        except (
            ImportError,
            OSError,
            ValueError,
        ) as exc:

            st.warning(
                f"Cannot read rule-based benchmark: {exc}"
            )

    else:

        st.info(
            "No rule_based_benchmark.parquet in this run; "
            "only ML and No Hedge are compared."
        )

    # -----------------------------------------------------
    # MATCHED HISTORICAL POLICY COMPARISON
    # -----------------------------------------------------

    try:

        matched, summary, coverage = (
            compare_historical_policies(
                recommendations,
                target=target,
                horizon=int(horizon),
                rule_results=rules,
                rule_units=rule_units,
            )
        )

    except (
        ValueError,
        TypeError,
        KeyError,
    ) as exc:

        st.error(
            f"Cannot compare historical policies: {exc}"
        )
        return

    count = matched[
        "Decision Date"
    ].nunique()

    st.write(
        f"**Matched decision dates: {count}**"
    )

    # Fractional returns are displayed as percentages.
    # Example: -0.0105 -> -1.05%.

    # Plot the matched decision-date policies before showing raw values.
    # This is a display change only: no new ranking or re-fitting occurs.
    from dashboard_visuals import model_ranking_chart
    st.plotly_chart(
        model_ranking_chart(
            summary, metric="Mean return improvement", name_col="Policy",
            title="Matched-date mean return improvement (fraction)",
        ),
        use_container_width=True,
        key="policy_matched_return_chart",
    )

    st.dataframe(
        summary.style.format({
            "Mean return improvement": "{:+.2%}",
            "Median return improvement": "{:+.2%}",
            "Positive-return frequency": "{:.1%}",
        }),
        hide_index=True,
        use_container_width=True,
    )

    if count < 20:

        st.warning(
            "Fewer than 20 shared historical decision dates. "
            "Results are exploratory and cannot establish "
            "model superiority."
        )

    st.caption(
        "All return improvements are expressed relative to "
        "No Hedge. No Hedge and Rule: No Hedge may both appear: "
        "they represent the explicit zero baseline and the "
        "saved rule policy. Likewise, ML: Historical Mean "
        "and Rule: Historical Mean are different recorded "
        "policies and need not have the same outcome. "
        "This table measures terminal return, not "
        "drawdown protection."
    )

    # -----------------------------------------------------
    # DATE COVERAGE AUDIT
    # -----------------------------------------------------

    with st.expander(
        "Date coverage audit"
    ):

        st.dataframe(
            coverage,
            hide_index=True,
            use_container_width=True,
        )

    # -----------------------------------------------------
    # INDIVIDUAL MATCHED HISTORICAL OUTCOMES
    # -----------------------------------------------------

    with st.expander(
        "Matched date-by-date realized outcomes"
    ):

        display = matched.copy()

        display["Return Improvement (%)"] = (
            display["Return Improvement"] * 100
        )

        st.dataframe(
            display,
            hide_index=True,
            use_container_width=True,
        )
