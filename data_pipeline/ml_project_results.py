
"""Views of saved stock-risk and hedge-candidate ML evaluations.

Uses one evaluation run at a time.
No fitting, LSEG requests, or trained-model loading.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


CLASSIFICATION_COLUMNS = (
    "Target",
    "Model",
    "Rows",
    "Events",
    "Prevalence",
    "Brier",
    "Average Precision",
    "ROC AUC",
)


def read_classification_validation(
    project_root: str | Path,
) -> pd.DataFrame:

    path = (
        Path(project_root)
        / "data"
        / "processed"
        / "ml_baselines"
        / "equity_63d_classifier_suite_v1"
        / "validation_all_models.csv"
    )

    if not path.is_file():
        raise FileNotFoundError(
            f"Classification validation missing: {path}"
        )

    data = pd.read_csv(path)

    missing = (
        set(CLASSIFICATION_COLUMNS)
        - set(data.columns)
    )

    if missing:
        raise ValueError(
            f"Classification fields missing: "
            f"{sorted(missing)}"
        )

    data = data.loc[
        :,
        list(CLASSIFICATION_COLUMNS),
    ].copy()

    for column in (
        "Rows",
        "Events",
        "Prevalence",
        "Brier",
        "Average Precision",
        "ROC AUC",
    ):
        data[column] = pd.to_numeric(
            data[column],
            errors="raise",
        )

    for column in (
        "Prevalence",
        "Brier",
        "Average Precision",
        "ROC AUC",
    ):
        if not data[column].between(0, 1).all():
            raise ValueError(
                f"Invalid classification {column}"
            )

    if data.duplicated(
        ["Target", "Model"]
    ).any():
        raise ValueError(
            "Duplicate classifier target/model rows"
        )

    if data.empty:
        raise ValueError(
            "Classification validation is empty"
        )

    return data


def read_recorded_classifier_selection(
    project_root: str | Path,
) -> pd.DataFrame:

    path = (
        Path(project_root)
        / "data/processed/ml_baselines"
        / "equity_63d_classifier_suite_v1"
        / "validation_selected.csv"
    )

    if not path.is_file():
        return pd.DataFrame()

    data = pd.read_csv(path)

    if not {
        "Target",
        "Selected Model",
    }.issubset(data.columns):
        raise ValueError(
            f"Invalid classifier selection: {path}"
        )

    return data


def read_exploratory_regression_test(
    project_root: str | Path,
) -> pd.DataFrame:

    path = (
        Path(project_root)
        / "data/processed/ml_baselines"
        / "equity_63d_suite_v1/full"
        / "exploratory_selected_vs_median.csv"
    )

    if not path.is_file():
        raise FileNotFoundError(
            f"Regression test missing: {path}"
        )

    data = pd.read_csv(path)

    columns = [
        "Target",
        "Model",
        "Test Rows",
        "MAE",
        "RMSE",
        "R2",
        "Bias",
    ]

    missing = set(columns) - set(data.columns)

    if missing:
        raise ValueError(
            f"Regression fields missing: "
            f"{sorted(missing)}"
        )

    return data[columns].copy()


def find_walkforward_runs(
    project_root: str | Path,
) -> list[Path]:

    root = (
        Path(project_root)
        / "data/processed/ml_evaluations"
    )

    if not root.is_dir():
        return []

    return sorted({
        path.parent
        for path in root.rglob(
            "walk_forward_recommendations.parquet"
        )
    })


def read_walkforward_recommendations(
    run_dir: str | Path,
) -> pd.DataFrame:

    path = (
        Path(run_dir)
        / "walk_forward_recommendations.parquet"
    )

    if not path.is_file():
        raise FileNotFoundError(
            f"Walk-forward results missing: {path}"
        )

    data = pd.read_parquet(path)

    required = {
        "Decision Date",
        "Hedge Horizon Days",
        "Target",
        "Model",
        "Selected Strategy",
        "Best Strategy",
        "Selected Actual",
        "Best Actual",
        "Regret",
        "Correct Selection",
    }

    missing = required - set(data.columns)

    if missing:
        raise ValueError(
            f"Walk-forward fields missing: "
            f"{sorted(missing)}"
        )

    if data.empty:
        raise ValueError(
            "Walk-forward results are empty"
        )

    return data


def summarize_walkforward(
    data: pd.DataFrame,
    target: str,
    horizon: int,
) -> pd.DataFrame:
    """Describe one historical experiment, without selecting a winner."""

    filtered = data.loc[
        data["Target"].eq(target)
        & pd.to_numeric(
            data["Hedge Horizon Days"],
            errors="coerce",
        ).eq(horizon)
    ].copy()

    if filtered.empty:
        return pd.DataFrame()

    for column in (
        "Regret",
        "Selected Actual",
    ):
        filtered[column] = pd.to_numeric(
            filtered[column],
            errors="coerce",
        )

    filtered["Correct Selection"] = (
        filtered["Correct Selection"]
        .astype("boolean")
    )

    summary = (
        filtered.groupby(
            "Model",
            dropna=False,
        )
        .agg(
            **{
                "Recorded decisions": (
                    "Decision Date",
                    "nunique",
                ),
                "Mean regret (stored units)": (
                    "Regret",
                    "mean",
                ),
                "Mean selected outcome (stored units)": (
                    "Selected Actual",
                    "mean",
                ),
                "Correct selection rate": (
                    "Correct Selection",
                    "mean",
                ),
            }
        )
        .reset_index()
        .sort_values("Model")
    )

    return summary


def find_candidate_validation_tables(
    project_root: str | Path,
) -> list[Path]:

    root = (
        Path(project_root)
        / "data/processed/ml_datasets"
    )

    if not root.is_dir():
        return []

    return sorted(
        root.glob(
            "*/model_benchmarks/validation_ranking.csv"
        )
    )


def read_candidate_validation(
    path: str | Path,
) -> pd.DataFrame:

    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(
            f"Candidate validation missing: {path}"
        )

    frame = pd.read_csv(path)

    required = {
        "Model",
        "Validation MAE",
        "Mean Realized Improvement",
    }

    missing = required - set(frame.columns)

    if missing:
        raise ValueError(
            f"Candidate fields missing: "
            f"{sorted(missing)}"
        )

    return frame.sort_values(
        "Validation MAE",
        kind="stable",
    )



def render_project_ml_results(
    project_root: str | Path,
) -> None:
    """Render the saved historical ML evaluation dashboard."""

    import streamlit as st

    root = Path(project_root)

    st.subheader(
        "Saved ML evaluations — verified project formats"
    )

    st.caption(
        "Historical evaluations from separate experiments. "
        "No retraining, live predictions, or selection "
        "using exploratory test results."
    )

    (
        tab_class,
        tab_regression,
        tab_candidate,
        tab_candidate_validation,
        tab_audit,
    ) = st.tabs(
        [
            "63-session classification",
            "Regression exploratory test",
            "Hedge-candidate walk-forward",
            "Candidate validation",
            "Sources and limitations",
        ]
    )

    # =====================================================
    # 1. CLASSIFICATION VALIDATION
    # =====================================================

    with tab_class:

        st.write(
            "**Classification: full-run validation**"
        )

        selected_classification_target = None

        try:

            classes = read_classification_validation(
                root
            )

            selected = read_recorded_classifier_selection(
                root
            )

            targets = sorted(
                classes["Target"].unique()
            )

            target = st.selectbox(
                "Loss threshold",
                targets,
                key="real_class_target",
            )

            selected_classification_target = target

            # Display model results using Average Precision.
            view = (
                classes.loc[
                    classes["Target"].eq(target)
                ]
                .sort_values(
                    ["Average Precision", "Brier"],
                    ascending=[False, True],
                    kind="stable",
                )
            )

            st.dataframe(
                view,
                hide_index=True,
                use_container_width=True,
            )

            # Show the original model selection
            # recorded by the training experiment.
            if not selected.empty:

                selected_view = selected.loc[
                    selected["Target"].eq(target)
                ]

                if not selected_view.empty:

                    st.caption(
                        "Original training-run selection:"
                    )

                    st.dataframe(
                        selected_view,
                        hide_index=True,
                        use_container_width=True,
                    )

            st.caption(
                "Brier: lower is better. "
                "Average Precision and ROC AUC: "
                "higher is better. "
                "Only validation scores are used here."
            )

        except (
            FileNotFoundError,
            ValueError,
            OSError,
        ) as exc:

            st.warning(
                f"Unable to load classification validation: {exc}"
            )

        # -------------------------------------------------
        # EXPLORATORY CLASSIFICATION TEST
        # -------------------------------------------------

        test_file = (
            root
            / "data/processed/ml_baselines"
            / "equity_63d_classifier_suite_v1"
            / "exploratory_top30_test.csv"
        )

        with st.expander(
            "Exploratory classification top-30 test results"
        ):

            if test_file.is_file():

                try:

                    tests = pd.read_csv(
                        test_file
                    )

                    if (
                        "Target" in tests.columns
                        and selected_classification_target
                        is not None
                    ):

                        tests = tests.loc[
                            tests["Target"].eq(
                                selected_classification_target
                            )
                        ].copy()

                    if tests.empty:

                        st.info(
                            "No exploratory test records "
                            "for the selected target."
                        )

                    else:

                        st.dataframe(
                            tests,
                            hide_index=True,
                            use_container_width=True,
                        )

                    st.caption(
                        "This evaluates historical alert "
                        "policies. It is not a fresh "
                        "independent test of every classifier."
                    )

                except (
                    OSError,
                    ValueError,
                    KeyError,
                ) as exc:

                    st.warning(
                        f"Unable to read classification "
                        f"test results: {exc}"
                    )

            else:

                st.info(
                    f"No saved top-30 test table: {test_file}"
                )

    # =====================================================
    # 2. REGRESSION EXPLORATORY TEST
    # =====================================================

    with tab_regression:

        st.write(
            "**Exploratory test: selected models "
            "versus training median**"
        )

        try:

            table = read_exploratory_regression_test(
                root
            )

            st.dataframe(
                table,
                hide_index=True,
                use_container_width=True,
            )

            st.warning(
                "This test window was recorded as "
                "reused/exploratory. Do not select "
                "models using these results or "
                "describe them as an independent "
                "final test."
            )

        except (
            FileNotFoundError,
            ValueError,
            OSError,
        ) as exc:

            st.warning(
                f"Unable to load exploratory "
                f"regression test: {exc}"
            )

    # =====================================================
    # 3. HEDGE-CANDIDATE WALK-FORWARD
    # =====================================================

    with tab_candidate:

        runs = find_walkforward_runs(
            root
        )

        if not runs:

            st.warning(
                "No saved walk-forward Parquet "
                "recommendations found."
            )

        else:

            run = st.selectbox(
                "Select ONE historical walk-forward run",
                runs,
                format_func=lambda path: (
                    path.relative_to(root).as_posix()
                ),
                key="real_walkforward_run",
            )

            try:

                data = read_walkforward_recommendations(
                    run
                )

                # -----------------------------------------
                # SELECT PREDICTION TARGET
                # -----------------------------------------

                targets = sorted(
                    data["Target"]
                    .dropna()
                    .unique()
                )

                target = st.selectbox(
                    "Hedge-candidate target",
                    targets,
                    key="real_walkforward_target",
                )

                filtered = data.loc[
                    data["Target"].eq(target)
                ].copy()

                # -----------------------------------------
                # SELECT HEDGE HORIZON
                # -----------------------------------------

                horizons = sorted(
                    pd.to_numeric(
                        filtered["Hedge Horizon Days"],
                        errors="coerce",
                    )
                    .dropna()
                    .unique()
                )

                if len(horizons) == 0:

                    st.warning(
                        "No valid hedge horizons "
                        "for the selected target."
                    )

                else:

                    horizon = st.selectbox(
                        "Recorded hedge horizon (days)",
                        horizons,
                        key="real_walkforward_horizon",
                    )

                    # -------------------------------------
                    # CALCULATE MODEL COMPARISON
                    # -------------------------------------

                    summary = summarize_walkforward(
                        data,
                        target,
                        horizon,
                    )

                    if summary.empty:

                        st.warning(
                            "No walk-forward results "
                            "for the selected target "
                            "and hedge horizon."
                        )

                    else:

                        st.dataframe(
                            summary,
                            hide_index=True,
                            use_container_width=True,
                        )

                        # ---------------------------------
                        # SAMPLE-SIZE VALIDATION
                        # ---------------------------------

                        minimum_decisions = int(
                            summary[
                                "Recorded decisions"
                            ].min()
                        )

                        maximum_decisions = int(
                            summary[
                                "Recorded decisions"
                            ].max()
                        )

                        st.caption(
                            "Recorded decision dates per model: "
                            f"{minimum_decisions}–"
                            f"{maximum_decisions}"
                        )

                        if minimum_decisions < 20:

                            st.warning(
                                "At least one model has fewer "
                                "than 20 recorded decision "
                                "dates. Results remain "
                                "exploratory and do not "
                                "establish reliable model "
                                "superiority."
                            )

                        if (
                            summary[
                                "Recorded decisions"
                            ].nunique() > 1
                        ):

                            st.warning(
                                "Models have different "
                                "numbers of evaluated "
                                "decision dates. Their "
                                "mean outcomes may not "
                                "be directly comparable."
                            )

                        # ---------------------------------
                        # CHECK MODEL SELECTION EVIDENCE
                        # ---------------------------------

                        if (
                            "Historical Mean"
                            in summary["Model"].values
                        ):

                            benchmark = summary.loc[
                                summary["Model"].eq(
                                    "Historical Mean"
                                )
                            ].iloc[0]

                            st.caption(
                                "Historical Mean benchmark: "
                                "mean regret "
                                f"{benchmark['Mean regret (stored units)']:.6f}; "
                                "mean selected outcome "
                                f"{benchmark['Mean selected outcome (stored units)']:.6f}."
                            )

                        st.caption(
                            "Regret and outcomes use the "
                            "original saved units. Correct "
                            "selection rate compares the "
                            "recorded recommendation with "
                            "the hindsight-best candidate. "
                            "Runs may overlap and remain "
                            "exploratory."
                        )

                    # -------------------------------------
                    # INDIVIDUAL RECOMMENDATION RECORDS
                    # -------------------------------------

                    with st.expander(
                        "Individual recommendation records"
                    ):

                        selected_rows = filtered.loc[
                            pd.to_numeric(
                                filtered[
                                    "Hedge Horizon Days"
                                ],
                                errors="coerce",
                            ).eq(horizon)
                        ].copy()

                        st.dataframe(
                            selected_rows,
                            hide_index=True,
                            use_container_width=True,
                        )

            except (
                ImportError,
                FileNotFoundError,
                ValueError,
                OSError,
                KeyError,
                TypeError,
            ) as exc:

                st.error(
                    f"Unable to read or summarize "
                    f"Parquet run: {exc}"
                )

    # =====================================================
    # 4. HISTORICAL HEDGE-CANDIDATE VALIDATION
    # =====================================================

    with tab_candidate_validation:

        tables = find_candidate_validation_tables(
            root
        )

        if not tables:

            st.info(
                "No candidate validation rankings found."
            )

        else:

            ranking_path = st.selectbox(
                "Select ONE candidate validation experiment",
                tables,
                format_func=lambda path: (
                    path.relative_to(root).as_posix()
                ),
                key="real_candidate_validation_run",
            )

            try:

                ranking = read_candidate_validation(
                    ranking_path
                )

                if ranking.empty:

                    st.warning(
                        "The selected candidate validation "
                        "table is empty."
                    )

                else:

                    st.dataframe(
                        ranking,
                        hide_index=True,
                        use_container_width=True,
                    )

                st.caption(
                    "Saved validation results are shown "
                    "in the original experiment's units. "
                    "Candidate models may use different "
                    "hedge horizons and features from "
                    "the 63-session stock-risk models."
                )

            except (
                FileNotFoundError,
                ValueError,
                OSError,
                KeyError,
            ) as exc:

                st.warning(
                    f"Unable to load candidate "
                    f"validation: {exc}"
                )

    # =====================================================
    # 5. SOURCES AND LIMITATIONS
    # =====================================================

    with tab_audit:

        sources = [
            (
                root
                / "data/processed/ml_baselines"
                / "equity_63d_classifier_suite_v1"
                / "validation_all_models.csv"
            ),
            (
                root
                / "data/processed/ml_baselines"
                / "equity_63d_classifier_suite_v1"
                / "validation_selected.csv"
            ),
            (
                root
                / "data/processed/ml_baselines"
                / "equity_63d_classifier_suite_v1"
                / "exploratory_top30_test.csv"
            ),
            (
                root
                / "data/processed/ml_baselines"
                / "equity_63d_suite_v1"
                / "full"
                / "exploratory_selected_vs_median.csv"
            ),
        ]

        audit_rows = [
            {
                "File": path.relative_to(
                    root
                ).as_posix(),
                "Exists": path.is_file(),
            }
            for path in sources
        ]

        st.dataframe(
            pd.DataFrame(audit_rows),
            hide_index=True,
            use_container_width=True,
        )

        walkforward_runs = find_walkforward_runs(
            root
        )

        st.write(
            f"Found **{len(walkforward_runs)}** "
            "saved walk-forward run directories."
        )

        candidate_tables = find_candidate_validation_tables(
            root
        )

        st.write(
            f"Found **{len(candidate_tables)}** "
            "candidate validation ranking files."
        )

        st.caption(
            "63-session stock-risk research and "
            "historical hedge-candidate experiments "
            "have different horizons, targets, "
            "training datasets and evaluation schemes. "
            "Their results must not be pooled or "
            "treated as live hedging recommendations."
        )

