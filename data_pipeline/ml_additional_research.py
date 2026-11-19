
"""Read classification and hedge-candidate ML evaluations.

No retraining, model deserialization, or network requests.
Classification selection uses validation metrics only.
Candidate walk-forward results remain exploratory.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd


CLASS_COLUMNS = [
    "Target",
    "Model",
    "Split",
    "Brier",
    "ROC AUC",
    "Log Loss",
    "Source",
]

FOLD_COLUMNS = [
    "Decision Date",
    "Model",
    "Return MAE (pp)",
    "Risk MAE (pp)",
    "Return Benefit (pp)",
    "Risk Benefit (pp)",
    "Utility (pp)",
    "Decision",
    "Eligible",
    "Source",
]


def _key(value: object) -> str:
    return re.sub(
        r"[^a-z0-9]",
        "",
        str(value).lower(),
    )


def _column(
    frame: pd.DataFrame,
    *aliases: str,
) -> str | None:

    names = {
        _key(column): column
        for column in frame.columns
    }

    for alias in aliases:
        result = names.get(_key(alias))
        if result is not None:
            return result

    return None


def _value(
    record: pd.Series,
    frame: pd.DataFrame,
    *aliases: str,
):
    name = _column(frame, *aliases)
    return record[name] if name is not None else None


def _metric(
    record: pd.Series,
    frame: pd.DataFrame,
    *aliases: str,
) -> float | None:

    value = pd.to_numeric(
        pd.Series([
            _value(record, frame, *aliases)
        ]),
        errors="coerce",
    ).iloc[0]

    return (
        float(value)
        if pd.notna(value)
        else None
    )


def _split(raw: object) -> str | None:

    name = str(raw).lower()

    if "valid" in name or name == "val":
        return "validation"

    if "test" in name or "holdout" in name:
        return "test"

    return None


def _read_small_table(path: Path) -> pd.DataFrame:

    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)

    with path.open(encoding="utf-8") as stream:
        obj = json.load(stream)

    if isinstance(obj, list):
        return pd.DataFrame(obj)

    if isinstance(obj, dict):

        for name in (
            "metrics",
            "results",
            "scores",
            "evaluation",
            "folds",
            "rows",
        ):
            if isinstance(obj.get(name), list):
                return pd.DataFrame(obj[name])

        return pd.DataFrame([obj])

    return pd.DataFrame()


def _class_rows(
    frame: pd.DataFrame,
    source: str,
) -> list[dict]:

    model_col = _column(
        frame,
        "Model",
        "Model Name",
        "Estimator",
        "Algorithm",
    )

    target_col = _column(
        frame,
        "Target",
        "Outcome",
        "Prediction Target",
    )

    threshold_col = _column(
        frame,
        "Threshold",
        "Loss Threshold",
    )

    split_col = _column(
        frame,
        "Split",
        "Dataset",
        "Partition",
    )

    if model_col is None:
        return []

    if target_col is None and threshold_col is None:
        return []

    has_wide = any(
        _column(
            frame,
            f"{split} {field}",
        ) is not None
        for split in ("Validation", "Test")
        for field in (
            "Brier",
            "Brier Score",
            "ROC AUC",
            "AUC",
            "Log Loss",
        )
    )

    rows = []

    for _, record in frame.iterrows():

        if pd.isna(record[model_col]):
            continue

        if target_col is not None:
            target = str(
                record[target_col]
            ).strip()
        else:
            target = (
                f"Loss threshold "
                f"{record[threshold_col]}"
            )

        model = str(
            record[model_col]
        ).strip()

        if (
            target.lower() in ("nan", "none", "")
            or model.lower() in ("nan", "none", "")
        ):
            continue

        if has_wide:
            splits = ("validation", "test")
        else:
            splits = (
                _split(record[split_col])
                if split_col is not None
                else _split(source),
            )

        for split in splits:

            if split is None:
                continue

            prefix = split if has_wide else ""

            brier = _metric(
                record,
                frame,
                f"{prefix} Brier",
                f"{prefix} Brier Score",
                f"{prefix} Brier Loss",
            )

            auc = _metric(
                record,
                frame,
                f"{prefix} ROC AUC",
                f"{prefix} AUC",
                f"{prefix} AUROC",
            )

            logloss = _metric(
                record,
                frame,
                f"{prefix} Log Loss",
                f"{prefix} LogLoss",
                f"{prefix} Cross Entropy",
            )

            if all(
                value is None
                for value in (
                    brier,
                    auc,
                    logloss,
                )
            ):
                continue

            rows.append({
                "Target": target,
                "Model": model,
                "Split": split,
                "Brier": brier,
                "ROC AUC": auc,
                "Log Loss": logloss,
                "Source": source,
            })

    return rows


def _fold_rows(
    frame: pd.DataFrame,
    source: str,
) -> list[dict]:

    date_col = _column(
        frame,
        "Decision Date",
        "Fold Date",
    )

    model_col = _column(
        frame,
        "Model",
        "Model Name",
    )

    if date_col is None or model_col is None:
        return []

    metric_aliases = {
        "Return MAE (pp)": (
            "Return MAE (pp)",
            "Return MAE pp",
        ),
        "Risk MAE (pp)": (
            "Risk MAE (pp)",
            "Risk MAE pp",
        ),
        "Return Benefit (pp)": (
            "Return Benefit (pp)",
        ),
        "Risk Benefit (pp)": (
            "Risk Benefit (pp)",
        ),
        "Utility (pp)": (
            "Utility (pp)",
            "Utility pp",
        ),
    }

    if not any(
        _column(frame, *aliases)
        for aliases in metric_aliases.values()
    ):
        return []

    rows = []

    for _, record in frame.iterrows():

        day = pd.to_datetime(
            record[date_col],
            errors="coerce",
        )

        model = str(
            record[model_col]
        ).strip()

        if (
            pd.isna(day)
            or model.lower() in ("nan", "none", "")
        ):
            continue

        row = {
            "Decision Date": day.date(),
            "Model": model,
            "Source": source,
        }

        for field, aliases in metric_aliases.items():
            row[field] = _metric(
                record,
                frame,
                *aliases,
            )

        row["Decision"] = _value(
            record,
            frame,
            "Decision",
            "Selected Strategy",
        )

        row["Eligible"] = _metric(
            record,
            frame,
            "Eligible",
            "Eligible Candidates",
        )

        rows.append(row)

    return rows


def scan_ml_evaluations(
    root: str | Path,
    *,
    max_file_mb: float = 8.0,
    max_files: int = 2000,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Find supported saved evaluation tables."""

    root = Path(root)

    if not root.is_dir():
        raise FileNotFoundError(
            f"Directory not found: {root}"
        )

    class_rows = []
    fold_rows = []
    audit = []

    seen = 0

    for path in sorted(root.rglob("*")):

        if not path.is_file():
            continue

        if path.suffix.lower() not in (
            ".csv",
            ".json",
        ):
            continue

        seen += 1

        if seen > max_files:

            audit.append({
                "File": "...",
                "Status": (
                    f"Stopped after {max_files} "
                    "CSV/JSON files"
                ),
            })

            break

        relative = path.relative_to(
            root
        ).as_posix()

        if "pilot" in {
            part.lower()
            for part in path.relative_to(root).parts
        }:
            continue

        if path.stat().st_size > (
            max_file_mb * 1024 * 1024
        ):

            audit.append({
                "File": relative,
                "Status": "Skipped: large file",
            })

            continue

        # Search evaluation-like filenames.
        if not any(
            token in path.stem.lower()
            for token in (
                "metric",
                "valid",
                "test",
                "score",
                "evaluat",
                "classif",
                "walk",
                "fold",
                "compar",
                "bench",
                "candidate",
                "ranking",
            )
        ):
            continue

        try:

            table = _read_small_table(path)

            classification = _class_rows(
                table,
                relative,
            )

            candidates = _fold_rows(
                table,
                relative,
            )

            class_rows.extend(classification)
            fold_rows.extend(candidates)

            audit.append({
                "File": relative,
                "Status": (
                    f"Classification: "
                    f"{len(classification)}; "
                    f"candidate folds: "
                    f"{len(candidates)}"
                ),
            })

        except (
            OSError,
            ValueError,
            KeyError,
            TypeError,
            UnicodeError,
            json.JSONDecodeError,
        ) as exc:

            audit.append({
                "File": relative,
                "Status": (
                    f"Unreadable: "
                    f"{type(exc).__name__}: {exc}"
                ),
            })

    classes = pd.DataFrame(
        class_rows,
        columns=CLASS_COLUMNS,
    ).drop_duplicates()

    folds = pd.DataFrame(
        fold_rows,
        columns=FOLD_COLUMNS,
    ).drop_duplicates()

    audit_frame = pd.DataFrame(
        audit,
        columns=["File", "Status"],
    )

    return classes, folds, audit_frame


def rank_classification_validation(
    scores: pd.DataFrame,
    target: str,
) -> tuple[pd.DataFrame, str | None]:
    """Choose by validation metrics, never test metrics."""

    rows = scores.loc[
        scores["Target"].eq(target)
        & scores["Split"].eq("validation")
    ].copy()

    if rows.empty:
        return rows, None

    for metric, ascending in (
        ("Brier", True),
        ("Log Loss", True),
        ("ROC AUC", False),
    ):

        if not rows[metric].notna().any():
            continue

        rows = rows.loc[
            rows[metric].notna()
        ].copy()

        for model, grouped in rows.groupby("Model"):

            if grouped[metric].nunique() > 1:

                raise ValueError(
                    f"Conflicting {metric} validation "
                    f"scores for {model}. Select one "
                    "training-run directory."
                )

        rows = rows.drop_duplicates(
            subset=[
                "Target",
                "Model",
                "Split",
                metric,
            ]
        )

        rows = rows.sort_values(
            [metric, "Model"],
            ascending=[ascending, True],
            kind="stable",
        )

        baseline = rows["Model"].str.contains(
            "baseline|prevalence|historical.mean",
            case=False,
            regex=True,
        )

        trained = rows.loc[
            ~baseline
        ]

        winner = (
            str(trained.iloc[0]["Model"])
            if not trained.empty
            else None
        )

        return rows, winner

    return rows, None


def render_additional_ml_research(
    default_root: str | Path,
) -> None:
    """Display classification and hedge-candidate results."""

    import streamlit as st

    st.subheader(
        "Classification and hedge-candidate ML research"
    )

    st.caption(
        "Saved evaluations only. No retraining or live "
        "inference. Classification models are selected "
        "using validation results, never test results."
    )

    directory = st.text_input(
        "Search evaluation files under",
        value=str(default_root),
        key="ml_extension_directory",
    )

    try:
        classes, folds, audit = scan_ml_evaluations(
            directory
        )

    except (FileNotFoundError, OSError) as exc:
        st.error(str(exc))
        return

    classification_tab, candidate_tab, audit_tab = (
        st.tabs([
            "Loss classification",
            "Hedge-candidate walk-forward",
            "Artifact audit",
        ])
    )

    with classification_tab:

        if classes.empty:

            st.warning(
                "No saved classification evaluation "
                "metrics were recognized. Forecast "
                "probabilities alone cannot establish "
                "Brier scores or ROC AUC without "
                "realized outcomes."
            )

        else:

            target = st.selectbox(
                "Classification target",
                sorted(classes["Target"].unique()),
                key="additional_ml_class_target",
            )

            try:
                ranking, selected = (
                    rank_classification_validation(
                        classes,
                        target,
                    )
                )

            except ValueError as exc:
                st.error(str(exc))
                ranking = pd.DataFrame()
                selected = None

            if not ranking.empty:

                st.write(
                    "**Validation-selected model:** "
                    f"{selected or 'None'}"
                )

                st.dataframe(
                    ranking,
                    hide_index=True,
                    use_container_width=True,
                )

            else:

                st.info(
                    "No identifiable validation "
                    "metrics for this target."
                )

            tests = classes.loc[
                classes["Target"].eq(target)
                & classes["Split"].eq("test")
            ]

            if not tests.empty:

                with st.expander(
                    "Test results — not used for selection"
                ):

                    st.dataframe(
                        tests,
                        hide_index=True,
                        use_container_width=True,
                    )

    with candidate_tab:

        if folds.empty:

            st.warning(
                "No saved candidate-level walk-forward "
                "evaluation table was found. A training "
                "command that only prints results to the "
                "terminal does not produce a reusable "
                "CSV evaluation artifact."
            )

        else:

            source = st.selectbox(
                "Walk-forward comparison file",
                sorted(folds["Source"].unique()),
                key="additional_ml_candidate_source",
            )

            chosen = folds.loc[
                folds["Source"].eq(source)
            ].copy()

            st.caption(
                "Exploratory fold results. Means are "
                "equally weighted across recorded rows. "
                "These are not a pooled backtest or an "
                "independent final holdout evaluation."
            )

            metrics = [
                "Return MAE (pp)",
                "Risk MAE (pp)",
                "Return Benefit (pp)",
                "Risk Benefit (pp)",
                "Utility (pp)",
            ]

            summary = (
                chosen.groupby(
                    "Model",
                    dropna=False,
                )
                .agg(
                    **{
                        "Decision dates": (
                            "Decision Date",
                            "nunique",
                        )
                    },
                    **{
                        metric: (
                            metric,
                            "mean",
                        )
                        for metric in metrics
                    },
                )
                .reset_index()
            )

            st.dataframe(
                summary,
                hide_index=True,
                use_container_width=True,
            )

            with st.expander(
                "Inspect individual fold results"
            ):

                st.dataframe(
                    chosen,
                    hide_index=True,
                    use_container_width=True,
                )

    with audit_tab:

        st.dataframe(
            audit,
            hide_index=True,
            use_container_width=True,
        )

        st.caption(
            "The scanner reads small CSV/JSON files "
            "whose filenames suggest model evaluations. "
            "Other formats or locations may need a "
            "separate reader."
        )
