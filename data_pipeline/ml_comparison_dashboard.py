
"""Read and compare saved ML evaluation metrics, without retraining.

Only validation metrics are used for model selection.
Test metrics are shown separately.

This module does not load pickle or joblib model files.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd


FIELDS = {
    "target": (
        "target",
        "targetname",
        "predictiontarget",
        "outcome",
    ),
    "model": (
        "model",
        "modelname",
        "selectedmodel",
        "estimator",
        "algorithm",
    ),
    "split": (
        "split",
        "dataset",
        "partition",
        "datasplit",
    ),
    "mae": (
        "mae",
        "meanabsoluteerror",
    ),
    "rmse": (
        "rmse",
        "rootmeansquarederror",
    ),
    "r2": (
        "r2",
        "r2score",
        "rsquared",
    ),
    "auc": (
        "rocauc",
        "auroc",
        "auc",
    ),
    "brier": (
        "brier",
        "brierscore",
        "brierloss",
    ),
    "log_loss": (
        "logloss",
        "logarithmicloss",
        "crossentropy",
    ),
}

METRICS = (
    "mae",
    "rmse",
    "r2",
    "auc",
    "brier",
    "log_loss",
)

DISPLAY = {
    "mae": "MAE",
    "rmse": "RMSE",
    "r2": "R2",
    "auc": "ROC AUC",
    "brier": "Brier",
    "log_loss": "Log Loss",
}

OUT_COLUMNS = [
    "Task",
    "Target",
    "Model",
    "Split",
    "MAE",
    "RMSE",
    "R2",
    "ROC AUC",
    "Brier",
    "Log Loss",
    "Source",
]


def _key(value: str) -> str:
    """Normalize a column name."""

    return re.sub(
        r"[^a-z0-9]",
        "",
        str(value).casefold(),
    )


def _col(
    frame: pd.DataFrame,
    field: str,
    prefix: str = "",
) -> str | None:
    """Find a column through its normalized aliases."""

    available = {
        _key(column): column
        for column in frame.columns
    }

    for alias in FIELDS[field]:

        match = available.get(
            _key(prefix + alias)
        )

        if match is not None:
            return match

    return None


def _split(value: object) -> str | None:
    """Identify a dataset split without guessing."""

    value = str(value).casefold()

    if "valid" in value or value == "val":
        return "validation"

    if "test" in value:
        return "test"

    if "train" in value:
        return "train"

    return None


def _task(
    target: str,
    metrics: dict,
) -> str:
    """Identify the target's ML application."""

    name = target.casefold()

    regression_targets = (
        "max_drawdown",
        "max_loss_from_entry",
        "realized_vol",
    )

    if any(
        token in name
        for token in regression_targets
    ):
        return "Regression"

    if (
        re.search(
            r"(?:10|15|20)(?:pct|percent|%)",
            name,
        )
        and "loss" in name
    ):
        return "Classification"

    classification_tokens = (
        "classification",
        "probability",
        "loss_event",
        "tail_event",
    )

    if any(
        token in name
        for token in classification_tokens
    ):
        return "Classification"

    if any(
        pd.notna(metrics[metric])
        for metric in ("auc", "brier", "log_loss")
    ):
        return "Classification"

    return "Regression"


def _rows(
    frame: pd.DataFrame,
    source: str,
) -> list[dict]:
    """Convert saved metric tables to a common schema."""

    target_column = _col(frame, "target")
    model_column = _col(frame, "model")

    if (
        target_column is None
        or model_column is None
    ):
        return []

    # Supports:
    # Target, Model, Split, MAE
    #
    # And:
    # Target, Model, Validation MAE, Test MAE
    wide = any(
        _col(frame, metric, split)
        for split in ("validation", "test")
        for metric in METRICS
    )

    split_column = _col(frame, "split")

    filename_split = _split(
        Path(source).name
    )

    output = []

    for _, record in frame.iterrows():

        if (
            pd.isna(record[target_column])
            or pd.isna(record[model_column])
        ):
            continue

        target = str(
            record[target_column]
        ).strip()

        model = str(
            record[model_column]
        ).strip()

        if not target or not model:
            continue

        if wide:
            splits = (
                "validation",
                "test",
            )

        else:
            splits = (
                _split(record[split_column])
                if split_column
                else filename_split,
            )

        for split in splits:

            # Never silently treat unknown scores
            # as validation results.
            if split is None:
                continue

            prefix = (
                split
                if wide
                else ""
            )

            values = {}

            for name in METRICS:

                column = _col(
                    frame,
                    name,
                    prefix,
                )

                raw = (
                    record[column]
                    if column is not None
                    else None
                )

                numeric = pd.to_numeric(
                    pd.Series([raw]),
                    errors="coerce",
                ).iloc[0]

                values[name] = (
                    float(numeric)
                    if pd.notna(numeric)
                    else None
                )

            if all(
                value is None
                for value in values.values()
            ):
                continue

            output.append({
                "Task": _task(target, values),
                "Target": target,
                "Model": model,
                "Split": split,
                **{
                    DISPLAY[name]: values[name]
                    for name in METRICS
                },
                "Source": source,
            })

    return output


def _read_file(
    path: Path,
) -> pd.DataFrame:
    """Read a supported metrics file."""

    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)

    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)

    with path.open(
        encoding="utf-8"
    ) as file:
        data = json.load(file)

    if isinstance(data, list):
        return pd.DataFrame(data)

    if isinstance(data, dict):

        for key in (
            "metrics",
            "results",
            "scores",
            "evaluation",
            "validation_metrics",
            "test_metrics",
        ):

            value = data.get(key)

            if isinstance(value, list):
                return pd.DataFrame(value)

        if any(
            _key(key) in FIELDS["target"]
            for key in data
        ):
            return pd.DataFrame([data])

    return pd.DataFrame()


def load_saved_metrics(
    root: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read available evaluation metrics without loading models."""

    root = Path(root)

    if not root.is_dir():
        raise FileNotFoundError(
            f"ML metrics folder not found: {root}"
        )

    rows = []
    audit = []

    for path in sorted(
        root.rglob("*")
    ):

        if not path.is_file():
            continue

        if path.suffix.lower() not in (
            ".csv",
            ".parquet",
            ".json",
        ):
            continue

        relative = path.relative_to(root)

        # Never mix pilot results into the full run.
        if any(
            part.casefold() == "pilot"
            for part in relative.parts
        ):
            continue

        filename = str(relative)

        if path.stat().st_size > 20_000_000:

            audit.append({
                "File": filename,
                "Status": "Skipped: over 20 MB",
            })

            continue

        try:

            frame = _read_file(path)

            extracted = _rows(
                frame,
                filename,
            )

            rows.extend(extracted)

            audit.append({
                "File": filename,
                "Status": (
                    f"{len(extracted)} metric rows"
                    if extracted
                    else "Not a recognized metric table"
                ),
            })

        except (
            ValueError,
            OSError,
            ImportError,
            TypeError,
            KeyError,
            json.JSONDecodeError,
        ) as exc:

            audit.append({
                "File": filename,
                "Status": (
                    f"Unreadable: "
                    f"{type(exc).__name__}: {exc}"
                ),
            })

    metrics = pd.DataFrame(
        rows,
        columns=OUT_COLUMNS,
    )

    if not metrics.empty:
        metrics = (
            metrics.drop_duplicates()
            .reset_index(drop=True)
        )

    audit_frame = pd.DataFrame(
        audit,
        columns=["File", "Status"],
    )

    return metrics, audit_frame


def rank_validation_models(
    metrics: pd.DataFrame,
    target: str,
) -> tuple[pd.DataFrame, str | None]:
    """Rank models using validation metrics only."""

    rows = metrics.loc[
        metrics["Target"].eq(target)
        & metrics["Split"].eq("validation")
    ].copy()

    if rows.empty:
        return rows, None

    task = rows["Task"].iloc[0]

    if task == "Regression":

        primary = (
            "MAE"
            if rows["MAE"].notna().any()
            else "RMSE"
        )

        ascending = True

    elif rows["Brier"].notna().any():

        primary = "Brier"
        ascending = True

    elif rows["ROC AUC"].notna().any():

        primary = "ROC AUC"
        ascending = False

    else:
        return rows, None

    rows = rows.loc[
        rows[primary].notna()
    ].copy()

    if rows.empty:
        return rows, None

    # Reject conflicting validation scores from
    # different evaluation runs.
    for model, group in rows.groupby("Model"):

        if group[primary].nunique() > 1:
            raise ValueError(
                f"Conflicting validation {primary} "
                f"results for {model} on {target}. "
                "Choose a single evaluation run folder."
            )

    rows = rows.drop_duplicates(
        subset=[
            "Task",
            "Target",
            "Model",
            "Split",
            primary,
        ]
    )

    rows = rows.sort_values(
        [primary, "Model"],
        ascending=[ascending, True],
        kind="stable",
    )

    baseline_pattern = (
        r"median|baseline|prevalence|historical.mean"
    )

    trained = rows.loc[
        ~rows["Model"]
        .str.casefold()
        .str.contains(
            baseline_pattern,
            regex=True,
        )
    ]

    winner = (
        str(trained.iloc[0]["Model"])
        if not trained.empty
        else None
    )

    return rows, winner


def render_ml_comparison(
    root: str | Path,
) -> None:
    """Display saved ML model comparisons in Streamlit."""

    import streamlit as st

    st.subheader(
        "63-session ML model comparison"
    )

    st.caption(
        "Saved evaluation results only. "
        "No retraining, live inference, or "
        "test-based model selection."
    )

    try:

        metrics, audit = load_saved_metrics(
            root
        )

    except FileNotFoundError as exc:

        st.warning(str(exc))
        return

    with st.expander(
        "Saved artifacts and import audit"
    ):

        st.dataframe(
            audit,
            hide_index=True,
            use_container_width=True,
        )

    if metrics.empty:

        st.warning(
            "No supported model metric tables found. "
            "Open the audit to inspect the saved files. "
            "The loader expects CSV, Parquet or JSON "
            "metrics with a target, model and identified "
            "validation or test split."
        )

        return

    for task in (
        "Regression",
        "Classification",
    ):

        subset = metrics.loc[
            metrics["Task"].eq(task)
        ]

        st.markdown(
            f"#### {task}"
        )

        if subset.empty:

            st.info(
                f"No {task.lower()} evaluation metrics "
                "found in this folder."
            )

            continue

        targets = sorted(
            subset["Target"].unique()
        )

        target = st.selectbox(
            f"{task} prediction target",
            targets,
            key=(
                f"ml_comparison_"
                f"{task.lower()}_target"
            ),
        )

        try:

            ranking, winner = rank_validation_models(
                metrics,
                target,
            )

        except ValueError as exc:

            st.error(str(exc))
            continue

        if ranking.empty:

            st.warning(
                "No usable validation metrics "
                "for this target."
            )

            continue

        if task == "Regression":

            primary = (
                "MAE"
                if ranking["MAE"].notna().any()
                else "RMSE"
            )

        else:

            primary = (
                "Brier"
                if ranking["Brier"].notna().any()
                else "ROC AUC"
            )

        if winner:

            st.write(
                f"**Validation leader: {winner}** "
                f"(metric: {primary}). "
                "This is not yet a live-hedge recommendation."
            )

        else:

            st.info(
                "No trained-model leader identified. "
                "The available rows may be baselines only."
            )

        st.dataframe(
            ranking.drop(columns=["Task"]),
            hide_index=True,
            use_container_width=True,
        )

        if ranking[primary].notna().any():

            st.bar_chart(
                ranking.set_index("Model")[
                    [primary]
                ]
            )

        test_rows = metrics.loc[
            metrics["Target"].eq(target)
            & metrics["Split"].eq("test")
        ]

        with st.expander(
            "Recorded test metrics — not used for selection"
        ):

            if test_rows.empty:

                st.info(
                    "No test metrics found for this target."
                )

            else:

                st.dataframe(
                    test_rows.drop(
                        columns=["Task"]
                    ),
                    hide_index=True,
                    use_container_width=True,
                )

                st.caption(
                    "Check whether the test period was "
                    "truly untouched. An exploratory or "
                    "reused test window is not independent "
                    "final validation."
                )
