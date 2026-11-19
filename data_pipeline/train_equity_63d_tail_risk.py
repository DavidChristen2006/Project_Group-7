
"""Train exploratory 63-day loss-breach probability classifiers.

No LSEG requests or modifications to existing prices, features, and targets.
This reuses an already inspected test period; results are exploratory.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Reuse the feature whitelist and chronological split
# from the existing 63-day regression benchmark.
from data_pipeline.train_equity_63d_baseline import (
    FEATURES,
    KEY,
    get_dataset,
    split_data,
)


ROOT = Path(__file__).resolve().parents[1]

BASELINE_DIR = (
    ROOT / "data/processed/ml_baselines/equity_63d_v1"
)

OUTPUT = (
    ROOT / "data/processed/ml_baselines/equity_63d_tail_v1"
)

LOSS_COLUMN = "target_max_loss_from_entry_63d"

LEVELS = (0.10, 0.15, 0.20)

VALIDATION_ALERT_BUDGET = 0.30

RANDOM_STATE = 42


def verify_frozen_split(
    split: pd.DataFrame,
    name: str,
) -> None:
    """Ensure sample keys match the completed regression benchmark."""

    path = BASELINE_DIR / f"{name}_sample_keys.csv"

    if not path.is_file():
        raise FileNotFoundError(path)

    old = pd.read_csv(
        path,
        usecols=KEY,
    )

    new = split[KEY].copy()

    for frame in (old, new):
        frame["Decision Date"] = pd.to_datetime(
            frame["Decision Date"],
            errors="raise",
        ).dt.normalize()

        frame["Underlying RIC"] = (
            frame["Underlying RIC"].astype(str)
        )

    def sorted_keys(frame):
        return frame.sort_values(
            KEY
        ).reset_index(drop=True)

    pd.testing.assert_frame_equal(
        sorted_keys(old),
        sorted_keys(new),
        check_dtype=False,
        obj=f"frozen {name} sample keys",
    )


def make_models() -> dict[str, object]:
    return {
        "train_prevalence": DummyClassifier(
            strategy="prior"
        ),

        "logistic_regression": Pipeline([
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "classifier",
                LogisticRegression(
                    C=0.3,
                    max_iter=4000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]),

        "hist_gradient_boosting": Pipeline([
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "classifier",
                HistGradientBoostingClassifier(
                    max_iter=160,
                    learning_rate=0.05,
                    max_leaf_nodes=15,
                    min_samples_leaf=70,
                    l2_regularization=1.0,
                    early_stopping=False,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]),
    }


def probabilities(
    model: object,
    x: pd.DataFrame,
) -> np.ndarray:

    classes = list(model.classes_)

    if classes != [0, 1]:
        raise ValueError(
            f"Unexpected classifier classes: {classes}"
        )

    probs = np.asarray(
        model.predict_proba(x)[:, 1],
        dtype=float,
    )

    if (
        not np.isfinite(probs).all()
        or np.any((probs < 0) | (probs > 1))
    ):
        raise ValueError(
            "Classifier returned invalid probabilities"
        )

    return probs


def probability_metrics(
    y: np.ndarray,
    probs: np.ndarray,
) -> dict:

    if not np.isin(y, [0, 1]).all():
        raise ValueError("Labels must be binary")

    return {
        "Observations": int(len(y)),
        "Events": int(y.sum()),
        "Prevalence": float(y.mean()),

        "Average Precision": (
            float(average_precision_score(y, probs))
            if y.sum()
            else np.nan
        ),

        "ROC AUC": (
            float(roc_auc_score(y, probs))
            if len(np.unique(y)) == 2
            else np.nan
        ),

        "Brier Score": float(
            brier_score_loss(y, probs)
        ),
    }


def alert_metrics(
    y: np.ndarray,
    probs: np.ndarray,
    cutoff: float,
) -> dict:

    alerted = probs >= cutoff

    tp = int(
        np.count_nonzero(
            alerted & (y == 1)
        )
    )

    fp = int(
        np.count_nonzero(
            alerted & (y == 0)
        )
    )

    fn = int(
        np.count_nonzero(
            ~alerted & (y == 1)
        )
    )

    return {
        "Alerts": int(alerted.sum()),
        "Alert Rate": float(alerted.mean()),
        "True Positives": tp,
        "False Positives": fp,
        "False Negatives": fn,

        "Recall": (
            float(tp / (tp + fn))
            if tp + fn
            else np.nan
        ),

        "Precision": (
            float(tp / (tp + fp))
            if tp + fp
            else np.nan
        ),
    }


def choose_validation_threshold(
    y: np.ndarray,
    probs: np.ndarray,
    alert_budget: float,
) -> tuple[float, dict]:
    """Maximize recall subject to the validation alert budget.

    Tied probability scores are kept together.
    """

    order = np.argsort(
        -probs,
        kind="stable",
    )

    sorted_prob = probs[order]
    sorted_y = y[order]

    ends = np.r_[
        np.flatnonzero(
            np.diff(sorted_prob) != 0
        ) + 1,
        len(y),
    ]

    cumulative_events = np.cumsum(
        sorted_y
    )

    events_at_end = cumulative_events[
        ends - 1
    ]

    max_alerts = int(
        np.floor(alert_budget * len(y))
    )

    eligible = ends <= max_alerts

    if not eligible.any():
        cutoff = float(
            np.nextafter(
                sorted_prob[0],
                np.inf,
            )
        )

        return cutoff, alert_metrics(
            y,
            probs,
            cutoff,
        )

    valid_ends = ends[eligible]
    valid_events = events_at_end[eligible]

    recall = (
        valid_events / int(y.sum())
    )

    precision = (
        valid_events / valid_ends
    )

    # Highest recall, then precision;
    # finally fewer alerts.
    ranking = np.lexsort(
        (
            valid_ends,
            -precision,
            -recall,
        )
    )

    selected_end = int(
        valid_ends[int(ranking[0])]
    )

    cutoff = float(
        sorted_prob[selected_end - 1]
    )

    return cutoff, alert_metrics(
        y,
        probs,
        cutoff,
    )


def main() -> None:

    panel = get_dataset(ROOT)

    train, validation, test = split_data(
        panel
    )

    for name, part in (
        ("train", train),
        ("validation", validation),
        ("test", test),
    ):
        verify_frozen_split(
            part,
            name,
        )

    if train[FEATURES].isna().all(axis=0).any():
        raise ValueError(
            "At least one feature is missing "
            "throughout training"
        )

    x_train = train[FEATURES]
    x_validation = validation[FEATURES]
    x_test = test[FEATURES]

    OUTPUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    (OUTPUT / "models").mkdir(
        exist_ok=True
    )

    validation_rows = []
    summary_rows = []
    prediction_frames = []
    monthly_rows = []

    print(
        "\n--- 63-DAY TAIL-LOSS "
        "PROBABILITY BENCHMARK ---"
    )

    print(
        "Conservative candidate rows:",
        len(panel),
    )

    print(
        "Predictor features:",
        len(FEATURES),
    )

    print(
        "Validation alert budget:",
        f"{VALIDATION_ALERT_BUDGET:.0%}",
    )

    print(
        "Warning: reused diagnostic test period, "
        "not a fresh holdout."
    )

    for loss_threshold in LEVELS:

        pct = int(
            round(loss_threshold * 100)
        )

        target = (
            f"loss_at_least_{pct}pct_63d"
        )

        y_train = (
            train[LOSS_COLUMN].to_numpy()
            >= loss_threshold
        ).astype(int)

        y_val = (
            validation[LOSS_COLUMN].to_numpy()
            >= loss_threshold
        ).astype(int)

        y_test = (
            test[LOSS_COLUMN].to_numpy()
            >= loss_threshold
        ).astype(int)

        if len(np.unique(y_train)) != 2:
            raise ValueError(
                f"Training labels for {target} "
                "have only one class"
            )

        if y_val.sum() == 0:
            raise ValueError(
                f"No validation positives for {target}"
            )

        models = make_models()

        val_scores = {}
        fitted = {}

        for name, model in models.items():

            model.fit(
                x_train,
                y_train,
            )

            probs = probabilities(
                model,
                x_validation,
            )

            scores = probability_metrics(
                y_val,
                probs,
            )

            val_scores[name] = scores
            fitted[name] = model

            validation_rows.append({
                "Target": target,
                "Model": name,
                **scores,
            })

        # Select using validation results only.
        chosen_name = min(
            models,
            key=lambda name: (
                -val_scores[name][
                    "Average Precision"
                ],
                val_scores[name][
                    "Brier Score"
                ],
                name,
            ),
        )

        chosen = fitted[chosen_name]

        val_prob = probabilities(
            chosen,
            x_validation,
        )

        cutoff, val_alert = (
            choose_validation_threshold(
                y_val,
                val_prob,
                VALIDATION_ALERT_BUDGET,
            )
        )

        test_prob = probabilities(
            chosen,
            x_test,
        )

        reference_prob = probabilities(
            fitted["train_prevalence"],
            x_test,
        )

        test_scores = probability_metrics(
            y_test,
            test_prob,
        )

        reference_scores = probability_metrics(
            y_test,
            reference_prob,
        )

        test_alert = alert_metrics(
            y_test,
            test_prob,
            cutoff,
        )

        summary_rows.append({
            "Target": target,
            "Selected Model": chosen_name,

            "Train Prevalence": float(
                y_train.mean()
            ),

            "Validation Prevalence": float(
                y_val.mean()
            ),

            "Test Prevalence": float(
                y_test.mean()
            ),

            "Validation AP": val_scores[
                chosen_name
            ]["Average Precision"],

            "Test AP": test_scores[
                "Average Precision"
            ],

            "Test ROC AUC": test_scores[
                "ROC AUC"
            ],

            "Test Brier": test_scores[
                "Brier Score"
            ],

            "Test Prior Brier": reference_scores[
                "Brier Score"
            ],

            "Validation Cutoff": cutoff,

            "Validation Alert Rate": val_alert[
                "Alert Rate"
            ],

            "Validation Recall": val_alert[
                "Recall"
            ],

            "Test Alert Rate": test_alert[
                "Alert Rate"
            ],

            "Test Precision": test_alert[
                "Precision"
            ],

            "Test Recall": test_alert[
                "Recall"
            ],

            "Test Positive Cases": int(
                y_test.sum()
            ),

            "Test Missed Cases": test_alert[
                "False Negatives"
            ],
        })

        prediction = test[
            KEY + ["target_exit_date"]
        ].copy()

        prediction["Target"] = target
        prediction["Actual Breach"] = y_test

        prediction[
            "Predicted Breach Probability"
        ] = test_prob

        prediction[
            "Training Prevalence Probability"
        ] = reference_prob

        prediction[
            "Validation Selected Cutoff"
        ] = cutoff

        prediction["Alert"] = (
            test_prob >= cutoff
        )

        prediction_frames.append(
            prediction
        )

        for date, indices in prediction.groupby(
            "Decision Date"
        ).indices.items():

            idx = np.asarray(
                indices,
                dtype=int,
            )

            monthly_rows.append({
                "Target": target,
                "Decision Date": date,

                **probability_metrics(
                    y_test[idx],
                    test_prob[idx],
                ),

                **alert_metrics(
                    y_test[idx],
                    test_prob[idx],
                    cutoff,
                ),
            })

        joblib.dump(
            chosen,
            OUTPUT / "models"
            / f"{target}.joblib",
        )

        print(
            f"{target}: "
            f"chosen={chosen_name}, "
            f"validation_AP="
            f"{val_scores[chosen_name]['Average Precision']:.4f}, "
            f"test_AP={test_scores['Average Precision']:.4f}, "
            f"test_recall={test_alert['Recall']:.3f}, "
            f"test_alert_rate={test_alert['Alert Rate']:.3f}"
        )

    pd.DataFrame(
        validation_rows
    ).to_csv(
        OUTPUT / "validation_model_comparison.csv",
        index=False,
    )

    pd.DataFrame(
        summary_rows
    ).to_csv(
        OUTPUT / "exploratory_test_summary.csv",
        index=False,
    )

    pd.concat(
        prediction_frames,
        ignore_index=True,
    ).to_csv(
        OUTPUT / "exploratory_test_predictions.csv",
        index=False,
    )

    pd.DataFrame(
        monthly_rows
    ).to_csv(
        OUTPUT / "exploratory_test_by_decision_date.csv",
        index=False,
    )

    specification = {
        "horizon_trading_sessions": 63,
        "target_source": LOSS_COLUMN,
        "thresholds": list(LEVELS),
        "features": FEATURES,

        "selection": (
            "Highest validation average precision; "
            "Brier tie-break"
        ),

        "alert_policy": (
            "Maximize validation recall subject to "
            "30% validation alert budget; "
            "fixed cutoff subsequently applied to test"
        ),

        "training_data": (
            "OBSERVED_SNAPSHOT_MATCH conservative sample"
        ),

        "split_sample_keys": (
            "Verified against previous baseline files"
        ),

        "test_status": (
            "EXPLORATORY_REUSED_TEST_NOT_INDEPENDENT_CONFIRMATION"
        ),

        "limitations": [
            "Consecutive 63-session outcome windows overlap.",
            "2019-2021 absent from conservative sample.",
            "Terminal security outcomes remain unresolved.",
            "Classifier probabilities not independently calibrated.",
            "No option hedging transaction costs or P&L.",
        ],
    }

    (
        OUTPUT / "experiment_specification.json"
    ).write_text(
        json.dumps(
            specification,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\n--- EXPLORATORY TEST SUMMARY ---"
    )

    print(
        pd.DataFrame(summary_rows)[
            [
                "Target",
                "Selected Model",
                "Test AP",
                "Test Brier",
                "Test Prior Brier",
                "Validation Cutoff",
                "Test Alert Rate",
                "Test Precision",
                "Test Recall",
                "Test Missed Cases",
            ]
        ].to_string(index=False)
    )

    print(
        "\nSaved:",
        OUTPUT,
    )

    print(
        "Exploratory diagnostics only; "
        "NOT production hedge triggers."
    )


if __name__ == "__main__":
    main()
