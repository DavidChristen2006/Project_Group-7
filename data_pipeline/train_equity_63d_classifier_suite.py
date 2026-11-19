
"""Compare 63-session loss-breach classifiers with a fixed monthly alert budget.

Research-only: the 2022-2023 test period has been examined previously.
No LSEG calls, no input-data edits, no use of test labels for selection.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data_pipeline.train_equity_63d_baseline import (
    FEATURES,
    KEY,
    get_dataset,
    split_data,
)
from data_pipeline.train_equity_63d_model_suite import (
    check_frozen_keys,
)


ROOT = Path(__file__).resolve().parents[1]

OUTPUT = (
    ROOT
    / "data/processed/ml_baselines"
    / "equity_63d_classifier_suite_v1"
)

LOSS_TARGET = "target_max_loss_from_entry_63d"
THRESHOLDS = (0.10, 0.15, 0.20)
ALERT_BUDGET = 0.30
RISK_HEURISTIC = "realized_vol_252d"
SEED = 42


def make_models() -> dict[str, object]:
    """Construct five classifiers with fixed parameters."""

    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise ImportError(
            "Install XGBoost with: py -m pip install xgboost"
        ) from exc

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
                    random_state=SEED,
                ),
            ),
        ]),

        "random_forest": Pipeline([
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=180,
                    min_samples_leaf=25,
                    max_features=0.8,
                    n_jobs=4,
                    random_state=SEED,
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
                    random_state=SEED,
                ),
            ),
        ]),

        "xgboost": Pipeline([
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "classifier",
                XGBClassifier(
                    objective="binary:logistic",
                    eval_metric="logloss",
                    n_estimators=160,
                    max_depth=3,
                    learning_rate=0.05,
                    subsample=0.85,
                    colsample_bytree=0.9,
                    min_child_weight=35,
                    reg_lambda=5.0,
                    tree_method="hist",
                    n_jobs=4,
                    random_state=SEED,
                ),
            ),
        ]),
    }


def probability(
    model: object,
    x: pd.DataFrame,
) -> np.ndarray:
    """Return valid positive-class probabilities."""

    classes = list(model.classes_)

    if classes != [0, 1]:
        raise ValueError(
            f"Unexpected classifier classes: {classes}"
        )

    p = np.asarray(
        model.predict_proba(x)[:, 1],
        dtype=float,
    )

    if (
        len(p) != len(x)
        or not np.isfinite(p).all()
        or ((p < 0) | (p > 1)).any()
    ):
        raise ValueError(
            "Invalid classifier probabilities"
        )

    return p


def scoring(
    y: np.ndarray,
    p: np.ndarray,
) -> dict:
    """Calculate probability metrics."""

    result = {
        "Rows": len(y),
        "Events": int(y.sum()),
        "Prevalence": float(y.mean()),
        "Brier": float(
            brier_score_loss(y, p)
        ),
    }

    if len(np.unique(y)) == 2:
        result["Average Precision"] = float(
            average_precision_score(y, p)
        )

        result["ROC AUC"] = float(
            roc_auc_score(y, p)
        )

    else:
        result["Average Precision"] = np.nan
        result["ROC AUC"] = np.nan

    return result


def top_monthly_fraction(
    frame: pd.DataFrame,
    scores: np.ndarray,
    fraction: float = ALERT_BUDGET,
) -> np.ndarray:
    """Select highest-risk stocks separately on each decision date.

    Ties are broken using the RIC to ensure reproducible results.
    No outcome labels are used to select alerts.
    """

    scores = np.asarray(
        scores,
        dtype=float,
    )

    if len(frame) != len(scores):
        raise ValueError(
            "Score and observation counts differ"
        )

    if not np.isfinite(scores).all():
        raise ValueError(
            "Nonfinite risk scores"
        )

    if not 0 <= fraction <= 1:
        raise ValueError(
            "Invalid alert fraction"
        )

    work = frame[KEY].reset_index(
        drop=True
    )

    flags = np.zeros(
        len(work),
        dtype=bool,
    )

    for _, positions in work.groupby(
        "Decision Date",
        sort=True,
    ).indices.items():

        positions = np.asarray(
            positions,
            dtype=int,
        )

        count = int(
            np.floor(fraction * len(positions))
        )

        if count == 0:
            continue

        rics = (
            work.iloc[positions]["Underlying RIC"]
            .astype(str)
            .to_numpy()
        )

        order = np.lexsort(
            (
                rics,
                -scores[positions],
            )
        )

        flags[
            positions[order[:count]]
        ] = True

    return flags


def alert_scores(
    y: np.ndarray,
    flags: np.ndarray,
) -> dict:
    """Calculate event detection at the selected alert budget."""

    y = np.asarray(
        y,
        dtype=int,
    )

    flags = np.asarray(
        flags,
        dtype=bool,
    )

    tp = int(
        ((y == 1) & flags).sum()
    )

    fp = int(
        ((y == 0) & flags).sum()
    )

    fn = int(
        ((y == 1) & ~flags).sum()
    )

    return {
        "Alert Rate": float(flags.mean()),
        "Alerts": int(flags.sum()),
        "True Positives": tp,
        "False Positives": fp,
        "Missed Events": fn,
        "Recall": (
            tp / (tp + fn)
            if tp + fn
            else np.nan
        ),
        "Precision": (
            tp / (tp + fp)
            if tp + fp
            else np.nan
        ),
    }


def validate_data(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
) -> None:
    """Verify the original samples and predictor availability."""

    for name, frame in (
        ("train", train),
        ("validation", val),
        ("test", test),
    ):
        check_frozen_keys(
            frame,
            name,
        )

        if frame[FEATURES].isna().all(
            axis=1
        ).any():
            raise ValueError(
                f"{name}: observations with no features"
            )

        losses = pd.to_numeric(
            frame[LOSS_TARGET],
            errors="coerce",
        )

        if not np.isfinite(
            losses.to_numpy(dtype=float)
        ).all():
            raise ValueError(
                f"{name}: invalid loss targets"
            )

        if (losses < 0).any():
            raise ValueError(
                f"{name}: negative loss target"
            )

        if RISK_HEURISTIC not in frame.columns:
            raise ValueError(
                f"{name}: missing {RISK_HEURISTIC}"
            )

        historical_risk = frame[
            RISK_HEURISTIC
        ].to_numpy(dtype=float)

        if not np.isfinite(
            historical_risk
        ).all():
            raise ValueError(
                f"{name}: incomplete historical risk scores"
            )


def main() -> None:

    panel = get_dataset(ROOT)

    train, val, test = split_data(
        panel
    )

    validate_data(
        train,
        val,
        test,
    )

    OUTPUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    (OUTPUT / "models").mkdir(
        exist_ok=True
    )

    validation_rows = []
    selected_rows = []
    test_rows = []
    monthly_rows = []
    prediction_frames = []

    print(
        "\n--- CLASSIFICATION MODEL SUITE: "
        "FIXED TOP-30% PER MONTH ---"
    )

    print(
        "Training rows:",
        len(train),
    )

    print(
        "Validation rows:",
        len(val),
    )

    print(
        "Exploratory test rows:",
        len(test),
    )

    print(
        "Feature count:",
        len(FEATURES),
    )

    print(
        "Classification targets:",
        len(THRESHOLDS),
    )

    print(
        "Alert budget:",
        f"{ALERT_BUDGET:.0%}",
    )

    print(
        "Test period already examined: EXPLORATORY ONLY"
    )

    for threshold in THRESHOLDS:

        pct = int(
            round(threshold * 100)
        )

        target_name = (
            f"loss_at_least_{pct}pct_63d"
        )

        y_train = (
            train[LOSS_TARGET].to_numpy(dtype=float)
            >= threshold
        ).astype(int)

        y_val = (
            val[LOSS_TARGET].to_numpy(dtype=float)
            >= threshold
        ).astype(int)

        y_test = (
            test[LOSS_TARGET].to_numpy(dtype=float)
            >= threshold
        ).astype(int)

        if any(
            len(np.unique(y)) != 2
            for y in (y_train, y_val, y_test)
        ):
            raise ValueError(
                f"{target_name}: both classes required"
            )

        models = make_models()
        fitted = {}
        val_stats = {}

        print(
            f"\nTARGET: {target_name}",
            flush=True,
        )

        for name, model in models.items():

            print(
                f"  Fitting {name}...",
                flush=True,
            )

            model.fit(
                train[FEATURES],
                y_train,
            )

            p_val = probability(
                model,
                val[FEATURES],
            )

            metrics = scoring(
                y_val,
                p_val,
            )

            val_stats[name] = metrics
            fitted[name] = model

            validation_rows.append({
                "Target": target_name,
                "Model": name,
                **metrics,
            })

            print(
                f"    AP={metrics['Average Precision']:.5f}, "
                f"Brier={metrics['Brier']:.5f}",
                flush=True,
            )

        # Select exclusively on validation AP.
        # Brier and model name are deterministic tie-breaks.
        chosen_name = min(
            models,
            key=lambda name: (
                -val_stats[name]["Average Precision"],
                val_stats[name]["Brier"],
                name,
            ),
        )

        chosen = fitted[
            chosen_name
        ]

        p_val_selected = probability(
            chosen,
            val[FEATURES],
        )

        p_test = probability(
            chosen,
            test[FEATURES],
        )

        p_prior = probability(
            fitted["train_prevalence"],
            test[FEATURES],
        )

        selected_test = scoring(
            y_test,
            p_test,
        )

        prior_test = scoring(
            y_test,
            p_prior,
        )

        val_flags = top_monthly_fraction(
            val,
            p_val_selected,
        )

        test_flags = top_monthly_fraction(
            test,
            p_test,
        )

        heuristic_scores = (
            test[RISK_HEURISTIC]
            .to_numpy(dtype=float)
        )

        heuristic_flags = top_monthly_fraction(
            test,
            heuristic_scores,
        )

        val_alert = alert_scores(
            y_val,
            val_flags,
        )

        model_alert = alert_scores(
            y_test,
            test_flags,
        )

        heuristic_alert = alert_scores(
            y_test,
            heuristic_flags,
        )

        selected_rows.append({
            "Target": target_name,
            "Selected Model": chosen_name,
            "Validation AP": (
                val_stats[chosen_name]["Average Precision"]
            ),
            "Validation Brier": (
                val_stats[chosen_name]["Brier"]
            ),
            "Validation Prior Brier": (
                val_stats["train_prevalence"]["Brier"]
            ),
            "Validation Top30 Recall": (
                val_alert["Recall"]
            ),
            "Validation Top30 Precision": (
                val_alert["Precision"]
            ),
        })

        heuristic_metrics = {
            "Rows": len(test),
            "Events": int(y_test.sum()),
            "Prevalence": float(y_test.mean()),
            "Average Precision": float(
                average_precision_score(
                    y_test,
                    heuristic_scores,
                )
            ),
            "ROC AUC": float(
                roc_auc_score(
                    y_test,
                    heuristic_scores,
                )
            ),
            "Brier": np.nan,
        }

        for name, prob_metrics, alert_metrics in (
            (
                "selected_model_top30",
                selected_test,
                model_alert,
            ),
            (
                "historical_volatility_top30",
                heuristic_metrics,
                heuristic_alert,
            ),
        ):
            test_rows.append({
                "Target": target_name,
                "Evaluation": name,
                **prob_metrics,
                **alert_metrics,
                "Test Prior Brier": prior_test["Brier"],
            })

        predictions = (
            test[KEY + ["target_exit_date"]]
            .copy()
            .reset_index(drop=True)
        )

        predictions["Target"] = target_name
        predictions["Actual Breach"] = y_test
        predictions["Selected Model"] = chosen_name

        predictions["Predicted Probability"] = (
            p_test
        )

        predictions["Training Prevalence Probability"] = (
            p_prior
        )

        predictions["Historical Volatility Score"] = (
            heuristic_scores
        )

        predictions["Top30 Model Alert"] = (
            test_flags
        )

        predictions["Top30 Historical Vol Alert"] = (
            heuristic_flags
        )

        prediction_frames.append(
            predictions
        )

        for date, positions in predictions.groupby(
            "Decision Date",
            sort=True,
        ).indices.items():

            indices = np.asarray(
                positions,
                dtype=int,
            )

            for name, flags in (
                ("selected_model_top30", test_flags),
                (
                    "historical_volatility_top30",
                    heuristic_flags,
                ),
            ):
                monthly_rows.append({
                    "Target": target_name,
                    "Decision Date": date,
                    "Evaluation": name,
                    "Rows": len(indices),
                    "Events": int(
                        y_test[indices].sum()
                    ),
                    **alert_scores(
                        y_test[indices],
                        flags[indices],
                    ),
                })

        joblib.dump(
            chosen,
            OUTPUT
            / "models"
            / f"{target_name}.joblib",
        )

        print(
            f"  SELECTED: {chosen_name} | "
            f"test AP={selected_test['Average Precision']:.4f} | "
            f"top30 recall={model_alert['Recall']:.3f} | "
            f"historical-volatility recall="
            f"{heuristic_alert['Recall']:.3f}",
            flush=True,
        )

    pd.DataFrame(
        validation_rows
    ).to_csv(
        OUTPUT / "validation_all_models.csv",
        index=False,
    )

    pd.DataFrame(
        selected_rows
    ).to_csv(
        OUTPUT / "validation_selected.csv",
        index=False,
    )

    pd.DataFrame(
        test_rows
    ).to_csv(
        OUTPUT / "exploratory_top30_test.csv",
        index=False,
    )

    pd.DataFrame(
        monthly_rows
    ).to_csv(
        OUTPUT / "exploratory_top30_by_month.csv",
        index=False,
    )

    all_predictions = pd.concat(
        prediction_frames,
        ignore_index=True,
    )

    all_predictions.to_csv(
        OUTPUT / "exploratory_test_predictions.csv",
        index=False,
    )

    # Check consistency of the three independently
    # trained probability models.
    wide = all_predictions.pivot(
        index=KEY,
        columns="Target",
        values="Predicted Probability",
    )

    p10 = wide["loss_at_least_10pct_63d"]
    p15 = wide["loss_at_least_15pct_63d"]
    p20 = wide["loss_at_least_20pct_63d"]

    violations = int(
        (
            (p20 > p15 + 1e-12)
            | (p15 > p10 + 1e-12)
        ).sum()
    )

    specification = {
        "features": FEATURES,
        "loss_target": LOSS_TARGET,
        "thresholds": list(THRESHOLDS),
        "model_families": list(make_models()),
        "selection": (
            "Validation average precision; "
            "Brier as tie-break"
        ),
        "alert_policy": (
            "Top floor(30% * monthly universe size) "
            "scores per decision date"
        ),
        "ordering_violations": violations,
        "test_status": (
            "REUSED_EXPLORATORY_NOT_FRESH_HOLDOUT"
        ),
        "limitations": [
            "Probabilities are not independently calibrated.",
            "Only 11 exploratory test decision dates.",
            "Forward outcome windows overlap.",
            "Terminal outcomes remain partially unresolved.",
            "Historical-volatility baseline supplies rankings, not probabilities.",
            "Separate classifiers may violate nested probability ordering.",
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
        "\n--- VALIDATION RANKING BY TARGET ---"
    )

    print(
        pd.DataFrame(
            validation_rows
        ).sort_values(
            ["Target", "Average Precision"],
            ascending=[True, False],
        )[
            [
                "Target",
                "Model",
                "Average Precision",
                "ROC AUC",
                "Brier",
            ]
        ].to_string(index=False)
    )

    print(
        "\n--- EXPLORATORY MONTHLY TOP-30% ALERT RESULTS ---"
    )

    print(
        pd.DataFrame(
            test_rows
        )[
            [
                "Target",
                "Evaluation",
                "Events",
                "Average Precision",
                "Brier",
                "Test Prior Brier",
                "Alert Rate",
                "Precision",
                "Recall",
                "Missed Events",
            ]
        ].to_string(index=False)
    )

    print(
        "\nNested probability-order violations:",
        violations,
        "/",
        len(wide),
    )

    print(
        "\nSaved:",
        OUTPUT,
    )

    print(
        "Exploratory only: test period has "
        "already informed model development."
    )


if __name__ == "__main__":
    main()
