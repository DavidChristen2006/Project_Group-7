
"""Compare nine 63-session equity regression families using frozen research splits.

All models are fit on training observations only.
Selection uses validation MAE.

The previously viewed test window is exploratory,
not a clean new holdout.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import (
    ElasticNet,
    Lasso,
    LinearRegression,
    Ridge,
)
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data_pipeline.train_equity_63d_baseline import (
    FEATURES,
    KEY,
    TARGETS,
    get_dataset,
    split_data,
)


SEED = 42

ROOT = Path(__file__).resolve().parents[1]

BASELINE_DIR = (
    ROOT
    / "data/processed/ml_baselines/equity_63d_v1"
)

OUTPUT_DIR = (
    ROOT
    / "data/processed/ml_baselines/equity_63d_suite_v1"
)

MODEL_FAMILIES = (
    "train_median",
    "ols",
    "ridge",
    "lasso",
    "elastic_net",
    "pcr",
    "pls",
    "random_forest",
    "hist_gradient_boosting",
    "xgboost",
)


def check_frozen_keys(
    split: pd.DataFrame,
    label: str,
) -> None:
    """Ensure that the model sample matches the original baseline."""

    path = (
        BASELINE_DIR
        / f"{label}_sample_keys.csv"
    )

    if not path.is_file():
        raise FileNotFoundError(path)

    columns = KEY + ["target_exit_date"]

    expected = pd.read_csv(
        path,
        usecols=columns,
    )

    actual = split[columns].copy()

    for frame in (expected, actual):

        for date_column in (
            "Decision Date",
            "target_exit_date",
        ):
            frame[date_column] = pd.to_datetime(
                frame[date_column],
                errors="raise",
            ).dt.normalize()

        frame["Underlying RIC"] = (
            frame["Underlying RIC"].astype(str)
        )

        if (
            frame.duplicated(KEY).any()
            or frame[KEY].isna().any().any()
        ):
            raise ValueError(
                f"Invalid keys in {label}"
            )

    expected = expected.sort_values(
        KEY
    ).reset_index(drop=True)

    actual = actual.sort_values(
        KEY
    ).reset_index(drop=True)

    pd.testing.assert_frame_equal(
        actual,
        expected,
        check_dtype=False,
        obj=f"frozen {label} rows",
    )


def pilot_sample(
    frame: pd.DataFrame,
    per_date: int = 65,
) -> pd.DataFrame:
    """Sample every decision date reproducibly for a fast pilot."""

    rng = np.random.default_rng(SEED)

    indices = []

    for _, group in frame.groupby(
        "Decision Date",
        sort=True,
    ):
        chosen = rng.choice(
            group.index.to_numpy(),
            size=min(per_date, len(group)),
            replace=False,
        )

        indices.extend(
            chosen.tolist()
        )

    return (
        frame.loc[indices]
        .sort_values(KEY)
        .reset_index(drop=True)
    )


def linear_pipeline(
    regressor: object,
) -> Pipeline:
    """Impute and standardize features using training data only."""

    return Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median"),
        ),
        (
            "scaler",
            StandardScaler(),
        ),
        (
            "regressor",
            regressor,
        ),
    ])


def make_models(
    quick: bool,
) -> dict[str, object]:
    """Build the complete regression model comparison."""

    try:
        from xgboost import XGBRegressor

    except ImportError as exc:
        raise ImportError(
            "xgboost is required for this comparison. "
            "Install it with: py -m pip install xgboost"
        ) from exc

    n_trees = 55 if quick else 160
    n_boost = 65 if quick else 160

    return {
        "train_median": DummyRegressor(
            strategy="median",
        ),

        "ols": linear_pipeline(
            LinearRegression(),
        ),

        "ridge": linear_pipeline(
            Ridge(alpha=10.0),
        ),

        "lasso": linear_pipeline(
            Lasso(
                alpha=0.0005,
                max_iter=15000,
                random_state=SEED,
            ),
        ),

        "elastic_net": linear_pipeline(
            ElasticNet(
                alpha=0.001,
                l1_ratio=0.5,
                max_iter=15000,
                random_state=SEED,
            ),
        ),

        "pcr": Pipeline([
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "pca",
                PCA(
                    n_components=10,
                    svd_solver="full",
                ),
            ),
            (
                "regressor",
                LinearRegression(),
            ),
        ]),

        "pls": Pipeline([
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "regressor",
                PLSRegression(
                    n_components=8,
                    scale=False,
                ),
            ),
        ]),

        "random_forest": Pipeline([
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=n_trees,
                    max_features=0.8,
                    min_samples_leaf=(
                        10 if quick else 25
                    ),
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
                "regressor",
                HistGradientBoostingRegressor(
                    max_iter=n_boost,
                    learning_rate=0.05,
                    max_leaf_nodes=15,
                    min_samples_leaf=(
                        20 if quick else 70
                    ),
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
                "regressor",
                XGBRegressor(
                    n_estimators=n_boost,
                    objective="reg:squarederror",
                    eval_metric="mae",
                    max_depth=3,
                    learning_rate=0.05,
                    subsample=0.85,
                    colsample_bytree=0.9,
                    min_child_weight=(
                        10 if quick else 35
                    ),
                    reg_lambda=5.0,
                    tree_method="hist",
                    n_jobs=4,
                    random_state=SEED,
                ),
            ),
        ]),
    }


def predict_valid(
    model: object,
    x: pd.DataFrame,
    target: str,
) -> np.ndarray:
    """Validate prediction shape and enforce target bounds."""

    raw = np.asarray(
        model.predict(x),
        dtype=float,
    ).reshape(-1)

    if (
        len(raw) != len(x)
        or not np.isfinite(raw).all()
    ):
        raise ValueError(
            f"Invalid predictions for {target}"
        )

    if target == "target_max_drawdown_63d":
        return np.minimum(raw, 0.0)

    return np.maximum(raw, 0.0)


def scores(
    y: np.ndarray,
    pred: np.ndarray,
) -> dict:
    """Evaluate regression predictions."""

    return {
        "MAE": float(
            mean_absolute_error(y, pred)
        ),
        "RMSE": float(
            np.sqrt(
                mean_squared_error(y, pred)
            )
        ),
        "R2": float(
            r2_score(y, pred)
        ),
        "Bias": float(
            np.mean(pred - y)
        ),
    }


def evaluate_target(
    target: str,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame | None,
    quick: bool,
    output: Path,
) -> tuple[list[dict], dict, list[dict]]:
    """Fit all models, select on validation, then evaluate the winner."""

    x_train = train[FEATURES]
    x_val = validation[FEATURES]

    y_train = train[target].to_numpy(
        dtype=float
    )

    y_val = validation[target].to_numpy(
        dtype=float
    )

    models = make_models(quick)

    comparison = []
    fitted_models = {}
    val_results = {}

    for name in MODEL_FAMILIES:

        model = models[name]

        print(
            f"  Fitting {name}...",
            flush=True,
        )

        try:
            model.fit(
                x_train,
                y_train,
            )

            val_pred = predict_valid(
                model,
                x_val,
                target,
            )

        except Exception as exc:
            raise RuntimeError(
                f"{target}, model={name}: {exc}"
            ) from exc

        metric = scores(
            y_val,
            val_pred,
        )

        comparison.append({
            "Target": target,
            "Model": name,
            "Validation Rows": len(validation),
            **metric,
        })

        val_results[name] = metric
        fitted_models[name] = model

        print(
            f"    validation MAE={metric['MAE']:.6f}, "
            f"RMSE={metric['RMSE']:.6f}",
            flush=True,
        )

    baseline_mae = val_results[
        "train_median"
    ]["MAE"]

    for record in comparison:
        record[
            "MAE Improvement vs Median (%)"
        ] = (
            100.0
            * (baseline_mae - record["MAE"])
            / baseline_mae
            if baseline_mae > 0
            else np.nan
        )

    # No test information enters model selection.
    chosen_name = min(
        MODEL_FAMILIES,
        key=lambda name: (
            val_results[name]["MAE"],
            val_results[name]["RMSE"],
            name,
        ),
    )

    print(
        f"  SELECTED on validation: {chosen_name}",
        flush=True,
    )

    selection = {
        "Target": target,
        "Selected Model": chosen_name,
        "Validation MAE": (
            val_results[chosen_name]["MAE"]
        ),
        "Validation Median MAE": baseline_mae,
        "Validation Rows": len(validation),
        "Test Status": (
            "NOT_RUN_PILOT"
            if quick
            else "REUSED_EXPLORATORY_WINDOW"
        ),
    }

    test_metrics = []

    if test is not None:

        x_test = test[FEATURES]

        y_test = test[target].to_numpy(
            dtype=float
        )

        selected_pred = predict_valid(
            fitted_models[chosen_name],
            x_test,
            target,
        )

        median_pred = predict_valid(
            fitted_models["train_median"],
            x_test,
            target,
        )

        for name, pred in (
            (chosen_name, selected_pred),
            ("train_median_reference", median_pred),
        ):
            test_metrics.append({
                "Target": target,
                "Model": name,
                "Test Rows": len(test),
                **scores(y_test, pred),
            })

        predictions = test[
            KEY + ["target_exit_date"]
        ].copy()

        predictions["Target"] = target
        predictions["Selected Model"] = chosen_name
        predictions["Actual"] = y_test
        predictions["Predicted"] = selected_pred
        predictions["Train Median"] = median_pred

        predictions.to_csv(
            output
            / f"exploratory_predictions_{target}.csv",
            index=False,
        )

        by_date = []

        for date, sub in predictions.groupby(
            "Decision Date",
            sort=True,
        ):
            metrics = scores(
                sub["Actual"].to_numpy(
                    dtype=float
                ),
                sub["Predicted"].to_numpy(
                    dtype=float
                ),
            )

            by_date.append({
                "Target": target,
                "Decision Date": date,
                "Rows": len(sub),
                **metrics,
            })

        pd.DataFrame(
            by_date
        ).to_csv(
            output
            / f"exploratory_by_month_{target}.csv",
            index=False,
        )

        joblib.dump(
            fitted_models[chosen_name],
            output
            / "selected_models"
            / f"{target}.joblib",
        )

    return (
        comparison,
        selection,
        test_metrics,
    )


def main() -> None:

    """Run the module entry point and its workflow.
    """
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--quick",
        action="store_true",
        help=(
            "Small stratified pilot, validation only; "
            "no held-out predictions"
        ),
    )

    args = parser.parse_args()

    if (
        len(FEATURES) != 30
        or len(TARGETS) != 3
    ):
        raise ValueError(
            "Feature/target contract changed unexpectedly"
        )

    panel = get_dataset(ROOT)

    train, val, test = split_data(
        panel
    )

    for name, split in (
        ("train", train),
        ("validation", val),
        ("test", test),
    ):
        check_frozen_keys(
            split,
            name,
        )

    if train[FEATURES].isna().all().any():
        raise ValueError(
            "Some predictors are entirely missing in training"
        )

    output = (
        OUTPUT_DIR / "pilot"
        if args.quick
        else OUTPUT_DIR / "full"
    )

    (output / "selected_models").mkdir(
        parents=True,
        exist_ok=True,
    )

    if args.quick:
        train = pilot_sample(train)
        val = pilot_sample(val)
        test = None

    print(
        "\n--- NINE-FAMILY 63-DAY "
        "REGRESSION BENCHMARK ---"
    )

    print(
        "Mode:",
        (
            "SMALL VALIDATION PILOT"
            if args.quick
            else "FULL EXPLORATORY COMPARISON"
        ),
    )

    print("Training rows:", len(train))
    print("Validation rows:", len(val))

    print(
        "Test rows:",
        0 if test is None else len(test),
    )

    print(
        "Feature columns:",
        len(FEATURES),
    )

    print(
        "Models plus median reference:",
        len(MODEL_FAMILIES),
    )

    print(
        "Note: fixed hyperparameters, "
        "not a hyperparameter search."
    )

    all_validation = []
    all_selected = []
    all_tests = []

    for target in TARGETS:

        print(
            f"\nTARGET: {target}",
            flush=True,
        )

        comparison, selection, tests = evaluate_target(
            target,
            train,
            val,
            test,
            args.quick,
            output,
        )

        all_validation.extend(
            comparison
        )

        all_selected.append(
            selection
        )

        all_tests.extend(
            tests
        )

        # Persist each completed target.
        pd.DataFrame(
            all_validation
        ).to_csv(
            output / "validation_all_models.csv",
            index=False,
        )

        pd.DataFrame(
            all_selected
        ).to_csv(
            output / "validation_selected_models.csv",
            index=False,
        )

        if all_tests:
            pd.DataFrame(
                all_tests
            ).to_csv(
                output
                / "exploratory_selected_vs_median.csv",
                index=False,
            )

    specification = {
        "features": FEATURES,
        "targets": TARGETS,
        "model_families": MODEL_FAMILIES,
        "training_rows": len(train),
        "validation_rows": len(val),
        "test_rows": (
            0 if test is None else len(test)
        ),
        "mode": (
            "PILOT"
            if args.quick
            else "EXPLORATORY_REUSED_TEST"
        ),
        "selection": (
            "Minimum validation MAE; "
            "RMSE then model name tie breaks"
        ),
        "hyperparameters": (
            "Fixed predefined specifications, "
            "not tuned via test"
        ),
        "sklearn_version": sklearn.__version__,
        "cautions": [
            "The 2022-2023 test period has already been inspected.",
            "Incomplete terminal paths are excluded.",
            "Reconstructed 2019-2021 membership is excluded.",
            "Adjusted prices require historical point-in-time validation.",
            "Cross-sectional observations share market dates and overlapping labels.",
            "No option hedging costs or strategy P&L are modeled.",
        ],
    }

    (
        output / "run_specification.json"
    ).write_text(
        json.dumps(
            specification,
            indent=2,
        ),
        encoding="utf-8",
    )

    validation_table = pd.DataFrame(
        all_validation
    )

    print(
        "\n--- VALIDATION: "
        "RANKED WITHIN EACH TARGET ---"
    )

    print(
        validation_table.sort_values(
            ["Target", "MAE", "RMSE"]
        )[
            [
                "Target",
                "Model",
                "MAE",
                "RMSE",
                "R2",
                "MAE Improvement vs Median (%)",
            ]
        ].to_string(index=False)
    )

    print(
        "\n--- VALIDATION SELECTED ---"
    )

    print(
        pd.DataFrame(
            all_selected
        ).to_string(index=False)
    )

    if all_tests:

        print(
            "\n--- EXPLORATORY REUSED TEST: "
            "SELECTED VS MEDIAN ---"
        )

        print(
            pd.DataFrame(
                all_tests
            ).to_string(index=False)
        )

    print(
        "\nSaved:",
        output,
    )

    print(
        "No existing feature, outcome, or "
        "baseline model files were changed."
    )


if __name__ == "__main__":
    main()
