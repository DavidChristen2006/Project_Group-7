
"""Train leakage-controlled, exploratory 63-session equity risk baselines.

Input: local point-in-time equity feature and forward-outcome Parquet files.
Output: validation metrics, one-time held-out tests, and fitted models.
No LSEG requests. No modification to input files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


KEY = ["Decision Date", "Underlying RIC"]

TARGETS = [
    "target_max_loss_from_entry_63d",
    "target_max_drawdown_63d",
    "target_realized_vol_63d",
]

FEATURES = [
    f"{metric}_{period}d"
    for period in (21, 63, 126, 252)
    for metric in (
        "momentum",
        "realized_vol",
        "downside_dev",
        "max_drawdown",
        "daily_var95",
        "daily_es95",
    )
] + [
    f"{metric}_{period}d"
    for period in (63, 252)
    for metric in (
        "beta_spy",
        "corr_spy",
        "excess_momentum_spy",
    )
]

SPLITS = {
    "train_end": "2018-12-31",
    "validation_start": "2022-01-01",
    "validation_end": "2022-06-30",
    "test_start": "2022-10-01",
    "test_end": "2023-08-31",
}


def check_keys(data: pd.DataFrame, source: str) -> None:
    missing = [col for col in KEY if col not in data]

    if missing:
        raise ValueError(f"{source}: missing keys {missing}")

    if (
        data[KEY].isna().any().any()
        or data.duplicated(KEY).any()
        or data["Underlying RIC"].astype(str).str.len().eq(0).any()
    ):
        raise ValueError(f"{source}: invalid or duplicate keys")


def bool_column(series: pd.Series) -> pd.Series:
    values = series.astype("string").str.strip().str.lower()

    result = values.map({
        "true": True,
        "false": False,
        "1": True,
        "0": False,
    })

    if result.isna().any():
        raise ValueError("Missing or invalid model eligibility flags")

    return result.astype(bool)


def get_dataset(project: Path) -> pd.DataFrame:
    features_file = (
        project
        / "data/processed/research_features/equity_features_v1"
        / "point_in_time_equity_features.parquet"
    )

    outcome_file = (
        project
        / "data/processed/research_outcomes/equity_forward_63d_v1"
        / "point_in_time_equity_forward_outcomes.parquet"
    )

    for path in (features_file, outcome_file):
        if not path.is_file():
            raise FileNotFoundError(path)

    feature_data = pd.read_parquet(features_file)
    outcome_data = pd.read_parquet(outcome_file)

    for source, frame in (
        ("features", feature_data),
        ("outcomes", outcome_data),
    ):
        frame["Decision Date"] = pd.to_datetime(
            frame["Decision Date"],
            errors="raise",
        ).dt.normalize()

        frame["Underlying RIC"] = (
            frame["Underlying RIC"]
            .astype("string")
            .str.strip()
        )

        check_keys(frame, source)

    needed_features = KEY + FEATURES

    needed_outcome = KEY + [
        "target_exit_date",
        "target_status",
        "Conservative Observed-Snapshot Candidate",
        "Membership Source",
    ] + TARGETS

    for source, frame, required in (
        ("features", feature_data, needed_features),
        ("outcomes", outcome_data, needed_outcome),
    ):
        missing = [
            col for col in required
            if col not in frame.columns
        ]

        if missing:
            raise ValueError(
                f"{source}: missing columns {missing}"
            )

    panel = feature_data[needed_features].merge(
        outcome_data[needed_outcome],
        on=KEY,
        how="outer",
        validate="one_to_one",
        indicator=True,
    )

    if not panel["_merge"].eq("both").all():
        raise ValueError(
            "Feature and outcome row keys do not match"
        )

    panel = panel.drop(columns="_merge")

    panel["target_exit_date"] = pd.to_datetime(
        panel["target_exit_date"],
        errors="coerce",
    ).dt.normalize()

    panel["Conservative Observed-Snapshot Candidate"] = (
        bool_column(
            panel["Conservative Observed-Snapshot Candidate"]
        )
    )

    eligible = panel.loc[
        panel["Conservative Observed-Snapshot Candidate"]
        & panel["target_status"].eq(
            "COMPLETE_63D_PRICE_PATH"
        )
    ].copy()

    if (
        eligible.empty
        or eligible["target_exit_date"].isna().any()
    ):
        raise ValueError(
            "No valid eligible observations or missing label end dates"
        )

    if (
        eligible["target_exit_date"]
        <= eligible["Decision Date"]
    ).any():
        raise ValueError(
            "Forward labels must end after their decision date"
        )

    if not eligible["Membership Source"].eq(
        "OBSERVED_SNAPSHOT_MATCH"
    ).all():
        raise ValueError(
            "Conservative sample contains reconstructed membership"
        )

    for col in FEATURES + TARGETS:
        eligible[col] = pd.to_numeric(
            eligible[col],
            errors="coerce",
        )

        eligible[col] = eligible[col].replace(
            [np.inf, -np.inf],
            np.nan,
        )

    if eligible[TARGETS].isna().any().any():
        raise ValueError(
            "A complete forward target is nonfinite"
        )

    if (
        eligible["target_max_loss_from_entry_63d"]
        < -1e-12
    ).any():
        raise ValueError(
            "Max loss target cannot be negative"
        )

    if (
        eligible["target_max_drawdown_63d"]
        > 1e-12
    ).any():
        raise ValueError(
            "Max drawdown target cannot be positive"
        )

    return eligible.sort_values(
        KEY
    ).reset_index(drop=True)


def split_data(
    panel: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    d = panel["Decision Date"]

    train = panel.loc[
        d <= pd.Timestamp(SPLITS["train_end"])
    ].copy()

    val = panel.loc[
        d.between(
            SPLITS["validation_start"],
            SPLITS["validation_end"],
        )
    ].copy()

    test = panel.loc[
        d.between(
            SPLITS["test_start"],
            SPLITS["test_end"],
        )
    ].copy()

    for name, data in (
        ("train", train),
        ("validation", val),
        ("test", test),
    ):
        if (
            len(data) < 500
            or data["Decision Date"].nunique() < 4
        ):
            raise ValueError(
                f"{name}: insufficient observations/dates: "
                f"{len(data)}"
            )

    # Purge observations whose future target
    # extends into the next chronological split.
    val_start = val["Decision Date"].min()
    test_start = test["Decision Date"].min()

    train_before = len(train)
    val_before = len(val)

    train = train.loc[
        train["target_exit_date"] < val_start
    ].copy()

    val = val.loc[
        val["target_exit_date"] < test_start
    ].copy()

    if train.empty or val.empty:
        raise ValueError(
            "Purging removed a training or validation split"
        )

    if train["target_exit_date"].max() >= val_start:
        raise AssertionError(
            "Training target overlaps validation"
        )

    if val["target_exit_date"].max() >= test_start:
        raise AssertionError(
            "Validation target overlaps test"
        )

    print(
        "Purged train observations:",
        train_before - len(train),
    )

    print(
        "Purged validation observations:",
        val_before - len(val),
    )

    for name, data in (
        ("train", train),
        ("validation", val),
        ("test", test),
    ):
        print(
            f"{name}: {len(data):,} rows, "
            f"{data['Decision Date'].nunique()} decision dates, "
            f"{data['Decision Date'].min().date()} to "
            f"{data['Decision Date'].max().date()}"
        )

    return train, val, test


def make_models() -> dict:
    return {
        "train_median": DummyRegressor(
            strategy="median"
        ),

        "elastic_net": Pipeline([
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
                ElasticNet(
                    alpha=0.001,
                    l1_ratio=0.5,
                    max_iter=15000,
                    random_state=42,
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
                    max_iter=160,
                    learning_rate=0.05,
                    max_leaf_nodes=15,
                    min_samples_leaf=70,
                    l2_regularization=1.0,
                    early_stopping=False,
                    random_state=42,
                ),
            ),
        ]),
    }


def admissible_prediction(
    pred: np.ndarray,
    target: str,
) -> np.ndarray:

    pred = np.asarray(
        pred,
        dtype=float,
    )

    if target == "target_max_drawdown_63d":
        return np.minimum(pred, 0.0)

    return np.maximum(pred, 0.0)


def evaluate(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> dict:

    return {
        "MAE": float(
            mean_absolute_error(actual, predicted)
        ),
        "RMSE": float(
            np.sqrt(
                mean_squared_error(actual, predicted)
            )
        ),
        "R2": float(
            r2_score(actual, predicted)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )

    args = parser.parse_args()

    project = args.project_root.resolve()

    output = (
        project
        / "data/processed/ml_baselines/equity_63d_v1"
    )

    models_dir = output / "models"

    models_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    panel = get_dataset(project)

    print(
        "\n--- CONSERVATIVE 63-DAY ML BENCHMARK ---"
    )

    print(
        "Eligible feature + outcome rows:",
        len(panel),
    )

    print(
        "Numerical predictor features:",
        len(FEATURES),
    )

    train, val, test = split_data(panel)

    no_training_data = [
        col for col in FEATURES
        if train[col].notna().sum() == 0
    ]

    if no_training_data:
        raise ValueError(
            f"Features entirely missing in training: "
            f"{no_training_data}"
        )

    X_train = train[FEATURES]
    X_val = val[FEATURES]
    X_test = test[FEATURES]

    val_metrics = []
    test_metrics = []
    selections = []
    test_predictions = []

    for target in TARGETS:
        y_train = train[target].to_numpy(
            dtype=float
        )

        y_val = val[target].to_numpy(
            dtype=float
        )

        y_test = test[target].to_numpy(
            dtype=float
        )

        models = make_models()
        scores = {}

        for name, model in models.items():
            print(
                f"Training {name} for {target}...",
                flush=True,
            )

            model.fit(
                X_train,
                y_train,
            )

            predicted = admissible_prediction(
                model.predict(X_val),
                target,
            )

            scores[name] = evaluate(
                y_val,
                predicted,
            )

            val_metrics.append({
                "Target": target,
                "Model": name,
                "Rows": len(val),
                **scores[name],
            })

        # Select the model using validation MAE only.
        selected_name = min(
            models,
            key=lambda name: (
                scores[name]["MAE"],
                name,
            ),
        )

        selections.append({
            "Target": target,
            "Selected Model": selected_name,
            "Validation MAE": (
                scores[selected_name]["MAE"]
            ),
            "Validation Baseline MAE": (
                scores["train_median"]["MAE"]
            ),
        })

        print(
            f"Selected by validation MAE: {selected_name}"
        )

        chosen = models[selected_name]

        # Do not fit, tune, or select using the test data.
        model_predictions = admissible_prediction(
            chosen.predict(X_test),
            target,
        )

        baseline_predictions = admissible_prediction(
            models["train_median"].predict(X_test),
            target,
        )

        for name, pred in (
            (selected_name, model_predictions),
            ("train_median_reference", baseline_predictions),
        ):
            test_metrics.append({
                "Target": target,
                "Model": name,
                "Rows": len(test),
                **evaluate(y_test, pred),
            })

        prediction_frame = test[
            KEY + ["target_exit_date"]
        ].copy()

        prediction_frame["Target"] = target
        prediction_frame["Selected Model"] = selected_name
        prediction_frame["Actual"] = y_test

        prediction_frame["Predicted"] = (
            model_predictions
        )

        prediction_frame["Train-Median Baseline"] = (
            baseline_predictions
        )

        test_predictions.append(
            prediction_frame
        )

        joblib.dump(
            chosen,
            models_dir / f"{target}.joblib",
        )

    pd.DataFrame(
        val_metrics
    ).to_csv(
        output / "validation_model_comparison.csv",
        index=False,
    )

    pd.DataFrame(
        test_metrics
    ).to_csv(
        output / "heldout_test_metrics.csv",
        index=False,
    )

    pd.DataFrame(
        selections
    ).to_csv(
        output / "validation_selected_models.csv",
        index=False,
    )

    pd.concat(
        test_predictions,
        ignore_index=True,
    ).to_csv(
        output / "heldout_test_predictions.csv",
        index=False,
    )

    for name, split in (
        ("train", train),
        ("validation", val),
        ("test", test),
    ):
        split[
            KEY + ["target_exit_date"]
        ].to_csv(
            output / f"{name}_sample_keys.csv",
            index=False,
        )

    specification = {
        "horizon_trading_sessions": 63,
        "feature_columns": FEATURES,
        "targets": TARGETS,
        "split_boundaries": SPLITS,
        "training_membership": (
            "OBSERVED_SNAPSHOT_MATCH only"
        ),
        "model_selection": (
            "Lowest validation MAE before held-out test"
        ),
        "limitations": [
            "Incomplete terminal paths are excluded.",
            "This may introduce selection bias.",
            "The conservative sample omits 2019-2021.",
            "No option hedging transaction costs.",
            "No option contract adjustments.",
            "Adjusted prices are not total shareholder returns.",
        ],
    }

    (
        output / "feature_specification.json"
    ).write_text(
        json.dumps(
            specification,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\n--- VALIDATION MODEL COMPARISON ---"
    )

    print(
        pd.DataFrame(
            val_metrics
        ).to_string(index=False)
    )

    print(
        "\n--- HELD-OUT TEST "
        "(SELECTED VS BASELINE) ---"
    )

    print(
        pd.DataFrame(
            test_metrics
        ).to_string(index=False)
    )

    print(
        "\nSaved under:",
        output,
    )

    print(
        "EXPLORATORY ONLY: terminal-event and "
        "membership reviews remain open."
    )


if __name__ == "__main__":
    main()
