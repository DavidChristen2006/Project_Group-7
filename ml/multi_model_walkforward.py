
"""Research-only, point-in-time-label hedge policy evaluation.

Can optimise net Hedge Improvement alone, or an explicitly weighted
combination of net return improvement and drawdown improvement.
Daily LSEG quotes do not establish intraday execution synchronisation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml.ml_dataset_preparation import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    load_ml_dataset,
)

TARGET = "Hedge Improvement"
KEY = ["Decision Date", "Hedge Horizon Days", "Strategy"]


def build_models() -> dict[str, Pipeline]:
    def preprocessing():
        return ColumnTransformer([
            ("numeric", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]), NUMERIC_FEATURES),
            ("categorical", OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
            ), CATEGORICAL_FEATURES),
        ], sparse_threshold=0.0)

    def model(regressor):
        return Pipeline([
            ("preprocessing", preprocessing()),
            ("regression", regressor),
        ])

    models = {
        "OLS": model(LinearRegression()),
        "Ridge": model(Ridge(alpha=1.0)),
        "Lasso": model(Lasso(
            alpha=0.001,
            max_iter=20000,
        )),
        "Elastic Net": model(ElasticNet(
            alpha=0.001,
            l1_ratio=0.5,
            max_iter=20000,
        )),
        "PCR": Pipeline([
            ("preprocessing", preprocessing()),
            ("pca", PCA(n_components=3)),
            ("regression", LinearRegression()),
        ]),
        "PLS": model(PLSRegression(
            n_components=2,
            scale=False,
        )),
        "Random Forest": model(RandomForestRegressor(
            n_estimators=150,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=1,
        )),
        "Gradient Boosting": model(
            HistGradientBoostingRegressor(
                max_iter=100,
                learning_rate=0.05,
                max_leaf_nodes=7,
                min_samples_leaf=10,
                l2_regularization=1.0,
                random_state=42,
            )
        ),
    }

    try:
        from xgboost import XGBRegressor
    except ImportError as exc:
        raise ImportError(
            "Install XGBoost: py -m pip install xgboost"
        ) from exc

    models["XGBoost"] = model(XGBRegressor(
        objective="reg:squarederror",
        n_estimators=150,
        max_depth=2,
        learning_rate=0.05,
        subsample=1.0,
        colsample_bytree=1.0,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=1,
        verbosity=0,
    ))

    return models


def validate_dataset(
    data: pd.DataFrame,
    drawdown_weight: float = 0.0,
) -> pd.DataFrame:

    if not np.isfinite(drawdown_weight) or drawdown_weight < 0:
        raise ValueError(
            "drawdown_weight must be finite and nonnegative"
        )

    required = set(FEATURE_COLUMNS + KEY + [
        "Outcome Date",
        TARGET,
        "Final Return",
        "Unhedged Final Return",
    ])

    missing = required - set(data.columns)
    if missing:
        raise ValueError(
            f"Missing required fields: {sorted(missing)}"
        )

    df = data.copy()

    for column in ("Decision Date", "Outcome Date"):
        df[column] = pd.to_datetime(
            df[column],
            errors="coerce",
        ).dt.normalize()

        if df[column].isna().any():
            raise ValueError(f"Invalid {column}")

    for column in (
        TARGET,
        "Final Return",
        "Unhedged Final Return",
    ):
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

        if not np.isfinite(
            df[column].to_numpy(dtype=float)
        ).all():
            raise ValueError(f"Nonfinite {column}")

    if (
        df["Outcome Date"] <= df["Decision Date"]
    ).any():
        raise ValueError(
            "Outcome Date must follow Decision Date"
        )

    if df.duplicated(KEY).any():
        raise ValueError(
            "Duplicate decision/horizon/strategy candidate"
        )

    if (
        df["Strategy"].isna().any()
        or df["Strategy"].astype(str).str.strip().eq("").any()
    ):
        raise ValueError("Missing strategy name")

    if df["Strategy"].eq("No Hedge").any():
        raise ValueError(
            "No Hedge is added as a benchmark separately"
        )

    if not np.isfinite(
        df[NUMERIC_FEATURES].to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "Nonfinite numerical feature"
        )

    reconstructed_return = (
        df["Final Return"]
        - df["Unhedged Final Return"]
    )

    if not np.allclose(
        df[TARGET],
        reconstructed_return,
        rtol=0.0,
        atol=1e-9,
    ):
        raise ValueError(
            "Hedge Improvement differs from recorded final returns"
        )

    by_decision = df.groupby(KEY[:2])

    if (
        by_decision["Outcome Date"].nunique() != 1
    ).any():
        raise ValueError(
            "Outcome dates disagree within a decision/horizon"
        )

    nohedge_range = by_decision[
        "Unhedged Final Return"
    ].agg(lambda x: float(x.max() - x.min()))

    if (nohedge_range > 1e-9).any():
        raise ValueError(
            "Unhedged outcome disagrees within decision/horizon"
        )

    dd_columns = {
        "Maximum Drawdown",
        "Unhedged Maximum Drawdown",
    }

    if dd_columns.issubset(df.columns):

        for name in dd_columns:
            df[name] = pd.to_numeric(
                df[name],
                errors="coerce",
            )

            if not np.isfinite(
                df[name].to_numpy(dtype=float)
            ).all():
                raise ValueError(f"Nonfinite {name}")

        df["Drawdown Improvement"] = (
            df["Maximum Drawdown"]
            - df["Unhedged Maximum Drawdown"]
        )

        if "Drawdown Improvement" in data.columns:

            original = pd.to_numeric(
                data["Drawdown Improvement"],
                errors="coerce",
            )

            if (
                not np.isfinite(
                    original.to_numpy(dtype=float)
                ).all()
                or not np.allclose(
                    df["Drawdown Improvement"],
                    original,
                    rtol=0.0,
                    atol=1e-9,
                )
            ):
                raise ValueError(
                    "Drawdown Improvement differs from recorded drawdowns"
                )

        dd_range = df.groupby(KEY[:2])[
            "Unhedged Maximum Drawdown"
        ].agg(lambda x: float(x.max() - x.min()))

        if (dd_range > 1e-9).any():
            raise ValueError(
                "Unhedged drawdown disagrees within decision/horizon"
            )

    elif drawdown_weight > 0:
        raise ValueError(
            "Risk-weighted mode needs both drawdown columns"
        )

    else:
        df["Drawdown Improvement"] = np.nan

    df["Decision Utility"] = df[
        TARGET
    ].to_numpy(dtype=float)

    if drawdown_weight > 0:
        df["Decision Utility"] += (
            drawdown_weight
            * df["Drawdown Improvement"].to_numpy(
                dtype=float
            )
        )

    return df.sort_values(
        KEY
    ).reset_index(drop=True)


def chronological_fold(
    data,
    decision_date,
    *,
    final_test_start,
):
    date = pd.Timestamp(
        decision_date
    ).normalize()

    test_start = pd.Timestamp(
        final_test_start
    ).normalize()

    phase = (
        "Validation"
        if date < test_start
        else "Test"
    )

    cutoff = (
        date
        if phase == "Validation"
        else test_start
    )

    train = data.loc[
        (data["Decision Date"] < cutoff)
        & (data["Outcome Date"] < cutoff)
    ].copy()

    test = data.loc[
        data["Decision Date"].eq(date)
    ].copy()

    return train, test, phase, cutoff


def mean_benchmark(train, test):
    groups = [
        "Hedge Horizon Days",
        "Strategy",
    ]

    means = train.groupby(
        groups
    )["Decision Utility"].mean()

    lookup = pd.MultiIndex.from_frame(
        test[groups]
    )

    pred = means.reindex(
        lookup
    ).to_numpy(dtype=float)

    return np.where(
        np.isnan(pred),
        float(train["Decision Utility"].mean()),
        pred,
    )


def score_decisions(predictions):
    group_keys = [
        "Phase",
        "Decision Date",
        "Hedge Horizon Days",
        "Target",
        "Model",
    ]

    records = []

    for keys, group in predictions.groupby(
        group_keys,
        sort=True,
    ):

        group = group.assign(
            _priority=(
                group["Strategy"]
                .ne("No Hedge")
                .astype(int)
            )
        ).sort_values(
            ["_priority", "Strategy"],
            kind="stable",
        )

        chosen = group.iloc[
            int(np.argmax(
                group["Predicted"].to_numpy(
                    dtype=float
                )
            ))
        ]

        best = group.iloc[
            int(np.argmax(
                group["Actual"].to_numpy(
                    dtype=float
                )
            ))
        ]

        records.append(
            dict(zip(group_keys, keys)) | {
                "Selected Strategy": (
                    chosen["Strategy"]
                ),
                "Best Strategy": (
                    best["Strategy"]
                ),
                "Selected Actual": float(
                    chosen["Actual"]
                ),
                "Best Actual": float(
                    best["Actual"]
                ),
                "Regret": max(
                    0.0,
                    float(
                        best["Actual"]
                        - chosen["Actual"]
                    ),
                ),
                "Correct Selection": bool(
                    np.isclose(
                        chosen["Actual"],
                        best["Actual"],
                        rtol=0.0,
                        atol=1e-10,
                    )
                ),
                "Selected Hedge Improvement": float(
                    chosen["Realized Hedge Improvement"]
                ),
                "Selected Drawdown Improvement": float(
                    chosen["Realized Drawdown Improvement"]
                ),
                "Available Strategies": len(group),
            }
        )

    return pd.DataFrame(records)


def run_walkforward(
    dataset,
    *,
    validation_start="2023-07-03",
    test_start="2024-01-02",
    test_end="2024-12-31",
    min_training_dates=6,
    drawdown_weight=0.0,
    estimators=None,
):
    frame = validate_dataset(
        dataset,
        drawdown_weight,
    )

    val = pd.Timestamp(
        validation_start
    ).normalize()

    start = pd.Timestamp(
        test_start
    ).normalize()

    end = pd.Timestamp(
        test_end
    ).normalize()

    if not val < start <= end:
        raise ValueError(
            "Require validation_start < test_start <= test_end"
        )

    if min_training_dates < 2:
        raise ValueError(
            "min_training_dates must be >= 2"
        )

    if estimators is None:
        estimators = build_models()

    if (
        not estimators
        or set(estimators)
        & {"Historical Mean", "No Hedge"}
    ):
        raise ValueError(
            "Missing models or reserved model name"
        )

    prediction_rows = []
    fold_rows = []

    decisions = sorted(
        d
        for d in frame["Decision Date"].unique()
        if val <= d <= end
    )

    for date in decisions:

        train, test, phase, cutoff = (
            chronological_fold(
                frame,
                date,
                final_test_start=start,
            )
        )

        if (
            test.empty
            or train["Decision Date"].nunique()
            < min_training_dates
        ):
            continue

        X_train = train[FEATURE_COLUMNS]
        X_test = test[FEATURE_COLUMNS]

        y_train = train[
            "Decision Utility"
        ].to_numpy(dtype=float)

        pred_by_model = {
            "Historical Mean": mean_benchmark(
                train,
                test,
            )
        }

        for name, estimator in estimators.items():

            try:
                fitted = clone(estimator)

                fitted.fit(
                    X_train,
                    y_train,
                )

                predicted = np.asarray(
                    fitted.predict(X_test),
                    dtype=float,
                ).ravel()

            except Exception as exc:
                raise RuntimeError(
                    f"Cannot fit {name} at {date}: {exc}"
                ) from exc

            if (
                len(predicted) != len(test)
                or not np.isfinite(
                    predicted
                ).all()
            ):
                raise ValueError(
                    f"Invalid forecasts for {name} at {date}"
                )

            pred_by_model[name] = predicted

        for name, predicted in pred_by_model.items():

            for i, (_, candidate) in enumerate(
                test.iterrows()
            ):

                prediction_rows.append({
                    "Phase": phase,
                    "Decision Date": date,
                    "Outcome Date": candidate["Outcome Date"],
                    "Hedge Horizon Days": int(
                        candidate["Hedge Horizon Days"]
                    ),
                    "Target": "Decision Utility",
                    "Model": name,
                    "Strategy": candidate["Strategy"],
                    "Actual": float(
                        candidate["Decision Utility"]
                    ),
                    "Predicted": float(predicted[i]),
                    "Prediction Kind": "forecast",
                    "Realized Hedge Improvement": float(
                        candidate[TARGET]
                    ),
                    "Realized Drawdown Improvement": float(
                        candidate["Drawdown Improvement"]
                    ),
                })

        distinct_groups = test.drop_duplicates(
            "Hedge Horizon Days"
        )

        for _, row in distinct_groups.iterrows():

            for name in [
                *pred_by_model,
                "No Hedge",
            ]:

                prediction_rows.append({
                    "Phase": phase,
                    "Decision Date": date,
                    "Outcome Date": row["Outcome Date"],
                    "Hedge Horizon Days": int(
                        row["Hedge Horizon Days"]
                    ),
                    "Target": "Decision Utility",
                    "Model": name,
                    "Strategy": "No Hedge",
                    "Actual": 0.0,
                    "Predicted": 0.0,
                    "Prediction Kind": (
                        "known_zero_baseline"
                        if name != "No Hedge"
                        else "static_policy_score_not_forecast"
                    ),
                    "Realized Hedge Improvement": 0.0,
                    "Realized Drawdown Improvement": 0.0,
                })

        for _, candidate in test.iterrows():

            prediction_rows.append({
                "Phase": phase,
                "Decision Date": date,
                "Outcome Date": candidate["Outcome Date"],
                "Hedge Horizon Days": int(
                    candidate["Hedge Horizon Days"]
                ),
                "Target": "Decision Utility",
                "Model": "No Hedge",
                "Strategy": candidate["Strategy"],
                "Actual": float(
                    candidate["Decision Utility"]
                ),
                "Predicted": -1.0,
                "Prediction Kind": (
                    "static_policy_score_not_forecast"
                ),
                "Realized Hedge Improvement": float(
                    candidate[TARGET]
                ),
                "Realized Drawdown Improvement": float(
                    candidate["Drawdown Improvement"]
                ),
            })

        fold_rows.append({
            "Phase": phase,
            "Test Date": date,
            "Training Cutoff (exclusive)": cutoff,
            "Training Observations": len(train),
            "Training Decision Dates": (
                train["Decision Date"].nunique()
            ),
            "Training Latest Outcome Date": (
                train["Outcome Date"].max()
            ),
            "Test Observations": len(test),
            "Excluded Unfinished Outcomes": int((
                (frame["Decision Date"] < cutoff)
                & (frame["Outcome Date"] >= cutoff)
            ).sum()),
        })

    if not prediction_rows:
        raise ValueError(
            "No eligible walk-forward folds"
        )

    predictions = pd.DataFrame(
        prediction_rows
    )

    duplicate_keys = [
        "Phase",
        "Decision Date",
        "Hedge Horizon Days",
        "Model",
        "Strategy",
    ]

    if predictions.duplicated(
        duplicate_keys
    ).any():
        raise ValueError(
            "Duplicate model/strategy forecast"
        )

    return (
        predictions,
        score_decisions(predictions),
        pd.DataFrame(fold_rows),
    )


def summarize_recommendations(recommendations):
    return recommendations.groupby(
        [
            "Phase",
            "Hedge Horizon Days",
            "Model",
        ],
        as_index=False,
    ).agg(**{
        "Decisions": (
            "Decision Date",
            "nunique",
        ),
        "Mean Hedge Improvement": (
            "Selected Hedge Improvement",
            "mean",
        ),
        "Mean Drawdown Improvement": (
            "Selected Drawdown Improvement",
            "mean",
        ),
        "Mean Utility": (
            "Selected Actual",
            "mean",
        ),
        "Mean Regret": (
            "Regret",
            "mean",
        ),
        "Optimal Choice Rate": (
            "Correct Selection",
            "mean",
        ),
        "No Hedge Choice Rate": (
            "Selected Strategy",
            lambda s: float(
                s.eq("No Hedge").mean()
            ),
        ),
    }).sort_values(
        [
            "Phase",
            "Hedge Horizon Days",
            "Model",
        ]
    ).reset_index(drop=True)


def summarize_prediction_accuracy(predictions):
    forecast = predictions.loc[
        predictions["Prediction Kind"].eq(
            "forecast"
        )
    ].copy()

    forecast["Absolute Error"] = (
        forecast["Actual"]
        - forecast["Predicted"]
    ).abs()

    return forecast.groupby(
        [
            "Phase",
            "Hedge Horizon Days",
            "Model",
        ],
        as_index=False,
    ).agg(**{
        "Candidate Forecasts": (
            "Absolute Error",
            "size",
        ),
        "Decision Dates": (
            "Decision Date",
            "nunique",
        ),
        "MAE": (
            "Absolute Error",
            "mean",
        ),
    })


def choose_on_validation(
    summary,
    min_validation_decisions=10,
):
    rows = []

    validation = summary.loc[
        summary["Phase"].eq("Validation")
    ]

    for horizon, group in validation.groupby(
        "Hedge Horizon Days"
    ):

        eligible = group.loc[
            group["Decisions"]
            >= min_validation_decisions
        ].copy()

        winner = None

        if not eligible.empty:

            eligible["Tie Priority"] = (
                eligible["Model"]
                .map({
                    "No Hedge": 0,
                    "Historical Mean": 1,
                })
                .fillna(2)
            )

            winner = eligible.sort_values([
                "Mean Regret",
                "Tie Priority",
                "Model",
            ]).iloc[0]["Model"]

        rows.append({
            "Hedge Horizon Days": int(horizon),
            "Validation Selected Model": winner,
            "Selection Status": (
                "Selected using matured validation outcomes"
                if winner is not None
                else (
                    "Insufficient matured validation dates "
                    f"(<{min_validation_decisions})"
                )
            ),
        })

    return pd.DataFrame(
        rows,
        columns=[
            "Hedge Horizon Days",
            "Validation Selected Model",
            "Selection Status",
        ],
    )


def choose_matured_validation(
    predictions,
    test_start,
    min_validation_decisions=10,
):
    """Exclude validation outcomes unavailable at test start."""

    cutoff = pd.Timestamp(
        test_start
    ).normalize()

    horizons = sorted(
        int(h)
        for h in predictions[
            "Hedge Horizon Days"
        ].unique()
    )

    matured = predictions.loc[
        predictions["Phase"].eq("Validation")
        & (
            pd.to_datetime(
                predictions["Outcome Date"]
            ).dt.normalize()
            < cutoff
        )
    ].copy()

    result = pd.DataFrame({
        "Hedge Horizon Days": horizons
    })

    if matured.empty:
        result["Validation Selected Model"] = None
        result["Selection Status"] = (
            "No matured validation outcomes"
        )
        result["Matured Validation Decisions"] = 0
        return result

    scored = score_decisions(matured)

    summary = summarize_recommendations(
        scored
    )

    choices = choose_on_validation(
        summary,
        min_validation_decisions,
    )

    counts = scored.groupby(
        "Hedge Horizon Days"
    )["Decision Date"].nunique()

    result = result.merge(
        choices,
        on="Hedge Horizon Days",
        how="left",
        validate="one_to_one",
    )

    result["Matured Validation Decisions"] = (
        result["Hedge Horizon Days"]
        .map(counts)
        .fillna(0)
        .astype(int)
    )

    result["Selection Status"] = (
        result["Selection Status"].fillna(
            "No matured validation outcomes"
        )
    )

    return result


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        required=True,
    )
    parser.add_argument(
        "--validation-start",
        default="2023-07-03",
    )
    parser.add_argument(
        "--test-start",
        default="2024-01-02",
    )
    parser.add_argument(
        "--test-end",
        default="2024-12-31",
    )
    parser.add_argument(
        "--min-training-dates",
        type=int,
        default=6,
    )
    parser.add_argument(
        "--min-validation-decisions",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--drawdown-weight",
        type=float,
        default=0.0,
    )

    args = parser.parse_args()

    source = Path(args.dataset)
    output = Path(args.output_dir)

    if not source.is_file():
        parser.error(
            f"Dataset missing: {source}"
        )

    if output.exists():
        parser.error(
            f"Output directory exists: {output}"
        )

    dataset = load_ml_dataset(
        source
    )

    predictions, recommendations, folds = (
        run_walkforward(
            dataset,
            validation_start=args.validation_start,
            test_start=args.test_start,
            test_end=args.test_end,
            min_training_dates=args.min_training_dates,
            drawdown_weight=args.drawdown_weight,
        )
    )

    summary = summarize_recommendations(
        recommendations
    )

    accuracy = summarize_prediction_accuracy(
        predictions
    )

    choices = choose_matured_validation(
        predictions,
        args.test_start,
        args.min_validation_decisions,
    )

    output.mkdir(
        parents=True,
        exist_ok=False,
    )

    predictions.to_parquet(
        output / "predictions.parquet",
        index=False,
    )

    recommendations.to_parquet(
        output / "recommendations.parquet",
        index=False,
    )

    folds.to_parquet(
        output / "folds.parquet",
        index=False,
    )

    summary.to_csv(
        output / "summary.csv",
        index=False,
    )

    accuracy.to_csv(
        output / "forecast_accuracy.csv",
        index=False,
    )

    choices.to_csv(
        output / "validation_selection.csv",
        index=False,
    )

    experiment = {
        "dataset": str(source.resolve()),
        "dataset_sha256": hashlib.sha256(
            source.read_bytes()
        ).hexdigest(),
        "target": (
            "Net hedge improvement + "
            "drawdown_weight * drawdown improvement"
        ),
        "drawdown_weight": args.drawdown_weight,
        "validation_start": args.validation_start,
        "test_start": args.test_start,
        "test_end": args.test_end,
        "training_mode": (
            "rolling validation; frozen test cutoff"
        ),
        "selection": (
            "Matured validation outcomes strictly "
            "before test start"
        ),
        "min_training_dates": args.min_training_dates,
        "min_validation_decisions": (
            args.min_validation_decisions
        ),
        "costs": (
            "Use already recorded net Final Return, "
            "do not subtract again"
        ),
        "limitations": (
            "Daily bid/ask quotes lack intraday synchronisation; "
            "test explored previously"
        ),
    }

    (
        output / "experiment.json"
    ).write_text(
        json.dumps(
            experiment,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n--- REALIZED HEDGE POLICY RESULTS ---")
    print(
        summary.to_string(index=False)
    )

    print("\n--- FORECAST ACCURACY (UTILITY TARGET) ---")
    print(
        accuracy.to_string(index=False)
    )

    print("\n--- MATURED VALIDATION SELECTION ---")
    print(
        choices.to_string(index=False)
    )

    print("\nOutput:", output.resolve())


if __name__ == "__main__":
    main()
