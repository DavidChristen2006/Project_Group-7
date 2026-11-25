"""ml/train_strategy_models.py: hedge research implementation.

Functions are documented at their point of definition.
Historical outputs are research estimates, not executable orders.
"""


from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

from xgboost import XGBRegressor


ROOT = (Path(__file__).resolve().parents[1] / "data/processed/ml_datasets/aapl_h21")
OUTPUT = ROOT / "model_benchmarks"
TARGET = "Hedge Improvement"


def load_dataset():
    """Load dataset from stored research artifacts.

    Returns:
        Computed result or status for the caller.
    """
    features = pd.read_parquet(ROOT / "features.parquet")
    targets = pd.read_parquet(ROOT / "targets.parquet")
    metadata = pd.read_parquet(ROOT / "metadata.parquet")
    dataset = pd.read_parquet(ROOT / "candidate_dataset.parquet")

    lengths = {
        len(features), len(targets),
        len(metadata), len(dataset),
    }

    if len(lengths) != 1:
        raise ValueError("Dataset files have different row counts.")

    dates = pd.to_datetime(metadata["Decision Date"])

    if not dates.equals(pd.to_datetime(dataset["Decision Date"])):
        raise ValueError("Metadata and dataset dates do not align.")

    # These columns describe realized outcomes,
    # not information available at the decision date.
    forbidden = {
        "Final Return",
        "Maximum Drawdown",
        "Hedge Improvement",
        "Drawdown Improvement",
        "Total Transaction Costs",
        "Unhedged Final Return",
        "Unhedged Maximum Drawdown",
        "Outcome Date",
    }

    leakage = forbidden.intersection(features.columns)

    if leakage:
        raise ValueError(f"Potential target leakage: {leakage}")

    x = features.apply(pd.to_numeric, errors="raise")
    x = x.replace([np.inf, -np.inf], np.nan)
    y = targets[TARGET].astype(float)

    if not np.isfinite(y).all():
        raise ValueError("Invalid target values.")

    return x, y, metadata, dataset


def make_models():
    """Construct models from the supplied inputs.

    Returns:
        Computed result or status for the caller.
    """
    def linear_steps():
        """Perform the linear steps operation for this research workflow.

        Returns:
            Computed result or status for the caller.
        """
        return [
            ("imputer", SimpleImputer(
                strategy="median",
                keep_empty_features=True,
            )),
            ("scaler", StandardScaler()),
        ]

    return {
        "OLS": Pipeline(
            linear_steps() + [
                ("model", LinearRegression())
            ]
        ),
        "ElasticNet": Pipeline(
            linear_steps() + [
                ("model", ElasticNet(
                    alpha=0.001,
                    l1_ratio=0.5,
                    max_iter=20000,
                ))
            ]
        ),
        "PCR": Pipeline(
            linear_steps() + [
                ("pca", PCA(n_components=5)),
                ("model", LinearRegression()),
            ]
        ),
        "PLS": Pipeline(
            linear_steps() + [
                ("model", PLSRegression(
                    n_components=3
                )),
            ]
        ),
        "RandomForest": Pipeline([
            ("imputer", SimpleImputer(
                strategy="median",
                keep_empty_features=True,
            )),
            ("model", RandomForestRegressor(
                n_estimators=200,
                min_samples_leaf=10,
                max_features=0.8,
                random_state=42,
                n_jobs=-1,
            )),
        ]),
        "XGBoost": Pipeline([
            ("imputer", SimpleImputer(
                strategy="median",
                keep_empty_features=True,
            )),
            ("model", XGBRegressor(
                n_estimators=250,
                max_depth=3,
                learning_rate=0.04,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="reg:squarederror",
                tree_method="hist",
                random_state=42,
                n_jobs=4,
            )),
        ]),
    }


def evaluate_policy(
    predictions,
    indices,
    metadata,
    dataset,
):
    """Perform the evaluate policy operation for this research workflow.

    Args:
        predictions: Input to the evaluate_policy operation.
        indices: Input to the evaluate_policy operation.
        metadata: Input to the evaluate_policy operation.
        dataset: Input to the evaluate_policy operation.

    Returns:
        Computed result or status for the caller.
    """
    rows = []

    frame = dataset.iloc[indices].copy()
    frame["Predicted Improvement"] = np.asarray(
        predictions
    ).reshape(-1)

    for date, group in frame.groupby(
        "Decision Date", sort=True
    ):
        baseline_return = group[
            "Unhedged Final Return"
        ].to_numpy(dtype=float)

        baseline_drawdown = group[
            "Unhedged Maximum Drawdown"
        ].to_numpy(dtype=float)

        # All candidates on the same decision date
        # must share the same unhedged benchmark.
        if (
            not np.allclose(baseline_return, baseline_return[0])
            or not np.allclose(baseline_drawdown, baseline_drawdown[0])
        ):
            raise ValueError(
                f"Inconsistent unhedged baseline: {date}"
            )

        best = group.loc[
            group["Predicted Improvement"].idxmax()
        ]

        # Zero is the predicted improvement
        # from not purchasing a hedge.
        if best["Predicted Improvement"] <= 0:
            strategy = "No Hedge"
            improvement = 0.0
            final_return = baseline_return[0]
            drawdown = baseline_drawdown[0]
            candidate_id = None
        else:
            strategy = best["Strategy"]
            improvement = float(best[TARGET])
            final_return = float(best["Final Return"])
            drawdown = float(best["Maximum Drawdown"])
            candidate_id = best["Candidate ID"]

        rows.append({
            "Decision Date": date,
            "Strategy": strategy,
            "Candidate ID": candidate_id,
            "Predicted Best Improvement":
                float(best["Predicted Improvement"]),
            "Realized Improvement": improvement,
            "Realized Return": final_return,
            "Maximum Drawdown": drawdown,
            "Unhedged Return": baseline_return[0],
            "Unhedged Drawdown": baseline_drawdown[0],
        })

    return pd.DataFrame(rows)


def main():
    """Run the module entry point and its workflow.
    """
    x, y, metadata, dataset = load_dataset()

    dates = pd.to_datetime(metadata["Decision Date"])

    train = dates.dt.year <= 2018
    validation = dates.dt.year.between(2022, 2023)
    test = dates.dt.year == 2024

    assert (train | validation | test).all()
    assert train.sum() > 0
    assert validation.sum() > 0
    assert test.sum() > 0

    print("\n--- CHRONOLOGICAL SPLIT ---")

    for name, mask in [
        ("Train", train),
        ("Validation", validation),
        ("Test", test),
    ]:
        print(
            name,
            "| candidates:", int(mask.sum()),
            "| decision dates:", dates[mask].nunique(),
        )

    models = make_models()
    summary = []
    validation_policies = {}

    print("\n--- VALIDATION BENCHMARK ---")

    for name, model in models.items():
        model.fit(x.loc[train], y.loc[train])

        predicted = np.asarray(
            model.predict(x.loc[validation])
        ).reshape(-1)

        policy = evaluate_policy(
            predicted,
            np.flatnonzero(validation.to_numpy()),
            metadata,
            dataset,
        )

        validation_policies[name] = policy

        result = {
            "Model": name,
            "Validation MAE": mean_absolute_error(
                y.loc[validation], predicted
            ),
            "Mean Realized Improvement":
                policy["Realized Improvement"].mean(),
            "Hedge Decisions":
                int(policy["Strategy"].ne("No Hedge").sum()),
            "No Hedge Decisions":
                int(policy["Strategy"].eq("No Hedge").sum()),
        }

        summary.append(result)

    ranking = (
        pd.DataFrame(summary)
        .sort_values(
            ["Mean Realized Improvement", "Validation MAE"],
            ascending=[False, True],
        )
        .reset_index(drop=True)
    )

    print(ranking.to_string(index=False))

    winner = ranking.iloc[0]["Model"]
    print("\nSelected model:", winner)

    # Fit the selected model again using all
    # observations available through 2023.
    final_train = train | validation
    final_model = make_models()[winner]

    final_model.fit(
        x.loc[final_train],
        y.loc[final_train],
    )

    predicted_test = np.asarray(
        final_model.predict(x.loc[test])
    ).reshape(-1)

    test_policy = evaluate_policy(
        predicted_test,
        np.flatnonzero(test.to_numpy()),
        metadata,
        dataset,
    )

    print("\n--- 2024 OUT-OF-SAMPLE TEST ---")

    print("Model:", winner)
    print(
        "Prediction MAE:",
        mean_absolute_error(
            y.loc[test], predicted_test
        ),
    )
    print(
        "Mean realized hedge improvement:",
        test_policy["Realized Improvement"].mean(),
    )
    print(
        "Mean strategy return:",
        test_policy["Realized Return"].mean(),
    )
    print(
        "Mean unhedged return:",
        test_policy["Unhedged Return"].mean(),
    )

    print("\n--- MONTHLY RECOMMENDATIONS ---")

    print(
        test_policy[
            [
                "Decision Date",
                "Strategy",
                "Predicted Best Improvement",
                "Realized Improvement",
                "Realized Return",
                "Maximum Drawdown",
            ]
        ].to_string(index=False)
    )

    OUTPUT.mkdir(parents=True, exist_ok=True)

    ranking.to_csv(
        OUTPUT / "validation_ranking.csv",
        index=False,
    )

    for name, policy in validation_policies.items():
        policy.to_csv(
            OUTPUT / f"validation_{name}.csv",
            index=False,
        )

    test_policy.to_csv(
        OUTPUT / "test_2024_recommendations.csv",
        index=False,
    )

    print("\nResults saved to:", OUTPUT)


if __name__ == "__main__":
    main()

