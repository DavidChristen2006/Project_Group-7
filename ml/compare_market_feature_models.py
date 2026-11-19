
"""Compare 20-feature and 28-feature risk-aware ML models.

Run:
    py -m ml.compare_market_feature_models

Imports:
    pathlib: Path
    numpy: numerical validation and predictions
    pandas: Parquet and tabular operations
    sklearn.metrics: prediction error
    ml.train_risk_aware_models: existing training and
        candidate-selection functions
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from ml.train_risk_aware_models import (
    LAMBDAS,
    build_model,
    fit_pair,
    load_data,
    make_policy,
    predict_pair,
)


ROOT = Path("data/processed/ml_datasets/aapl_h21")
OUTPUT = ROOT / "market_feature_comparison"

MARKET_FEATURES = [
    "Return 5D",
    "Return 21D",
    "Return 63D",
    "Volatility 21D",
    "Volatility 63D",
    "Drawdown 63D",
    "Distance SMA 21",
    "Distance SMA 63",
]

MODEL_NAMES = (
    "OLS",
    "ElasticNet",
    "PCR",
    "PLS",
    "RandomForest",
    "XGBoost",
)


def load_feature_sets():
    baseline, raw = load_data()

    enhanced = pd.read_parquet(
        ROOT / "features_with_market.parquet"
    )

    enhanced = enhanced.reset_index(drop=True)

    if len(enhanced) != len(baseline):
        raise ValueError(
            "Enhanced and baseline row counts differ."
        )

    missing = set(MARKET_FEATURES) - set(
        enhanced.columns
    )

    if missing:
        raise ValueError(
            f"Missing market features: {sorted(missing)}"
        )

    if enhanced.columns.duplicated().any():
        raise ValueError(
            "Duplicate enhanced feature columns."
        )

    original_columns = baseline.columns.tolist()

    if not all(
        column in enhanced.columns
        for column in original_columns
    ):
        raise ValueError(
            "Original features missing from enhanced data."
        )

    original_values = enhanced[
        original_columns
    ].apply(pd.to_numeric, errors="raise")

    if not np.allclose(
        baseline.to_numpy(dtype=float),
        original_values.to_numpy(dtype=float),
        rtol=1e-10,
        atol=1e-8,
        equal_nan=True,
    ):
        raise ValueError(
            "Original features changed or row order differs."
        )

    market = enhanced[MARKET_FEATURES].apply(
        pd.to_numeric,
        errors="raise",
    )

    if not np.isfinite(
        market.to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "Invalid market-feature values."
        )

    # Market-state features must be identical
    # across candidates on the same decision date.
    check = market.copy()

    check["Decision Date"] = (
        raw["Decision Date"].to_numpy()
    )

    inconsistent = (
        check.groupby("Decision Date")[
            MARKET_FEATURES
        ].nunique(dropna=False) > 1
    )

    if inconsistent.any().any():
        raise ValueError(
            "Market features differ within a decision date."
        )

    enhanced = enhanced.apply(
        pd.to_numeric,
        errors="raise",
    ).replace([np.inf, -np.inf], np.nan)

    if enhanced[MARKET_FEATURES].isna().any().any():
        raise ValueError(
            "Missing market-state feature values."
        )

    return {
        "Baseline_20": baseline,
        "Market_28": enhanced,
    }, raw


def chronological_splits(raw):
    years = raw["Decision Date"].dt.year

    train = years.le(2018)
    validation = years.between(2022, 2023)
    holdout = years.eq(2024)

    if not (
        train | validation | holdout
    ).all():
        raise ValueError(
            "Unexpected year in historical dataset."
        )

    for label, mask in (
        ("Training", train),
        ("Validation", validation),
        ("2024 holdout", holdout),
    ):
        if not mask.any():
            raise ValueError(
                f"Empty chronological split: {label}"
            )

        print(
            f"{label}: {int(mask.sum())} candidates | "
            f"{raw.loc[mask, 'Decision Date'].nunique()} dates"
        )

    return train, validation, holdout


def evaluate_validation(
    feature_sets,
    raw,
    train,
    validation,
):
    results = []

    for feature_set_name, features in feature_sets.items():
        print(
            f"\n--- TRAINING {feature_set_name} ---"
        )

        x_train = features.loc[train]
        x_val = features.loc[validation]

        raw_train = raw.loc[train]
        raw_val = raw.loc[validation]

        for name in MODEL_NAMES:
            print("Model:", name)

            models = fit_pair(
                name,
                x_train,
                raw_train,
            )

            predicted_return, predicted_protection = (
                predict_pair(models, x_val)
            )

            return_mae = mean_absolute_error(
                raw_val["Return Benefit"],
                predicted_return,
            )

            protection_mae = mean_absolute_error(
                raw_val["Protection Benefit"],
                predicted_protection,
            )

            for weight in LAMBDAS:
                policy = make_policy(
                    raw_val,
                    predicted_return,
                    predicted_protection,
                    weight,
                )

                results.append({
                    "Feature Set": feature_set_name,
                    "Model": name,
                    "Lambda": weight,
                    "Validation Mean Utility": (
                        policy["Realized Utility"].mean()
                    ),
                    "Validation Return Benefit": (
                        policy[
                            "Realized Return Benefit"
                        ].mean()
                    ),
                    "Validation Protection Benefit": (
                        policy[
                            "Realized Protection Benefit"
                        ].mean()
                    ),
                    "Return MAE": return_mae,
                    "Protection MAE": protection_mae,
                    "Hedge Decisions": int(
                        policy["Strategy"].ne(
                            "No Hedge"
                        ).sum()
                    ),
                    "No Hedge Decisions": int(
                        policy["Strategy"].eq(
                            "No Hedge"
                        ).sum()
                    ),
                })

    return pd.DataFrame(results)


def select_models(ranking):
    selected = []

    for feature_set in (
        "Baseline_20",
        "Market_28",
    ):
        for weight in LAMBDAS:
            group = ranking.loc[
                ranking["Feature Set"].eq(feature_set)
                & ranking["Lambda"].eq(weight)
            ]

            best = group.sort_values(
                [
                    "Validation Mean Utility",
                    "Return MAE",
                    "Protection MAE",
                    "Model",
                ],
                ascending=[
                    False,
                    True,
                    True,
                    True,
                ],
                kind="stable",
            ).iloc[0]

            utility = float(
                best["Validation Mean Utility"]
            )

            selected.append({
                "Feature Set": feature_set,
                "Lambda": weight,
                "Selected Model": (
                    str(best["Model"])
                    if utility > 0
                    else "No Hedge"
                ),
                "Validation Mean Utility": max(
                    0.0,
                    utility,
                ),
                "Best ML Validation Utility": utility,
                "Validation Return MAE": float(
                    best["Return MAE"]
                ),
                "Validation Protection MAE": float(
                    best["Protection MAE"]
                ),
            })

    return pd.DataFrame(selected)


def evaluate_holdout(
    selections,
    feature_sets,
    raw,
    train,
    validation,
    holdout,
):
    summaries = []
    policies = {}

    final_train = train | validation

    for _, selected in selections.iterrows():
        feature_set = str(
            selected["Feature Set"]
        )

        name = str(
            selected["Selected Model"]
        )

        weight = float(
            selected["Lambda"]
        )

        features = feature_sets[feature_set]

        raw_holdout = raw.loc[holdout]

        if name == "No Hedge":
            predicted_return = np.full(
                len(raw_holdout),
                -1.0,
            )

            predicted_protection = np.zeros(
                len(raw_holdout)
            )

        else:
            models = fit_pair(
                name,
                features.loc[final_train],
                raw.loc[final_train],
            )

            predicted_return, predicted_protection = (
                predict_pair(
                    models,
                    features.loc[holdout],
                )
            )

        policy = make_policy(
            raw_holdout,
            predicted_return,
            predicted_protection,
            weight,
        )

        policies[
            (feature_set, weight)
        ] = policy

        summaries.append({
            "Feature Set": feature_set,
            "Lambda": weight,
            "Selected Model": name,
            "2024 Mean Utility": (
                policy["Realized Utility"].mean()
            ),
            "2024 Return Benefit": (
                policy[
                    "Realized Return Benefit"
                ].mean()
            ),
            "2024 Protection Benefit": (
                policy[
                    "Realized Protection Benefit"
                ].mean()
            ),
            "2024 Hedge Decisions": int(
                policy["Strategy"].ne(
                    "No Hedge"
                ).sum()
            ),
        })

        suffix = str(weight).replace(
            ".",
            "p",
        )

        policy.to_csv(
            OUTPUT / (
                f"{feature_set}_lambda_{suffix}_2024.csv"
            ),
            index=False,
        )

    return pd.DataFrame(summaries), policies


def compare_decisions(policies):
    records = []

    for weight in LAMBDAS:
        baseline = policies[
            ("Baseline_20", weight)
        ].sort_values(
            "Decision Date"
        ).reset_index(drop=True)

        enhanced = policies[
            ("Market_28", weight)
        ].sort_values(
            "Decision Date"
        ).reset_index(drop=True)

        if not baseline["Decision Date"].equals(
            enhanced["Decision Date"]
        ):
            raise ValueError(
                "Holdout policy dates do not match."
            )

        baseline_ids = (
            baseline["Candidate ID"]
            .astype("string")
            .fillna("No Hedge")
        )

        enhanced_ids = (
            enhanced["Candidate ID"]
            .astype("string")
            .fillna("No Hedge")
        )

        differences = (
            baseline_ids.to_numpy()
            != enhanced_ids.to_numpy()
        )

        records.append({
            "Lambda": weight,
            "Different Decisions": int(
                differences.sum()
            ),
            "Decision Dates": len(baseline),
        })

    return pd.DataFrame(records)


def main():
    feature_sets, raw = load_feature_sets()

    print("\n--- DATASET VALIDATION ---")

    print("Candidates:", len(raw))

    print(
        "Decision dates:",
        raw["Decision Date"].nunique(),
    )

    for name, features in feature_sets.items():
        print(
            name,
            "| features:",
            features.shape[1],
        )

    train, validation, holdout = (
        chronological_splits(raw)
    )

    OUTPUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    ranking = evaluate_validation(
        feature_sets,
        raw,
        train,
        validation,
    )

    ranking.to_csv(
        OUTPUT / "validation_ranking.csv",
        index=False,
    )

    selected = select_models(ranking)

    selected.to_csv(
        OUTPUT / "selected_models.csv",
        index=False,
    )

    print("\n--- SELECTED MODELS ---")

    print(
        selected.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    holdout_results, policies = evaluate_holdout(
        selected,
        feature_sets,
        raw,
        train,
        validation,
        holdout,
    )

    holdout_results.to_csv(
        OUTPUT / "holdout_2024_comparison.csv",
        index=False,
    )

    print("\n--- 2024 HISTORICAL HOLDOUT ---")

    print(
        holdout_results.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    decision_comparison = compare_decisions(
        policies
    )

    decision_comparison.to_csv(
        OUTPUT / "decision_comparison.csv",
        index=False,
    )

    print("\n--- DECISION DIFFERENCES ---")

    print(
        decision_comparison.to_string(
            index=False
        )
    )

    print("\nResults saved to:", OUTPUT)
    print("Existing datasets were not modified.")


if __name__ == "__main__":
    main()
