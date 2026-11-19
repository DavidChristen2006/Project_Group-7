
import argparse

import numpy as np
import pandas as pd

from pipeline.historical_ml_benchmark import (
    split_dataset,
)

from pipeline.historical_ml_feature_comparison import (
    DATA_PATH,
    make_model,
)


RETURN_TARGET = "Hedge Improvement"
RISK_TARGET = "Drawdown Improvement"


def load_data():
    df = pd.read_parquet(DATA_PATH)

    required = [
        "Candidate Key",
        "Decision Date",
        "Outcome Date",
        "Strategy",
        "Expiry",
        RETURN_TARGET,
        RISK_TARGET,
    ]

    missing = [
        col for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    for col in ["Decision Date", "Outcome Date"]:
        df[col] = pd.to_datetime(
            df[col]
        ).astype("datetime64[ns]")

    if not df["Candidate Key"].is_unique:
        raise ValueError("Duplicate candidate keys.")

    for target in [RETURN_TARGET, RISK_TARGET]:
        df[target] = pd.to_numeric(
            df[target],
            errors="raise",
        )

        if not np.isfinite(df[target]).all():
            raise ValueError(
                f"Invalid target values: {target}"
            )

    return df.sort_values(
        ["Decision Date", "Candidate Key"]
    ).reset_index(drop=True)


def train_two_models(train, use_market_features):
    return_model, features = make_model(
        use_market_features
    )

    risk_model, risk_features = make_model(
        use_market_features
    )

    if features != risk_features:
        raise ValueError(
            "Feature sets are inconsistent."
        )

    return_model.fit(
        train[features],
        train[RETURN_TARGET],
    )

    risk_model.fit(
        train[features],
        train[RISK_TARGET],
    )

    return return_model, risk_model, features


def predict_two_targets(
    df,
    return_model,
    risk_model,
    features,
):
    result = df.copy()

    result["Predicted Return Improvement"] = (
        return_model.predict(df[features])
    )

    result["Predicted Drawdown Improvement"] = (
        risk_model.predict(df[features])
    )

    predictions = result[
        [
            "Predicted Return Improvement",
            "Predicted Drawdown Improvement",
        ]
    ].to_numpy(dtype=float)

    if not np.isfinite(predictions).all():
        raise ValueError(
            "Non-finite model predictions."
        )

    return result


def select_hedges(predictions, risk_weight):
    if risk_weight < 0:
        raise ValueError(
            "Risk weight must be nonnegative."
        )

    ranked = predictions.copy()

    ranked["Predicted Score"] = (
        ranked["Predicted Return Improvement"]
        + risk_weight
        * ranked["Predicted Drawdown Improvement"]
    )

    # Compare every candidate on each decision date.
    best_indices = ranked.groupby(
        "Decision Date"
    )["Predicted Score"].idxmax()

    selected = ranked.loc[
        best_indices
    ].sort_values("Decision Date").copy()

    # No hedge has zero improvement relative
    # to the unhedged benchmark.
    selected["Choose Hedge"] = (
        selected["Predicted Score"] > 0
    )

    selected["Decision"] = np.where(
        selected["Choose Hedge"],
        selected["Strategy"],
        "No Hedge",
    )

    selected["Realized Return Benefit"] = np.where(
        selected["Choose Hedge"],
        selected[RETURN_TARGET],
        0.0,
    )

    selected["Realized Drawdown Benefit"] = np.where(
        selected["Choose Hedge"],
        selected[RISK_TARGET],
        0.0,
    )

    selected["Realized Utility"] = (
        selected["Realized Return Benefit"]
        + risk_weight
        * selected["Realized Drawdown Benefit"]
    )

    selected["Predicted Utility"] = np.where(
        selected["Choose Hedge"],
        selected["Predicted Score"],
        0.0,
    )

    return selected


def summarize(
    predictions,
    selected,
    risk_weight,
    period,
    feature_set,
):
    # Oracle is calculated only as a
    # retrospective diagnostic benchmark.
    oracle_scores = (
        predictions[RETURN_TARGET]
        + risk_weight
        * predictions[RISK_TARGET]
    )

    oracle = predictions.assign(
        Oracle_Score=oracle_scores
    ).groupby("Decision Date")[
        "Oracle_Score"
    ].max().clip(lower=0.0)

    realized_utility = selected.set_index(
        "Decision Date"
    )["Realized Utility"]

    regret = (
        oracle.reindex(realized_utility.index)
        - realized_utility
    )

    return {
        "Period": period,
        "Features": feature_set,
        "Risk Weight": risk_weight,
        "Dates": len(selected),
        "Hedges": int(
            selected["Choose Hedge"].sum()
        ),
        "Return Benefit (pp)": (
            selected[
                "Realized Return Benefit"
            ].mean() * 100
        ),
        "Drawdown Benefit (pp)": (
            selected[
                "Realized Drawdown Benefit"
            ].mean() * 100
        ),
        "Realized Utility (pp)": (
            realized_utility.mean() * 100
        ),
        "Utility Regret (pp)": (
            regret.mean() * 100
        ),
        "Return MAE (pp)": (
            np.mean(
                np.abs(
                    predictions[RETURN_TARGET]
                    - predictions[
                        "Predicted Return Improvement"
                    ]
                )
            ) * 100
        ),
        "Drawdown MAE (pp)": (
            np.mean(
                np.abs(
                    predictions[RISK_TARGET]
                    - predictions[
                        "Predicted Drawdown Improvement"
                    ]
                )
            ) * 100
        ),
    }


def evaluate_period(
    train,
    evaluation,
    period,
    feature_set,
    use_market_features,
    risk_weights,
):
    return_model, risk_model, features = (
        train_two_models(
            train,
            use_market_features,
        )
    )

    predictions = predict_two_targets(
        evaluation,
        return_model,
        risk_model,
        features,
    )

    rows = []

    for risk_weight in risk_weights:
        selected = select_hedges(
            predictions,
            risk_weight,
        )

        rows.append(
            summarize(
                predictions,
                selected,
                risk_weight,
                period,
                feature_set,
            )
        )

    return rows


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--risk-weights",
        default="0,0.25,0.5,1.0",
    )

    args = parser.parse_args()

    risk_weights = [
        float(value.strip())
        for value in args.risk_weights.split(",")
    ]

    if (
        not risk_weights
        or not np.isfinite(risk_weights).all()
        or any(value < 0 for value in risk_weights)
    ):
        raise ValueError(
            "All risk weights must be finite "
            "and nonnegative."
        )

    df = load_data()

    train, validation, final_train, test = (
        split_dataset(df)
    )

    print("\n--- DATASET ---")
    print("Candidates:", len(df))
    print("Decision dates:", df["Decision Date"].nunique())
    print("Training:", len(train))
    print("Validation:", len(validation))
    print("Final training:", len(final_train))
    print("2024 evaluation:", len(test))

    configurations = [
        ("Original", False),
        ("Market-Aware", True),
    ]

    results = []

    for feature_set, use_market in configurations:
        results.extend(
            evaluate_period(
                train,
                validation,
                "2023 validation",
                feature_set,
                use_market,
                risk_weights,
            )
        )

        results.extend(
            evaluate_period(
                final_train,
                test,
                "2024 exploratory",
                feature_set,
                use_market,
                risk_weights,
            )
        )

    table = pd.DataFrame(results)

    print("\n--- RISK-AWARE ML COMPARISON ---")

    display_columns = [
        "Period",
        "Features",
        "Risk Weight",
        "Hedges",
        "Return Benefit (pp)",
        "Drawdown Benefit (pp)",
        "Realized Utility (pp)",
        "Utility Regret (pp)",
        "Return MAE (pp)",
        "Drawdown MAE (pp)",
    ]

    print(
        table[display_columns]
        .round(4)
        .to_string(index=False)
    )

    print("\n--- IMPORTANT ---")
    print(
        "2024 is exploratory, not an untouched "
        "test period."
    )

    print(
        "Risk weights are investor preferences, "
        "not fitted model parameters."
    )

    print("No files modified.")


if __name__ == "__main__":
    main()
