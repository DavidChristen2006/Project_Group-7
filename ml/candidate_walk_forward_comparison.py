
import argparse

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    GradientBoostingRegressor,
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
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)

from ml.walk_forward_validation import (
    get_walk_forward_split,
)

from pipeline.historical_ml_feature_comparison import (
    DATA_PATH,
    MARKET_FEATURES,
)

from pipeline.historical_ml_benchmark import (
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
)

from pipeline.historical_ml_constraints import (
    HedgeConstraints,
    filter_eligible_candidates,
)


TARGET_RETURN = "Hedge Improvement"
TARGET_RISK = "Drawdown Improvement"

MODEL_NAMES = [
    "Historical Mean",
    "OLS",
    "Ridge",
    "Lasso",
    "Elastic Net",
    "PCR",
    "PLS",
    "Random Forest",
    "Gradient Boosting",
    "XGBoost",
]


def build_preprocessor(numeric_features):
    numeric = Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="constant",
                fill_value=0.0,
                add_indicator=True,
            ),
        ),
        ("scaler", StandardScaler()),
    ])

    categorical = Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent",
            ),
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
            ),
        ),
    ])

    return ColumnTransformer([
        ("numeric", numeric, numeric_features),
        (
            "categorical",
            categorical,
            CATEGORICAL_FEATURES,
        ),
    ])


def build_estimator(name):
    if name == "Historical Mean":
        return DummyRegressor(strategy="mean")

    if name == "OLS":
        return LinearRegression()

    if name == "Ridge":
        return Ridge(alpha=10.0)

    if name == "Lasso":
        return Lasso(
            alpha=0.001,
            max_iter=20000,
        )

    if name == "Elastic Net":
        return ElasticNet(
            alpha=0.001,
            l1_ratio=0.5,
            max_iter=20000,
            random_state=42,
        )

    if name == "PCR":
        return Pipeline([
            ("pca", PCA(n_components=5)),
            ("regression", LinearRegression()),
        ])

    if name == "PLS":
        return PLSRegression(n_components=3)

    if name == "Random Forest":
        return RandomForestRegressor(
            n_estimators=200,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=-1,
        )

    if name == "Gradient Boosting":
        return GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=2,
            random_state=42,
        )

    if name == "XGBoost":
        from xgboost import XGBRegressor

        return XGBRegressor(
            objective="reg:squarederror",
            n_estimators=150,
            max_depth=2,
            learning_rate=0.05,
            subsample=1.0,
            colsample_bytree=1.0,
            random_state=42,
            n_jobs=2,
            tree_method="hist",
        )

    raise ValueError(f"Unknown model: {name}")


def make_model(name, numeric_features):
    return Pipeline([
        (
            "preprocessor",
            build_preprocessor(numeric_features),
        ),
        ("estimator", build_estimator(name)),
    ])


def load_dataset():
    if not DATA_PATH.is_file():
        raise FileNotFoundError(DATA_PATH)

    df = pd.read_parquet(DATA_PATH)

    required = (
        NUMERIC_FEATURES
        + CATEGORICAL_FEATURES
        + MARKET_FEATURES
        + [
            "Candidate Key",
            "Decision Date",
            "Outcome Date",
            "Strategy",
            "Short Call Strike Pct",
            "Long Put Strike Pct",
            "Short Put Strike Pct",
            "Net Premium Pct",
            TARGET_RETURN,
            TARGET_RISK,
        ]
    )

    missing = [
        column for column in set(required)
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    for col in ["Decision Date", "Outcome Date"]:
        df[col] = (
            pd.to_datetime(df[col], errors="raise")
            .astype("datetime64[ns]")
            .dt.normalize()
        )

    if not df["Candidate Key"].is_unique:
        raise ValueError("Duplicate Candidate Key.")

    for target in [TARGET_RETURN, TARGET_RISK]:
        df[target] = pd.to_numeric(
            df[target],
            errors="raise",
        )

        if not np.isfinite(df[target]).all():
            raise ValueError(
                f"Non-finite target: {target}"
            )

    return df.sort_values(
        ["Decision Date", "Candidate Key"]
    ).reset_index(drop=True)


def fit_predict(
    name,
    train,
    test,
    numeric_features,
):
    features = (
        numeric_features
        + CATEGORICAL_FEATURES
    )

    predictions = test[
        [
            "Candidate Key",
            "Decision Date",
            "Strategy",
            TARGET_RETURN,
            TARGET_RISK,
        ]
    ].copy()

    for target, predicted_column in [
        (TARGET_RETURN, "Predicted Return"),
        (TARGET_RISK, "Predicted Risk Benefit"),
    ]:
        model = make_model(
            name,
            numeric_features,
        )

        model.fit(
            train[features],
            train[target],
        )

        predicted = np.asarray(
            model.predict(test[features]),
            dtype=float,
        ).reshape(-1)

        if (
            len(predicted) != len(test)
            or not np.isfinite(predicted).all()
        ):
            raise ValueError(
                f"Invalid predictions: {name}, {target}"
            )

        predictions[predicted_column] = predicted

    return predictions


def evaluate_policy(
    test,
    predictions,
    constraints,
    weight,
):
    # Candidate Key, rather than Strategy, is
    # the unique observation identifier.
    scored = test.merge(
        predictions[
            [
                "Candidate Key",
                "Predicted Return",
                "Predicted Risk Benefit",
            ]
        ],
        on="Candidate Key",
        how="left",
        validate="one_to_one",
    )

    eligible = filter_eligible_candidates(
        scored,
        constraints,
    ).copy()

    eligible["Predicted Utility"] = (
        eligible["Predicted Return"]
        + weight * eligible["Predicted Risk Benefit"]
    )

    records = []

    for date, group in test.groupby("Decision Date"):
        month = eligible.loc[
            eligible["Decision Date"].eq(date)
        ].sort_values(
            ["Predicted Utility", "Candidate Key"],
            ascending=[False, True],
        )

        record = {
            "Decision Date": date,
            "Decision": "No Hedge",
            "Candidate Key": None,
            "Expiry": pd.NaT,
            "Long Put Strike Pct": np.nan,
            "Short Call Strike Pct": np.nan,
            "Short Put Strike Pct": np.nan,
            "Net Premium Pct": np.nan,
            "Predicted Return": 0.0,
            "Predicted Risk Benefit": 0.0,
            "Predicted Utility": 0.0,
            "Return Benefit": 0.0,
            "Risk Benefit": 0.0,
            "Eligible": len(month),
        }

        if not month.empty:
            best = month.iloc[0]

            if best["Predicted Utility"] > 0:
                record.update({
                    "Decision": best["Strategy"],
                    "Candidate Key": best["Candidate Key"],
                    "Expiry": best["Expiry"],
                    "Long Put Strike Pct": best[
                        "Long Put Strike Pct"
                    ],
                    "Short Call Strike Pct": best[
                        "Short Call Strike Pct"
                    ],
                    "Short Put Strike Pct": best[
                        "Short Put Strike Pct"
                    ],
                    "Net Premium Pct": best[
                        "Net Premium Pct"
                    ],
                    "Predicted Return": best[
                        "Predicted Return"
                    ],
                    "Predicted Risk Benefit": best[
                        "Predicted Risk Benefit"
                    ],
                    "Predicted Utility": best[
                        "Predicted Utility"
                    ],
                    "Return Benefit": best[TARGET_RETURN],
                    "Risk Benefit": best[TARGET_RISK],
                })

        records.append(record)

    return pd.DataFrame(records)


def evaluate_fold(
    train,
    test,
    model_names,
    numeric_features,
    constraints,
    risk_weight,
):
    records = []

    for name in model_names:
        predicted = fit_predict(
            name,
            train,
            test,
            numeric_features,
        )

        selected = evaluate_policy(
            test,
            predicted,
            constraints,
            risk_weight,
        )

        counts = selected["Decision"].value_counts()

        choice = selected.iloc[0]

        records.append({
            "Decision Date": test["Decision Date"].iloc[0],
            "Model": name,
            "Training Dates": train["Decision Date"].nunique(),
            "Training Candidates": len(train),
            "Test Candidates": len(test),
            "Return MAE (pp)": (
                np.abs(
                    predicted[TARGET_RETURN]
                    - predicted["Predicted Return"]
                ).mean() * 100
            ),
            "Risk MAE (pp)": (
                np.abs(
                    predicted[TARGET_RISK]
                    - predicted["Predicted Risk Benefit"]
                ).mean() * 100
            ),
            "Return Benefit (pp)": (
                selected["Return Benefit"].mean() * 100
            ),
            "Risk Benefit (pp)": (
                selected["Risk Benefit"].mean() * 100
            ),
            "Utility (pp)": (
                (
                    selected["Return Benefit"]
                    + risk_weight
                    * selected["Risk Benefit"]
                ).mean() * 100
            ),
            "Decision": selected["Decision"].iloc[0],
            "Eligible": selected["Eligible"].iloc[0],
            
            "Candidate Key": choice["Candidate Key"],
            "Expiry": choice["Expiry"],
            "Long Put Strike Pct": choice["Long Put Strike Pct"],
            "Short Call Strike Pct": choice["Short Call Strike Pct"],
            "Short Put Strike Pct": choice["Short Put Strike Pct"],
            "Net Premium Pct": choice["Net Premium Pct"],
            "Predicted Return (pp)": (
                choice["Predicted Return"] * 100
            ),
            "Predicted Risk (pp)": (
                choice["Predicted Risk Benefit"] * 100
            ),
            "Predicted Utility (pp)": (
                choice["Predicted Utility"] * 100
            ),

        })

    return records


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--test-start",
        default="2023-01-03",
    )

    parser.add_argument(
        "--max-dates",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--models",
        default="Historical Mean,OLS,Ridge,Lasso,"
                "Elastic Net,PCR,PLS,Random Forest,"
                "Gradient Boosting,XGBoost",
    )

    parser.add_argument(
        "--market-features",
        action="store_true",
    )

    parser.add_argument(
        "--risk-weight",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--min-call-otm",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--max-net-debit",
        type=float,
        default=0.02,
    )

    args = parser.parse_args()

    model_names = [
        name.strip()
        for name in args.models.split(",")
    ]

    if (
        not model_names
        or len(model_names) != len(set(model_names))
        or any(name not in MODEL_NAMES for name in model_names)
    ):
        raise ValueError(
            "Unknown or duplicate model names."
        )

    if (
        args.max_dates < 1
        or not np.isfinite(args.risk_weight)
        or args.risk_weight < 0
    ):
        raise ValueError(
            "Invalid evaluation parameters."
        )

    numeric_features = NUMERIC_FEATURES.copy()

    if args.market_features:
        numeric_features += MARKET_FEATURES

    constraints = HedgeConstraints(
        min_call_otm_pct=args.min_call_otm,
        max_net_debit_pct=args.max_net_debit,
    )

    df = load_dataset()

    start = pd.Timestamp(args.test_start).normalize()

    dates = sorted(
        df.loc[
            df["Decision Date"] >= start,
            "Decision Date",
        ].unique()
    )[:args.max_dates]

    records = []

    print("\n--- CANDIDATE WALK-FORWARD COMPARISON ---")

    for date in dates:
        train, test, excluded = get_walk_forward_split(
            df,
            date,
        )

        n_dates = train["Decision Date"].nunique()

        if test.empty or n_dates < 6:
            print(f"Skipping {date}: insufficient data")
            continue

        if not (
            train["Outcome Date"] < pd.Timestamp(date)
        ).all():
            raise ValueError(
                "Training uses an unfinished outcome."
            )

        print(
            f"Test {pd.Timestamp(date).date()} | "
            f"train dates={n_dates} | "
            f"train rows={len(train)} | "
            f"test rows={len(test)} | "
            f"excluded={len(excluded)}"
        )

        records.extend(
            evaluate_fold(
                train,
                test,
                model_names,
                numeric_features,
                constraints,
                args.risk_weight,
            )
        )

    if not records:
        raise ValueError(
            "No eligible walk-forward folds."
        )

    results = pd.DataFrame(records)

    print("\n--- FOLD RESULTS ---")
    print(
        results.round(4).to_string(index=False)
    )

    summary = results.groupby("Model").agg(
        Dates=("Decision Date", "nunique"),
        Return_MAE_pp=("Return MAE (pp)", "mean"),
        Risk_MAE_pp=("Risk MAE (pp)", "mean"),
        Return_pp=("Return Benefit (pp)", "mean"),
        Risk_pp=("Risk Benefit (pp)", "mean"),
        Utility_pp=("Utility (pp)", "mean"),
    ).reset_index()

    summary = summary.sort_values(
        ["Utility_pp", "Model"],
        ascending=[False, True],
    )

    print("\n--- WALK-FORWARD SUMMARY ---")
    print(summary.round(4).to_string(index=False))

    print(
        "\nHistorical Mean is the baseline. "
        "2024 results are exploratory."
    )

    print("No files modified.")


if __name__ == "__main__":
    main()
