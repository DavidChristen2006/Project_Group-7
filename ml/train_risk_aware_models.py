
"""Risk-aware, chronological AAPL candidate-policy ML benchmark.

Run from project root: py -m ml.train_risk_aware_models
Requires the files produced by ml.consolidate_candidate_dataset.
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

ROOT = (Path(__file__).resolve().parents[1] / "data/processed/ml_datasets/aapl_h21")
OUTPUT = ROOT / "risk_aware_benchmarks"
LAMBDAS = (0.0, 0.5, 1.0, 2.0)
SEED = 42


def load_data():
    """Load data from stored research artifacts.

    Returns:
        Computed result or status for the caller.
    """
    x = pd.read_parquet(ROOT / "features.parquet").reset_index(drop=True)
    y = pd.read_parquet(ROOT / "targets.parquet").reset_index(drop=True)
    meta = pd.read_parquet(ROOT / "metadata.parquet").reset_index(drop=True)
    raw = pd.read_parquet(ROOT / "candidate_dataset.parquet").reset_index(drop=True)

    if len({len(x), len(y), len(meta), len(raw)}) != 1:
        raise ValueError(
            "Consolidated tables have different numbers of rows"
        )

    for column in ("Decision Date", "Candidate ID", "Strategy"):
        if column in meta.columns and column in raw.columns:
            if column == "Decision Date":
                a = pd.to_datetime(meta[column]).dt.normalize()
                b = pd.to_datetime(raw[column]).dt.normalize()
            else:
                a = meta[column].astype(str)
                b = raw[column].astype(str)

            if not a.equals(b):
                raise ValueError(
                    f"Metadata misaligned at: {column}"
                )

    forbidden = {
        "Outcome Date",
        "Final Return",
        "Final P&L",
        "Maximum Drawdown",
        "Unhedged Final Return",
        "Unhedged Maximum Drawdown",
        "Hedge Improvement",
        "Drawdown Improvement",
        "Gross Final P&L",
        "Gross Final Return",
        "Gross Maximum Drawdown",
        "Total Transaction Costs",
        "Option Commissions",
        "Option Fixed Fees",
        "Option Slippage",
        "Stock Commissions",
        "Observations",
    }

    leak = forbidden.intersection(x.columns)

    if leak:
        raise ValueError(
            f"Future outcome columns found in features: {sorted(leak)}"
        )

    x = x.apply(
        pd.to_numeric,
        errors="raise",
    ).replace([np.inf, -np.inf], np.nan)

    if x.empty or x.columns.duplicated().any():
        raise ValueError(
            "Empty or duplicate feature columns"
        )

    required = [
        "Final Return",
        "Unhedged Final Return",
        "Maximum Drawdown",
        "Unhedged Maximum Drawdown",
        "Decision Date",
        "Outcome Date",
        "Candidate ID",
        "Strategy",
    ]

    missing = set(required).difference(raw.columns)

    if missing:
        raise ValueError(
            f"Missing required dataset columns: {sorted(missing)}"
        )

    # Positive protection means maximum drawdown is less negative.
    raw["Return Benefit"] = (
        raw["Final Return"]
        - raw["Unhedged Final Return"]
    )

    raw["Protection Benefit"] = (
        raw["Maximum Drawdown"]
        - raw["Unhedged Maximum Drawdown"]
    )

    if not np.isfinite(
        raw[
            ["Return Benefit", "Protection Benefit"]
        ].to_numpy(float)
    ).all():
        raise ValueError("Nonfinite training targets")

    if "Hedge Improvement" in raw.columns:
        if not np.allclose(
            raw["Return Benefit"],
            raw["Hedge Improvement"],
            rtol=1e-10,
            atol=1e-8,
        ):
            raise ValueError(
                "Return benefit does not reconcile "
                "with Hedge Improvement"
            )

    if "Drawdown Improvement" in raw.columns:
        source = pd.to_numeric(
            raw["Drawdown Improvement"],
            errors="coerce",
        )

        protection = raw["Protection Benefit"]

        same_sign = np.allclose(
            protection,
            source,
            rtol=1e-10,
            atol=1e-8,
            equal_nan=False,
        )

        opposite_sign = np.allclose(
            protection,
            -source,
            rtol=1e-10,
            atol=1e-8,
            equal_nan=False,
        )

        if same_sign:
            print(
                "Drawdown Improvement: same sign as "
                "Protection Benefit"
            )
        elif opposite_sign:
            print(
                "Drawdown Improvement: opposite sign "
                "to Protection Benefit"
            )
        else:
            raise ValueError(
                "Drawdown Improvement cannot be reconciled "
                "with calculated Protection Benefit"
            )

        print("Drawdown sign convention: PASS")


    raw["Decision Date"] = pd.to_datetime(
        raw["Decision Date"]
    ).dt.normalize()

    raw["Outcome Date"] = pd.to_datetime(
        raw["Outcome Date"]
    ).dt.normalize()

    if (
        raw["Outcome Date"] < raw["Decision Date"]
    ).any():
        raise ValueError(
            "Outcome precedes decision date"
        )

    return x, raw


def build_model(name):
    """Construct model from the supplied inputs.

    Args:
        name: Input to the build_model operation.

    Returns:
        Computed result or status for the caller.
    """
    def impute():
        """Perform the impute operation for this research workflow.

        Returns:
            Computed result or status for the caller.
        """
        return SimpleImputer(
            strategy="median",
            keep_empty_features=True,
        )

    if name in ("OLS", "ElasticNet", "PCR", "PLS"):
        steps = [
            ("imputer", impute()),
            ("scaler", StandardScaler()),
        ]

        if name == "OLS":
            steps.append(
                ("model", LinearRegression())
            )

        elif name == "ElasticNet":
            steps.append(
                (
                    "model",
                    ElasticNet(
                        alpha=0.001,
                        l1_ratio=0.5,
                        max_iter=20000,
                    ),
                )
            )

        elif name == "PCR":
            steps.extend([
                ("pca", PCA(n_components=5)),
                ("model", LinearRegression()),
            ])

        else:
            steps.append(
                (
                    "model",
                    PLSRegression(n_components=3),
                )
            )

        return Pipeline(steps)

    if name == "RandomForest":
        return Pipeline([
            ("imputer", impute()),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=200,
                    min_samples_leaf=10,
                    max_features=0.8,
                    random_state=SEED,
                    n_jobs=-1,
                ),
            ),
        ])

    if name == "XGBoost":
        return Pipeline([
            ("imputer", impute()),
            (
                "model",
                XGBRegressor(
                    n_estimators=250,
                    max_depth=3,
                    learning_rate=0.04,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective="reg:squarederror",
                    tree_method="hist",
                    random_state=SEED,
                    n_jobs=4,
                ),
            ),
        ])

    raise ValueError(
        f"Unknown model: {name}"
    )


def fit_pair(name, x_train, data_train):
    """Fit pair on the supplied research sample.

    Args:
        name: Input to the fit_pair operation.
        x_train: Input to the fit_pair operation.
        data_train: Input to the fit_pair operation.

    Returns:
        Computed result or status for the caller.
    """
    models = {}

    for target in (
        "Return Benefit",
        "Protection Benefit",
    ):
        model = build_model(name)

        model.fit(
            x_train,
            data_train[target].to_numpy(float),
        )

        models[target] = model

    return models


def predict_pair(models, x):
    """Perform the predict pair operation for this research workflow.

    Args:
        models: Input to the predict_pair operation.
        x: Input to the predict_pair operation.

    Returns:
        Computed result or status for the caller.
    """
    return (
        np.asarray(
            models["Return Benefit"].predict(x),
            dtype=float,
        ).reshape(-1),
        np.asarray(
            models["Protection Benefit"].predict(x),
            dtype=float,
        ).reshape(-1),
    )


def make_policy(
    data,
    predicted_return,
    predicted_protection,
    risk_weight,
):
    """Construct policy from the supplied inputs.

    Args:
        data: Input to the make_policy operation.
        predicted_return: Input to the make_policy operation.
        predicted_protection: Input to the make_policy operation.
        risk_weight: Input to the make_policy operation.

    Returns:
        Computed result or status for the caller.
    """
    if (
        len(data) != len(predicted_return)
        or len(data) != len(predicted_protection)
    ):
        raise ValueError(
            "Prediction length mismatch"
        )

    frame = data.copy()

    frame["Predicted Return Benefit"] = (
        predicted_return
    )

    frame["Predicted Protection Benefit"] = (
        predicted_protection
    )

    frame["Predicted Utility"] = (
        predicted_return
        + risk_weight * predicted_protection
    )

    if not np.isfinite(
        frame["Predicted Utility"]
    ).all():
        raise ValueError(
            "Nonfinite predicted utility"
        )

    records = []

    for date, group in frame.groupby(
        "Decision Date",
        sort=True,
    ):
        baseline_return = group[
            "Unhedged Final Return"
        ].to_numpy(float)

        baseline_dd = group[
            "Unhedged Maximum Drawdown"
        ].to_numpy(float)

        if not np.allclose(
            baseline_return,
            baseline_return[0],
            atol=1e-8,
            rtol=1e-10,
        ):
            raise ValueError(
                f"Different unhedged returns on {date.date()}"
            )

        if not np.allclose(
            baseline_dd,
            baseline_dd[0],
            atol=1e-8,
            rtol=1e-10,
        ):
            raise ValueError(
                f"Different unhedged drawdowns on {date.date()}"
            )

        best = group.loc[
            group["Predicted Utility"].idxmax()
        ]

        hedge = (
            float(best["Predicted Utility"]) > 0
        )

        ret_benefit = (
            float(best["Return Benefit"])
            if hedge else 0.0
        )

        protect_benefit = (
            float(best["Protection Benefit"])
            if hedge else 0.0
        )

        records.append({
            "Decision Date": date,
            "Strategy": (
                str(best["Strategy"])
                if hedge else "No Hedge"
            ),
            "Candidate ID": (
                str(best["Candidate ID"])
                if hedge else None
            ),
            "Predicted Utility": max(
                0.0,
                float(best["Predicted Utility"]),
            ),
            "Realized Return Benefit": ret_benefit,
            "Realized Protection Benefit": protect_benefit,
            "Realized Utility": (
                ret_benefit
                + risk_weight * protect_benefit
            ),
            "Realized Return": (
                float(best["Final Return"])
                if hedge else baseline_return[0]
            ),
            "Maximum Drawdown": (
                float(best["Maximum Drawdown"])
                if hedge else baseline_dd[0]
            ),
            "Unhedged Return": baseline_return[0],
            "Unhedged Drawdown": baseline_dd[0],
        })

    return pd.DataFrame.from_records(records)


def main():
    """Run the module entry point and its workflow.
    """
    x, raw = load_data()

    years = raw["Decision Date"].dt.year

    train_mask = years.le(2018)
    val_mask = years.between(2022, 2023)
    test_mask = years.eq(2024)

    if not (
        train_mask | val_mask | test_mask
    ).all():
        raise ValueError(
            "Unexpected years in consolidated data"
        )

    for label, mask in (
        ("Train", train_mask),
        ("Validation", val_mask),
        ("2024 holdout", test_mask),
    ):
        print(
            f"{label}: {int(mask.sum())} candidates | "
            f"{raw.loc[mask, 'Decision Date'].nunique()} dates"
        )

    if not (
        train_mask.any()
        and val_mask.any()
        and test_mask.any()
    ):
        raise ValueError(
            "A chronological split is empty"
        )

    names = (
        "OLS",
        "ElasticNet",
        "PCR",
        "PLS",
        "RandomForest",
        "XGBoost",
    )

    x_train = x.loc[train_mask]
    raw_train = raw.loc[train_mask]

    x_val = x.loc[val_mask]
    raw_val = raw.loc[val_mask]

    rows = []
    validation_policies = {}

    for name in names:
        models = fit_pair(
            name,
            x_train,
            raw_train,
        )

        pred_ret, pred_protect = predict_pair(
            models,
            x_val,
        )

        mae_ret = mean_absolute_error(
            raw_val["Return Benefit"],
            pred_ret,
        )

        mae_protect = mean_absolute_error(
            raw_val["Protection Benefit"],
            pred_protect,
        )

        for weight in LAMBDAS:
            policy = make_policy(
                raw_val,
                pred_ret,
                pred_protect,
                weight,
            )

            validation_policies[
                (weight, name)
            ] = policy

            rows.append({
                "Lambda": weight,
                "Model": name,
                "Mean Realized Utility": (
                    policy["Realized Utility"].mean()
                ),
                "Mean Return Benefit": (
                    policy["Realized Return Benefit"].mean()
                ),
                "Mean Protection Benefit": (
                    policy["Realized Protection Benefit"].mean()
                ),
                "Return MAE": mae_ret,
                "Protection MAE": mae_protect,
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

    ranking = pd.DataFrame(rows).sort_values(
        [
            "Lambda",
            "Mean Realized Utility",
            "Return MAE",
            "Protection MAE",
        ],
        ascending=[
            True,
            False,
            True,
            True,
        ],
        kind="stable",
    )

    OUTPUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    ranking.to_csv(
        OUTPUT / "validation_ranking.csv",
        index=False,
    )

    print("\n--- VALIDATION RANKING ---")

    print(
        ranking[
            [
                "Lambda",
                "Model",
                "Mean Realized Utility",
                "Hedge Decisions",
            ]
        ].to_string(index=False)
    )

    final_train = train_mask | val_mask

    x_final = x.loc[final_train]
    raw_final = raw.loc[final_train]

    x_test = x.loc[test_mask]
    raw_test = raw.loc[test_mask]

    selections = []

    for weight in LAMBDAS:
        candidates = ranking.loc[
            ranking["Lambda"].eq(weight)
        ]

        selected = candidates.iloc[0]

        name = str(selected["Model"])

        val_utility = float(
            selected["Mean Realized Utility"]
        )

        # If every model has nonpositive
        # validation utility, prefer No Hedge.
        use_model = val_utility > 0.0

        if use_model:
            model_pair = fit_pair(
                name,
                x_final,
                raw_final,
            )

            predicted_ret, predicted_protect = (
                predict_pair(
                    model_pair,
                    x_test,
                )
            )

            policy = make_policy(
                raw_test,
                predicted_ret,
                predicted_protect,
                weight,
            )

        else:
            # No Hedge has zero improvement
            # relative to itself.
            policy = make_policy(
                raw_test,
                np.full(len(raw_test), -1.0),
                np.zeros(len(raw_test)),
                weight,
            )

            name = "No Hedge (validation gate)"

        policy.insert(
            1,
            "Lambda",
            weight,
        )

        policy.insert(
            2,
            "Model",
            name,
        )

        file_weight = str(weight).replace(
            ".",
            "p",
        )

        policy.to_csv(
            OUTPUT
            / f"holdout_2024_lambda_{file_weight}.csv",
            index=False,
        )

        selections.append({
            "Lambda": weight,
            "Selected Policy": name,
            "Validation Mean Utility": val_utility,
            "Holdout Mean Utility": (
                policy["Realized Utility"].mean()
            ),
            "Holdout Mean Return Benefit": (
                policy["Realized Return Benefit"].mean()
            ),
            "Holdout Mean Protection Benefit": (
                policy["Realized Protection Benefit"].mean()
            ),
            "Holdout Hedge Decisions": int(
                policy["Strategy"].ne(
                    "No Hedge"
                ).sum()
            ),
        })

    selection_table = pd.DataFrame(
        selections
    )

    selection_table.to_csv(
        OUTPUT / "selected_policies_2024.csv",
        index=False,
    )

    with (
        OUTPUT / "configuration.json"
    ).open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "risk_weights": list(LAMBDAS),
                "model_names": list(names),
                "train_years": "2014-2018",
                "validation_years": "2022-2023",
                "holdout_year": 2024,
                "random_seed": SEED,
                "selection": (
                    "highest mean realized validation "
                    "utility; No Hedge if <= 0"
                ),
                "note": (
                    "2024 was previously inspected, "
                    "so it is not an untouched test set"
                ),
            },
            fh,
            indent=2,
        )

    print("\n--- SELECTED POLICIES / 2024 ---")

    print(
        selection_table.to_string(
            index=False
        )
    )

    print("\nResults saved to:", OUTPUT)


if __name__ == "__main__":
    main()
