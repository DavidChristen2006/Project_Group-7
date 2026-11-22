"""Compare ten prediction models for hedge selection with walk-forward tests.

For every test decision date, each model is trained only on outcomes
that were already known before that date and predicts two targets for
all hedge candidates of the date:

    Hedge Improvement    (return benefit of the hedge)
    Drawdown Improvement (risk benefit of the hedge)

The candidates that satisfy the hedge constraints are ranked by

    predicted utility = predicted return + risk weight * predicted risk

and the best candidate is chosen if its predicted utility is above 0;
otherwise the decision is "No Hedge". The chosen hedge is then scored
with its actual outcomes. "Historical Mean" (the average of the training
targets) serves as the baseline against which all other models are
compared.

All MAE, benefit and utility columns are in percentage points (pp).

The functions load_dataset() and fit_predict() are also imported by
other audit scripts, so importing this module has no side effects.

Run from project root, for example:
    py -m ml.candidate_walk_forward_comparison --max-dates 6
        --models "Ridge,XGBoost" --market-features

Reads:  the dataset at DATA_PATH, via load_dataset().
Writes: nothing; results are printed only.

GenAI usage: Claude (Anthropic) was used to explain the code and to
draft, review and correct comments and docstrings. The comments were
checked and adapted by the author(s).
"""

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

# splits the dataset into training and test candidates for one date
from ml.walk_forward_validation import (
    get_walk_forward_split,
)

# DATA_PATH = location of the dataset; MARKET_FEATURES = optional
# market-state columns (see build_market_features)
from pipeline.historical_ml_feature_comparison import (
    DATA_PATH,
    MARKET_FEATURES,
)

from pipeline.historical_ml_benchmark import (
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
)

# hedge rules and the filter that removes candidates violating them
from pipeline.historical_ml_constraints import (
    HedgeConstraints,
    filter_eligible_candidates,
)


# target 1: return benefit of the hedge (hedged minus unhedged return)
TARGET_RETURN = "Hedge Improvement"

# target 2: risk benefit of the hedge (reduction of the drawdown)
TARGET_RISK = "Drawdown Improvement"

# models that can be compared; --models must use these exact names
# Historical Mean = baseline, predicts the training average
# OLS, Ridge, Lasso, Elastic Net = linear models (the last three
#     with regularization)
# PCR = principal component regression, PLS = partial least squares
# Random Forest, Gradient Boosting, XGBoost = tree ensembles
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
    """Build the preprocessing step shared by all models.

    Numeric columns: missing values are filled with 0 and an indicator
    column marks which values were missing; then all columns are
    standardized. Categorical columns: missing values are replaced by
    the most frequent category and the result is one-hot encoded.

    Args:
        numeric_features: List of numeric feature column names.

    Returns:
        Unfitted sklearn ColumnTransformer. It must be fitted on
        training data only (this happens inside make_model()).
    """

    # add_indicator=True adds one 0/1 column per feature with missing
    # values, so the model can tell "missing" from a real 0
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

    # handle_unknown="ignore": a category that did not occur in the
    # training data becomes all zeros instead of causing an error;
    # sparse_output=False returns a normal (dense) array
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
    """Create the regression model with the given name.

    The hyperparameters are fixed values and are not tuned in this
    script. random_state=42 makes models that use randomness
    reproducible.

    Args:
        name: One of MODEL_NAMES.

    Returns:
        Unfitted sklearn-compatible regressor.

    Raises:
        ValueError: if the name is not a known model.
    """

    if name == "Historical Mean":
        return DummyRegressor(strategy="mean")

    if name == "OLS":
        return LinearRegression()

    if name == "Ridge":
        return Ridge(alpha=10.0)

    # max_iter is raised so that the solver can converge
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

    # PCR: reduce the features to 5 principal components, then fit a
    # linear regression on them
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

    # shallow trees (max_depth=2) and a small learning rate keep the
    # boosting models simple, which limits overfitting on small data
    if name == "Gradient Boosting":
        return GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=2,
            random_state=42,
        )

    if name == "XGBoost":
        # imported only here, so the other models work without xgboost
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
    """Combine preprocessing and estimator into one pipeline.

    Because both steps are in one pipeline, fitting it on training data
    fits the preprocessing on training data only, so no information from
    the test data leaks into the scaling or imputation.

    Args:
        name: One of MODEL_NAMES.
        numeric_features: List of numeric feature column names.

    Returns:
        Unfitted sklearn Pipeline.
    """

    return Pipeline([
        (
            "preprocessor",
            build_preprocessor(numeric_features),
        ),
        ("estimator", build_estimator(name)),
    ])


def load_dataset():
    """Load the candidate dataset and check that it is usable.

    Verifies that all required columns exist, converts the dates to
    plain dates (no time of day), checks that every candidate key is
    unique and that both targets are finite numbers. Rows are sorted by
    decision date and candidate key.

    Returns:
        DataFrame with one row per hedge candidate, sorted and with a
        fresh index.

    Raises:
        FileNotFoundError: if the dataset file at DATA_PATH is missing.
        ValueError: if required columns are missing, candidate keys are
            duplicated, or a target is non-numeric or not finite.
    """

    if not DATA_PATH.is_file():
        raise FileNotFoundError(DATA_PATH)

    df = pd.read_parquet(DATA_PATH)

    # all columns that the models, the constraints and the decisions
    # need
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

    # set() removes columns that appear in several lists
    missing = [
        column for column in set(required)
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    # errors="raise" stops at an unparsable date instead of hiding it
    for col in ["Decision Date", "Outcome Date"]:
        df[col] = (
            pd.to_datetime(df[col], errors="raise")
            .astype("datetime64[ns]")
            .dt.normalize()
        )

    # the key identifies one observation and is used for all merges
    if not df["Candidate Key"].is_unique:
        raise ValueError("Duplicate Candidate Key.")

    # a missing or infinite target would break training
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
    """Train a model on the training data and predict the test candidates.

    Two separate models are trained with the same settings, one for the
    return benefit and one for the risk benefit.

    Args:
        name: One of MODEL_NAMES.
        train: DataFrame of training candidates (known outcomes only).
        test: DataFrame of candidates to predict.
        numeric_features: List of numeric feature column names.

    Returns:
        DataFrame with one row per test candidate: "Candidate Key",
        "Decision Date", "Strategy", the actual targets and the columns
        "Predicted Return" and "Predicted Risk Benefit" (fractions, not
        percentage points).

    Raises:
        ValueError: if the model returns the wrong number of
            predictions or non-finite values.
    """

    features = (
        numeric_features
        + CATEGORICAL_FEATURES
    )

    # start with the identifiers and the actual outcomes, then add the
    # predictions as new columns
    predictions = test[
        [
            "Candidate Key",
            "Decision Date",
            "Strategy",
            TARGET_RETURN,
            TARGET_RISK,
        ]
    ].copy()

    # (target to learn, name of the prediction column)
    for target, predicted_column in [
        (TARGET_RETURN, "Predicted Return"),
        (TARGET_RISK, "Predicted Risk Benefit"),
    ]:
        # a fresh model for every target
        model = make_model(
            name,
            numeric_features,
        )

        model.fit(
            train[features],
            train[target],
        )

        # reshape(-1) turns models that return a column (e.g. PLS)
        # into a flat array
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
    """Choose a hedge for each decision date of the test data.

    Merges the predictions onto the candidates, removes candidates that
    violate the constraints, ranks the rest by predicted utility
    (predicted return + weight * predicted risk benefit) and chooses the
    best one if its predicted utility is above 0. Otherwise the decision
    is "No Hedge", with all benefits set to 0.

    Args:
        test: DataFrame of test candidates with their actual targets.
        predictions: DataFrame as returned by fit_predict().
        constraints: HedgeConstraints that candidates must satisfy.
        weight: Weight of the risk benefit in the utility.

    Returns:
        DataFrame with one row per decision date: the decision, the
        contract details, the predicted values, the actual "Return
        Benefit" and "Risk Benefit" of the chosen hedge and the number
        of eligible candidates. Values are fractions, not percentage
        points.

    Raises:
        ValueError: if candidate keys are not one-to-one with the
            predictions (raised by the merge).
    """

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

    # candidates that violate the hedge constraints are removed
    eligible = filter_eligible_candidates(
        scored,
        constraints,
    ).copy()

    eligible["Predicted Utility"] = (
        eligible["Predicted Return"]
        + weight * eligible["Predicted Risk Benefit"]
    )

    records = []

    # one decision per decision date
    for date, group in test.groupby("Decision Date"):
        # eligible candidates of this date, best predicted utility
        # first; ties are broken alphabetically by key so that the
        # choice is reproducible
        month = eligible.loc[
            eligible["Decision Date"].eq(date)
        ].sort_values(
            ["Predicted Utility", "Candidate Key"],
            ascending=[False, True],
        )

        # default record: "No Hedge" with benefits of 0
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

        # replace the default by the best candidate, but only if its
        # predicted utility is positive
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
    """Evaluate all models on one walk-forward fold.

    A fold consists of one decision date: the models are trained on the
    training data and tested on the candidates of that date. For each
    model, the prediction error (MAE) over all test candidates and the
    outcome of the chosen hedge are recorded. The function assumes that
    the test data contains a single decision date.

    Args:
        train: DataFrame of training candidates (known outcomes only).
        test: DataFrame of test candidates of one decision date.
        model_names: List of model names from MODEL_NAMES.
        numeric_features: List of numeric feature column names.
        constraints: HedgeConstraints that candidates must satisfy.
        risk_weight: Weight of the risk benefit in the utility.

    Returns:
        List of dicts, one per model, with the decision date, the model,
        the sizes of the training and test data, "Return MAE (pp)" and
        "Risk MAE (pp)", the benefits and the utility of the chosen
        hedge (pp), the decision and the contract details. Strike and
        premium columns are fractions of the entry price; the columns
        with "(pp)" are percentage points.
    """

    records = []

    for name in model_names:
        # predictions for ALL test candidates
        predicted = fit_predict(
            name,
            train,
            test,
            numeric_features,
        )

        # decision of this model for the test date
        selected = evaluate_policy(
            test,
            predicted,
            constraints,
            risk_weight,
        )

        counts = selected["Decision"].value_counts()

        # the single decision row of this fold
        choice = selected.iloc[0]

        records.append({
            "Decision Date": test["Decision Date"].iloc[0],
            "Model": name,
            "Training Dates": train["Decision Date"].nunique(),
            "Training Candidates": len(train),
            "Test Candidates": len(test),
            # mean absolute error of the predictions over all test
            # candidates, including those that are not eligible
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
            # actual benefits of the chosen hedge ("No Hedge" = 0)
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
    """Run the walk-forward comparison and print the results.

    For each of the first --max-dates decision dates from --test-start,
    the data is split into training and test candidates, every selected
    model is trained and evaluated, and the results are printed as a
    table per fold and as a summary per model, sorted by utility.

    Command-line arguments:
        --test-start: First test date (default 2023-01-03).
        --max-dates: Number of test dates to evaluate (default 1).
        --models: Comma-separated model names from MODEL_NAMES
            (default: all ten models).
        --market-features: Add the market-state features to the
            numeric features (flag, off by default).
        --risk-weight: Weight of the risk benefit in the utility
            (default 0.5).
        --min-call-otm: Minimum distance of the call strike from the
            price as a fraction (default 0.05 = 5%).
        --max-net-debit: Maximum net premium as a fraction
            (default 0.02 = 2%).

    Raises:
        ValueError: if model names are unknown or duplicated, the
            evaluation parameters are invalid, a training outcome lies
            in the future, or no fold could be evaluated.

    Side effects:
        Prints progress and result tables; no files are written.
    """

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

    # reject empty lists, duplicates and names that do not exist
    if (
        not model_names
        or len(model_names) != len(set(model_names))
        or any(name not in MODEL_NAMES for name in model_names)
    ):
        raise ValueError(
            "Unknown or duplicate model names."
        )

    # at least one date must be evaluated and the weight must be a
    # finite, non-negative number
    if (
        args.max_dates < 1
        or not np.isfinite(args.risk_weight)
        or args.risk_weight < 0
    ):
        raise ValueError(
            "Invalid evaluation parameters."
        )

    # copy so that the shared feature list is not modified
    numeric_features = NUMERIC_FEATURES.copy()

    if args.market_features:
        numeric_features += MARKET_FEATURES

    constraints = HedgeConstraints(
        min_call_otm_pct=args.min_call_otm,
        max_net_debit_pct=args.max_net_debit,
    )

    df = load_dataset()

    start = pd.Timestamp(args.test_start).normalize()

    # the first --max-dates decision dates on or after the start date
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

        # too little training data: skip this date instead of stopping
        # (collect_decisions() raises an error in this case)
        if test.empty or n_dates < 6:
            print(f"Skipping {date}: insufficient data")
            continue

        # no training outcome may lie in the future of the test date
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

    # averages per model over all evaluated dates
    summary = results.groupby("Model").agg(
        Dates=("Decision Date", "nunique"),
        Return_MAE_pp=("Return MAE (pp)", "mean"),
        Risk_MAE_pp=("Risk MAE (pp)", "mean"),
        Return_pp=("Return Benefit (pp)", "mean"),
        Risk_pp=("Risk Benefit (pp)", "mean"),
        Utility_pp=("Utility (pp)", "mean"),
    ).reset_index()

    # best utility first; ties are broken alphabetically
    summary = summary.sort_values(
        ["Utility_pp", "Model"],
        ascending=[False, True],
    )

    print("\n--- WALK-FORWARD SUMMARY ---")
    print(summary.round(4).to_string(index=False))

    # closing notes that help to interpret the output
    print(
        "\nHistorical Mean is the baseline. "
        "2024 results are exploratory."
    )

    print("No files modified.")


if __name__ == "__main__":
    main()