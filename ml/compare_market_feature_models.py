"""Compare 20-feature and 28-feature risk-aware ML models.

The script tests whether adding eight market-state features (returns,
volatility, drawdown, distance to moving averages) to the 20 original
candidate features improves the hedge decisions of the models.

Procedure:
    1. Load both feature sets and verify that the market features were
       attached correctly (same rows, unchanged original features,
       identical market values within a decision date).
    2. Split the data chronologically: training (up to 2018),
       validation (2022-2023) and a 2024 holdout.
    3. For each feature set, model and risk weight (lambda), train on
       the training data and measure the validation utility of the
       resulting hedge decisions.
    4. For each feature set and lambda, select the model with the best
       validation utility. If no model has a positive utility, the
       selection is "No Hedge".
    5. Retrain the selected models on training plus validation data and
       evaluate them once on the 2024 holdout. The holdout is not used
       for any selection.
    6. Count on how many decision dates the two feature sets make
       different decisions.

Each model is trained twice: once for the return benefit and once for
the protection benefit (risk reduction). Lambda is the weight of the
protection benefit in the utility (see make_policy).

Run:
    py -m ml.compare_market_feature_models

Imports:
    pathlib: Path
    numpy: numerical validation and predictions
    pandas: Parquet and tabular operations
    sklearn.metrics: prediction error
    ml.train_risk_aware_models: existing training and
        candidate-selection functions

Reads:  the baseline data via load_data() and
        features_with_market.parquet in ROOT.
Writes: CSV files in OUTPUT: validation_ranking.csv,
        selected_models.csv, holdout_2024_comparison.csv,
        decision_comparison.csv and one 2024 policy file per feature
        set and lambda. Existing datasets are not modified.

GenAI usage: Claude (Anthropic) was used to explain the code and to
draft, review and correct comments and docstrings. The comments were
checked and adapted by the author(s).
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

# LAMBDAS = list of risk weights; fit_pair() trains the return and the
# protection model of one model type; predict_pair() returns both
# predictions; make_policy() chooses the hedge for each decision date
# and computes the realized results; load_data() returns the baseline
# features and the raw candidate data
from ml.train_risk_aware_models import (
    LAMBDAS,
    build_model,
    fit_pair,
    load_data,
    make_policy,
    predict_pair,
)


# folder of the AAPL dataset with the 21-day hedge horizon
ROOT = Path("data/processed/ml_datasets/aapl_h21")

# all result files of this script are written here
OUTPUT = ROOT / "market_feature_comparison"

# the eight market-state features that distinguish the 28-feature set
# from the 20-feature baseline
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

# model types that are compared
MODEL_NAMES = (
    "OLS",
    "ElasticNet",
    "PCR",
    "PLS",
    "RandomForest",
    "XGBoost",
)


def load_feature_sets():
    """Load the baseline and the enhanced feature set and validate them.

    The enhanced set must contain the same rows and the same original
    feature values as the baseline, plus the market features, which must
    be finite and identical for all candidates of a decision date.

    Returns:
        Tuple (feature_sets, raw). feature_sets is a dict with the keys
        "Baseline_20" and "Market_28" (feature DataFrames); raw is the
        raw candidate DataFrame (decision dates, outcomes).

    Raises:
        ValueError: if row counts differ, market features or original
            columns are missing or duplicated, original values changed,
            market features are not finite, differ within a decision
            date or contain missing values.
    """

    baseline, raw = load_data()

    enhanced = pd.read_parquet(
        ROOT / "features_with_market.parquet"
    )

    # reset the index so that rows can be compared by position
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

    # the original features of the enhanced set must equal the baseline
    # value by value; this also proves that the row order is the same
    # (allclose tolerates rounding, equal_nan treats NaN as equal)
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

    # rejects NaN as well as positive and negative infinity
    if not np.isfinite(
        market.to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "Invalid market-feature values."
        )

    # Market-state features must be identical
    # across candidates on the same decision date.
    # nunique(dropna=False) counts the distinct values per date and
    # feature; more than 1 means the values differ within a date
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

    # all columns to numbers; infinite values become NaN so that the
    # models never see them
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
    """Split the candidates by decision year into three periods.

    Training: up to 2018. Validation: 2022 and 2023. Holdout: 2024.
    Every row must belong to one of these periods, so the dataset must
    not contain other years.

    Args:
        raw: Raw candidate DataFrame with a datetime column
            "Decision Date".

    Returns:
        Tuple (train, validation, holdout) of boolean masks over the
        rows of raw.

    Raises:
        ValueError: if a row has a year outside the three periods or a
            period is empty.

    Side effects:
        Prints the number of candidates and dates of each period.
    """

    years = raw["Decision Date"].dt.year

    train = years.le(2018)
    validation = years.between(2022, 2023)
    holdout = years.eq(2024)

    # every row must fall into exactly one of the periods
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
    """Evaluate every feature set, model and lambda on the validation data.

    Each model is trained on the training period and predicts the
    validation period. The prediction errors (MAE) and, for every
    lambda, the realized results of the resulting hedge decisions are
    recorded.

    Args:
        feature_sets: Dict of feature DataFrames, as returned by
            load_feature_sets().
        raw: Raw candidate DataFrame.
        train: Boolean mask of the training rows.
        validation: Boolean mask of the validation rows.

    Returns:
        DataFrame with one row per feature set, model and lambda:
        validation mean utility, return and protection benefit, return
        and protection MAE and the numbers of hedge and "No Hedge"
        decisions.

    Side effects:
        Prints the progress of the training.
    """

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

            # one model for the return benefit and one for the
            # protection benefit
            models = fit_pair(
                name,
                x_train,
                raw_train,
            )

            predicted_return, predicted_protection = (
                predict_pair(models, x_val)
            )

            # prediction errors do not depend on lambda, so they are
            # computed once per model
            return_mae = mean_absolute_error(
                raw_val["Return Benefit"],
                predicted_return,
            )

            protection_mae = mean_absolute_error(
                raw_val["Protection Benefit"],
                predicted_protection,
            )

            # lambda changes only the decision rule, not the
            # predictions
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
    """Select the best model per feature set and lambda.

    The best model has the highest validation mean utility. Ties are
    broken by the lower return MAE, then the lower protection MAE and
    finally the model name, so the result is reproducible. If even the
    best model has a utility of 0 or less, the selection is "No Hedge"
    with a utility of 0.

    Args:
        ranking: DataFrame as returned by evaluate_validation().

    Returns:
        DataFrame with one row per feature set and lambda: the selected
        model, its validation utility (at least 0), the utility of the
        best ML model before the floor and its validation MAEs.
    """

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

            # best utility first; the other columns only break ties;
            # kind="stable" keeps the original order of equal rows
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
                # hedging is only chosen if it beats not hedging
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
    """Evaluate the selected models once on the 2024 holdout.

    Each selected model is retrained on training plus validation data
    and applied to 2024. For the selection "No Hedge", constant
    predictions are used that make every candidate unattractive, so the
    policy chooses "No Hedge" on every date. The 2024 policy of every
    selection is saved as a CSV file.

    Args:
        selections: DataFrame as returned by select_models().
        feature_sets: Dict of feature DataFrames.
        raw: Raw candidate DataFrame.
        train: Boolean mask of the training rows.
        validation: Boolean mask of the validation rows.
        holdout: Boolean mask of the 2024 rows.

    Returns:
        Tuple (summary, policies). summary is a DataFrame with the 2024
        mean utility, return and protection benefit and the number of
        hedge decisions per selection. policies is a dict that maps
        (feature set, lambda) to the 2024 policy DataFrame.

    Side effects:
        Writes one CSV file per selection to OUTPUT, named
        <feature set>_lambda_<lambda>_2024.csv.
    """

    summaries = []
    policies = {}

    # for the final model, the validation years are added to the
    # training data
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
            # a predicted return of -1.0 (-100%) and no protection
            # make the utility of every candidate negative, so no
            # hedge is ever chosen
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

        # "." is not wanted in a file name: 0.5 becomes 0p5
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
    """Count the 2024 decisions that differ between the two feature sets.

    For each lambda, the decisions of the baseline and the enhanced
    feature set are compared date by date using the chosen candidate.
    A date without a hedge counts as "No Hedge".

    Args:
        policies: Dict that maps (feature set, lambda) to the 2024 policy
            DataFrame, as returned by evaluate_holdout().

    Returns:
        DataFrame with one row per lambda: the number of different
        decisions and the number of decision dates.

    Raises:
        ValueError: if the two policies do not cover the same dates.
    """

    records = []

    for weight in LAMBDAS:
        # sort by date so that both policies line up row by row
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

        # a missing candidate ID means that no hedge was chosen
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

        # True on every date where the two sets chose differently
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
    """Run the comparison of the 20-feature and the 28-feature models.

    Validates the data, evaluates all models on the validation period,
    selects the best model per feature set and lambda, evaluates the
    selections on the 2024 holdout, compares the decisions of both
    feature sets and saves all results as CSV files.

    Raises:
        ValueError: if the data validation or the chronological split
            fails (raised by the called functions).

    Side effects:
        Creates OUTPUT if needed, writes the result CSV files and
        prints all result tables.
    """

    feature_sets, raw = load_feature_sets()

    print("\n--- DATASET VALIDATION ---")

    print("Candidates:", len(raw))

    print(
        "Decision dates:",
        raw["Decision Date"].nunique(),
    )

    # number of features in each set (20 and 28)
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

    # step 1: validation results of every combination
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

    # step 2: best model per feature set and lambda
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

    # step 3: one-time evaluation of the selections on 2024
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

    # step 4: how often do the two feature sets decide differently?
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