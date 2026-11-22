"""Audit the monthly hedge decisions of ML models for 2024.

For every decision date in 2024 and every model, the script trains the
model on outcomes that were already known before that date (walk-forward),
selects the eligible hedge candidate with the highest predicted utility
(or "No Hedge" if that utility is not positive) and compares the result
with the best eligible alternative in hindsight (oracle).

Utility = hedge improvement + RISK_WEIGHT * drawdown improvement.

Run from project root:
    py -m ml.<module_name> --models "Random Forest,XGBoost"

Reads:  dataset from load_dataset()
Writes: nothing; results are printed only.

GenAI usage: Claude (Anthropic) was used to explain the code and to
draft, review and correct comments and docstrings. The comments were
checked and adapted by the author(s).
"""

import argparse

import numpy as np
import pandas as pd

from ml.walk_forward_validation import (
    get_walk_forward_split,
)

from ml.candidate_walk_forward_comparison import (
    load_dataset,
    fit_predict,
)

from pipeline.historical_ml_benchmark import (
    NUMERIC_FEATURES,
)

from pipeline.historical_ml_constraints import (
    HedgeConstraints,
    filter_eligible_candidates,
)

# weight of the risk benefit in the utility; 
# return = hedge improvement, risk benefit = drawdown improvement
# predicted utility = predicted return + 0.5 x predicted risk benefit
# realized utility = actual return + 0.5 x actual risk benefit 
RISK_WEIGHT = 0.5

# only hedge candidates satisfying these restrictions
# call strike must be at least 5% out of the money
# net premium/ debit cannot exceed 2%
CONSTRAINTS = HedgeConstraints(
    min_call_otm_pct=0.05,
    max_net_debit_pct=0.02,
)

# decision dates with poor results whose contract details are printed 
AUDIT_DATES = [
    pd.Timestamp("2024-05-01"),
    pd.Timestamp("2024-06-03"),
    pd.Timestamp("2024-11-01"),
]


def evaluate_decision(
    test,
    predictions,
    risk_weight,
):
    """Choose a hedge for one decision date and score the choice.

    Merges the model predictions onto the candidates, removes candidates
    violating CONSTRAINTS, ranks the rest by predicted utility and
    picks the best one if its predicted utility is positive; otherwise
    the decision is "No Hedge". The oracle (best realized utility among
    the eligible candidates, at least 0) is used only for comparison.

    Args:
        test: DataFrame of candidates for one decision date, including
            "Candidate Key", "Hedge Improvement" and
            "Drawdown Improvement".
        predictions: DataFrame with "Candidate Key", "Predicted Return"
            and "Predicted Risk Benefit".
        risk_weight: Weight of the risk benefit in the utility.

    Returns:
        Dict with the decision, contract details, predicted and actual
        return, risk and utility, oracle utility, regret and the number
        of eligible candidates. All values except "Eligible" are in
        percentage points (fraction * 100). Regret is never negative.

    Raises:
        ValueError: if candidate keys are not one-to-one with the
            predictions (raised by the merge).
    """
    
    # add model predictions to the candidate 
    # combines each candidate in test with that candidate's model
    # prediction based on the Key, the predicted return and 
    # the predicted risk benefit (risk reduction)
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

    # candidates that violate the configured hedge constraints
    # are removed 
    eligible = filter_eligible_candidates(
        scored,
        CONSTRAINTS,
    ).copy()

    # calculate predicted utility (one score from which candidates
    # can be ranked)
    eligible["Predicted Utility"] = (
        eligible["Predicted Return"]
        + risk_weight
        * eligible["Predicted Risk Benefit"]
    )

    # calculate realized utility based on historical outcomes 
    eligible["Realized Utility"] = (
        eligible["Hedge Improvement"]
        + risk_weight
        * eligible["Drawdown Improvement"]
    )

    # sort candidates on highest predicted utility (first) and
    # if tied alphabetically 
    # the first row is therefore the model's preferred hedge 
    eligible = eligible.sort_values(
        ["Predicted Utility", "Candidate Key"],
        ascending=[False, True],
    )

    # decide whether to hedge
    # possible decisions are: 
    # 1. No Hedge if there are no eligible candidates
    # 2. No Hedge if the best candidate has predicted utility equal
    # to or below zero
    # 3. Otherwise, choose the highest-scoring eligible candidate
    if eligible.empty:
        chosen = None
    else:
        best = eligible.iloc[0]
        chosen = (
            best
            if best["Predicted Utility"] > 0
            else None
        )

    # Retrospective best eligible alternative.
    # It is NEVER used to choose the hedge, because it would have 
    # required knowing the outcomes beforehand (used for comparison)
    oracle = max(
        0.0,
        eligible["Realized Utility"].max()
        if not eligible.empty
        else 0.0,
    )

    # returns a No Hedge record: predicted and actual values are 0, 
    # oracle and regret are still computed  
    if chosen is None:
        return {
            "Decision": "No Hedge",
            "Candidate Key": None,
            "Expiry": pd.NaT,
            "Put Strike %": np.nan,
            "Call Strike %": np.nan,
            "Short Put Strike %": np.nan,
            "Net Premium %": np.nan,
            "Predicted Return (pp)": 0.0,
            "Predicted Risk (pp)": 0.0,
            "Predicted Utility (pp)": 0.0,
            "Actual Return (pp)": 0.0,
            "Actual Risk (pp)": 0.0,
            "Actual Utility (pp)": 0.0,
            "Oracle Utility (pp)": oracle * 100,
            "Regret (pp)": oracle * 100,
            "Eligible": len(eligible),
        }

    # returns the record of the chosen hedge; all returns, risks 
    # and utilities are converted to percentage points (fraction × 100) and 
    # contains: strategy, candidate key, expiry, put/call strike percentages
    # net premium, predicted return and risk, actual return and risk,
    # predicted and realized utility, oracle utility and 
    # regret (oracle minus realized utility)
    return {
        "Decision": chosen["Strategy"],
        "Candidate Key": chosen["Candidate Key"],
        "Expiry": chosen["Expiry"],
        "Put Strike %": (
            chosen["Long Put Strike Pct"] * 100
        ),
        "Call Strike %": (
            chosen["Short Call Strike Pct"] * 100
            if pd.notna(
                chosen["Short Call Strike Pct"]
            )
            else np.nan
        ),
        "Short Put Strike %": (
            chosen["Short Put Strike Pct"] * 100
            if pd.notna(
                chosen["Short Put Strike Pct"]
            )
            else np.nan
        ),
        "Net Premium %": (
            chosen["Net Premium Pct"] * 100
        ),
        "Predicted Return (pp)": (
            chosen["Predicted Return"] * 100
        ),
        "Predicted Risk (pp)": (
            chosen["Predicted Risk Benefit"] * 100
        ),
        "Predicted Utility (pp)": (
            chosen["Predicted Utility"] * 100
        ),
        "Actual Return (pp)": (
            chosen["Hedge Improvement"] * 100
        ),
        "Actual Risk (pp)": (
            chosen["Drawdown Improvement"] * 100
        ),
        "Actual Utility (pp)": (
            chosen["Realized Utility"] * 100
        ),
        "Oracle Utility (pp)": oracle * 100,
        "Regret (pp)": (
            oracle - chosen["Realized Utility"]
        ) * 100,
        "Eligible": len(eligible),
    }


def main():
    """Run the 2024 walk-forward decision audit for the chosen models.

    For each decision date in 2024 the data is split into training and
    test candidates, every model is retrained on past outcomes only and
    its decision is evaluated. Prints a model summary, the monthly
    recommendations, contract details for AUDIT_DATES and the utility
    forecast errors of the chosen hedges.

    Command-line arguments:
        --models: Comma-separated model names
            (default "Random Forest,XGBoost").

    Raises:
        ValueError: if the model list is invalid, a date has no test
            candidates, the training data contains unfinished outcomes
            or duplicate records are found.
    """
    parser = argparse.ArgumentParser()

    # read the model argument 
    parser.add_argument(
        "--models",
        default="Random Forest,XGBoost",
    )

    args = parser.parse_args()

    models = [
        value.strip()
        for value in args.models.split(",")
    ]

    # validate that the model list contains no empty names 
    if not models or any(not name for name in models):
        raise ValueError("Invalid model list.")

    # load the dataset into a pandas table 
    df = load_dataset()

    # finds unique decision dates between 2024-01-01 and 2025-01-01
    # the code therefore audits only 2024 (end date exclusive)
    dates = sorted(
        df.loc[
            (df["Decision Date"] >= "2024-01-01")
            & (df["Decision Date"] < "2025-01-01"),
            "Decision Date",
        ].unique()
    )

    records = []

    print("\n--- CANDIDATE DECISION AUDIT ---")

    # for each date, split the data and retrain each model 
    # on past outcomes only 
    for date in dates:
        train, test, excluded = (
            get_walk_forward_split(df, date)
        )

        # ensure that the test is not empty 
        if test.empty:
            raise ValueError(
                f"Missing test candidates: {date}"
            )

        # ensures that the training outcome happened before the 
        # decision date 
        if not (
            train["Outcome Date"]
            < pd.Timestamp(date)
        ).all():
            raise ValueError(
                "Unfinished outcome in training."
            )

        print(
            f"{pd.Timestamp(date).date()} | "
            f"training={len(train)} | "
            f"candidates={len(test)}"
        )

        # generate predictions and make decisions for every date and model
        for model in models:
            # produce predictions for the test candidates 
            predictions = fit_predict(
                model,
                train,
                test,
                NUMERIC_FEATURES.copy(),
            )

            # evaluates the predictions and returns one decision record 
            result = evaluate_decision(
                test,
                predictions,
                RISK_WEIGHT,
            )

            result["Decision Date"] = date
            result["Model"] = model

            # the result is added to records 
            records.append(result)

    results = pd.DataFrame(records)

    # prevents duplicate recommendation records per decision date and
    # model 
    if results.duplicated(
        ["Decision Date", "Model"]
    ).any():
        raise ValueError(
            "Duplicate recommendation records."
        )

    # the records are grouped by model:
    # number of unique decision dates
    # mean predicted utility
    # mean actual utility
    # mean regret 
    # the summary therefore shows how accurate and useful each model was
    # but includes No Hedge months as well, which count as 0
    summary = (
        results.groupby("Model")
        .agg(
            Dates=("Decision Date", "nunique"),
            Average_Predicted_Utility=(
                "Predicted Utility (pp)", "mean"
            ),
            Average_Realized_Utility=(
                "Actual Utility (pp)", "mean"
            ),
            Average_Regret=(
                "Regret (pp)", "mean"
            ),
        )
        .reset_index()
    )

    print("\n--- MODEL SUMMARY ---")

    print(
        summary.round(4).to_string(
            index=False
        )
    )

    print("\n--- MONTHLY RECOMMENDATIONS ---")

    # defines a list of columns to display containing 
    # date, model, decision, predicted utility, actual utility and regret 
    monthly_columns = [
        "Decision Date",
        "Model",
        "Decision",
        "Predicted Utility (pp)",
        "Actual Utility (pp)",
        "Regret (pp)",
    ]

    # prints the table defined above 
    print(
        results[monthly_columns]
        .round(3)
        .to_string(index=False)
    )

    print("\n--- PROBLEMATIC MONTHS: CONTRACT DETAILS ---")

    # for each date in AUDIT_DATES, the code prints the selected
    # contract details 
    details = results.loc[
        results["Decision Date"].isin(
            AUDIT_DATES
        )
    ].copy()

    # detailed contract
    detail_columns = [
        "Decision Date",
        "Model",
        "Decision",
        "Candidate Key",
        "Expiry",
        "Put Strike %",
        "Call Strike %",
        "Short Put Strike %",
        "Net Premium %",
        "Predicted Return (pp)",
        "Actual Return (pp)",
        "Predicted Risk (pp)",
        "Actual Risk (pp)",
        "Predicted Utility (pp)",
        "Actual Utility (pp)",
        "Oracle Utility (pp)",
        "Regret (pp)",
    ]

    # prints the exact hedge contract that was recommended 
    print(
        details[detail_columns]
        .round(3)
        .to_string(index=False)
    )

    print("\n--- PREDICTION ERRORS ---")

    # analyze only actual hedge selections where the decision 
    # is not equal to "No Hedge"
    hedged = results.loc[
        results["Decision"] != "No Hedge"
    ].copy()

    # calculates the utility forecast error and adds it to hedged
    hedged["Utility Forecast Error (pp)"] = (
        hedged["Actual Utility (pp)"]
        - hedged["Predicted Utility (pp)"]
    )

    # for every model, the script calculates the
    # average forecast error, the average absolute forecast error,
    # the lowest or most overoptimistic error and the highest or most
    # underestimating error 
    # positive errors mean that the hedge performed better than forecast,
    # negative errors mean that the hedge performed worse than forecast 
    print(
        hedged.groupby("Model")[
            "Utility Forecast Error (pp)"
        ]
        .agg(
            Mean="mean",
            MAE=lambda x: x.abs().mean(),
            Worst="min",
            Best="max",
        )
        .round(4)
        .to_string()
    )

    print("\nNo historical files modified.")
    print(
        "Oracle utility is retrospective only."
    )
    print(
        "2024 remains an exploratory period."
    )


if __name__ == "__main__":
    main()
