
import numpy as np
import pandas as pd

from pipeline.historical_ml_benchmark import split_dataset

from pipeline.historical_ml_risk_optimizer import (
    load_data,
    train_two_models,
    predict_two_targets,
)

from pipeline.historical_ml_constraints import (
    HedgeConstraints,
    filter_eligible_candidates,
)


RISK_WEIGHTS = [0.0, 0.5, 1.0]

CONSTRAINTS = {
    "Unrestricted": HedgeConstraints(
        min_call_otm_pct=0.0,
        max_net_debit_pct=1.0,
    ),
    "5% upside / 2% debit": HedgeConstraints(
        min_call_otm_pct=0.05,
        max_net_debit_pct=0.02,
    ),
    "10% upside / 2% debit": HedgeConstraints(
        min_call_otm_pct=0.10,
        max_net_debit_pct=0.02,
    ),
}


def recommend(predictions, constraints, risk_weight):
    """
    Apply investor constraints after prediction.
    Always preserve the No Hedge alternative.
    """
    eligible = filter_eligible_candidates(
        predictions,
        constraints,
    ).copy()

    eligible["Predicted Score"] = (
        eligible["Predicted Return Improvement"]
        + risk_weight
        * eligible["Predicted Drawdown Improvement"]
    )

    all_dates = sorted(
        predictions["Decision Date"].unique()
    )

    records = []

    for date in all_dates:
        month = eligible.loc[
            eligible["Decision Date"].eq(date)
        ]

        # No Hedge has predicted and realized
        # improvement equal to zero.
        record = {
            "Decision Date": date,
            "Decision": "No Hedge",
            "Candidate Key": None,
            "Eligible Candidates": len(month),
            "Return Benefit": 0.0,
            "Drawdown Benefit": 0.0,
            "Predicted Utility": 0.0,
        }

        if not month.empty:
            chosen = month.sort_values(
                [
                    "Predicted Score",
                    "Candidate Key",
                ],
                ascending=[False, True],
            ).iloc[0]

            if chosen["Predicted Score"] > 0:
                record.update({
                    "Decision": chosen["Strategy"],
                    "Candidate Key": chosen["Candidate Key"],
                    "Return Benefit": chosen[
                        "Hedge Improvement"
                    ],
                    "Drawdown Benefit": chosen[
                        "Drawdown Improvement"
                    ],
                    "Predicted Utility": chosen[
                        "Predicted Score"
                    ],
                })

        records.append(record)

    return pd.DataFrame(records)


def summarize(selected, period, constraint_name, weight):
    utility = (
        selected["Return Benefit"]
        + weight * selected["Drawdown Benefit"]
    )

    counts = selected["Decision"].value_counts()

    return {
        "Period": period,
        "Constraint": constraint_name,
        "Risk Weight": weight,
        "Dates": len(selected),
        "Collar": int(counts.get("Collar", 0)),
        "Protective Put": int(
            counts.get("Protective Put", 0)
        ),
        "Put Spread": int(
            counts.get("Put Spread", 0)
        ),
        "No Hedge": int(
            counts.get("No Hedge", 0)
        ),
        "Return (pp)": (
            selected["Return Benefit"].mean() * 100
        ),
        "Drawdown (pp)": (
            selected["Drawdown Benefit"].mean() * 100
        ),
        "Utility (pp)": utility.mean() * 100,
        "Min Eligible": int(
            selected["Eligible Candidates"].min()
        ),
    }


def evaluate_period(training, evaluation, period):
    return_model, risk_model, features = (
        train_two_models(
            training,
            use_market_features=False,
        )
    )

    predictions = predict_two_targets(
        evaluation,
        return_model,
        risk_model,
        features,
    )

    records = []

    for constraint_name, constraints in CONSTRAINTS.items():
        for weight in RISK_WEIGHTS:
            selected = recommend(
                predictions,
                constraints,
                weight,
            )

            records.append(
                summarize(
                    selected,
                    period,
                    constraint_name,
                    weight,
                )
            )

    return records


def main():
    df = load_data()

    train, validation, final_train, test = (
        split_dataset(df)
    )

    records = []

    records.extend(
        evaluate_period(
            train,
            validation,
            "2023 validation",
        )
    )

    records.extend(
        evaluate_period(
            final_train,
            test,
            "2024 exploratory",
        )
    )

    results = pd.DataFrame(records)

    print("\n--- CONSTRAINED ML COMPARISON ---")

    print(
        results.round(4).to_string(index=False)
    )

    print("\n--- VALIDATION ---")

    if not results["Dates"].eq(12).all():
        raise ValueError(
            "Some decision dates were lost."
        )

    if not (
        results[
            ["Collar", "Protective Put",
             "Put Spread", "No Hedge"]
        ].sum(axis=1)
        == results["Dates"]
    ).all():
        raise ValueError(
            "Strategy counts do not match dates."
        )

    if results["Min Eligible"].lt(0).any():
        raise ValueError(
            "Invalid candidate counts."
        )

    print("All decision dates accounted for.")
    print("No Hedge fallback preserved.")
    print("No historical files modified.")
    print(
        "2024 results are exploratory, "
        "not a fresh out-of-sample test."
    )


if __name__ == "__main__":
    main()
