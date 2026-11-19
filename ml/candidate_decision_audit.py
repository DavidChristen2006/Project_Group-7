
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


RISK_WEIGHT = 0.5

CONSTRAINTS = HedgeConstraints(
    min_call_otm_pct=0.05,
    max_net_debit_pct=0.02,
)

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
        CONSTRAINTS,
    ).copy()

    eligible["Predicted Utility"] = (
        eligible["Predicted Return"]
        + risk_weight
        * eligible["Predicted Risk Benefit"]
    )

    eligible["Realized Utility"] = (
        eligible["Hedge Improvement"]
        + risk_weight
        * eligible["Drawdown Improvement"]
    )

    eligible = eligible.sort_values(
        ["Predicted Utility", "Candidate Key"],
        ascending=[False, True],
    )

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
    # It is NEVER used to choose the hedge.
    oracle = max(
        0.0,
        eligible["Realized Utility"].max()
        if not eligible.empty
        else 0.0,
    )

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
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--models",
        default="Random Forest,XGBoost",
    )

    args = parser.parse_args()

    models = [
        value.strip()
        for value in args.models.split(",")
    ]

    if not models or any(not name for name in models):
        raise ValueError("Invalid model list.")

    df = load_dataset()

    dates = sorted(
        df.loc[
            (df["Decision Date"] >= "2024-01-01")
            & (df["Decision Date"] < "2025-01-01"),
            "Decision Date",
        ].unique()
    )

    records = []

    print("\n--- CANDIDATE DECISION AUDIT ---")

    for date in dates:
        train, test, excluded = (
            get_walk_forward_split(df, date)
        )

        if test.empty:
            raise ValueError(
                f"Missing test candidates: {date}"
            )

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

        for model in models:
            predictions = fit_predict(
                model,
                train,
                test,
                NUMERIC_FEATURES.copy(),
            )

            result = evaluate_decision(
                test,
                predictions,
                RISK_WEIGHT,
            )

            result["Decision Date"] = date
            result["Model"] = model

            records.append(result)

    results = pd.DataFrame(records)

    if results.duplicated(
        ["Decision Date", "Model"]
    ).any():
        raise ValueError(
            "Duplicate recommendation records."
        )

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

    monthly_columns = [
        "Decision Date",
        "Model",
        "Decision",
        "Predicted Utility (pp)",
        "Actual Utility (pp)",
        "Regret (pp)",
    ]

    print(
        results[monthly_columns]
        .round(3)
        .to_string(index=False)
    )

    print("\n--- PROBLEMATIC MONTHS: CONTRACT DETAILS ---")

    details = results.loc[
        results["Decision Date"].isin(
            AUDIT_DATES
        )
    ].copy()

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

    print(
        details[detail_columns]
        .round(3)
        .to_string(index=False)
    )

    print("\n--- PREDICTION ERRORS ---")

    hedged = results.loc[
        results["Decision"] != "No Hedge"
    ].copy()

    hedged["Utility Forecast Error (pp)"] = (
        hedged["Actual Utility (pp)"]
        - hedged["Predicted Utility (pp)"]
    )

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
