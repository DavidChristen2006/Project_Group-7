
import numpy as np
import pandas as pd

from ml.walk_forward_validation import (
    get_walk_forward_split,
)

from ml.candidate_walk_forward_comparison import (
    load_dataset,
    fit_predict,
)

from ml.candidate_decision_audit import (
    evaluate_decision,
    RISK_WEIGHT,
)

from pipeline.historical_ml_benchmark import (
    NUMERIC_FEATURES,
)


MODELS = ["Random Forest", "XGBoost"]

CALIBRATION_START = pd.Timestamp("2023-01-01")
CALIBRATION_END = pd.Timestamp("2024-01-01")

TEST_START = pd.Timestamp("2024-01-01")
TEST_END = pd.Timestamp("2025-01-01")

# Prespecified exploratory safety margin:
# 80th percentile of historical optimism.
OPTIMISM_QUANTILE = 0.80


def collect_decisions(df, model, start, end):
    dates = sorted(
        df.loc[
            (df["Decision Date"] >= start)
            & (df["Decision Date"] < end),
            "Decision Date",
        ].unique()
    )

    records = []

    for date in dates:
        train, test, excluded = (
            get_walk_forward_split(df, date)
        )

        if train["Decision Date"].nunique() < 6:
            raise ValueError(
                f"Insufficient training dates: {date}"
            )

        if not (
            train["Outcome Date"] < pd.Timestamp(date)
        ).all():
            raise ValueError(
                "Future outcome detected in training."
            )

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

        result["Decision Date"] = pd.Timestamp(date)
        result["Model"] = model
        result["Excluded Training Rows"] = len(excluded)

        records.append(result)

    return pd.DataFrame(records)


def estimate_margin(calibration):
    hedges = calibration.loc[
        calibration["Decision"] != "No Hedge"
    ].copy()

    if len(hedges) < 5:
        raise ValueError(
            "Too few selected hedges for calibration."
        )

    optimism = (
        hedges["Predicted Utility (pp)"]
        - hedges["Actual Utility (pp)"]
    )

    margin = max(
        0.0,
        float(
            optimism.quantile(
                OPTIMISM_QUANTILE
            )
        ),
    )

    return margin, len(hedges)


def evaluate_threshold(test, margin):
    result = test.copy()

    # Original policy uses a zero threshold.
    result["Original Utility (pp)"] = (
        result["Actual Utility (pp)"]
    )

    # Conservative policy:
    # reject the recommended hedge unless
    # its predicted utility exceeds margin.
    accepted = (
        result["Decision"].ne("No Hedge")
        & (
            result["Predicted Utility (pp)"]
            > margin
        )
    )

    result["Conservative Decision"] = np.where(
        accepted,
        result["Decision"],
        "No Hedge",
    )

    result["Conservative Utility (pp)"] = np.where(
        accepted,
        result["Actual Utility (pp)"],
        0.0,
    )

    result["Conservative Return (pp)"] = np.where(
        accepted,
        result["Actual Return (pp)"],
        0.0,
    )

    result["Conservative Drawdown (pp)"] = np.where(
        accepted,
        result["Actual Risk (pp)"],
        0.0,
    )

    result["Accepted"] = accepted

    return result


def main():
    df = load_dataset()

    print("\n--- UNCERTAINTY CALIBRATION ---")
    print(
        "Calibration: 2023 walk-forward"
    )
    print(
        "Evaluation: 2024 exploratory"
    )
    print(
        f"Optimism quantile: {OPTIMISM_QUANTILE:.0%}"
    )

    summaries = []

    for model in MODELS:
        print(f"\n--- {model} ---")

        calibration = collect_decisions(
            df,
            model,
            CALIBRATION_START,
            CALIBRATION_END,
        )

        margin, n_hedges = estimate_margin(
            calibration
        )

        print(
            "Calibration dates:",
            len(calibration)
        )

        print(
            "Calibration hedge decisions:",
            n_hedges
        )

        print(
            f"Uncertainty margin: {margin:.4f} pp"
        )

        test = collect_decisions(
            df,
            model,
            TEST_START,
            TEST_END,
        )

        comparison = evaluate_threshold(
            test,
            margin,
        )

        summaries.append({
            "Model": model,
            "Margin (pp)": margin,
            "Dates": len(comparison),
            "Original Hedges": int(
                comparison["Decision"]
                .ne("No Hedge")
                .sum()
            ),
            "Conservative Hedges": int(
                comparison["Accepted"].sum()
            ),
            "Original Utility (pp)": (
                comparison[
                    "Original Utility (pp)"
                ].mean()
            ),
            "Conservative Utility (pp)": (
                comparison[
                    "Conservative Utility (pp)"
                ].mean()
            ),
            "Conservative Return (pp)": (
                comparison[
                    "Conservative Return (pp)"
                ].mean()
            ),
            "Conservative Drawdown (pp)": (
                comparison[
                    "Conservative Drawdown (pp)"
                ].mean()
            ),
        })

        print("\nMONTHLY DECISIONS")

        columns = [
            "Decision Date",
            "Decision",
            "Predicted Utility (pp)",
            "Original Utility (pp)",
            "Conservative Decision",
            "Conservative Utility (pp)",
        ]

        print(
            comparison[columns]
            .round(3)
            .to_string(index=False)
        )

    summary = pd.DataFrame(summaries)

    print("\n--- POLICY COMPARISON ---")
    print(
        summary.round(4).to_string(index=False)
    )

    print("\nNo files modified.")
    print(
        "2024 evaluation remains exploratory."
    )
    print(
        "The margin is not a guaranteed "
        "risk bound."
    )


if __name__ == "__main__":
    main()
