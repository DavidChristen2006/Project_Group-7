
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ml.risk_return_recommender import (
    PROJECT_ROOT,
    KEY_COLUMNS,
    TARGET_RISK,
    load_data,
    evaluate_policy,
)


def resolve_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def adjust_risk_predictions(
    predictions,
    unhedged,
    confidence=0.90,
    min_samples=10,
):
    adjusted = predictions.copy()

    # Only maximum-drawdown forecasts are calibrated.
    risk_rows = predictions.loc[
        predictions["Target"] == TARGET_RISK
    ].copy()


    # Use the validated outcome dates from unhedged
    # to avoid duplicate Outcome Date columns.

    risk_rows = risk_rows.drop(
        columns=["Outcome Date"],
        errors="ignore",
    )

    risk_rows = risk_rows.merge(
        unhedged[KEY_COLUMNS + ["Outcome Date"]],
        on=KEY_COLUMNS,
        how="left",
        validate="many_to_one",
    )


    if risk_rows["Outcome Date"].isna().any():
        raise ValueError(
            "Missing outcome dates for risk predictions."
        )

    # Positive error means drawdown was worse
    # than the prediction.
    risk_rows["Adverse Error"] = (
        risk_rows["Predicted"] - risk_rows["Actual"]
    )

    calibration_records = []

    groups = risk_rows.groupby(
        KEY_COLUMNS + ["Model"],
        sort=True,
    )

    for (date, horizon, model), group in groups:
        # Use previous out-of-sample errors only.
        # A previous decision is insufficient unless
        # its actual hedge outcome is already known.
        history = risk_rows.loc[
            (risk_rows["Model"] == model)
            & (
                risk_rows["Hedge Horizon Days"]
                == horizon
            )
            & (risk_rows["Decision Date"] < date)
            & (risk_rows["Outcome Date"] < date)
        ]

        count = len(history)
        calibrated = count >= min_samples

        if calibrated:
            margin = max(
                0.0,
                float(
                    np.quantile(
                        history["Adverse Error"],
                        confidence,
                    )
                ),
            )
        else:
            # Keep the original prediction when
            # there is insufficient calibration data.
            margin = 0.0

        mask = (
            (adjusted["Decision Date"] == date)
            & (
                adjusted["Hedge Horizon Days"]
                == horizon
            )
            & (adjusted["Model"] == model)
            & (adjusted["Target"] == TARGET_RISK)
        )

        adjusted.loc[mask, "Predicted"] -= margin

        calibration_records.append({
            "Decision Date": date,
            "Horizon": int(horizon),
            "Model": model,
            "Calibration Samples": count,
            "Calibrated": calibrated,
            "Risk Margin": margin,
        })

    calibration = pd.DataFrame(
        calibration_records
    )

    return adjusted, calibration


def evaluate_comparison(
    predictions,
    unhedged,
    risk_limit,
    confidence,
    min_samples,
):
    adjusted, calibration = adjust_risk_predictions(
        predictions,
        unhedged,
        confidence=confidence,
        min_samples=min_samples,
    )

    original = evaluate_policy(
        predictions,
        unhedged,
        maximum_drawdown=risk_limit,
    )

    original["Policy"] = "Original"

    conservative = evaluate_policy(
        adjusted,
        unhedged,
        maximum_drawdown=risk_limit,
    )

    conservative["Policy"] = "Uncertainty Adjusted"

    details = pd.concat(
        [original, conservative],
        ignore_index=True,
    )

    details = details.merge(
        calibration,
        on=["Decision Date", "Horizon", "Model"],
        how="left",
        validate="many_to_one",
    )

    summary = (
        details.groupby(
            ["Model", "Policy"]
        )
        .agg(
            Decisions=("Status", "size"),
            Feasible=(
                "Status",
                lambda x: (x == "Feasible").sum(),
            ),
            Calibrated_Decisions=(
                "Calibrated",
                "sum",
            ),
            Mean_Improvement=(
                "Actual Improvement",
                "mean",
            ),
            Mean_Drawdown=(
                "Actual Drawdown",
                "mean",
            ),
            Violation_Rate=(
                "Realized Risk Violation",
                "mean",
            ),
        )
        .reset_index()
    )

    summary["Feasibility_Rate"] = (
        summary["Feasible"]
        / summary["Decisions"]
        * 100
    )

    for column in [
        "Mean_Improvement",
        "Mean_Drawdown",
        "Violation_Rate",
    ]:
        summary[column] *= 100

    # These realized outcome statistics are calculated
    # on each policy's feasible decisions, so compare
    # feasibility counts alongside their averages.
    return details, summary, calibration


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compare original and uncertainty-adjusted "
            "historical hedging recommendations."
        )
    )

    parser.add_argument(
        "--evaluation-dir",
        required=True,
    )

    parser.add_argument(
        "--dataset",
        required=True,
    )

    parser.add_argument(
        "--max-drawdown",
        type=float,
        default=0.10,
    )

    parser.add_argument(
        "--confidence",
        type=float,
        default=0.90,
    )

    parser.add_argument(
        "--min-samples",
        type=int,
        default=10,
    )

    args = parser.parse_args()

    if not 0 < args.max_drawdown <= 1:
        raise ValueError("Invalid drawdown limit.")

    if not 0 < args.confidence < 1:
        raise ValueError("Invalid confidence quantile.")

    if args.min_samples < 1:
        raise ValueError("Minimum samples must be positive.")

    evaluation_dir = resolve_path(
        args.evaluation_dir
    )

    dataset_path = resolve_path(
        args.dataset
    )

    predictions, unhedged = load_data(
        evaluation_dir,
        dataset_path,
    )

    details, summary, calibration = (
        evaluate_comparison(
            predictions,
            unhedged,
            risk_limit=args.max_drawdown,
            confidence=args.confidence,
            min_samples=args.min_samples,
        )
    )

    print("\n--- UNCERTAINTY-ADJUSTED RISK ---")

    print(
        summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print(
        "\nMean improvement and drawdown: "
        "percentage points."
    )

    print(
        "Violation and feasibility rates: percent."
    )

    print(
        "\nOnly feasible recommendations are included "
        "in realized-outcome averages."
    )

    print(
        "\nCalibration uses completed prior "
        "walk-forward predictions only."
    )

    details_path = (
        evaluation_dir
        / "uncertainty_comparison.parquet"
    )

    summary_path = (
        evaluation_dir
        / "uncertainty_summary.csv"
    )

    calibration_path = (
        evaluation_dir
        / "uncertainty_calibration.csv"
    )

    details.to_parquet(
        details_path,
        index=False,
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    calibration.to_csv(
        calibration_path,
        index=False,
    )

    print(f"\nSaved: {details_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {calibration_path}")


if __name__ == "__main__":
    main()
