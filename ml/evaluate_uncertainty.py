"""Compare original and uncertainty-adjusted hedging recommendations.

The recommendation policy only accepts hedges whose predicted maximum
drawdown stays within a risk limit. If the models underestimate the
drawdown, this limit is violated more often than expected. This script
tests whether making the drawdown predictions more cautious reduces such
violations.

Procedure:
    1. For each decision date, horizon and model, measure how wrong the
       model's earlier drawdown predictions were (predicted minus
       actual; positive = the drawdown was worse than predicted). Only
       predictions whose outcome was already known before the decision
       date are used.
    2. Set the risk margin to the chosen quantile (--confidence) of
       these errors, but not below 0, and subtract it from the
       predicted drawdown. With fewer than --min-samples earlier
       predictions, the prediction stays unchanged (margin 0).
    3. Run the recommendation policy once with the original and once
       with the adjusted predictions.
    4. Summarize feasibility, realized improvement, realized drawdown
       and the rate of risk violations per model and policy.

Only maximum-drawdown predictions are adjusted; return predictions are
not changed. Improvement, drawdown and the violation and feasibility
rates are shown in percentage points or percent.

Run from project root:
    py -m ml.<module_name> --evaluation-dir <folder> --dataset <file>

Reads:  the walk-forward predictions and the dataset, via load_data().
Writes: uncertainty_comparison.parquet, uncertainty_summary.csv and
        uncertainty_calibration.csv in the evaluation folder. The
        parquet file is the input of the matched-comparison script.

GenAI usage: Claude (Anthropic) was used to explain the code and to
draft, review and correct comments and docstrings. The comments were
checked and adapted by the author(s).
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# PROJECT_ROOT = project folder; KEY_COLUMNS = columns that identify a
# decision (date and horizon); TARGET_RISK = name of the drawdown
# target; load_data() loads predictions and unhedged results;
# evaluate_policy() applies the recommendation policy
from ml.risk_return_recommender import (
    PROJECT_ROOT,
    KEY_COLUMNS,
    TARGET_RISK,
    load_data,
    evaluate_policy,
)


def resolve_path(path):
    """Turn a relative path into a path inside the project.

    Args:
        path: Absolute or relative path (string or Path).

    Returns:
        The path itself if it is absolute, otherwise the path relative
        to PROJECT_ROOT.
    """

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
    """Make the drawdown predictions more cautious using past errors.

    For each decision date, horizon and model, the margin is the
    confidence quantile of the adverse errors (predicted minus actual
    drawdown) of earlier predictions of the same model and horizon, but
    at least 0. Only predictions with a decision date and an outcome
    date before the current decision date are used, so no future
    information enters the margin. The margin is subtracted from the
    predicted drawdown. The errors of all strategies are pooled.

    Args:
        predictions: DataFrame of walk-forward predictions with the
            columns "Target", "Model", "Predicted", "Actual", the
            decision date and the horizon.
        unhedged: DataFrame with the validated "Outcome Date" for each
            decision date and horizon.
        confidence: Quantile of the adverse errors used as margin
            (default 0.90).
        min_samples: Minimum number of earlier prediction rows needed
            for a calibration (default 10). Rows, not decisions, are
            counted: each decision has one row per strategy.

    Returns:
        Tuple (adjusted, calibration). adjusted is a copy of the
        predictions with adjusted drawdown predictions. calibration is
        a DataFrame with one row per decision date, horizon and model:
        number of calibration samples, whether a calibration was done
        and the risk margin that was applied.

    Raises:
        ValueError: if an outcome date is missing for a risk prediction.
    """

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

    # many_to_one: several strategies share one decision, but each
    # decision has exactly one outcome date
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
            # quantile of the past errors; negative values are cut off
            # because the adjustment must never make a prediction
            # less cautious
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

        # select the prediction rows of this decision, horizon and
        # model, and subtract the margin from the predicted drawdown
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
    """Run the recommendation policy with original and adjusted predictions.

    Args:
        predictions: DataFrame of walk-forward predictions.
        unhedged: DataFrame with the unhedged results and outcome dates.
        risk_limit: Maximum drawdown limit for a recommended hedge,
            passed to evaluate_policy().
        confidence: Quantile used for the risk margin.
        min_samples: Minimum number of earlier prediction rows needed
            for a calibration.

    Returns:
        Tuple (details, summary, calibration). details has one row per
        decision and policy ("Original" or "Uncertainty Adjusted") with
        the calibration data. summary has one row per model and policy:
        decisions, feasible decisions, calibrated decisions, mean
        improvement and drawdown (percentage points) and violation and
        feasibility rates (percent). calibration is the table from
        adjust_risk_predictions().
    """

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

    # attach the margin that was used for each decision
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

    # share of decisions with a feasible recommendation, in percent
    summary["Feasibility_Rate"] = (
        summary["Feasible"]
        / summary["Decisions"]
        * 100
    )

    # convert fractions into percentage points or percent
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
    """Compare the original and the uncertainty-adjusted policy.

    Reads and validates the arguments, loads the data, runs the
    comparison, prints the summary and saves the results.

    Command-line arguments:
        --evaluation-dir: Folder with the walk-forward predictions;
            results are saved here too (required).
        --dataset: Consolidated historical ML dataset (required).
        --max-drawdown: Drawdown limit as a fraction (default 0.10,
            allowed range above 0 up to 1).
        --confidence: Quantile of past errors used as margin
            (default 0.90, allowed range between 0 and 1).
        --min-samples: Minimum number of earlier prediction rows for a
            calibration (default 10, at least 1).

    Raises:
        ValueError: if an argument is outside its allowed range or the
            data is inconsistent (raised by the called functions).

    Side effects:
        Prints the summary and writes three result files to the
        evaluation folder.
    """

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

    # reject values that would make the calculation meaningless
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