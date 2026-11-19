
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ml.risk_return_recommender import (
    PROJECT_ROOT,
    KEY_COLUMNS,
    TARGET_RETURN,
    TARGET_RISK,
    load_data,
    evaluate_policy,
)


DEFAULT_RISK_LIMITS = [0.05, 0.10, 0.15, 0.20]

STRATEGY_PRIORITY = [
    "Collar",
    "Protective Put",
    "Put Spread",
]


def resolve_path(path):
    path = Path(path)

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def get_actual_strategies(predictions):
    # Retain realized outcomes once per strategy.
    actual = predictions[
        KEY_COLUMNS + ["Strategy", "Target", "Actual"]
    ].copy()

    consistency = actual.groupby(
        KEY_COLUMNS + ["Strategy", "Target"]
    )["Actual"].agg(["min", "max"])

    if not np.allclose(
        consistency["min"],
        consistency["max"],
        atol=1e-12,
        rtol=0.0,
    ):
        raise ValueError(
            "Realized outcomes differ across models."
        )

    actual = actual.drop_duplicates(
        KEY_COLUMNS + ["Strategy", "Target"]
    )

    actual = actual.pivot(
        index=KEY_COLUMNS + ["Strategy"],
        columns="Target",
        values="Actual",
    ).reset_index()

    if actual[[TARGET_RETURN, TARGET_RISK]].isna().any().any():
        raise ValueError(
            "Missing actual return or drawdown values."
        )

    return actual


def build_benchmarks(predictions, unhedged):
    actual = get_actual_strategies(predictions)

    reference = unhedged.set_index(
        KEY_COLUMNS
    )

    records = []

    for (date, horizon), group in actual.groupby(
        KEY_COLUMNS,
        sort=True,
    ):
        key = (date, int(horizon))

        if key not in reference.index:
            raise ValueError(
                f"Missing No Hedge outcome for {key}."
            )

        no_hedge_mdd = float(
            reference.loc[
                key, "Unhedged Maximum Drawdown"
            ]
        )

        records.append({
            "Decision Date": date,
            "Horizon": int(horizon),
            "Approach": "No Hedge",
            "Selected Strategy": "No Hedge",
            "Actual Improvement": 0.0,
            "Actual Drawdown": no_hedge_mdd,
        })

        selected = None

        for strategy in STRATEGY_PRIORITY:
            matches = group.loc[
                group["Strategy"] == strategy
            ]

            if not matches.empty:
                selected = matches.iloc[0]
                break

        if selected is None:
            raise ValueError(
                f"No available strategy for {key}."
            )

        records.append({
            "Decision Date": date,
            "Horizon": int(horizon),
            "Approach": "Always Collar Rule",
            "Selected Strategy": selected["Strategy"],
            "Actual Improvement": float(
                selected[TARGET_RETURN]
            ),
            "Actual Drawdown": float(
                selected[TARGET_RISK]
            ),
        })

    return pd.DataFrame(records)


def build_frontier(
    predictions,
    unhedged,
    risk_limits,
):
    benchmarks = build_benchmarks(
        predictions,
        unhedged,
    )

    details = []
    coverage_records = []

    for risk_limit in risk_limits:
        recommendations = evaluate_policy(
            predictions,
            unhedged,
            maximum_drawdown=risk_limit,
        )

        for model, group in recommendations.groupby(
            "Model",
            sort=True,
        ):
            eligible = group.loc[
                group["Status"] == "Feasible"
            ].copy()

            coverage_records.append({
                "Risk Limit": risk_limit,
                "Model": model,
                "Decisions": len(group),
                "Feasible Decisions": len(eligible),
                "Infeasible Decisions": (
                    len(group) - len(eligible)
                ),
                "Feasibility Rate": (
                    len(eligible) / len(group)
                ),
            })

            # Use matched evaluation samples.
            # Do not hide infeasible decisions:
            # they remain in the coverage report.
            for _, selected in eligible.iterrows():
                date = selected["Decision Date"]
                horizon = int(selected["Horizon"])

                matching = benchmarks.loc[
                    (benchmarks["Decision Date"] == date)
                    & (benchmarks["Horizon"] == horizon)
                ]

                if len(matching) != 2:
                    raise ValueError(
                        "Benchmark observations are missing."
                    )

                details.append({
                    "Risk Limit": risk_limit,
                    "Model": model,
                    "Decision Date": date,
                    "Horizon": horizon,
                    "Approach": "Risk-Return Optimizer",
                    "Selected Strategy": (
                        selected["Selected Strategy"]
                    ),
                    "Actual Improvement": float(
                        selected["Actual Improvement"]
                    ),
                    "Actual Drawdown": float(
                        selected["Actual Drawdown"]
                    ),
                })

                for _, baseline in matching.iterrows():
                    details.append({
                        "Risk Limit": risk_limit,
                        "Model": model,
                        "Decision Date": date,
                        "Horizon": horizon,
                        "Approach": baseline["Approach"],
                        "Selected Strategy": (
                            baseline["Selected Strategy"]
                        ),
                        "Actual Improvement": float(
                            baseline["Actual Improvement"]
                        ),
                        "Actual Drawdown": float(
                            baseline["Actual Drawdown"]
                        ),
                    })

    details = pd.DataFrame(details)
    coverage = pd.DataFrame(coverage_records)

    if details.empty:
        raise ValueError(
            "No feasible recommendations were generated."
        )

    details["Risk Violation"] = (
        details["Actual Drawdown"]
        < -details["Risk Limit"]
    )

    summary = (
        details.groupby(
            ["Risk Limit", "Model", "Approach"]
        )
        .agg(
            Evaluated_Decisions=(
                "Actual Improvement", "size"
            ),
            Mean_Improvement=(
                "Actual Improvement", "mean"
            ),
            Mean_Drawdown=(
                "Actual Drawdown", "mean"
            ),
            Violation_Rate=(
                "Risk Violation", "mean"
            ),
        )
        .reset_index()
    )

    summary = summary.merge(
        coverage,
        on=["Risk Limit", "Model"],
        how="left",
        validate="many_to_one",
    )

    # Convert display metrics to percentages.
    summary["Risk Limit"] *= 100
    summary["Mean_Improvement"] *= 100
    summary["Mean_Drawdown"] *= 100
    summary["Violation_Rate"] *= 100
    summary["Feasibility Rate"] *= 100

    summary = summary.sort_values(
        ["Risk Limit", "Model", "Approach"]
    ).reset_index(drop=True)

    return details, summary, coverage


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate risk-return frontiers "
            "against Collar and No Hedge."
        )
    )

    parser.add_argument(
        "--evaluation-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--risk-limits",
        type=float,
        nargs="+",
        default=DEFAULT_RISK_LIMITS,
        help="Risk limits as fractions: 0.05 0.10 0.15",
    )

    args = parser.parse_args()

    risk_limits = sorted(set(args.risk_limits))

    if any(
        not np.isfinite(x) or not (0 < x <= 1)
        for x in risk_limits
    ):
        raise ValueError(
            "Risk limits must be between 0 and 1."
        )

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

    details, summary, coverage = build_frontier(
        predictions,
        unhedged,
        risk_limits,
    )

    print("\n--- RISK-RETURN FRONTIER ---")

    print(
        summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print(
        "\nRisk limit, mean improvement, "
        "mean drawdown, violation and feasibility "
        "rates are percentages."
    )

    print(
        "\nBenchmark comparisons use the same "
        "feasible decision-date/horizon combinations "
        "as each model."
    )

    details_path = (
        evaluation_dir
        / "risk_return_frontier_details.parquet"
    )

    summary_path = (
        evaluation_dir
        / "risk_return_frontier_summary.csv"
    )

    coverage_path = (
        evaluation_dir
        / "risk_return_frontier_coverage.csv"
    )

    details.to_parquet(
        details_path,
        index=False,
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    coverage.to_csv(
        coverage_path,
        index=False,
    )

    print(f"\nDetails saved to: {details_path}")
    print(f"Summary saved to: {summary_path}")
    print(f"Coverage saved to: {coverage_path}")

    print("\nRisk-return frontier completed.")


if __name__ == "__main__":
    main()
