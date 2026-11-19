
import argparse
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--evaluation-dir",
        required=True,
    )

    args = parser.parse_args()

    evaluation_dir = Path(args.evaluation_dir)

    if not evaluation_dir.is_absolute():
        evaluation_dir = PROJECT_ROOT / evaluation_dir

    path = (
        evaluation_dir / "uncertainty_comparison.parquet"
    )

    df = pd.read_parquet(path)

    keys = [
        "Decision Date",
        "Horizon",
        "Model",
    ]

    original = df.loc[
        df["Policy"] == "Original"
    ].copy()

    adjusted = df.loc[
        df["Policy"] == "Uncertainty Adjusted"
    ].copy()

    paired = original.merge(
        adjusted,
        on=keys,
        suffixes=("_Original", "_Adjusted"),
        validate="one_to_one",
    )

    paired = paired.loc[
        (paired["Status_Original"] == "Feasible")
        & (paired["Status_Adjusted"] == "Feasible")
    ].copy()

    if paired.empty:
        raise ValueError(
            "No common feasible decisions."
        )

    records = []

    for model, group in paired.groupby("Model"):
        for policy, suffix in [
            ("Original", "Original"),
            ("Uncertainty Adjusted", "Adjusted"),
        ]:
            records.append({
                "Model": model,
                "Policy": policy,
                "Matched Decisions": len(group),
                "Mean Improvement": (
                    group[
                        f"Actual Improvement_{suffix}"
                    ].mean() * 100
                ),
                "Mean Drawdown": (
                    group[
                        f"Actual Drawdown_{suffix}"
                    ].mean() * 100
                ),
                "Violation Rate": (
                    group[
                        f"Realized Risk Violation_{suffix}"
                    ].astype(bool).mean() * 100
                ),
                "Changed Selections": (
                    (
                        group[
                            "Selected Strategy_Original"
                        ]
                        != group[
                            "Selected Strategy_Adjusted"
                        ]
                    ).sum()
                ),
            })

    summary = pd.DataFrame(records)

    print("\n--- MATCHED UNCERTAINTY COMPARISON ---")

    print(
        summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    output_path = (
        evaluation_dir
        / "uncertainty_matched_summary.csv"
    )

    summary.to_csv(
        output_path,
        index=False,
    )

    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
