"""Compare the original and the uncertainty-adjusted policy on matched decisions.

Only decisions that are feasible under both policies are used, so both
policies are judged on exactly the same decisions. For each model, the
script reports the mean improvement, the mean drawdown, the share of
decisions with a realized risk violation and the number of decisions
where the selected strategy changed. Improvement and drawdown are in
percentage points, the violation rate in percent.

Run from project root:
    py -m ml.compare_uncertainty_policies --evaluation-dir ml

Reads:  uncertainty_comparison.parquet in the evaluation folder.
Writes: uncertainty_matched_summary.csv in the evaluation folder.

GenAI usage: Claude (Anthropic) was used to explain the code and to
draft, review and correct comments and docstrings. The comments were
checked and adapted by the author(s).
"""

import argparse
from pathlib import Path

import pandas as pd


# two folders above this file
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main():
    """Build and save the matched comparison of the two policies.

    Command-line arguments:
        --evaluation-dir: Folder with uncertainty_comparison.parquet
            (required). A relative path is resolved from PROJECT_ROOT.

    Raises:
        ValueError: if no decision is feasible under both policies, or if
            the policies cannot be paired one-to-one (raised by merge).

    Side effects:
        Prints the summary table and writes the summary CSV file.
    """

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--evaluation-dir",
        required=True,
    )

    args = parser.parse_args()

    evaluation_dir = Path(args.evaluation_dir)

    # relative paths are taken from the project root
    if not evaluation_dir.is_absolute():
        evaluation_dir = PROJECT_ROOT / evaluation_dir

    path = (
        evaluation_dir / "uncertainty_comparison.parquet"
    )

    df = pd.read_parquet(path)

    # one decision is identified by date, horizon and model
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

    # put both policies side by side, one row per decision;
    # validate="one_to_one" fails if a decision occurs more than once
    paired = original.merge(
        adjusted,
        on=keys,
        suffixes=("_Original", "_Adjusted"),
        validate="one_to_one",
    )

    # keep only decisions that are feasible under both policies,
    # so the comparison is fair
    paired = paired.loc[
        (paired["Status_Original"] == "Feasible")
        & (paired["Status_Adjusted"] == "Feasible")
    ].copy()

    if paired.empty:
        raise ValueError(
            "No common feasible decisions."
        )

    records = []

    # one summary row per model and policy
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
                # share of decisions whose realized risk exceeded
                # the allowed limit, in percent
                "Violation Rate": (
                    group[
                        f"Realized Risk Violation_{suffix}"
                    ].astype(bool).mean() * 100
                ),
                # number of decisions where the two policies chose
                # different strategies (same value in both rows)
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