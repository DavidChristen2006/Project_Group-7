
"""Read-only diagnostics for the held-out 63-day ML benchmark.

Uses existing predictions; does not retrain or modify models.
"""

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

INPUT = (
    ROOT
    / "data/processed/ml_baselines/equity_63d_v1"
    / "heldout_test_predictions.csv"
)

OUTPUT = (
    ROOT
    / "data/processed/ml_baselines/equity_63d_v1"
    / "diagnostics"
)

BLOCK_SIZE = 3
BOOTSTRAP_REPLICATIONS = 2000
SEED = 42

LOSS_TARGET = "target_max_loss_from_entry_63d"


def moving_block_interval(
    part: pd.DataFrame,
) -> tuple[float, float]:
    """Resample contiguous decision dates, keeping equities together."""

    per_date = part.groupby("Decision Date").agg(
        n=("Actual", "size"),
        delta_sum=("Absolute Error Improvement", "sum"),
    ).sort_index()

    n_dates = len(per_date)

    if n_dates < BLOCK_SIZE + 1:
        raise ValueError(
            "Too few decision dates for block resampling"
        )

    rng = np.random.default_rng(SEED)

    starts = np.arange(
        n_dates - BLOCK_SIZE + 1
    )

    sums = per_date[
        "delta_sum"
    ].to_numpy(dtype=float)

    sizes = per_date[
        "n"
    ].to_numpy(dtype=float)

    boot = np.empty(
        BOOTSTRAP_REPLICATIONS,
        dtype=float,
    )

    for b in range(BOOTSTRAP_REPLICATIONS):
        chosen = []

        while len(chosen) < n_dates:
            start = int(
                rng.choice(starts)
            )

            chosen.extend(
                range(start, start + BLOCK_SIZE)
            )

        indices = np.asarray(
            chosen[:n_dates],
            dtype=int,
        )

        boot[b] = (
            sums[indices].sum()
            / sizes[indices].sum()
        )

    return (
        float(np.quantile(boot, 0.025)),
        float(np.quantile(boot, 0.975)),
    )


def model_metrics(
    part: pd.DataFrame,
) -> dict:

    actual = part[
        "Actual"
    ].to_numpy(dtype=float)

    ml = part[
        "Predicted"
    ].to_numpy(dtype=float)

    baseline = part[
        "Train-Median Baseline"
    ].to_numpy(dtype=float)

    model_mae = float(
        np.mean(np.abs(actual - ml))
    )

    base_mae = float(
        np.mean(np.abs(actual - baseline))
    )

    model_rmse = float(
        np.sqrt(np.mean((actual - ml) ** 2))
    )

    base_rmse = float(
        np.sqrt(np.mean((actual - baseline) ** 2))
    )

    return {
        "Rows": len(part),
        "Decision Dates": part[
            "Decision Date"
        ].nunique(),
        "Model MAE": model_mae,
        "Baseline MAE": base_mae,
        "MAE Improvement": base_mae - model_mae,
        "MAE Skill Percent": (
            100.0 * (base_mae - model_mae) / base_mae
            if base_mae > 0
            else np.nan
        ),
        "Model RMSE": model_rmse,
        "Baseline RMSE": base_rmse,
        "Mean Error (predicted - actual)": float(
            np.mean(ml - actual)
        ),
    }


def main() -> None:
    if not INPUT.is_file():
        raise FileNotFoundError(INPUT)

    frame = pd.read_csv(INPUT)

    required = {
        "Decision Date",
        "Underlying RIC",
        "target_exit_date",
        "Target",
        "Selected Model",
        "Actual",
        "Predicted",
        "Train-Median Baseline",
    }

    missing = required - set(frame.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    for column in (
        "Decision Date",
        "target_exit_date",
    ):
        frame[column] = pd.to_datetime(
            frame[column],
            errors="raise",
        ).dt.normalize()

    for column in (
        "Actual",
        "Predicted",
        "Train-Median Baseline",
    ):
        frame[column] = pd.to_numeric(
            frame[column],
            errors="raise",
        )

        if not np.isfinite(
            frame[column].to_numpy(dtype=float)
        ).all():
            raise ValueError(
                f"Nonfinite values in {column}"
            )

    keys = [
        "Target",
        "Decision Date",
        "Underlying RIC",
    ]

    if (
        frame[keys].isna().any().any()
        or frame.duplicated(keys).any()
    ):
        raise ValueError(
            "Missing or duplicate prediction keys"
        )

    if (
        frame["target_exit_date"]
        <= frame["Decision Date"]
    ).any():
        raise ValueError(
            "Invalid target exit dates"
        )

    frame["Model Absolute Error"] = (
        frame["Predicted"] - frame["Actual"]
    ).abs()

    frame["Baseline Absolute Error"] = (
        frame["Train-Median Baseline"]
        - frame["Actual"]
    ).abs()

    frame["Absolute Error Improvement"] = (
        frame["Baseline Absolute Error"]
        - frame["Model Absolute Error"]
    )

    OUTPUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = []
    monthly = []

    for target, part in frame.groupby(
        "Target",
        sort=True,
    ):
        if part["Selected Model"].nunique() != 1:
            raise ValueError(
                f"Multiple selected models for {target}"
            )

        interval = moving_block_interval(
            part
        )

        metrics = model_metrics(
            part
        )

        metrics.update({
            "Target": target,
            "Selected Model": (
                part["Selected Model"].iloc[0]
            ),
            "Block Bootstrap Low (MAE Improvement)": (
                interval[0]
            ),
            "Block Bootstrap High (MAE Improvement)": (
                interval[1]
            ),
        })

        rows.append(metrics)

        for date, date_part in part.groupby(
            "Decision Date",
            sort=True,
        ):
            monthly.append({
                "Target": target,
                "Decision Date": date,
                **model_metrics(date_part),
            })

    summary = pd.DataFrame(
        rows
    ).sort_values("Target")

    monthly_df = pd.DataFrame(
        monthly
    ).sort_values(
        ["Target", "Decision Date"]
    )

    loss = frame.loc[
        frame["Target"].eq(LOSS_TARGET)
    ].copy()

    if loss.empty:
        raise ValueError(
            "Maximum-loss target was not found"
        )

    tail = []

    for threshold in (
        0.10,
        0.15,
        0.20,
    ):
        reality = (
            loss["Actual"] >= threshold
        )

        model_alert = (
            loss["Predicted"] >= threshold
        )

        baseline_alert = (
            loss["Train-Median Baseline"] >= threshold
        )

        for name, alert in (
            ("Selected ML", model_alert),
            ("Train median", baseline_alert),
        ):
            tp = int(
                (reality & alert).sum()
            )

            fp = int(
                (~reality & alert).sum()
            )

            fn = int(
                (reality & ~alert).sum()
            )

            tail.append({
                "Loss Threshold": threshold,
                "Predictor": name,
                "Rows": len(loss),
                "Actual Breaches": int(
                    reality.sum()
                ),
                "Predicted Alerts": int(
                    alert.sum()
                ),
                "True Positives": tp,
                "False Positives": fp,
                "False Negatives": fn,
                "Recall": (
                    tp / (tp + fn)
                    if tp + fn
                    else np.nan
                ),
                "Precision": (
                    tp / (tp + fp)
                    if tp + fp
                    else np.nan
                ),
            })

    loss["Underprediction"] = (
        loss["Actual"] - loss["Predicted"]
    )

    worst = (
        loss.sort_values(
            "Underprediction",
            ascending=False,
        )
        .head(25)
        .copy()
    )

    summary.to_csv(
        OUTPUT / "overall_and_block_bootstrap.csv",
        index=False,
    )

    monthly_df.to_csv(
        OUTPUT / "metrics_by_decision_date.csv",
        index=False,
    )

    pd.DataFrame(
        tail
    ).to_csv(
        OUTPUT / "loss_threshold_diagnostics.csv",
        index=False,
    )

    worst.to_csv(
        OUTPUT / "largest_max_loss_underpredictions.csv",
        index=False,
    )

    print(
        "\n--- HOLDOUT ERROR AND DATE-BLOCK DIAGNOSTICS ---"
    )

    print(
        f"Rows: {len(frame):,}; "
        f"distinct decision dates: "
        f"{frame['Decision Date'].nunique()}"
    )

    print(
        f"Block size: {BLOCK_SIZE} "
        "consecutive monthly decisions"
    )

    print(
        "\nPer-target results "
        "(positive MAE improvement favors ML):"
    )

    print(
        summary[
            [
                "Target",
                "Rows",
                "Model MAE",
                "Baseline MAE",
                "MAE Skill Percent",
                "Block Bootstrap Low (MAE Improvement)",
                "Block Bootstrap High (MAE Improvement)",
            ]
        ].to_string(index=False)
    )

    print(
        "\nLoss threshold diagnostics:"
    )

    print(
        pd.DataFrame(tail)[
            [
                "Loss Threshold",
                "Predictor",
                "Actual Breaches",
                "Predicted Alerts",
                "Recall",
                "Precision",
            ]
        ].to_string(index=False)
    )

    print(
        "\nLargest model underpredictions:"
    )

    print(
        worst[
            [
                "Decision Date",
                "Underlying RIC",
                "Actual",
                "Predicted",
                "Train-Median Baseline",
                "Underprediction",
            ]
        ].head(10).to_string(index=False)
    )

    print(
        "\nSaved:",
        OUTPUT,
    )

    print(
        "Diagnostic only. Do not tune model selection "
        "against the held-out test."
    )


if __name__ == "__main__":
    main()
