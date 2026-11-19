
import numpy as np
import pandas as pd

from pipeline.historical_ml_benchmark import (
    split_dataset,
)

from pipeline.historical_ml_risk_optimizer import (
    load_data,
    train_two_models,
    predict_two_targets,
    select_hedges,
)

from pipeline.historical_rule_benchmarks import (
    RULES,
    choose_candidate,
)


RISK_WEIGHTS = [0.0, 0.25, 0.5, 1.0]


def summarize_policy(
    period,
    policy,
    weight,
    return_benefits,
    drawdown_benefits,
):
    returns = np.asarray(
        return_benefits,
        dtype=float,
    )

    drawdowns = np.asarray(
        drawdown_benefits,
        dtype=float,
    )

    utility = returns + weight * drawdowns

    return {
        "Period": period,
        "Policy": policy,
        "Risk Weight": weight,
        "Dates": len(returns),
        "Return (pp)": returns.mean() * 100,
        "Drawdown (pp)": drawdowns.mean() * 100,
        "Utility (pp)": utility.mean() * 100,
    }


def evaluate_period(
    period,
    training,
    evaluation,
):
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

    monthly_groups = list(
        evaluation.groupby("Decision Date", sort=True)
    )

    rows = []

    for weight in RISK_WEIGHTS:
        # ML recommendations
        selected = select_hedges(
            predictions,
            weight,
        )

        rows.append(
            summarize_policy(
                period,
                "Random Forest",
                weight,
                selected["Realized Return Benefit"],
                selected["Realized Drawdown Benefit"],
            )
        )

        # No-hedge baseline
        rows.append(
            summarize_policy(
                period,
                "No Hedge",
                weight,
                np.zeros(len(monthly_groups)),
                np.zeros(len(monthly_groups)),
            )
        )

        # Fixed-strike strategy benchmarks
        for name, rule in RULES.items():
            returns = []
            drawdowns = []

            for date, month in monthly_groups:
                chosen = choose_candidate(
                    month,
                    rule,
                )

                if chosen is None:
                    raise ValueError(
                        f"{name} unavailable on {date}"
                    )

                returns.append(
                    chosen["Hedge Improvement"]
                )

                drawdowns.append(
                    chosen["Drawdown Improvement"]
                )

            rows.append(
                summarize_policy(
                    period,
                    name,
                    weight,
                    returns,
                    drawdowns,
                )
            )

    return rows


def main():
    df = load_data()

    train, validation, final_train, test = (
        split_dataset(df)
    )

    rows = []

    rows.extend(
        evaluate_period(
            "2023 validation",
            train,
            validation,
        )
    )

    rows.extend(
        evaluate_period(
            "2024 exploratory",
            final_train,
            test,
        )
    )

    results = pd.DataFrame(rows)

    results = results.sort_values(
        [
            "Period",
            "Risk Weight",
            "Utility (pp)",
        ],
        ascending=[True, True, False],
    )

    print("\n--- COMMON UTILITY COMPARISON ---")

    print(
        results.round(4).to_string(index=False)
    )

    print("\n--- HIGHEST REALIZED UTILITY ---")

    winners = results.loc[
        results.groupby(
            ["Period", "Risk Weight"]
        )["Utility (pp)"].idxmax()
    ]

    print(
        winners.round(4).to_string(index=False)
    )

    print(
        "\nWinners are retrospective and must "
        "not be treated as deployable policy "
        "choices without independent validation."
    )

    print("\nNo historical files modified.")


if __name__ == "__main__":
    main()
