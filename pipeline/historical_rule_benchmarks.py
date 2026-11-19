
import numpy as np
import pandas as pd

from pipeline.historical_ml_risk_optimizer import load_data


RULES = {
    "Protective Put 90": {
        "strategy": "Protective Put",
        "put": 0.90,
        "call": None,
        "short_put": None,
    },
    "Collar 90/110": {
        "strategy": "Collar",
        "put": 0.90,
        "call": 1.10,
        "short_put": None,
    },
    "Put Spread 90/80": {
        "strategy": "Put Spread",
        "put": 0.90,
        "call": None,
        "short_put": 0.80,
    },
}


def choose_candidate(month, rule):
    candidates = month.loc[
        month["Strategy"].eq(rule["strategy"])
    ].copy()

    if candidates.empty:
        return None

    candidates = candidates.loc[
        candidates["Long Put Strike Pct"].le(1.0)
    ]

    if rule["call"] is not None:
        candidates = candidates.loc[
            candidates["Short Call Strike Pct"].ge(1.0)
        ]

    if rule["short_put"] is not None:
        candidates = candidates.loc[
            candidates["Short Put Strike Pct"]
            < candidates["Long Put Strike Pct"]
        ]

    if candidates.empty:
        return None

    # Minimize total strike deviation from
    # the predetermined rule.
    candidates["Strike Distance"] = (
        candidates["Long Put Strike Pct"]
        .sub(rule["put"]).abs()
    )

    if rule["call"] is not None:
        candidates["Strike Distance"] += (
            candidates["Short Call Strike Pct"]
            .sub(rule["call"]).abs()
        )

    if rule["short_put"] is not None:
        candidates["Strike Distance"] += (
            candidates["Short Put Strike Pct"]
            .sub(rule["short_put"]).abs()
        )

    # Deterministic ranking, using no
    # realized future outcome variables.
    candidates = candidates.sort_values(
        [
            "Strike Distance",
            "Days After Exit",
            "Candidate Key",
        ],
        ascending=[True, True, True],
    )

    return candidates.iloc[0]


def evaluate_rules(df):
    records = []

    for period, start, end in [
        ("2023 validation", "2023-01-01", "2024-01-01"),
        ("2024 exploratory", "2024-01-01", "2025-01-01"),
    ]:
        period_data = df.loc[
            (df["Decision Date"] >= start)
            & (df["Decision Date"] < end)
        ]

        for date, month in period_data.groupby(
            "Decision Date"
        ):
            for rule_name, rule in RULES.items():
                chosen = choose_candidate(month, rule)

                if chosen is None:
                    records.append({
                        "Period": period,
                        "Decision Date": date,
                        "Rule": rule_name,
                        "Available": False,
                        "Return Benefit (pp)": np.nan,
                        "Drawdown Benefit (pp)": np.nan,
                    })
                    continue

                records.append({
                    "Period": period,
                    "Decision Date": date,
                    "Rule": rule_name,
                    "Available": True,
                    "Return Benefit (pp)": (
                        chosen["Hedge Improvement"] * 100
                    ),
                    "Drawdown Benefit (pp)": (
                        chosen["Drawdown Improvement"] * 100
                    ),
                })

    return pd.DataFrame(records)


def main():
    df = load_data()

    results = evaluate_rules(df)

    summary = results.groupby(
        ["Period", "Rule"]
    ).agg(
        Decision_Dates=("Available", "size"),
        Available_Dates=("Available", "sum"),
        Mean_Return_Benefit=(
            "Return Benefit (pp)", "mean"
        ),
        Mean_Drawdown_Benefit=(
            "Drawdown Benefit (pp)", "mean"
        ),
    ).reset_index()

    print("\n--- RULE-BASED HEDGING BENCHMARK ---")
    print(
        summary.round(4).to_string(index=False)
    )

    print("\n--- MONTHLY RULE RESULTS ---")
    print(
        results.round(4).to_string(index=False)
    )

    print("\nNo historical files modified.")


if __name__ == "__main__":
    main()
