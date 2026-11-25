
"""Six-horizon, nine-model fixed-hedge ML research and Streamlit interface."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.multioutput import MultiOutputRegressor

from train_nine_candidate_models import (
    FEATURES as OLD_FEATURES,
    model_factory,
)

ROOT = Path(__file__).resolve().parent

SOURCE = (
    ROOT
    / "data/processed/ml_datasets/"
    "six_horizon_reconciled_v1/"
    "research_eligible_candidates.parquet"
)

OUTPUT = (
    ROOT / "data/processed/ml_experiments/"
    "six_horizon_final"
)

ANCHORS = (21, 42, 63, 84, 126, 252)

MODEL_NAMES = tuple(model_factory(seed=42))

RISKS = {
    "Low": 1.5,
    "Balanced": 0.5,
    "High": 0.0,
}

FEATURES = list(OLD_FEATURES) + [
    "Hedge Horizon Days"
]

LABELS = [
    "Hedge Improvement",
    "Drawdown Improvement",
]

EVENT = [
    "Underlying RIC",
    "Decision Date",
    "Hedge Horizon Days",
]

IDENTITY = EVENT + ["Candidate ID"]


def prepare(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate completed historical candidates."""

    required = (
        set(
            IDENTITY + LABELS + [
                "Outcome Date",
                "Strategy",
                "Spot Price",
                "Long Put Ask",
                "Short Call Bid",
                "Short Put Bid",
            ]
        )
        | (
            set(OLD_FEATURES)
            - {
                "Long Put Ask Pct",
                "Short Call Bid Pct",
                "Short Put Bid Pct",
            }
        )
    )

    missing = required - set(frame.columns)

    if missing:
        raise ValueError(
            f"Missing historical columns: {sorted(missing)}"
        )

    df = frame.copy()

    for key in ("Decision Date", "Outcome Date"):
        df[key] = pd.to_datetime(
            df[key],
            errors="raise",
        ).dt.normalize()

    if (
        df[IDENTITY].isna().any().any()
        or df.duplicated(IDENTITY).any()
    ):
        raise ValueError(
            "Missing or duplicate candidate keys"
        )

    if not df["Outcome Date"].gt(
        df["Decision Date"]
    ).all():
        raise ValueError("Invalid outcome timing")

    df["Hedge Horizon Days"] = pd.to_numeric(
        df["Hedge Horizon Days"],
        errors="raise",
    ).astype(int)

    unexpected = (
        set(df["Hedge Horizon Days"].unique())
        - set(ANCHORS)
    )

    if unexpected:
        raise ValueError(
            f"Unexpected horizons: {sorted(unexpected)}"
        )

    if "Observations" in df:
        observations = pd.to_numeric(
            df["Observations"],
            errors="raise",
        )

        if not observations.eq(
            df["Hedge Horizon Days"] + 1
        ).all():
            raise ValueError(
                "Incomplete historical hedge path"
            )

    for flag in (
        "Membership Provisional",
        "Same-Day Timing Unverified",
    ):
        if flag in df:
            false = (
                df[flag]
                .astype("string")
                .str.lower()
                .isin(("false", "0"))
            )

            if not false.all():
                raise ValueError(
                    f"Unverified rows: {flag}"
                )

    if "Research Eligible" in df:
        eligible = (
            df["Research Eligible"]
            .astype("string")
            .str.lower()
            .isin(("true", "1"))
        )

        if not eligible.all():
            raise ValueError(
                "Ineligible candidate in training pool"
            )

    if "Root Identity Verified" in df:
        explicitly_false = (
            df["Root Identity Verified"]
            .astype("string")
            .str.lower()
            .isin(("false", "0"))
        )

        if explicitly_false.any():
            raise ValueError(
                "Unverified historical option roots"
            )

    if (
        df.groupby(EVENT)["Outcome Date"]
        .nunique()
        .gt(1)
        .any()
    ):
        raise ValueError(
            "Conflicting outcomes within decision"
        )

    for name in (
        "Spot Price",
        "Long Put Ask",
        "Short Call Bid",
        "Short Put Bid",
    ):
        df[name] = pd.to_numeric(
            df[name],
            errors="coerce",
        )

    if (
        df["Spot Price"].isna().any()
        or df["Spot Price"].le(0).any()
    ):
        raise ValueError("Invalid spot prices")

    for name in ("Short Call Bid", "Short Put Bid"):
        df[name] = df[name].fillna(0.0)

    for leg in (
        "Long Put Ask",
        "Short Call Bid",
        "Short Put Bid",
    ):
        df[leg + " Pct"] = (
            df[leg] / df["Spot Price"]
        )

    for name in FEATURES:
        if name == "Strategy":
            if df[name].isna().any():
                raise ValueError("Missing strategy")
            df[name] = df[name].astype(str)
        else:
            df[name] = pd.to_numeric(
                df[name],
                errors="coerce",
            ).fillna(0.0)

    for name in LABELS:
        df[name] = pd.to_numeric(
            df[name],
            errors="coerce",
        )

    numeric_features = [
        name for name in FEATURES
        if name != "Strategy"
    ]

    if not np.isfinite(
        df[LABELS].to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "Invalid historical labels"
        )

    if not np.isfinite(
        df[numeric_features].to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "Invalid training features"
        )

    if {
        "Final Return",
        "Unhedged Final Return",
    }.issubset(df.columns):

        calculated = (
            pd.to_numeric(df["Final Return"])
            - pd.to_numeric(
                df["Unhedged Final Return"]
            )
        )

        if not np.allclose(
            calculated,
            df["Hedge Improvement"],
            atol=1e-6,
            rtol=0,
        ):
            raise ValueError(
                "Return labels do not reconcile"
            )

    if {
        "Maximum Drawdown",
        "Unhedged Maximum Drawdown",
    }.issubset(df.columns):

        calculated = (
            pd.to_numeric(df["Maximum Drawdown"])
            - pd.to_numeric(
                df["Unhedged Maximum Drawdown"]
            )
        )

        if not np.allclose(
            calculated,
            df["Drawdown Improvement"],
            atol=1e-6,
            rtol=0,
        ):
            raise ValueError(
                "Drawdown labels do not reconcile"
            )

    return (
        df.sort_values(IDENTITY)
        .reset_index(drop=True)
    )


def nine_estimators():
    """Reuse the existing nine model specifications."""

    models = model_factory(seed=42)

    for pipeline in models.values():
        prep = pipeline.named_steps["features"]

        prep.transformers = [
            (
                name,
                transformer,
                (
                    list(columns)
                    + ["Hedge Horizon Days"]
                    if name == "numeric"
                    else columns
                ),
            )
            for name, transformer, columns
            in prep.transformers
        ]

    return {
        name: MultiOutputRegressor(model)
        for name, model in models.items()
    }


def split_history(df):
    """Chronological, outcome-matured evaluation."""

    dates = sorted(
        pd.Timestamp(x)
        for x in df["Decision Date"].unique()
    )

    if len(dates) < 12:
        raise ValueError(
            "At least 12 decision dates required"
        )

    validation_start = dates[
        int(len(dates) * 0.60)
    ]

    test_start = dates[
        int(len(dates) * 0.80)
    ]

    train = df.loc[
        (df["Decision Date"] < validation_start)
        & (df["Outcome Date"] < validation_start)
    ].copy()

    validation = df.loc[
        (df["Decision Date"] >= validation_start)
        & (df["Decision Date"] < test_start)
        & (df["Outcome Date"] < test_start)
    ].copy()

    train_before_test = df.loc[
        (df["Decision Date"] < test_start)
        & (df["Outcome Date"] < test_start)
    ].copy()

    test = df.loc[
        df["Decision Date"] >= test_start
    ].copy()

    if any(
        part.empty
        for part in (
            train,
            validation,
            train_before_test,
            test,
        )
    ):
        raise ValueError(
            "Empty chronological evaluation partition"
        )

    return (
        train,
        validation,
        train_before_test,
        test,
        str(validation_start.date()),
        str(test_start.date()),
    )


def decisions(frame, predictions, risk_weight):
    """Select one strategy or No Hedge per opportunity."""

    if (
        predictions.shape != (len(frame), 2)
        or not np.isfinite(predictions).all()
    ):
        raise ValueError(
            "Invalid model predictions"
        )

    temp = frame[
        IDENTITY + LABELS + ["Strategy"]
    ].copy()

    temp["Predicted Utility"] = (
        predictions[:, 0]
        + risk_weight * predictions[:, 1]
    )

    temp["Realized Utility"] = (
        temp["Hedge Improvement"]
        + risk_weight * temp["Drawdown Improvement"]
    )

    rows = []

    for key, part in temp.groupby(
        EVENT,
        sort=True,
    ):
        best = part.sort_values(
            ["Predicted Utility", "Candidate ID"],
            ascending=[False, True],
        ).iloc[0]

        hedge = (
            float(best["Predicted Utility"]) > 0
        )

        rows.append({
            "Underlying RIC": key[0],
            "Decision Date": str(key[1].date()),
            "Hedge Horizon Days": int(key[2]),
            "Strategy": (
                str(best["Strategy"])
                if hedge
                else "No Hedge"
            ),
            "Realized Utility": (
                float(best["Realized Utility"])
                if hedge
                else 0.0
            ),
            "Realized Return Improvement": (
                float(best["Hedge Improvement"])
                if hedge
                else 0.0
            ),
            "Realized Drawdown Improvement": (
                float(best["Drawdown Improvement"])
                if hedge
                else 0.0
            ),
        })

    return pd.DataFrame(rows)


def evaluate(train, evaluation):
    """Evaluate all nine models at independent decisions."""

    report = []

    for name, model in nine_estimators().items():
        print(
            f"Evaluating {name}",
            flush=True,
        )

        model.fit(
            train[FEATURES],
            train[LABELS],
        )

        predictions = np.asarray(
            model.predict(evaluation[FEATURES]),
            dtype=float,
        )

        for appetite, weight in RISKS.items():
            chosen = decisions(
                evaluation,
                predictions,
                weight,
            )

            for horizon in ("ALL", *ANCHORS):
                subset = (
                    chosen
                    if horizon == "ALL"
                    else chosen.loc[
                        chosen["Hedge Horizon Days"]
                        .eq(horizon)
                    ]
                )

                if subset.empty:
                    continue

                report.append({
                    "Model": name,
                    "Risk appetite": appetite,
                    "Horizon": horizon,
                    "Opportunities": len(subset),
                    "Stocks": int(
                        subset["Underlying RIC"]
                        .nunique()
                    ),
                    "Mean utility": float(
                        subset["Realized Utility"].mean()
                    ),
                    "Mean return improvement": float(
                        subset[
                            "Realized Return Improvement"
                        ].mean()
                    ),
                    "Mean drawdown improvement": float(
                        subset[
                            "Realized Drawdown Improvement"
                        ].mean()
                    ),
                    "Hedge frequency": float(
                        subset["Strategy"]
                        .ne("No Hedge")
                        .mean()
                    ),
                })

    return pd.DataFrame(report)


def train():
    """Train, validate, test and export the nine models."""

    if not SOURCE.is_file():
        raise FileNotFoundError(
            f"Historical dataset missing: {SOURCE}"
        )

    data = prepare(
        pd.read_parquet(SOURCE)
    )

    coverage = (
        data.groupby("Hedge Horizon Days")
        .agg(
            Candidates=("Candidate ID", "size"),
            Stocks=("Underlying RIC", "nunique"),
        )
    )

    opportunities = (
        data[EVENT]
        .drop_duplicates()
        .groupby("Hedge Horizon Days")
        .size()
    )

    coverage["Opportunities"] = opportunities

    print(
        "\n=== SIX-HORIZON TRAINING COVERAGE ==="
    )
    print(coverage.to_string())

    missing = set(ANCHORS) - set(coverage.index)

    if missing:
        raise ValueError(
            "No completed training outcomes for "
            f"horizons {sorted(missing)}. "
            "Historical stock prices alone "
            "cannot replace option-path labels."
        )

    (
        train_v,
        validation,
        train_t,
        test,
        validation_start,
        test_start,
    ) = split_history(data)

    print(
        f"\nValidation starts: {validation_start}"
    )
    print(f"Test starts: {test_start}")

    validated = evaluate(
        train_v,
        validation,
    )

    best_global = {}
    best_by_horizon = {}
    scores = {}

    for appetite in RISKS:
        overall = (
            validated.loc[
                validated["Risk appetite"].eq(appetite)
                & validated["Horizon"].eq("ALL")
            ]
            .sort_values(
                [
                    "Mean utility",
                    "Mean return improvement",
                    "Model",
                ],
                ascending=[False, False, True],
            )
        )

        best_global[appetite] = str(
            overall.iloc[0]["Model"]
        )

        best_by_horizon[appetite] = {}
        scores[appetite] = {}

        for horizon in ANCHORS:
            supported = (
                validated.loc[
                    validated[
                        "Risk appetite"
                    ].eq(appetite)
                    & validated["Horizon"]
                      .astype(str)
                      .eq(str(horizon))
                    & validated["Opportunities"].ge(8)
                    & validated["Stocks"].ge(2)
                ]
                .sort_values(
                    [
                        "Mean utility",
                        "Mean return improvement",
                        "Model",
                    ],
                    ascending=[False, False, True],
                )
            )

            selected = (
                supported.iloc[0]
                if not supported.empty
                else overall.iloc[0]
            )

            best_by_horizon[appetite][str(horizon)] = (
                str(selected["Model"])
            )

            scores[appetite][str(horizon)] = (
                float(selected["Mean utility"])
            )

    # Held-out test results do not choose the winner.
    tested = evaluate(
        train_t,
        test,
    )

    OUTPUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    files = {}

    for name, model in nine_estimators().items():
        print(
            f"Fitting final {name}",
            flush=True,
        )

        model.fit(
            data[FEATURES],
            data[LABELS],
        )

        filename = (
            name.lower().replace(" ", "_")
            + ".joblib"
        )

        joblib.dump(
            model,
            OUTPUT / filename,
        )

        files[name] = filename

    validated.to_csv(
        OUTPUT / "validation_models.csv",
        index=False,
    )

    tested.to_csv(
        OUTPUT / "test_models.csv",
        index=False,
    )

    coverage.to_csv(
        OUTPUT / "horizon_coverage.csv",
    )

    manifest = {
        "scope": "Research prototype",
        "horizons": list(ANCHORS),
        "features": FEATURES,
        "targets": LABELS,
        "risk_weights": RISKS,
        "best_model_by_risk_appetite": best_global,
        "best_by_horizon": best_by_horizon,
        "winner_validation_utility": scores,
        "files": files,
        "training_candidates": len(data),
        "training_opportunities": int(
            data[EVENT].drop_duplicates().shape[0]
        ),
        "training_stocks": int(
            data["Underlying RIC"].nunique()
        ),
        "stock_ric_list": sorted(
            data["Underlying RIC"]
            .astype(str)
            .unique()
            .tolist()
        ),
        "source_dataset": str(SOURCE),
        "coverage_by_horizon": (
            coverage.reset_index()
            .to_dict(orient="records")
        ),
        "first_training_date": str(
            data["Decision Date"].min().date()
        ),
        "last_training_date": str(
            data["Decision Date"].max().date()
        ),
        "validation_start": validation_start,
        "test_start": test_start,
        "warnings": [
            "Only six predefined horizons were trained.",
            "Other durations are approximations.",
            "Outside 21-252 sessions use a boundary proxy.",
            "Historical root and quote selection may be biased.",
            "Available historical stock coverage is not equivalent to option-label coverage.",
            "Models are research estimates, not guaranteed returns.",
        ],
    }

    (
        OUTPUT / "manifest.json"
    ).write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print(
        "\n=== BEST MODELS BY RISK APPETITE ==="
    )
    print(json.dumps(best_global, indent=2))

    print(
        "\n=== FINAL MODEL EXPORT COMPLETE ==="
    )
    print("Stock coverage:", manifest["stock_ric_list"])
    print("Model directory:", OUTPUT)


@lru_cache(maxsize=18)
def _trusted_model(filename):
    """Handle the internal trusted model step.

    Args:
        filename: Input to the _trusted_model operation.

    Returns:
        Computed result or status for the caller.
    """
    return joblib.load(filename)


def infer(snapshot, appetite):
    """Generate the nine current hedge recommendations."""

    from data_pipeline.live_research_candidate_ml import (
        make_live_candidates,
    )

    if appetite not in RISKS:
        raise ValueError(
            f"Unknown risk appetite: {appetite}"
        )

    manifest = json.loads(
        (
            OUTPUT / "manifest.json"
        ).read_text(encoding="utf-8")
    )

    if manifest["features"] != FEATURES:
        raise ValueError(
            "Model feature schema mismatch"
        )

    requested = int(
        snapshot["meta"]["research_horizon_sessions"]
    )

    effective = min(
        max(requested, min(ANCHORS)),
        max(ANCHORS),
    )

    total_shares = int(
        snapshot["inputs"].shares
    )

    covered_shares = (
        total_shares // 100
    ) * 100

    if covered_shares < 100:
        raise ValueError(
            "Standard option hedges require "
            "at least 100 shares"
        )

    if covered_shares != total_shares:
        proxy = dict(snapshot)
        proxy["inputs"] = replace(
            snapshot["inputs"],
            shares=covered_shares,
        )
        candidates = make_live_candidates(proxy)
    else:
        candidates = make_live_candidates(
            snapshot
        )

    candidates["Hedge Horizon Days"] = effective

    weight = RISKS[appetite]
    fraction = covered_shares / total_shares
    rankings = []

    for name in MODEL_NAMES:
        filename = (
            OUTPUT / manifest["files"][name]
        ).resolve()

        if filename.parent != OUTPUT.resolve():
            raise ValueError(
                "Unsafe model file path"
            )

        model = _trusted_model(
            str(filename)
        )

        predicted = np.asarray(
            model.predict(
                candidates[FEATURES]
            ),
            dtype=float,
        )

        if (
            predicted.shape != (len(candidates), 2)
            or not np.isfinite(predicted).all()
        ):
            raise ValueError(
                f"Invalid predictions from {name}"
            )

        utility = (
            predicted[:, 0]
            + weight * predicted[:, 1]
        )

        selected_index = int(
            np.argmax(utility)
        )

        hedge = (
            float(utility[selected_index]) > 0
        )

        candidate = candidates.iloc[selected_index]

        rankings.append({
            "Model": name,
            "Strategy": (
                str(candidate["Strategy"])
                if hedge
                else "No Hedge"
            ),
            "Candidate ID": (
                str(candidate["Candidate ID"])
                if hedge
                else "NO_HEDGE"
            ),
            "Forecast return improvement (pp)": (
                float(
                    100
                    * predicted[selected_index, 0]
                    * fraction
                )
                if hedge else 0.0
            ),
            "Forecast drawdown improvement (pp)": (
                float(
                    100
                    * predicted[selected_index, 1]
                    * fraction
                )
                if hedge else 0.0
            ),
            "Forecast utility (pp)": (
                float(
                    100
                    * utility[selected_index]
                    * fraction
                )
                if hedge else 0.0
            ),
        })

    return (
        candidates,
        pd.DataFrame(rankings),
        manifest,
        requested,
        effective,
    )


def render_recommendation(snapshot, appetite):
    """Default ML hedge policy and optional comparison; expiry payoffs only."""
    import streamlit as st
    from data_pipeline.live_research_candidate_ml import (
        compare_recommendations,
        expiration_payoff_summary,
    )

    st.subheader("ML hedge recommendation")

    try:
        candidates, ranking, manifest, requested, effective = infer(
            snapshot, appetite
        )
    except Exception as exc:
        st.warning(
            f"ML recommendation unavailable: {type(exc).__name__}: {exc}. "
            "Standard hedge comparisons remain accessible."
        )
        return

    nearest = min(ANCHORS, key=lambda h: abs(h - effective))
    winner = manifest["best_by_horizon"][appetite][str(nearest)]
    choice = ranking.loc[ranking["Model"].eq(winner)].iloc[0].copy()

    # Preserve the ML model's original recommendation before
    # applying the historical-validation safeguard.
    raw_choice = choice.copy()

    # The training code chooses a horizon-specific winner only when
    # validation contains at least 8 decisions and 2 stocks.
    validation = pd.read_csv(OUTPUT / "validation_models.csv")
    validation = validation.loc[validation["Risk appetite"].eq(appetite)]

    local = validation.loc[
        validation["Horizon"].astype(str).eq(str(nearest))
    ]

    enough = (
        not local.empty
        and int(local["Opportunities"].max()) >= 8
        and int(local["Stocks"].max()) >= 2
    )

    selection_scope = str(nearest) if enough else "ALL"

    record = validation.loc[
        validation["Horizon"].astype(str).eq(selection_scope)
        & validation["Model"].eq(winner)
    ].iloc[0]

    validation_utility = float(record["Mean utility"])
    conservative_no_hedge = validation_utility <= 0

    if conservative_no_hedge:
        choice["Strategy"] = "No Hedge"
        choice["Candidate ID"] = "NO_HEDGE"

        for key in (
            "Forecast return improvement (pp)",
            "Forecast drawdown improvement (pp)",
            "Forecast utility (pp)",
        ):
            choice[key] = 0.0

    st.markdown(f"### {choice['Strategy']}")

    st.caption(
        f"Chosen hedge model: {winner} | {appetite} risk appetite | "
        f"Hedge period: {requested} NYSE sessions"
    )

    if not enough:
        st.warning(
            f"H{nearest} has too little validation evidence for "
            "reliable horizon-specific model selection. "
            "Using the pooled six-horizon winner instead."
        )

    st.caption(
        f"Model-selection evidence: {int(record['Opportunities'])} "
        f"historical decisions from {int(record['Stocks'])} stocks; "
        f"mean risk-adjusted utility: {100 * validation_utility:+.2f} pp."
    )

    if conservative_no_hedge:
        st.info(
            "No Hedge is the conservative default because this "
            "model's historical validation utility did not beat "
            "remaining unhedged. Other models may recommend "
            "a put or collar; compare them below."
        )

    if conservative_no_hedge:
        st.caption(
            "The +0.00 pp values below describe the No Hedge "
            "baseline, not the ML model's original predictions."
        )

        if raw_choice["Strategy"] != "No Hedge":
            st.markdown(
                "**Best alternative proposed by the selected ML model**"
            )

            st.write(
                f"Strategy: **{raw_choice['Strategy']}**"
            )

            st.write(
                "Predicted return improvement: "
                f"{raw_choice['Forecast return improvement (pp)']:+.2f} pp"
            )

            st.write(
                "Predicted drawdown improvement: "
                f"{raw_choice['Forecast drawdown improvement (pp)']:+.2f} pp"
            )

            st.caption(
                "This is the model's original prediction. "
                "It is not the default recommendation because "
                "historical validation did not establish "
                "a positive average risk-adjusted benefit."
            )

    if requested != effective:
        st.warning(
            f"H{requested} is outside the trained 21–252-session range. "
            f"The model uses H{effective} as a proxy, not a validated "
            "forecast for that duration."
        )
    elif requested not in ANCHORS:
        st.info(
            f"H{requested} is interpolated from six predefined "
            "horizons. This interpolation is not independently validated."
        )

    n_shares = int(snapshot["inputs"].shares)
    covered = (n_shares // 100) * 100

    if covered != n_shares:
        st.caption(
            f"Standard options cover {covered} of {n_shares} shares; "
            "the remainder is unhedged."
        )

    columns = st.columns(3)

    for column, label, field in zip(
        columns,
        (
            "Return effect vs unhedged",
            "Drawdown improvement",
            "Combined utility",
        ),
        (
            "Forecast return improvement (pp)",
            "Forecast drawdown improvement (pp)",
            "Forecast utility (pp)",
        ),
    ):
        column.metric(label, f"{float(choice[field]):+.2f} pp")

    st.caption(
        "Model estimates are research-only and are not guaranteed "
        "returns. Utility combines return and downside protection."
    )

    spot = float(snapshot["inputs"].spot)
    fee = float(snapshot["meta"].get("fee_per_contract", 0.0))

    lookup = candidates.set_index(
        candidates["Candidate ID"].astype(str),
        verify_integrity=True,
    )

    chosen_id = str(choice["Candidate ID"])
    candidate = (
        None
        if chosen_id == "NO_HEDGE"
        else lookup.loc[chosen_id]
    )

    if candidate is None:
        st.caption(
            "No option is selected. See the alternative hedge "
            "recommendations below; no-hedge stock P&L is "
            "already displayed in the standard comparison."
        )

    else:
        details, scenarios = expiration_payoff_summary(
            candidate,
            spot=spot,
            shares=covered,
            opening_fee_per_contract=fee,
        )

        st.metric(
            "Net opening option cashflow (quote currency)",
            f"{details['Net Opening Cashflow']:+,.2f}",
        )

        st.caption(
            f"Put strike: {details['Long Put Strike']} | "
            f"Call strike: {details['Short Call Strike']} | "
            f"Short put strike: {details['Short Put Strike']} | "
            f"Expiry: {details['Expiry']}"
        )

        target = pd.Timestamp(
            snapshot["meta"]["research_target_date"]
        ).date()

        expiry = pd.Timestamp(details["Expiry"]).date()

        if expiry != target:
            st.warning(
                f"Requested hedge end: {target}; option expiry: {expiry}. "
                "The scenarios below are expiration payoffs, NOT "
                "liquidation values at the requested hedge end. "
                "A target-date option valuation is not available."
            )

        with st.expander("Selected strategy: expiration scenarios"):
            st.dataframe(
                scenarios.round(2),
                hide_index=True,
                use_container_width=True,
            )

            st.caption(
                "Stock shocks are hypothetical, not probabilities. "
                "Closing commissions and time-value changes are excluded."
            )

    with st.expander("Compare all nine ML hedge recommendations"):
        choices = ranking.rename(columns={
            "Strategy": "Decision",
            "Forecast utility (pp)": "Predicted improvement (pp)",
        })

        comparison = compare_recommendations(
            choices,
            candidates,
            spot=spot,
            shares=covered,
            opening_fee_per_contract=fee,
        ).merge(
            ranking[[
                "Model",
                "Forecast return improvement (pp)",
                "Forecast drawdown improvement (pp)",
            ]],
            on="Model",
            validate="one_to_one",
        )

        st.dataframe(
            comparison.drop(columns=["Candidate ID"]).round(2),
            hide_index=True,
            use_container_width=True,
        )

        st.caption(
            "These are each model's raw choices before the conservative "
            "validation veto. Different option expiries are not directly "
            "comparable at one fixed future date."
        )

        for file, heading in (
            (
                "validation_models.csv",
                "Historical validation (model selection)",
            ),
            (
                "test_models.csv",
                "Later held-out results (not model selection)",
            ),
        ):
            path = OUTPUT / file

            if not path.is_file():
                continue

            results = pd.read_csv(path)

            results = results.loc[
                results["Risk appetite"].eq(appetite)
                & results["Horizon"]
                .astype(str)
                .isin(("ALL", str(nearest)))
            ].copy()

            if results.empty:
                continue

            for name in (
                "Mean utility",
                "Mean return improvement",
                "Mean drawdown improvement",
            ):
                results[name] = (results[name] * 100).round(2)

            results["Hedge frequency"] = (
                results["Hedge frequency"] * 100
            ).round(1)

            st.markdown(f"**{heading}**")

            st.dataframe(
                results.rename(columns={
                    "Mean utility": "Mean utility (pp)",
                    "Mean return improvement": "Return improvement (pp)",
                    "Mean drawdown improvement": "Drawdown improvement (pp)",
                    "Hedge frequency": "Hedge frequency (%)",
                }),
                hide_index=True,
                use_container_width=True,
            )

    st.caption(
        f"Historical hedge training: {manifest['training_stocks']} stocks, "
        f"{manifest['training_opportunities']} stock-date-horizon "
        "opportunities over six anchor horizons. "
        "Historical option-root and corporate-action limitations remain."
    )




if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
    )

    parser.add_argument(
        "--train",
        action="store_true",
    )

    args = parser.parse_args()

    if not args.train:
        parser.error(
            "Run with --train to train "
            "the historical models."
        )

    train()
