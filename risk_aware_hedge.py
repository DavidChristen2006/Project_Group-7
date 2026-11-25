
"""Risk-aware extension of the existing six_horizon_ml hedge recommender.

Trains option-only and market-risk-enhanced versions of the EXISTING nine models
on identical historical decisions. Uses previous-session adjusted market data.
No history downloads, no new option backtests, no change to existing exports.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from dataclasses import replace
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import six_horizon_ml as base
from data_pipeline.build_equity_features import calculate_window


ROOT = Path(__file__).resolve().parent

DB = (
    ROOT
    / "data/processed/unified_research_store_v1/"
    "master_research.sqlite"
)

OUT = (
    ROOT
    / "data/processed/ml_experiments/"
    "risk_aware_hedge_final"
)

# Keep the feature set small because actual option-outcome
# coverage is substantially narrower than stock-price coverage.
RISK_FEATURES = (
    "momentum_21d",
    "realized_vol_21d",
    "momentum_63d",
    "realized_vol_63d",
    "max_drawdown_63d",
    "daily_es95_63d",
    "beta_spy_63d",
    "realized_vol_252d",
)

EXTENDED_FEATURES = list(base.FEATURES) + list(RISK_FEATURES)


def price_history(conn, ric):
    """Perform the price history operation for this research workflow.

    Args:
        conn: Input to the price_history operation.
        ric: Input to the price_history operation.

    Returns:
        Computed result or status for the caller.
    """
    frame = pd.read_sql_query(
        "SELECT date, close FROM prices "
        "WHERE ric=? AND mode='adjusted' ORDER BY date",
        conn,
        params=(ric,),
    )

    if frame.empty:
        return pd.Series(dtype=float)

    frame["date"] = pd.to_datetime(
        frame["date"],
        errors="raise",
    ).dt.normalize()

    if frame["date"].duplicated().any():
        raise ValueError(
            f"Duplicate adjusted closes for {ric}"
        )

    return pd.Series(
        pd.to_numeric(
            frame["close"],
            errors="coerce",
        ).to_numpy(float),
        index=pd.DatetimeIndex(frame["date"]),
    ).sort_index()


def attach_prior_risk_features(candidates):
    """Use prices strictly before historical hedge decisions."""

    if not DB.exists():
        raise FileNotFoundError(DB)

    events = candidates[
        ["Underlying RIC", "Decision Date"]
    ].drop_duplicates().copy()

    events["Decision Date"] = pd.to_datetime(
        events["Decision Date"]
    ).dt.normalize()

    found = []

    with sqlite3.connect(
        f"file:{DB.resolve().as_posix()}?mode=ro",
        uri=True,
    ) as conn:

        spy = price_history(conn, "SPY.P")

        if (
            len(spy) < 253
            or not spy.index.is_monotonic_increasing
        ):
            raise ValueError(
                "Insufficient SPY.P adjusted history"
            )

        for ric, part in events.groupby(
            "Underlying RIC",
            sort=True,
        ):

            stock = price_history(
                conn, str(ric)
            ).reindex(spy.index)

            for decision in part[
                "Decision Date"
            ].sort_values():

                # Explicitly exclude the decision-date close.
                end = int(
                    spy.index.searchsorted(
                        decision, side="left"
                    )
                )

                if end < 253:
                    continue

                p = stock.iloc[
                    end - 253:end
                ].to_numpy(float)

                m = spy.iloc[
                    end - 253:end
                ].to_numpy(float)

                if not (
                    np.isfinite(p).all()
                    and np.isfinite(m).all()
                    and (p > 0).all()
                    and (m > 0).all()
                ):
                    continue

                features = {}

                for h in (21, 63, 126, 252):
                    features.update(
                        calculate_window(
                            p[-h - 1:],
                            m[-h - 1:],
                            h,
                        )
                    )

                if not all(
                    np.isfinite(features[k])
                    for k in RISK_FEATURES
                ):
                    continue

                found.append({
                    "Underlying RIC": str(ric),
                    "Decision Date": decision,
                    **{
                        name: float(features[name])
                        for name in RISK_FEATURES
                    },
                })

    if not found:
        raise ValueError(
            "No point-in-time market features "
            "for hedge decisions"
        )

    features = pd.DataFrame(found)

    if features.duplicated(
        ["Underlying RIC", "Decision Date"]
    ).any():
        raise ValueError(
            "Duplicate risk feature decision keys"
        )

    joined = candidates.merge(
        features,
        on=["Underlying RIC", "Decision Date"],
        how="inner",
        validate="many_to_one",
    )

    before = (
        candidates[base.EVENT]
        .drop_duplicates()
        .shape[0]
    )

    after = (
        joined[base.EVENT]
        .drop_duplicates()
        .shape[0]
    )

    print(
        f"Risk history: retained {after}/{before} "
        "complete hedge opportunities"
    )

    if after < 12:
        raise ValueError(
            "Too few hedge decisions have "
            "prior-session market features"
        )

    return joined


def estimators(enhanced):
    """Reuse the original nine model specifications."""

    models = base.nine_estimators()

    if enhanced:
        for model in models.values():

            prep = model.estimator.named_steps[
                "features"
            ]

            prep.transformers = [
                (
                    name,
                    transformer,
                    (
                        list(columns)
                        + list(RISK_FEATURES)
                        if name == "numeric"
                        else columns
                    ),
                )
                for name, transformer, columns
                in prep.transformers
            ]

    return models


def predict(model, frame, enhanced):
    """Perform the predict operation for this research workflow.

    Args:
        model: Input to the predict operation.
        frame: Input to the predict operation.
        enhanced: Input to the predict operation.

    Returns:
        Computed result or status for the caller.
    """
    fields = (
        EXTENDED_FEATURES
        if enhanced
        else base.FEATURES
    )

    output = np.asarray(
        model.predict(frame[fields]),
        dtype=float,
    )

    if (
        output.shape != (len(frame), 2)
        or not np.isfinite(output).all()
    ):
        raise ValueError(
            "Invalid two-target hedge forecasts"
        )

    return output


def evaluate(train, evaluation, phase):
    """Compare both approaches on identical opportunities."""

    results = []

    for enhanced in (False, True):

        version = (
            "risk_aware"
            if enhanced
            else "option_only"
        )

        fields = (
            EXTENDED_FEATURES
            if enhanced
            else base.FEATURES
        )

        for name, model in estimators(
            enhanced
        ).items():

            print(
                f"{phase}: {version} / {name}",
                flush=True,
            )

            model.fit(
                train[fields],
                train[base.LABELS],
            )

            predictions = predict(
                model,
                evaluation,
                enhanced,
            )

            for appetite, weight in base.RISKS.items():

                chosen = base.decisions(
                    evaluation,
                    predictions,
                    weight,
                )

                results.append({
                    "Phase": phase,
                    "Version": version,
                    "Model": name,
                    "Risk appetite": appetite,
                    "Opportunities": len(chosen),
                    "Stocks": int(
                        chosen["Underlying RIC"].nunique()
                    ),
                    "Mean utility": float(
                        chosen["Realized Utility"].mean()
                    ),
                    "Mean return improvement": float(
                        chosen[
                            "Realized Return Improvement"
                        ].mean()
                    ),
                    "Mean drawdown improvement": float(
                        chosen[
                            "Realized Drawdown Improvement"
                        ].mean()
                    ),
                    "Hedge frequency": float(
                        chosen["Strategy"]
                        .ne("No Hedge")
                        .mean()
                    ),
                })

    return pd.DataFrame(results)


def train():
    """Perform the train operation for this research workflow.
    """
    df = base.prepare(
        pd.read_parquet(base.SOURCE)
    )

    df = attach_prior_risk_features(df)

    (
        train_val,
        validation,
        pretest,
        test,
        validation_start,
        test_start,
    ) = base.split_history(df)

    print(
        f"\nValidation starts: {validation_start}"
    )
    print(f"Test starts: {test_start}")

    print(
        "Validation opportunities:",
        validation[base.EVENT]
        .drop_duplicates()
        .shape[0],
    )

    validation_scores = evaluate(
        train_val,
        validation,
        "VALIDATION",
    )

    policy = {}

    for appetite in base.RISKS:

        candidates = (
            validation_scores.loc[
                validation_scores[
                    "Risk appetite"
                ].eq(appetite)
            ]
            .sort_values(
                [
                    "Mean utility",
                    "Model",
                    "Version",
                ],
                ascending=[False, True, True],
            )
        )

        traditional = candidates.loc[
            candidates["Version"].eq(
                "option_only"
            )
        ].iloc[0]

        enriched = candidates.loc[
            candidates["Version"].eq(
                "risk_aware"
            )
        ].iloc[0]

        # Require improvement over both No Hedge
        # and the best traditional option-only model.
        enough = (
            int(enriched["Opportunities"]) >= 12
            and int(enriched["Stocks"]) >= 2
        )

        promote = (
            enough
            and float(enriched["Mean utility"])
            > max(
                0.0,
                float(traditional["Mean utility"]),
            ) + 0.0025
        )

        selected = (
            enriched
            if promote
            else traditional
        )

        policy[appetite] = {
            "version": str(selected["Version"]),
            "model": str(selected["Model"]),
            "validation_utility": float(
                selected["Mean utility"]
            ),
            "validation_opportunities": int(
                selected["Opportunities"]
            ),
            "validation_stocks": int(
                selected["Stocks"]
            ),
            "option_only_best_utility": float(
                traditional["Mean utility"]
            ),
            "risk_aware_best_utility": float(
                enriched["Mean utility"]
            ),
            "risk_aware_promoted": bool(promote),
        }

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    validation_scores.to_csv(
        OUT / "same_sample_validation.csv",
        index=False,
    )

    # Evaluate selected models on later outcomes.
    # Test results are NOT used to select the policy.
    reports = []

    for appetite, info in policy.items():

        enhanced = (
            info["version"] == "risk_aware"
        )

        model = estimators(enhanced)[
            info["model"]
        ]

        fields = (
            EXTENDED_FEATURES
            if enhanced
            else base.FEATURES
        )

        model.fit(
            pretest[fields],
            pretest[base.LABELS],
        )

        chosen = base.decisions(
            test,
            predict(model, test, enhanced),
            base.RISKS[appetite],
        )

        reports.append({
            "Risk appetite": appetite,
            "Version": info["version"],
            "Model": info["model"],
            "Opportunities": len(chosen),
            "Stocks": int(
                chosen["Underlying RIC"].nunique()
            ),
            "Mean utility": float(
                chosen["Realized Utility"].mean()
            ),
            "Mean return improvement": float(
                chosen[
                    "Realized Return Improvement"
                ].mean()
            ),
            "Mean drawdown improvement": float(
                chosen[
                    "Realized Drawdown Improvement"
                ].mean()
            ),
        })

    pd.DataFrame(reports).to_csv(
        OUT / "exploratory_test.csv",
        index=False,
    )

    # Final model exports, trained on the same
    # research opportunities for both approaches.
    for enhanced in (False, True):

        subdir = OUT / (
            "risk_aware"
            if enhanced
            else "option_only"
        )

        subdir.mkdir(
            parents=True,
            exist_ok=True,
        )

        fields = (
            EXTENDED_FEATURES
            if enhanced
            else base.FEATURES
        )

        for name, model in estimators(
            enhanced
        ).items():

            print(
                f"Final fit: {subdir.name} / {name}",
                flush=True,
            )

            model.fit(
                df[fields],
                df[base.LABELS],
            )

            joblib.dump(
                model,
                subdir / (
                    name.replace(" ", "_")
                    + ".joblib"
                ),
            )

    manifest = {
        "status": "RESEARCH_ONLY",
        "policy": policy,
        "risk_features": list(RISK_FEATURES),
        "anchor_horizons": list(base.ANCHORS),
        "candidate_features": list(base.FEATURES),
        "first_decision": str(
            df["Decision Date"].min().date()
        ),
        "last_decision": str(
            df["Decision Date"].max().date()
        ),
        "training_stocks": int(
            df["Underlying RIC"].nunique()
        ),
        "training_opportunities": int(
            df[base.EVENT].drop_duplicates().shape[0]
        ),
        "validation_start": validation_start,
        "test_start": test_start,
        "warning": (
            "Historical option roots are not "
            "independently verified. "
            "Hedge outcomes cover a limited stock universe."
        ),
    }

    (
        OUT / "manifest.json"
    ).write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n=== CHOSEN POLICIES ===")
    print(json.dumps(policy, indent=2))
    print("\nSaved:", OUT)


@lru_cache(maxsize=18)
def load_model(version, name):
    """Load only models exported by this project's trainer."""

    return joblib.load(
        OUT
        / version
        / (
            name.replace(" ", "_")
            + ".joblib"
        )
    )



def render_recommendation(snapshot, appetite):
    """Display policy, raw ML hedge, and optional model comparisons."""
    import streamlit as st
    from data_pipeline.live_research_candidate_ml import (
        make_live_candidates,
        expiration_payoff_summary,
    )

    manifest_path = OUT / "manifest.json"

    if not manifest_path.is_file():
        st.info(
            "Risk-aware models unavailable; using the existing hedge policy."
        )
        base.render_recommendation(snapshot, appetite)
        return

    try:
        manifest = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )
        policy = manifest["policy"][appetite]
        version = policy["version"]
        model_name = policy["model"]

        shares = int(snapshot["inputs"].shares)
        covered = (shares // 100) * 100

        if covered < 100:
            raise ValueError(
                "At least 100 shares are required for standard options."
            )

        proxy = snapshot
        if covered != shares:
            proxy = dict(snapshot)
            proxy["inputs"] = replace(
                snapshot["inputs"], shares=covered
            )

        candidates = make_live_candidates(proxy)

        if candidates.empty:
            raise ValueError("No eligible live option candidates.")

        candidates = candidates.copy()
        candidates["Candidate ID"] = (
            candidates["Candidate ID"].astype(str)
        )

        if candidates["Candidate ID"].duplicated().any():
            raise ValueError("Duplicate option candidate IDs.")

        lookup = candidates.set_index(
            "Candidate ID", verify_integrity=True
        )

        requested = int(
            snapshot["meta"]["research_horizon_sessions"]
        )
        effective = int(np.clip(requested, 21, 252))
        candidates["Hedge Horizon Days"] = effective

        prior_date = None

        if version == "risk_aware":
            from broad_equity_six_horizon import live_features

            decision = pd.Timestamp(
                snapshot["meta"]["retrieved_at_new_york"]
            ).date()

            with st.spinner("Loading previous-session risk features..."):
                features, prior_date = live_features(
                    str(snapshot["inputs"].ric),
                    decision,
                )

            for name in RISK_FEATURES:
                value = float(features[name])

                if not np.isfinite(value):
                    raise ValueError(
                        f"Invalid stock-risk feature: {name}"
                    )

                candidates[name] = value

        spot = float(snapshot["inputs"].spot)
        fee = float(
            snapshot["meta"].get("fee_per_contract", 0.0)
        )
        fraction = covered / shares
        risk_weight = base.RISKS[appetite]

        rows = []

        for name in base.MODEL_NAMES:
            prediction = predict(
                load_model(version, name),
                candidates,
                version == "risk_aware",
            )

            utility = (
                prediction[:, 0]
                + risk_weight * prediction[:, 1]
            )

            best_index = int(np.argmax(utility))
            hedge = bool(utility[best_index] > 0)

            candidate = (
                candidates.iloc[best_index]
                if hedge else None
            )

            details, _ = expiration_payoff_summary(
                candidate,
                spot=spot,
                shares=covered,
                opening_fee_per_contract=fee,
            )

            rows.append({
                "Model": name,
                "Strategy": (
                    str(candidate["Strategy"])
                    if hedge else "No Hedge"
                ),
                "Candidate ID": (
                    str(candidate["Candidate ID"])
                    if hedge else "NO_HEDGE"
                ),
                "Put strike": details["Long Put Strike"],
                "Call strike": details["Short Call Strike"],
                "Short put strike": details["Short Put Strike"],
                "Option expiry": details["Expiry"],
                "Net opening cashflow": details[
                    "Net Opening Cashflow"
                ],
                "Return effect (pp)": (
                    float(
                        100 * prediction[best_index, 0]
                        * fraction
                    )
                    if hedge else 0.0
                ),
                "Drawdown effect (pp)": (
                    float(
                        100 * prediction[best_index, 1]
                        * fraction
                    )
                    if hedge else 0.0
                ),
                "Utility (pp)": (
                    float(
                        100 * utility[best_index]
                        * fraction
                    )
                    if hedge else 0.0
                ),
            })

        comparison = pd.DataFrame(rows)

        raw_choice = comparison.loc[
            comparison["Model"].eq(model_name)
        ].iloc[0]

        conservative_veto = (
            float(policy["validation_utility"]) <= 0
        )

        default_action = (
            "No Hedge"
            if conservative_veto
            else str(raw_choice["Strategy"])
        )

    except Exception as exc:
        st.warning(
            "ML recommendation unavailable: "
            f"{type(exc).__name__}: {exc}"
        )
        base.render_recommendation(snapshot, appetite)
        return

    # --------------------------------------------------
    # 1. Primary recommendation
    # --------------------------------------------------

    st.subheader("ML hedge recommendation")

    
    left, right = st.columns(2)

    with left:
        st.caption("ML-selected strategy (experimental)")
        st.markdown(f"### {raw_choice['Strategy']}")
        st.caption(f"Selected model: {model_name}")

    with right:
        st.caption("Conservative validation-based alternative")
        st.markdown(f"### {default_action}")

    if conservative_veto:
        st.caption(
            "The conservative policy remains unhedged because "
            "historical validation did not establish a positive "
            "average hedging benefit. The ML-selected hedge is "
            "shown for comparison, not as a guaranteed outcome."
        )

    st.caption(
        f"{appetite} risk | H{requested} NYSE sessions | "
        f"{version.replace('_', ' ')} model | "
        f"Validation: {policy['validation_opportunities']} decisions "
        f"from {policy['validation_stocks']} stocks | "
        f"Mean utility: "
        f"{100 * policy['validation_utility']:+.2f} pp"
    )

    if prior_date is not None:
        st.caption(
            "Stock-risk features use prices strictly before "
            f"the decision date, through {prior_date}."
        )

    if conservative_veto and raw_choice["Strategy"] != "No Hedge":
        st.info(
            "The ML model currently favors a hedge, but the "
            "conservative policy chooses No Hedge because "
            "historical validation utility was not positive. "
            "The model's preferred hedge is shown below."
        )

    if (
        int(policy["validation_opportunities"]) < 30
        or int(policy["validation_stocks"]) < 5
    ):
        st.warning(
            "Historical hedge validation is limited. "
            "Both choices are experimental."
        )

    if requested not in base.ANCHORS:
        if requested < 21 or requested > 252:
            st.warning(
                f"H{requested} is outside the trained range. "
                f"The model uses H{effective} as a proxy."
            )
        else:
            st.info(
                f"H{requested} is interpolated from six "
                "predefined training horizons."
            )

    if covered != shares:
        st.warning(
            f"Options cover {covered} of {shares} shares. "
            "The remaining shares are unhedged."
        )

    # Show raw ML forecasts even if the conservative
    # safeguard overrides the trade to No Hedge.

    a, b, c = st.columns(3)

    a.metric(
        "Model: return effect",
        f"{raw_choice['Return effect (pp)']:+.2f} pp",
    )
    b.metric(
        "Model: drawdown effect",
        f"{raw_choice['Drawdown effect (pp)']:+.2f} pp",
    )
    c.metric(
        "Model: combined utility",
        f"{raw_choice['Utility (pp)']:+.2f} pp",
    )

    st.caption(
        "These are the model's original estimates relative "
        "to No Hedge, not guaranteed investment returns. "
        "The No Hedge baseline has zero incremental effect."
    )

    # Compare all available model predictions visually. The selected policy
    # and validation safeguard are unchanged by this graphical display.
    from dashboard_visuals import model_ranking_chart

    st.plotly_chart(
        model_ranking_chart(
            comparison, metric="Utility (pp)",
            title="Nine-model predicted hedging utility (pp)",
        ),
        use_container_width=True,
        key="live_nine_model_utility",
    )

    # --------------------------------------------------
    # 2. The actual hedge the ML prefers
    # --------------------------------------------------

    raw_id = str(raw_choice["Candidate ID"])

    if raw_id != "NO_HEDGE":
        candidate = lookup.loc[raw_id]

        details, scenarios = expiration_payoff_summary(
            candidate,
            spot=spot,
            shares=covered,
            opening_fee_per_contract=fee,
        )

        st.markdown("**Model-selected hedge: contract details**")

        st.write(
            f"Put strike: **{details['Long Put Strike']}** | "
            f"Call strike: **{details['Short Call Strike'] or '—'}** | "
            f"Short put strike: **{details['Short Put Strike'] or '—'}** | "
            f"Expiry: **{details['Expiry']}**"
        )

        left, right = st.columns(2)

        left.metric(
            "Net opening cashflow",
            f"{details['Net Opening Cashflow']:+,.2f}",
        )

        right.metric(
            "Worst mechanical P&L at option expiry",
            f"{details['Minimum Expiry P&L']:+,.2f}",
        )

        st.caption(
            f"{details['Downside Description']}. "
            f"{details['Upside Description']}. "
            "Amounts are in the stock's quote currency. "
            "Positive opening cashflow means a credit; "
            "negative means a debit."
        )

        target = str(
            pd.Timestamp(
                snapshot["meta"]["research_target_date"]
            ).date()
        )

        if details["Expiry"] != target:
            st.warning(
                f"Requested hedge end: {target}. "
                f"Option expiry: {details['Expiry']}. "
                "The mechanical P&L and scenarios apply "
                "at option expiry, not at your requested "
                "hedge end date. They are not a guaranteed "
                "target-date protection floor."
            )

        if covered != shares:
            extra_pnl = (
                (shares - covered)
                * (
                    scenarios["Underlying price at expiry"]
                    - spot
                )
            )

            scenarios["Unhedged P&L"] += extra_pnl
            scenarios["Selected P&L"] += extra_pnl

            scenarios["Hedge benefit vs unhedged"] = (
                scenarios["Selected P&L"]
                - scenarios["Unhedged P&L"]
            )

            st.caption(
                "The minimum-expiry-P&L metric above "
                "covers only the option-covered shares."
            )

        with st.expander("View option-expiry payoff scenarios"):
            st.dataframe(
                scenarios.round(2),
                hide_index=True,
                use_container_width=True,
            )

            st.caption(
                "Hypothetical price shocks, not probabilities. "
                "Future closing costs are excluded."
            )

    else:
        st.caption(
            "The selected ML model itself favors No Hedge."
        )

    # --------------------------------------------------
    # 3. Optional comparison of all nine models
    # --------------------------------------------------

    with st.expander("Compare all nine ML hedge models"):
        st.dataframe(
            comparison.drop(
                columns="Candidate ID"
            ).round(2),
            hide_index=True,
            use_container_width=True,
        )

        st.caption(
            "These are each model's original choices. "
            "The conservative historical-validation safeguard "
            "is applied to the primary policy separately. "
            "Different expiries cannot be compared directly "
            "as valuations at the same hedge end date."
        )

    # --------------------------------------------------
    # 4. Optional historical performance
    # --------------------------------------------------

    with st.expander("Historical validation and held-out results"):
        validation_path = OUT / "same_sample_validation.csv"

        if validation_path.is_file():
            validation = pd.read_csv(validation_path)

            validation = validation.loc[
                validation["Risk appetite"].eq(appetite)
            ].copy()

            for column in (
                "Mean utility",
                "Mean return improvement",
                "Mean drawdown improvement",
            ):
                validation[column] = (
                    validation[column] * 100
                ).round(2)

            validation = validation.rename(columns={
                "Mean utility": "Mean utility (pp)",
                "Mean return improvement": "Return improvement (pp)",
                "Mean drawdown improvement": "Drawdown improvement (pp)",
            })

            st.write("**Chronological validation**")

            st.dataframe(
                validation.sort_values(
                    "Mean utility (pp)",
                    ascending=False,
                ),
                hide_index=True,
                use_container_width=True,
            )

        test_path = OUT / "exploratory_test.csv"

        if test_path.is_file():
            test = pd.read_csv(test_path)

            st.write("**Later held-out evaluation**")

            st.dataframe(
                test.loc[
                    test["Risk appetite"].eq(appetite)
                ],
                hide_index=True,
                use_container_width=True,
            )

            st.caption(
                "Held-out results were not used "
                "to choose the model."
            )

    st.caption(
        f"Research prototype: {manifest['training_stocks']} "
        f"historical hedge-training stocks, "
        f"{manifest['training_opportunities']} "
        "independent hedge opportunities, six predefined "
        "horizons. Historical option roots and corporate "
        "actions are not fully verified. No order execution."
    )



def main():
    """Run the module entry point and its workflow.
    """
    parser = argparse.ArgumentParser(
        description=__doc__,
    )

    parser.add_argument(
        "--train",
        action="store_true",
    )

    args = parser.parse_args()

    if not args.train:
        parser.error("Use --train")

    train()


if __name__ == "__main__":
    main()
