
"""Experimental H21 live candidate ranking and payoff explanation.

Uses existing indicative LSEG option quotes and saved models.
No LSEG requests, retraining, or order execution in this module.

Mechanical payoffs are at option expiration, NOT necessarily
at the requested 21-session research target.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from pipeline.historical_candidate_generation import (
    CandidateConfig,
    generate_hedge_candidates,
)
from train_nine_candidate_models import FEATURES
from data_pipeline.quote_time_evidence import (
    selected_option_quote_evidence,
)
from data_pipeline.stock_price_timing import (
    stock_time_evidence,
)


ROOT = Path(__file__).resolve().parents[1]

MODEL_ROOT = (
    ROOT
    / "data/processed/ml_experiments/"
      "SP500_h21_research_model_exports"
)

MODEL_NAMES = (
    "OLS",
    "Ridge",
    "Lasso",
    "Elastic Net",
    "PCR",
    "PLS",
    "Random Forest",
    "Gradient Boosting",
    "XGBoost",
)


def make_live_candidates(
    snapshot: dict,
) -> pd.DataFrame:
    """Create entry-time model features from existing quotes."""

    meta = snapshot["meta"]
    inputs = snapshot["inputs"]

    if meta.get("used_close_fallback"):
        raise ValueError(
            "Unsynchronized closing stock prices "
            "cannot be used for ML scoring."
        )

    if not snapshot.get("has_direct_options"):
        raise ValueError(
            "No eligible direct-stock options available."
        )

    if (
        not meta.get("assumed_multiplier_accepted")
        and not meta.get("multiplier_verified_from_fields")
    ):
        raise ValueError(
            "Option multiplier has not been accepted."
        )

    shares = int(inputs.shares)
    spot = float(inputs.spot)

    if shares < 100 or shares % 100:
        raise ValueError(
            "ML scoring requires whole 100-share "
            "contracts without uncovered shares."
        )

    if not np.isfinite(spot) or spot <= 0:
        raise ValueError("Invalid stock price.")

    raw = snapshot["liquid_options"].copy()

    required = {
        "RIC",
        "CallPutOption",
        "StrikePrice",
        "ExpiryDate",
        "BID",
        "ASK",
    }

    missing = required - set(raw.columns)

    if missing:
        raise ValueError(
            f"Missing option fields: {sorted(missing)}"
        )

    if raw.empty:
        raise ValueError("Empty option quote set.")

    if "QuoteSource" in raw.columns:
        historical = (
            raw["QuoteSource"]
            .astype("string")
            .str.contains(
                "historical",
                case=False,
                na=False,
            )
        )
        raw = raw.loc[~historical].copy()

    if raw.empty:
        raise ValueError(
            "Only historical fallback quotes available."
        )

    for column in (
        "ContractSize",
        "ContractMultiplier",
    ):
        if column in raw.columns:
            multiplier = pd.to_numeric(
                raw[column],
                errors="coerce",
            )
            compatible = (
                multiplier.isna()
                | np.isclose(multiplier, 100)
            )
            raw = raw.loc[compatible].copy()

    raw["Option Type"] = (
        raw["CallPutOption"]
        .astype("string")
        .str.strip()
        .str.lower()
        .replace({
            "p": "put",
            "c": "call",
        })
    )

    raw["Expiry"] = (
        pd.to_datetime(
            raw["ExpiryDate"],
            errors="coerce",
            utc=True,
        )
        .dt.tz_convert(None)
        .dt.normalize()
    )

    raw["Strike"] = pd.to_numeric(
        raw["StrikePrice"],
        errors="coerce",
    )

    for column in ("BID", "ASK"):
        raw[column] = pd.to_numeric(
            raw[column],
            errors="coerce",
        )

    options = raw[
        [
            "RIC",
            "Option Type",
            "Strike",
            "Expiry",
            "BID",
            "ASK",
        ]
    ].copy()

    if options.duplicated("RIC").any():
        raise ValueError(
            "Duplicate option RICs require inspection."
        )

    candidates = generate_hedge_candidates(
        options=options,
        underlying_ric=str(inputs.ric),
        spot_price=spot,
        hedge_end_date=meta["research_target_date"],
        shares=shares,
        option_multiplier=100,
        config=CandidateConfig(
            max_spread_pct=float(
                meta["max_spread_fraction"]
            ),
            max_expiry_gap_days=45,
        ),
    )

    if candidates.empty:
        raise ValueError(
            "No eligible option combinations match "
            "the historical candidate schema."
        )

    decision_date = pd.Timestamp(
        meta["retrieved_at_new_york"]
    ).date()

    candidates["Days to Expiry"] = (
        candidates["Expiry"]
        - pd.Timestamp(decision_date)
    ).dt.days

    if candidates["Days to Expiry"].le(0).any():
        raise ValueError(
            "Option expiry must follow the quote date."
        )

    for source, feature in (
        ("Long Put Ask", "Long Put Ask Pct"),
        ("Short Call Bid", "Short Call Bid Pct"),
        ("Short Put Bid", "Short Put Bid Pct"),
    ):
        value = pd.to_numeric(
            candidates[source],
            errors="coerce",
        )
        candidates[feature] = (
            value.fillna(0.0) / spot
        )

    for column in (
        "Short Call Strike Pct",
        "Short Put Strike Pct",
    ):
        candidates[column] = pd.to_numeric(
            candidates[column],
            errors="coerce",
        ).fillna(0.0)

    numeric_features = [
        feature
        for feature in FEATURES
        if feature != "Strategy"
    ]

    if not np.isfinite(
        candidates[numeric_features]
        .to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "Invalid live model feature values."
        )

    return candidates


def load_local_models(
    objective: str,
    model_root: Path = MODEL_ROOT,
) -> tuple[dict, dict]:
    """Load trusted local research model exports."""

    if objective not in {"risk", "return"}:
        raise ValueError(
            "Objective must be risk or return."
        )

    directory = (
        Path(model_root) / objective
    ).resolve()

    manifest = json.loads(
        (directory / "manifest.json").read_text(
            encoding="utf-8"
        )
    )

    expected_weight = (
        1.0 if objective == "risk" else 0.0
    )

    if int(manifest["horizon_sessions"]) != 21:
        raise ValueError(
            "Model export does not support H21."
        )

    if not np.isclose(
        float(manifest["drawdown_weight"]),
        expected_weight,
    ):
        raise ValueError(
            "Model objective differs from manifest."
        )

    if list(
        manifest["features_in_order"]
    ) != list(FEATURES):
        raise ValueError(
            "Model feature schema differs from application."
        )

    if set(manifest["files"]) != set(MODEL_NAMES):
        raise ValueError(
            "Exactly nine named models are required."
        )

    models = {}

    for name in MODEL_NAMES:
        filename = (
            directory
            / manifest["files"][name]
        ).resolve()

        if filename.parent != directory:
            raise ValueError(
                "Unsafe model path in manifest."
            )

        # Only load locally trusted joblib files.
        models[name] = joblib.load(filename)

    return models, manifest


def rank_live_candidates(
    candidates: pd.DataFrame,
    models: dict,
) -> pd.DataFrame:
    """Choose the best predicted hedge or No Hedge."""

    if (
        candidates.empty
        or not set(FEATURES).issubset(
            candidates.columns
        )
    ):
        raise ValueError(
            "No valid candidate features to score."
        )

    decisions = []

    for name in MODEL_NAMES:
        if name not in models:
            raise ValueError(
                f"Missing model: {name}"
            )

        predictions = np.asarray(
            models[name].predict(
                candidates[FEATURES]
            ),
            dtype=float,
        ).reshape(-1)

        if (
            len(predictions) != len(candidates)
            or not np.isfinite(predictions).all()
        ):
            raise ValueError(
                f"Invalid predictions from {name}"
            )

        ranked = (
            candidates.assign(
                _score=predictions
            )
            .sort_values(
                ["_score", "Candidate ID"],
                ascending=[False, True],
                kind="stable",
            )
        )

        best = ranked.iloc[0]
        score = float(best["_score"])
        no_hedge = score <= 0.0

        decisions.append({
            "Model": name,
            "Decision": (
                "No Hedge"
                if no_hedge
                else str(best["Strategy"])
            ),
            "Predicted improvement (pp)": max(
                0.0,
                100.0 * score,
            ),
            "Best hedge forecast (pp)": (
                100.0 * score
            ),
            "Candidate ID": (
                "NO_HEDGE"
                if no_hedge
                else str(best["Candidate ID"])
            ),
        })

    return pd.DataFrame(decisions)


def expiration_payoff_summary(
    candidate: pd.Series | None,
    *,
    spot: float,
    shares: int,
    opening_fee_per_contract: float,
    multiplier: int = 100,
) -> tuple[dict, pd.DataFrame]:
    """Calculate mechanical stock-plus-option expiration P&L.

    Purchased put: entry ASK.
    Written call/put: entry BID.
    Opening fees included; future closing costs excluded.

    All option legs cover the full stock position.
    """

    if (
        shares < multiplier
        or shares % multiplier != 0
        or multiplier <= 0
        or not np.isfinite(spot)
        or spot <= 0
        or not np.isfinite(opening_fee_per_contract)
        or opening_fee_per_contract < 0
    ):
        raise ValueError(
            "Invalid position, multiplier or fee."
        )

    strategy = (
        "No Hedge"
        if candidate is None
        else str(candidate["Strategy"])
    )

    allowed = {
        "No Hedge",
        "Protective Put",
        "Collar",
        "Put Spread",
    }

    if strategy not in allowed:
        raise ValueError(
            f"Unsupported strategy: {strategy}"
        )

    if candidate is None:
        expiry = None
        k_put = 0.0
        k_call = 0.0
        k_short_put = 0.0
        put_ask = 0.0
        call_bid = 0.0
        short_put_bid = 0.0
        legs = 0

    else:
        expiry = (
            pd.Timestamp(
                candidate["Expiry"]
            ).date().isoformat()
        )

        k_put = float(
            candidate["Long Put Strike"]
        )

        put_ask = float(
            candidate["Long Put Ask"]
        )

        k_call = (
            float(candidate["Short Call Strike"])
            if strategy == "Collar"
            else 0.0
        )

        call_bid = (
            float(candidate["Short Call Bid"])
            if strategy == "Collar"
            else 0.0
        )

        k_short_put = (
            float(candidate["Short Put Strike"])
            if strategy == "Put Spread"
            else 0.0
        )

        short_put_bid = (
            float(candidate["Short Put Bid"])
            if strategy == "Put Spread"
            else 0.0
        )

        legs = {
            "Protective Put": 1,
            "Collar": 2,
            "Put Spread": 2,
        }[strategy]

        prices = (
            k_put,
            put_ask,
            k_call,
            call_bid,
            k_short_put,
            short_put_bid,
        )

        if not all(
            np.isfinite(value)
            for value in prices
        ):
            raise ValueError(
                "Invalid strike or option premium."
            )

        if k_put <= 0 or put_ask <= 0:
            raise ValueError(
                "Invalid long-put strike or ASK."
            )

        if (
            strategy == "Collar"
            and not (
                k_call > k_put
                and call_bid > 0
            )
        ):
            raise ValueError(
                "Invalid Collar option legs."
            )

        if (
            strategy == "Put Spread"
            and not (
                0 < k_short_put < k_put
                and short_put_bid > 0
            )
        ):
            raise ValueError(
                "Invalid Put Spread option legs."
            )

    # Positive debit = premium paid.
    debit_per_share = (
        put_ask
        - call_bid
        - short_put_bid
    )

    # Positive cashflow = net premium received.
    option_premium_cashflow = (
        -shares * debit_per_share
    )

    contracts = shares // multiplier

    opening_fees = (
        contracts
        * legs
        * opening_fee_per_contract
    )

    net_opening_cashflow = (
        option_premium_cashflow
        - opening_fees
    )

    def pnl_at(
        terminal: float,
    ) -> tuple[float, float]:
        """Stock P&L and stock-plus-option P&L."""

        unhedged = (
            shares * (terminal - spot)
        )

        if candidate is None:
            option_payoff = 0.0

        else:
            long_put_payoff = max(
                k_put - terminal,
                0.0,
            )

            short_call_liability = (
                max(
                    terminal - k_call,
                    0.0,
                )
                if strategy == "Collar"
                else 0.0
            )

            short_put_liability = (
                max(
                    k_short_put - terminal,
                    0.0,
                )
                if strategy == "Put Spread"
                else 0.0
            )

            option_payoff = shares * (
                long_put_payoff
                - short_call_liability
                - short_put_liability
            )

        selected = (
            unhedged
            + option_payoff
            + net_opening_cashflow
        )

        return unhedged, selected

    shocks = (
        -1.00,
        -0.30,
        -0.20,
        -0.10,
        -0.05,
        0.00,
        0.10,
        0.20,
        0.30,
    )

    rows = []

    for shock in shocks:
        terminal = max(
            0.0,
            spot * (1.0 + shock),
        )

        unhedged, selected = pnl_at(
            terminal
        )

        rows.append({
            "Underlying move": f"{shock:+.0%}",
            "Underlying price at expiry": terminal,
            "Unhedged P&L": unhedged,
            "Selected P&L": selected,
            "Hedge benefit vs unhedged": (
                selected - unhedged
            ),
        })

    minimum_pnl = pnl_at(0.0)[1]

    maximum_pnl = (
        pnl_at(k_call)[1]
        if strategy == "Collar"
        else None
    )

    if strategy in (
        "Protective Put",
        "Collar",
    ):
        downside_description = (
            f"Put protection floor at {k_put:,.2f}"
        )

    elif strategy == "Put Spread":
        downside_description = (
            "Protection is limited between "
            f"{k_short_put:,.2f} and {k_put:,.2f}; "
            f"losses resume below {k_short_put:,.2f}"
        )

    else:
        downside_description = (
            "No put protection"
        )

    upside_description = (
        f"Stock upside capped above {k_call:,.2f}"
        if strategy == "Collar"
        else "Stock upside is not capped"
    )

    summary = {
        "Strategy": strategy,
        "Expiry": expiry,
        "Long Put Strike": (
            k_put if candidate is not None else None
        ),
        "Short Call Strike": (
            k_call if strategy == "Collar" else None
        ),
        "Short Put Strike": (
            k_short_put
            if strategy == "Put Spread"
            else None
        ),
        "Option Premium Cashflow": (
            option_premium_cashflow
        ),
        "Opening Fees": opening_fees,
        "Net Opening Cashflow": (
            net_opening_cashflow
        ),
        "Minimum Expiry P&L": minimum_pnl,
        "Maximum Expiry P&L": maximum_pnl,
        "Downside Description": (
            downside_description
        ),
        "Upside Description": (
            upside_description
        ),
        "Long Put RIC": (
            None
            if candidate is None
            else str(candidate["Long Put RIC"])
        ),
        "Short Call RIC": (
            str(candidate["Short Call RIC"])
            if strategy == "Collar"
            else None
        ),
        "Short Put RIC": (
            str(candidate["Short Put RIC"])
            if strategy == "Put Spread"
            else None
        ),
    }

    return summary, pd.DataFrame(rows)


def compare_recommendations(
    ranking: pd.DataFrame,
    candidates: pd.DataFrame,
    *,
    spot: float,
    shares: int,
    opening_fee_per_contract: float,
) -> pd.DataFrame:
    """Attach mechanical opening cashflows to ML selections."""

    if (
        candidates["Candidate ID"]
        .astype(str)
        .duplicated()
        .any()
    ):
        raise ValueError(
            "Duplicate Candidate IDs."
        )

    lookup = candidates.set_index(
        candidates["Candidate ID"].astype(str)
    )

    rows = []

    for _, choice in ranking.iterrows():
        cid = str(
            choice["Candidate ID"]
        )

        candidate = (
            None
            if cid == "NO_HEDGE"
            else lookup.loc[cid]
        )

        details, _ = expiration_payoff_summary(
            candidate,
            spot=spot,
            shares=shares,
            opening_fee_per_contract=(
                opening_fee_per_contract
            ),
        )

        rows.append({
            "Model": choice["Model"],
            "Decision": choice["Decision"],
            "Predicted improvement (pp)": (
                choice["Predicted improvement (pp)"]
            ),
            "Put strike": details[
                "Long Put Strike"
            ],
            "Call strike": details[
                "Short Call Strike"
            ],
            "Short put": details[
                "Short Put Strike"
            ],
            "Opening premium cashflow": details[
                "Option Premium Cashflow"
            ],
            "Opening fees": details[
                "Opening Fees"
            ],
            "Net opening cashflow": details[
                "Net Opening Cashflow"
            ],
            "Expiry": details["Expiry"],
            "Candidate ID": cid,
        })

    return pd.DataFrame(rows)


def render_live_candidate_models(
    snapshot: dict,
) -> None:
    """Render optional ML decisions and expiration scenarios."""

    import streamlit as st

    with st.expander(
        "Experimental ML option ranking (21 sessions)",
        expanded=False,
    ):
        st.caption(
            "Nine experimental models score indicative "
            "option combinations. No orders are placed, "
            "and the main hedge comparison is unchanged."
        )

        meta = snapshot["meta"]

        if int(
            meta["research_horizon_sessions"]
        ) != 21:
            st.info(
                "Saved ML candidate models support "
                "21 NYSE sessions only."
            )
            return

        if not st.checkbox(
            "Show experimental scores from saved models",
            value=False,
            key="show_live_research_candidate_ml",
        ):
            return

        goal = st.radio(
            "Scoring objective",
            (
                "Reduce drawdown (return + drawdown)",
                "Preserve returns (return only)",
            ),
            horizontal=True,
            key="live_research_candidate_ml_objective",
        )

        objective = (
            "risk"
            if goal.startswith("Reduce")
            else "return"
        )

        try:
            candidates = make_live_candidates(
                snapshot
            )

            models, manifest = load_local_models(
                objective
            )

            ranking = rank_live_candidates(
                candidates,
                models,
            )

            inputs = snapshot["inputs"]

            spot = float(inputs.spot)
            shares = int(inputs.shares)

            fee = float(
                meta.get("fee_per_contract", 0.0)
            )

            comparison = compare_recommendations(
                ranking,
                candidates,
                spot=spot,
                shares=shares,
                opening_fee_per_contract=fee,
            )

        except (
            FileNotFoundError,
            KeyError,
            ValueError,
            TypeError,
            OSError,
        ) as exc:
            st.warning(
                f"ML ranking unavailable: {exc}"
            )
            return

        except Exception as exc:
            st.warning(
                "ML inference failed: "
                f"{type(exc).__name__}: {exc}"
            )
            return

        if objective == "return":
            st.caption(
                "**Forecast:** Predicted return effect "
                "relative to holding the stock without "
                "hedging, measured in percentage points."
            )
        else:
            st.caption(
                "**Forecast:** Predicted return effect "
                "plus signed maximum-drawdown benefit, "
                "equally weighted. This is not a "
                "predicted investment return."
            )

        st.caption(
            "Snapshot retrieved (New York): "
            f"{meta['retrieved_at_new_york']}. "
            "Changing the ML setting does not "
            "refresh the market quotes."
        )

        st.caption(
            f"{len(candidates)} option combinations | "
            f"{manifest['training_rows']:,} historical "
            "candidate records from "
            f"{manifest['training_opportunities']} "
            "stock-date opportunities."
        )

        visible = comparison.drop(
            columns=["Candidate ID", "Expiry"]
        ).rename(
            columns={
                "Predicted improvement (pp)": (
                    "Forecast return effect (pp)"
                    if objective == "return"
                    else "Forecast combined score (pp)"
                ),
                "Opening premium cashflow": (
                    "Option premium cashflow"
                ),
            }
        )

        st.dataframe(
            visible,
            hide_index=True,
            use_container_width=True,
        )

        votes = (
            ranking["Decision"]
            .value_counts()
        )

        summary = "; ".join(
            f"{int(count)} × {strategy}"
            for strategy, count in votes.items()
        )

        st.write(
            f"**Model choices:** {summary}."
        )

        if ranking["Candidate ID"].eq(
            "NO_HEDGE"
        ).any():
            st.caption(
                "No Hedge means that model predicts "
                "no positive improvement from any "
                "available option hedge."
            )

        # Allow the user to inspect each model's
        # selected contract without repeating nine
        # separate payoff tables.
        display_names = [
            f"{row['Model']} — {row['Decision']}"
            for _, row in ranking.iterrows()
        ]

        selected_name = st.selectbox(
            "Inspect a model's hedge and payoff",
            options=display_names,
            index=1,
            key="live_research_candidate_ml_inspect",
        )

        chosen = ranking.iloc[
            display_names.index(selected_name)
        ]

        selected_id = str(
            chosen["Candidate ID"]
        )

        if selected_id == "NO_HEDGE":
            candidate = None
        else:
            candidate = candidates.loc[
                candidates["Candidate ID"]
                .astype(str)
                .eq(selected_id)
            ].iloc[0]

        details, scenarios = expiration_payoff_summary(
            candidate,
            spot=spot,
            shares=shares,
            opening_fee_per_contract=fee,
        )

        cashflow = details[
            "Net Opening Cashflow"
        ]

        cash_type = (
            "Credit"
            if cashflow > 0
            else "Debit"
            if cashflow < 0
            else "None"
        )

        a, b, c = st.columns(3)

        a.metric(
            "Net opening cashflow",
            f"{cashflow:+,.2f}",
        )

        b.metric(
            "Cashflow type",
            cash_type,
        )

        c.metric(
            "Option expiry",
            details["Expiry"] or "—",
        )

        st.write(
            "**Downside at option expiration:** "
            f"{details['Downside Description']}."
        )

        st.write(
            "**Upside at option expiration:** "
            f"{details['Upside Description']}."
        )

        st.write(
            "**Opening option premium cashflow:** "
            f"{details['Option Premium Cashflow']:+,.2f}; "
            "**opening contract fees:** "
            f"{details['Opening Fees']:,.2f}. "
            "Positive cashflow is a credit. "
            "Quotes use the long put ASK and "
            "the written options' BID prices."
        )

        st.write(
            "**Minimum stock-plus-option P&L "
            "at expiration:** "
            f"{details['Minimum Expiry P&L']:+,.2f}. "
            + (
                "**Maximum P&L at expiration:** "
                f"{details['Maximum Expiry P&L']:+,.2f}."
                if details["Maximum Expiry P&L"] is not None
                else "Stock upside is not capped."
            )
        )

        if candidate is not None:
            legs = []

            for label, field in (
                ("Long put", "Long Put RIC"),
                ("Short call", "Short Call RIC"),
                ("Short put", "Short Put RIC"),
            ):
                if details[field] is not None:
                    legs.append(
                        f"{label}: {details[field]}"
                    )

            st.caption(
                "Selected option contracts: "
                + " | ".join(legs)
            )

        st.write(
            "**LSEG underlying stock timing evidence**"
        )

        st.dataframe(
            stock_time_evidence(meta),
            hide_index=True,
            use_container_width=True,
        )

        st.caption(
            "The stock trade and quote clocks come "
            "from the same stock-price request. "
            "They are time-of-day fields without "
            "a verified event date. Stock and "
            "option snapshots are retrieved "
            "separately, so their synchronization "
            "is not independently established."
        )

        if candidate is not None:

            selected_rics = [
                details[field]
                for field in (
                    "Long Put RIC",
                    "Short Call RIC",
                    "Short Put RIC",
                )
                if details[field] is not None
            ]

            quote_evidence = (
                selected_option_quote_evidence(
                    snapshot["liquid_options"],
                    selected_rics,
                )
            )

            st.write(
                "**LSEG option quote timing evidence**"
            )

            st.dataframe(
                quote_evidence,
                hide_index=True,
                use_container_width=True,
            )

            st.caption(
                "QUOTIM is a UTC time of day, "
                "not a verified event datetime. "
                "The observed clock lag is consistent "
                "with delayed data when near 15 minutes, "
                "but does not independently establish "
                "the event date, feed delay, or quote "
                "freshness. BID and ASK come from the "
                "same retrieved pricing snapshot."
            )

        st.write(
            "**Mechanical P&L scenarios**"
        )

        st.dataframe(
            scenarios.style.format({
                "Underlying price at expiry": "{:,.2f}",
                "Unhedged P&L": "{:+,.2f}",
                "Selected P&L": "{:+,.2f}",
                "Hedge benefit vs unhedged": "{:+,.2f}",
            }),
            hide_index=True,
            use_container_width=True,
        )

        target = str(
            meta["research_target_date"]
        )

        expiry = details["Expiry"]

        if expiry is None:
            st.info(
                "No Hedge has no option expiration. "
                "The scenarios use hypothetical "
                "stock prices for comparison."
            )

        elif expiry != target:
            st.warning(
                "The 21-session research target is "
                f"**{target}**, but these options expire "
                f"on **{expiry}**. The table shows "
                "expiration payoffs only. It is NOT "
                "a guaranteed or forecast P&L at the "
                "research target. Earlier liquidation "
                "requires future option quotes, remaining "
                "time value and possibly additional fees."
            )

        else:
            st.info(
                "These are hypothetical expiration "
                "payoffs, not stock-price forecasts."
            )

        st.warning(
            "ML calculations assume 100 shares per "
            "option contract. The main snapshot's "
            "multiplier check does not independently "
            "verify every ML-selected option leg. "
            "Confirm each contract's multiplier "
            "and deliverable before execution."
        )

        if fee > 0:
            st.caption(
                "Opening fees are included in the "
                "mechanical payoffs. Closing, exercise "
                "and assignment fees, taxes, dividends "
                "and financing are excluded. Custom "
                "fees are not inputs to the ML models."
            )
        else:
            st.caption(
                "Your selected opening fee is zero. "
                "Other trading fees, taxes, dividends "
                "and financing are excluded."
            )

        st.caption(
            "LSEG quotes may be stale or unsynchronized. "
            "Exported models were refitted after prior "
            "historical evaluation; AAPL dominates "
            "training. Historical quote-completeness, "
            "corporate-action and membership limitations "
            "remain unresolved. ML predictions are "
            "not guaranteed results."
        )
