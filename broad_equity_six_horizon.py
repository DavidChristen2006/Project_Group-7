
"""Train nine pooled stock-risk models on six historical hedge horizons.

Reuses existing point-in-time features and SQLite equity labels.
No LSEG calls during training. No historical option returns invented.

Usage:
    py broad_equity_six_horizon.py --train
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import timedelta
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from data_pipeline.train_equity_63d_baseline import FEATURES as PRIOR_FEATURES
from data_pipeline.train_equity_63d_model_suite import make_models
from data_pipeline.build_equity_features import calculate_window


ROOT = Path(__file__).resolve().parent

DB = (
    ROOT
    / "data/processed/unified_research_store_v1"
    / "master_research.sqlite"
)

FEATURE_FILE = (
    ROOT
    / "data/processed/research_features/equity_features_v1"
    / "point_in_time_equity_features.parquet"
)

OUT = (
    ROOT
    / "data/processed/ml_experiments"
    / "broad_equity_six_horizon_v1"
)

ANCHORS = (21, 42, 63, 84, 126, 252)

FEATURES = list(PRIOR_FEATURES) + ["Hedge Horizon Days"]

TARGETS = [
    "final_return",
    "maximum_drawdown",
    "worst_return",
]

WEIGHTS = {
    "Low": (0.20, 0.40, 0.40),
    "Balanced": (0.34, 0.33, 0.33),
    "High": (0.60, 0.20, 0.20),
}

DISPLAY = {
    "ols": "OLS",
    "ridge": "Ridge",
    "lasso": "Lasso",
    "elastic_net": "Elastic Net",
    "pcr": "PCR",
    "pls": "PLS",
    "random_forest": "Random Forest",
    "hist_gradient_boosting": "Gradient Boosting",
    "xgboost": "XGBoost",
}

KEY = [
    "Underlying RIC",
    "Decision Date",
    "Hedge Horizon Days",
]


def read_data():
    """Load data from stored research artifacts.

    Returns:
        Computed result or status for the caller.
    """
    if not DB.is_file():
        raise FileNotFoundError(DB)

    if not FEATURE_FILE.is_file():
        raise FileNotFoundError(FEATURE_FILE)

    needed = [
        "Underlying RIC",
        "Decision Date",
        "Last Prior Equity Price Date",
        "Verified Membership Feature Candidate",
        *PRIOR_FEATURES,
    ]

    feature = pd.read_parquet(
        FEATURE_FILE,
        columns=needed,
    )

    if feature.duplicated(KEY[:2]).any():
        raise ValueError(
            "Duplicate point-in-time feature keys"
        )

    feature["Decision Date"] = pd.to_datetime(
        feature["Decision Date"]
    ).dt.normalize()

    feature["Last Prior Equity Price Date"] = (
        pd.to_datetime(
            feature["Last Prior Equity Price Date"],
            errors="coerce",
        ).dt.normalize()
    )

    verified = (
        feature[
            "Verified Membership Feature Candidate"
        ]
        .astype("string")
        .str.lower()
        .isin(("true", "1"))
    )

    feature = feature.loc[verified].copy()

    if not (
        feature["Last Prior Equity Price Date"]
        < feature["Decision Date"]
    ).all():
        raise ValueError(
            "Future information in market features"
        )

    with sqlite3.connect(
        f"file:{DB.resolve().as_posix()}?mode=ro",
        uri=True,
    ) as cn:

        outcome = pd.read_sql_query(
            """
            SELECT
                ric AS "Underlying RIC",
                decision_date AS "Decision Date",
                outcome_date AS "Outcome Date",
                horizon AS "Hedge Horizon Days",
                final_return,
                maximum_drawdown,
                worst_return
            FROM equity_forward_labels
            WHERE eligibility='READY_EQUITY_ONLY'
            """,
            cn,
        )

        event = pd.read_sql_query(
            """
            SELECT ric, event_date
            FROM corporate_action_review
            WHERE event_date IS NOT NULL
            """,
            cn,
        )

        splits = pd.read_sql_query(
            "SELECT ric, date FROM split_review",
            cn,
        )

    if outcome.empty:
        raise ValueError(
            "No READY_EQUITY_ONLY records available"
        )

    outcome["Decision Date"] = pd.to_datetime(
        outcome["Decision Date"]
    ).dt.normalize()

    outcome["Outcome Date"] = pd.to_datetime(
        outcome["Outcome Date"]
    ).dt.normalize()

    outcome["Hedge Horizon Days"] = (
        outcome["Hedge Horizon Days"].astype(int)
    )

    outcome = outcome.loc[
        outcome["Hedge Horizon Days"].isin(ANCHORS)
    ].copy()

    if outcome.duplicated(KEY).any():
        raise ValueError(
            "Duplicate equity-label opportunities"
        )

    if not (
        outcome["Outcome Date"]
        > outcome["Decision Date"]
    ).all():
        raise ValueError(
            "Invalid forward outcome dates"
        )

    # Exclude unresolved terminal-event and split windows.
    # Do not assume historical option deliverables are unchanged.

    for events, date_column in (
        (event, "event_date"),
        (splits, "date"),
    ):
        for rec in events.itertuples(index=False):
            dt = pd.to_datetime(
                getattr(rec, date_column),
                errors="coerce",
            )

            if pd.isna(dt):
                continue

            mask = (
                outcome["Underlying RIC"].eq(rec.ric)
                & outcome["Decision Date"].le(dt)
                & outcome["Outcome Date"].ge(dt)
            )

            outcome = outcome.loc[~mask].copy()

    data = outcome.merge(
        feature.drop(
            columns=[
                "Verified Membership Feature Candidate"
            ]
        ),
        on=KEY[:2],
        how="inner",
        validate="many_to_one",
    )

    for field in FEATURES + TARGETS:
        data[field] = pd.to_numeric(
            data[field],
            errors="coerce",
        )

    good = (
        np.isfinite(
            data[TARGETS].to_numpy(float)
        ).all(axis=1)
        & data["final_return"].ge(-1)
        & data["maximum_drawdown"].between(-1, 0)
        & data["worst_return"].between(-1, 0)
    )

    data = data.loc[good].copy()

    # Missing historical features are handled by
    # existing model imputers fitted on training data.

    if (
        data.empty
        or data["Underlying RIC"].nunique() < 3
    ):
        raise ValueError(
            "Insufficient observed, validated, "
            "as-of equity labels"
        )

    missing_horizons = (
        set(ANCHORS)
        - set(data["Hedge Horizon Days"].unique())
    )

    if missing_horizons:
        raise ValueError(
            "Missing horizons: "
            f"{sorted(missing_horizons)}"
        )

    return (
        data.sort_values(KEY)
        .reset_index(drop=True)
    )


def chronological_split(data):
    """Perform the chronological split operation for this research workflow.

    Args:
        data: Input to the chronological_split operation.

    Returns:
        Computed result or status for the caller.
    """
    dates = sorted(
        data["Decision Date"].unique()
    )

    if len(dates) < 15:
        raise ValueError(
            "At least 15 historical decision dates required"
        )

    vstart = pd.Timestamp(
        dates[int(len(dates) * 0.60)]
    )

    tstart = pd.Timestamp(
        dates[int(len(dates) * 0.80)]
    )

    # Training outcomes must be known before validation.
    train = data.loc[
        data["Decision Date"].lt(vstart)
        & data["Outcome Date"].lt(vstart)
    ].copy()

    # Validation outcomes must mature before test.
    valid = data.loc[
        data["Decision Date"].ge(vstart)
        & data["Decision Date"].lt(tstart)
        & data["Outcome Date"].lt(tstart)
    ].copy()

    # Larger historical training sample for test.
    pretest = data.loc[
        data["Decision Date"].lt(tstart)
        & data["Outcome Date"].lt(tstart)
    ].copy()

    test = data.loc[
        data["Decision Date"].ge(tstart)
    ].copy()

    if any(
        part.empty
        for part in (
            train,
            valid,
            pretest,
            test,
        )
    ):
        raise ValueError(
            "Chronological split empty after "
            "forward-label purging"
        )

    return (
        train,
        valid,
        pretest,
        test,
        vstart,
        tstart,
    )


def models():
    """Perform the models operation for this research workflow.

    Returns:
        Computed result or status for the caller.
    """
    existing = make_models(quick=True)

    return {
        key: model
        for key, model in existing.items()
        if key in DISPLAY
    }


def checked_predict(model, x):
    """Perform the checked predict operation for this research workflow.

    Args:
        model: Input to the checked_predict operation.
        x: Input to the checked_predict operation.

    Returns:
        Computed result or status for the caller.
    """
    prediction = np.asarray(
        model.predict(x),
        dtype=float,
    )

    expected = (len(x), len(TARGETS))

    if (
        prediction.shape != expected
        or not np.isfinite(prediction).all()
    ):
        raise ValueError(
            "Invalid multi-target risk forecasts"
        )

    return prediction


def risk_score(
    actual,
    predicted,
    scales,
    weights,
):
    """Perform the risk score operation for this research workflow.

    Args:
        actual: Input to the risk_score operation.
        predicted: Input to the risk_score operation.
        scales: Input to the risk_score operation.
        weights: Input to the risk_score operation.

    Returns:
        Computed result or status for the caller.
    """
    errors = (
        np.abs(actual - predicted)
        / scales
    )

    return float(
        np.mean(
            errors.mean(axis=0)
            * np.asarray(weights)
        )
    )


def score_rows(
    actual,
    predicted,
    scales,
    label,
    model_name,
    sample,
):
    """Perform the score rows operation for this research workflow.

    Args:
        actual: Input to the score_rows operation.
        predicted: Input to the score_rows operation.
        scales: Input to the score_rows operation.
        label: Input to the score_rows operation.
        model_name: Input to the score_rows operation.
        sample: Input to the score_rows operation.

    Returns:
        Computed result or status for the caller.
    """
    rows = []

    for appetite, weights in WEIGHTS.items():

        for horizon in ("ALL", *ANCHORS):
            if horizon == "ALL":
                selected = np.ones(
                    len(sample),
                    dtype=bool,
                )
            else:
                selected = (
                    sample["Hedge Horizon Days"]
                    .eq(horizon)
                    .to_numpy(dtype=bool)
                )

            if not selected.any():
                continue

            subset = sample.loc[selected]

            record = {
                "period": label,
                "model": model_name,
                "risk_appetite": appetite,
                "horizon": str(horizon),
                "rows": int(selected.sum()),
                "stocks": int(
                    subset["Underlying RIC"].nunique()
                ),
                "dates": int(
                    subset["Decision Date"].nunique()
                ),
                "weighted_normalized_MAE": risk_score(
                    actual[selected],
                    predicted[selected],
                    scales,
                    weights,
                ),
            }

            for j, target in enumerate(TARGETS):
                record[f"MAE_{target}"] = float(
                    np.mean(
                        np.abs(
                            actual[selected, j]
                            - predicted[selected, j]
                        )
                    )
                )

            rows.append(record)

    return rows


def train():
    """Perform the train operation for this research workflow.
    """
    from sklearn.multioutput import MultiOutputRegressor

    data = read_data()

    coverage = (
        data.groupby("Hedge Horizon Days")
        .agg(
            Stocks=(
                "Underlying RIC",
                "nunique",
            ),
            Decision_Dates=(
                "Decision Date",
                "nunique",
            ),
            Opportunities=(
                "Underlying RIC",
                "size",
            ),
        )
    )

    print(
        "\n=== BROAD STOCK-RISK COVERAGE ==="
    )
    print(
        coverage.to_string(),
        flush=True,
    )

    (
        train_set,
        valid,
        pretest,
        test,
        vstart,
        tstart,
    ) = chronological_split(data)

    print(
        f"\nValidation starts: {vstart.date()}"
    )
    print(
        f"Test starts: {tstart.date()}"
    )

    scales = np.maximum(
        train_set[TARGETS]
        .std()
        .to_numpy(float),
        0.01,
    )

    outcomes = []

    # Select model using validation only.
    for name, model in models().items():
        print(
            "Validating",
            DISPLAY[name],
            flush=True,
        )

        estimator = MultiOutputRegressor(
            model,
            n_jobs=1,
        )

        estimator.fit(
            train_set[FEATURES],
            train_set[TARGETS],
        )

        predicted = checked_predict(
            estimator,
            valid[FEATURES],
        )

        outcomes.extend(
            score_rows(
                valid[TARGETS].to_numpy(float),
                predicted,
                scales,
                "VALIDATION",
                name,
                valid,
            )
        )

    validation = pd.DataFrame(outcomes)

    best = {}

    for appetite in WEIGHTS:
        group = validation.loc[
            validation["risk_appetite"].eq(
                appetite
            )
            & validation["horizon"].eq("ALL")
        ]

        selected = group.sort_values(
            [
                "weighted_normalized_MAE",
                "model",
            ]
        ).iloc[0]

        best[appetite] = str(
            selected["model"]
        )

    # The held-out results are descriptive only.
    # They are NOT used to select the model.

    test_rows = []

    for name in sorted(set(best.values())):
        print(
            "Testing selected",
            DISPLAY[name],
            flush=True,
        )

        estimator = MultiOutputRegressor(
            models()[name],
            n_jobs=1,
        )

        estimator.fit(
            pretest[FEATURES],
            pretest[TARGETS],
        )

        predicted = checked_predict(
            estimator,
            test[FEATURES],
        )

        test_rows.extend(
            score_rows(
                test[TARGETS].to_numpy(float),
                predicted,
                scales,
                "EXPLORATORY_TEST",
                name,
                test,
            )
        )

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    coverage.to_csv(
        OUT / "coverage.csv"
    )

    validation.to_csv(
        OUT / "validation.csv",
        index=False,
    )

    pd.DataFrame(test_rows).to_csv(
        OUT / "exploratory_test.csv",
        index=False,
    )

    # Refit all nine models on the full eligible
    # history for the optional live comparison.
    saved = {}

    for name, model in models().items():
        print(
            "Fitting final",
            DISPLAY[name],
            flush=True,
        )

        fitted = MultiOutputRegressor(
            model,
            n_jobs=1,
        )

        fitted.fit(
            data[FEATURES],
            data[TARGETS],
        )

        filename = f"{name}.joblib"

        joblib.dump(
            fitted,
            OUT / filename,
        )

        saved[name] = filename

    manifest = {
        "status": "RESEARCH_ONLY",
        "anchors": list(ANCHORS),
        "features": FEATURES,
        "targets": TARGETS,
        "best_model_by_risk": best,
        "models": saved,
        "train_opportunities": int(len(data)),
        "unique_stocks": int(
            data["Underlying RIC"].nunique()
        ),
        "unique_decision_dates": int(
            data["Decision Date"].nunique()
        ),
        "validation_start": str(
            vstart.date()
        ),
        "test_start": str(
            tstart.date()
        ),
        "training_last_date": str(
            data["Decision Date"].max().date()
        ),
        "warnings": [
            "Only six historical horizons were trained.",
            "Intermediate durations are model interpolations.",
            "Stock-risk forecasts are not option-hedge P&L.",
            "Adjusted prices are not total shareholder returns.",
            "Corporate actions and historical membership have incomplete verification.",
            "Historical stock behavior may not generalize.",
            "Previously researched test dates are exploratory.",
        ],
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

    print(
        "\n=== VALIDATION WINNERS ==="
    )

    for appetite, key in best.items():
        print(
            f"{appetite}: {DISPLAY[key]}"
        )

    print(
        "\nFinal model directory:",
        OUT,
    )


@lru_cache(maxsize=12)
def _saved_model(filename):
    """Handle the internal saved model step.

    Args:
        filename: Input to the _saved_model operation.

    Returns:
        Computed result or status for the caller.
    """
    return joblib.load(
        OUT / filename
    )


def live_features(
    ric,
    decision_date,
):
    """Recreate prior-session features from fresh adjusted data."""

    import lseg.data as ld

    from data_pipeline.collect_daily_market_prices import (
        ADJUSTMENTS,
        FIELDS,
        normalize_history,
    )

    decision_date = pd.Timestamp(
        decision_date
    ).date()

    start = (
        decision_date
        - timedelta(days=560)
    )

    history = {}

    ld.open_session()

    try:
        for instrument in (
            ric,
            "SPY.P",
        ):
            if instrument in history:
                continue

            response = ld.get_history(
                universe=instrument,
                fields=FIELDS,
                interval="daily",
                start=str(start),
                end=str(decision_date),
                count=3000,
                adjustments=ADJUSTMENTS["adjusted"],
                header_type=ld.HeaderType.NAME,
            )

            prices = normalize_history(
                response,
                instrument,
                "adjusted",
            )

            prices = prices.loc[
                pd.to_datetime(
                    prices["Date"]
                ).dt.date < decision_date
            ]

            series = pd.Series(
                pd.to_numeric(
                    prices["Close"],
                    errors="coerce",
                ).to_numpy(float),
                index=pd.DatetimeIndex(
                    pd.to_datetime(
                        prices["Date"]
                    ).dt.normalize()
                ),
            ).sort_index()

            if series.index.has_duplicates:
                raise ValueError(
                    "Duplicate recent LSEG closes"
                )

            history[instrument] = series

    finally:
        ld.close_session()

    market = history["SPY.P"].dropna()

    stock = history[ric].reindex(
        market.index
    )

    if (
        len(market) < 253
        or not np.isfinite(
            stock.iloc[-253:].to_numpy(float)
        ).all()
    ):
        raise ValueError(
            "253 complete prior stock and SPY "
            "sessions are required"
        )

    if (
        pd.Timestamp(decision_date)
        - market.index[-1]
    ).days > 7:
        raise ValueError(
            "Latest market close too old"
        )

    features = {}

    for window in (
        21,
        63,
        126,
        252,
    ):
        features.update(
            calculate_window(
                stock.iloc[
                    -window - 1:
                ].to_numpy(float),
                market.iloc[
                    -window - 1:
                ].to_numpy(float),
                window,
            )
        )

    if not np.isfinite(
        np.array(
            [
                features[name]
                for name in PRIOR_FEATURES
            ],
            dtype=float,
        )
    ).all():
        raise ValueError(
            "Recent risk features invalid"
        )

    return (
        features,
        str(market.index[-1].date()),
    )


def forecast(
    snapshot,
    appetite="Balanced",
):
    """Perform the forecast operation for this research workflow.

    Args:
        snapshot: Input to the forecast operation.
        appetite: Input to the forecast operation.

    Returns:
        Computed result or status for the caller.
    """
    manifest = json.loads(
        (
            OUT / "manifest.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    if manifest["features"] != FEATURES:
        raise ValueError(
            "Stock-risk feature schema mismatch"
        )

    if appetite not in WEIGHTS:
        raise ValueError(
            "Invalid risk appetite"
        )

    ric = str(
        snapshot["inputs"].ric
    )

    requested = int(
        snapshot["meta"][
            "research_horizon_sessions"
        ]
    )

    effective = min(
        max(requested, 21),
        252,
    )

    decision = pd.Timestamp(
        snapshot["meta"][
            "retrieved_at_new_york"
        ]
    ).date()

    features, prior_date = live_features(
        ric,
        decision,
    )

    features[
        "Hedge Horizon Days"
    ] = effective

    row = pd.DataFrame(
        [features]
    )[FEATURES]

    recommendations = []

    for name, filename in manifest[
        "models"
    ].items():
        prediction = checked_predict(
            _saved_model(filename),
            row,
        )[0].copy()

        prediction[0] = max(
            -1.0,
            prediction[0],
        )

        prediction[1] = np.clip(
            prediction[1],
            -1,
            0,
        )

        prediction[2] = np.clip(
            prediction[2],
            -1,
            0,
        )

        recommendations.append({
            "Model": DISPLAY[name],
            "Key": name,
            "Expected stock return (%)":
                100 * prediction[0],
            "Predicted max drawdown (%)":
                100 * prediction[1],
            "Predicted worst return (%)":
                100 * prediction[2],
        })

    return (
        pd.DataFrame(recommendations),
        manifest,
        requested,
        effective,
        prior_date,
    )


def render_stock_risk(
    snapshot,
    appetite="Balanced",
):
    """Display stock risk in the user interface.

    Args:
        snapshot: Input to the render_stock_risk operation.
        appetite: Input to the render_stock_risk operation.
    """
    import streamlit as st

    with st.expander(
        "Broad S&P 500 ML stock-risk forecasts (optional)",
        expanded=False,
    ):
        st.caption(
            "These forecasts describe underlying "
            "stock risk, not the realized return "
            "of an option hedge."
        )

        cache_key = (
            str(snapshot["inputs"].ric),
            str(
                snapshot["meta"][
                    "retrieved_at_new_york"
                ]
            )[:10],
            int(
                snapshot["meta"][
                    "research_horizon_sessions"
                ]
            ),
        )

        # Fetch at most once per stock/date/horizon; a previous failure is
        # cached as well to avoid refetching on every UI widget interaction.
        cached = st.session_state.get("broad_risk_result")
        if not cached or cached[0] != cache_key:
            with st.spinner(
                "Retrieving recent adjusted stock "
                "and SPY histories..."
            ):
                try:
                    result = forecast(
                        snapshot,
                        appetite,
                    )

                    st.session_state[
                        "broad_risk_result"
                    ] = (
                        cache_key,
                        result,
                    )

                except Exception as exc:
                    st.session_state["broad_risk_result"] = (
                        cache_key, None,
                    )
                    st.warning(
                        "Broad-market risk unavailable: "
                        f"{exc}"
                    )

        cached = st.session_state.get(
            "broad_risk_result"
        )

        if (
            not cached
            or cached[0] != cache_key
            or cached[1] is None
        ):
            return

        (
            table,
            manifest,
            requested,
            effective,
            last_date,
        ) = cached[1]

        best_key = manifest[
            "best_model_by_risk"
        ][appetite]

        selection = table.loc[
            table["Key"].eq(best_key)
        ].iloc[0]

        st.write(
            "**Best historical stock-risk model "
            f"({appetite}): {DISPLAY[best_key]}**"
        )

        st.write(
            "Estimated stock return: "
            f"{selection['Expected stock return (%)']:+.2f}%"
        )

        st.write(
            "Estimated maximum drawdown: "
            f"{selection['Predicted max drawdown (%)']:.2f}%"
        )

        st.caption(
            f"Latest historical close used: {last_date}. "
            f"Model duration: H{effective}."
        )

        if requested not in ANCHORS:
            st.warning(
                "This duration was not directly "
                "represented as a training anchor. "
                "Interpolation or a boundary proxy "
                "is not independently validated."
            )

        st.caption(
            f"Trained on {manifest['unique_stocks']} "
            f"stocks and "
            f"{manifest['train_opportunities']:,} "
            "historical stock-date-horizon observations. "
            "These observations do not establish "
            "option-hedging effectiveness."
        )

        # A default comparison makes the broad-market forecasts visible
        # after their data has been retrieved, without another display toggle.
        from dashboard_visuals import model_ranking_chart
        st.plotly_chart(
            model_ranking_chart(
                table, metric="Predicted max drawdown (%)",
                name_col="Model", title="Nine-model predicted stock drawdown",
            ),
            use_container_width=True,
            key="broad_nine_model_drawdown_chart",
        )
        with st.expander("Recorded nine-model stock-risk estimates"):
            st.dataframe(
                table.drop(columns=["Key"]),
                hide_index=True,
                use_container_width=True,
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
