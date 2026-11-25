
"""Historical ML research view for the Equity Hedge Explorer.

Displays saved 63-session research forecasts.
Does not retrieve option quotes or treat past forecasts as live.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]

FORECAST_DIR = (
    ROOT
    / "data/processed/ml_inference/equity_63d_v1"
)

REGRESSION_COLUMNS = {
    "Predicted maximum loss from entry":
        "pred_max_loss_from_entry_63d",
    "Predicted maximum drawdown":
        "pred_max_drawdown_63d",
    "Predicted realized volatility":
        "pred_realized_vol_63d",
}

PROBABILITY_COLUMNS = {
    "Loss ≥10%": "prob_loss_ge_10pct_63d",
    "Loss ≥15%": "prob_loss_ge_15pct_63d",
    "Loss ≥20%": "prob_loss_ge_20pct_63d",
}


def read_forecasts(path: Path) -> pd.DataFrame:
    """Load forecasts from stored research artifacts.

    Args:
        path: Input to the read_forecasts operation.

    Returns:
        Computed result or status for the caller.
    """
    frame = pd.read_csv(path)

    required = {
        "Decision Date",
        "Underlying RIC",
        *REGRESSION_COLUMNS.values(),
        *PROBABILITY_COLUMNS.values(),
    }

    missing = required - set(frame.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    if frame.empty:
        raise ValueError("Forecast file is empty")

    frame["Decision Date"] = pd.to_datetime(
        frame["Decision Date"],
        errors="raise",
    ).dt.normalize()

    frame["Underlying RIC"] = (
        frame["Underlying RIC"]
        .astype("string")
        .str.strip()
    )

    if (
        frame["Decision Date"].nunique() != 1
        or frame["Underlying RIC"].isna().any()
        or frame["Underlying RIC"].eq("").any()
        or frame.duplicated(
            ["Decision Date", "Underlying RIC"]
        ).any()
    ):
        raise ValueError(
            "Expected one decision date "
            "and unique underlying RICs"
        )

    columns = [
        *REGRESSION_COLUMNS.values(),
        *PROBABILITY_COLUMNS.values(),
    ]

    for column in columns:
        frame[column] = pd.to_numeric(
            frame[column],
            errors="raise",
        )

        if not np.isfinite(
            frame[column].to_numpy(dtype=float)
        ).all():
            raise ValueError(
                f"Invalid numeric forecast: {column}"
            )

    for column in PROBABILITY_COLUMNS.values():
        if not frame[column].between(0, 1).all():
            raise ValueError(
                f"Invalid probability: {column}"
            )

    return frame


def render_historical_ml_view() -> None:
    """Display historical ml view in the user interface.
    """
    st.subheader("Historical ML Research")

    st.info(
        "These are historical 63-session model predictions, "
        "not live forecasts. They are not combined with "
        "current LSEG option quotes. Predicted loss "
        "probabilities are not independently calibrated."
    )

    if not FORECAST_DIR.is_dir():
        st.error(
            f"Forecast directory not found: {FORECAST_DIR}"
        )
        return

    files = sorted(
        FORECAST_DIR.glob("risk_forecasts_*.csv"),
        key=lambda path: (
            path.stat().st_mtime,
            path.name,
        ),
        reverse=True,
    )

    if not files:
        st.warning(
            "No historical ML forecasts found. "
            "Generate them using "
            "data_pipeline.run_ml_risk_inference."
        )
        return

    selected_file = st.selectbox(
        "Saved research forecast",
        files,
        format_func=lambda path: path.name,
    )

    try:
        frame = read_forecasts(selected_file)
    except Exception as exc:
        st.error(
            f"Unable to load forecast: {exc}"
        )
        return

    decision_date = (
        frame["Decision Date"]
        .iloc[0]
        .date()
    )

    st.caption(
        f"Research decision date: {decision_date} | "
        f"Available securities: {len(frame):,}"
    )

    rics = sorted(
        frame["Underlying RIC"]
        .astype(str)
        .tolist()
    )

    default_index = (
        rics.index("NVDA.OQ")
        if "NVDA.OQ" in rics
        else 0
    )

    ric = st.selectbox(
        "Select historical equity",
        rics,
        index=default_index,
    )

    record = frame.loc[
        frame["Underlying RIC"].eq(ric)
    ].iloc[0]

    if "Membership Source" in frame.columns:
        membership = str(
            record["Membership Source"]
        )

        st.caption(
            f"Membership evidence: {membership}"
        )

        if membership != "OBSERVED_SNAPSHOT_MATCH":
            st.warning(
                "This observation has provisional or "
                "otherwise unconfirmed historical "
                "index-membership evidence."
            )

    columns = st.columns(3)

    for container, (label, field) in zip(
        columns,
        REGRESSION_COLUMNS.items(),
    ):
        container.metric(
            label,
            f"{float(record[field]) * 100:.2f}%",
        )


    # ---------------------------------------------------------
    # PREDICTED LOSS-BREACH PROBABILITIES
    # ---------------------------------------------------------

    st.subheader(
        "Predicted loss-breach probabilities"
    )

    probability_table = pd.DataFrame(
        {
            "Estimated probability (%)": [
                float(record[field]) * 100
                for field in PROBABILITY_COLUMNS.values()
            ]
        },
        index=list(PROBABILITY_COLUMNS),
    )

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=probability_table.index.tolist(),
            y=probability_table[
                "Estimated probability (%)"
            ].tolist(),
            text=[
                f"{value:.2f}%"
                for value in probability_table[
                    "Estimated probability (%)"
                ]
            ],
            textposition="outside",
            cliponaxis=False,
            hovertemplate=(
                "%{x}<br>"
                "Estimated probability: %{y:.2f}%"
                "<extra></extra>"
            ),
            name="Loss probability",
        )
    )

    fig.update_yaxes(
        range=[0, 100],
        dtick=20,
        title_text="Estimated probability (%)",
        ticksuffix="%",
        zeroline=True,
    )

    fig.update_xaxes(
        title_text=None,
        tickangle=0,
    )

    fig.update_layout(
        height=420,
        showlegend=False,
        bargap=0.40,
        margin=dict(
            l=20,
            r=20,
            t=30,
            b=30,
        ),
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        key="historical_loss_probability_chart",
    )

    st.caption(
        "Estimated probabilities that the stock "
        "will experience a loss of at least "
        "10%, 15%, or 20% within 63 trading sessions. "
        "These are historical model predictions "
        "and are not independently calibrated."
    )

    if len(frame) > 1:
        st.subheader(
            "Highest estimated 20% loss risk"
        )

        display_columns = [
            "Underlying RIC",
            *PROBABILITY_COLUMNS.values(),
            *REGRESSION_COLUMNS.values(),
        ]

        ranked = frame.sort_values(
            "prob_loss_ge_20pct_63d",
            ascending=False,
        )

        st.dataframe(
            ranked[display_columns].head(50),
            hide_index=True,
            use_container_width=True,
        )

    st.download_button(
        "Download research forecasts",
        data=selected_file.read_bytes(),
        file_name=selected_file.name,
        mime="text/csv",
    )

    manifest_path = selected_file.with_name(
        f"{selected_file.stem}_manifest.json"
    )

    if manifest_path.is_file():
        with st.expander(
            "Forecast audit and model provenance"
        ):
            try:
                manifest = json.loads(
                    manifest_path.read_text(
                        encoding="utf-8"
                    )
                )
                st.json(manifest)

            except (
                json.JSONDecodeError,
                UnicodeError,
            ) as exc:
                st.warning(
                    f"Cannot read manifest: {exc}"
                )

    st.caption(
        "These risk forecasts do not include "
        "option premiums, transaction costs, "
        "or realized hedging performance."
    )
