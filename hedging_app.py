
"""Equity Hedge Explorer: LSEG research-only live hedging interface.

Run from the project root:
    py -m streamlit run hedging_app.py

Stock and ETF prices/quotes may be delayed or stale.
No orders are placed.
Historical 63-session ML research remains separate from live hedging.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
import streamlit as st

from data_pipeline.compare_lseg_hedges import (
    prepare_candidates,
    choose_pair,
)

from data_pipeline.hedge_strategy_payoffs import (
    HedgeInputs,
    expiration_scenarios,
)

from data_pipeline.expanded_hedge_strategies import (
    select_lower_put,
    extra_option_payoffs,
    etf_cross_hedge_stresses,
)

from data_pipeline.historical_ml_view import (
    render_historical_ml_view,
)

from data_pipeline.ml_comparison_dashboard import (
    render_ml_comparison,
)

from data_pipeline.estimate_etf_beta import (
    retrieve_beta_lseg,
    last_completed_nyse_session,
)

from data_pipeline.etf_option_cross_hedges import (
    retrieve_etf_option_structures,
    option_cross_hedge_stresses,
)

from data_pipeline.etf_option_deltas import (
    retrieve_structure_option_deltas,
)

from data_pipeline.etf_option_risk_audit import (
    summarize_etf_option_risks,
)

from data_pipeline.horizon_option_selection import (
    select_nearest_direct_option_pair,
)

from data_pipeline.continuous_hedge_charts import (
    direct_expiration_curves,
    cross_hedge_expiration_curves,
    payoff_figure,
)

from data_pipeline.joint_hedge_heatmap import (
    build_joint_payoff_surface,
    make_joint_heatmap,
    selected_joint_scenario,
)

from lseg_api.instrument_resolver import (
    resolve_lseg_ric,
)

from data_pipeline.ml_project_results import (
    render_project_ml_results,
)

from data_pipeline.ml_decision_audit import (
    render_ml_decision_audit,
)

from data_pipeline.ml_policy_benchmarks import (
    render_policy_benchmarks,
)

from data_pipeline.risk_aware_recommendations import (
    render_risk_aware_recommendations,
)

from data_pipeline.historical_basis_calibration import (
    render_historical_basis_calibration,
)

from data_pipeline.observed_joint_scenarios import (
    render_observed_joint_comparison,
)

from data_pipeline.block_bootstrap import (
    render_block_bootstrap,
)

from data_pipeline.simple_overview import (
    render_simple_overview,
)

from risk_aware_hedge import render_recommendation

from broad_equity_six_horizon import render_stock_risk



ROOT = Path(__file__).resolve().parent

SAVE_DIR = (
    ROOT
    / "data/processed/hedge_comparisons/streamlit_v1"
)

NYSE = mcal.get_calendar("NYSE")
NEW_YORK = ZoneInfo("America/New_York")

MAX_HORIZON_SESSIONS = 5040
HEDGE_FRACTIONS = (0.25, 0.50, 0.75, 1.0)


# ---------------------------------------------------------
# STOCK POSITION WITHOUT DIRECT OPTIONS
# ---------------------------------------------------------

@dataclass(frozen=True)
class StockOnlyInputs:
    """Stock position without eligible direct option contracts.

    Offers the same interface as ``HedgeInputs`` so the rest of the app
    can run when no direct option hedge was found (e.g. for ETF hedging).

    Attributes:
        ric: LSEG instrument code of the stock.
        as_of: Retrieval date (New York time).
        shares: Number of shares owned.
        spot: Reference price of the stock.
        fee_per_contract: Opening fee per option contract (kept for
            interface compatibility).
        expiry: Always ``None`` because no option is selected.
    """

    ric: str
    as_of: date
    shares: int
    spot: float
    fee_per_contract: float
    expiry: date | None = None

    @property
    def contracts(self) -> int:
        """Perform the contracts operation for this research workflow.

        Returns:
            Computed result or status for the caller. (always 0)
        """
        return 0

    @property
    def covered_shares(self) -> int:
        """Perform the covered shares operation for this research workflow.

        Returns:
            Computed result or status for the caller. (always 0)
        """
        return 0

    @property
    def uncovered_shares(self) -> int:
        """Perform the uncovered shares operation for this research workflow.

        Returns:
            Computed result or status for the caller. (all shares)
        """
        return self.shares


# ---------------------------------------------------------
# NYSE TRADING CALENDAR
# ---------------------------------------------------------

def future_market_days(
    as_of: date,
    count: int,
) -> pd.DatetimeIndex:
    """Return the next ``count`` NYSE trading sessions after ``as_of``.

    Weekends and exchange holidays are skipped. ``as_of`` itself is not
    included.

    Args:
        as_of: Reference date (typically today, New York time).
        count: Number of sessions, between 1 and ``MAX_HORIZON_SESSIONS``.

    Returns:
        Timezone-naive, midnight-normalized ``pd.DatetimeIndex`` of exactly
        ``count`` ascending session dates. The last entry is the hedge target date.

    Raises:
        ValueError: If ``count`` is out of range or the NYSE calendar does not
        cover enough sessions.
    """

    if not 1 <= count <= MAX_HORIZON_SESSIONS:
        raise ValueError(
            f"Hedge horizon must be "
            f"1–{MAX_HORIZON_SESSIONS} sessions."
        )

    end = as_of + timedelta(
        days=max(60, int(count * 1.6) + 30)
    )

    days = NYSE.valid_days(
        start_date=as_of + timedelta(days=1),
        end_date=end,
    ).tz_localize(None).normalize()

    if len(days) < count:
        raise ValueError(
            "NYSE calendar did not cover the selected horizon."
        )

    return days[:count]


def sessions_for_end_date(
    as_of: date,
    end_date: date,
) -> int:
    """Convert an end date into a number of NYSE trading sessions.

    A date that is not a trading day is rolled forward to the next session.

    Args:
        as_of: Reference date (today, New York time); not counted.
        end_date: Requested end date of the hedge.

    Returns:
        Number of sessions after ``as_of`` up to and including the first
        session on or after ``end_date``.

    Raises:
        ValueError: If ``end_date`` is not after ``as_of``, lies more than
            20 years ahead, has no following NYSE session, or exceeds
            ``MAX_HORIZON_SESSIONS``.
    """

    if end_date <= as_of:
        raise ValueError(
            "Choose an end date after today (New York time)."
        )

    if (end_date - as_of).days > 365 * 20:
        raise ValueError(
            "Choose an end date within the next 20 years."
        )

    days = NYSE.valid_days(
        start_date=as_of + timedelta(days=1),
        end_date=end_date + timedelta(days=14),
    )

    future_days = days[
        days.date >= end_date
    ]

    if len(future_days) == 0:
        raise ValueError(
            "No NYSE session found after the chosen end date."
        )

    count = int(
        (days <= future_days[0]).sum()
    )

    if not 1 <= count <= MAX_HORIZON_SESSIONS:
        raise ValueError(
            "Requested date exceeds the supported "
            "trading-session limit."
        )

    return count


def trading_sessions_until(
    as_of: date,
    expiry: date,
) -> int:

    """Perform the trading sessions until operation for this research workflow.

    Args:
        as_of: Input to the trading_sessions_until operation.
        expiry: Input to the trading_sessions_until operation.

    Returns:
        Computed result or status for the caller.
    """
    if expiry <= as_of:
        return 0

    return len(
        NYSE.valid_days(
            start_date=as_of + timedelta(days=1),
            end_date=expiry,
        )
    )


# ---------------------------------------------------------
# OPTION MULTIPLIER VERIFICATION
# ---------------------------------------------------------

def verify_multiplier(
    put: pd.Series,
    call: pd.Series,
    multiplier: int,
) -> bool:
    """Check reported option sizes for both contracts."""

    for leg in (put, call):

        reported = []

        for field in (
            "ContractSize",
            "ContractMultiplier",
        ):

            if (
                field in leg.index
                and pd.notna(leg[field])
            ):

                numeric = pd.to_numeric(
                    pd.Series([leg[field]]),
                    errors="coerce",
                ).iloc[0]

                if pd.notna(numeric):
                    reported.append(
                        float(numeric)
                    )

        if not reported or not all(
            np.isclose(v, multiplier)
            for v in reported
        ):
            return False

    return True


# ---------------------------------------------------------
# LSEG STOCK SNAPSHOT
# ---------------------------------------------------------

def get_snapshot(
    ric: str,
    shares: int,
    sessions: int,
    spread: float,
    fee: float,
    assume_standard_contracts: bool,
) -> dict:
    """Fetch stock price and attempt direct-stock option hedges.

    Missing direct options do not prevent ETF cross-hedging.
    """

    import lseg.data as ld

    from data_pipeline.stock_price_timing import (
    get_reference_price_with_timing,)

    from lseg_api.options import (
        get_option_chain,
        add_option_market_data,
        filter_liquid_options,
    )

    stamp = datetime.now(
        NEW_YORK
    )

    as_of = stamp.date()

    target = future_market_days(
        as_of,
        sessions,
    )[-1].date()

    selected = None
    direct_issue = None

    put = None
    call = None

    attempted_direct_expiries = ()

    ld.open_session()

    try:

        stock_price = get_reference_price_with_timing(ric)

        spot = stock_price.value

        if stock_price.is_close_fallback:

            direct_issue = (
                "Only a closing-price reference is available "
                "for the stock. Direct option strategies are "
                "disabled to avoid combining nonsynchronous "
                "stock and option quotes."
            )

        elif shares < 100:

            direct_issue = (
                "Holding contains fewer than 100 shares. "
                "Standard 100-share option contracts "
                "are not used for this position."
            )

        else:

            try:

                chain = get_option_chain(
                    ric,
                    minimum_strike=0.70 * spot,
                    maximum_strike=1.20 * spot,
                    max_results=2000,
                )

                if chain is None or chain.empty:
                    raise ValueError(
                        "No direct-stock option chain returned"
                    )

                option_match = select_nearest_direct_option_pair(
                    chain=chain,
                    spot=spot,
                    target_date=target,
                    max_spread=spread,
                    assume_standard_contracts=(
                        assume_standard_contracts
                    ),
                    prepare_candidates=prepare_candidates,
                    add_option_market_data=add_option_market_data,
                    filter_liquid_options=filter_liquid_options,
                    choose_pair=choose_pair,
                    max_expiries=12,
                )

                put = option_match.put
                call = option_match.call

                selected = option_match.scored_candidates

                attempted_direct_expiries = (
                    option_match.attempted_expiries
                )

            except Exception as exc:

                direct_issue = (
                    "No eligible direct-stock option hedge: "
                    f"{exc}"
                )

                selected = None

    finally:
        ld.close_session()

    meta = {
        "ric": ric,
        "retrieved_at_new_york": stamp.isoformat(),
        "shares": shares,
        "underlying_spot": spot,
        "underlying_price_source": stock_price.source,
        "underlying_stock_retrieved_utc": (
            stock_price.retrieved_utc
        ),
        "underlying_stock_trade_clock_utc": (
            stock_price.trade_clock_utc
        ),
        "underlying_stock_quote_clock_utc": (
            stock_price.quote_clock_utc
        ),
        "underlying_stock_evidence_status": (
            stock_price.evidence_status
        ),
        "underlying_price_date": (
            stock_price.price_date.isoformat()
            if stock_price.price_date is not None
            else None
        ),
        "used_close_fallback": stock_price.is_close_fallback,
        "research_horizon_sessions": sessions,
        "research_target_date": target.isoformat(),
        "max_spread_fraction": spread,
        "fee_per_contract": fee,
        "assumed_multiplier_accepted": (
            assume_standard_contracts
        ),
        "direct_option_issue": direct_issue,
        "has_direct_options": selected is not None,
        "attempted_direct_expiries": [
            str(d)
            for d in attempted_direct_expiries
        ],
        "mode": (
            "REFERENCE_CLOSE_RESEARCH"
            if stock_price.is_close_fallback
            else "INDICATIVE_LSEG_LATEST_AVAILABLE"
        ),
        "historical_ml_combined": False,
        "limitations": [
            "US/NYSE trading calendar assumed.",
            "Quotes may be stale, including over weekends.",
            "Option deliverables and liquidity require verification.",
            "Option expiry may differ from the requested hedge horizon.",
            "Expiry P&L omits taxes, financing, dividends and early exercise.",
            "ETF share cross hedges have borrowing and basis risks.",
        ],
    }

    if selected is None:

        inputs = StockOnlyInputs(
            ric=ric,
            as_of=as_of,
            shares=shares,
            spot=spot,
            fee_per_contract=fee,
        )

        meta["option_expiry"] = None

        return {
            "inputs": inputs,
            "meta": meta,
            "table": pd.DataFrame(),
            "liquid_options": pd.DataFrame(),
            "has_direct_options": False,
        }

    expiry = pd.Timestamp(
        put["ExpiryDate"]
    ).date()

    inputs = HedgeInputs(
        ric=ric,
        as_of=as_of,
        expiry=expiry,
        shares=shares,
        spot=spot,
        put_strike=float(
            put["StrikePrice"]
        ),
        put_ask=float(
            put["ASK"]
        ),
        call_strike=float(
            call["StrikePrice"]
        ),
        call_bid=float(
            call["BID"]
        ),
        source="live",
        multiplier=100,
        fee_per_contract=fee,
    )

    meta.update({
        "option_expiry": expiry.isoformat(),
        "option_expiration_lag_calendar_days": (
            expiry - target
        ).days,
        "actual_calendar_sessions_to_expiry": (
            trading_sessions_until(
                as_of,
                expiry,
            )
        ),
        "selected_liquid_candidates": len(selected),
        "contracts_per_leg": inputs.contracts,
        "covered_shares": inputs.covered_shares,
        "uncovered_shares": inputs.uncovered_shares,
        "multiplier_verified_from_fields": (
            verify_multiplier(
                put,
                call,
                100,
            )
        ),
        "put": {
            "ric": str(put["RIC"]),
            "strike": inputs.put_strike,
            "bid": float(put["BID"]),
            "ask": inputs.put_ask,
        },
        "call": {
            "ric": str(call["RIC"]),
            "strike": inputs.call_strike,
            "bid": inputs.call_bid,
            "ask": float(call["ASK"]),
        },
    })

    return {
        "inputs": inputs,
        "meta": meta,
        "table": expiration_scenarios(inputs),
        "liquid_options": selected.copy(),
        "has_direct_options": True,
    }


# ---------------------------------------------------------
# INTERACTIVE PAYOFF CHART
# ---------------------------------------------------------

def render_payoff_chart(
    curves: pd.DataFrame,
    x_name: str,
    upper: float,
    key: str,
) -> None:
    """Dynamic scenario slider and payoff display."""

    slider_key = f"{key}_cursor"

    previous = float(
        st.session_state.get(
            slider_key,
            0.0,
        )
    )

    st.session_state[slider_key] = min(
        max(previous, -100.0),
        upper,
    )

    x = st.slider(
        f"Scenario cursor for {x_name}",
        min_value=-100.0,
        max_value=float(upper),
        value=0.0,
        step=1.0,
        key=slider_key,
    )

    figure = payoff_figure(
        curves,
        x_name,
        upper,
    )

    figure.add_vline(
        x=x,
        line_dash="dash",
        line_color="gray",
    )

    x_values = curves[
        x_name
    ].to_numpy(dtype=float)

    rows = []

    for name in curves.columns:

        if name == x_name:
            continue

        pnl = float(
            np.interp(
                x,
                x_values,
                curves[name].to_numpy(dtype=float),
            )
        )

        rows.append({
            "Strategy": name,
            "P&L": pnl,
        })

        figure.add_scatter(
            x=[x],
            y=[pnl],
            mode="markers",
            marker={"size": 9},
            showlegend=False,
            hovertemplate=(
                f"{name}<br>"
                "%{x:.1f}%: %{y:,.2f}"
                "<extra></extra>"
            ),
        )

    st.plotly_chart(
        figure,
        use_container_width=True,
        key=f"{key}_plot",
    )

    st.dataframe(
        pd.DataFrame(rows).style.format({
            "P&L": "{:,.2f}",
        }),
        hide_index=True,
        use_container_width=True,
    )


# ---------------------------------------------------------
# DIRECT STOCK-OPTION RESULTS
# ---------------------------------------------------------

def render_direct_options(
    result: dict,
) -> None:
    """Graph a stock position and available direct option hedge payoffs.

    Args:
        result: Validated stock snapshot including eligible option contracts,
            expiration information, shares and reference price.

    Notes:
        Payoffs refer to option expiration. If options are unavailable, the
        unhedged underlying payoff curve remains visible.
    """
    inputs = result["inputs"]
    meta = result["meta"]

    if not result["has_direct_options"]:

        st.warning(
            meta["direct_option_issue"]
        )

        st.info(
            "ETF cross-hedging is still available "
            "for this stock position."
        )

        st.subheader(
            "Unhedged stock payoff at the research target"
        )

        upper = float(
            st.number_input(
                "Maximum stock gain shown (%)",
                min_value=0.0,
                value=100.0,
                step=50.0,
                key="stock_only_upper",
            )
        )

        x_values = np.linspace(
            -100.0,
            upper,
            401,
        )

        stock_curve = pd.DataFrame({
            "Underlying move (%)": x_values,
            "Unhedged": (
                inputs.shares
                * inputs.spot
                * x_values
                / 100.0
            ),
        })

        render_payoff_chart(
            stock_curve,
            "Underlying move (%)",
            upper,
            "stock_only",
        )

        return

    a, b, c, d = st.columns(4)

    a.metric(
        "Stock quote",
        f"{inputs.spot:,.2f}",
    )

    b.metric(
        "Put strike / ask",
        f"{inputs.put_strike:.2f} / "
        f"{inputs.put_ask:.2f}",
    )

    c.metric(
        "Call strike / bid",
        f"{inputs.call_strike:.2f} / "
        f"{inputs.call_bid:.2f}",
    )

    d.metric(
        "Collar premium cashflow",
        f"{inputs.covered_shares * (inputs.call_bid - inputs.put_ask):,.2f}",
    )

    st.write(
        f"Research target: **{meta['research_target_date']}** "
        f"({meta['research_horizon_sessions']} NYSE sessions). "
        f"Direct option expiry: **{inputs.expiry}** "
        f"({meta['actual_calendar_sessions_to_expiry']} NYSE sessions)."
    )

    st.caption(
        f"Contracts per leg: {inputs.contracts}; "
        f"covered shares: {inputs.covered_shares}; "
        f"uncovered shares: {inputs.uncovered_shares}. "
        f"Multiplier verified: "
        f"{meta['multiplier_verified_from_fields']}."
    )

    if (
        meta["actual_calendar_sessions_to_expiry"]
        != meta["research_horizon_sessions"]
    ):
        st.warning(
            "Direct-option expiration P&L is not at "
            "the chosen research target date."
        )

    lower = select_lower_put(
        result["liquid_options"],
        long_put_strike=inputs.put_strike,
        expiry=inputs.expiry,
        spot=inputs.spot,
        max_spread_fraction=1.0,
    )

    if lower is not None:

        extra = extra_option_payoffs(
            inputs,
            lower_put_strike=float(
                lower["StrikePrice"]
            ),
            lower_put_bid=float(
                lower["BID"]
            ),
        )

        meta["put_spread_short_put"] = {
            "ric": str(lower["RIC"]),
            "strike": float(
                lower["StrikePrice"]
            ),
            "bid": float(
                lower["BID"]
            ),
        }

    else:

        extra = extra_option_payoffs(inputs)

        st.info(
            "No eligible lower-strike put; "
            "direct put spread omitted."
        )

    scenarios = pd.concat(
        [
            result["table"],
            extra,
        ],
        ignore_index=True,
    )

    st.subheader(
        "Direct-option strategy comparison at expiration"
    )

    upper = float(
        st.number_input(
            "Maximum stock gain shown (%)",
            min_value=0.0,
            value=100.0,
            step=50.0,
            key="direct_upper",
        )
    )

    curves = direct_expiration_curves(
        inputs,
        upper_pct=upper,
        lower_put_strike=(
            float(lower["StrikePrice"])
            if lower is not None
            else None
        ),
        lower_put_bid=(
            float(lower["BID"])
            if lower is not None
            else None
        ),
    )

    render_payoff_chart(
        curves,
        "Underlying move (%)",
        upper,
        "direct",
    )

    st.caption(
        "X-axis is bounded below at −100%. "
        "The positive range is editable. "
        "All option payoffs are evaluated at expiration."
    )

    st.dataframe(
        scenarios,
        hide_index=True,
        use_container_width=True,
    )

    minimums = (
        scenarios.groupby(
            "Strategy"
        )["Total P&L"]
        .min()
        .to_dict()
    )

    st.caption(
        "Worst P&L among displayed scenarios "
        "(not necessarily the global maximum loss): "
        + ", ".join(
            f"{name}: {value:,.2f}"
            for name, value in minimums.items()
        )
    )

    st.download_button(
        "Download direct-option scenario CSV",
        scenarios.to_csv(index=False),
        file_name=(
            f"{inputs.ric}_{inputs.expiry}_direct_options.csv"
        ),
        mime="text/csv",
    )

    with st.expander(
        "Direct option quote audit"
    ):
        st.json(meta)

    SAVE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    retrieval_time = datetime.fromisoformat(
        meta["retrieved_at_new_york"]
    )

    filename = (
        f"{inputs.ric.replace('/', '_')}_"
        f"{retrieval_time.strftime('%Y%m%d_%H%M%S_%f')}"
    )

    csv_file = SAVE_DIR / f"{filename}_payoffs.csv"

    if not csv_file.exists():

        scenarios.to_csv(
            csv_file,
            index=False,
        )

        audit_file = (
            SAVE_DIR / f"{filename}_audit.json"
        )

        audit_file.write_text(
            json.dumps(meta, indent=2),
            encoding="utf-8",
        )


# ---------------------------------------------------------
# ETF CROSS-HEDGE RETRIEVAL
# ---------------------------------------------------------

def evaluate_cross_hedge(
    result: dict,
    etf_text: str,
    beta_mode: str,
    manual_beta: float,
    stock_shock: float,
    etf_shock: float,
    include_options: bool,
) -> dict:
    """Retrieve ETF price, beta, option quotes and Greeks."""

    import lseg.data as ld

    from lseg_api.price_reference import get_reference_price

    inputs = result["inputs"]
    meta = result["meta"]

    etf_ric = resolve_lseg_ric(
        etf_text
    )

    if etf_ric.upper() == inputs.ric.upper():
        raise ValueError(
            "ETF must differ from the stock being hedged."
        )

    structures = {}
    net_deltas = {}
    leg_deltas = {}

    option_issue = None
    beta_report = None

    ld.open_session()

    try:

        etf_price = get_reference_price(etf_ric)

        etf_spot = etf_price.value

        if beta_mode == "Automatic — 252 sessions":

            beta_report = retrieve_beta_lseg(
                inputs.ric,
                etf_ric,
                last_completed_nyse_session(),
            )

            beta = float(
                beta_report["beta_stock_on_etf"]
            )

        else:
            beta = manual_beta

        if not np.isfinite(beta):
            raise ValueError(
                "Beta must be finite"
            )

        if include_options:

            if (
                meta.get("used_close_fallback", False)
                or etf_price.is_close_fallback
            ):

                option_issue = (
                    "ETF option comparison temporarily "
                    "disabled: at least one underlying "
                    "price is available only as a closing "
                    "reference. Use synchronized market "
                    "quotes before comparing option premiums."
                )

            elif beta <= 0:

                option_issue = (
                    "Bearish ETF option structures "
                    "require positive beta."
                )

            else:

                try:

                    structures = retrieve_etf_option_structures(
                        etf_ric,
                        etf_spot,
                        date.fromisoformat(
                            meta["research_target_date"]
                        ),
                        max_spread=meta[
                            "max_spread_fraction"
                        ],
                        assume_standard_contracts=meta[
                            "assumed_multiplier_accepted"
                        ],
                    )

                    (
                        net_deltas,
                        leg_deltas,
                    ) = retrieve_structure_option_deltas(
                        structures
                    )

                except Exception as exc:

                    structures = {}
                    net_deltas = {}
                    leg_deltas = {}

                    option_issue = (
                        "ETF option discovery/quotes/Greeks: "
                        f"{exc}"
                    )

    finally:
        ld.close_session()

    share_table = etf_cross_hedge_stresses(
        stock_spot=inputs.spot,
        shares=inputs.shares,
        etf_spot=etf_spot,
        beta=beta,
    )

    selected_share = share_table.loc[
        np.isclose(
            share_table["Stock Shock"],
            stock_shock,
        )
        & np.isclose(
            share_table["ETF Shock"],
            etf_shock,
        )
    ].copy()

    if selected_share.empty:
        raise ValueError(
            "Selected ETF-share scenario was not produced"
        )

    option_table = None
    selected_option = None

    if structures:

        option_table = option_cross_hedge_stresses(
            stock_spot=inputs.spot,
            shares=inputs.shares,
            etf_spot=etf_spot,
            beta=beta,
            structures=structures,
            option_net_deltas=net_deltas,
            fee_per_contract=inputs.fee_per_contract,
        )

        selected_option = option_table.loc[
            np.isclose(
                option_table["Stock Shock"],
                stock_shock,
            )
            & np.isclose(
                option_table["ETF Shock"],
                etf_shock,
            )
        ].copy()

    return {
        "snapshot": meta["retrieved_at_new_york"],
        "etf_ric": etf_ric,
        "etf_spot": etf_spot,
        "etf_price_source": etf_price.source,
        "etf_price_date": (
            etf_price.price_date.isoformat()
            if etf_price.price_date is not None
            else None),
        "etf_close_fallback": etf_price.is_close_fallback,
        "beta": beta,
        "beta_report": beta_report,
        "stock_shock": stock_shock,
        "etf_shock": etf_shock,
        "selected": selected_share,
        "all_scenarios": share_table,
        "option_structures": structures,
        "option_net_deltas": net_deltas,
        "option_leg_deltas": leg_deltas,
        "option_issue": option_issue,
        "option_selected": selected_option,
        "option_scenarios": option_table,
        "option_fee_per_contract": inputs.fee_per_contract,
    }


# ---------------------------------------------------------
# ETF OPTION RESULTS
# ---------------------------------------------------------

def render_cross_options(
    cross: dict,
) -> None:

    """Display cross options in the user interface.

    Args:
        cross: Input to the render_cross_options operation.
    """
    selected = cross.get(
        "option_selected"
    )

    if selected is None or selected.empty:

        if cross.get("option_issue"):

            st.warning(
                cross["option_issue"]
            )

        else:

            st.info(
                "No ETF options evaluated. "
                "ETF share hedge remains available."
            )

        return

    st.subheader(
        "ETF Option Cross Hedges — Expiration P&L"
    )

    st.caption(
        "Contract quantities use the selected options' "
        "LSEG TR.DELTA values. Initial delta exposure "
        "does not guarantee a loss floor."
    )

    columns = [
        "Strategy",
        "Option Expiry",
        "Hedge Fraction",
        "Net Delta Per Structure",
        "Delta Source",
        "Contracts Per Leg",
        "Unhedged P&L",
        "Option Premium Cashflow",
        "ETF Option P&L",
        "Total P&L",
    ]

    st.dataframe(
        selected[columns].style.format({
            "Hedge Fraction": "{:.0%}",
            "Net Delta Per Structure": "{:.4f}",
            "Unhedged P&L": "{:,.2f}",
            "Option Premium Cashflow": "{:,.2f}",
            "ETF Option P&L": "{:,.2f}",
            "Total P&L": "{:,.2f}",
        }),
        hide_index=True,
        use_container_width=True,
    )

    risk = summarize_etf_option_risks(
        cross["option_structures"],
        selected,
        fee_per_contract=cross[
            "option_fee_per_contract"
        ],
    )

    st.subheader(
        "ETF option risk limits — options only"
    )

    st.dataframe(
        risk,
        hide_index=True,
        use_container_width=True,
    )

    st.caption(
        "Risk limits do not bound losses on the stock "
        "position or cover assignment and financing costs."
    )

    details = []

    for name, structure in cross[
        "option_structures"
    ].items():

        for leg in structure.legs:

            details.append({
                "Strategy": name,
                "Side": (
                    "BUY"
                    if leg.side == 1
                    else "SELL"
                ),
                "Right": leg.right,
                "RIC": leg.ric,
                "Strike": leg.strike,
                "Premium": leg.premium,
                "Expiry": leg.expiry.isoformat(),
                "LSEG Delta": cross[
                    "option_leg_deltas"
                ][leg.ric.upper()],
            })

    with st.expander(
        "ETF option contracts, premiums and Greeks"
    ):

        st.dataframe(
            pd.DataFrame(details),
            hide_index=True,
            use_container_width=True,
        )

    st.download_button(
        "Download ETF option scenario CSV",
        cross["option_scenarios"].to_csv(
            index=False
        ),
        file_name="etf_option_cross_hedges.csv",
        mime="text/csv",
    )


# ---------------------------------------------------------
# ETF CROSS-HEDGE CHARTS AND HEATMAP
# ---------------------------------------------------------

def render_cross_charts(
    inputs,
    meta: dict,
    cross: dict,
) -> None:
    """Show ETF payoff curves and a joint stock–ETF stress heatmap.

    Args:
        inputs: Stock position and any validated direct option information.
        meta: Underlying quote and hedge/option-expiry timing metadata.
        cross: ETF price, beta, hedge quantities, candidate option structures,
            and calculated scenarios from ``evaluate_cross_hedge``.

    Notes:
        Heatmap axes are hypothetical returns, not outcome probabilities.
        Option structures are valued at their expiration, not necessarily the
        user's selected target date.
    """
    st.subheader(
        "ETF Cross-Hedge Payoff Comparison"
    )

    st.caption(
        "Stock return is held fixed while ETF return "
        "varies. ETF basis risk remains."
    )

    fraction = st.select_slider(
        "ETF hedge fraction (chart)",
        options=list(HEDGE_FRACTIONS),
        value=0.50,
        format_func=lambda v: f"{v:.0%}",
        key="cross_fraction_chart",
    )

    stock_change = float(
        st.number_input(
            "Assumed stock change (%)",
            min_value=-100.0,
            value=float(
                cross["stock_shock"] * 100
            ),
            step=5.0,
            key="cross_stock_change",
        )
    )

    upper = float(
        st.number_input(
            "Maximum ETF gain on graph (%)",
            min_value=0.0,
            value=100.0,
            step=50.0,
            key="cross_upper",
        )
    )

    curves = cross_hedge_expiration_curves(
        stock_spot=inputs.spot,
        stock_shares=inputs.shares,
        stock_move_pct=stock_change,
        etf_spot=cross["etf_spot"],
        upper_etf_pct=upper,
        hedge_fraction=fraction,
        share_stresses=cross["all_scenarios"],
        option_structures=cross["option_structures"],
        option_stresses=cross["option_scenarios"],
        fee_per_contract=cross[
            "option_fee_per_contract"
        ],
    )

    render_payoff_chart(
        curves,
        "ETF move (%)",
        upper,
        "cross",
    )

    structures = cross[
        "option_structures"
    ]

    if structures:

        expiries = {
            leg.expiry
            for structure in structures.values()
            for leg in structure.legs
        }

        target = date.fromisoformat(
            meta["research_target_date"]
        )

        if expiries != {target}:

            st.info(
                "ETF option expiry "
                f"{', '.join(map(str, sorted(expiries)))} "
                f"differs from target {target}. "
                "These are expiration payoffs, not "
                "guaranteed target-date values."
            )

        if (
            inputs.expiry is not None
            and expiries != {inputs.expiry}
        ):

            st.warning(
                "ETF and direct-stock options expire "
                "on different dates. Do not rank them "
                "as same-horizon results."
            )

    st.subheader(
        "Joint Stock–ETF Hedge Risk Heatmap"
    )

    left, right = st.columns(2)

    with left:

        stock_upper = float(
            st.number_input(
                "Heatmap maximum stock gain (%)",
                min_value=0.0,
                value=100.0,
                step=50.0,
                key="heat_stock_upper",
            )
        )

    with right:

        etf_upper = float(
            st.number_input(
                "Heatmap maximum ETF gain (%)",
                min_value=0.0,
                value=100.0,
                step=50.0,
                key="heat_etf_upper",
            )
        )

    heat_fraction = st.select_slider(
        "ETF hedge fraction (heatmap)",
        options=list(HEDGE_FRACTIONS),
        value=0.50,
        format_func=lambda v: f"{v:.0%}",
        key="heat_fraction",
    )

    surface = build_joint_payoff_surface(
        stock_spot=inputs.spot,
        stock_shares=inputs.shares,
        etf_spot=cross["etf_spot"],
        hedge_fraction=heat_fraction,
        share_stresses=cross["all_scenarios"],
        option_structures=structures,
        option_stresses=cross["option_scenarios"],
        fee_per_contract=cross[
            "option_fee_per_contract"
        ],
        stock_max_pct=stock_upper,
        etf_max_pct=etf_upper,
    )

    heat_mode = st.radio(
        "Heatmap metric",
        [
            "Total P&L",
            "Benefit versus unhedged",
        ],
        horizontal=True,
        key="heat_mode",
    )

    strategies = list(
        surface["payoffs"]
    )

    if heat_mode == "Benefit versus unhedged":

        strategies.remove(
            "Unhedged stock"
        )

    if (
        st.session_state.get(
            "heat_strategy"
        )
        not in strategies
    ):

        st.session_state.pop(
            "heat_strategy",
            None,
        )

    default_index = (
        strategies.index(
            "Short/long ETF shares"
        )
        if "Short/long ETF shares" in strategies
        else 0
    )

    strategy = st.selectbox(
        "Strategy for heatmap",
        options=strategies,
        index=default_index,
        key="heat_strategy",
    )

    for key, maximum in (
        ("heat_stock_cursor", stock_upper),
        ("heat_etf_cursor", etf_upper),
    ):

        if key in st.session_state:

            st.session_state[key] = min(
                max(
                    float(
                        st.session_state[key]
                    ),
                    -100.0,
                ),
                maximum,
            )

    left_cursor, right_cursor = st.columns(2)

    with left_cursor:

        stock_cursor = st.slider(
            "Stock change (%)",
            min_value=-100.0,
            max_value=stock_upper,
            value=-20.0,
            step=1.0,
            key="heat_stock_cursor",
        )

    with right_cursor:

        etf_cursor = st.slider(
            "ETF change (%)",
            min_value=-100.0,
            max_value=etf_upper,
            value=-10.0,
            step=1.0,
            key="heat_etf_cursor",
        )

    heat_figure = make_joint_heatmap(
        surface,
        strategy,
        mode=heat_mode,
        stock_cursor=stock_cursor,
        etf_cursor=etf_cursor,
    )

    st.plotly_chart(
        heat_figure,
        use_container_width=True,
        key="joint_heatmap",
    )

    scenario_results = selected_joint_scenario(
        surface,
        stock_cursor,
        etf_cursor,
    )

    st.dataframe(
        scenario_results.style.format({
            "Total P&L": "{:,.2f}",
            "Benefit vs unhedged": "{:,.2f}",
        }),
        hide_index=True,
        use_container_width=True,
    )

    st.caption(
        "Heatmap return combinations carry no "
        "probabilities. ETF options are valued "
        "at expiration only."
    )


# ---------------------------------------------------------
# COMPACT, ONE-ACTION HEDGING DASHBOARD
# ---------------------------------------------------------

FX_DISCLAIMER = (
    "Currency warning: the stock and ETF may be quoted in different "
    "currencies. The app does not verify currency parity, convert ETF "
    "prices, premiums or P&L to the stock currency, or model FX moves. "
    "Cross-currency combined P&L, hedge quantities and strategy rankings "
    "are illustrative and may be materially wrong. Use a common-currency "
    "valuation before making a financial decision."
)


RESEARCH_PAGES = {
    "Overview": "A concise summary of saved stock-risk forecasts and model validation.",
    "Stock Risk Forecasts": "Historical 63-session loss probabilities, volatility and drawdown.",
    "Model Comparison": "Validation metrics for saved regression and classification models.",
    "Experiment Results": "Recorded classification, regression and hedge-candidate experiments.",
    "Hedging Performance": "Matched-date ML, rule-based and unhedged historical outcomes.",
    "Decision Quality Audit": "Paired decision regret versus the Historical Mean benchmark.",
}


def _render_quick_direct_payoff(result: dict) -> None:
    """Plot the stock's expiration payoff without exposing research controls.

    Args:
        result: Validated stock snapshot and its eligible direct option quotes.

    Notes:
        All options are valued at their actual expiry, not necessarily the
        requested hedge date; chart movement values are hypothetical.
    """
    inputs = result["inputs"]
    meta = result["meta"]
    st.subheader("Stock protection and expiration payoff")

    if not result["has_direct_options"]:
        st.warning(meta.get("direct_option_issue", "No eligible stock option found."))
        x = np.linspace(-100.0, 100.0, 401)
        curves = pd.DataFrame({
            "Underlying move (%)": x,
            "Unhedged stock": inputs.shares * inputs.spot * x / 100.0,
        })
    else:
        # Find the optional lower put for the put-spread comparison; no
        # additional LSEG request is required because quotes are cached.
        lower = select_lower_put(
            result["liquid_options"],
            long_put_strike=inputs.put_strike,
            expiry=inputs.expiry,
            spot=inputs.spot,
            max_spread_fraction=1.0,
        )
        curves = direct_expiration_curves(
            inputs,
            upper_pct=100.0,
            lower_put_strike=(float(lower["StrikePrice"]) if lower is not None else None),
            lower_put_bid=(float(lower["BID"]) if lower is not None else None),
        )
        st.caption(
            f"Direct option expiry: {inputs.expiry}. "
            f"Hedge target: {meta['research_target_date']}. "
            "The two dates can differ."
        )

    st.plotly_chart(
        payoff_figure(curves, "Underlying move (%)", 100.0),
        use_container_width=True,
        key="quick_direct_payoff",
    )
    st.caption("Hover over the lines to compare hypothetical expiration P&L.")


def _render_quick_etf_payoffs(result: dict, cross: dict) -> None:
    """Display an ETF comparison and joint heatmap from a saved snapshot.

    Args:
        result: Underlying position snapshot containing its price and shares.
        cross: ETF snapshot and already-calculated scenario/option structures.

    Notes:
        Rerendering either graph makes no network requests. Joint returns
        are hypothetical, without probabilities or cross-currency conversion.
    """
    inputs = result["inputs"]
    st.subheader("ETF protection scenarios")
    st.caption(
        f"Comparison ETF: {cross['etf_ric']} · "
        f"ETF reference price: {cross['etf_spot']:,.2f} · "
        f"Stock-on-ETF beta: {cross['beta']:.2f}"
    )
    if cross.get("option_issue"):
        st.info(f"ETF options: {cross['option_issue']}")

    # A conditional 20% stock loss is a visual scenario, not a forecast.
    curves = cross_hedge_expiration_curves(
        stock_spot=inputs.spot,
        stock_shares=inputs.shares,
        stock_move_pct=-20.0,
        etf_spot=cross["etf_spot"],
        upper_etf_pct=100.0,
        hedge_fraction=0.5,
        share_stresses=cross["all_scenarios"],
        option_structures=cross["option_structures"],
        option_stresses=cross["option_scenarios"],
        fee_per_contract=cross["option_fee_per_contract"],
    )
    st.plotly_chart(
        payoff_figure(curves, "ETF move (%)", 100.0),
        use_container_width=True,
        key="quick_etf_payoff",
    )

    # Preserve the research engine's original joint-stress calculations;
    # only chart controls and presentation are simplified here.
    surface = build_joint_payoff_surface(
        stock_spot=inputs.spot,
        stock_shares=inputs.shares,
        etf_spot=cross["etf_spot"],
        hedge_fraction=0.5,
        share_stresses=cross["all_scenarios"],
        option_structures=cross["option_structures"],
        option_stresses=cross["option_scenarios"],
        fee_per_contract=cross["option_fee_per_contract"],
        stock_max_pct=100.0,
        etf_max_pct=100.0,
    )
    strategies = list(surface["payoffs"])
    default = (
        strategies.index("Short/long ETF shares")
        if "Short/long ETF shares" in strategies
        else 0
    )
    strategy = st.selectbox(
        "Strategy displayed on the heatmap",
        options=strategies,
        index=default,
        key="quick_heat_strategy",
    )
    st.plotly_chart(
        make_joint_heatmap(
            surface,
            strategy,
            mode="Total P&L",
            stock_cursor=-20.0,
            etf_cursor=-10.0,
        ),
        use_container_width=True,
        key="quick_joint_heatmap",
    )
    st.caption(
        "Fixed 50% ETF hedge shown. The grid represents possible stock/ETF "
        "changes, not statistical probabilities. ETF options are priced "
        "at expiration. Currency conversion and FX risk are omitted."
    )


def _render_quick_historical_risk(result: dict, cross: dict) -> None:
    """Plot historical stock/ETF tail-risk and risk/return comparisons.

    Args:
        result: Validated current stock snapshot.
        cross: Matching ETF result for the same retrieval timestamp.

    Notes:
        Only previously cached horizon-matched market observations are used.
        This is a descriptive comparison, not a live prediction or proof of
        model superiority. No new LSEG calls or user run buttons are needed.
    """
    from dashboard_visuals import (
        strategy_tail_loss_chart,
        strategy_tradeoff_chart,
    )
    from data_pipeline.simple_overview import (
        AVERAGE,
        RISK,
        build_strategy_overview,
    )

    saved = st.session_state.get("empirical_basis_result")
    if not isinstance(saved, dict):
        return

    try:
        summary, ready = build_strategy_overview(result, cross, saved)
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        st.info(f"Historical comparison cannot be assembled: {exc}")
        return
    if not ready:
        return

    st.subheader("Historical downside risk and return")
    left, right = st.columns(2)
    with left:
        st.plotly_chart(
            strategy_tail_loss_chart(summary, RISK),
            use_container_width=True,
            key="quick_historical_tail_chart",
        )
    with right:
        st.plotly_chart(
            strategy_tradeoff_chart(summary, RISK, AVERAGE),
            use_container_width=True,
            key="quick_historical_tradeoff_chart",
        )
    st.caption(
        "Descriptive overlapping historical windows, not live forecasts. "
        "CVaR estimates can be unstable with few independent periods. "
        "Cross-currency FX conversion and short-ETF costs are excluded."
    )


def _auto_calibrate_history(result: dict, cross: dict | None) -> None:
    """Retrieve shared historical returns once after a successful calculation.

    Args:
        result: Current stock snapshot.
        cross: Current ETF analysis, or None if ETF retrieval failed.

    Notes:
        Requires a measured beta and R-squared. Failure never discards
        the live snapshot or triggers repeated retrieval during chart changes.
    """
    if not cross or not isinstance(cross.get("beta_report"), dict):
        return

    from data_pipeline.quick_calibration import fetch_quick_calibration

    with st.spinner("Preparing the historical risk comparison..."):
        try:
            st.session_state["empirical_basis_result"] = (
                fetch_quick_calibration(result, cross, lookback_years=8)
            )
            st.session_state.pop("ui_calibration_warning", None)
        except Exception as exc:
            # The historical step is optional. LSEG entitlement and network
            # errors must not invalidate successful stock/ETF snapshots.
            st.session_state.pop("empirical_basis_result", None)
            st.session_state["ui_calibration_warning"] = (
                f"Historical calibration unavailable: {type(exc).__name__}: {exc}"
            )


def _render_historical_research() -> None:
    """Navigate concise historical results or the five audit-grade panels.

    Saved research never retrieves current option quotes, retrains models,
    or silently treats exploratory tests as a final independent holdout.
    """
    page = st.sidebar.selectbox(
        "Historical research view",
        options=list(RESEARCH_PAGES),
        key="historical_research_page_v2",
    )
    st.caption(RESEARCH_PAGES[page])

    if page == "Overview":
        from data_pipeline.historical_research_overview import (
            render_historical_research_overview,
        )
        render_historical_research_overview(ROOT)
    elif page == "Stock Risk Forecasts":
        render_historical_ml_view()
    elif page == "Model Comparison":
        metrics_dir = (
            ROOT / "data" / "processed" / "ml_baselines"
            / "equity_63d_suite_v1" / "full"
        )
        render_ml_comparison(metrics_dir)
    elif page == "Experiment Results":
        render_project_ml_results(ROOT)
    elif page == "Hedging Performance":
        render_policy_benchmarks(ROOT)
    else:
        render_ml_decision_audit(ROOT)


def main() -> None:
    """Run an approachable calculator while retaining the advanced research.

    Workflow:
        1. Navigate between a concise calculator, advanced live research and
           historical saved experiments.
        2. Enter a position once and press Calculate: load the stock, eligible
           direct option hedges and optionally the ETF cross hedge.
        3. Automatically attempt paired historical calibration if available.
        4. Reuse the saved results in both views; chart edits do not re-query LSEG.

    Notes:
        There is no automated execution or currency conversion. Even when
        the ETF and stock prices are quoted in different currencies, the
        experimental plots can be shown, but monetary combined results
        may not be economically interpretable.
    """
    st.set_page_config(page_title="Equity Hedge Explorer", layout="wide")
    st.title("Equity Hedge Explorer")
    st.caption("LSEG-powered academic research · No order execution")

    page = st.sidebar.radio(
        "Navigation",
        ["Hedge Calculator", "Advanced Hedge Research", "Historical ML Research"],
        key="application_page_v2",
    )
    if page == "Historical ML Research":
        _render_historical_research()
        return

    advanced = page == "Advanced Hedge Research"
    st.write(
        "Enter your position once. The program retrieves available hedges, "
        "estimates risk and prepares its payoff charts together."
    )

    # A single form prevents an expensive LSEG call on every keystroke.
    # Charts outside the form rerender from saved snapshot data automatically.
    with st.form("calculate_position_v2"):
        left, middle, right = st.columns(3)
        with left:
            ticker = st.text_input(
                "Stock ticker / LSEG RIC", value="NVDA", key="calc_ticker_v2"
            ).strip()
            shares = int(st.number_input(
                "Shares owned", min_value=1, value=1000, step=1,
                key="calc_shares_v2",
            ))
        with middle:
            if advanced:
                horizon_mode = st.radio(
                    "Hedge duration input",
                    ["Trading sessions", "End date"], horizontal=True,
                    key="calc_horizon_mode_v2",
                )
            else:
                horizon_mode = "Trading sessions"
            if horizon_mode == "Trading sessions":
                sessions = int(st.number_input(
                    "Hedge duration (NYSE trading sessions)",
                    min_value=1, max_value=MAX_HORIZON_SESSIONS,
                    value=63, step=1, key="calc_sessions_v2",
                ))
                end_date = None
            else:
                now = datetime.now(NEW_YORK).date()
                end_date = st.date_input(
                    "Requested end date",
                    value=future_market_days(now, 63)[-1].date(),
                    min_value=now + timedelta(days=1),
                    max_value=now + timedelta(days=365 * 20),
                    key="calc_end_date_v2",
                )
                sessions = None
        with right:
            etf = st.text_input(
                "Comparison ETF (optional)", value="SPY", key="calc_etf_v2"
            ).strip()
            risk_appetite = st.select_slider(
                "Risk appetite", options=["Low", "Balanced", "High"],
                value="Balanced", key="calc_risk_v2",
            )

        # Research-only controls are not present on the main calculator.
        if advanced:
            with st.expander("Market-data and calculation settings"):
                settings_a, settings_b = st.columns(2)
                with settings_a:
                    spread = st.slider(
                        "Maximum option bid/ask spread (%)",
                        min_value=1, max_value=30, value=10,
                        key="calc_spread_v2",
                    )
                    fee = float(st.number_input(
                        "Opening fee per option contract", min_value=0.0,
                        value=0.0, step=0.5, key="calc_fee_v2",
                    ))
                with settings_b:
                    beta_mode = st.radio(
                        "ETF beta estimation",
                        ["Automatic — 252 sessions", "Manual beta assumption"],
                        key="calc_beta_mode_v2",
                    )
                    beta = float(st.number_input(
                        "Manual stock-to-ETF beta", min_value=-5.0,
                        max_value=5.0, value=1.0, step=0.1,
                        disabled=beta_mode != "Manual beta assumption",
                        key="calc_beta_v2",
                    ))
                    include_options = st.checkbox(
                        "Include ETF option cross hedges", value=True,
                        key="calc_etf_options_v2",
                    )
        else:
            spread, fee = 10, 0.0
            beta_mode, beta, include_options = "Automatic — 252 sessions", 1.0, True

        calculate = st.form_submit_button("Calculate hedge", type="primary")

    if calculate:
        # Save the complete result only after stock retrieval succeeds;
        # a failed ETF lookup must not erase an otherwise usable stock hedge.
        st.session_state.pop("ui_etf_warning", None)
        st.session_state.pop("ui_calibration_warning", None)
        st.session_state.pop("etf_cross_result", None)
        st.session_state.pop("empirical_basis_result", None)
        st.session_state.pop("hedge_result", None)
        try:
            if not ticker:
                raise ValueError("Enter a stock ticker or full LSEG RIC.")
            now = datetime.now(NEW_YORK).date()
            requested_sessions = (
                sessions if end_date is None else sessions_for_end_date(now, end_date)
            )
            with st.spinner("Retrieving stock and direct option hedges..."):
                result = get_snapshot(
                    ric=resolve_lseg_ric(ticker),
                    shares=shares,
                    sessions=requested_sessions,
                    spread=spread / 100.0,
                    fee=fee,
                    assume_standard_contracts=True,
                )
            result["meta"].update({
                "user_ticker": ticker.upper(),
                "horizon_input_type": horizon_mode,
                "requested_end_date": end_date.isoformat() if end_date else None,
            })
            st.session_state["hedge_result"] = result
        except Exception as exc:
            st.error(
                "Unable to calculate the stock hedge. Check the ticker, "
                f"LSEG session and entitlements. {type(exc).__name__}: {exc}"
            )
            return

        if etf:
            with st.spinner("Retrieving ETF hedges and preparing graphs..."):
                try:
                    cross = evaluate_cross_hedge(
                        result=result,
                        etf_text=etf,
                        beta_mode=beta_mode,
                        manual_beta=beta,
                        stock_shock=-0.20,
                        etf_shock=-0.10,
                        include_options=include_options,
                    )
                    st.session_state["etf_cross_result"] = cross
                    _auto_calibrate_history(result, cross)
                except Exception as exc:
                    st.session_state["ui_etf_warning"] = (
                        f"ETF comparison unavailable: {type(exc).__name__}: {exc}"
                    )

    result = st.session_state.get("hedge_result")
    if result is None:
        st.info("Choose your stock and press Calculate hedge to see the results.")
        return

    inputs = result["inputs"]
    meta = result["meta"]
    cross = st.session_state.get("etf_cross_result")
    if not isinstance(cross, dict) or cross.get("snapshot") != meta.get("retrieved_at_new_york"):
        cross = None

    st.divider()
    st.subheader(f"{meta.get('user_ticker', inputs.ric)} · {inputs.shares:,} shares")
    a, b, c = st.columns(3)
    a.metric("Reference price", f"{inputs.spot:,.2f}")
    b.metric("Position value (stock quote currency)", f"{inputs.spot * inputs.shares:,.2f}")
    c.metric("Hedge horizon", f"{meta['research_horizon_sessions']} sessions")
    st.caption(
        f"LSEG stock RIC: {inputs.ric} · Source: "
        f"{meta.get('underlying_price_source', 'unknown')} · "
        f"Target: {meta['research_target_date']} · "
        f"Fetched: {meta['retrieved_at_new_york']}"
    )
    if meta.get("used_close_fallback"):
        st.warning("Stock price is a previous close, not a confirmed live quote.")

    # The currency issue is a disclosure, never a blocker or checkbox.
    if cross is not None:
        st.warning(FX_DISCLAIMER)
    if st.session_state.get("ui_etf_warning"):
        st.warning(st.session_state["ui_etf_warning"])
    if st.session_state.get("ui_calibration_warning"):
        st.info(st.session_state["ui_calibration_warning"])

    render_recommendation(result, risk_appetite)

    if not advanced:
        # The ordinary calculator has no diagnostic switches or extra
        # LSEG commands; the graphs are derived from the existing snapshot.
        _render_quick_direct_payoff(result)
        if cross:
            _render_quick_etf_payoffs(result, cross)
        if cross:
            _render_quick_historical_risk(result, cross)
        st.caption(
            "Research-only outputs: no guaranteed loss floor. Quoted option "
            "expiries, transaction costs and different-currency ETFs require "
            "independent verification."
        )
        return

    # All audit-focused calculations and adjustable charts are kept in the
    # separate advanced page; they reuse the same stock and ETF snapshots.
    render_stock_risk(result, risk_appetite)
    render_direct_options(result)
    with st.expander("Stock snapshot and research assumptions"):
        st.json(meta)

    if cross is None:
        st.info("Enter an ETF in the calculator form to compare cross hedges.")
        render_risk_aware_recommendations(result, None)
        return

    render_simple_overview(result, cross)
    render_cross_options(cross)
    render_cross_charts(inputs, meta, cross)
    render_historical_basis_calibration(result, cross)
    render_observed_joint_comparison(result, cross)
    render_block_bootstrap(result, cross)
    render_risk_aware_recommendations(result, cross)


if __name__ == "__main__":
    import sys
    from UI import render_consumer_ui

    render_consumer_ui(sys.modules[__name__])
