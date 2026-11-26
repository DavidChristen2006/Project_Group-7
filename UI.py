"""Provides streamlit user interface.

Input:
    The existing hedging_app module is supplied to render_consumer_ui(). User choices
    come from Streamlit widgets; saved stock, ETF and historical snapshots come from
    st.session_state.

Output:
    An interactive interface for position inputs, stock and ETF scenarios, historical
    results and optional model guidance. Financial calculations and market-data access
    remain in the existing application.
"""
from __future__ import annotations
from html import escape
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from data_pipeline.simple_overview import AVERAGE, RISK, build_strategy_overview


def render_consumer_ui(engine) -> None:
    """Build and display the consumer interface using the existing app engine.

    Args:
    engine (module): The running hedging_app module. It supplies ROOT,
    MAX_HORIZON_SESSIONS, market-data functions, payoff calculations and the
    original advanced-research renderers.

    Returns:
    Displays the interface and runs the selected page.

    Notes:
    Defines nested helpers that share engine, project paths and theme settings.
    Reads and updates Streamlit session state as the user interacts. Opening the
    interface alone does not submit a new market-data request.
    """
    ROOT = Path(engine.ROOT)
    try:
        dark_mode = st.context.theme.type == "dark"
    except (AttributeError, KeyError):
        dark_mode = st.get_option("theme.base") == "dark"
    TEAL, GREY = "#087f76", "#91a4af"
    PERIODS = {"1 month": 21, "3 months": 63, "6 months": 126, "1 year": 252}
    PREFERENCES = {"More protection": "Low", "Balanced": "Balanced", "More upside": "High"}
    NAMES = {
        "Unhedged": "Stock only",
        "Unhedged stock": "Stock only",
        "Protective put": "Downside protection · put",
        "Collar": "Protection with an upside limit · collar",
        "Covered call": "Premium income · covered call",
        "Protective put spread": "Protection within a range · put spread",
        "ETF share cross hedge": "ETF share hedge",
        "Short/long ETF shares": "ETF share hedge",
    }
    EXPLAIN = {
        "Unhedged": "You retain the stock's gains and remain exposed to its losses.",
        "Protective put": "You pay a premium for a put. At expiry, its payout can offset losses below the strike on covered shares. Uncovered shares remain exposed.",
        "Collar": "A put helps limit downside on covered shares. Selling a call helps fund it, while limiting gains above the call strike.",
        "Covered call": "A sold call earns a premium and limits gains. It provides only a small downside cushion; substantial losses remain possible.",
        "Protective put spread": "Buying a put and selling a lower-strike put reduces the premium. Protection is limited to the distance between the strikes.",
    }
    CSS = """
    <style>
    .stApp {background:#f5f7f9;color:#18313e}
    [data-testid="stHeader"] {background:rgba(245,247,249,.96)}
    [data-testid="stSidebar"] {background:white;border-right:1px solid #e1e8ec}
    .block-container {max-width:1320px;padding:2.5rem 3rem 3rem}
    h1,h2,h3 {color:#18313e;letter-spacing:-.045em}
    h1 {font-size:2.7rem!important;font-weight:800!important;line-height:1.15!important}
    h3 {font-size:1.15rem!important}
    p {line-height:1.65}
    .brand {font-size:1.25rem;font-weight:800;letter-spacing:-.04em;margin:.7rem 0 2rem;color:#087f76}
    .brand small {display:block;font-size:.64rem;letter-spacing:.12em;text-transform:uppercase;color:#81949f}
    .eyebrow {font-size:.67rem;font-weight:700;letter-spacing:.15em;text-transform:uppercase;color:#087f76;margin-bottom:.6rem}
    .subtitle {color:#748b98;font-size:1rem;margin-bottom:1.6rem}
    .section {font-weight:700;font-size:.94rem;color:#284b56;margin:.4rem 0 .7rem}
    [data-testid="stForm"] {background:white;border:1px solid #e1e8ec;border-radius:16px;padding:1.3rem 1.5rem}
    [data-testid="stWidgetLabel"] p {color:#536e7c;font-size:.76rem;font-weight:600}
    .stButton>button,.stFormSubmitButton>button {border-radius:9px;min-height:2.6rem;font-weight:600}
    .metric {border:1px solid #e1e8ec;background:white;border-radius:13px;padding:1.05rem 1.2rem;min-height:112px}
    .metric .name {color:#7a8e99;font-size:.65rem;font-weight:600;letter-spacing:.08em;text-transform:uppercase}
    .metric .value {font-size:1.65rem;font-weight:800;letter-spacing:-.045em;margin:.2rem 0;color:#18313e}
    .metric .detail {font-size:.7rem;color:#8296a1}
    .chips {display:flex;gap:.45rem;flex-wrap:wrap;margin:1rem 0 .7rem}
    .chip {padding:.25rem .55rem;background:#e5f2ee;color:#087f76;font-size:.64rem;font-weight:700;border-radius:5px}
    .chip.amber {background:#fff1d6;color:#8c641c}
    .notice {background:#fff7e7;border:1px solid #eee0bd;border-radius:10px;padding:.8rem 1rem;color:#876320;font-size:.78rem;line-height:1.6;margin:.7rem 0}
    .empty {background:#edf3f5;border:1px dashed #cadbe2;border-radius:14px;padding:2rem 1.4rem;text-align:center;margin:1rem 0}
    .empty strong {display:block;font-size:1.05rem;color:#2d525a;margin-bottom:.4rem}
    .empty span {font-size:.82rem;color:#748d9a}
    .outcome {background:#153e3d;color:white;border-radius:15px;padding:1.4rem;margin-top:.35rem}
    .outcome .label {font-size:.65rem;letter-spacing:.1em;text-transform:uppercase;color:#b6d6d0}
    .outcome .number {font-size:2.3rem;font-weight:800;letter-spacing:-.06em;margin:.35rem 0}
    .outcome .context {font-size:.75rem;color:#c2d9d5;line-height:1.6}
    .outcome .compare {border-top:1px solid #ffffff25;margin-top:1rem;padding-top:.8rem;font-size:.76rem;line-height:1.9;color:#c2d9d5}
    .explanation {font-size:.81rem;color:#5d7987;line-height:1.7;margin-top:1.1rem}
    .footer {border-top:1px solid #e1e8ec;font-size:.68rem;color:#8d9da6;margin-top:2rem;padding-top:1rem}
    [data-testid="stExpander"] {border-color:#e1e8ec;border-radius:10px}
    [data-testid="stSidebar"] [role="radiogroup"] label {padding:.5rem .2rem}
    @media(max-width:800px) {.block-container{padding:1.6rem 1rem 2rem}h1{font-size:2rem!important}}
    </style>
    """

    if dark_mode:
        replacements = {
            "#f5f7f9": "#0e151c", "#18313e": "#e4edf2",
            "background:white": "background:#151f29",
            "#e1e8ec": "#2e3d49", "#81949f": "#a8b9c5",
            "#748b98": "#a8bdc9", "#284b56": "#bdd8df",
            "#536e7c": "#b6c8d2", "#7a8e99": "#adc0cc",
            "#8296a1": "#a7bac6", "#e5f2ee": "#153e38",
            "#087f76": "#65cabb", "#fff1d6": "#302719",
            "#8c641c": "#e1c184", "#fff7e7": "#2c261b",
            "#eee0bd": "#635534", "#876320": "#e2c78d",
            "#edf3f5": "#16232c", "#cadbe2": "#3b515e",
            "#2d525a": "#c0dfe4", "#748d9a": "#a6bcc8",
            "#5d7987": "#b0c5cf", "#8d9da6": "#a0b2be",
            "rgba(245,247,249,.96)": "rgba(14,21,28,.96)",
        }
        for before, after in replacements.items():
            CSS = CSS.replace(before, after)
        TEAL = "#65cabb"

    def html(value: str) -> None:
        """Display application-controlled HTML or CSS in the current container.

        Args:
        value (str): Markup to render. Callers must escape dynamic text with
        escape() before inserting it; this helper does not sanitize the supplied
        markup.

        Returns:
        Inserts markup through st.markdown() with HTML enabled.
        """
        st.markdown(value, unsafe_allow_html=True)

    def name(value: str) -> str:
        """Convert an internal strategy identifier into a readable display label.

        Args:
        value (str): A strategy identifier used by the calculation engine.

        Returns:
        str: Its label from NAMES, or the original text with underscores replaced by
        spaces when no predefined label exists.

        Notes:
        Only displayed text changes; the engine's strategy identifiers stay the same.
        """
        return NAMES.get(value, value.replace("_", " "))

    def money(value: float, signed: bool = False) -> str:
        """Format an amount without assuming a currency or converting its value.

        Args:
            value (float): Amount in the relevant instrument's quote units.
            signed (bool): If True, include a plus or minus sign. Defaults to False,
                which still shows a minus sign for negative amounts.

        Returns:
            str: Formatted amount with comma thousands separators and zero decimal
            places.

        Notes:
            Rounding affects display only; it does not change stored calculation values.
        """
        return f"{float(value):+,.0f}" if signed else f"{float(value):,.0f}"

    def notice(message: str) -> None:
        """Display a concise limitation or data-quality message.

        Args:
            message (str): Plain text explaining a relevant limitation.

        Returns:
            None: Shows the escaped message in a styled notice box.
        """
        html(f'<div class="notice">{escape(message)}</div>')

    def empty(title: str, message: str) -> None:
        """Explain why a panel has no data available to display.

        Args:
            title (str): Short heading describing the missing-data state.
            message (str): Explanation or suggested next step for the user.

        Returns:
            None: Displays an escaped heading and explanation in a placeholder box.

        Notes:
            Does not substitute zero values or generate artificial results.
        """
        html(f'<div class="empty"><strong>{escape(title)}</strong><span>{escape(message)}</span></div>')

    def debug_error(exc: object) -> None:
        """Show error information inside an expandable technical-details panel.

        Args:
            exc (object): Usually a caught exception. Its type name and string
                representation help diagnose the failed action.

        Returns:
            None: Adds an expander containing the error type and message.

        Notes:
            Does not raise, fix or retry the error, and does not display a traceback.
        """
        with st.expander("Technical details"):
            st.code(f"{type(exc).__name__}: {exc}", language=None)

    def have_hedge_models() -> bool:
        """Check whether either supported hedge-model manifest exists locally.

        Args:
            None. Uses the enclosing configuration or Streamlit state as described below.

        Returns:
            bool: True if manifest.json exists in either the risk_aware_hedge_final or
            six_horizon_final experiment folder; otherwise False.

        Notes:
            Uses the enclosing ROOT/data/processed/ml_experiments path. A manifest
            indicates possible availability, but this check does not load a model,
            validate all its files or guarantee successful inference.
        """
        return any((ROOT / f"data/processed/ml_experiments/{folder}/manifest.json").is_file()
                   for folder in ("risk_aware_hedge_final", "six_horizon_final"))

    def matching_cross(result: dict | None) -> dict | None:
        """Retrieve the saved ETF comparison only when its stock snapshot matches.

        Args:
            result (dict | None): Current stock snapshot with
                meta['retrieved_at_new_york'], or None before a position has been
                loaded.

        Returns:
            dict | None: Saved etf_cross_result when its snapshot marker matches the
            stock retrieval marker; None when absent or mismatched.

        Notes:
            Reads session state without changing it. This association check does not
            verify synchronized market quotes or currency parity.
        """
        cross = st.session_state.get("etf_cross_result")
        if result and isinstance(cross, dict) and cross.get("snapshot") == result["meta"].get("retrieved_at_new_york"):
            return cross
        return None

    def chart_style(fig: go.Figure, height: int = 370) -> go.Figure:
        """Apply the interface's current light or dark appearance to a Plotly chart.

        Args:
            fig (go.Figure): Existing figure to style.
            height (int): Chart height in pixels. Defaults to 370.

        Returns:
            go.Figure: The same figure with updated theme, axes, margins, legend and
            hover-label formatting.

        Notes:
            Modifies fig in place without changing plotted data values. Individual views
            may override the default percentage suffix.
        """
        fig.update_layout(
            template="plotly_dark" if dark_mode else "plotly_white", height=height,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Arial, sans-serif", color="#b7c9d2" if dark_mode else "#617d8b", size=11),
            margin=dict(l=10,r=10,t=18,b=10), hovermode="x unified",
            legend=dict(orientation="h",x=0,y=-.2,font=dict(size=10)),
            hoverlabel=dict(bgcolor="#1a2a35" if dark_mode else "white",font_color="#e4edf2" if dark_mode else "#18313e"),
        )
        fig.update_xaxes(gridcolor="#2c3b46" if dark_mode else "#e6edf1",zerolinecolor="#3d525f" if dark_mode else "#d3dfe5",ticksuffix="%")
        fig.update_yaxes(gridcolor="#2c3b46" if dark_mode else "#e6edf1",zerolinecolor="#3d525f" if dark_mode else "#cbd9df",tickformat=",.0f")
        return fig

    def stock_curves(result: dict) -> pd.DataFrame:
        """Prepare stock-only or stock-option profit/loss curves for a snapshot.

        Args:
        result (dict): Stock snapshot containing inputs and has_direct_options.
        Optional liquid_options holds contracts used to select a lower-strike
        put for a put spread.

        Returns:
        pd.DataFrame: An 'Underlying move (%)' column and one profit/loss column per
        available strategy, expressed in stock quote units.

        Notes:
        Without direct options, builds a stock-only baseline from -100% to +100%.
        Otherwise delegates option selection and expiration payoffs to the existing
        engine. This helper does not request new market data.
        """
        inputs = result["inputs"]
        if not result.get("has_direct_options"):
            x = np.linspace(-100.,100.,401)
            return pd.DataFrame({"Underlying move (%)":x,"Unhedged":inputs.shares*inputs.spot*x/100.})
        frame = result.get("liquid_options")
        lower = None
        if isinstance(frame,pd.DataFrame) and not frame.empty:
            lower = engine.select_lower_put(
                frame,long_put_strike=inputs.put_strike,expiry=inputs.expiry,
                spot=inputs.spot,max_spread_fraction=1.0,
            )
        return engine.direct_expiration_curves(
            inputs,upper_pct=100.,
            lower_put_strike=float(lower["StrikePrice"]) if lower is not None else None,
            lower_put_bid=float(lower["BID"]) if lower is not None else None,
        )

    def values_at(curves: pd.DataFrame, axis: str, move: float) -> dict:
        """Find each strategy's profit/loss at one selected price change.

        Args:
        curves (pd.DataFrame): Numerical scenario table with a move column and
        profit/loss columns for the strategies.
        axis (str): Name of the move column, ordered in increasing order.
        move (float): Selected change in percentage points: -20 means a 20% fall,
        rather than the decimal fraction -0.20.

        Returns:
        dict: Mapping from each strategy column name to its interpolated profit/loss
        as a float. The move column is excluded.

        Notes:
        Uses linear interpolation between existing curve points, including the
        engine's strike knots. Values outside the grid use the nearest endpoint
        rather than extrapolation.
        """
        return {column:float(np.interp(move,curves[axis],curves[column]))
                for column in curves if column != axis}

    def payoff_plot(curves: pd.DataFrame,axis: str,chosen: str,move: float,full: bool=False) -> go.Figure:
        """Build a chart comparing one strategy with holding the stock alone.

        Args:
        curves (pd.DataFrame): Scenario curves including the relevant unhedged
        baseline.
        axis (str): Name of the stock-move or ETF-move percentage column.
        chosen (str): Existing strategy column to highlight.
        move (float): Change in percentage points at the selected scenario marker.
        full (bool): If True, display the full -100% to +100% horizontal range.
        Otherwise use a narrower range containing the selected move. Defaults to
        False.

        Returns:
        go.Figure: Styled payoff lines and a marker at the selected outcome.

        Notes:
        Returns the figure for its caller to display. Does not alter curve values or
        present the scenario as a forecast.
        """
        base = "Unhedged" if "Unhedged" in curves else "Unhedged stock"
        fig = go.Figure()
        for strategy in dict.fromkeys([base,chosen]):
            active = strategy == chosen and chosen != base
            fig.add_trace(go.Scatter(
                x=curves[axis],y=curves[strategy],name=name(strategy),mode="lines",
                line=dict(color=TEAL if active else GREY,width=3 if active else 2,
                          dash="solid" if active else "dot"),
                hovertemplate="%{y:,.0f} quote units<extra>"+escape(name(strategy))+"</extra>",
            ))
        point = values_at(curves,axis,move)[chosen]
        fig.add_vline(x=move,line_dash="dash",line_color="#a6babd",line_width=1)
        fig.add_trace(go.Scatter(x=[move],y=[point],mode="markers",
            marker=dict(size=10,color=TEAL,line=dict(color="white",width=2)),
            showlegend=False,hoverinfo="skip"))
        chart_style(fig)
        bounds = [-100,100] if full else [min(-40,move-5),max(40,move+5)]
        fig.update_xaxes(title="Stock price change" if axis.startswith("Underlying") else "ETF price change",range=bounds)
        fig.update_yaxes(title="Profit / loss · quote units")
        return fig

    def outcome(selected: float,base: float,context: str) -> None:
        """Display a selected scenario beside its stock-only comparison.

        Args:
        selected (float): Chosen strategy's hypothetical profit/loss.
        base (float): Stock-only profit/loss for the same scenario.
        context (str): Description of the assumed stock and/or ETF price changes.

        Returns:
        None: Displays selected and base amounts plus their difference, selected base.

        Notes:
        Inputs must already use a comparable valuation basis. The function formats
        quote-unit amounts without performing currency conversion.
        """
        html('<div class="outcome"><div class="label">Selected scenario</div>'
             f'<div class="number">{money(selected,True)}</div>'
             f'<div class="context">{escape(context)}<br>Quote units · hypothetical outcome</div>'
             '<div class="compare">'
             f'Stock only <b style="float:right;color:white">{money(base,True)}</b><br>'
             f'Difference <b style="float:right;color:#b7e1d5">{money(selected-base,True)}</b>'
             '</div></div>')

    def navigation(result: dict | None) -> str:
        """Display sidebar navigation, data status and short term explanations.

        Args:
        result (dict | None): Current stock snapshot, or None before a position is
        loaded. Its metadata provides quote-status information.

        Returns:
        str: Selected page: 'Your plan', 'Advanced tools' or 'Research library'.

        Notes:
        Streamlit stores the page selection under friendly_page. The model-status
        label reflects manifest detection, not proof of successful model inference.
        """
        with st.sidebar:
            html('<div class="brand">↗ Equity Hedge<small>Explore. Compare. Understand.</small></div>')
            page = st.radio("Workspace",["Your plan","Advanced tools","Research library"],
                            label_visibility="collapsed",key="friendly_page")
            st.divider()
            st.caption("DATA STATUS")
            st.write("Stock price: "+("not loaded" if result is None else
                     "closing reference" if result["meta"].get("used_close_fallback") else "latest available"))
            st.write("Stock options: "+("not loaded" if result is None else
                     "available" if result.get("has_direct_options") else "unavailable"))
            st.write("ML guidance: "+("files detected" if have_hedge_models() else "files not loaded"))
            with st.expander("What is a hedge?"):
                st.write("A hedge adds a position intended to offset some losses. It can cost money, limit gains, and introduce new risks.")
            with st.expander("Explain the terms"):
                st.markdown("**Put:** an option that can pay out when prices fall.\n\n"
                    "**Collar:** a put combined with a sold call; it limits downside and upside on covered shares.\n\n"
                    "**ETF:** a fund holding a basket of assets. A cross hedge uses a related fund rather than the stock itself.\n\n"
                    "**Tail loss / CVaR:** the average loss in the worst historical outcomes at the stated threshold.")
            st.caption("Academic research. No orders are placed.")
        return page

    def calculate(ticker,shares,sessions,etf,preference,spread,fee,beta_mode,beta,include_options) -> None:
        """Load a stock snapshot and an optional ETF comparison after form submission.

        Args:
        ticker (str): Stock ticker or LSEG instrument code. An empty value displays a message without requesting data.
        shares (int): Number of stock shares in the user's position.
        sessions (int): Planning horizon measured in trading sessions.
        etf (str): Optional fund ticker or instrument code. An empty string skips the ETF comparison.
        preference (str): Internal optional-model preference, such as 'Low','Balanced' or 'High', saved for subsequent model guidance.
        spread (float): Maximum option spread as a decimal fraction: 0.10 represents 10%.
        fee (float): Opening fee per option contract in quote units.
        beta_mode (str): 'Estimate from market data' selects automatic stock-to-ETF sensitivity; the manual selection uses beta instead.
        beta (float): Manual stock-to-ETF sensitivity assumption.
        include_options (bool): Whether to request eligible ETF options.

        Returns:
        None: Stores the stock snapshot as hedge_result and, when available, the ETF comparison as etf_cross_result in st.session_state.

        Notes:
            For a nonempty ticker, clears previous snapshots and error messages, then
            delegates retrieval to the existing engine. May save historical calibration
            as empirical_basis_result when the ETF result has a beta report. Caught
            errors are saved as friendly_error, friendly_etf_error or
            friendly_history_error so the UI can explain partial failures. Does not
            train models or execute trades.
        """
        if not ticker:
            notice("Enter a stock ticker to start.")
            return
        for key in ("hedge_result","etf_cross_result","empirical_basis_result",
                    "friendly_error","friendly_etf_error","friendly_history_error"):
            st.session_state.pop(key,None)
        try:
            with st.spinner("Getting your stock snapshot and eligible option prices…"):
                result = engine.get_snapshot(
                    ric=engine.resolve_lseg_ric(ticker),shares=shares,sessions=sessions,
                    spread=spread,fee=fee,assume_standard_contracts=True)
            result["meta"].update(user_ticker=ticker.upper(),horizon_input_type="Trading sessions",requested_end_date=None)
            st.session_state["hedge_result"] = result
            st.session_state["friendly_preference"] = preference
        except Exception as exc:
            st.session_state["friendly_error"] = exc
            return
        if etf:
            try:
                with st.spinner("Preparing the fund comparison…"):
                    cross = engine.evaluate_cross_hedge(
                        result=result,etf_text=etf,
                        beta_mode="Automatic - 252 sessions" if beta_mode == "Estimate from market data" else "Manual beta assumption",
                        manual_beta=beta,stock_shock=-.2,etf_shock=-.1,include_options=include_options)
                st.session_state["etf_cross_result"] = cross
                if isinstance(cross.get("beta_report"),dict):
                    from data_pipeline.quick_calibration import fetch_quick_calibration
                    try:
                        with st.spinner("Checking the historical comparison…"):
                            st.session_state["empirical_basis_result"] = fetch_quick_calibration(result,cross,lookback_years=8)
                    except Exception as exc:
                        st.session_state["friendly_history_error"] = exc
            except Exception as exc:
                st.session_state["friendly_etf_error"] = exc

    def position_form() -> None:
        """Collect position details and comparison assumptions in one submitted form.

        Args:
        Uses the enclosing configuration or Streamlit state as described below.

        Returns:
        None: Displays the form and calls calculate() only when the user submits it.

        Notes:
        Inputs come from widgets for ticker, shares, horizon, ETF, preference,
        spread, fees, beta and ETF-option eligibility. Converts the spread
        percentage to a decimal fraction and maps the preference label to the engine
        value. A nonzero custom horizon overrides the preset period. Typing or
        changing result sliders does not itself submit the form or retrieve quotes.
        """
        with st.form("friendly_position"):
            html('<div class="section">01 · Your investment</div>')
            a,b,c = st.columns([1.3,1,1.2])
            ticker = a.text_input("Which stock do you own?",value="NVDA",help="Stock ticker or full LSEG instrument code.").strip()
            shares = int(b.number_input("Number of shares",min_value=1,value=100,step=1))
            period = c.selectbox("How long are you planning for?",list(PERIODS),index=1,
                                help="Approximate months based on NYSE trading days, not calendar months.")
            html('<div class="section" style="margin-top:.6rem">02 · Your comparison</div>')
            a,b = st.columns([1.3,2])
            etf = a.text_input("Market fund to compare · optional",value="SPY",
                              help="Leave blank to explore the stock alone.").strip()
            preference = b.select_slider("What matters most to you?",options=list(PREFERENCES),value="Balanced",
                              help="Used by optional ML policies. It does not change your manual scenarios or guarantee protection.")
            with st.expander("Fine-tune the assumptions"):
                a,b = st.columns(2)
                custom = int(a.number_input("Custom trading days · 0 uses the selected period",0,engine.MAX_HORIZON_SESSIONS,0))
                spread = int(b.slider("Maximum option spread (%)",1,30,10))
                fee = float(a.number_input("Fee per option contract",min_value=0.,value=0.,step=.5))
                beta_mode = b.selectbox("Stock–fund sensitivity",["Estimate from market data","Use a manual assumption"])
                beta = float(a.number_input("Manual sensitivity (beta)",-5.,5.,1.,step=.1,
                                           help="Used only with the manual sensitivity assumption."))
                include_options = b.checkbox("Include ETF options when eligible",value=True)
            submitted = st.form_submit_button("Explore protection  →",type="primary",width="stretch")
        if submitted:
            calculate(ticker,shares,custom or PERIODS[period],etf,PREFERENCES[preference],
                      spread/100.,fee,beta_mode,beta,include_options)

    def position_summary(result: dict) -> None:
        """Display position value, reference price, horizon and quote availability.

        Args:
        result (dict): Stock snapshot with inputs such as spot, shares and ric, plus
        metadata describing the horizon, source and closing-price fallback.

        Returns:
        None: Displays three summary cards, data-status labels and any relevant
        closing-price limitation.

        Notes:
        Position value is spot multiplied by shares. Amounts remain in stock quote
        units; a previous close is not described as a live price.
        """
        inputs,meta = result["inputs"],result["meta"]
        st.markdown("### Your position, at a glance")
        cards = [
            ("Stock position",money(inputs.spot*inputs.shares),
             f"{inputs.shares:,} shares · {meta.get('user_ticker',inputs.ric)}"),
            ("Reference price",f"{inputs.spot:,.2f}",
             "Previous close" if meta.get("used_close_fallback") else "Latest available; may be delayed"),
            ("Planning horizon",f"{meta['research_horizon_sessions']} days",
             f"Target · {meta.get('research_target_date','not supplied')}"),
        ]
        for col,(title,value,detail) in zip(st.columns(3),cards):
            with col:
                html(f'<div class="metric"><div class="name">{escape(title)}</div>'
                     f'<div class="value">{escape(value)}</div><div class="detail">{escape(detail)}</div></div>')
        source = "CLOSING REFERENCE" if meta.get("used_close_fallback") else "LATEST AVAILABLE"
        options = "STOCK OPTIONS AVAILABLE" if result.get("has_direct_options") else "STOCK OPTIONS UNAVAILABLE"
        html(f'<div class="chips"><span class="chip amber">{source}</span><span class="chip">{options}</span></div>')
        st.caption("Amounts use the stock's quote units. No currency symbol is assumed.")
        if meta.get("used_close_fallback"):
            notice("This snapshot uses a closing price. Stock-option comparisons are paused until eligible, comparable quotes are available.")

    def stock_view(result: dict) -> None:
        """Display an interactive stock-protection scenario for the saved position.

        Args:
        result (dict): Current stock snapshot used to build eligible payoff curves
        and explain option expiry, coverage and configured fees.

        Returns:
        None: Displays strategy selection, a price-change slider, payoff chart, outcome comparison and an optional scenario CSV download.

        Notes:
        Uses the saved snapshot without retrieving fresh quotes. If options are
        unavailable, offers only the stock baseline. The CSV contains the selected
        scenario amounts rather than the full payoff grid.
        """
        curves = stock_curves(result)
        names = [column for column in curves if column != "Underlying move (%)"]
        left,right = st.columns([2.4,1],gap="large")
        with left:
            st.markdown("### What if your stock price changes?")
            chosen = st.selectbox("Protection to explore",names,
                index=names.index("Protective put") if "Protective put" in names else 0,
                format_func=name,key="friendly_stock_strategy")
            move = st.slider("Stock price change (%)",-100,100,-20,key="friendly_stock_move")
            full = st.checkbox("Show full −100% to +100% chart range",key="friendly_full")
            st.plotly_chart(payoff_plot(curves,"Underlying move (%)",chosen,move,full),
                            width="stretch",theme=None,key="friendly_stock_chart",config={"displayModeBar":False})
        points = values_at(curves,"Underlying move (%)",move)
        with right:
            outcome(points[chosen],points["Unhedged"],f"If the stock moves {move:+d}%")
            html(f'<div class="explanation">{escape(EXPLAIN.get(chosen,"Compare this scenario with holding the stock alone."))}</div>')
        if result.get("has_direct_options"):
            inputs = result["inputs"]
            st.caption(f"Options are valued at expiry: {inputs.expiry}. Planning target: {result['meta']['research_target_date']}. These dates can differ.")
            st.caption(f"{inputs.covered_shares:,} of {inputs.shares:,} shares covered. Premiums and configured opening fees are included. Tax, financing and dividends are omitted.")
        else:
            notice("Only the stock-only baseline is available for this snapshot.")
        st.caption("Hypothetical price changes, not forecasts or probabilities.")
        with st.expander("See the scenario numbers"):
            frame = pd.DataFrame({"Strategy":[name(key) for key in points],"Scenario profit / loss":list(points.values())})
            st.dataframe(frame,hide_index=True,width="stretch")
            st.download_button("Download scenario CSV",frame.to_csv(index=False),"stock_scenario.csv","text/csv")

    def etf_view(result: dict,cross: dict | None) -> None:
        """Display ETF hedge scenarios with separate stock and ETF price assumptions.

        Args:
        result (dict): Stock snapshot containing position size and stock spot price.
        cross (dict | None): Matching ETF snapshot with etf_ric, etf_spot, beta and all_scenarios, plus any eligible option structures and scenario tables.
        None displays a missing-comparison message.

        Returns:
        None: Displays hedge-size and price-move controls, payoff charts, outcome amounts and an expandable joint-move heatmap.

        Notes:
        Delegates payoffs and heatmap data to the existing engine. Hedge size is a
        fraction of beta-based exposure, not guaranteed loss protection. No currency
        conversion is performed, so combined amounts remain illustrative when the
        currencies are unverified.
        """
        if cross is None:
            empty("No fund comparison yet","Enter an ETF and calculate again to compare a fund-based hedge.")
            if st.session_state.get("friendly_etf_error"):
                debug_error(st.session_state["friendly_etf_error"])
            return
        notice("Currency match is unverified. Combined amounts and rankings are illustrative: the engine does not convert currencies or model FX changes.")
        st.markdown("### Explore a market-fund hedge")
        st.caption(f"{cross['etf_ric']} · reference {cross['etf_spot']:,.2f} · sensitivity (beta) {cross['beta']:.2f}")
        fractions = sorted(pd.to_numeric(cross["all_scenarios"]["Hedge Fraction"],errors="raise").unique().tolist())
        if not fractions:
            empty("No saved hedge positions","The ETF result did not contain valid share-hedge sizes.")
            return
        fraction = st.select_slider("Modelled hedge size",fractions,
            value=.5 if .5 in fractions else fractions[0],format_func=lambda value:f"{value:.0%}",
            help="A fraction of beta-based exposure, not a guaranteed percentage of loss protection.")
        a,b = st.columns(2)
        stock_move = a.slider("Assume the stock changes by (%)",-100,100,-20,key="friendly_etf_stock")
        etf_move = b.slider("Assume the ETF changes by (%)",-100,100,-10,key="friendly_etf_move")
        inputs = result["inputs"]
        kwargs = dict(stock_spot=inputs.spot,stock_shares=inputs.shares,etf_spot=cross["etf_spot"],
            hedge_fraction=float(fraction),share_stresses=cross["all_scenarios"],
            option_structures=cross.get("option_structures",{}),
            option_stresses=cross.get("option_scenarios"),fee_per_contract=cross.get("option_fee_per_contract",0.))
        curves = engine.cross_hedge_expiration_curves(
            **kwargs,stock_move_pct=float(stock_move),upper_etf_pct=100.)
        names = [column for column in curves if column != "ETF move (%)"]
        left,right = st.columns([2.4,1],gap="large")
        with left:
            chosen = st.selectbox("Fund-based strategy",names,
                index=names.index("ETF share cross hedge") if "ETF share cross hedge" in names else 0,
                format_func=name,key="friendly_etf_strategy")
            st.caption(f"The stock stays fixed at {stock_move:+d}% on this chart; only the ETF move changes.")
            st.plotly_chart(payoff_plot(curves,"ETF move (%)",chosen,etf_move),
                            width="stretch",theme=None,key="friendly_etf_chart",config={"displayModeBar":False})
        points = values_at(curves,"ETF move (%)",etf_move)
        with right:
            outcome(points[chosen],points["Unhedged stock"],f"Stock {stock_move:+d}% · ETF {etf_move:+d}% · illustrative")
            html('<div class="explanation">A fund can move differently from your stock. A hedge can reduce a loss in one scenario and increase it in another. Short positions require borrowing.</div>')
        if cross.get("option_issue"):
            notice("ETF options are unavailable for this snapshot. The ETF-share comparison remains available.")
        with st.expander("Explore stock and ETF moves together"):
            surface = engine.build_joint_payoff_surface(**kwargs,stock_max_pct=100.,etf_max_pct=100.)
            strategies = list(surface["payoffs"])
            strategy = st.selectbox("Heatmap strategy",strategies,
                index=1 if len(strategies)>1 else 0,format_func=name,key="friendly_heat_strategy")
            mode = st.radio("Map value",["Profit / loss","Difference from stock only"],horizontal=True,key="friendly_heat_mode")
            fig = engine.make_joint_heatmap(
                surface,strategy,mode="Total P&L" if mode=="Profit / loss" else "Benefit versus unhedged",
                stock_cursor=float(stock_move),etf_cursor=float(etf_move))
            chart_style(fig,430)
            fig.update_layout(hovermode="closest")
            fig.update_xaxes(range=[-100,100])
            fig.update_yaxes(range=[-100,100],ticksuffix="%",tickformat=".0f")
            fig.data[0].update(colorscale=[[0,"#bd5c64"],[.5,"#fafcfb"],[1,TEAL]])
            st.plotly_chart(fig,width="stretch",theme=None,key="friendly_heatmap",config={"displayModeBar":False})
            st.caption("Each square is a hypothetical pair of moves, not a probability. ETF options are valued at their own expiry. Borrowing, financing, FX effects and tax are omitted.")

    def history_view(result: dict,cross: dict | None) -> None:
        """Compare strategies using available matched historical observations.

        Args:
        result (dict): Current stock snapshot.
        cross (dict | None): Matching ETF comparison, if available.

        Returns:
        None: Displays historical tail-loss and average-outcome charts and an expandable table, or explains that matching history is unavailable.

        Notes:
        Also reads empirical_basis_result from session state.
        build_strategy_overview() supplies the comparison table. Tail loss describes
        the worst 5% of observed outcomes. Historical results are descriptive
        evidence, not new forecasts, and missing observations are not replaced with fabricated scores.
        """
        table,ready = build_strategy_overview(result,cross,st.session_state.get("empirical_basis_result"))
        if not ready:
            empty("No matching history available","This view needs paired stock–ETF observations for your horizon. Scenario charts still work.")
            if st.session_state.get("friendly_history_error"):
                debug_error(st.session_state["friendly_history_error"])
            return
        notice("These are descriptive historical comparisons, not forecasts. Currency match is unverified; FX conversion and short-ETF costs are omitted.")
        st.markdown("### How did strategies compare historically?")
        st.caption("Severe-tail loss means average loss in the worst 5% of observed historical outcomes. Lower values indicate smaller losses in this sample.")
        view = table.sort_values(RISK).copy()
        labels = view["Strategy"].str.replace("No Hedge","Stock only",regex=False)
        a,b = st.columns([1.15,1])
        with a:
            fig = go.Figure(go.Bar(x=view[RISK],y=labels,orientation="h",
                marker_color=[GREY if item=="No Hedge" else TEAL for item in view["Strategy"]],
                hovertemplate="%{y}<br>Historical tail loss: %{x:,.0f}<extra></extra>"))
            chart_style(fig,340)
            fig.update_xaxes(title="Historical severe-tail loss · quote units",ticksuffix="",tickformat=",.0f")
            fig.update_yaxes(autorange="reversed",tickformat=None)
            st.plotly_chart(fig,width="stretch",theme=None,key="friendly_history_loss",config={"displayModeBar":False})
        with b:
            fig = go.Figure(go.Scatter(x=view[RISK],y=view[AVERAGE],mode="markers",text=labels,
                marker=dict(size=13,color=TEAL),
                hovertemplate="%{text}<br>Tail loss: %{x:,.0f}<br>Average outcome: %{y:,.0f}<extra></extra>"))
            chart_style(fig,340)
            fig.update_xaxes(title="Historical tail loss · lower is better",ticksuffix="",tickformat=",.0f")
            fig.update_yaxes(title="Average historical profit / loss")
            st.plotly_chart(fig,width="stretch",theme=None,key="friendly_history_return",config={"displayModeBar":False})
        st.caption("Historical windows overlap. Few independent observations can make tail estimates unstable. Historical results are not a current recommendation.")
        with st.expander("Compare exact historical values"):
            st.dataframe(table,hide_index=True,width="stretch")

    def model_view(result: dict) -> None:
        """Offer optional model guidance when the required inputs are available.

        Args:
        result (dict): Stock snapshot containing eligible direct-option candidates.
        The saved model preference is read from session state.

        Returns:
        None: Explains missing models or options, or shows a button that delegates
        recommendation rendering to the existing engine.

        Notes:
        Runs the recommendation renderer only when its button is pressed. Manifest
        detection does not validate every model file. This function does not train,
        download or repair models.
        """
        st.markdown("### Optional model guidance")
        if not have_hedge_models():
            empty("Model guidance is not loaded","Add the matching exported model files to enable it. Stock and ETF scenarios remain available.")
            with st.expander("How to enable it"):
                st.write("Copy the project's matching model exports, manifests and research outputs into their original folders under data/processed/. Installing a Python package does not install the trained models.")
            return
        if not result.get("has_direct_options"):
            notice("This option-based model comparison needs eligible stock-option candidates, unavailable for this snapshot.")
            return
        st.caption("Files were detected; successful evaluation still depends on compatible models and inputs.")
        if st.button("Load model guidance",key="friendly_run_ml"):
            engine.render_recommendation(result,st.session_state.get("friendly_preference","Balanced"))

    def diagnostics(result: dict,cross: dict | None) -> None:
        """Make snapshot metadata, assumptions and saved errors available on demand.

        Args:
        result (dict): Stock snapshot whose meta dictionary describes retrieval and
        calculation assumptions.
        cross (dict | None): Optional ETF snapshot with source information and any
        option-availability issue.

        Returns:
        None: Displays an expander with limitations, stock metadata, ETF source
        details and saved ETF or historical error messages.

        Notes:
        Reads existing data and errors without retrying failed requests.
        """
        with st.expander("Data quality, assumptions & technical details"):
            st.write("Quotes may be delayed. Scenarios are hypothetical, not probabilities. Option expiry may differ from the planning horizon. FX conversion, tax, financing, dividends and all short-position costs are not modelled.")
            st.json(result["meta"])
            if cross:
                st.write("ETF source:",cross.get("etf_price_source","Not supplied"))
                if cross.get("option_issue"):
                    st.code(str(cross["option_issue"]),language=None)
            for key in ("friendly_etf_error","friendly_history_error"):
                if st.session_state.get(key):
                    exc=st.session_state[key]
                    st.code(f"{type(exc).__name__}: {exc}",language=None)

    def advanced_view(result: dict | None,cross: dict | None) -> None:
        """Open the original app's detailed tools from a simpler navigation panel.

        Args:
        result (dict | None): Current stock snapshot. None prompts the user to
        calculate a position first.
        cross (dict | None): Matching ETF snapshot, required by the detailed ETF,
        calibration and joint-comparison panels.

        Returns:
        None: Displays a tool selector and delegates the selected analysis to an
        existing engine renderer or the optional-model guidance panel.

        Notes:
        Preserves access to contract details, forecasts, calibration, observed
        outcomes, bootstrap simulation and risk-aware research. The selected
        original tool handles its own additional data and model requirements.
        """
        html('<div class="eyebrow">Advanced tools</div>')
        st.title("Go one level deeper.")
        html('<div class="subtitle">Contracts, assumptions and research methods, when you need them.</div>')
        if result is None:
            empty("Start with your position","Calculate a position on Your plan first.")
            return
        choice = st.selectbox("Analysis to open",[
            "Stock option contracts","ETF option structures","Historical calibration",
            "Observed outcomes","Bootstrap simulation","Risk-aware research",
            "Stock risk forecasts","Model guidance"])
        if choice=="Stock option contracts":
            engine.render_direct_options(result)
        elif choice=="Model guidance":
            model_view(result)
        elif choice=="Stock risk forecasts":
            # Its model family is separate from the option-hedge models.
            # Call the existing renderer, which handles its own files.
            engine.render_stock_risk(result,st.session_state.get("friendly_preference","Balanced"))
        elif cross is None:
            empty("This view needs a fund comparison","Add an ETF on Your plan and calculate again.")
        elif choice=="ETF option structures":
            engine.render_cross_options(cross)
        elif choice=="Historical calibration":
            engine.render_historical_basis_calibration(result,cross)
        elif choice=="Observed outcomes":
            engine.render_observed_joint_comparison(result,cross)
        elif choice=="Bootstrap simulation":
            engine.render_block_bootstrap(result,cross)
        else:
            engine.render_risk_aware_recommendations(result,cross)

    def research_view() -> None:
        """Display saved project forecasts, validation and research results.

        Args:
        Uses the enclosing configuration or Streamlit state as described below.

        Returns:
        None: Displays the selected research panel or explains that the required
        folder or validation results have not been supplied.

        Notes:
        Uses ROOT/data/processed and the research category chosen in a widget.
        Delegates to existing historical-research renderers. Saved forecasts and
        metrics are not live quotes. Does not train models or create missing
        research outputs.
        """
        html('<div class="eyebrow">Research library</div>')
        st.title("Evidence behind the models.")
        html('<div class="subtitle">Recorded forecasts, validation and past hedge outcomes.</div>')
        st.caption("These are saved historical results. Opening this page does not retrain models or retrieve current prices.")
        processed=ROOT/"data/processed"
        if not processed.is_dir():
            empty("Saved research has not been added","Copy matching project forecasts and evaluation outputs into data/processed/.")
            return
        choice=st.selectbox("Research to explore",["Summary","Stock forecasts","Model accuracy","Experiments","Hedge outcomes","Decision audit"])
        if choice=="Summary":
            from data_pipeline.historical_research_overview import render_historical_research_overview
            render_historical_research_overview(ROOT)
        elif choice=="Stock forecasts":
            engine.render_historical_ml_view()
        elif choice=="Model accuracy":
            folder=processed/"ml_baselines/equity_63d_suite_v1/full"
            if folder.is_dir():
                engine.render_ml_comparison(folder)
            else:
                empty("Validation results are not loaded","Add the saved validation metrics to enable model comparisons.")
        elif choice=="Experiments":
            engine.render_project_ml_results(ROOT)
        elif choice=="Hedge outcomes":
            engine.render_policy_benchmarks(ROOT)
        else:
            engine.render_ml_decision_audit(ROOT)

    def main() -> None:
        """Configure the page and route the user's selected interface view.

        Args:
        Uses the enclosing configuration or Streamlit state as described below.

        Returns:
        None: Displays the chosen page, position results, diagnostics and footer.
        Stops with an upgrade message for Streamlit below 1.50.

        Notes:
        Uses enclosing theme settings and saved Streamlit state. Calls the position
        form on Your plan and refreshes the local snapshot after submission.
        matching_cross() associates ETF results with that snapshot. Catches panel-
        rendering failures and displays a readable message with expandable technical
        information.
        """
        st.set_page_config(page_title="Equity Hedge · Explore your protection",page_icon="↗",       # change ????
                           layout="wide",initial_sidebar_state="expanded")
        if tuple(int(part) for part in st.__version__.split(".")[:2]) < (1,50):
            st.error("This interface needs Streamlit 1.50 or newer.")
            st.code('py -m pip install "streamlit>=1.50,<2"')
            st.stop()
        html(CSS)
        result=st.session_state.get("hedge_result")
        page=navigation(result)
        try:
            if page=="Research library":
                research_view()
            elif page=="Advanced tools":
                advanced_view(result,matching_cross(result))
            else:
                html('<div class="eyebrow">Stock protection, made clearer</div>')
                st.title("Plan for the downside.")          #change ???
                html('<div class="subtitle">Understand your exposure. Explore the trade-offs. Stay in control.</div>')      #delete ???
                position_form()
                result=st.session_state.get("hedge_result")
                cross=matching_cross(result)
                if st.session_state.get("friendly_error"):
                    notice("We could not load that stock. Check the ticker, your LSEG connection and your account's data access.")
                    debug_error(st.session_state["friendly_error"])
                if result is None:
                    empty("Your comparison starts here","Enter a stock and choose Explore protection. Your position summary and interactive scenarios will appear below.")      #change ?
                else:
                    st.write("")
                    position_summary(result)
                    view=st.radio("Explore your results",[
                        "Stock protection","ETF comparison","Past risk & return","Model guidance"],
                        horizontal=True,label_visibility="collapsed",key="friendly_result_view")
                    try:
                        if view=="Stock protection":
                            stock_view(result)
                        elif view=="ETF comparison":
                            etf_view(result,cross)
                        elif view=="Past risk & return":
                            history_view(result,cross)
                        else:
                            model_view(result)
                    except Exception as exc:
                        notice("This comparison could not be displayed with the available snapshot.")
                        debug_error(exc)
                    diagnostics(result,cross)
        except Exception as exc:
            notice("This research panel could not be loaded with the available data.")
            debug_error(exc)
        html('<div class="footer">Equity Hedge · Academic research only. No orders are executed. Scenarios are not forecasts or guarantees of protection.</div>')

    main()
