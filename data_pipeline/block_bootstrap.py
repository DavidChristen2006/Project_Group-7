
"""Paired moving-block bootstrap for research-only stock/ETF hedges.

Each resampled block retains paired daily returns and within-block order.
The bootstrap estimates a distribution conditional on historical data,
not an independent backtest or probability guarantee.
"""

from __future__ import annotations

from datetime import date
from math import isfinite

import numpy as np
import pandas as pd


# =========================================================
# 1. PREPARE SYNCHRONIZED DAILY RETURNS
# =========================================================

def prepare_paired_daily_returns(
    stock_prices: pd.Series,
    etf_prices: pd.Series,
    *,
    calendar_index: pd.DatetimeIndex | None = None,
    as_of: date | None = None,
) -> pd.DataFrame:
    """Prepare daily returns on a common NYSE session calendar.

    Missing prices remain NaN. We never forward-fill them.
    A resampled block cannot cross a missing observation.
    """

    def normalize(
        prices: pd.Series,
        name: str,
    ) -> pd.Series:

        if not isinstance(prices, pd.Series) or prices.empty:
            raise ValueError(
                f"Missing {name} price series"
            )

        values = pd.to_numeric(
            prices,
            errors="coerce",
        )

        index = pd.to_datetime(
            prices.index,
            errors="coerce",
            utc=True,
        )

        if index.isna().any():
            raise ValueError(
                f"Invalid {name} dates"
            )

        values.index = (
            index.tz_convert(None)
            .normalize()
        )

        if values.index.has_duplicates:
            raise ValueError(
                f"Duplicate {name} dates"
            )

        observed = values.dropna()

        if (
            observed.empty
            or (observed <= 0).any()
            or not np.isfinite(observed).all()
        ):
            raise ValueError(
                f"Invalid {name} adjusted prices"
            )

        return values.sort_index()

    stock = normalize(
        stock_prices,
        "stock",
    )

    etf = normalize(
        etf_prices,
        "ETF",
    )

    if calendar_index is None:

        import pandas_market_calendars as mcal

        first = min(
            stock.index.min(),
            etf.index.min(),
        )

        last = max(
            stock.index.max(),
            etf.index.max(),
        )

        if as_of is not None:
            last = min(
                last,
                pd.Timestamp(as_of),
            )

        if last < first:
            raise ValueError(
                "No calendar sessions before as-of date"
            )

        dates = (
            mcal.get_calendar("NYSE")
            .valid_days(
                start_date=first,
                end_date=last,
            )
            .tz_localize(None)
            .normalize()
        )

    else:

        dates = pd.DatetimeIndex(
            pd.to_datetime(
                calendar_index,
                errors="coerce",
                utc=True,
            )
        )

        if (
            dates.isna().any()
            or dates.has_duplicates
        ):
            raise ValueError(
                "Invalid or duplicate calendar dates"
            )

        dates = (
            dates.tz_convert(None)
            .normalize()
            .sort_values()
        )

        if dates.has_duplicates:
            raise ValueError(
                "Duplicate normalized calendar dates"
            )

    if as_of is not None:
        dates = dates[
            dates <= pd.Timestamp(as_of)
        ]

    if len(dates) < 2:
        raise ValueError(
            "At least two calendar sessions are required"
        )

    paired = pd.DataFrame({
        "Stock close": stock.reindex(dates),
        "ETF close": etf.reindex(dates),
    })

    # Explicitly prevent forward-filling of missing closes.
    returns = paired.pct_change(
        fill_method=None
    )

    returns.columns = [
        "Stock return",
        "ETF return",
    ]

    return returns


# =========================================================
# 2. MOVING-BLOCK BOOTSTRAP
# =========================================================

def bootstrap_paired_horizon_returns(
    daily_returns: pd.DataFrame,
    *,
    horizon_sessions: int,
    block_sessions: int = 10,
    simulations: int = 3000,
    seed: int = 42,
) -> tuple[pd.DataFrame, int]:
    """Generate synthetic joint returns over a hedge horizon.

    Draw contiguous blocks of paired stock/ETF daily returns.
    Concatenate blocks until the selected horizon is reached.
    Compound returns separately for the stock and ETF.

    Returns:
        simulated_returns, number_of_valid_block_starts
    """

    for label, value in (
        ("horizon_sessions", horizon_sessions),
        ("block_sessions", block_sessions),
        ("simulations", simulations),
    ):

        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 1
        ):
            raise ValueError(
                f"{label} must be a positive integer"
            )

    if block_sessions > horizon_sessions:
        raise ValueError(
            "Block length cannot exceed hedge horizon"
        )

    if (
        isinstance(seed, bool)
        or not isinstance(seed, int)
        or not 0 <= seed <= 2**32 - 1
    ):
        raise ValueError(
            "Seed must be an unsigned 32-bit integer"
        )

    required = [
        "Stock return",
        "ETF return",
    ]

    if (
        not isinstance(daily_returns, pd.DataFrame)
        or not set(required).issubset(
            daily_returns.columns
        )
    ):
        raise ValueError(
            "Paired daily return fields are missing"
        )

    if (
        daily_returns.empty
        or not daily_returns.index.is_monotonic_increasing
        or daily_returns.index.has_duplicates
    ):
        raise ValueError(
            "Daily return dates must be sorted and unique"
        )

    values = (
        daily_returns[required]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .to_numpy(dtype=float)
    )

    if (
        np.isinf(values).any()
        or np.any(
            values[np.isfinite(values)] <= -1
        )
    ):
        raise ValueError(
            "Invalid daily returns"
        )

    valid_day = (
        np.isfinite(values)
        .all(axis=1)
    )

    if len(values) < block_sessions:
        raise ValueError(
            "Insufficient daily observations for block length"
        )

    # Every day in the original block must have
    # both the stock and ETF return available.
    complete_counts = np.convolve(
        valid_day.astype(int),
        np.ones(
            block_sessions,
            dtype=int,
        ),
        mode="valid",
    )

    valid_starts = np.flatnonzero(
        complete_counts == block_sessions
    )

    if not valid_starts.size:
        raise ValueError(
            "No complete paired daily blocks "
            "of requested length"
        )

    blocks_per_path = (
        horizon_sessions
        + block_sessions
        - 1
    ) // block_sessions

    rng = np.random.default_rng(
        seed
    )

    sampled_starts = rng.choice(
        valid_starts,
        size=(
            simulations,
            blocks_per_path,
        ),
        replace=True,
    )

    offsets = np.arange(
        block_sessions
    )

    indices = (
        sampled_starts[..., None]
        + offsets
    ).reshape(
        simulations,
        -1,
    )[:, :horizon_sessions]

    # Shape:
    # simulations x horizon_sessions x two assets
    sampled = values[
        indices
    ]

    # Compound the daily returns over the horizon.
    compounded = np.expm1(
        np.log1p(sampled)
        .sum(axis=1)
    )

    if not np.isfinite(compounded).all():
        raise ValueError(
            "Bootstrap produced nonfinite compounded returns"
        )

    result = pd.DataFrame(
        compounded,
        columns=required,
    )

    result.index.name = (
        "Simulated path"
    )

    return result, int(
        valid_starts.size
    )


# =========================================================
# 3. APPLY SIMULATED RETURNS TO THE HEDGE
# =========================================================

def evaluate_bootstrap_share_hedges(
    candidates: list,
    simulated_returns: pd.DataFrame,
    *,
    stock_spot: float,
    shares: int,
    etf_spot: float,
    confidence: float = 0.95,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate simulated P&L, VaR and CVaR for share hedges."""

    from data_pipeline.tail_risk_analysis import (
        empirical_var_cvar,
    )

    for label, price in (
        ("stock_spot", stock_spot),
        ("etf_spot", etf_spot),
    ):

        if (
            not isfinite(float(price))
            or float(price) <= 0
        ):
            raise ValueError(
                f"{label} must be positive and finite"
            )

    if (
        isinstance(shares, bool)
        or not isinstance(shares, int)
        or shares < 1
    ):
        raise ValueError(
            "Shares must be a positive integer"
        )

    required = {
        "Stock return",
        "ETF return",
    }

    if (
        not isinstance(simulated_returns, pd.DataFrame)
        or not required.issubset(
            simulated_returns.columns
        )
    ):
        raise ValueError(
            "Missing simulated stock/ETF return fields"
        )

    paired = (
        simulated_returns[
            ["Stock return", "ETF return"]
        ]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
    )

    if (
        paired.empty
        or not np.isfinite(
            paired.to_numpy()
        ).all()
        or (paired < -1).any().any()
    ):
        raise ValueError(
            "Simulated returns must be finite "
            "and no lower than -100%"
        )

    # Historical options are intentionally excluded.
    eligible = [
        item
        for item in candidates
        if (
            item.expiry is None
            and not item.legs
        )
    ]

    names = [
        str(item.name)
        for item in eligible
    ]

    if (
        names.count("No Hedge") != 1
        or len(names) != len(set(names))
    ):
        raise ValueError(
            "Unique strategies including exactly "
            "one No Hedge are required"
        )

    if any(
        isinstance(item.etf_shares, bool)
        or not isinstance(item.etf_shares, int)
        for item in eligible
    ):
        raise ValueError(
            "ETF share positions must be integers"
        )

    baseline_candidate = next(
        item
        for item in eligible
        if item.name == "No Hedge"
    )

    if baseline_candidate.etf_shares != 0:
        raise ValueError(
            "No Hedge must own zero ETF shares"
        )

    stock_return = (
        paired["Stock return"]
        .to_numpy(dtype=float)
    )

    etf_return = (
        paired["ETF return"]
        .to_numpy(dtype=float)
    )

    baseline_pnl = (
        int(shares)
        * float(stock_spot)
        * stock_return
    )

    metrics = []
    pnl_records = []

    for candidate in eligible:

        benefit = (
            candidate.etf_shares
            * float(etf_spot)
            * etf_return
        )

        pnl = (
            baseline_pnl + benefit
        )

        var, cvar, tail_weight = empirical_var_cvar(
            -pnl,
            confidence,
        )

        metrics.append({
            "Strategy": str(candidate.name),
            "ETF shares (signed)":
                candidate.etf_shares,
            "Simulation paths": len(pnl),
            "VaR loss": var,
            "CVaR loss": cvar,
            "CVaR / stock notional": (
                cvar / (stock_spot * shares)
            ),
            "Worst simulated loss": float(
                (-pnl).max()
            ),
            "Mean simulated P&L": float(
                pnl.mean()
            ),
            "Mean benefit vs No Hedge": float(
                benefit.mean()
            ),
            "Tail path weight": tail_weight,
        })

        pnl_records.append(
            pd.DataFrame({
                "Simulated path": np.arange(len(pnl)),
                "Strategy": str(candidate.name),
                "Stock return": stock_return,
                "ETF return": etf_return,
                "P&L": pnl,
                "Benefit vs No Hedge": benefit,
            })
        )

    ranking = (
        pd.DataFrame(metrics)
        .sort_values(
            ["CVaR loss", "Strategy"],
            kind="stable",
        )
        .reset_index(drop=True)
    )

    ranking.insert(
        0,
        "Rank",
        np.arange(
            1,
            len(ranking) + 1,
        ),
    )

    details = pd.concat(
        pnl_records,
        ignore_index=True,
    )

    return ranking, details


# =========================================================
# 4. STREAMLIT INTERFACE
# =========================================================

def render_block_bootstrap(
    hedge_result: dict,
    cross_result: dict | None,
) -> None:
    """Reuse the daily LSEG prices cached by historical calibration."""

    import streamlit as st

    from data_pipeline.risk_aware_recommendations import (
        build_candidates,
    )

    if (
        not cross_result
        or cross_result.get("snapshot")
        != hedge_result["meta"].get(
            "retrieved_at_new_york"
        )
    ):
        return

    st.subheader(
        "Paired moving-block bootstrap"
    )

    st.caption(
        "Simulates horizon-matched joint stock/ETF "
        "returns by resampling contiguous blocks of "
        "historical paired daily returns. "
        "Paths are synthetic, not out-of-sample "
        "observations. Historical option contracts "
        "are excluded."
    )

    saved = st.session_state.get(
        "empirical_basis_result"
    )

    if (
        not isinstance(saved, dict)
        or not {
            "key",
            "stock_prices",
            "etf_prices",
        }.issubset(saved)
    ):
        st.info(
            "Click 'Calibrate from LSEG adjusted daily "
            "closes' above again to cache the daily "
            "prices needed for bootstrap."
        )

        return

    inputs = hedge_result["inputs"]
    meta = hedge_result["meta"]

    report = (
        cross_result.get("beta_report")
        or {}
    )

    key = saved["key"]

    # -----------------------------------------------------
    # PREVENT CROSS-INSTRUMENT OR CROSS-HORIZON CACHE REUSE
    # -----------------------------------------------------

    try:

        compatible = (
            len(key) == 7

            and key[0] == str(inputs.ric)

            and key[1] == str(
                cross_result["etf_ric"]
            )

            and int(key[2]) == int(
                meta["research_horizon_sessions"]
            )

            and np.isclose(
                float(key[5]),
                float(cross_result["beta"]),
                rtol=0,
                atol=1e-12,
            )

            and np.isclose(
                float(key[6]),
                float(report["r_squared"]),
                rtol=0,
                atol=1e-12,
            )
        )

    except (
        IndexError,
        KeyError,
        TypeError,
        ValueError,
    ):

        compatible = False

    if not compatible:

        st.warning(
            "Cached daily prices are for a different "
            "stock, ETF, horizon or beta estimate. "
            "Recalibrate first."
        )

        return

    horizon = int(
        meta["research_horizon_sessions"]
    )

    # -----------------------------------------------------
    # BOOTSTRAP CONTROLS
    # -----------------------------------------------------

    left, middle, right = st.columns(3)

    with left:

        block = int(
            st.number_input(
                "Block length (sessions)",
                min_value=1,
                max_value=horizon,
                value=min(10, horizon),
                step=1,
                key=f"joint_bootstrap_block_{horizon}",
            )
        )

    with middle:

        n_paths = int(
            st.number_input(
                "Simulated paths",
                min_value=200,
                max_value=20000,
                value=3000,
                step=200,
                key="joint_bootstrap_paths",
            )
        )

    with right:

        seed = int(
            st.number_input(
                "Random seed",
                min_value=0,
                max_value=999999,
                value=42,
                step=1,
                key="joint_bootstrap_seed",
            )
        )

    confidence = st.selectbox(
        "Bootstrap CVaR confidence",
        [
            0.90,
            0.95,
            0.975,
            0.99,
        ],
        index=1,
        format_func=lambda value: (
            f"{value:.1%}"
        ),
        key="joint_bootstrap_confidence",
    )

    # -----------------------------------------------------
    # REUSE CURRENT ETF SHARE POSITIONS
    # -----------------------------------------------------

    try:

        candidates, _, etf_spot, notices = (
            build_candidates(
                hedge_result,
                cross_result,
            )
        )

        if etf_spot is None:
            raise ValueError(
                "No ETF reference price available"
            )

        identity = (
            tuple(key),
            str(cross_result["snapshot"]),
            float(inputs.spot),
            int(inputs.shares),
            float(etf_spot),

            tuple(sorted(
                (
                    str(candidate.name),
                    int(candidate.etf_shares),
                )
                for candidate in candidates
                if (
                    candidate.expiry is None
                    and not candidate.legs
                )
            )),

            block,
            n_paths,
            seed,
            float(confidence),
        )

    except (
        ValueError,
        TypeError,
        KeyError,
        AttributeError,
    ) as exc:

        st.error(
            f"Cannot assemble bootstrap candidates: {exc}"
        )

        return

    # -----------------------------------------------------
    # RUN ON REQUEST, THEN CACHE ONLY FOR IDENTICAL INPUTS
    # -----------------------------------------------------

    if st.button(
        "Run paired moving-block bootstrap",
        key="joint_bootstrap_run",
    ):

        try:

            # The calibration key records the
            # last completed market session.
            as_of = (
                pd.Timestamp(key[4])
                .date()
            )

            daily = prepare_paired_daily_returns(
                saved["stock_prices"],
                saved["etf_prices"],
                as_of=as_of,
            )

            simulations, n_starts = (
                bootstrap_paired_horizon_returns(
                    daily,
                    horizon_sessions=horizon,
                    block_sessions=block,
                    simulations=n_paths,
                    seed=seed,
                )
            )

            ranking, details = (
                evaluate_bootstrap_share_hedges(
                    candidates,
                    simulations,
                    stock_spot=float(inputs.spot),
                    shares=int(inputs.shares),
                    etf_spot=float(etf_spot),
                    confidence=float(confidence),
                )
            )

            st.session_state[
                "joint_bootstrap_result"
            ] = {
                "identity": identity,
                "ranking": ranking,
                "details": details,
                "valid_days": int(
                    daily.dropna().shape[0]
                ),
                "n_starts": n_starts,
            }

        except (
            ValueError,
            TypeError,
            KeyError,
            AttributeError,
            ImportError,
        ) as exc:

            st.session_state.pop(
                "joint_bootstrap_result",
                None,
            )

            st.error(
                f"Bootstrap unavailable: {exc}"
            )

            return

    outcome = st.session_state.get(
        "joint_bootstrap_result"
    )

    if (
        not isinstance(outcome, dict)
        or outcome.get("identity") != identity
    ):

        st.info(
            "Run the bootstrap to evaluate "
            "the current inputs."
        )

        return

    ranking = outcome["ranking"]
    details = outcome["details"]

    n_starts = outcome["n_starts"]
    valid_days = outcome["valid_days"]

    # -----------------------------------------------------
    # DISPLAY RESULTS
    # -----------------------------------------------------

    st.write(
        f"**{n_starts:,}** valid historical paired-block "
        f"starts | **{n_paths:,}** synthetic paths | "
        f"**{valid_days:,}** valid paired daily returns"
    )

    st.dataframe(
        ranking.style.format({
            "VaR loss": "{:,.2f}",
            "CVaR loss": "{:,.2f}",
            "CVaR / stock notional": "{:.2%}",
            "Worst simulated loss": "{:,.2f}",
            "Mean simulated P&L": "{:+,.2f}",
            "Mean benefit vs No Hedge": "{:+,.2f}",
            "Tail path weight": "{:.1f}",
        }),
        hide_index=True,
        use_container_width=True,
    )

    st.warning(
        "Bootstrap paths reuse historical market data; "
        "more simulated paths do not mean more independent "
        "historical evidence. Block length and historical "
        "regimes can substantially change results. "
        "Trading costs, borrowing, dividends, financing "
        "and option pricing are excluded."
    )

    with st.expander(
        "Inspect simulated paired returns and strategy P&L"
    ):

        st.dataframe(
            details,
            hide_index=True,
            use_container_width=True,
        )
