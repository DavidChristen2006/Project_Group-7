
"""Research-only hedge comparison on a common stress grid.

Share hedges are evaluated at the requested hedge target.
Option hedges are evaluated at their own expiration.

Different option expirations are never ranked together.

No ML forecast probabilities, financing costs, dividends,
borrow costs or assignment outcomes are inferred.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from math import isfinite

import pandas as pd


OBJECTIVES = (
    "Capital protection",
    "Balanced downside and upside",
    "Preserve upside",
    "Lowest opening cash outlay",
)


@dataclass(frozen=True)
class OptionLeg:
    underlying: str
    right: str
    side: int
    strike: float
    premium: float
    contracts: int
    multiplier: int = 100


@dataclass(frozen=True)
class Candidate:
    name: str
    expiry: date | None = None
    etf_shares: int = 0
    legs: tuple[OptionLeg, ...] = ()


def _positive(
    value: object,
    label: str,
) -> float:

    number = float(value)

    if not isfinite(number) or number <= 0:
        raise ValueError(
            f"{label} must be positive and finite"
        )

    return number


def _leg_payoff(
    leg: OptionLeg,
    terminal: float,
) -> float:
    """One option leg's expiration P&L."""

    if leg.underlying not in ("stock", "etf"):
        raise ValueError(
            "Invalid option underlying"
        )

    if leg.right not in ("put", "call"):
        raise ValueError(
            "Invalid option right"
        )

    if leg.side not in (-1, 1):
        raise ValueError(
            "Invalid option side"
        )

    if (
        not isinstance(leg.contracts, int)
        or leg.contracts < 1
    ):
        raise ValueError(
            "Option quantity must be positive"
        )

    if (
        not isinstance(leg.multiplier, int)
        or leg.multiplier < 1
    ):
        raise ValueError(
            "Option multiplier must be positive"
        )

    strike = _positive(
        leg.strike,
        "Option strike",
    )

    premium = float(
        leg.premium
    )

    if not isfinite(premium) or premium < 0:
        raise ValueError(
            "Invalid option premium"
        )

    if leg.right == "put":

        intrinsic = max(
            strike - terminal,
            0.0,
        )

    else:

        intrinsic = max(
            terminal - strike,
            0.0,
        )

    return (
        leg.side
        * (intrinsic - premium)
        * leg.contracts
        * leg.multiplier
    )


def opening_debit(
    candidate: Candidate,
    fee_per_leg_contract: float,
) -> float:
    """Positive means net option cash paid; negative means credit."""

    fee = float(
        fee_per_leg_contract
    )

    if not isfinite(fee) or fee < 0:
        raise ValueError(
            "Opening fee must be nonnegative"
        )

    debit = 0.0

    for leg in candidate.legs:

        _positive(
            leg.strike,
            "Option strike",
        )

        if (
            leg.right not in ("put", "call")
            or leg.side not in (-1, 1)
            or leg.contracts < 1
            or leg.multiplier < 1
        ):
            raise ValueError(
                "Invalid option leg"
            )

        if (
            not isfinite(float(leg.premium))
            or leg.premium < 0
        ):
            raise ValueError(
                "Invalid option premium"
            )

        debit += (
            leg.side
            * leg.premium
            * leg.contracts
            * leg.multiplier
            + fee * leg.contracts
        )

    return debit


def scenario_pnl(
    candidate: Candidate,
    *,
    stock_spot: float,
    shares: int,
    stock_return: float,
    etf_spot: float | None,
    etf_return: float,
    fee_per_leg_contract: float,
) -> float:
    """Calculate full-position P&L for one hypothetical scenario."""

    stock_spot = _positive(
        stock_spot,
        "Stock spot",
    )

    if (
        not isfinite(float(fee_per_leg_contract))
        or fee_per_leg_contract < 0
    ):
        raise ValueError(
            "Invalid opening fee"
        )

    if not isinstance(shares, int) or shares < 1:
        raise ValueError(
            "Shares must be a positive integer"
        )

    if (
        not isfinite(stock_return)
        or stock_return < -1
    ):
        raise ValueError(
            "Stock return cannot be below -100%"
        )

    if (
        not isfinite(etf_return)
        or etf_return < -1
    ):
        raise ValueError(
            "ETF return cannot be below -100%"
        )

    needs_etf = (
        bool(candidate.etf_shares)
        or any(
            leg.underlying == "etf"
            for leg in candidate.legs
        )
    )

    if needs_etf:

        etf_spot = _positive(
            etf_spot,
            "ETF spot",
        )

    stock_final = (
        stock_spot
        * (1.0 + stock_return)
    )

    etf_final = (
        etf_spot * (1.0 + etf_return)
        if needs_etf
        else 0.0
    )

    # The entire original stock position.
    pnl = (
        shares
        * stock_spot
        * stock_return
    )

    # Signed ETF shares: negative means short.
    if candidate.etf_shares:

        pnl += (
            candidate.etf_shares
            * etf_spot
            * etf_return
        )

    # Option intrinsic value, paid/received
    # premium, and opening fees.
    for leg in candidate.legs:

        terminal = (
            stock_final
            if leg.underlying == "stock"
            else etf_final
        )

        pnl += _leg_payoff(
            leg,
            terminal,
        )

        pnl -= (
            fee_per_leg_contract
            * leg.contracts
        )

    return float(pnl)


def build_candidates(
    hedge_result: dict,
    cross_result: dict | None = None,
) -> tuple[
    list[Candidate],
    date,
    float | None,
    list[str],
]:
    """Reuse eligible instruments and quantities from the app."""

    inputs = hedge_result["inputs"]
    meta = hedge_result["meta"]

    _positive(
        inputs.spot,
        "Stock spot",
    )

    if (
        not isinstance(inputs.shares, int)
        or inputs.shares < 1
    ):
        raise ValueError(
            "Invalid stock holding"
        )

    target = date.fromisoformat(
        meta["research_target_date"]
    )

    candidates = [
        Candidate("No Hedge")
    ]

    notices = []

    # -----------------------------------------------------
    # DIRECT STOCK OPTIONS
    # -----------------------------------------------------

    contracts = int(
        getattr(inputs, "contracts", 0)
    )

    if (
        hedge_result.get("has_direct_options")
        and not meta.get(
            "used_close_fallback",
            False,
        )
    ):

        verified = (
            meta.get("multiplier_verified_from_fields")
            or meta.get("assumed_multiplier_accepted")
        )

        if contracts > 0 and verified:

            expiry = pd.Timestamp(
                inputs.expiry
            ).date()

            if expiry < target:
                raise ValueError(
                    "Direct option expires before the hedge target"
                )

            put = OptionLeg(
                underlying="stock",
                right="put",
                side=1,
                strike=float(inputs.put_strike),
                premium=float(inputs.put_ask),
                contracts=contracts,
            )

            call = OptionLeg(
                underlying="stock",
                right="call",
                side=-1,
                strike=float(inputs.call_strike),
                premium=float(inputs.call_bid),
                contracts=contracts,
            )

            candidates.extend([
                Candidate(
                    "Protective Put",
                    expiry,
                    legs=(put,),
                ),
                Candidate(
                    "Collar",
                    expiry,
                    legs=(put, call),
                ),
                Candidate(
                    "Covered Call",
                    expiry,
                    legs=(call,),
                ),
            ])

            # The existing app saves the selected
            # lower-strike put in metadata.
            lower = meta.get(
                "put_spread_short_put"
            )

            if lower:

                lower_strike = float(
                    lower["strike"]
                )

                if lower_strike < put.strike:

                    short_put = OptionLeg(
                        underlying="stock",
                        right="put",
                        side=-1,
                        strike=lower_strike,
                        premium=float(
                            lower["bid"]
                        ),
                        contracts=contracts,
                    )

                    candidates.append(
                        Candidate(
                            "Put Spread",
                            expiry,
                            legs=(
                                put,
                                short_put,
                            ),
                        )
                    )

        else:

            notices.append(
                "Direct options excluded: no positive "
                "contract quantity or unverified multiplier."
            )

    elif meta.get(
        "used_close_fallback",
        False,
    ):

        notices.append(
            "Direct options excluded because the stock "
            "uses an unsynchronized closing-price reference."
        )

    # -----------------------------------------------------
    # ETF CROSS HEDGES
    # -----------------------------------------------------

    etf_spot = None

    if cross_result is None:

        return (
            candidates,
            target,
            etf_spot,
            notices,
        )

    # Avoid using ETF analysis from another stock snapshot.
    if (
        cross_result.get("snapshot")
        != meta.get("retrieved_at_new_york")
    ):

        notices.append(
            "ETF analysis excluded: snapshot mismatch."
        )

        return (
            candidates,
            target,
            etf_spot,
            notices,
        )

    etf_spot = _positive(
        cross_result["etf_spot"],
        "ETF spot",
    )

    stock_date = meta.get(
        "underlying_price_date"
    )

    etf_date = cross_result.get(
        "etf_price_date"
    )

    if (
        stock_date
        and etf_date
        and stock_date != etf_date
    ):

        notices.append(
            "Stock and ETF reference prices have "
            "different observation dates. "
            "Results are illustrative."
        )

    etf_name = str(
        cross_result["etf_ric"]
    )

    # -----------------------------------------------------
    # ETF SHARE HEDGES
    # -----------------------------------------------------

    share_rows = cross_result.get(
        "selected"
    )

    if (
        isinstance(share_rows, pd.DataFrame)
        and not share_rows.empty
    ):

        needed = {
            "Hedge Fraction",
            "ETF Position Shares (signed)",
        }

        missing = (
            needed
            - set(share_rows.columns)
        )

        if missing:
            raise ValueError(
                f"Missing ETF share columns: "
                f"{sorted(missing)}"
            )

        for fraction, group in share_rows.groupby(
            "Hedge Fraction"
        ):

            counts = pd.to_numeric(
                group["ETF Position Shares (signed)"],
                errors="raise",
            ).unique()

            if (
                len(counts) != 1
                or not float(
                    counts[0]
                ).is_integer()
            ):
                raise ValueError(
                    "Inconsistent ETF-share quantity"
                )

            count = int(
                counts[0]
            )

            if count != 0:

                candidates.append(
                    Candidate(
                        name=(
                            f"{etf_name} "
                            f"shares {float(fraction):.0%}"
                        ),
                        etf_shares=count,
                    )
                )

    # -----------------------------------------------------
    # ETF OPTION HEDGES
    # -----------------------------------------------------

    structures = (
        cross_result.get(
            "option_structures"
        )
        or {}
    )

    selected = cross_result.get(
        "option_selected"
    )

    if (
        not structures
        or not isinstance(
            selected,
            pd.DataFrame,
        )
        or selected.empty
    ):

        return (
            candidates,
            target,
            etf_spot,
            notices,
        )

    if (
        meta.get("used_close_fallback")
        or cross_result.get(
            "etf_close_fallback"
        )
    ):

        notices.append(
            "ETF options excluded because at least "
            "one underlying uses a closing-price fallback."
        )

        return (
            candidates,
            target,
            etf_spot,
            notices,
        )

    needed = {
        "Strategy",
        "Hedge Fraction",
        "Contracts Per Leg",
    }

    missing = (
        needed
        - set(selected.columns)
    )

    if missing:
        raise ValueError(
            f"Missing ETF option fields: "
            f"{sorted(missing)}"
        )

    # Match raw strategy IDs and display labels.
    def normalize(value: object) -> str:

        return "".join(
            char
            for char in str(value).lower()
            if char.isalnum()
        )

    for name, structure in structures.items():

        rows = selected.loc[
            selected["Strategy"]
            .astype(str)
            .map(normalize)
            .eq(normalize(name))
        ]

        if rows.empty:

            notices.append(
                f"ETF option structure {name} has "
                "no recorded contract quantities."
            )

            continue

        expiries = {
            pd.Timestamp(
                leg.expiry
            ).date()
            for leg in structure.legs
        }

        if len(expiries) != 1:
            raise ValueError(
                f"Mismatched expiration in {name}"
            )

        expiry = next(
            iter(expiries)
        )

        if expiry < target:

            notices.append(
                f"ETF option structure {name} expires "
                "before the requested hedge target."
            )

            continue

        multiplier = int(
            structure.multiplier
        )

        for fraction, group in rows.groupby(
            "Hedge Fraction"
        ):

            counts = pd.to_numeric(
                group["Contracts Per Leg"],
                errors="raise",
            ).unique()

            if (
                len(counts) != 1
                or not float(
                    counts[0]
                ).is_integer()
            ):
                raise ValueError(
                    "Inconsistent ETF option quantity"
                )

            contracts = int(
                counts[0]
            )

            if contracts <= 0:
                continue

            legs = tuple(
                OptionLeg(
                    underlying="etf",
                    right=str(
                        leg.right
                    ).lower(),
                    side=int(leg.side),
                    strike=float(
                        leg.strike
                    ),
                    premium=float(
                        leg.premium
                    ),
                    contracts=contracts,
                    multiplier=multiplier,
                )
                for leg in structure.legs
            )

            candidates.append(
                Candidate(
                    name=(
                        f"{etf_name} "
                        f"{name.replace('_', ' ')} "
                        f"{float(fraction):.0%}"
                    ),
                    expiry=expiry,
                    legs=legs,
                )
            )

    return (
        candidates,
        target,
        etf_spot,
        notices,
    )


def evaluate_candidates(
    candidates: list[Candidate],
    *,
    evaluation_date: date,
    stock_spot: float,
    shares: int,
    etf_spot: float | None,
    fee_per_leg_contract: float,
    objective: str = "Capital protection",
    worst_stock_return: float = -0.30,
    max_abs_etf_return: float = 0.20,
    downside_weight: float = 0.60,
    stress_model: str = "independent",
    stock_on_etf_beta: float | None = None,
    stock_etf_r_squared: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate same-date hedge candidates.

    Modes
    -----
    independent:
        Combine each stock return with independent ETF shocks.

    historical_conditional:
        Center ETF returns using the reverse linear projection
        implied by the historical stock-on-ETF beta and R2.

        ETF center = (R2 / beta) * stock return.

        Add researcher-selected ETF residual shocks.

    These are hypothetical stress grids, not forecasts.
    """

    if objective not in OBJECTIVES:
        raise ValueError(
            "Unrecognized research objective"
        )

    if (
        not isfinite(float(downside_weight))
        or not 0 <= downside_weight <= 1
    ):
        raise ValueError(
            "Downside weight must be between 0 and 1"
        )

    if (
        not isfinite(float(worst_stock_return))
        or not -1 <= worst_stock_return <= -0.10
    ):
        raise ValueError(
            "Worst stock shock must be from -100% to -10%"
        )

    if (
        not isfinite(float(max_abs_etf_return))
        or not 0 <= max_abs_etf_return <= 1
    ):
        raise ValueError(
            "ETF stress width must be between 0% and 100%"
        )

    if not isfinite(float(stock_spot)) or stock_spot <= 0:
        raise ValueError(
            "Stock spot must be positive and finite"
        )

    if not isinstance(shares, int) or shares <= 0:
        raise ValueError(
            "Shares must be a positive integer"
        )

    if (
        not isfinite(float(fee_per_leg_contract))
        or fee_per_leg_contract < 0
    ):
        raise ValueError(
            "Opening fee must be nonnegative"
        )

    if stress_model not in (
        "independent",
        "historical_conditional",
    ):
        raise ValueError(
            "Unknown ETF stress model"
        )

    # -----------------------------------------------------
    # HISTORICAL STOCK-ETF LINKAGE
    # -----------------------------------------------------

    conditional_slope = None

    if stress_model == "historical_conditional":

        if (
            stock_on_etf_beta is None
            or stock_etf_r_squared is None
        ):
            raise ValueError(
                "Historical beta and R-squared required"
            )

        beta = float(stock_on_etf_beta)
        r_squared = float(stock_etf_r_squared)

        if not isfinite(beta) or abs(beta) <= 1e-8:
            raise ValueError(
                "Historical beta must be finite and nonzero"
            )

        if (
            not isfinite(r_squared)
            or not 0 <= r_squared <= 1
        ):
            raise ValueError(
                "Historical R-squared must be within 0 to 1"
            )

        # Beta was estimated from:
        # stock_return ~ beta * ETF_return
        #
        # For a reverse linear projection, the coefficient
        # ETF_return ~ slope * stock_return is R2 / beta.
        #
        # Do not simply use 1 / beta as a conditional mean.

        conditional_slope = r_squared / beta

    # -----------------------------------------------------
    # ELIGIBLE STRATEGIES FOR THIS DATE
    # -----------------------------------------------------

    eligible = [
        candidate
        for candidate in candidates
        if (
            candidate.expiry is None
            or candidate.expiry == evaluation_date
        )
    ]

    if (
        not eligible
        or not any(
            candidate.name == "No Hedge"
            for candidate in eligible
        )
    ):
        raise ValueError(
            "No Hedge baseline required"
        )

    if len({
        candidate.name
        for candidate in eligible
    }) != len(eligible):
        raise ValueError(
            "Strategy names must be unique within a date"
        )

    # -----------------------------------------------------
    # BUILD THE STRESS GRID
    # -----------------------------------------------------

    stock_shocks = sorted(set([
        worst_stock_return,
        2 * worst_stock_return / 3,
        worst_stock_return / 3,
        0.0,
        0.10,
        0.20,
    ]))

    offsets = sorted(set([
        -max_abs_etf_return,
        -max_abs_etf_return / 2,
        0.0,
        max_abs_etf_return / 2,
        max_abs_etf_return,
    ]))

    records = []

    for stock_return in stock_shocks:

        if conditional_slope is not None:

            center = (
                conditional_slope
                * stock_return
            )

        else:

            center = 0.0

        for offset in offsets:

            if conditional_slope is not None:

                # ETF residual shock around the
                # historical linear-projection center.
                # ETF prices cannot fall below zero.

                etf_return = max(
                    -1.0,
                    center + offset,
                )

            else:

                # Original fully independent stress grid.
                etf_return = offset

            baseline = (
                shares
                * stock_spot
                * stock_return
            )

            for candidate in eligible:

                pnl = scenario_pnl(
                    candidate,
                    stock_spot=stock_spot,
                    shares=shares,
                    stock_return=stock_return,
                    etf_spot=etf_spot,
                    etf_return=etf_return,
                    fee_per_leg_contract=fee_per_leg_contract,
                )

                records.append({
                    "Strategy": candidate.name,
                    "Stock return": stock_return,
                    "ETF return": etf_return,
                    "ETF linear center": (
                        center
                        if conditional_slope is not None
                        else float("nan")
                    ),
                    "ETF stress offset": offset,
                    "P&L": pnl,
                    "Benefit vs No Hedge": (
                        pnl - baseline
                    ),
                })

    details = pd.DataFrame(records)

    # -----------------------------------------------------
    # SCORE STRATEGIES
    # -----------------------------------------------------

    notional = (
        stock_spot
        * shares
    )

    summaries = []

    for candidate in eligible:

        subset = details.loc[
            details["Strategy"].eq(
                candidate.name
            )
        ]

        downside = subset.loc[
            subset["Stock return"] < 0
        ]

        upside = subset.loc[
            subset["Stock return"] > 0
        ]

        worst = float(
            downside["P&L"].min()
        )

        mean_downside = float(
            downside["P&L"].mean()
        )

        mean_upside = float(
            upside["P&L"].mean()
        )

        debit = opening_debit(
            candidate,
            fee_per_leg_contract,
        )

        if objective == "Capital protection":

            score = (
                worst / notional
            )

        elif objective in (
            "Balanced downside and upside",
            "Preserve upside",
        ):

            score = (
                downside_weight * mean_downside
                + (1 - downside_weight) * mean_upside
            ) / notional

        else:

            # Net opening option debit only.
            # ETF short margin and borrowing costs excluded.
            score = (
                -debit / notional
            )

        summaries.append({
            "Strategy": candidate.name,
            "Objective score": score,
            "Worst downside P&L": worst,
            "Mean downside benefit": float(
                downside[
                    "Benefit vs No Hedge"
                ].mean()
            ),
            "Mean upside P&L": mean_upside,
            "Opening option debit": debit,
            "Option legs": len(
                candidate.legs
            ),
            "ETF shares (signed)": (
                candidate.etf_shares
            ),
        })

    ranking = (
        pd.DataFrame(summaries)
        .sort_values(
            ["Objective score", "Strategy"],
            ascending=[False, True],
            kind="stable",
        )
        .reset_index(drop=True)
    )

    ranking.insert(
        0,
        "Rank",
        range(1, len(ranking) + 1),
    )

    return ranking, details



def render_risk_aware_recommendations(
    hedge_result: dict,
    cross_result: dict | None = None,
) -> None:
    """Show independent and historically linked stress rankings."""

    import streamlit as st

    st.subheader(
        "Risk-aware strategy comparison"
    )

    st.caption(
        "Hypothetical stress-test ranking, not a "
        "probability forecast or investment instruction. "
        "Financing, dividends, borrowing, margin and "
        "early assignment are excluded."
    )

    # -----------------------------------------------------
    # PREPARE ELIGIBLE STRATEGIES
    # -----------------------------------------------------

    try:

        (
            candidates,
            target,
            etf_spot,
            notices,
        ) = build_candidates(
            hedge_result,
            cross_result,
        )

    except (
        ValueError,
        TypeError,
        KeyError,
        AttributeError,
    ) as exc:

        st.error(
            f"Unable to build eligible strategies: {exc}"
        )

        return

    for message in notices:
        st.info(message)

    if cross_result is None:

        st.info(
            "No ETF analysis is attached to this stock "
            "snapshot. Use ETF Cross-Hedging Research "
            "to evaluate an ETF."
        )

    elif not any(
        candidate.etf_shares != 0
        for candidate in candidates
    ):

        st.info(
            "No nonzero ETF-share hedge is currently eligible."
        )

    # -----------------------------------------------------
    # EVALUATION DATE
    # -----------------------------------------------------

    dates = sorted({
        target,
        *(
            candidate.expiry
            for candidate in candidates
            if candidate.expiry is not None
        ),
    })

    if (
        st.session_state.get(
            "risk_aware_evaluation_date"
        )
        not in dates
    ):

        st.session_state.pop(
            "risk_aware_evaluation_date",
            None,
        )

    evaluation_date = st.selectbox(
        "Scenario evaluation date",
        dates,
        format_func=lambda day: (
            f"{day} — "
            + (
                "requested hedge target"
                if day == target
                else "option-expiration research"
            )
        ),
        key="risk_aware_evaluation_date",
    )

    if evaluation_date != target:

        st.warning(
            "Option strategies are evaluated at their "
            "actual expiration. Rankings across different "
            "evaluation dates are not directly comparable."
        )

    # -----------------------------------------------------
    # RESEARCH OBJECTIVE
    # -----------------------------------------------------

    objective = st.selectbox(
        "Research priority",
        OBJECTIVES,
        key="risk_aware_objective",
    )

    downside_weight = 0.60

    if objective in (
        "Balanced downside and upside",
        "Preserve upside",
    ):

        default_weight = (
            60
            if objective == "Balanced downside and upside"
            else 25
        )

        downside_weight = (
            st.slider(
                "Weight on downside scenarios (%)",
                min_value=0,
                max_value=100,
                value=default_weight,
                step=5,
                key="risk_aware_downside_weight",
            )
            / 100
        )

    # -----------------------------------------------------
    # HISTORICAL BETA / R-SQUARED AVAILABILITY
    # -----------------------------------------------------

    report = (
        (cross_result or {})
        .get("beta_report")
    )

    beta = (
        (cross_result or {})
        .get("beta")
    )

    r_squared = (
        report.get("r_squared")
        if isinstance(report, dict)
        else None
    )

    try:

        linked_available = (
            beta is not None
            and r_squared is not None
            and isfinite(float(beta))
            and abs(float(beta)) > 1e-8
            and isfinite(float(r_squared))
            and 0 <= float(r_squared) <= 1
        )

    except (
        TypeError,
        ValueError,
    ):

        linked_available = False

    # -----------------------------------------------------
    # STRESS-MODE SELECTOR
    # -----------------------------------------------------

    independent_label = (
        "Independent shocks — extreme basis-risk audit"
    )

    linked_label = (
        "Historically linked shocks + residual stress"
    )

    options = [
        independent_label
    ]

    if linked_available:
        options.append(linked_label)

    if (
        st.session_state.get(
            "risk_aware_stress_model"
        )
        not in options
    ):

        st.session_state.pop(
            "risk_aware_stress_model",
            None,
        )

    display_model = st.radio(
        "Stock–ETF stress relationship",
        options,
        index=(
            1
            if linked_available
            else 0
        ),
        key="risk_aware_stress_model",
    )

    stress_model = (
        "historical_conditional"
        if display_model == linked_label
        else "independent"
    )

    if stress_model == "historical_conditional":

        st.caption(
            "Uses the zero-centered historical linear "
            "projection ETF return ≈ "
            "(R² / stock-on-ETF beta) × stock return, "
            "plus a user-selected ETF residual shock. "
            "The 252-session daily beta and R² are not "
            "calibrated 63-session joint forecasts or "
            "statistical confidence bounds."
        )

        st.write(
            f"Historical stock-on-ETF beta: "
            f"**{float(beta):.3f}** | "
            f"R²: **{float(r_squared):.3f}** | "
            f"ETF-on-stock linear coefficient: "
            f"**{float(r_squared) / float(beta):.3f}**"
        )

    else:

        st.caption(
            "ETF and stock shocks vary independently, "
            "including scenarios where the stock falls "
            "while the ETF rises. This is an adverse "
            "basis-risk stress test, not a prediction."
        )

    # -----------------------------------------------------
    # STRESS-SCENARIO SETTINGS
    # -----------------------------------------------------

    left, right = st.columns(2)

    with left:

        worst_pct = st.slider(
            "Worst stock shock included (%)",
            min_value=-100,
            max_value=-10,
            value=-30,
            step=5,
            key="risk_aware_worst_stock",
        )

    with right:

        if stress_model == "historical_conditional":

            width_pct = st.slider(
                "ETF residual shock (+/- percentage points)",
                min_value=0,
                max_value=100,
                value=5,
                step=5,
                key="risk_aware_etf_residual_width",
            )

        else:

            width_pct = st.slider(
                "Independent ETF shock (+/- %)",
                min_value=0,
                max_value=100,
                value=20,
                step=5,
                key="risk_aware_etf_width",
            )

    # -----------------------------------------------------
    # EVALUATE THE SELECTED STRESS MODEL
    # -----------------------------------------------------

    inputs = hedge_result["inputs"]

    common = {
        "candidates": candidates,
        "evaluation_date": evaluation_date,
        "stock_spot": float(inputs.spot),
        "shares": int(inputs.shares),
        "etf_spot": etf_spot,
        "fee_per_leg_contract": float(
            inputs.fee_per_contract
        ),
        "objective": objective,
        "worst_stock_return": (
            worst_pct / 100
        ),
        "max_abs_etf_return": (
            width_pct / 100
        ),
        "downside_weight": downside_weight,
        "stress_model": stress_model,
        "stock_on_etf_beta": (
            float(beta)
            if linked_available
            else None
        ),
        "stock_etf_r_squared": (
            float(r_squared)
            if linked_available
            else None
        ),
    }

    try:

        ranking, stress = evaluate_candidates(
            **common
        )

    except (
        ValueError,
        TypeError,
        KeyError,
        AttributeError,
    ) as exc:

        st.error(
            f"Unable to evaluate strategies: {exc}"
        )

        return

    st.dataframe(
        ranking.style.format({
            "Objective score": "{:+.3%}",
            "Worst downside P&L": "{:,.2f}",
            "Mean downside benefit": "{:+,.2f}",
            "Mean upside P&L": "{:,.2f}",
            "Opening option debit": "{:+,.2f}",
        }),
        hide_index=True,
        use_container_width=True,
    )

    if len(ranking) == 1:

        st.info(
            "Only No Hedge is eligible "
            "on this evaluation date."
        )

    else:

        st.caption(
            f"Highest score under '{objective}' "
            f"and the selected stress assumptions: "
            f"{ranking.iloc[0]['Strategy']}. "
            "This is a scenario ranking, not a "
            "validated ML recommendation."
        )

    # -----------------------------------------------------
    # KEEP EXTREME BASIS RISK VISIBLE
    # -----------------------------------------------------

    if stress_model == "historical_conditional":

        audit_args = dict(common)

        audit_args.update({
            "stress_model": "independent",
            "max_abs_etf_return": 0.20,
        })

        independent_ranking, _ = evaluate_candidates(
            **audit_args
        )

        independent_worst = (
            independent_ranking[
                [
                    "Strategy",
                    "Worst downside P&L",
                ]
            ].rename(columns={
                "Worst downside P&L":
                    "Independent worst P&L (+/-20% ETF)"
            })
        )

        linked_worst = (
            ranking[
                [
                    "Strategy",
                    "Worst downside P&L",
                ]
            ].rename(columns={
                "Worst downside P&L":
                    "Linked worst P&L"
            })
        )

        with st.expander(
            "Independent basis-risk audit (same strategies)"
        ):

            st.warning(
                "A favorable linked-scenario ranking "
                "does not guarantee protection. "
                "The independent stress permits a falling "
                "stock and rising ETF and should be "
                "reviewed as an adverse basis-risk case."
            )

            st.dataframe(
                linked_worst.merge(
                    independent_worst,
                    on="Strategy",
                ),
                hide_index=True,
                use_container_width=True,
            )

    # -----------------------------------------------------
    # FULL SCENARIO AUDIT
    # -----------------------------------------------------

    with st.expander(
        "Inspect all hypothetical stock/ETF scenario results"
    ):

        st.dataframe(
            stress,
            hide_index=True,
            use_container_width=True,
        )
