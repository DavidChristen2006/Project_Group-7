
"""ETF option cross hedges: selection and expiration stress tests.

The caller must open the LSEG session before retrieval.
Stock and ETF shocks remain separate to preserve basis risk.
Research only; option Greeks and contract deliverables unverified.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from math import floor, isfinite

import numpy as np
import pandas as pd

from data_pipeline.horizon_option_selection import (
    select_nearest_etf_option_structures,
)


@dataclass(frozen=True)
class OptionLeg:
    ric: str
    right: str
    side: int
    strike: float
    premium: float
    expiry: date


@dataclass(frozen=True)
class ETFOptionStructure:
    name: str
    legs: tuple[OptionLeg, ...]
    multiplier: int = 100

    def __post_init__(self):
        if self.multiplier < 1 or not self.legs:
            raise ValueError("Invalid multiplier or legs")

        if len({leg.expiry for leg in self.legs}) != 1:
            raise ValueError(
                "All option legs must have the same expiry"
            )

        for leg in self.legs:
            if (
                leg.right not in ("put", "call")
                or leg.side not in (-1, 1)
                or not isfinite(leg.strike)
                or leg.strike <= 0
                or not isfinite(leg.premium)
                or leg.premium < 0
            ):
                raise ValueError("Invalid option leg")


def normalize_option_quotes(
    quotes: pd.DataFrame,
    target_date: date,
    max_days_after: int,
    max_spread: float,
) -> pd.DataFrame:

    required = {
        "RIC",
        "CallPutOption",
        "StrikePrice",
        "ExpiryDate",
        "BID",
        "ASK",
    }

    missing = required - set(quotes.columns)

    if missing:
        raise ValueError(
            f"Missing option fields: {sorted(missing)}"
        )

    frame = quotes.copy()

    frame["RIC"] = (
        frame["RIC"].astype("string").str.strip()
    )

    frame["ExpiryDate"] = (
        pd.to_datetime(
            frame["ExpiryDate"],
            errors="coerce",
            utc=True,
        )
        .dt.tz_convert(None)
        .dt.normalize()
    )

    frame["right"] = (
        frame["CallPutOption"]
        .astype("string")
        .str.strip()
        .str.lower()
        .replace({"p": "put", "c": "call"})
    )

    for column in ("StrikePrice", "BID", "ASK"):
        frame[column] = pd.to_numeric(
            frame[column],
            errors="coerce",
        )

    frame = frame.loc[
        frame["RIC"].notna()
        & frame["RIC"].ne("")
        & frame["right"].isin(["put", "call"])
        & frame["StrikePrice"].gt(0)
        & frame["BID"].gt(0)
        & frame["ASK"].ge(frame["BID"])
        & frame["ExpiryDate"].between(
            pd.Timestamp(target_date),
            pd.Timestamp(target_date)
            + pd.DateOffset(days=max_days_after),
        )
    ].copy()

    if frame.empty:
        return frame

    frame["spread"] = (
        (frame["ASK"] - frame["BID"])
        / ((frame["ASK"] + frame["BID"]) / 2)
    )

    return frame.loc[
        frame["spread"].le(max_spread)
    ].copy()


def select_etf_option_structures(
    quotes: pd.DataFrame,
    *,
    etf_spot: float,
    target_date: date,
    multiplier: int = 100,
    assume_standard_contracts: bool = False,
    max_days_after: int = 90,
    max_spread: float = 0.10,
) -> dict[str, ETFOptionStructure]:
    """Select up to three bearish ETF option structures."""

    if not isfinite(etf_spot) or etf_spot <= 0:
        raise ValueError("Invalid ETF price")

    if not 0 < max_spread <= 1:
        raise ValueError("Invalid maximum spread")

    frame = normalize_option_quotes(
        quotes,
        target_date,
        max_days_after,
        max_spread,
    )

    if frame.empty:
        raise ValueError(
            "No eligible ETF option bid/ask quotes"
        )

    verified = pd.Series(
        False,
        index=frame.index,
    )

    for column in ("ContractSize", "ContractMultiplier"):
        if column in frame.columns:
            sizes = pd.to_numeric(
                frame[column],
                errors="coerce",
            )

            incompatible = (
                sizes.notna()
                & ~np.isclose(sizes, multiplier)
            )

            frame = frame.loc[
                ~incompatible
            ].copy()

            if not frame.empty:
                verified = (
                    verified.reindex(
                        frame.index,
                        fill_value=False,
                    )
                    | sizes.reindex(
                        frame.index
                    ).eq(multiplier)
                )

    if not assume_standard_contracts:
        frame = frame.loc[
            verified.reindex(
                frame.index,
                fill_value=False,
            )
        ].copy()

    if frame.empty:
        raise ValueError(
            "No eligible standard-multiplier contracts. "
            "Verify deliverables or enable the "
            "research-only multiplier assumption."
        )

    def pick(
        group,
        right,
        target_fraction,
        min_strike=None,
        max_strike=None,
    ):
        candidates = group.loc[
            group["right"].eq(right)
        ].copy()

        if min_strike is not None:
            candidates = candidates.loc[
                candidates["StrikePrice"].gt(min_strike)
            ]

        if max_strike is not None:
            candidates = candidates.loc[
                candidates["StrikePrice"].lt(max_strike)
            ]

        if candidates.empty:
            return None

        candidates["rank"] = (
            (
                candidates["StrikePrice"] / etf_spot
                - target_fraction
            ).abs()
            + 0.15 * candidates["spread"]
        )

        return candidates.sort_values(
            ["rank", "RIC"]
        ).iloc[0]

    best = None

    for expiry, group in frame.groupby(
        "ExpiryDate",
        sort=True,
    ):
        long_put = pick(
            group, "put", 0.95
        )

        if long_put is None:
            continue

        short_put = pick(
            group,
            "put",
            0.85,
            max_strike=float(
                long_put["StrikePrice"]
            ),
        )

        short_call = pick(
            group, "call", 1.00
        )

        long_call = (
            pick(
                group,
                "call",
                1.10,
                min_strike=float(
                    short_call["StrikePrice"]
                ),
            )
            if short_call is not None
            else None
        )

        expiry_date = pd.Timestamp(
            expiry
        ).date()

        def leg(row, side, right):
            return OptionLeg(
                ric=str(row["RIC"]),
                right=right,
                side=side,
                strike=float(row["StrikePrice"]),
                premium=float(
                    row["ASK"]
                    if side == 1
                    else row["BID"]
                ),
                expiry=expiry_date,
            )

        structures = {
            "long_etf_put": ETFOptionStructure(
                name="long_etf_put",
                legs=(
                    leg(long_put, 1, "put"),
                ),
                multiplier=multiplier,
            )
        }

        if short_put is not None:
            structures["bear_put_spread"] = (
                ETFOptionStructure(
                    name="bear_put_spread",
                    legs=(
                        leg(long_put, 1, "put"),
                        leg(short_put, -1, "put"),
                    ),
                    multiplier=multiplier,
                )
            )

        if (
            short_call is not None
            and long_call is not None
            and float(short_call["BID"])
            > float(long_call["ASK"])
        ):
            structures["bear_call_credit_spread"] = (
                ETFOptionStructure(
                    name="bear_call_credit_spread",
                    legs=(
                        leg(short_call, -1, "call"),
                        leg(long_call, 1, "call"),
                    ),
                    multiplier=multiplier,
                )
            )

        if best is None or len(structures) > len(best):
            best = structures

        if len(best) == 3:
            break

    if best is None:
        raise ValueError(
            "No suitable ETF put in the requested window"
        )

    return best



def retrieve_etf_option_structures(
    etf_ric: str,
    etf_spot: float,
    target_date: date,
    *,
    max_spread: float = 0.10,
    assume_standard_contracts: bool = False,
    multiplier: int = 100,
    max_days_after: int = 90,
) -> dict[str, ETFOptionStructure]:
    """Retrieve option structures for any ETF RIC.

    The caller must already have an open LSEG session.
    No ETF-specific identifier mappings are required.
    """

    from lseg_api.options import (
        get_option_chain,
        add_option_market_data,
        filter_liquid_options,
    )

    from lseg_api.etf_option_discovery import (
        discover_etf_option_chain,
    )

    if not 1 <= max_days_after <= 365:
        raise ValueError(
            "max_days_after must be between 1 and 365"
        )

    start = pd.Timestamp(target_date).normalize()
    end = start + pd.Timedelta(days=max_days_after)

    chain = None
    discovery_errors = []

    # First search the equity-derivatives view using
    # dynamically generated underlying identifiers.
    try:
        chain = discover_etf_option_chain(
            etf_ric=etf_ric,
            etf_spot=etf_spot,
            target_date=target_date,
            max_days_after=max_days_after,
            max_results=2000,
        )

    except Exception as exc:
        discovery_errors.append(
            f"Derivative discovery: {exc}"
        )

    # Fall back to the existing stock/ETF option wrapper.
    # This is particularly important when the derivatives
    # view does not index an ETF in the same way.
    if chain is None or chain.empty:

        try:
            chain = get_option_chain(
                etf_ric,
                minimum_strike=etf_spot * 0.65,
                maximum_strike=etf_spot * 1.20,
                max_results=2000,
            )

        except Exception as exc:
            discovery_errors.append(
                f"Standard chain: {exc}"
            )

    if chain is None or chain.empty:
        raise ValueError(
            "No option contracts discovered for "
            f"{etf_ric}. "
            + " | ".join(discovery_errors)
        )

    required = {
        "RIC",
        "CallPutOption",
        "StrikePrice",
        "ExpiryDate",
    }

    missing = required - set(chain.columns)

    if missing:
        raise ValueError(
            f"Option chain missing fields: {sorted(missing)}"
        )

    work = chain.copy()

    work["ExpiryDate"] = (
        pd.to_datetime(
            work["ExpiryDate"],
            errors="coerce",
            utc=True,
        )
        .dt.tz_convert(None)
        .dt.normalize()
    )

    work["StrikePrice"] = pd.to_numeric(
        work["StrikePrice"],
        errors="coerce",
    )

    work["right"] = (
        work["CallPutOption"]
        .astype("string")
        .str.strip()
        .str.lower()
        .replace({
            "p": "put",
            "c": "call",
        })
    )

    # Preserve an informative list of expirations
    # before applying the target-date filter.
    available_expiries = sorted(
        work["ExpiryDate"].dropna().dt.date.unique()
    )

    work = work.loc[
        work["ExpiryDate"].between(start, end)
        & work["StrikePrice"].gt(0)
        & work["right"].isin(["put", "call"])
    ].copy()

    if work.empty:
        raise ValueError(
            f"No eligible {etf_ric} option contracts "
            f"expiring between {start.date()} "
            f"and {end.date()}. "
            f"Available expirations in returned chain: "
            f"{available_expiries[:20]}"
        )

    # Require at least two put and two call candidates
    # before spending time on bid/ask retrieval.
    counts = (
        work.groupby(
            ["ExpiryDate", "right"]
        )
        .size()
        .unstack(fill_value=0)
    )

    if "put" not in counts.columns:
        counts["put"] = 0

    if "call" not in counts.columns:
        counts["call"] = 0

    eligible_expiries = counts.loc[
        (counts["put"] >= 2)
        & (counts["call"] >= 2)
    ].sort_index()

    if eligible_expiries.empty:
        raise ValueError(
            f"{etf_ric}: contracts were found, "
            "but no expiration has enough put "
            "and call candidates."
        )

    # Prioritize the nearest four sufficiently
    # populated expiration dates.
    expiries = eligible_expiries.index[:12].tolist()

    work = work.loc[
        work["ExpiryDate"].isin(expiries)
    ].copy()

    parts = []

    for (_, right), group in work.groupby(
        ["ExpiryDate", "right"]
    ):

        target_strikes = (
            (0.85, 0.95)
            if right == "put"
            else (1.00, 1.10)
        )

        for fraction in target_strikes:
            subset = group.copy()

            subset["distance"] = (
                subset["StrikePrice"] / etf_spot
                - fraction
            ).abs()

            parts.append(
                subset.nsmallest(
                    5,
                    "distance",
                )
            )

    candidates = (
        pd.concat(parts, ignore_index=True)
        .drop_duplicates("RIC")
        .drop(columns=["right", "distance"])
    )

    quotes = add_option_market_data(
        candidates,
        batch_size=50,
    )

    if quotes is None or quotes.empty:
        raise ValueError(
            f"{etf_ric}: no option bid/ask quotes"
        )

    # Never substitute historical fallback quotes
    # for current-market strategy calculations.
    if "QuoteSource" in quotes.columns:
        historical = (
            quotes["QuoteSource"]
            .astype("string")
            .str.lower()
            .str.contains(
                "historical",
                na=False,
            )
        )

        quotes = quotes.loc[
            ~historical
        ].copy()

    if quotes.empty:
        raise ValueError(
            f"{etf_ric}: only historical fallback "
            "option quotes available"
        )

    liquid = filter_liquid_options(
        quotes,
        max_spread_pct=max_spread,
    )

    if liquid is None or liquid.empty:
        raise ValueError(
            f"{etf_ric}: no liquid option quotes "
            f"at the {max_spread:.0%} spread limit"
        )

    return select_nearest_etf_option_structures(
        liquid=liquid,
        etf_spot=etf_spot,
        target_date=target_date,
        select_structures=select_etf_option_structures,
        multiplier=multiplier,
        assume_standard_contracts=assume_standard_contracts,
        max_spread=max_spread,
        max_expiries=12,
    )



def option_cross_hedge_stresses(
    *,
    stock_spot: float,
    shares: int,
    etf_spot: float,
    beta: float,
    structures: dict[str, ETFOptionStructure],
    assumed_abs_net_delta: float = 0.35,
    option_net_deltas: dict[str, float] | None = None,
    hedge_fractions=(0.25, 0.50, 0.75, 1.00),
    stock_shocks=(-0.30, -0.20, -0.10, 0.0, 0.10, 0.20),
    etf_shocks=(-0.30, -0.20, -0.10, 0.0, 0.10, 0.20),
    fee_per_contract: float = 0.0,
) -> pd.DataFrame:
    """Calculate stock plus ETF-option expiration P&L.

    Option counts approximate initial beta-adjusted
    delta coverage; no actual option Greeks are inferred.
    """

    if not isinstance(shares, int) or shares <= 0:
        raise ValueError("Invalid share quantity")

    parameters = (
        stock_spot,
        etf_spot,
        beta,
        assumed_abs_net_delta,
        fee_per_contract,
    )

    if not all(isfinite(value) for value in parameters):
        raise ValueError("Nonfinite input")

    if (
        stock_spot <= 0
        or etf_spot <= 0
        or beta <= 0
        or not 0 < assumed_abs_net_delta <= 1
        or fee_per_contract < 0
    ):
        raise ValueError(
            "Invalid price, beta, delta or fee"
        )

    if not structures:
        raise ValueError(
            "No ETF option structures"
        )

    if any(
        not 0 < float(fraction) <= 1
        for fraction in hedge_fractions
    ):
        raise ValueError("Invalid hedge fraction")

    all_shocks = (
        *stock_shocks,
        *etf_shocks,
    )

    if any(
        not isfinite(float(shock))
        or float(shock) < -1
        for shock in all_shocks
    ):
        raise ValueError("Invalid return shock")

    stock_notional = shares * stock_spot
    rows = []


    for name, structure in structures.items():

        multiplier = structure.multiplier

        if option_net_deltas is None:
            # Legacy research fallback only.
            net_delta = -assumed_abs_net_delta
            delta_source = "ASSUMED"

        else:
            if name not in option_net_deltas:
                raise ValueError(
                    f"Missing net delta for {name}"
                )

            net_delta = float(
                option_net_deltas[name]
            )

            if (
                not isfinite(net_delta)
                or net_delta >= -0.01
            ):
                raise ValueError(
                    f"Invalid bearish net delta "
                    f"for {name}: {net_delta}"
                )

            delta_source = "LSEG_TR.DELTA"

        delta_magnitude = abs(net_delta)

        for fraction in hedge_fractions:

            target_exposure = (
                stock_notional
                * beta
                * float(fraction)
            )

            per_contract_exposure = (
                etf_spot
                * multiplier
                * delta_magnitude
            )

            contracts = floor(
                target_exposure / per_contract_exposure
                + 0.5
            )

            # Long options are bought at ASK.
            # Short options are sold at BID.
            premium_cashflow = (
                sum(
                    -leg.side * leg.premium
                    for leg in structure.legs
                )
                * multiplier
                * contracts
            )

            fees = (
                contracts
                * len(structure.legs)
                * fee_per_contract
            )

            for stock_return in stock_shocks:
                stock_pnl = (
                    stock_notional
                    * float(stock_return)
                )

                for etf_return in etf_shocks:
                    terminal = (
                        etf_spot
                        * (1 + float(etf_return))
                    )

                    intrinsic = (
                        sum(
                            leg.side
                            * (
                                max(
                                    leg.strike - terminal,
                                    0.0,
                                )
                                if leg.right == "put"
                                else max(
                                    terminal - leg.strike,
                                    0.0,
                                )
                            )
                            for leg in structure.legs
                        )
                        * multiplier
                        * contracts
                    )

                    option_pnl = (
                        intrinsic
                        + premium_cashflow
                        - fees
                    )

                    rows.append({
                        "Strategy": name,
                        "Option Expiry": (
                            structure.legs[0]
                            .expiry.isoformat()
                        ),
                        "Hedge Fraction": float(fraction),
                        "Contracts Per Leg": contracts,
                        "Net Delta Per Structure": net_delta,
                        "Delta Source": delta_source,   
                        "Approximate Initial Hedge Fraction": (
                            contracts
                            * per_contract_exposure
                            / (stock_notional * beta)
                        ),
                        "Stock Shock": float(stock_return),
                        "ETF Shock": float(etf_return),
                        "Unhedged P&L": stock_pnl,
                        "Option Premium Cashflow": premium_cashflow,
                        "Option Expiration Intrinsic Payoff": intrinsic,
                        "Option Fees": -fees,
                        "ETF Option P&L": option_pnl,
                        "Total P&L": stock_pnl + option_pnl,
                    })

    return pd.DataFrame(rows)
