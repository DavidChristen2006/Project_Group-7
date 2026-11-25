
"""Compare LSEG option snapshots using the tested expiration-payoff engine.

Use --research-historical-fallback only for non-executable scenarios.

No historical ML forecast is attached to contemporary option quotes.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from data_pipeline.hedge_strategy_payoffs import (
    HedgeInputs,
    expiration_scenarios,
)


ROOT = Path(__file__).resolve().parents[1]

OUTPUT = (
    ROOT
    / "data/processed/hedge_comparisons/lseg_snapshot_v1"
)


def prepare_candidates(
    chain: pd.DataFrame,
    *,
    spot: float,
    target_expiry: pd.Timestamp,
    max_expiry_days: int = 35,
    max_expiries: int = 4,
    per_type_per_expiry: int = 16,
) -> pd.DataFrame:
    """Preselect contracts to reduce LSEG quote requests."""

    required = {
        "RIC",
        "CallPutOption",
        "StrikePrice",
        "ExpiryDate",
    }

    missing = required - set(chain.columns)

    if missing:
        raise ValueError(
            f"Option chain missing: {sorted(missing)}"
        )

    frame = chain.copy()

    frame["RIC"] = (
        frame["RIC"]
        .astype("string")
        .str.strip()
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

    frame["StrikePrice"] = pd.to_numeric(
        frame["StrikePrice"],
        errors="coerce",
    )

    frame["Option Type"] = (
        frame["CallPutOption"]
        .astype("string")
        .str.strip()
        .str.lower()
        .replace({
            "p": "put",
            "c": "call",
        })
    )

    frame = frame.loc[
        frame["RIC"].notna()
        & frame["RIC"].ne("")
        & frame["Option Type"].isin(
            ["put", "call"]
        )
        & frame["StrikePrice"].gt(0)
        & frame["ExpiryDate"].between(
            pd.Timestamp(target_expiry),
            pd.Timestamp(target_expiry)
            + pd.DateOffset(days=max_expiry_days),
        )
    ].copy()

    if frame.empty:
        raise ValueError(
            "No contracts near the requested hedge expiry"
        )

    expiries = sorted(
        frame["ExpiryDate"]
        .dropna()
        .unique()
    )[:max_expiries]

    frame = frame.loc[
        frame["ExpiryDate"].isin(expiries)
    ].copy()

    frame["Target Strike"] = np.where(
        frame["Option Type"].eq("put"),
        spot * 0.90,
        spot * 1.10,
    )

    frame["Distance"] = (
        frame["StrikePrice"]
        - frame["Target Strike"]
    ).abs()

    frame = frame.sort_values(
        [
            "ExpiryDate",
            "Option Type",
            "Distance",
            "RIC",
        ]
    )

    frame = (
        frame.groupby(
            ["ExpiryDate", "Option Type"],
            group_keys=False,
        )
        .head(per_type_per_expiry)
    )

    return (
        frame.drop(
            columns=[
                "Option Type",
                "Target Strike",
                "Distance",
            ]
        )
        .drop_duplicates("RIC")
    )


def choose_pair(
    liquid: pd.DataFrame,
    *,
    spot: float,
    target_expiry: pd.Timestamp,
    multiplier: int,
    assume_standard_contracts: bool,
) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    """Select a same-expiry put/call pair.

    Selection uses strike proximity and quoted spread.
    It does not require Greeks or estimate hedge utility.
    """

    required = {
        "RIC",
        "CallPutOption",
        "StrikePrice",
        "ExpiryDate",
        "BID",
        "ASK",
    }

    missing = required - set(liquid.columns)

    if missing:
        raise ValueError(
            f"Liquid quotes missing: {sorted(missing)}"
        )

    frame = liquid.copy()

    frame["ExpiryDate"] = (
        pd.to_datetime(
            frame["ExpiryDate"],
            errors="coerce",
            utc=True,
        )
        .dt.tz_convert(None)
        .dt.normalize()
    )

    frame["Option Type"] = (
        frame["CallPutOption"]
        .astype("string")
        .str.strip()
        .str.lower()
        .replace({
            "p": "put",
            "c": "call",
        })
    )

    for column in (
        "StrikePrice",
        "BID",
        "ASK",
    ):
        frame[column] = pd.to_numeric(
            frame[column],
            errors="coerce",
        )

    frame = frame.loc[
        frame["Option Type"].isin(
            ["put", "call"]
        )
        & frame["ExpiryDate"].ge(target_expiry)
        & frame["StrikePrice"].gt(0)
        & frame["BID"].gt(0)
        & frame["ASK"].ge(frame["BID"])
    ].copy()

    if frame.empty:
        raise ValueError(
            "No valid positive bid/ask quotes"
        )

    # Exclude explicitly incompatible contract sizes.
    for column in (
        "ContractSize",
        "ContractMultiplier",
    ):
        if column in frame.columns:

            sizes = pd.to_numeric(
                frame[column],
                errors="coerce",
            )

            incompatible = (
                sizes.notna()
                & ~np.isclose(
                    sizes,
                    multiplier,
                )
            )

            frame = frame.loc[
                ~incompatible
            ].copy()

    if not assume_standard_contracts:

        multiplier_verified = False

        for column in (
            "ContractSize",
            "ContractMultiplier",
        ):
            if (
                column in frame.columns
                and frame[column].notna().all()
            ):
                multiplier_verified = True

        if not multiplier_verified:
            raise ValueError(
                "Contract multiplier not verified. "
                "For research with an assumed standard "
                "contract, add --assume-standard-contracts."
            )

    frame["Spread Fraction"] = (
        (frame["ASK"] - frame["BID"])
        / (
            (frame["ASK"] + frame["BID"])
            / 2.0
        )
    )

    frame["Strike Distance"] = np.where(
        frame["Option Type"].eq("put"),
        abs(
            frame["StrikePrice"] / spot - 0.90
        ),
        abs(
            frame["StrikePrice"] / spot - 1.10
        ),
    )

    frame["Candidate Score"] = (
        10 * frame["Strike Distance"]
        + 2 * frame["Spread Fraction"]
    )

    alternatives = []

    for expiry, group in frame.groupby(
        "ExpiryDate",
        sort=True,
    ):
        puts = (
            group.loc[
                group["Option Type"].eq("put")
            ]
            .sort_values(
                ["Candidate Score", "RIC"]
            )
        )

        calls = (
            group.loc[
                group["Option Type"].eq("call")
            ]
            .sort_values(
                ["Candidate Score", "RIC"]
            )
        )

        if puts.empty or calls.empty:
            continue

        put = puts.iloc[0]
        call = calls.iloc[0]

        if not (
            put["StrikePrice"]
            < call["StrikePrice"]
        ):
            continue

        expiry_penalty = (
            expiry - target_expiry
        ).days / 365.0

        combined_score = (
            float(put["Candidate Score"])
            + float(call["Candidate Score"])
            + expiry_penalty
        )

        alternatives.append(
            (
                combined_score,
                put,
                call,
            )
        )

    if not alternatives:
        raise ValueError(
            "No liquid put/call pair with the same "
            "expiry and appropriate strike ordering"
        )

    alternatives.sort(
        key=lambda item: (
            item[0],
            str(item[1]["RIC"]),
            str(item[2]["RIC"]),
        )
    )

    _, put, call = alternatives[0]

    return put, call, frame


def main() -> None:

    """Run the module entry point and its workflow.

    Returns:
        Computed result or status for the caller.
    """
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--ric",
        default="NVDA.O",
    )

    parser.add_argument(
        "--shares",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--horizon-sessions",
        type=int,
        default=63,
    )

    parser.add_argument(
        "--max-spread-pct",
        type=float,
        default=0.10,
    )

    parser.add_argument(
        "--fee-per-contract",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--multiplier",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--assume-standard-contracts",
        action="store_true",
    )

    parser.add_argument(
        "--research-historical-fallback",
        action="store_true",
    )

    args = parser.parse_args()

    if args.horizon_sessions <= 0:
        raise ValueError(
            "Hedge horizon must be positive"
        )

    if not (
        0 < args.max_spread_pct <= 1
    ):
        raise ValueError(
            "Invalid maximum spread"
        )

    if (
        args.multiplier < 1
        or args.shares < args.multiplier
    ):
        raise ValueError(
            "Shares must cover at least one contract"
        )

    # Import existing LSEG project functions.
    from lseg_api.equity import (
        get_current_equity_price,
    )

    from lseg_api.options import (
        get_option_chain,
        add_option_market_data,
        filter_liquid_options,
    )

    import lseg.data as ld

    timestamp = datetime.now().astimezone()

    as_of = timestamp.date()

    # Approximation only: actual exchange trading calendars
    # must be used for precise session-to-expiry mapping.
    target_days = round(
        args.horizon_sessions * 365 / 252
    )

    target_expiry = pd.Timestamp(
        as_of + timedelta(days=target_days)
    )

    ld.open_session()

    try:
        spot = float(
            get_current_equity_price(
                args.ric
            )
        )

        if (
            not math.isfinite(spot)
            or spot <= 0
        ):
            raise ValueError(
                "Invalid underlying price"
            )

        chain = get_option_chain(
            args.ric,
            minimum_strike=spot * 0.70,
            maximum_strike=spot * 1.20,
            max_results=1500,
        )

        if (
            chain is None
            or chain.empty
        ):
            raise ValueError(
                "Option discovery returned no contracts"
            )

        candidates = prepare_candidates(
            chain,
            spot=spot,
            target_expiry=target_expiry,
        )

        quoted = add_option_market_data(
            candidates
        )

        if args.research_historical_fallback:

            from lseg_api.option_history import (
                fill_historical_option_quotes,
            )

            quoted = fill_historical_option_quotes(
                quoted,
                batch_size=25,
                count=5,
                max_age_days=7,
            )

        elif "QuoteSource" in quoted.columns:

            historical = (
                quoted["QuoteSource"]
                .astype("string")
                .str.lower()
                .eq("historical")
                .fillna(False)
            )

            quoted = quoted.loc[
                ~historical
            ].copy()

        liquid = filter_liquid_options(
            quoted,
            max_spread_pct=args.max_spread_pct,
        )

        if (
            liquid is None
            or liquid.empty
        ):
            raise ValueError(
                "No liquid option contracts found. "
                "For non-executable indicative research, "
                "try --research-historical-fallback."
            )

        put, call, eligible = choose_pair(
            liquid,
            spot=spot,
            target_expiry=target_expiry,
            multiplier=args.multiplier,
            assume_standard_contracts=(
                args.assume_standard_contracts
            ),
        )

    finally:
        ld.close_session()

    # When fallback is enabled, the entire comparison is
    # an indicative research what-if, not a live quote.
    source = (
        "illustrative"
        if args.research_historical_fallback
        else "live"
    )

    hedge = HedgeInputs(
        ric=args.ric,
        as_of=as_of,
        expiry=pd.Timestamp(
            put["ExpiryDate"]
        ).date(),
        shares=args.shares,
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
        source=source,
        multiplier=args.multiplier,
        fee_per_contract=args.fee_per_contract,
    )

    payoff = expiration_scenarios(
        hedge
    )

    OUTPUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    safe_ric = "".join(
        ch if (
            ch.isalnum()
            or ch in "._-"
        ) else "_"
        for ch in args.ric
    )

    filename = (
        f"{safe_ric}_"
        f"{timestamp:%Y%m%d_%H%M%S}_"
        f"{hedge.expiry}"
    )

    csv_path = (
        OUTPUT / f"{filename}_payoffs.csv"
    )

    audit_path = (
        OUTPUT / f"{filename}_audit.json"
    )

    payoff.to_csv(
        csv_path,
        index=False,
    )

    def quote_info(
        row: pd.Series,
    ) -> dict:

        """Perform the quote info operation for this research workflow.

        Args:
            row: Input to the quote_info operation.

        Returns:
            Computed result or status for the caller.
        """
        date_value = row.get(
            "QuoteDate",
            None,
        )

        return {
            "ric": str(row["RIC"]),
            "strike": float(
                row["StrikePrice"]
            ),
            "expiry": str(
                pd.Timestamp(
                    row["ExpiryDate"]
                ).date()
            ),
            "bid": float(row["BID"]),
            "ask": float(row["ASK"]),
            "quote_source": str(
                row.get(
                    "QuoteSource",
                    "UNVERIFIED",
                )
            ),
            "quote_date": (
                pd.Timestamp(
                    date_value
                ).isoformat()
                if pd.notna(date_value)
                else None
            ),
        }

    audit = {
        "data_source": (
            "LSEG latest-available snapshot; "
            "quote freshness unverified"
        ),
        "retrieval_time_local": (
            timestamp.isoformat()
        ),
        "underlying_ric": args.ric,
        "spot": spot,
        "shares": args.shares,
        "hedge_horizon_sessions": (
            args.horizon_sessions
        ),
        "approximate_target_date": (
            str(target_expiry.date())
        ),
        "option_expiry": str(
            hedge.expiry
        ),
        "contract_multiplier": (
            args.multiplier
        ),
        "contract_terms_independently_verified": False,
        "put": quote_info(put),
        "call": quote_info(call),
        "liquid_candidate_count": (
            len(eligible)
        ),
        "research_fallback_enabled": (
            args.research_historical_fallback
        ),
        "executable_trade_recommendation": False,
        "limitations": [
            "Quotes may be delayed or stale.",
            "Put purchase uses ASK.",
            "Call sale uses BID.",
            "Contract deliverables require verification.",
            "ML forecasts are not yet live.",
            "Option expiry may differ from 63 sessions.",
            "Expiration-only payoff calculations.",
            "No dividends, financing, taxes or early exercise.",
        ],
    }

    audit_path.write_text(
        json.dumps(
            audit,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\n--- LSEG HEDGE STRATEGY COMPARISON ---"
    )

    print(
        f"Underlying: {args.ric}"
    )

    print(
        f"Spot price: {spot:.4f}"
    )

    print(
        f"Shares: {args.shares}"
    )

    print(
        f"Hedge horizon: "
        f"{args.horizon_sessions} sessions"
    )

    print(
        f"Approximate target date: "
        f"{target_expiry.date()}"
    )

    print(
        f"Selected option expiry: "
        f"{hedge.expiry}"
    )

    print(
        f"Covered shares: "
        f"{hedge.covered_shares}"
    )

    print(
        f"Liquid candidates: "
        f"{len(eligible)}"
    )

    print(
        f"\nPUT: {put['RIC']} | "
        f"strike={put['StrikePrice']} | "
        f"ASK={put['ASK']}"
    )

    print(
        f"CALL: {call['RIC']} | "
        f"strike={call['StrikePrice']} | "
        f"BID={call['BID']}"
    )

    grid = payoff.pivot(
        index="Stock Price Change",
        columns="Strategy",
        values="Total P&L",
    )

    grid = grid[
        [
            "unhedged",
            "protective_put",
            "collar",
        ]
    ]

    print(
        "\nExpiration P&L scenarios:"
    )

    print(
        grid.to_string(
            float_format=lambda x: f"{x:,.2f}"
        )
    )

    print(
        "\nPayoffs:",
        csv_path,
    )

    print(
        "Audit:",
        audit_path,
    )

    print(
        "\nRESEARCH ONLY: contract terms, "
        "quote freshness and execution "
        "have not been independently verified."
    )


if __name__ == "__main__":
    main()
