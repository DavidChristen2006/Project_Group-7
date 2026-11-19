
from pathlib import Path

import numpy as np
import pandas as pd
import json


from dataclasses import asdict

from pipeline.run_identifiers import (
    transaction_cost_tag
)

from pipeline.historical_data_pipeline import (
    load_price_history
)
from pipeline.historical_option_pipeline import (
    download_option_quote_history,
    load_option_quote_history
)
from pipeline.expired_option_pipeline import (
    discover_expired_options
)

from pipeline.market_calendar import (
    next_monthly_expiry
)
from ml.historical_strategy_selection import (
    select_historical_strategy
)
from ml.strategy_paths import build_historical_strategy_path
from pipeline.historical_discovery_cache import discover_expired_options_cached
from ml.transaction_costs import TransactionCostConfig, apply_transaction_costs_to_path
from pipeline.option_quote_retry_cache import get_missing_quote_dates, should_skip_quote_download, record_quote_download_attempt
from pipeline.historical_candidate_generation import CandidateConfig, generate_hedge_candidates
from pipeline.historical_strike_discovery import generate_extended_strikes
from pipeline.experiment_identifiers import build_historical_run_name
from pipeline.historical_expiry_discovery import (
    ExpiryConfig,
    generate_monthly_expiry_candidates,
)
from pipeline.historical_corporate_actions import (
    normalize_historical_stock_prices,
    get_aapl_historical_split_factor,
)



PROJECT_ROOT = Path(__file__).resolve().parent.parent

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

STRATEGY_NAMES = [
    "Protective Put",
    "Collar",
    "Put Spread"
]

LEG_SIDES = {
    "Long Put": 1,
    "Short Put": -1,
    "Short Call": -1
}


def generate_candidate_strikes(
    spot,
    strike_step=5.0,
    target_pcts=(0.80, 0.90, 1.10)
):
    if spot <= 0 or strike_step <= 0:
        raise ValueError("Invalid spot or strike step.")

    strikes = set()

    for target_pct in target_pcts:
        target = spot * target_pct
        central_strike = (
            round(target / strike_step) * strike_step
        )

        for offset in (-1, 0, 1):
            strike = central_strike + offset * strike_step

            if 0 < strike < 1000:
                strikes.add(round(strike, 2))

    return sorted(strikes)



def ensure_local_option_history(
    option_ric,
    start_date,
    end_date,
    required_dates,
    refresh=False,
    retry_after_days=7
):
    required_dates = pd.DatetimeIndex(
        pd.to_datetime(required_dates)
    ).normalize().unique().sort_values()

    existing = None

    try:
        existing = load_option_quote_history(
            option_ric
        )
    except FileNotFoundError:
        pass

    if existing is not None:
        existing = existing.copy()

        existing.index = pd.DatetimeIndex(
            pd.to_datetime(existing.index)
        ).normalize()

        missing = get_missing_quote_dates(
            existing,
            required_dates
        )

        if not missing:
            print(
                f"Using cached quotes: {option_ric}"
            )
            return existing

        skip = should_skip_quote_download(
            option_ric=option_ric,
            required_dates=required_dates,
            existing_quotes=existing,
            retry_after_days=retry_after_days,
            refresh=refresh
        )

        if skip:
            print(
                f"Skipping recently retried "
                f"incomplete quotes: {option_ric}"
            )
            print(
                f"  Missing observations: {len(missing)}"
            )
            return existing

    # Preserve the date buffer needed by the
    # historical LSEG data request.
    download_start = (
        pd.Timestamp(start_date)
        - pd.Timedelta(days=7)
    ).strftime("%Y-%m-%d")

    download_end = (
        pd.Timestamp(end_date)
        + pd.Timedelta(days=7)
    ).strftime("%Y-%m-%d")

    downloaded = download_option_quote_history(
        option_ric=option_ric,
        start_date=download_start,
        end_date=download_end
    )

    downloaded = downloaded.copy()

    downloaded.index = pd.DatetimeIndex(
        pd.to_datetime(downloaded.index)
    ).normalize()

    if existing is not None:
        combined = pd.concat(
            [existing, downloaded]
        )

        combined = combined[
            ~combined.index.duplicated(keep="last")
        ].sort_index()

    else:
        combined = downloaded.sort_index()

    # Preserve the original local quote-history
    # storage convention.
    safe_ric = (
        option_ric
        .replace(".", "_")
        .replace("/", "_")
    )

    quote_path = (
        PROJECT_ROOT
        / "data"
        / "raw"
        / "options"
        / f"{safe_ric}.parquet"
    )

    quote_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    combined.to_parquet(quote_path)

    record_quote_download_attempt(
        option_ric=option_ric,
        required_dates=required_dates,
        quote_df=combined
    )

    missing = get_missing_quote_dates(
        combined,
        required_dates
    )

    if missing:
        print(
            f"{option_ric}: "
            f"{len(missing)} required quotes missing"
        )

    return combined


def build_historical_ml_dataset(
    underlying_ric,
    decision_dates,
    hedge_horizons=(21, 63, 126),
    shares=1000,
    option_multiplier=100,
    max_spread_pct=0.10,
    max_strike_deviation_pct=0.05,
    transaction_cost_config=None,
    refresh_missing_quotes=False,
    export_candidates=False,
    candidate_config=None,
    strike_discovery_mode="fixed",

    minimum_strike_pct=0.65,
    maximum_strike_pct=1.25,
    strike_step=5.0,
    expiry_mode="single",
    expiry_config=None,
):

    if transaction_cost_config is None:
        transaction_cost_config = TransactionCostConfig()

    if candidate_config is None:
        candidate_config = CandidateConfig(
            max_spread_pct=max_spread_pct,
        )

    if expiry_config is None:
        expiry_config = ExpiryConfig(
            min_days_after_exit=1,
            max_days_after_exit=45,
            max_expiries=3,
        )

    if expiry_mode not in ("single", "multiple"):
        raise ValueError("Invalid expiry mode.")

    if expiry_mode == "multiple":
        if not export_candidates:
            raise ValueError(
                "Multiple expiry mode requires "
                "export_candidates=True."
            )

        if (
            expiry_config.max_days_after_exit
            > candidate_config.max_expiry_gap_days
        ):
            raise ValueError(
                "Expiry discovery window exceeds "
                "candidate expiry eligibility window."
            )


    if strike_discovery_mode not in ("fixed", "expanded"):
        raise ValueError("strike_discovery_mode must be 'fixed' or 'expanded'.")

    stock = load_price_history(underlying_ric).sort_index()

    stock.index = pd.to_datetime(stock.index).normalize()

    stock["Close"] = pd.to_numeric(
        stock["Close"],
        errors="coerce"
    )

    # The current corporate-action helper supports
    # AAPL observations from 2014-06-09 onward.
    if underlying_ric == "AAPL.O":
        stock = stock.loc[
            stock.index >= pd.Timestamp("2014-06-09")
        ].copy()

        stock["Close"] = normalize_historical_stock_prices(
            prices=stock["Close"],
            underlying_ric=underlying_ric,
        )

    results = []
    coverage = []

    candidate_inventory = []

    if candidate_config is None:
        candidate_config = CandidateConfig(
            max_spread_pct=max_spread_pct,
        )

    for decision_date in decision_dates:
        decision_date = pd.Timestamp(
            decision_date
        ).normalize()

        if decision_date not in stock.index:
            raise ValueError(
                f"Missing decision date: {decision_date.date()}"
            )

        start_position = stock.index.get_loc(decision_date)

        spot = float(stock.loc[decision_date, "Close"])

        for horizon in hedge_horizons:
            horizon = int(horizon)

            if horizon <= 0:
                raise ValueError("Invalid hedge horizon.")

            exit_position = start_position + horizon

            if exit_position >= len(stock):
                print(
                    f"Skipping {horizon} days: "
                    "insufficient future stock data."
                )
                continue

            exit_date = stock.index[exit_position]

            # A fixed-share, fixed-contract backtest cannot
            # safely cross a stock split without adjusting
            # quantities and contract terms.
            if underlying_ric == "AAPL.O":
                entry_factor = get_aapl_historical_split_factor(
                    decision_date
                )
                exit_factor = get_aapl_historical_split_factor(
                    exit_date
                )

                if entry_factor != exit_factor:
                    print(
                        f"Skipping {decision_date.date()}: "
                        "holding period crosses a stock split."
                    )
                    continue
                
            expiry = next_monthly_expiry(exit_date)
            
            stock_path = stock["Close"].iloc[
                start_position:exit_position + 1
            ].to_numpy(dtype=float)

            unhedged_values = stock_path / stock_path[0]

            unhedged_drawdowns = (
                unhedged_values
                / np.maximum.accumulate(unhedged_values)
                - 1.0
            )

            unhedged_final_return = float(
                unhedged_values[-1] - 1.0
            )

            unhedged_max_drawdown = float(
                unhedged_drawdowns.min()
            )

            print(
                f"\n--- {underlying_ric} | "
                f"{decision_date.date()} | "
                f"{horizon} trading days ---"
            )

            print(f"Spot: {spot:.2f}")
            print(f"Exit: {exit_date.date()}")
            print(f"Option expiry: {expiry.date()}")


            if strike_discovery_mode == "fixed":
                strikes = generate_candidate_strikes(
                    spot
                )

            else:
                strikes = generate_extended_strikes(
                    spot=spot,
                    minimum_strike_pct=minimum_strike_pct,
                    maximum_strike_pct=maximum_strike_pct,
                    strike_step=strike_step,
                )

            print(
                f"Discovery mode: {strike_discovery_mode}"
            )

            print(
                f"Requested strikes: {len(strikes)}"
            )


            discovered_df = discover_expired_options_cached(
                underlying_root=underlying_ric.split(".")[0],
                decision_date=decision_date,
                expiry=expiry,
                strikes=strikes,
                option_types=("put", "call"),
                max_spread_pct=max_spread_pct
            )



            if export_candidates:
                print("\n--- CANDIDATE GENERATION ---")

                # The original benchmark expiry remains
                # unchanged. Additional expiries are
                # considered only for candidate export.
                if expiry_mode == "single":
                    candidate_expiries = [expiry]

                else:
                    candidate_expiries = (
                        generate_monthly_expiry_candidates(
                            decision_date=decision_date,
                            exit_date=exit_date,
                            config=expiry_config,
                        )
                    )

                # Preserve the original expiry whenever
                # it satisfies the configured eligibility
                # window. Avoid duplicate expiry requests.
                candidate_expiries = sorted(set(
                    pd.Timestamp(value).normalize()
                    for value in candidate_expiries
                ))

                print(
                    "Candidate expiration dates:",
                    [
                        value.strftime("%Y-%m-%d")
                        for value in candidate_expiries
                    ],
                )

                discovered_chains = []

                for candidate_expiry in candidate_expiries:

                    if candidate_expiry == expiry:
                        chain = discovered_df.copy()

                    else:
                        chain = discover_expired_options_cached(
                            underlying_root=(
                                underlying_ric.split(".")[0]
                            ),
                            decision_date=decision_date,
                            expiry=candidate_expiry,
                            strikes=strikes,
                            option_types=("put", "call"),
                            max_spread_pct=max_spread_pct,
                        )

                    print(
                        f"Expiry {candidate_expiry.date()}: "
                        f"{len(chain)} discovered contracts"
                    )

                    if chain.empty:
                        continue

                    chain = chain.copy()

                    chain["Expiry"] = pd.to_datetime(
                        chain["Expiry"]
                    ).dt.normalize()

                    # Reject mismatched expiry data.
                    if not chain["Expiry"].eq(
                        candidate_expiry
                    ).all():
                        raise ValueError(
                            "Discovered contracts contain "
                            "an unexpected expiration date."
                        )

                    discovered_chains.append(chain)

                if discovered_chains:
                    candidate_options = pd.concat(
                        discovered_chains,
                        ignore_index=True,
                    )

                    # Detect unintended duplicate
                    # contract discoveries.
                    if candidate_options.duplicated(
                        ["RIC", "Expiry"]
                    ).any():
                        raise ValueError(
                            "Duplicate option contracts "
                            "across expiry discoveries."
                        )

                    candidates = generate_hedge_candidates(
                        options=candidate_options,
                        underlying_ric=underlying_ric,
                        spot_price=spot,
                        hedge_end_date=exit_date,
                        shares=shares,
                        option_multiplier=option_multiplier,
                        config=candidate_config,
                    )

                    if not candidates.empty:
                        candidates = candidates.copy()

                        candidates.insert(
                            1,
                            "Decision Date",
                            decision_date,
                        )

                        candidates.insert(
                            2,
                            "Outcome Date",
                            exit_date,
                        )

                        candidates.insert(
                            3,
                            "Hedge Horizon Days",
                            horizon,
                        )

                        candidates["Days to Expiry"] = (
                            candidates["Expiry"]
                            - decision_date
                        ).dt.days

                        candidates["Days After Exit"] = (
                            candidates["Expiry"]
                            - exit_date
                        ).dt.days

                        candidate_inventory.append(
                            candidates
                        )

                        print("\nGenerated candidates:")
                        print(
                            candidates["Strategy"]
                            .value_counts()
                            .to_string()
                        )

                        print(
                            "\nCandidates by expiry:"
                        )

                        print(
                            candidates.groupby(
                                "Expiry"
                            ).size().to_string()
                        )

                        print(
                            f"\nTotal candidates: "
                            f"{len(candidates)}"
                        )

                    else:
                        print(
                            "No eligible historical candidates."
                        )

                else:
                    print(
                        "No historical option contracts "
                        "discovered for eligible expiries."
                    )


            if discovered_df.empty:
                for strategy_name in STRATEGY_NAMES:
                    coverage.append({
                        "Decision Date": decision_date,
                        "Hedge Horizon Days": horizon,
                        "Strategy": strategy_name,
                        "Status": "No contracts discovered"
                    })
                continue

            selection = select_historical_strategy(
                underlying_ric=underlying_ric,
                discovered_df=discovered_df,
                decision_date=decision_date,
                hedge_horizon_days=horizon,
                max_strike_deviation_pct=max_strike_deviation_pct
            )

            # Diagnostic output only when no strategy qualifies.
            if all(
                legs is None
                for legs in selection["Strategies"].values()
            ):
                print("\n--- STRATEGY SELECTION DIAGNOSTICS ---")
                print(f"Underlying: {underlying_ric}")
                print(f"Decision date: {decision_date.date()}")
                print(f"Horizon: {horizon}")
                print(f"Stock price: {spot:.2f}")

                print("\nTarget strikes:")
                print(f"Long put: {spot * 0.90:.2f}")
                print(f"Short call: {spot * 1.10:.2f}")
                print(f"Short put: {spot * 0.80:.2f}")

                print("\nSelection constraints:")
                print(f"Maximum spread: {max_spread_pct:.2%}")
                print(
                    "Maximum strike deviation: "
                    f"{max_strike_deviation_pct:.2%}"
                )

                print("\nDiscovered option columns:")
                print(discovered_df.columns.tolist())

                print("\nAvailable historical option quotes:")
                print(discovered_df.to_string(index=False))

                print("\nSelected strategies:")
                for name, legs in selection["Strategies"].items():
                    print(f"{name}: {legs}")

            selected_contracts = {}

            for legs in selection["Strategies"].values():
                if legs is None:
                    continue

                for contract in legs.values():
                    selected_contracts[
                        contract["RIC"]
                    ] = contract

            required_dates = stock.index[
                start_position:exit_position + 1
            ]

            download_errors = {}

            for ric in selected_contracts:
                try:
                    ensure_local_option_history(
                        option_ric=ric,
                        start_date=decision_date,
                        end_date=exit_date,
                        required_dates=required_dates,
                        refresh=refresh_missing_quotes
                    )
                except Exception as exc:
                    download_errors[ric] = str(exc)
                    print(f"History unavailable: {ric}: {exc}")

            for strategy_name in STRATEGY_NAMES:
                legs = selection["Strategies"][strategy_name]

                if legs is None:
                    coverage.append({
                        "Decision Date": decision_date,
                        "Hedge Horizon Days": horizon,
                        "Strategy": strategy_name,
                        "Status": "No suitable contracts"
                    })
                    continue

                failed_rics = [
                    contract["RIC"]
                    for contract in legs.values()
                    if contract["RIC"] in download_errors
                ]

                if failed_rics:
                    coverage.append({
                        "Decision Date": decision_date,
                        "Hedge Horizon Days": horizon,
                        "Strategy": strategy_name,
                        "Status": "Download failed: "
                        + ", ".join(failed_rics)
                    })
                    continue

                try:
                    path_df, summary = (
                        build_historical_strategy_path(
                            selection=selection,
                            strategy_name=strategy_name,
                            shares=shares,
                            option_multiplier=option_multiplier
                        )
                    )

                    expected_observations = horizon + 1

                    if len(path_df) != expected_observations:
                        raise ValueError(
                            f"{strategy_name}: expected "
                            f"{expected_observations} observations, "
                            f"received {len(path_df)}."
                        )

                    if summary["Observations"] != expected_observations:
                        raise ValueError(
                            f"{strategy_name}: inconsistent path summary."
                        )

                    path_df, summary = apply_transaction_costs_to_path(
                        path_df=path_df,
                        summary=summary,
                        selection=selection,
                        strategy_name=strategy_name,
                        config=transaction_cost_config,
                        shares=shares,
                        option_multiplier=option_multiplier,
                        portfolio_value_column="Portfolio Value"
                    )

                except (ValueError, FileNotFoundError) as exc:
                    coverage.append({
                        "Decision Date": decision_date,
                        "Hedge Horizon Days": horizon,
                        "Strategy": strategy_name,
                        "Status": f"Incomplete path: {exc}"
                    })
                    continue

                # Entry features: available on the decision date.
                # Outcome columns: known only after the exit date.
                row = {
                    "Underlying RIC": underlying_ric,
                    "Decision Date": decision_date,
                    "Outcome Date": exit_date,
                    "Hedge Horizon Days": horizon,
                    "Strategy": strategy_name,
                    "Stock Price": spot,
                    "Option Expiry": expiry,
                    "Days to Expiry": (
                        expiry - decision_date
                    ).days,
                    "Final Return": summary["Final Return"],
                    "Maximum Drawdown": (
                        summary["Maximum Drawdown"]
                    ),
                    "Final P&L": summary["Final P&L"],

                    "Gross Final Return": summary["Gross Final Return"],
                    "Gross Maximum Drawdown": summary["Gross Maximum Drawdown"],
                    "Gross Final P&L": summary["Gross Final P&L"],
                    "Total Transaction Costs": summary["Total Transaction Costs"],
                    "Option Commissions": summary["Option Commissions"],
                    "Option Fixed Fees": summary["Option Fixed Fees"],
                    "Option Slippage": summary["Option Slippage"],
                    "Stock Commissions": summary["Stock Commissions"],

                    "Unhedged Final Return": unhedged_final_return,
                    "Unhedged Maximum Drawdown": unhedged_max_drawdown,
                    "Hedge Improvement": (
                        summary["Final Return"] - unhedged_final_return
                    ),
                    "Drawdown Improvement": (
                        summary["Maximum Drawdown"] - unhedged_max_drawdown
                    ),
                    "Observations": summary["Observations"]
                }

                net_premium = 0.0

                for leg_name, contract in legs.items():
                    side = LEG_SIDES[leg_name]

                    entry_price = (
                        contract["ASK"]
                        if side == 1
                        else contract["BID"]
                    )

                    net_premium += side * entry_price

                    prefix = leg_name.replace(" ", "_")

                    row[f"{prefix}_RIC"] = contract["RIC"]
                    row[f"{prefix}_Strike"] = contract["Strike"]
                    row[f"{prefix}_Entry_Price"] = entry_price
                    row[f"{prefix}_Spread_Pct"] = (
                        contract["SPREAD_PCT"]
                    )

                row["Net Premium Per Share"] = net_premium
                row["Net Premium Pct"] = net_premium / spot

                results.append(row)

                coverage.append({
                    "Decision Date": decision_date,
                    "Hedge Horizon Days": horizon,
                    "Strategy": strategy_name,
                    "Status": "Complete"
                })

                print(
                    f"{strategy_name}: "
                    f"return {summary['Final Return']:.2%}, "
                    f"drawdown {summary['Maximum Drawdown']:.2%}"
                )

    results_df = pd.DataFrame(results)
    coverage_df = pd.DataFrame(coverage)

    
    # Save each configuration in its own run directory.
    dates = pd.to_datetime(decision_dates).normalize()
    horizons = sorted(set(int(h) for h in hedge_horizons))


    run_name = build_historical_run_name(
        underlying_ric=underlying_ric,
        decision_dates=decision_dates,
        hedge_horizons=hedge_horizons,
        shares=shares,
        option_multiplier=option_multiplier,
        max_spread_pct=max_spread_pct,
        max_strike_deviation_pct=(
            max_strike_deviation_pct
        ),
        transaction_cost_config=(
            transaction_cost_config
        ),
        candidate_config=candidate_config,
        export_candidates=export_candidates,
        strike_discovery_mode=strike_discovery_mode,
        minimum_strike_pct=minimum_strike_pct,
        maximum_strike_pct=maximum_strike_pct,
        strike_step=strike_step,
        expiry_mode=expiry_mode,
        expiry_config=expiry_config,
    )


    run_dir = OUTPUT_DIR / "historical_runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    if export_candidates:
        if candidate_inventory:
            candidate_df = pd.concat(
                candidate_inventory,
                ignore_index=True,
            )

            identity_columns = [
                "Underlying RIC",
                "Decision Date",
                "Hedge Horizon Days",
                "Candidate ID",
            ]

            if candidate_df.duplicated(
                identity_columns
            ).any():
                raise ValueError(
                    "Duplicate historical candidate IDs."
                )

            candidate_df = candidate_df.sort_values(
                [
                    "Decision Date",
                    "Hedge Horizon Days",
                    "Strategy",
                    "Candidate ID",
                ]
            ).reset_index(drop=True)

            candidate_path = (
                OUTPUT_DIR
                / "candidate_inventories"
                / run_name
                / "historical_candidates.parquet"
            )

            candidate_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            candidate_df.to_parquet(
                candidate_path,
                index=False,
            )

            print(
                f"\nHistorical candidates saved to: "
                f"{candidate_path}"
            )

            print(
                f"Candidate observations: "
                f"{len(candidate_df)}"
            )

        else:
            print(
                "\nNo historical candidates to save."
            )


    if not results_df.empty:
        results_df = results_df.sort_values(
            ["Decision Date", "Hedge Horizon Days", "Strategy"]
        ).reset_index(drop=True)

        results_df.to_parquet(
            run_dir / "historical_ml_results.parquet",
            index=False
        )

    coverage_df.to_parquet(
        run_dir / "historical_ml_coverage.parquet",
        index=False
    )

    configuration = {
        "underlying_ric": underlying_ric,
        "decision_dates": [
            str(pd.Timestamp(d).date())
            for d in decision_dates
        ],
        "hedge_horizons": horizons,
        "shares": shares,
        "option_multiplier": option_multiplier,
        "max_spread_pct": max_spread_pct,
        "max_strike_deviation_pct": max_strike_deviation_pct,
        "strike_target_pcts": {
            "long_put": 0.90,
            "short_call": 1.10,
            "short_put": 0.80},
        "strike_discovery_mode": strike_discovery_mode,
        "minimum_strike_pct": minimum_strike_pct,
        "maximum_strike_pct": maximum_strike_pct,
        "strike_step": strike_step,

        "candidate_export_enabled": export_candidates,
        "candidate_configuration": (
            asdict(candidate_config)
            if export_candidates else None
        ),
        "transaction_costs": asdict(transaction_cost_config),
        "expiry_mode": expiry_mode,
        "expiry_configuration": asdict(expiry_config),
    }

    with open(
        run_dir / "config.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(configuration, file, indent=4)

    print(f"\nHistorical run saved to: {run_dir}")

    return results_df, coverage_df

