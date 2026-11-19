"""Backtest candidate inventories without modifying fixed-strike runs.

Run from the project root:
    py -m pipeline.historical_candidate_backtest --inventory <parquet>

This module deliberately does not open LSEG or download quotes. Missing histories
are recorded in coverage; use the existing authenticated quote pipeline to
populate the cache first.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ml.strategy_paths import build_historical_strategy_path
from ml.transaction_costs import TransactionCostConfig, apply_transaction_costs_to_path
from pipeline.historical_data_pipeline import load_price_history
from pipeline.v2_stock_bridge import load_v2_stock_window
from pipeline.historical_option_pipeline import load_option_quote_history
from pipeline.historical_corporate_actions import (
    normalize_historical_stock_prices,
    get_aapl_historical_split_factor,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

LEG_FIELDS = {
    "Long Put": ("Long Put RIC", "Long Put Strike", "Long Put Ask", 1),
    "Short Call": ("Short Call RIC", "Short Call Strike", "Short Call Bid", -1),
    "Short Put": ("Short Put RIC", "Short Put Strike", "Short Put Bid", -1),
}
STRATEGY_LEGS = {
    "Protective Put": ("Long Put",),
    "Collar": ("Long Put", "Short Call"),
    "Put Spread": ("Long Put", "Short Put"),
}
REQUIRED = {
    "Underlying RIC", "Decision Date", "Outcome Date", "Hedge Horizon Days",
    "Candidate ID", "Strategy", "Expiry", "Shares", "Spot Price",
    "Long Put RIC", "Long Put Strike", "Long Put Ask",
    "Short Call RIC", "Short Call Strike", "Short Call Bid",
    "Short Put RIC", "Short Put Strike", "Short Put Bid",
    "Net Premium Per Share",
}


def _date(value):
    value = pd.Timestamp(value).normalize()
    if pd.isna(value):
        raise ValueError("Missing or invalid date")
    return value


def _selection_for_candidate(row):
    """Convert one candidate row to the selection schema expected by existing modules."""
    strategy = str(row["Strategy"])
    if strategy not in STRATEGY_LEGS:
        raise ValueError(f"Unsupported strategy: {strategy}")
    start, end, expiry = map(_date, (row["Decision Date"], row["Outcome Date"], row["Expiry"]))
    if not (start < end < expiry):
        raise ValueError("Expected Decision Date < Outcome Date < Expiry")
    horizon = int(row["Hedge Horizon Days"])
    if horizon < 1 or float(row["Hedge Horizon Days"]) != horizon:
        raise ValueError("Invalid hedge horizon")
    spot = float(row["Spot Price"])
    if not np.isfinite(spot) or spot <= 0:
        raise ValueError("Invalid spot price")

    legs = {}
    strikes = {}
    for leg in STRATEGY_LEGS[strategy]:
        ric_col, strike_col, price_col, side = LEG_FIELDS[leg]
        ric = row[ric_col]
        if pd.isna(ric) or not str(ric).strip():
            raise ValueError(f"Missing {leg} RIC")
        strike, price = float(row[strike_col]), float(row[price_col])
        if not (np.isfinite(strike) and strike > 0 and np.isfinite(price) and price > 0):
            raise ValueError(f"Invalid {leg} strike/entry quote")
        strikes[leg] = strike
        # Include entry bid or ask for the existing transaction-cost function.
        contract = {"RIC": str(ric), "Strike": strike, "Expiry": expiry}
        contract["ASK" if side == 1 else "BID"] = price
        legs[leg] = contract

    if strategy == "Collar" and not (strikes["Long Put"] < strikes["Short Call"]):
        raise ValueError("Invalid collar strikes")
    if strategy == "Put Spread" and not (strikes["Short Put"] < strikes["Long Put"]):
        raise ValueError("Invalid put spread strikes")

    # Ensure unwanted legs are not silently included or mislabeled.
    expected_rics = set(STRATEGY_LEGS[strategy])
    for leg, (ric_col, _, _, _) in LEG_FIELDS.items():
        if leg not in expected_rics and pd.notna(row[ric_col]):
            raise ValueError(f"Unexpected {leg} on {strategy}")

    return {
        "Underlying RIC": str(row["Underlying RIC"]),
        "Decision Date": start,
        "Exit Date": end,
        "Hedge Horizon Days": horizon,
        "Stock Price": spot,
        "Strategies": {strategy: legs},
    }


def _enrich_selection_from_cache(selection, quote_cache, max_entry_difference_pct=0.05):
    """Supply complete decision-date BID/ASK, validate entry source against cache."""
    start = selection["Decision Date"]
    legs = next(iter(selection["Strategies"].values()))
    problems = []
    for leg, contract in legs.items():
        ric = contract["RIC"]
        if ric not in quote_cache:
            try:
                quotes = load_option_quote_history(ric).copy()
                quotes.index = pd.to_datetime(quotes.index).normalize()
                if quotes.index.has_duplicates or not {"BID", "ASK"}.issubset(quotes.columns):
                    raise ValueError("Duplicate dates or missing BID/ASK columns")
                quote_cache[ric] = quotes
            except (FileNotFoundError, ValueError, KeyError, TypeError) as exc:
                quote_cache[ric] = exc
        cached = quote_cache[ric]
        if isinstance(cached, Exception):
            problems.append(f"{leg} {ric}: {cached}")
            continue
        if start not in cached.index:
            problems.append(f"{leg} {ric}: entry quote missing on {start.date()}")
            continue
        bid, ask = float(cached.loc[start, "BID"]), float(cached.loc[start, "ASK"])
        if not (np.isfinite(bid) and np.isfinite(ask) and 0 <= bid <= ask and ask > 0):
            problems.append(f"{leg} {ric}: invalid entry BID/ASK")
            continue
        is_long = leg == "Long Put"
        cached_entry = ask if is_long else bid
        inventory_entry = float(contract["ASK" if is_long else "BID"])
        if abs(cached_entry - inventory_entry) > max(0.01, abs(inventory_entry) * max_entry_difference_pct):
            problems.append(
                f"{leg} {ric}: cached entry {cached_entry:.4f} differs "
                f"from discovery {inventory_entry:.4f}"
            )
            continue
        contract["BID"] = bid
        contract["ASK"] = ask
        midpoint = (bid + ask) / 2
        contract["SPREAD_PCT"] = (ask - bid) / midpoint if midpoint > 0 else np.nan
    return problems


def backtest_candidates(inventory, option_multiplier=100, cost_config=None,
                        max_candidates=None, price_source="legacy"):
    """Return (complete outcomes, per-candidate coverage) using existing backtest engines."""
    if cost_config is None:
        cost_config = TransactionCostConfig()
    missing = REQUIRED - set(inventory.columns)
    if missing:
        raise ValueError(f"Missing inventory columns: {sorted(missing)}")
    if inventory.empty:
        raise ValueError("Candidate inventory is empty")
    if not isinstance(option_multiplier, int) or option_multiplier <= 0:
        raise ValueError("option_multiplier must be a positive integer")
    if price_source not in ("legacy", "v2"):
        raise ValueError("price_source must be legacy or v2")

    records = []
    coverage = []
    items = inventory.copy()
    for col in ("Decision Date", "Outcome Date", "Expiry"):
        items[col] = pd.to_datetime(items[col]).dt.normalize()
    unique_cols = ["Underlying RIC", "Decision Date", "Hedge Horizon Days", "Candidate ID"]
    if items.duplicated(unique_cols).any():
        raise ValueError("Duplicate candidate keys")
    items = items.sort_values(unique_cols, kind="mergesort")
    if max_candidates is not None:
        if max_candidates <= 0:
            raise ValueError("max_candidates must be positive")
        items = items.head(max_candidates)

    # One stock history per underlying rather than repeated disk reads for the benchmark.
    stock_cache = {}
    quote_cache = {}
    for _, row in items.iterrows():
        key = {c: row[c] for c in unique_cols}
        key["Strategy"] = row["Strategy"]
        try:
            selection = _selection_for_candidate(row)
            ric = selection["Underlying RIC"]
            shares = int(row["Shares"])
            if shares < 1 or float(row["Shares"]) != shares or shares % option_multiplier:
                raise ValueError("Shares must be a positive whole-contract quantity")
            start, end = selection["Decision Date"], selection["Exit Date"]
            if price_source == "v2":
                # Exact RIC and historical trading basis; no alias substitution.
                window_key = (ric, start, end, float(selection["Stock Price"]),
                              int(selection["Hedge Horizon Days"]))
                if window_key not in stock_cache:
                    stock_cache[window_key] = load_v2_stock_window(
                        ric,
                        start,
                        end,
                        hedge_horizon_days=selection["Hedge Horizon Days"],
                        expected_spot=selection["Stock Price"],
                    )
                stock_window = stock_cache[window_key]
            else:
                # Legacy calculations retained exactly for old experiments.
                if ric not in stock_cache:
                    stock = load_price_history(ric)["Close"].copy()
                    stock.index = pd.to_datetime(stock.index).normalize()
                    if stock.index.has_duplicates:
                        raise ValueError("Duplicate stock price dates")
                    stock_cache[ric] = stock.sort_index()
                stock = stock_cache[ric]
                stock_window = stock.loc[start:end].astype(float)
                if ric == "AAPL.O":
                    if (
                        get_aapl_historical_split_factor(start)
                        != get_aapl_historical_split_factor(end)
                    ):
                        raise ValueError(
                            "Stock split inside holding period: "
                            "split-aware position accounting required."
                        )
                    stock_window = normalize_historical_stock_prices(
                        prices=stock_window, underlying_ric=ric,
                    )

            expected = selection["Hedge Horizon Days"] + 1
            if (len(stock_window) != expected or stock_window.index[0] != start
                    or stock_window.index[-1] != end):
                raise ValueError("Incomplete stock window")
            spot = float(stock_window.iloc[0])
            if not np.isclose(spot, selection["Stock Price"], rtol=1e-5, atol=1e-4):
                raise ValueError("Inventory spot differs from cached stock history")
            if not np.isfinite(stock_window).all() or (stock_window <= 0).any():
                raise ValueError("Invalid stock values")
            if problems := _enrich_selection_from_cache(selection, quote_cache):
                raise FileNotFoundError("; ".join(problems))

            strategy = str(row["Strategy"])
            path, summary = build_historical_strategy_path(
                selection=selection, strategy_name=strategy,
                shares=shares, option_multiplier=option_multiplier,
                stock_window_override=(stock_window if price_source == "v2" else None),
            )
            if len(path) != expected or summary["Observations"] != expected:
                raise ValueError("Unexpected number of path observations")
            path, summary = apply_transaction_costs_to_path(
                path_df=path, summary=summary, selection=selection,
                strategy_name=strategy, config=cost_config, shares=shares,
                option_multiplier=option_multiplier, portfolio_value_column="Portfolio Value",
            )
            prices = stock_window.to_numpy(dtype=float)
            relative = prices / prices[0]
            unhedged_mdd = float(np.min(relative / np.maximum.accumulate(relative) - 1.0))
            unhedged_return = float(relative[-1] - 1.0)
            final_return = float(summary["Final Return"])
            mdd = float(summary["Maximum Drawdown"])
            output = row.to_dict()
            output.update({
                "Option Expiry": selection["Strategies"][strategy]["Long Put"]["Expiry"],
                "Days to Expiry": int((row["Expiry"] - row["Decision Date"]).days),
                "Final Return": final_return,
                "Maximum Drawdown": mdd,
                "Final P&L": float(summary["Final P&L"]),
                "Unhedged Final Return": unhedged_return,
                "Unhedged Maximum Drawdown": unhedged_mdd,
                "Hedge Improvement": final_return - unhedged_return,
                "Drawdown Improvement": mdd - unhedged_mdd,
                "Observations": expected,
            })
            for field in (
                "Gross Final Return", "Gross Maximum Drawdown", "Gross Final P&L",
                "Total Transaction Costs", "Option Commissions", "Option Fixed Fees",
                "Option Slippage", "Stock Commissions",
            ):
                output[field] = summary[field]
            records.append(output)
            coverage.append({**key, "Status": "Complete"})
        except (ValueError, FileNotFoundError, KeyError, TypeError, IndexError) as exc:
            coverage.append({**key, "Status": "Incomplete", "Reason": str(exc)})
    return pd.DataFrame(records), pd.DataFrame(coverage)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--option-multiplier", type=int, default=100)
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument("--price-source", choices=["legacy", "v2"], default="legacy")
    args = parser.parse_args()
    inventory_path = args.inventory if args.inventory.is_absolute() else PROJECT_ROOT / args.inventory
    outdir = args.output_dir
    if outdir is None:
        outdir = PROJECT_ROOT / "data" / "processed" / "candidate_backtests" / inventory_path.parent.name
    elif not outdir.is_absolute():
        outdir = PROJECT_ROOT / outdir
    if not inventory_path.is_file():
        parser.error(f"Inventory does not exist: {inventory_path}")
    df = pd.read_parquet(inventory_path)
    costs = TransactionCostConfig()
    results, coverage = backtest_candidates(
        df, option_multiplier=args.option_multiplier,
        cost_config=costs, max_candidates=args.max_candidates,
        price_source=args.price_source,
    )
    outdir.mkdir(parents=True, exist_ok=True)
    results.to_parquet(outdir / "candidate_ml_results.parquet", index=False)
    coverage.to_parquet(outdir / "candidate_coverage.parquet", index=False)
    with (outdir / "backtest_config.json").open("w", encoding="utf-8") as f:
        json.dump({"inventory": str(inventory_path), "option_multiplier": args.option_multiplier,
                   "max_candidates": args.max_candidates, "price_source": args.price_source,
                   "transaction_costs": asdict(costs)}, f, indent=2)
    print("\n--- CANDIDATE BACKTEST ---")
    print(f"Candidates evaluated: {len(coverage)}")
    print(f"Completed: {len(results)}")
    print(f"Incomplete: {len(coverage) - len(results)}")
    if not coverage.empty:
        print(coverage["Status"].value_counts().to_string())
    print(f"Saved results and coverage to: {outdir}")
    if len(results) == 0:
        print("No completed candidates. Inspect candidate_coverage.parquet for missing quote histories.")


if __name__ == "__main__":
    main()
