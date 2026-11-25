
"""Expiration-only stock, protective-put, and collar scenario comparisons.

The demo uses fictional quotes. Real historical forecasts may only be attached
when contemporaneous historical option quotes are supplied explicitly.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data/processed/hedge_comparisons/expiry_payoff_v1"
)

SHOCKS = (
    -1.00, -0.30, -0.20, -0.15, -0.10,
    -0.05, 0.00, 0.10, 0.20, 0.40,
)

RIC_ALIASES = {
    "NVDA.O": "NVDA.OQ",
}


@dataclass(frozen=True)
class HedgeInputs:
    """Represent HedgeInputs within the hedge research workflow.
    """
    ric: str
    as_of: date
    expiry: date
    shares: int
    spot: float
    put_strike: float
    put_ask: float
    call_strike: float
    call_bid: float
    source: str
    multiplier: int = 100
    fee_per_contract: float = 0.0

    def __post_init__(self) -> None:
        """Handle the internal post init step.
        """
        if self.source not in {
            "illustrative",
            "historical_snapshot",
            "live",
        }:
            raise ValueError("Invalid quote source")

        if not self.ric.strip():
            raise ValueError("RIC must not be blank")

        if self.expiry <= self.as_of:
            raise ValueError(
                "Option expiry must follow the quote date"
            )

        if self.shares < 1 or self.multiplier < 1:
            raise ValueError(
                "Shares and multiplier must be positive integers"
            )

        if self.shares < self.multiplier:
            raise ValueError(
                "At least one covered option contract is required"
            )

        prices = (
            self.spot,
            self.put_strike,
            self.put_ask,
            self.call_strike,
            self.call_bid,
            self.fee_per_contract,
        )

        if not all(
            math.isfinite(value)
            for value in prices
        ):
            raise ValueError(
                "All prices and fees must be finite"
            )

        if (
            self.spot <= 0
            or self.put_strike <= 0
            or self.call_strike <= 0
        ):
            raise ValueError(
                "Spot and strike prices must be positive"
            )

        if min(
            self.put_ask,
            self.call_bid,
            self.fee_per_contract,
        ) < 0:
            raise ValueError(
                "Premiums and fees must be nonnegative"
            )

        if self.put_strike >= self.call_strike:
            raise ValueError(
                "Collar put strike must be below call strike"
            )

    @property
    def contracts(self) -> int:
        """Perform the contracts operation for this research workflow.

        Returns:
            Computed result or status for the caller.
        """
        return self.shares // self.multiplier

    @property
    def covered_shares(self) -> int:
        """Perform the covered shares operation for this research workflow.

        Returns:
            Computed result or status for the caller.
        """
        return self.contracts * self.multiplier

    @property
    def uncovered_shares(self) -> int:
        """Perform the uncovered shares operation for this research workflow.

        Returns:
            Computed result or status for the caller.
        """
        return self.shares - self.covered_shares


def expiration_scenarios(
    inputs: HedgeInputs,
) -> pd.DataFrame:
    """Compute intrinsic option payoffs at expiration."""

    n = inputs.shares
    q = inputs.covered_shares
    n_contracts = inputs.contracts

    put_premium = q * inputs.put_ask
    call_premium = q * inputs.call_bid

    put_fees = (
        n_contracts * inputs.fee_per_contract
    )

    collar_fees = 2 * put_fees

    initial_stock_value = n * inputs.spot

    rows = []

    for change in SHOCKS:
        terminal = max(
            0.0,
            inputs.spot * (1.0 + change),
        )

        stock_pl = (
            n * (terminal - inputs.spot)
        )

        put_payoff = (
            q * max(
                inputs.put_strike - terminal,
                0.0,
            )
        )

        call_liability = (
            q * max(
                terminal - inputs.call_strike,
                0.0,
            )
        )

        variants = {
            "unhedged": (
                0.0,
                0.0,
                0.0,
            ),

            "protective_put": (
                put_payoff,
                -put_premium,
                -put_fees,
            ),

            "collar": (
                put_payoff - call_liability,
                call_premium - put_premium,
                -collar_fees,
            ),
        }

        for strategy, (
            derivative_payoff,
            option_premium_cashflow,
            fees,
        ) in variants.items():

            pl = (
                stock_pl
                + derivative_payoff
                + option_premium_cashflow
                + fees
            )

            rows.append({
                "Stock Price Change": change,
                "Expiration Stock Price": terminal,
                "Strategy": strategy,
                "Stock P&L": stock_pl,
                "Derivative Expiration Payoff": (
                    derivative_payoff
                ),
                "Initial Option Premium Cashflow": (
                    option_premium_cashflow
                ),
                "Option Transaction Fees": fees,
                "Total P&L": pl,
                "P&L / Initial Stock Value": (
                    pl / initial_stock_value
                ),
            })

    return pd.DataFrame(rows)


def load_historical_forecast(
    path: Path,
    inputs: HedgeInputs,
) -> dict:
    """Only attach forecasts to same-date historical quotes."""

    if inputs.source != "historical_snapshot":
        raise ValueError(
            "A historical ML forecast requires "
            "historical_snapshot option quotes. "
            "Do not combine historical predictions "
            "with live or illustrative quotes."
        )

    if not path.is_file():
        raise FileNotFoundError(path)

    frame = pd.read_csv(path)

    required = {
        "Decision Date",
        "Underlying RIC",
        "pred_max_loss_from_entry_63d",
        "pred_max_drawdown_63d",
        "pred_realized_vol_63d",
        "prob_loss_ge_10pct_63d",
        "prob_loss_ge_15pct_63d",
        "prob_loss_ge_20pct_63d",
    }

    missing = required - set(frame.columns)

    if missing:
        raise ValueError(
            f"Forecast file missing columns: "
            f"{sorted(missing)}"
        )

    requested = RIC_ALIASES.get(
        inputs.ric.upper(),
        inputs.ric.upper(),
    )

    dates = pd.to_datetime(
        frame["Decision Date"],
        errors="raise",
    ).dt.date

    matching = frame.loc[
        dates.eq(inputs.as_of)
        & frame["Underlying RIC"]
        .astype(str)
        .str.upper()
        .eq(requested)
    ]

    if len(matching) != 1:
        raise ValueError(
            f"Expected one forecast for {requested} "
            f"on {inputs.as_of}; found {len(matching)}"
        )

    record = matching.iloc[0]

    forecast = {}

    for column in sorted(
        required - {
            "Decision Date",
            "Underlying RIC",
        }
    ):
        value = float(record[column])

        if not math.isfinite(value):
            raise ValueError(
                f"Invalid forecast: {column}"
            )

        if (
            column.startswith("prob_")
            and not (0 <= value <= 1)
        ):
            raise ValueError(
                f"Invalid probability: {column}"
            )

        forecast[column] = value

    return forecast


def get_arguments() -> argparse.Namespace:
    """Get arguments from the current context.

    Returns:
        Computed result or status for the caller.
    """
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--demo",
        action="store_true",
    )

    parser.add_argument("--ric")

    parser.add_argument(
        "--as-of",
        type=date.fromisoformat,
    )

    parser.add_argument(
        "--expiry",
        type=date.fromisoformat,
    )

    parser.add_argument(
        "--shares",
        type=int,
    )

    parser.add_argument(
        "--spot",
        type=float,
    )

    parser.add_argument(
        "--put-strike",
        type=float,
    )

    parser.add_argument(
        "--put-ask",
        type=float,
    )

    parser.add_argument(
        "--call-strike",
        type=float,
    )

    parser.add_argument(
        "--call-bid",
        type=float,
    )

    parser.add_argument(
        "--multiplier",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--fee-per-contract",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--quote-source",
        choices=(
            "illustrative",
            "historical_snapshot",
            "live",
        ),
        default="illustrative",
    )

    parser.add_argument(
        "--forecast-csv",
        type=Path,
    )

    return parser.parse_args()


def main() -> None:
    """Run the module entry point and its workflow.
    """
    args = get_arguments()

    if args.demo:
        if args.forecast_csv is not None:
            raise ValueError(
                "Demo quotes cannot be combined "
                "with historical ML predictions"
            )

        inputs = HedgeInputs(
            ric="DEMO",
            as_of=date(2024, 12, 2),
            expiry=date(2025, 3, 7),
            shares=1000,
            spot=100.0,
            put_strike=90.0,
            put_ask=3.0,
            call_strike=110.0,
            call_bid=1.5,
            source="illustrative",
            multiplier=100,
            fee_per_contract=0.65,
        )

    else:
        required = {
            "ric": args.ric,
            "as_of": args.as_of,
            "expiry": args.expiry,
            "shares": args.shares,
            "spot": args.spot,
            "put_strike": args.put_strike,
            "put_ask": args.put_ask,
            "call_strike": args.call_strike,
            "call_bid": args.call_bid,
        }

        missing = [
            key
            for key, value in required.items()
            if value is None
        ]

        if missing:
            raise ValueError(
                f"Missing required arguments: {missing}"
            )

        inputs = HedgeInputs(
            ric=args.ric,
            as_of=args.as_of,
            expiry=args.expiry,
            shares=args.shares,
            spot=args.spot,
            put_strike=args.put_strike,
            put_ask=args.put_ask,
            call_strike=args.call_strike,
            call_bid=args.call_bid,
            source=args.quote_source,
            multiplier=args.multiplier,
            fee_per_contract=args.fee_per_contract,
        )

    forecast = None

    if args.forecast_csv is not None:
        forecast = load_historical_forecast(
            args.forecast_csv,
            inputs,
        )

    table = expiration_scenarios(inputs)

    safe_ric = re.sub(
        r"[^A-Za-z0-9_.-]",
        "_",
        inputs.ric,
    )

    basename = (
        f"{safe_ric}_{inputs.as_of}_{inputs.expiry}"
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_path = (
        OUTPUT_DIR / f"{basename}.csv"
    )

    json_path = (
        OUTPUT_DIR / f"{basename}_audit.json"
    )

    table.to_csv(
        csv_path,
        index=False,
        encoding="utf-8-sig",
    )

    manifest = {
        "ric": inputs.ric,
        "quote_date": inputs.as_of.isoformat(),
        "option_expiry": inputs.expiry.isoformat(),
        "quote_source": inputs.source,
        "spot": inputs.spot,
        "shares": inputs.shares,
        "option_contracts_per_leg": (
            inputs.contracts
        ),
        "contract_multiplier": inputs.multiplier,
        "covered_shares": inputs.covered_shares,
        "uncovered_shares": inputs.uncovered_shares,
        "put_strike": inputs.put_strike,
        "put_ask": inputs.put_ask,
        "call_strike": inputs.call_strike,
        "call_bid": inputs.call_bid,
        "fee_per_contract": inputs.fee_per_contract,
        "ml_forecast_63_sessions": forecast,
        "limitations": [
            "Option payouts are modeled at expiration only.",
            "No interim option mark-to-market or early exercise.",
            "No dividends, taxes, interest, or stock trading costs.",
            "Premiums are per underlying share; fees are per contract.",
            "The 63-session ML horizon may differ from option expiry.",
            "Historical quote provenance is user-supplied, not externally verified.",
            "ML probabilities are research estimates, not calibrated hedge P&L.",
        ],
    }

    json_path.write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\n--- EXPIRATION HEDGE PAYOFF COMPARISON ---"
    )

    print(
        f"RIC: {inputs.ric} | "
        f"as of: {inputs.as_of} | "
        f"expiry: {inputs.expiry}"
    )

    print(
        f"Quote source: {inputs.source}"
    )

    print(
        f"Shares: {inputs.shares} | "
        f"contracts/leg: {inputs.contracts}"
    )

    print(
        f"Covered shares: {inputs.covered_shares} | "
        f"uncovered: {inputs.uncovered_shares}"
    )

    print(
        f"Put: strike {inputs.put_strike}, "
        f"ask {inputs.put_ask}"
    )

    print(
        f"Call: strike {inputs.call_strike}, "
        f"bid {inputs.call_bid}"
    )

    if forecast is not None:
        print(
            "\nSeparate 63-session research forecasts "
            "(not expiry P&L):"
        )

        for name, value in forecast.items():
            print(
                f"  {name}: {value:.4f}"
            )

    grid = table.pivot(
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
        "\nExpiration P&L scenarios "
        "(native quote currency; no FX conversion):"
    )

    print(
        grid.to_string(
            float_format=lambda v: f"{v:,.2f}"
        )
    )

    print(
        "\nScenario CSV:",
        csv_path,
    )

    print(
        "Audit JSON:",
        json_path,
    )

    print(
        "Not a historical backtest or a trade recommendation."
    )


if __name__ == "__main__":
    main()
