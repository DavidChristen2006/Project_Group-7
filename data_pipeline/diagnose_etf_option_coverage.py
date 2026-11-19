
"""Diagnose ETF option discovery, quote selection and Greeks.

Run from project root:
    py -m data_pipeline.diagnose_etf_option_coverage

This script only retrieves data. It never places orders.
"""

from __future__ import annotations

import traceback
from datetime import date
import argparse

import lseg.data as ld

from lseg_api.instrument_resolver import (
    resolve_lseg_ric,
)

from lseg_api.equity import (
    get_current_equity_price,
)

from lseg_api.etf_option_discovery import (
    discover_etf_option_chain,
    option_search_keys,
)

from data_pipeline.etf_option_cross_hedges import (
    retrieve_etf_option_structures,
)

from data_pipeline.etf_option_deltas import (
    retrieve_structure_option_deltas,
)


TICKERS = ("SPY", "QQQ", "VTI")
TARGET_DATE = date(2027, 2, 9)


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "tickers",
        nargs="+",
        help="ETF tickers or full LSEG RICs",
    )

    parser.add_argument(
        "--target-date",
        type=date.fromisoformat,
        required=True,
        help="Research target date, YYYY-MM-DD",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    tickers = args.tickers
    target_date = args.target_date

    # Resolve tickers before opening our data session.
    resolved = {}

    for ticker in TICKERS:
        try:
            resolved[ticker] = resolve_lseg_ric(ticker)
        except Exception as exc:
            print(
                f"{ticker}: ticker resolution failed: {exc}"
            )

    ld.open_session()

    try:
        for ticker, ric in resolved.items():

            print("\n" + "=" * 65)
            print(f"ETF: {ticker} | PRICE RIC: {ric}")
            print("=" * 65)

            print(
                "Option search keys:",
                option_search_keys(ric),
            )

            try:
                spot = float(
                    get_current_equity_price(ric)
                )
            except Exception as exc:
                print(
                    "PRICE ERROR:",
                    type(exc).__name__,
                    str(exc),
                )
                continue

            print("ETF price:", spot)

            print("\n[1] DERIVATIVE DISCOVERY")

            try:
                chain = discover_etf_option_chain(
                    etf_ric=ric,
                    etf_spot=spot,
                    target_date=TARGET_DATE,
                    max_days_after=35,
                )

                print("Valid contracts:", len(chain))

                if not chain.empty:
                    print(
                        chain.groupby(
                            [
                                "ExpiryDate",
                                "CallPutOption",
                            ]
                        ).size().head(20)
                    )

            except Exception as exc:
                print(
                    "DISCOVERY ERROR:",
                    type(exc).__name__,
                    str(exc),
                )

            print("\n[2] COMPLETE OPTION SELECTION")

            try:
                structures = retrieve_etf_option_structures(
                    etf_ric=ric,
                    etf_spot=spot,
                    target_date=TARGET_DATE,
                    max_spread=0.10,
                    assume_standard_contracts=True,
                )

                for name, structure in structures.items():
                    print(f"\n{name}")

                    for leg in structure.legs:
                        print(
                            " ",
                            "BUY" if leg.side == 1 else "SELL",
                            leg.right,
                            leg.ric,
                            "strike:",
                            leg.strike,
                            "premium:",
                            leg.premium,
                            "expiry:",
                            leg.expiry,
                        )

            except Exception as exc:
                print(
                    "SELECTION ERROR:",
                    type(exc).__name__,
                    str(exc),
                )
                continue

            print("\n[3] LSEG GREEKS")

            try:
                net_deltas, leg_deltas = (
                    retrieve_structure_option_deltas(
                        structures
                    )
                )

                print("Leg deltas:", leg_deltas)
                print("Net deltas:", net_deltas)

            except Exception as exc:
                print(
                    "GREEKS ERROR:",
                    type(exc).__name__,
                    str(exc),
                )
                traceback.print_exc()

    finally:
        ld.close_session()


if __name__ == "__main__":
    main()
