
import pandas as pd
import lseg.data as ld

from pipeline.expired_option_pipeline import (
    build_expired_opra_ric,
    validate_expired_option_candidate,
)

TESTS = [
    ("2019-12-02", "2020-02-21", 66.04),
    ("2018-12-03", "2019-02-15", 46.205),
]

SPLIT_FACTOR = 4
MAX_SPREAD_PCT = 0.10


def make_strikes(adjusted_price):
    historical_spot = adjusted_price * SPLIT_FACTOR

    def nearest_five(value):
        return round(value / 5) * 5

    return historical_spot, [
        ("put", nearest_five(historical_spot * 0.90)),
        ("put", nearest_five(historical_spot * 0.95)),
        ("call", nearest_five(historical_spot * 1.05)),
        ("call", nearest_five(historical_spot * 1.10)),
    ]


def main():
    rows = []

    ld.open_session()

    try:
        for decision_date, expiry, adjusted_price in TESTS:
            historical_spot, contracts = make_strikes(
                adjusted_price
            )

            print(f"\n--- {decision_date} ---")
            print(f"Split-adjusted price: {adjusted_price:.4f}")
            print(f"Estimated pre-split price: {historical_spot:.4f}")

            for option_type, strike in contracts:
                ric = build_expired_opra_ric(
                    underlying_root="AAPL",
                    expiry=expiry,
                    strike=strike,
                    option_type=option_type,
                )

                print(f"\nChecking {ric}")

                try:
                    quote = validate_expired_option_candidate(
                        ric=ric,
                        decision_date=decision_date,
                        max_spread_pct=MAX_SPREAD_PCT,
                    )

                    if quote is None:
                        status = "No valid quote"
                        bid = None
                        ask = None
                    else:
                        status = (
                            "Liquid"
                            if quote["Liquid"]
                            else "Wide spread"
                        )
                        bid = quote["BID"]
                        ask = quote["ASK"]

                except Exception as exc:
                    status = (
                        f"{type(exc).__name__}: {exc}"
                    )
                    bid = None
                    ask = None

                rows.append({
                    "Date": decision_date,
                    "Type": option_type,
                    "Strike": strike,
                    "RIC": ric,
                    "Status": status,
                    "BID": bid,
                    "ASK": ask,
                })

    finally:
        ld.close_session()

    results = pd.DataFrame(rows)

    print("\n--- PRE-SPLIT COVERAGE RESULTS ---")
    print(results.to_string(index=False))

    print("\nValid option quotes:",
          results["BID"].notna().sum())

    print("No local datasets modified.")


if __name__ == "__main__":
    main()
