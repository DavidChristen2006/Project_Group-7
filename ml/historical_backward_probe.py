
import pandas as pd
import lseg.data as ld

from pipeline.expired_option_pipeline import (
    build_expired_opra_ric,
    validate_expired_option_candidate,
)


UNDERLYING_RIC = "AAPL.O"
UNDERLYING_ROOT = "AAPL"

TEST_DATES = {
    "2021-06-01": "2021-08-20",
    "2020-12-01": "2021-02-19",
    "2019-12-02": "2020-02-21",
    "2018-12-03": "2019-02-15",
}

MAX_SPREAD_PCT = 0.10


def get_stock_price(decision_date):
    date = pd.Timestamp(decision_date)

    history = ld.get_history(
        universe=UNDERLYING_RIC,
        fields=["TRDPRC_1"],
        interval="daily",
        start=(date - pd.Timedelta(days=3)).strftime(
            "%Y-%m-%d"
        ),
        end=(date + pd.Timedelta(days=2)).strftime(
            "%Y-%m-%d"
        ),
    )

    if history is None or history.empty:
        return None

    if isinstance(history.columns, pd.MultiIndex):
        for level in range(history.columns.nlevels):
            if UNDERLYING_RIC in history.columns.get_level_values(level):
                history = history.xs(
                    UNDERLYING_RIC,
                    axis=1,
                    level=level,
                )
                break

    if "TRDPRC_1" not in history.columns:
        return None

    history.index = pd.to_datetime(history.index).normalize()

    if date not in history.index:
        return None

    price = pd.to_numeric(
        history.loc[date, "TRDPRC_1"],
        errors="coerce",
    )

    if isinstance(price, pd.Series):
        price = price.iloc[-1]

    if pd.isna(price) or price <= 0:
        return None

    return float(price)


def make_strikes(price):
    def nearest_five(value):
        return round(value / 5) * 5

    return [
        ("put", nearest_five(price * 0.90)),
        ("put", nearest_five(price * 0.95)),
        ("call", nearest_five(price * 1.05)),
        ("call", nearest_five(price * 1.10)),
    ]


def check_date(decision_date, expiry):
    print(f"\n--- {decision_date} ---")

    price = get_stock_price(decision_date)

    if price is None:
        print("No valid underlying price.")
        return {
            "Date": decision_date,
            "Stock Price": None,
            "Valid Options": 0,
            "Liquid Options": 0,
            "Liquid Puts": 0,
            "Liquid Calls": 0,
        }

    print(f"Stock price: {price:.4f}")

    valid = 0
    liquid = 0
    liquid_puts = 0
    liquid_calls = 0

    for option_type, strike in make_strikes(price):
        ric = build_expired_opra_ric(
            underlying_root=UNDERLYING_ROOT,
            expiry=expiry,
            strike=strike,
            option_type=option_type,
        )

        try:
            quote = validate_expired_option_candidate(
                ric=ric,
                decision_date=decision_date,
                max_spread_pct=MAX_SPREAD_PCT,
            )

            if quote is None:
                print(f"{option_type} {strike}: unavailable")
                continue

            valid += 1

            if quote["Liquid"]:
                liquid += 1

                if option_type == "put":
                    liquid_puts += 1
                else:
                    liquid_calls += 1

            print(
                f"{option_type} {strike}: "
                f"BID={quote['BID']:.4f}, "
                f"ASK={quote['ASK']:.4f}, "
                f"spread={quote['SPREAD_PCT'] * 100:.2f}%, "
                f"liquid={quote['Liquid']}"
            )

        except Exception as exc:
            print(
                f"{option_type} {strike}: "
                f"{type(exc).__name__}: {exc}"
            )

    return {
        "Date": decision_date,
        "Stock Price": price,
        "Valid Options": valid,
        "Liquid Options": liquid,
        "Liquid Puts": liquid_puts,
        "Liquid Calls": liquid_calls,
    }


def main():
    ld.open_session()

    try:
        results = []

        for decision_date, expiry in TEST_DATES.items():
            try:
                result = check_date(
                    decision_date,
                    expiry,
                )
            except Exception as exc:
                print(
                    f"Date-level error: "
                    f"{type(exc).__name__}: {exc}"
                )
                result = {
                    "Date": decision_date,
                    "Error": str(exc),
                }

            results.append(result)

        summary = pd.DataFrame(results)

        print("\n--- HISTORICAL BACKWARD COVERAGE ---")
        print(summary.to_string(index=False))

    finally:
        ld.close_session()

    print("\nNo historical files modified.")


if __name__ == "__main__":
    main()
