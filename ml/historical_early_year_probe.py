
import pandas as pd
import lseg.data as ld

from pipeline.expired_option_pipeline import (
    build_expired_opra_ric,
    validate_expired_option_candidate,
)


TEST_DATES = [
    ("2013-12-02", "2014-02-21"),
    ("2012-12-03", "2013-02-15"),
    ("2011-12-01", "2012-02-17"),
    ("2010-12-01", "2011-02-18"),
]

MAX_SPREAD_PCT = 0.10

# Applies to the tested dates after Apple's 2014 split
# and before its August 2020 split, provided the
# returned stock series is adjusted on that basis.
def get_split_factor(decision_date):
    date = pd.Timestamp(str(decision_date)[:10]).normalize()

    if date < pd.Timestamp("2014-06-09"):
        return 28

    if date < pd.Timestamp("2020-08-31"):
        return 4

    return 1


def get_stock_price(decision_date):
    date = pd.Timestamp(decision_date).normalize()

    history = ld.get_history(
        universe="AAPL.O",
        fields=["TRDPRC_1"],
        interval="daily",
        start=(
            date - pd.Timedelta(days=7)
        ).strftime("%Y-%m-%d"),
        end=(
            date + pd.Timedelta(days=3)
        ).strftime("%Y-%m-%d"),
    )

    if history is None or history.empty:
        return None

    if isinstance(history.columns, pd.MultiIndex):
        for level in range(history.columns.nlevels):
            if "AAPL.O" in history.columns.get_level_values(level):
                history = history.xs(
                    "AAPL.O",
                    axis=1,
                    level=level,
                )
                break

    if "TRDPRC_1" not in history.columns:
        return None

    history.index = pd.to_datetime(
        history.index
    ).normalize()

    if date not in history.index:
        return None

    value = history.loc[date, "TRDPRC_1"]

    if isinstance(value, pd.Series):
        value = value.iloc[-1]

    value = pd.to_numeric(value, errors="coerce")

    if pd.isna(value) or float(value) <= 0:
        return None

    return float(value)


def make_strikes(historical_spot):
    def nearest_five(value):
        return round(value / 5) * 5

    return [
        ("put", nearest_five(historical_spot * 0.90)),
        ("put", nearest_five(historical_spot * 0.95)),
        ("call", nearest_five(historical_spot * 1.05)),
        ("call", nearest_five(historical_spot * 1.10)),
    ]


def main():
    rows = []

    ld.open_session()

    try:
        for decision_date, expiry in TEST_DATES:
            print(f"\n--- {decision_date} ---")

            try:
                adjusted_spot = get_stock_price(
                    decision_date
                )

                if adjusted_spot is None:
                    print("No stock price available.")
                    continue

                historical_spot = (
                    adjusted_spot
                    * get_split_factor(decision_date)
                )

                print(
                    "Stock price returned:",
                    f"{adjusted_spot:.4f}",
                )

                print(
                    "Estimated historical spot:",
                    f"{historical_spot:.4f}",
                )

                for option_type, strike in make_strikes(
                    historical_spot
                ):
                    ric = build_expired_opra_ric(
                        underlying_root="AAPL",
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
                            status = "Unavailable"
                        elif quote["Liquid"]:
                            status = "Liquid"
                        else:
                            status = "Wide spread"

                    except Exception as exc:
                        status = (
                            f"{type(exc).__name__}: {exc}"
                        )

                    print(
                        option_type,
                        strike,
                        ric,
                        status,
                    )

                    rows.append({
                        "Date": decision_date,
                        "Type": option_type,
                        "Strike": strike,
                        "RIC": ric,
                        "Status": status,
                    })

            except Exception as exc:
                print(
                    "Date-level error:",
                    type(exc).__name__,
                    str(exc),
                )

    finally:
        ld.close_session()

    print("\n--- COVERAGE SUMMARY ---")

    if rows:
        results = pd.DataFrame(rows)

        print(
            results.groupby(
                ["Date", "Status"]
            ).size().unstack(fill_value=0)
        )
    else:
        print("No contracts tested.")

    print("\nNo local datasets modified.")


if __name__ == "__main__":
    main()
