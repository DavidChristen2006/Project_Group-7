
import pandas as pd
import lseg.data as ld

from pipeline.expired_option_pipeline import (
    build_expired_opra_ric,
    validate_expired_option_candidate,
)


UNDERLYING_RIC = "AAPL.O"
UNDERLYING_ROOT = "AAPL"

DECISION_DATE = "2021-12-01"
EXPIRY = "2022-02-18"

MAX_SPREAD_PCT = 0.10


def get_historical_stock_price(ric, decision_date):
    date = pd.Timestamp(decision_date).normalize()

    history = ld.get_history(
        universe=ric,
        fields=["TRDPRC_1"],
        interval="daily",
        start=(date - pd.Timedelta(days=5)).strftime(
            "%Y-%m-%d"
        ),
        end=(date + pd.Timedelta(days=2)).strftime(
            "%Y-%m-%d"
        ),
    )

    if history is None or history.empty:
        raise ValueError(
            "No historical stock prices returned."
        )

    if isinstance(history.columns, pd.MultiIndex):
        for level in range(history.columns.nlevels):
            if ric in history.columns.get_level_values(level):
                history = history.xs(
                    ric,
                    axis=1,
                    level=level,
                )
                break

    if "TRDPRC_1" not in history.columns:
        raise ValueError(
            f"TRDPRC_1 missing: {history.columns.tolist()}"
        )

    history.index = pd.to_datetime(
        history.index
    ).normalize()

    if date not in history.index:
        raise ValueError(
            f"No stock observation on {date.date()}."
        )

    value = history.loc[date, "TRDPRC_1"]

    if isinstance(value, pd.Series):
        value = value.iloc[-1]

    price = float(value)

    if not 0 < price < float("inf"):
        raise ValueError("Invalid stock price.")

    return price


def make_test_strikes(price):
    # A small sample of representative strikes.
    # This is only an availability probe.

    strike_step = 5.0

    def round_to_step(value):
        return (
            round(value / strike_step)
            * strike_step
        )

    candidates = [
        ("put", 0.80),
        ("put", 0.85),
        ("put", 0.90),
        ("put", 0.95),
        ("call", 1.05),
        ("call", 1.10),
    ]

    return [
        (
            option_type,
            round_to_step(price * strike_pct),
        )
        for option_type, strike_pct in candidates
    ]


def main():
    print("\n--- HISTORICAL COVERAGE PROBE ---")
    print("Underlying:", UNDERLYING_RIC)
    print("Decision date:", DECISION_DATE)
    print("Option expiry:", EXPIRY)

    ld.open_session()

    try:
        stock_price = get_historical_stock_price(
            UNDERLYING_RIC,
            DECISION_DATE,
        )

        print(
            "\nHistorical stock price:",
            f"{stock_price:.4f}",
        )

        rows = []

        for option_type, strike in make_test_strikes(
            stock_price
        ):
            option_ric = build_expired_opra_ric(
                underlying_root=UNDERLYING_ROOT,
                expiry=EXPIRY,
                strike=strike,
                option_type=option_type,
            )

            print(
                "\nChecking:",
                option_type,
                strike,
                option_ric,
            )

            try:
                quote = validate_expired_option_candidate(
                    ric=option_ric,
                    decision_date=DECISION_DATE,
                    max_spread_pct=MAX_SPREAD_PCT,
                )

                if quote is None:
                    status = "No valid quote"
                    bid = None
                    ask = None
                    spread_pct = None
                else:
                    bid = quote["BID"]
                    ask = quote["ASK"]
                    spread_pct = quote["SPREAD_PCT"]

                    status = (
                        "Liquid"
                        if quote["Liquid"]
                        else "Wide spread"
                    )

            except Exception as exc:
                status = (
                    f"Query error: "
                    f"{type(exc).__name__}: {exc}"
                )
                bid = None
                ask = None
                spread_pct = None

            rows.append({
                "Type": option_type,
                "Strike": strike,
                "RIC": option_ric,
                "Status": status,
                "BID": bid,
                "ASK": ask,
                "Spread (%)": (
                    None
                    if spread_pct is None
                    else spread_pct * 100
                ),
            })

        results = pd.DataFrame(rows)

        print("\n--- OPTION DISCOVERY RESULTS ---")
        print(
            results.to_string(
                index=False,
                float_format=lambda x: f"{x:.4f}",
            )
        )

        print("\n--- SUMMARY ---")

        print(
            "Valid quotes:",
            results["BID"].notna().sum(),
            "/",
            len(results),
        )

        print(
            "Liquid contracts:",
            (results["Status"] == "Liquid").sum(),
        )

    finally:
        ld.close_session()

    print("\nProbe complete.")
    print("No historical files modified.")


if __name__ == "__main__":
    main()
