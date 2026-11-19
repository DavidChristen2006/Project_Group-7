
import pandas as pd
import lseg.data as ld

from pipeline.expired_option_pipeline import (
    build_expired_opra_ric,
)

HEDGE_HORIZON_DAYS = 21
MIN_DAILY_COVERAGE = 0.90

# Previously confirmed liquid historical contracts.

TEST_CASES = [
    {
        "date": "2014-12-01",
        "put": "AAPLN201511000.U^B15",
        "call": "AAPLB201512000.U^B15",
    },
    {
        "date": "2015-12-01",
        "put": "AAPLN191611000.U^B16",
        "call": "AAPLB191612500.U^B16",
    },
    {
        "date": "2016-12-01",
        "put": "AAPLN171710500.U^B17",
        "call": "AAPLB171711500.U^B17",
    },
    {
        "date": "2017-12-01",
        "put": "AAPLN161816000.U^B18",
        "call": "AAPLB161818000.U^B18",
    },
]


def normalize_history(history, ric, fields):
    if history is None or history.empty:
        return pd.DataFrame()

    history = history.copy()

    if isinstance(history.columns, pd.MultiIndex):
        for level in range(history.columns.nlevels):
            if ric in history.columns.get_level_values(level):
                history = history.xs(
                    ric,
                    axis=1,
                    level=level,
                )
                break

    if not set(fields).issubset(history.columns):
        return pd.DataFrame()

    history = history[list(fields)].copy()

    history.index = pd.to_datetime(
        history.index
    ).normalize()

    history = history.sort_index()
    history = history[
        ~history.index.duplicated(keep="last")
    ]

    return history



def get_holding_dates(decision_date):
    decision_date = pd.Timestamp(
        decision_date
    ).normalize()

    # Request additional calendar days around the period.
    query_start = decision_date - pd.Timedelta(days=7)
    query_end = decision_date + pd.Timedelta(days=60)

    history = ld.get_history(
        universe="AAPL.O",
        fields=["TRDPRC_1"],
        interval="daily",
        start=query_start.strftime("%Y-%m-%d"),
        end=query_end.strftime("%Y-%m-%d"),
    )

    history = normalize_history(
        history,
        "AAPL.O",
        ["TRDPRC_1"],
    )

    if history.empty:
        raise ValueError(
            "No historical stock observations."
        )

    history["TRDPRC_1"] = pd.to_numeric(
        history["TRDPRC_1"],
        errors="coerce",
    )

    history = history.dropna(
        subset=["TRDPRC_1"]
    )

    # Restrict observations to the actual holding period.
    history = history.loc[
        history.index >= decision_date
    ]

    if history.empty:
        raise ValueError(
            "No stock observations on or after decision date."
        )

    actual_entry = pd.Timestamp(
        history.index[0]
    ).normalize()

    if actual_entry != decision_date:
        raise ValueError(
            f"Expected entry {decision_date.date()}, "
            f"but first available date is "
            f"{actual_entry.date()}."
        )

    required_observations = HEDGE_HORIZON_DAYS + 1

    if len(history) < required_observations:
        raise ValueError(
            f"Insufficient stock history: "
            f"{len(history)} observations, "
            f"{required_observations} required."
        )

    return pd.DatetimeIndex(
        history.index[:required_observations]
    )




def check_option_history(ric, trading_dates):
    start = pd.Timestamp(
        str(trading_dates[0])[:10]
    )
    end = pd.Timestamp(
        str(trading_dates[-1])[:10]
    )

    def fetch_quotes(query_start, query_end):
        history = ld.get_history(
            universe=ric,
            fields=["BID", "ASK"],
            interval="daily",
            start=query_start.strftime("%Y-%m-%d"),
            end=query_end.strftime("%Y-%m-%d"),
        )

        return normalize_history(
            history,
            ric,
            ["BID", "ASK"],
        )

    # Main holding-period request.
    history = fetch_quotes(
        start - pd.Timedelta(days=7),
        end + pd.Timedelta(days=3),
    )

    if history.empty:
        history = pd.DataFrame(
            columns=["BID", "ASK"]
        )
        history.index = pd.DatetimeIndex([])

    entry_recovered = False

    # Some longer requests omit the entry observation.
    # Recheck the exact date using a short query.
    if start not in history.index:
        entry_history = fetch_quotes(
            start - pd.Timedelta(days=5),
            start + pd.Timedelta(days=5),
        )

        if not entry_history.empty:
            if start in entry_history.index:
                history = pd.concat(
                    [history, entry_history.loc[[start]]]
                )

                entry_recovered = True

    if history.empty:
        return {
            "RIC": ric,
            "Entry": False,
            "Exit": False,
            "Coverage": 0.0,
            "Complete": False,
            "Entry Recovered": False,
        }

    history = history.sort_index()
    history = history[
        ~history.index.duplicated(keep="last")
    ]

    for col in ["BID", "ASK"]:
        history[col] = pd.to_numeric(
            history[col],
            errors="coerce",
        )

    valid = (
        history["BID"].notna()
        & history["ASK"].notna()
        & (history["BID"] >= 0)
        & (history["ASK"] > 0)
        & (history["ASK"] >= history["BID"])
    )

    valid_dates = history.index[valid]

    entry_ok = start in valid_dates
    exit_ok = end in valid_dates

    matched = trading_dates.isin(
        valid_dates
    ).sum()

    coverage = matched / len(trading_dates)

    complete = (
        entry_ok
        and exit_ok
        and coverage >= MIN_DAILY_COVERAGE
    )

    return {
        "RIC": ric,
        "Entry": bool(entry_ok),
        "Exit": bool(exit_ok),
        "Coverage": float(coverage),
        "Complete": bool(complete),
        "Entry Recovered": (
            entry_recovered and bool(entry_ok)
        ),
    }



def main():
    rows = []

    ld.open_session()

    try:
        for case in TEST_CASES:
            decision_date = case["date"]

            print(f"\n--- {decision_date} ---")

            try:
                trading_dates = get_holding_dates(
                    decision_date
                )

                exit_date = trading_dates[-1]

                print(
                    "Historical exit:",
                    exit_date.date(),
                )

                for option_type in ["put", "call"]:
                    ric = case[option_type]

                    try:
                        result = check_option_history(
                            ric,
                            trading_dates,
                        )

                        print(
                            option_type,
                            ric,
                            f"coverage={result['Coverage']:.1%}",
                            f"entry={result['Entry']}",
                            f"exit={result['Exit']}",
                        )

                        rows.append({
                            "Date": decision_date,
                            "Type": option_type,
                            "Exit Date": exit_date.date(),
                            **result,
                        })

                    except Exception as exc:
                        print(
                            option_type,
                            type(exc).__name__,
                            str(exc),
                        )

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
            results.to_string(
                index=False,
                float_format=lambda x: f"{x:.2%}",
            )
        )
    else:
        print("No successful option checks.")

    print("\nNo local files modified.")


if __name__ == "__main__":
    main()
