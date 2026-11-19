
from calendar import monthrange
from datetime import date
from pathlib import Path

import pandas as pd
import lseg.data as ld


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DISCOVERY_DIR = (
    PROJECT_ROOT / "data" / "processed" / "option_discovery"
)


def third_friday(year, month):
    first = date(year, month, 1)

    days_until_friday = (4 - first.weekday()) % 7

    day = 1 + days_until_friday + 14

    if day > monthrange(year, month)[1]:
        raise ValueError("Invalid third-Friday calculation.")

    return pd.Timestamp(year=year, month=month, day=day)


def build_expired_opra_ric(
    underlying_root,
    expiry,
    strike,
    option_type
):
    expiry = pd.Timestamp(expiry).normalize()

    option_type = option_type.lower()

    if option_type not in ("call", "put"):
        raise ValueError("option_type must be call or put.")

    root = underlying_root.upper()

    if not root.isalpha() or len(root) > 5:
        raise ValueError("Unsupported OPRA underlying root.")

    strike_units = round(float(strike) * 100)

    if (
        strike <= 0
        or strike >= 1000
        or abs(strike_units / 100 - float(strike)) > 1e-8
    ):
        raise ValueError(
            "This initial RIC builder supports strikes "
            "below 1000 in increments of 0.01."
        )

    month = expiry.month

    if option_type == "call":
        month_code = "ABCDEFGHIJKL"[month - 1]
    else:
        month_code = "MNOPQRSTUVWX"[month - 1]

    year_code = expiry.strftime("%y")
    strike_code = f"{strike_units:05d}"

    expiry_month_code = "ABCDEFGHIJKL"[month - 1]

    ric = (
        f"{root}{month_code}"
        f"{expiry.day:02d}{year_code}"
        f"{strike_code}.U"
        f"^{expiry_month_code}{year_code}"
    )

    return ric


def validate_expired_option_candidate(
    ric,
    decision_date,
    max_spread_pct=0.10
):
    decision_date = pd.Timestamp(
        decision_date
    ).normalize()

    history = ld.get_history(
        universe=ric,
        fields=["BID", "ASK"],
        interval="daily",
        start=(
            decision_date - pd.Timedelta(days=5)
        ).strftime("%Y-%m-%d"),
        end=(
            decision_date + pd.Timedelta(days=5)
        ).strftime("%Y-%m-%d")
    )

    if history is None or history.empty:
        print("  Empty LSEG history.")
        return None

    history = history.copy()

    if isinstance(history.columns, pd.MultiIndex):
        for level in range(history.columns.nlevels):
            if ric in history.columns.get_level_values(level):
                history = history.xs(
                    ric,
                    axis=1,
                    level=level
                )
                break

    if isinstance(history.columns, pd.MultiIndex):
        print(
            "  Unsupported column structure:",
            history.columns.tolist()
        )
        return None

    if not {"BID", "ASK"}.issubset(history.columns):
        print(
            "  BID/ASK columns missing:",
            history.columns.tolist()
        )
        return None

    history.index = pd.to_datetime(
        history.index
    ).normalize()

    history = history.sort_index()

    if decision_date not in history.index:
        print(
            f"  Decision date {decision_date.date()} "
            "not found. Available dates:",
            history.index.date.tolist()
        )
        return None

    quote = history.loc[decision_date]

    if isinstance(quote, pd.DataFrame):
        quote = quote.iloc[-1]

    bid = pd.to_numeric(
        quote["BID"],
        errors="coerce"
    )

    ask = pd.to_numeric(
        quote["ASK"],
        errors="coerce"
    )

    if pd.isna(bid) or pd.isna(ask):
        print(f"  Missing BID/ASK: {bid}, {ask}")
        return None

    bid = float(bid)
    ask = float(ask)

    if bid < 0 or ask <= 0 or ask < bid:
        print(f"  Invalid BID/ASK: {bid}, {ask}")
        return None

    mid = (bid + ask) / 2
    spread_pct = (ask - bid) / mid

    return {
        "RIC": ric,
        "Decision Date": decision_date,
        "BID": bid,
        "ASK": ask,
        "MID": mid,
        "SPREAD_PCT": spread_pct,
        "Liquid": spread_pct <= max_spread_pct
    }


def discover_expired_options(
    underlying_root,
    decision_date,
    expiry,
    strikes,
    option_types=("put", "call"),
    max_spread_pct=0.10
):
    decision_date = pd.Timestamp(decision_date).normalize()
    expiry = pd.Timestamp(expiry).normalize()

    if decision_date >= expiry:
        raise ValueError("Decision date must precede expiry.")

    results = []

    for option_type in option_types:
        for strike in strikes:
            ric = build_expired_opra_ric(
                underlying_root=underlying_root,
                expiry=expiry,
                strike=strike,
                option_type=option_type
            )

            print(f"Checking {ric}")

            try:
                quote = validate_expired_option_candidate(
                    ric=ric,
                    decision_date=decision_date,
                    max_spread_pct=max_spread_pct
                )

                if quote is None:
                    print("  No valid quote.")
                    continue

                quote["Option Type"] = option_type
                quote["Strike"] = float(strike)
                quote["Expiry"] = expiry

                results.append(quote)

                print(
                    f"  Found: BID {quote['BID']:.2f}, "
                    f"ASK {quote['ASK']:.2f}"
                )

            except Exception as exc:
                print(f"  Lookup failed: {exc}")

    result = pd.DataFrame(results)

    if result.empty:
        print("No valid contracts discovered.")
        return result

    result = result.sort_values(
        ["Option Type", "Strike"]
    ).reset_index(drop=True)

    DISCOVERY_DIR.mkdir(parents=True, exist_ok=True)

    filename = (
        f"{underlying_root.upper()}_"
        f"{decision_date:%Y%m%d}_"
        f"{expiry:%Y%m%d}.parquet"
    )

    result.to_parquet(
        DISCOVERY_DIR / filename,
        index=False
    )

    return result
