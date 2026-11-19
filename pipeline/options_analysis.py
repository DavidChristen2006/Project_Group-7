import pandas as pd

from lseg_api.options import get_option_chain, add_option_market_data, filter_liquid_options, add_option_analytics, rank_put_options, rank_call_options_same_expiry
from lseg_api.option_history import fill_historical_option_quotes


def prepare_option_universe(position, market):
    current_price = market.current_price

    options = get_option_chain(
        underlying_ric=position.ric,
        minimum_strike=current_price * 0.65,
        maximum_strike=current_price * 1.25,
        max_results=2000
    )

    print("\n--- OPTION UNIVERSE DIAGNOSTICS ---")
    print(f"Raw option chain: {len(options)}")

    target_expiry = calculate_target_expiry(
        position.hedge_horizon_days
    )

    options["ExpiryDate"] = pd.to_datetime(
        options["ExpiryDate"]
    )

    options = options[
        options["ExpiryDate"] >= target_expiry
    ].copy()

    print(f"After expiry filter: {len(options)}")

    options = add_option_market_data(options)

    options = fill_historical_option_quotes(
        options,
        batch_size=25,
        count=5,
        max_age_days=7
    )

    options = filter_liquid_options(
        options,
        max_spread_pct=0.10
    )

    print(f"After market data: {len(options)}")

    if not options.empty:
        print(
            "Available columns:",
            options.columns.tolist()
        )

        for column in ["BID", "ASK"]:
            if column in options.columns:
                print(
                    f"{column} non-null:",
                    options[column].notna().sum()
                )

    options = filter_liquid_options(
        options,
        max_spread_pct=0.10
    )

    print(f"After liquidity filter: {len(options)}")

    print(
        f"Options: {len(options)} liquid contracts "
        f"({(options['CallPutOption'] == 'Put').sum()} puts, "
        f"{(options['CallPutOption'] == 'Call').sum()} calls)"
    )

    print("\n--- OPTION QUOTE SOURCE SUMMARY ---")

    if "QuoteSource" in options.columns:
        print(options["QuoteSource"].value_counts(dropna=False))

    if "QuoteDate" in options.columns:
        print(
            "Historical quote dates:",
            options["QuoteDate"].dropna().nunique()
        )

    if "Spread %" in options.columns:
        print(
            f"Maximum accepted spread: "
            f"{options['Spread %'].max():.2%}"
        )

    return options

def calculate_target_expiry(hedge_horizon_days):
    calendar_horizon_days = int(round(hedge_horizon_days * 365 / 252))

    return pd.Timestamp.today().normalize() + pd.Timedelta(days=calendar_horizon_days)

def select_protective_put(
    position,
    market,
    options,
    target_put_strike_pct=0.90,
    target_put_delta=-0.25,
    selection_mode="balanced"):
    current_price = market.current_price

    target_expiry = calculate_target_expiry(position.hedge_horizon_days)
    target_put_strike = current_price * target_put_strike_pct

    put_candidates = options[options["CallPutOption"] == "Put"].copy()

    target_put_strike = (
        current_price
        * target_put_strike_pct
    )

    lower_bound = (
        target_put_strike * 0.90
    )

    upper_bound = (
        target_put_strike * 1.10
    )

    puts_in_range = put_candidates[
        (
            put_candidates["StrikePrice"]
            >= lower_bound
        )
        &
        (
            put_candidates["StrikePrice"]
            <= upper_bound
        )
    ].copy()

    put_candidates = put_candidates[put_candidates["ExpiryDate"] >= target_expiry].copy()

    if selection_mode == "strike":
        put_candidates = put_candidates[
            (put_candidates["StrikePrice"] >= target_put_strike * 0.95)
            &
            (put_candidates["StrikePrice"] <= target_put_strike * 1.05)
        ].copy()

    elif selection_mode == "delta":
        pass

    elif selection_mode == "balanced":
        put_candidates = put_candidates[
            (put_candidates["StrikePrice"] >= target_put_strike * 0.85)
            &
            (put_candidates["StrikePrice"] <= target_put_strike * 1.15)
        ].copy()

    else:
        raise ValueError("selection_mode must be 'strike', 'delta', or 'balanced'.")

    put_candidates["Expiry Distance"] = (put_candidates["ExpiryDate"] - target_expiry).dt.days

    put_candidates["Strike Distance"] = (put_candidates["StrikePrice"] - target_put_strike).abs()

    if selection_mode == "strike":
        put_candidates = (put_candidates.sort_values(
                by=[
                    "Strike Distance",
                    "Expiry Distance"
                ]).head(30))

    elif selection_mode == "delta":
        put_candidates = put_candidates.sort_values(by="Expiry Distance").head(50)

    elif selection_mode == "balanced":
        put_candidates = (put_candidates.sort_values(
                by=[
                    "Expiry Distance",
                    "Strike Distance"
                ]).head(40))

    put_candidates = add_option_analytics(
        options=put_candidates,
        current_price=current_price,
        risk_free_rate=market.risk_free_rate,
        dividend_yield=market.dividend_yield)

    if put_candidates.empty:
        raise ValueError("No valid protective-put candidates remain.")

    ranked_puts = rank_put_options(
        options=put_candidates,
        current_price=current_price,
        hedge_horizon_days=(
            position.hedge_horizon_days
        ),
        target_strike_pct=(
            target_put_strike_pct
        ),
        target_delta=(
            target_put_delta
        ),
        selection_mode=selection_mode)

    selected_put = ranked_puts.iloc[0]

    return {
        "selected_put": selected_put,
        "ranked_puts": ranked_puts}

def select_collar(
    position,
    market,
    options,
    target_put_strike_pct=0.90,
    target_put_delta=-0.25,
    target_call_strike_pct=1.10,
    target_call_delta=0.25,
    put_selection_mode="balanced",
    call_selection_mode="balanced"):
    put_result = (
        select_protective_put(
            position=position,
            market=market,
            options=options,
            target_put_strike_pct=(target_put_strike_pct),
            target_put_delta=(target_put_delta),
            selection_mode=put_selection_mode))

    selected_put = put_result["selected_put"]
    put_expiry = selected_put["ExpiryDate"]
    current_price = market.current_price
    target_call_strike = current_price * target_call_strike_pct

    call_candidates = options[options["CallPutOption"] == "Call"].copy()
    call_candidates = call_candidates[call_candidates["ExpiryDate"] == put_expiry].copy()
    if call_selection_mode == "strike":
        call_candidates = call_candidates[
            (
                call_candidates["StrikePrice"]
                >= target_call_strike * 0.95
            )
            &
            (
                call_candidates["StrikePrice"]
                <= target_call_strike * 1.05
            )
        ].copy()

    elif call_selection_mode == "delta":
        pass

    elif call_selection_mode == "balanced":
        call_candidates = call_candidates[
            (
                call_candidates["StrikePrice"]
                >= target_call_strike * 0.85
            )
            &
            (
                call_candidates["StrikePrice"]
                <= target_call_strike * 1.15
            )
        ].copy()

    else:
        raise ValueError(
            "call_selection_mode must be "
            "'strike', 'delta', or 'balanced'."
        )
    call_candidates["Strike Distance"] = (call_candidates["StrikePrice"] - target_call_strike).abs()
    if call_selection_mode == "strike":
        call_candidates = (
            call_candidates
            .sort_values(
                by="Strike Distance"
            )
            .head(30)
        )

    elif call_selection_mode == "delta":
        call_candidates = (
            call_candidates
            .head(50)
        )

    elif call_selection_mode == "balanced":
        call_candidates = (
            call_candidates
            .sort_values(
                by="Strike Distance"
            )
            .head(40)
        )
    call_candidates = add_option_analytics(
        options=call_candidates,
        current_price=current_price,
        risk_free_rate=market.risk_free_rate,
        dividend_yield=market.dividend_yield)

    if call_candidates.empty:
        raise ValueError("No valid collar call candidates remain.")

    ranked_calls = rank_call_options_same_expiry(
        options=call_candidates,
        current_price=current_price,
        expiry_date=put_expiry,
        target_strike_pct=target_call_strike_pct,
        target_delta=target_call_delta,
        selection_mode=call_selection_mode)

    selected_call = ranked_calls.iloc[0]

    return {
        "selected_put": selected_put,
        "selected_call": selected_call,
        "ranked_puts": put_result["ranked_puts"],
        "ranked_calls": ranked_calls}

def select_put_spread(
    position,
    market,
    options,
    target_long_put_strike_pct=0.90,
    target_long_put_delta=-0.25,
    target_short_put_strike_pct=0.80,
    target_short_put_delta=-0.15,
    long_put_selection_mode="balanced",
    short_put_selection_mode="balanced"
):
    current_price = market.current_price

    long_put_result = select_protective_put(
        position=position,
        market=market,
        options=options,
        target_put_strike_pct=target_long_put_strike_pct,
        target_put_delta=target_long_put_delta,
        selection_mode=long_put_selection_mode
    )

    ranked_long_puts = long_put_result["ranked_puts"]

    target_short_put_strike = (
        current_price * target_short_put_strike_pct
    )

    selected_long_put = None
    selected_short_put = None
    selected_short_put_candidates = None

    for _, candidate_long_put in ranked_long_puts.iterrows():

        long_put_strike = (
            candidate_long_put["StrikePrice"]
        )

        long_put_expiry = (
            candidate_long_put["ExpiryDate"]
        )

        short_put_candidates = options[
            options["CallPutOption"] == "Put"
        ].copy()

        short_put_candidates = short_put_candidates[
            short_put_candidates["ExpiryDate"]
            == long_put_expiry
        ].copy()

        short_put_candidates = short_put_candidates[
            short_put_candidates["StrikePrice"]
            < long_put_strike
        ].copy()

        if short_put_candidates.empty:
            continue

        short_put_candidates[
            "Strike Distance"
        ] = (
            short_put_candidates["StrikePrice"]
            - target_short_put_strike
        ).abs()

        short_put_candidates[
            "Strike Distance %"
        ] = (
            short_put_candidates["Strike Distance"]
            / current_price
        )

        if short_put_selection_mode == "strike":
            short_put_candidates = (
                short_put_candidates
                .sort_values(
                    by="Strike Distance"
                )
                .head(30)
            )

        elif short_put_selection_mode == "delta":
            short_put_candidates = (
                short_put_candidates
                .head(50)
            )

        elif short_put_selection_mode == "balanced":
            short_put_candidates = (
                short_put_candidates
                .sort_values(
                    by="Strike Distance"
                )
                .head(40)
            )

        else:
            raise ValueError(
                "short_put_selection_mode must be "
                "'strike', 'delta', or 'balanced'."
            )

        short_put_candidates = add_option_analytics(
            options=short_put_candidates,
            current_price=current_price,
            risk_free_rate=market.risk_free_rate,
            dividend_yield=market.dividend_yield
        )

        if short_put_candidates.empty:
            continue

        short_put_candidates[
            "Delta Distance"
        ] = (
            short_put_candidates["Delta"]
            - target_short_put_delta
        ).abs()

        short_put_candidates[
            "Premium %"
        ] = (
            short_put_candidates["BID"]
            / current_price
        )

        if short_put_selection_mode == "strike":

            short_put_candidates[
                "Selection Score"
            ] = (
                short_put_candidates[
                    "Strike Distance %"
                ] * 10
                + short_put_candidates[
                    "Spread %"
                ] * 5
                + short_put_candidates[
                    "Premium %"
                ] * 5
            )

        elif short_put_selection_mode == "delta":

            short_put_candidates[
                "Selection Score"
            ] = (
                short_put_candidates[
                    "Delta Distance"
                ] * 10
                + short_put_candidates[
                    "Spread %"
                ] * 5
                + short_put_candidates[
                    "Premium %"
                ] * 5
            )

        elif short_put_selection_mode == "balanced":

            short_put_candidates[
                "Selection Score"
            ] = (
                short_put_candidates[
                    "Strike Distance %"
                ] * 10
                + short_put_candidates[
                    "Delta Distance"
                ] * 2
                + short_put_candidates[
                    "Spread %"
                ] * 5
                + short_put_candidates[
                    "Premium %"
                ] * 5
            )

        candidate_short_put = (
            short_put_candidates
            .sort_values(
                by="Selection Score"
            )
            .iloc[0]
        )

        if (
            candidate_short_put["StrikePrice"]
            >= long_put_strike
        ):
            continue

        selected_long_put = candidate_long_put
        selected_short_put = candidate_short_put

        selected_short_put_candidates = (
            short_put_candidates
        )

        break

    if (
        selected_long_put is None
        or selected_short_put is None
    ):
        raise ValueError(
            "No valid long-put / short-put pair "
            "was found for the put spread."
        )

    return {
        "selected_long_put":
            selected_long_put,

        "selected_short_put":
            selected_short_put,

        "ranked_long_puts":
            ranked_long_puts,

        "short_put_candidates":
            selected_short_put_candidates}