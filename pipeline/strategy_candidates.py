def generate_protective_put_configs():
    configs = []

    strike_pcts = [0.85, 0.90, 0.95]
    deltas = [-0.20, -0.25, -0.30]

    for strike_pct in strike_pcts:
        configs.append({
            "strategy": "Protective Put",
            "put_strike_pct": strike_pct,
            "put_delta": -0.25,
            "put_selection_mode": "strike"
        })

    for delta in deltas:
        configs.append({
            "strategy": "Protective Put",
            "put_strike_pct": 0.90,
            "put_delta": delta,
            "put_selection_mode": "delta"
        })

    for strike_pct in strike_pcts:
        for delta in deltas:
            configs.append({
                "strategy": "Protective Put",
                "put_strike_pct": strike_pct,
                "put_delta": delta,
                "put_selection_mode": "balanced"
            })

    return configs

def generate_collar_configs():
    configs = []

    put_strike_pcts = [0.85, 0.90, 0.95]
    call_strike_pcts = [1.05, 1.10, 1.15]

    put_modes = [
        "strike",
        "delta",
        "balanced"
    ]

    call_modes = [
        "strike",
        "delta",
        "balanced"
    ]

    for put_mode in put_modes:
        for call_mode in call_modes:

            if put_mode == "strike":
                put_configs = [
                    {
                        "put_strike_pct": x,
                        "put_delta": -0.25
                    }
                    for x in put_strike_pcts
                ]

            elif put_mode == "delta":
                put_configs = [
                    {
                        "put_strike_pct": 0.90,
                        "put_delta": x
                    }
                    for x in [-0.20, -0.25, -0.30]
                ]

            else:
                put_configs = [
                    {
                        "put_strike_pct": strike,
                        "put_delta": delta
                    }
                    for strike in put_strike_pcts
                    for delta in [-0.20, -0.25, -0.30]
                ]

            if call_mode == "strike":
                call_configs = [
                    {
                        "call_strike_pct": x,
                        "call_delta": 0.25
                    }
                    for x in call_strike_pcts
                ]

            elif call_mode == "delta":
                call_configs = [
                    {
                        "call_strike_pct": 1.10,
                        "call_delta": x
                    }
                    for x in [0.20, 0.25, 0.30]
                ]

            else:
                call_configs = [
                    {
                        "call_strike_pct": strike,
                        "call_delta": delta
                    }
                    for strike in call_strike_pcts
                    for delta in [0.20, 0.25, 0.30]
                ]

            for put_config in put_configs:
                for call_config in call_configs:
                    configs.append({
                        "strategy": "Collar",

                        **put_config,
                        "put_selection_mode":
                            put_mode,

                        **call_config,
                        "call_selection_mode":
                            call_mode
                    })

    return configs

def generate_put_spread_configs():
    configs = []

    long_strikes = [0.90, 0.95]
    long_deltas = [-0.20, -0.25, -0.30]

    short_strikes = [
        0.75,
        0.80,
        0.85
    ]

    short_deltas = [
        -0.10,
        -0.15,
        -0.20
    ]

    long_modes = [
        "strike",
        "delta",
        "balanced"
    ]

    short_modes = [
        "strike",
        "delta",
        "balanced"
    ]

    for long_mode in long_modes:
        for short_mode in short_modes:

            if long_mode == "strike":
                long_configs = [
                    {
                        "put_strike_pct": x,
                        "put_delta": -0.25
                    }
                    for x in long_strikes
                ]

            elif long_mode == "delta":
                long_configs = [
                    {
                        "put_strike_pct": 0.90,
                        "put_delta": x
                    }
                    for x in long_deltas
                ]

            else:
                long_configs = [
                    {
                        "put_strike_pct": strike,
                        "put_delta": delta
                    }
                    for strike in long_strikes
                    for delta in long_deltas
                ]

            if short_mode == "strike":
                short_configs = [
                    {
                        "short_put_strike_pct": x,
                        "short_put_delta": -0.15
                    }
                    for x in short_strikes
                ]

            elif short_mode == "delta":
                short_configs = [
                    {
                        "short_put_strike_pct": 0.80,
                        "short_put_delta": x
                    }
                    for x in short_deltas
                ]

            else:
                short_configs = [
                    {
                        "short_put_strike_pct":
                            strike,

                        "short_put_delta":
                            delta
                    }
                    for strike in short_strikes
                    for delta in short_deltas
                ]

            for long_config in long_configs:
                for short_config in short_configs:

                    if (
                        short_config[
                            "short_put_strike_pct"
                        ]
                        >= long_config[
                            "put_strike_pct"
                        ]
                    ):
                        continue

                    configs.append({
                        "strategy": "Put Spread",

                        **long_config,

                        "put_selection_mode":
                            long_mode,

                        **short_config,

                        "short_put_selection_mode":
                            short_mode
                    })

    return configs

def generate_cross_hedge_configs():
    return [
        {
            "strategy": "Cross Hedge",
            "cross_hedge_ratio": ratio
        }
        for ratio in [
            0.25,
            0.50,
            0.75,
            1.00
        ]
    ]

def generate_all_strategy_configs():
    return (
        generate_protective_put_configs()
        + generate_collar_configs()
        + generate_put_spread_configs()
        + generate_cross_hedge_configs()
    )