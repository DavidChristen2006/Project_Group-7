from models.hedge_config import HedgeConfig


def generate_hedge_configs():
    put_strike_pcts = [0.85, 0.90, 0.95]
    call_strike_pcts = [1.05, 1.10, 1.15]
    short_put_strike_pcts = [0.75, 0.80, 0.85]
    cross_hedge_ratios = [0.25, 0.50, 0.75, 1.00]
    configs = []

    for put_strike_pct in put_strike_pcts:
        for call_strike_pct in call_strike_pcts:
            for short_put_strike_pct in short_put_strike_pcts:
                for cross_hedge_ratio in cross_hedge_ratios:
                    config = HedgeConfig(
                        put_strike_pct=put_strike_pct,
                        put_delta=-0.25,
                        call_strike_pct=call_strike_pct,
                        call_delta=0.25,
                        short_put_strike_pct=short_put_strike_pct,
                        cross_hedge_ratio=cross_hedge_ratio)

                    configs.append(config)
    return configs