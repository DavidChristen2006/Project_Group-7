import pandas as pd


def put_spread_scenarios(
    current_price,
    shares,
    scenarios,
    long_put_strike,
    long_put_premium_per_share,
    short_put_strike,
    short_put_premium_per_share,
    contract_size=100):
    number_of_contracts = shares // contract_size
    hedged_shares = number_of_contracts * contract_size
    unhedged_shares = shares - hedged_shares

    long_put_cost = long_put_premium_per_share * contract_size * number_of_contracts

    short_put_premium_received = short_put_premium_per_share * contract_size * number_of_contracts

    net_hedge_cost =long_put_cost - short_put_premium_received

    current_position_value = current_price * shares

    results = []

    for scenario in scenarios:
        scenario_price = current_price * (1 + scenario)

        scenario_position_value = scenario_price * shares

        unhedged_position_pnl = scenario_position_value - current_position_value

        long_put_payoff_per_share = max(long_put_strike - scenario_price, 0)

        long_put_payoff_total = long_put_payoff_per_share * contract_size * number_of_contracts

        short_put_payoff_per_share = max(short_put_strike - scenario_price, 0)

        short_put_payoff_total = short_put_payoff_per_share * contract_size * number_of_contracts

        hedged_stock_pnl = (scenario_price - current_price) * hedged_shares

        unhedged_stock_pnl = (scenario_price - current_price) * unhedged_shares

        put_spread_pnl = (
            hedged_stock_pnl
            + unhedged_stock_pnl
            + long_put_payoff_total
            - short_put_payoff_total
            - long_put_cost
            + short_put_premium_received)

        hedge_improvement = put_spread_pnl - unhedged_position_pnl

        if unhedged_position_pnl < 0:
            hedge_effectiveness = hedge_improvement / abs(unhedged_position_pnl)
        else:
            hedge_effectiveness = None

        results.append({
            "Scenario": scenario,
            "Scenario Price": scenario_price,
            "Unhedged P&L": unhedged_position_pnl,
            "Long Put Payoff": long_put_payoff_total,
            "Short Put Payoff": short_put_payoff_total,
            "Long Put Cost": long_put_cost,
            "Short Put Premium": short_put_premium_received,
            "Net Hedge Cost": net_hedge_cost,
            "Put Spread P&L": put_spread_pnl,
            "Hedge Improvement": hedge_improvement,
            "Hedge Effectiveness": hedge_effectiveness})

    return pd.DataFrame(results)