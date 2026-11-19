import pandas as pd

def protective_put_scenarios(
    current_price,
    shares,
    scenarios,
    put_strike,
    put_premium_per_share,
    contract_size=100):
    number_of_contracts = shares // contract_size
    hedged_shares = number_of_contracts * contract_size
    unhedged_shares = shares - hedged_shares

    put_cost = put_premium_per_share * contract_size * number_of_contracts

    current_position_value = current_price * shares

    results = []

    for scenario in scenarios:
        scenario_price = current_price * (1 + scenario)

        scenario_position_value = scenario_price * shares

        unhedged_position_pnl = scenario_position_value - current_position_value

        put_payoff_per_share = max(put_strike - scenario_price, 0)

        put_payoff_total = put_payoff_per_share * contract_size * number_of_contracts

        stock_pnl_hedged_shares = (scenario_price - current_price) * hedged_shares

        stock_pnl_unhedged_shares = (scenario_price - current_price) * unhedged_shares

        hedged_position_pnl = stock_pnl_hedged_shares + stock_pnl_unhedged_shares + put_payoff_total - put_cost

        hedge_improvement = hedged_position_pnl - unhedged_position_pnl

        if unhedged_position_pnl < 0:
            hedge_effectiveness = hedge_improvement / abs(unhedged_position_pnl)
        else:
            hedge_effectiveness = None

        results.append({
            "Scenario": scenario,
            "Scenario Price": scenario_price,
            "Position Value": scenario_position_value,
            "Unhedged P&L": unhedged_position_pnl,
            "Put Payoff": put_payoff_total,
            "Put Cost": put_cost,
            "Hedged P&L": hedged_position_pnl,
            "Hedge Improvement": hedge_improvement,
            "Hedge Effectiveness": hedge_effectiveness})

    return pd.DataFrame(results)