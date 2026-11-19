import pandas as pd

def collar_scenarios(
    current_price,
    shares,
    scenarios,
    put_strike,
    put_premium_per_share,
    call_strike,
    call_premium_per_share,
    contract_size=100):
    number_of_contracts = shares // contract_size
    hedged_shares = number_of_contracts * contract_size
    unhedged_shares = shares - hedged_shares

    put_cost = put_premium_per_share * contract_size * number_of_contracts

    call_premium_received = call_premium_per_share * contract_size * number_of_contracts

    net_hedge_cost = put_cost - call_premium_received

    current_position_value = current_price * shares

    results = []

    for scenario in scenarios:
        scenario_price = current_price * (1 + scenario)

        scenario_position_value = scenario_price * shares

        unhedged_position_pnl = scenario_position_value - current_position_value

        put_payoff_per_share = max(put_strike - scenario_price, 0)

        put_payoff_total = put_payoff_per_share * contract_size * number_of_contracts

        call_payoff_per_share = max(scenario_price - call_strike, 0)

        call_payoff_total = call_payoff_per_share * contract_size * number_of_contracts

        hedged_stock_pnl = (scenario_price - current_price) * hedged_shares

        unhedged_stock_pnl = (scenario_price - current_price) * unhedged_shares

        collar_pnl = hedged_stock_pnl + unhedged_stock_pnl + put_payoff_total - call_payoff_total - put_cost + call_premium_received

        hedge_improvement = collar_pnl - unhedged_position_pnl

        if unhedged_position_pnl < 0:
            hedge_effectiveness = hedge_improvement / abs(unhedged_position_pnl)
        else:
            hedge_effectiveness = None

        results.append({
            "Scenario": scenario,
            "Scenario Price": scenario_price,
            "Unhedged P&L": unhedged_position_pnl,
            "Put Payoff": put_payoff_total,
            "Call Payoff": call_payoff_total,
            "Put Cost": put_cost,
            "Call Premium": call_premium_received,
            "Net Hedge Cost": net_hedge_cost,
            "Collar P&L": collar_pnl,
            "Hedge Improvement": hedge_improvement,
            "Hedge Effectiveness": hedge_effectiveness})

    return pd.DataFrame(results)