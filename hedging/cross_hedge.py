import pandas as pd

def cross_hedge_scenarios(current_price, shares, scenarios, beta, hedge_ratio=1.0):
    position_value = current_price * shares
    hedge_notional = position_value * beta * hedge_ratio
    results = []
    for scenario in scenarios:
        scenario_price = current_price * (1 + scenario)
        stock_pnl = (scenario_price - current_price) * shares
        hedge_return = scenario / beta
        hedge_pnl = -hedge_notional * hedge_return
        hedged_pnl = stock_pnl + hedge_pnl
        hedge_improvement = hedged_pnl - stock_pnl
        if stock_pnl < 0:
            hedge_effectiveness = hedge_improvement / abs(stock_pnl)
        else:
            hedge_effectiveness = None
        results.append({
            "Scenario": scenario,
            "Scenario Price": scenario_price,
            "Unhedged P&L": stock_pnl,
            "Hedge Notional": hedge_notional,
            "Hedge Return": hedge_return,
            "Hedge P&L": hedge_pnl,
            "Cross Hedge P&L": hedged_pnl,
            "Hedge Improvement": hedge_improvement,
            "Hedge Effectiveness": hedge_effectiveness})
    return pd.DataFrame(results)

def calculate_cross_hedge_returns( stock_history, hedge_history, beta, hedge_ratio=1.0):
    stock_returns = (
        stock_history["Close"]
        .pct_change()
        .rename("Stock Return"))
    hedge_returns = (
        hedge_history["Close"]
        .pct_change()
        .rename("Hedge Return"))
    data = pd.concat([stock_returns, hedge_returns], axis=1).dropna()
    data["Hedge Position Return"] = -beta * hedge_ratio * data["Hedge Return"]
    data["Hedged Return"] = data["Stock Return"] + data["Hedge Position Return"]
    return data

def calculate_cross_hedge_metrics(cross_hedge_returns):
    stock_returns = cross_hedge_returns["Stock Return"]
    hedged_returns = cross_hedge_returns["Hedged Return"]
    unhedged_volatility = stock_returns.std() * (252 ** 0.5)
    hedged_volatility = hedged_returns.std() * (252 ** 0.5)
    volatility_reduction = 1 - hedged_volatility / unhedged_volatility
    return {
        "unhedged_volatility": unhedged_volatility,
        "hedged_volatility": hedged_volatility,
        "volatility_reduction": volatility_reduction}