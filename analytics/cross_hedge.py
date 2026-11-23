"""Comparing cross-hedge candidates and laod their historical prices. 

Input: 
Stock and hedge price histories, hedge RICs, dates and a history loader. 

Output: 
A ranked hedge comparaison and a directionary of available price histories 
"""

import pandas as pd

from analytics.beta import estimate_beta

"""Compare candidate hedge assets using historical return statistics.

Input: 
Stock price DataFrame and a dictionary of hedge price DataFrames, each containing a "close" column 

Output:
DataFrame of hedge metrics, sorted by descedning R-squared and then desceding correlation
"""
def compare_cross_hedge_candidates(
    stock_history,
    hedge_histories):
    stock_returns = (
        stock_history["Close"]
        .pct_change()
        .dropna())
    results = []
    for hedge_name, hedge_history in (hedge_histories.items()):
        hedge_returns = (
            hedge_history["Close"]
            .pct_change()
            .dropna())
        metrics = estimate_beta(stock_returns=stock_returns, hedge_returns=hedge_returns)
        results.append({
            "Hedge": hedge_name,
            "Alpha": metrics["alpha"],
            "Beta": metrics["beta"],
            "Correlation": (metrics["correlation"]),
            "R Squared": (metrics["r_squared"]),
            "Residual Volatility": (metrics["residual_volatility"]),
            "Observations": (metrics["observations"])})
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(
        by=["R Squared", "Correlation"],
        ascending=[False, False])
    return results_df

"""Load historical prices for candidate hedge insturments

Input: 
Hedge RICs, start/end dates, and a price-history laoder function. 

Output: 
Dicitionary mapping RICs to nonempty price histories (empty results and failed requests are skipped)
"""
def load_cross_hedge_histories(hedge_rics, start_date, end_date, history_loader):
    histories = {}
    for ric in hedge_rics:
        try:
            history = history_loader( ric=ric, start_date=start_date, end_date=end_date)
            if history.empty:
                continue
            histories[ric] = history
        except Exception as error:
            print(f"Skipping hedge candidate {ric}: {error}")
    return histories