"""Analyse historical drawdowns and their effect on a stock position

Input: 
Price history with "Close" and "Drawdown" columns, and shares held 

Output:
Drawdown, position P&L, and recovery statistics
"""

def analyze_historical_risk(
    history,
    shares):
    """Measure the largest recorded drawdown and subsequent recovery.

    Input:
    Chronologically ordered, date-indexed DataFrame with "Close" and precomputed "Drawdown" columns, and the number of shares held

    Output: 
    Dictionary with drawdown, peak/trough details, position P&L, and recovery timing; recovery values are None if no recovery occurs
    """
    max_drawdown = history["Drawdown"].min()
    max_drawdown_date = history["Drawdown"].idxmin()
    peak_date = history.loc[:max_drawdown_date, "Close"].idxmax()
    drawdown_duration = max_drawdown_date - peak_date
    peak_price = history.loc[peak_date, "Close"]
    trough_price = history.loc[max_drawdown_date,"Close"]
    price_change = trough_price - peak_price
    position_pnl = price_change * shares
    position_value_peak = peak_price * shares
    position_pnl_pct = position_pnl / position_value_peak
    position_value_trough = trough_price * shares
    after_trough = history.loc[max_drawdown_date:]
    recovery_rows = after_trough[after_trough["Close"] >= peak_price]

    if not recovery_rows.empty:
        recovery_date = recovery_rows.index[0]
        recovery_duration = recovery_date - max_drawdown_date
    else:
        recovery_date = None
        recovery_duration = None

    return {
        "max_drawdown": max_drawdown,
        "max_drawdown_date": max_drawdown_date,
        "peak_date": peak_date,
        "drawdown_duration": drawdown_duration,
        "peak_price": peak_price,
        "trough_price": trough_price,
        "price_change": price_change,
        "position_pnl": position_pnl,
        "position_value_peak": position_value_peak,
        "position_pnl_pct": position_pnl_pct,
        "position_value_trough": position_value_trough,
        "recovery_date": recovery_date,
        "recovery_duration": recovery_duration}