"""Calculate rolling returns and option-hedged scenario returns. 

Input:
Historical prices or return Series, horizons, current stock price, and option stirkes and premiums per share 

Output: 
Series of rolling returns and hedged returns using option payoffs at expiry
"""

import pandas as pd

def calculate_rolling_returns(history, horizon_days):
    """calculate historical price returns over a rolling horizon
    
    Input:
    Price DataFrame containing a "Close" column and horizon in observations.
    
    Output:
    Series of horizon returns as decimals, with missing results removed
    """
    rolling_returns = (
        history["Close"]
        .pct_change(periods=horizon_days)
        .dropna())
    return rolling_returns

def protective_put_backtest(stock_returns, current_price, put_strike, put_premium_per_share):
    """Calculate stock returns with a purchased put held to expire
    
    Input:
    Stock return Series as decimals, current stock price, put strike, and put premium per share 
    
    Output: 
    Series of stock-plus-put returns after deducting the put premium, with the same index as the input returns
    """
    strike_ratio = put_strike / current_price
    premium_ratio = put_premium_per_share / current_price
    results = []
    for stock_return in stock_returns:
        terminal_price_ratio = 1 + stock_return
        put_payoff_ratio = max(strike_ratio - terminal_price_ratio, 0)
        hedged_return = stock_return + put_payoff_ratio - premium_ratio
        results.append(hedged_return)
    return pd.Series(results, index=stock_returns.index, name="Protective Put Return")

def collar_backtest(stock_returns, current_price, put_strike, put_premium_per_share, call_strike, call_premium_per_share):
    """Calculate stock returns with a pruchased put and sold call at expiry
    
    Input:
    Stock return Series as decimals, current stock price, put and call strikes, and their premiums per share 
    
    Output:
    Series of collar returns including option payoffs and net premiums, with the same index as the input returns
    """
    put_strike_ratio = put_strike / current_price
    call_strike_ratio = call_strike / current_price
    put_premium_ratio = put_premium_per_share / current_price
    call_premium_ratio = call_premium_per_share / current_price
    results = []
    for stock_return in stock_returns:
        terminal_price_ratio =  1 + stock_return
        put_payoff_ratio = max(put_strike_ratio - terminal_price_ratio, 0)
        call_payoff_ratio = max(terminal_price_ratio - call_strike_ratio, 0)
        hedged_return = stock_return + put_payoff_ratio - call_payoff_ratio - put_premium_ratio + call_premium_ratio
        results.append(hedged_return)

    return pd.Series(results, index=stock_returns.index, name="Collar Return")

def put_spread_backtest(stock_returns, current_price, long_put_strike, long_put_premium_per_share, short_put_strike, short_put_premium_per_share):
    """Calculate stock returns with a purchased put and sold put at expiry 
    
    Input:
    Stock return Sereis as decimals, current stock price, long and short put strikes, and their premiums per share 
    
    Output:
    Series of stock-plus-put-spread returns including net premiums, with the same index as the input returns """
    long_strike_ratio = long_put_strike / current_price
    short_strike_ratio = short_put_strike / current_price
    long_premium_ratio = long_put_premium_per_share / current_price
    short_premium_ratio = short_put_premium_per_share / current_price
    results = []
    for stock_return in stock_returns:
        terminal_price_ratio = 1 + stock_return
        long_put_payoff = max(long_strike_ratio - terminal_price_ratio, 0)
        short_put_payoff = max(short_strike_ratio - terminal_price_ratio, 0)
        hedged_return = stock_return + long_put_payoff - short_put_payoff - long_premium_ratio + short_premium_ratio
        results.append(hedged_return)

    return pd.Series(results, index=stock_returns.index, name="Put Spread Return")

def calculate_rolling_compounded_returns(
    returns,
    horizon_days
):
    """Compound returns within each rolling window 
    
    Input:
    Return Series as decimal and a rolling window lenght in observations 
    
    Output:
    Series of compounded window returns, with missing results removed 
    """
    rolling_returns = (
        (1 + returns)
        .rolling(
            window=horizon_days
        )
        .apply(
            lambda values:
                values.prod() - 1,
            raw=False
        )
        .dropna()
    )

    return rolling_returns