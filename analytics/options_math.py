"""analytics/options_math.py: hedge research implementation.

Functions are documented at their point of definition.
Historical outputs are research estimates, not executable orders.
"""

from vollib.black_scholes_merton.implied_volatility import implied_volatility

from vollib.black_scholes_merton.greeks.analytical import delta, gamma, theta, vega

def calculate_option_analytics(
    option_type,
    option_price,
    stock_price,
    strike,
    time_to_expiry,
    risk_free_rate,
    dividend_yield):

    """Calculate Black–Scholes–Merton implied volatility and option Greeks.

    Args:
        option_type: `Call` or `Put` (case-sensitive).
        option_price: Observed option premium in the underlying quote currency.
        stock_price: Positive underlying reference price.
        strike: Contract strike in the same currency as the stock price.
        time_to_expiry: Positive calendar year fraction until expiration.
        risk_free_rate: Continuously compounded annual risk-free rate.
        dividend_yield: Continuously compounded annual dividend yield.

    Returns:
        Mapping with implied volatility, delta, gamma, theta and vega.

    Notes:
        A Black–Scholes–Merton approximation is not a contract-level execution
        quote. Pay attention to the Greek units used by py_vollib.
    """
    if option_type == "Call":
        flag = "c"
    elif option_type == "Put":
        flag = "p"
    else:
        raise ValueError("option_type must be either 'Call' or 'Put'.")

    iv = implied_volatility(
        option_price,
        stock_price,
        strike,
        time_to_expiry,
        risk_free_rate,
        dividend_yield,
        flag)

    option_delta = delta(
        flag,
        stock_price,
        strike,
        time_to_expiry,
        risk_free_rate,
        iv,
        dividend_yield)

    option_gamma = gamma(
        flag,
        stock_price,
        strike,
        time_to_expiry,
        risk_free_rate,
        iv,
        dividend_yield)

    option_theta = theta(
        flag,
        stock_price,
        strike,
        time_to_expiry,
        risk_free_rate,
        iv,
        dividend_yield)

    option_vega = vega(
        flag,
        stock_price,
        strike,
        time_to_expiry,
        risk_free_rate,
        iv,
        dividend_yield)

    return {
        "Implied Volatility": iv,
        "Delta": option_delta,
        "Gamma": option_gamma,
        "Theta": option_theta,
        "Vega": option_vega}