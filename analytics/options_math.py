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