"""Price options and simulate portfolio hedging strategies

Input:
Historical prices or paired returns, position and option details, valuation dates, horizons, annual rates, and simulation settings

Output:
Option values, maximum drawdowns, simulated portfolio paths, and strategy comparison summaries
"""

import numpy as np
import pandas as pd

from scipy.stats import norm


def black_scholes_put(
    stock_price,
    strike,
    time_to_expiry,
    risk_free_rate,
    dividend_yield,
    volatility
):
    """Price a European put using the Black-Scholes-Merton model
    
    Input:
    Stock price, strike, expiry in years, annual risk-free and dividend rates, and annual volatility, with rates and volatility as decimals 

    Output:
    Non-negative theoretical put price per share 
    """
    if time_to_expiry <= 0:
        return max(strike - stock_price, 0.0)

    if volatility <= 0:
        return max(
            strike * np.exp(-risk_free_rate * time_to_expiry)
            - stock_price * np.exp(-dividend_yield * time_to_expiry),
            0.0
        )

    d1 = (
        np.log(stock_price / strike)
        + (
            risk_free_rate
            - dividend_yield
            + 0.5 * volatility**2
        ) * time_to_expiry
    ) / (volatility * np.sqrt(time_to_expiry))

    d2 = d1 - volatility * np.sqrt(time_to_expiry)

    put_price = (
        strike
        * np.exp(-risk_free_rate * time_to_expiry)
        * norm.cdf(-d2)
        - stock_price
        * np.exp(-dividend_yield * time_to_expiry)
        * norm.cdf(-d1)
    )

    return float(max(put_price, 0.0))


def calculate_max_drawdown(portfolio_values):
    """Calculate the largest peak-to-trough decline in portfolio value
    
    Input: 
    Sequence of positive portfolio values; missing entires are removed 

    Output:
    Drawdown as a nonpositive decimal, or NaN if no value remain
    """
    values = pd.Series(
        portfolio_values,
        dtype=float
    ).dropna()

    if values.empty:
        return float("nan")

    if (values <= 0).any():
        raise ValueError(
            "Portfolio values must be positive "
            "to calculate percentage drawdown."
        )

    running_peak = values.cummax()
    drawdown = values / running_peak - 1.0

    return float(drawdown.min())


def protective_put_portfolio_backtest(
    historical_prices,
    current_price,
    shares,
    put_strike,
    put_premium_per_share,
    put_iv,
    put_expiry,
    valuation_date,
    hedge_horizon_days,
    risk_free_rate,
    dividend_yield
):
    """Replay recent historical returns fro a stock protected by a put
    
    Input:
    Stock price history, current position, put terms and premium, valuation/expiry dates, horison in business-day steps, and annual rates

    Output:
    Dictionary containing backtest_df with prices, values, returns, and drawdowns incl. metrics comparing hedged and unhedfed results 
    """
    prices = pd.to_numeric(
        pd.Series(historical_prices),
        errors="coerce"
    ).dropna()

    if len(prices) < hedge_horizon_days + 1:
        raise ValueError(
            "Insufficient historical prices "
            "for the selected hedge horizon."
        )

    if (prices <= 0).any():
        raise ValueError(
            "Historical stock prices must be positive."
        )

    if (
        current_price <= 0
        or shares <= 0
        or put_strike <= 0
        or put_premium_per_share < 0
        or put_iv < 0
        or hedge_horizon_days < 1
    ):
        raise ValueError("Invalid protective put parameters.")

    historical_returns = (
        prices.pct_change()
        .dropna()
        .tail(hedge_horizon_days)
        .to_numpy(dtype=float)
    )

    dates = pd.bdate_range(
        start=pd.Timestamp(valuation_date).normalize(),
        periods=hedge_horizon_days + 1
    )

    expiry = pd.Timestamp(put_expiry).normalize()

    if dates[-1] > expiry:
        raise ValueError(
            "The simulated hedge horizon exceeds "
            "the option expiry date."
        )

    stock_prices = np.empty(hedge_horizon_days + 1)
    stock_prices[0] = current_price

    stock_prices[1:] = (
        current_price
        * np.cumprod(1.0 + historical_returns)
    )

    put_values = np.empty(hedge_horizon_days + 1)

    # At entry, use the actual purchase premium.
    put_values[0] = put_premium_per_share

    for i in range(1, len(dates)):
        time_to_expiry = max(
            (expiry - dates[i]).days / 365.0,
            0.0
        )

        put_values[i] = calibrated_option_value(
            option_type="put",
            stock_price=stock_prices[i],
            initial_stock_price=current_price,
            strike=put_strike,
            market_premium=put_premium_per_share,
            volatility=put_iv,
            expiry=put_expiry,
            valuation_date=valuation_date,
            current_date=dates[i],
            risk_free_rate=risk_free_rate,
            dividend_yield=dividend_yield
        )

    initial_position_value = shares * current_price

    portfolio_values = shares * (
        stock_prices
        + put_values
        - put_premium_per_share
    )

    unhedged_values = shares * stock_prices

    backtest_df = pd.DataFrame({
        "Date": dates,
        "Stock Price": stock_prices,
        "Put Value": put_values,
        "Unhedged Value": unhedged_values,
        "Protective Put Value": portfolio_values
    })

    backtest_df["Unhedged Return"] = (
        backtest_df["Unhedged Value"]
        / initial_position_value
        - 1.0
    )

    backtest_df["Protective Put Return"] = (
        backtest_df["Protective Put Value"]
        / initial_position_value
        - 1.0
    )

    backtest_df["Unhedged Drawdown"] = (
        backtest_df["Unhedged Value"]
        / backtest_df["Unhedged Value"].cummax()
        - 1.0
    )

    backtest_df["Protective Put Drawdown"] = (
        backtest_df["Protective Put Value"]
        / backtest_df["Protective Put Value"].cummax()
        - 1.0
    )

    metrics = {
        "Unhedged Maximum Drawdown":
            calculate_max_drawdown(unhedged_values),

        "Protective Put Maximum Drawdown":
            calculate_max_drawdown(portfolio_values),

        "Unhedged Final Return":
            float(backtest_df["Unhedged Return"].iloc[-1]),

        "Protective Put Final Return":
            float(backtest_df["Protective Put Return"].iloc[-1])
    }

    return {
        "backtest_df": backtest_df,
        "metrics": metrics
    }


def black_scholes_call(
    stock_price,
    strike,
    time_to_expiry,
    risk_free_rate,
    dividend_yield,
    volatility
):
    """Price a European call using the Black-Scholes-Merton model
    
    Input: 
    Stock price, stirke, expiry in years, annual risk-free and dividend rates, and annual volatility, with rates and volatility as decimals

    Output:
    Non-negative theoretical call price per share 
    """
    if time_to_expiry <= 0:
        return max(stock_price - strike, 0.0)

    if volatility <= 0:
        return max(
            stock_price * np.exp(-dividend_yield * time_to_expiry)
            - strike * np.exp(-risk_free_rate * time_to_expiry),
            0.0
        )

    d1 = (
        np.log(stock_price / strike)
        + (
            risk_free_rate
            - dividend_yield
            + 0.5 * volatility**2
        ) * time_to_expiry
    ) / (volatility * np.sqrt(time_to_expiry))

    d2 = d1 - volatility * np.sqrt(time_to_expiry)

    call_price = (
        stock_price
        * np.exp(-dividend_yield * time_to_expiry)
        * norm.cdf(d1)
        - strike
        * np.exp(-risk_free_rate * time_to_expiry)
        * norm.cdf(d2)
    )

    return float(max(call_price, 0.0))


def option_strategy_path_simulation(
    historical_prices,
    current_price,
    shares,
    legs,
    valuation_date,
    hedge_horizon_days,
    risk_free_rate,
    dividend_yield,
    strategy_name, 
    scenario_returns=None
):
    """Simualte a stock postion with long and/or short option legs
    
    Input:
    Price history, position, dates, horizon, annual rates, strategy name, and optional daily scenario returns. Each leg supplies type (put/call), side (1 long, -1 short), strike, premium per share, iv, and expiry. 

    Output:
    Dictionary containing backtest_df with portfolio values and returns, plus maximum-drawdown and final-return metrics 
    """
    prices = pd.to_numeric(
        pd.Series(historical_prices),
        errors="coerce"
    ).dropna()

    if (
        hedge_horizon_days < 1
        or len(prices) < hedge_horizon_days + 1
    ):
        raise ValueError("Insufficient historical prices.")

    if (
        current_price <= 0
        or shares <= 0
        or (prices <= 0).any()
        or not legs
    ):
        raise ValueError("Invalid portfolio inputs.")

    if scenario_returns is None:
        returns = (
            prices.pct_change()
            .dropna()
            .tail(hedge_horizon_days)
            .to_numpy(dtype=float)
        )
    else:
        returns = np.asarray(
            scenario_returns,
            dtype=float
        )

        if (
            len(returns) != hedge_horizon_days
            or not np.isfinite(returns).all()
            or (returns <= -1.0).any()
        ):
            raise ValueError(
                "scenario_returns must contain "
                "one valid return per trading day."
            )

    dates = pd.bdate_range(
        start=pd.Timestamp(valuation_date).normalize(),
        periods=hedge_horizon_days + 1
    )

    stock_prices = np.r_[
        current_price,
        current_price * np.cumprod(1.0 + returns)
    ]

    net_premium = 0.0

    for leg in legs:
        if leg["type"] not in ("put", "call"):
            raise ValueError("Invalid option type.")

        if leg["side"] not in (-1, 1):
            raise ValueError("Option side must be 1 or -1.")

        if (
            leg["strike"] <= 0
            or leg["iv"] < 0
            or leg["premium"] < 0
        ):
            raise ValueError("Invalid option parameters.")

        if dates[-1] > pd.Timestamp(leg["expiry"]).normalize():
            raise ValueError(
                "Simulation horizon exceeds option expiry."
            )

        net_premium += leg["side"] * leg["premium"]

    # Pay the initial net premium from cash.
    # At entry, the quoted option value equals the
    # premium paid or received, so portfolio value
    # starts at shares * current_price.
    portfolio_values = np.empty(len(dates))
    portfolio_values[0] = shares * current_price

    for i in range(1, len(dates)):
        option_value = 0.0

        for leg in legs:
            expiry = pd.Timestamp(
                leg["expiry"]
            ).normalize()

            time_to_expiry = max(
                (expiry - dates[i]).days / 365.0,
                0.0
            )

            theoretical_price = calibrated_option_value(
                option_type=leg["type"],
                stock_price=stock_prices[i],
                initial_stock_price=current_price,
                strike=leg["strike"],
                market_premium=leg["premium"],
                volatility=leg["iv"],
                expiry=leg["expiry"],
                valuation_date=valuation_date,
                current_date=dates[i],
                risk_free_rate=risk_free_rate,
                dividend_yield=dividend_yield
            )

            option_value += (
                leg["side"] * theoretical_price
            )

        portfolio_values[i] = shares * (
            stock_prices[i]
            + option_value
            - net_premium
        )

    unhedged_values = shares * stock_prices
    initial_value = shares * current_price

    backtest_df = pd.DataFrame({
        "Date": dates,
        "Stock Price": stock_prices,
        "Unhedged Value": unhedged_values,
        "Hedged Value": portfolio_values
    })

    backtest_df["Unhedged Return"] = (
        unhedged_values / initial_value - 1.0
    )

    backtest_df["Hedged Return"] = (
        portfolio_values / initial_value - 1.0
    )

    metrics = {
        "Strategy": strategy_name,
        "Unhedged Maximum Drawdown":
            calculate_max_drawdown(unhedged_values),
        "Hedged Maximum Drawdown":
            calculate_max_drawdown(portfolio_values),
        "Unhedged Final Return":
            float(backtest_df["Unhedged Return"].iloc[-1]),
        "Hedged Final Return":
            float(backtest_df["Hedged Return"].iloc[-1])
    }

    return {
        "backtest_df": backtest_df,
        "metrics": metrics
    }


def simulate_multiple_hedge_paths(
    historical_prices,
    current_price,
    shares,
    strategies,
    valuation_date,
    hedge_horizon_days,
    risk_free_rate,
    dividend_yield,
    n_simulations=200,
    block_size=5,
    random_seed=42
):
    """Compare option strategies across block-resampled stock-return paths
    
    Input: 
    Stock price history, position, strategy-to-leg mapping, dates, horizon, annual rates, simulation count, block size, and random seed. 

    Output:
    Dictionary with simulation_df of drawdowns and final returns, and summary_df of strategy medians
    """
    prices = pd.to_numeric(
        pd.Series(historical_prices),
        errors="coerce"
    ).dropna()

    historical_returns = (
        prices.pct_change()
        .dropna()
        .to_numpy(dtype=float)
    )

    if (
        n_simulations < 1
        or block_size < 1
        or len(historical_returns) < block_size
    ):
        raise ValueError(
            "Invalid simulation count, block size, "
            "or insufficient historical data."
        )

    rng = np.random.default_rng(random_seed)
    results = []

    for simulation_id in range(n_simulations):
        simulated_returns = []

        # Sample blocks of consecutive historical returns.
        while len(simulated_returns) < hedge_horizon_days:
            start = rng.integers(
                0,
                len(historical_returns) - block_size + 1
            )

            block = historical_returns[
                start:start + block_size
            ]

            simulated_returns.extend(block.tolist())

        simulated_returns = np.asarray(
            simulated_returns[:hedge_horizon_days],
            dtype=float
        )

        # Unhedged benchmark using the same simulated returns
        unhedged_prices = current_price * np.r_[
            1.0,
            np.cumprod(1.0 + simulated_returns)
        ]

        unhedged_values = shares * unhedged_prices

        unhedged_drawdown = calculate_max_drawdown(
            unhedged_values
        )

        unhedged_final_return = (
            unhedged_values[-1] / unhedged_values[0] - 1.0
        )

        results.append({
            "Simulation": simulation_id,
            "Strategy": "Unhedged",
            "Maximum Drawdown": unhedged_drawdown,
            "Final Return": float(unhedged_final_return)
        })

        for strategy_name, legs in strategies.items():
            simulation = option_strategy_path_simulation(
                historical_prices=prices,
                current_price=current_price,
                shares=shares,
                legs=legs,
                valuation_date=valuation_date,
                hedge_horizon_days=hedge_horizon_days,
                risk_free_rate=risk_free_rate,
                dividend_yield=dividend_yield,
                strategy_name=strategy_name,
                scenario_returns=simulated_returns
            )

            metrics = simulation["metrics"]

            results.append({
                "Simulation": simulation_id,
                "Strategy": strategy_name,
                "Maximum Drawdown": metrics[
                    "Hedged Maximum Drawdown"
                ],
                "Final Return": metrics[
                    "Hedged Final Return"
                ]
            })

    simulation_df = pd.DataFrame(results)

    summary_df = (
        simulation_df
        .groupby("Strategy")
        .agg(
            Median_Drawdown=("Maximum Drawdown", "median"),
            Worst_5pct_Drawdown=(
                "Maximum Drawdown",
                lambda x: x.quantile(0.05)
            ),
            Median_Final_Return=("Final Return", "median"),
            Final_Return_5pct=(
                "Final Return",
                lambda x: x.quantile(0.05)
            )
        )
        .reset_index()
    )

    return {
        "simulation_df": simulation_df,
        "summary_df": summary_df
    }


def calibrated_option_value(
    option_type,
    stock_price,
    initial_stock_price,
    strike,
    market_premium,
    volatility,
    expiry,
    valuation_date,
    current_date,
    risk_free_rate,
    dividend_yield
):
    """Adjust an option model value to its initial market premium
    
    Input:
    Option type, current/initial stock prices, strike, market premium, volatility, expiry/valuation/current dates, and annual rates

    Output:
    Non-negative option value per share with an entry-price adjustment that fades linearly to zero at expiry 
    """
    pricing_function = (
        black_scholes_put
        if option_type == "put"
        else black_scholes_call
    )

    expiry = pd.Timestamp(expiry).normalize()
    valuation_date = pd.Timestamp(valuation_date).normalize()
    current_date = pd.Timestamp(current_date).normalize()

    initial_days = (expiry - valuation_date).days
    remaining_days = max((expiry - current_date).days, 0)

    if initial_days <= 0:
        raise ValueError("Option must expire after valuation date.")

    initial_time = initial_days / 365.0
    remaining_time = remaining_days / 365.0

    model_entry_price = pricing_function(
        initial_stock_price,
        strike,
        initial_time,
        risk_free_rate,
        dividend_yield,
        volatility
    )

    model_current_price = pricing_function(
        stock_price,
        strike,
        remaining_time,
        risk_free_rate,
        dividend_yield,
        volatility
    )

    entry_difference = market_premium - model_entry_price

    adjustment = (
        entry_difference
        * remaining_days
        / initial_days
    )

    return float(max(model_current_price + adjustment, 0.0))


def simulate_cross_hedge_paths(
    cross_hedge_returns,
    hedge_beta,
    hedge_ratio=1.0,
    n_simulations=1000,
    hedge_horizon_days=63,
    block_size=10,
    random_seed=42
):
    """Compare unhedged and daily-rebalanced cross-hedge return paths
    
    Input: 
    DataFrame with paired "Stock Return" and "Hedge Return" columns, hedge beta/ratio, simulatino count, horizon, block size, and random seed

    Output:
    Dictionary with simulation_df of drawdowns and final returns, and summary_df of strategy medians and lower 5%
    """
    paired_returns = (
        cross_hedge_returns[
            ["Stock Return", "Hedge Return"]
        ]
        .apply(pd.to_numeric, errors="coerce")
        .dropna()
    )

    if (
        len(paired_returns) < block_size
        or block_size < 1
        or n_simulations < 1
        or hedge_horizon_days < 1
        or not np.isfinite(hedge_beta)
        or not np.isfinite(hedge_ratio)
    ):
        raise ValueError("Invalid cross-hedge simulation inputs.")

    data = paired_returns.to_numpy(dtype=float)
    rng = np.random.default_rng(random_seed)

    results = []

    for simulation_id in range(n_simulations):
        blocks = []

        while sum(len(block) for block in blocks) < hedge_horizon_days:
            start = rng.integers(
                0,
                len(data) - block_size + 1
            )

            blocks.append(data[start:start + block_size])

        sampled = np.vstack(blocks)[:hedge_horizon_days]

        stock_returns = sampled[:, 0]
        etf_returns = sampled[:, 1]

        hedged_returns = (
            stock_returns
            - hedge_ratio * hedge_beta * etf_returns
        )

        if (
            (stock_returns <= -1).any()
            or (hedged_returns <= -1).any()
        ):
            raise ValueError(
                "Simulated daily return is -100% or lower."
            )

        stock_values = np.r_[
            1.0,
            np.cumprod(1.0 + stock_returns)
        ]

        hedged_values = np.r_[
            1.0,
            np.cumprod(1.0 + hedged_returns)
        ]

        for name, values in [
            ("Unhedged", stock_values),
            ("Cross Hedge", hedged_values)
        ]:
            results.append({
                "Simulation": simulation_id,
                "Strategy": name,
                "Maximum Drawdown":
                    calculate_max_drawdown(values),
                "Final Return":
                    float(values[-1] - 1.0)
            })

    simulation_df = pd.DataFrame(results)

    summary_df = (
        simulation_df
        .groupby("Strategy")
        .agg(
            Median_Drawdown=("Maximum Drawdown", "median"),
            Worst_5pct_Drawdown=(
                "Maximum Drawdown",
                lambda x: x.quantile(0.05)
            ),
            Median_Final_Return=("Final Return", "median"),
            Final_Return_5pct=(
                "Final Return",
                lambda x: x.quantile(0.05)
            )
        )
        .reset_index()
    )

    return {
        "simulation_df": simulation_df,
        "summary_df": summary_df
    }



def simulate_unified_hedge_paths(
    historical_prices,
    paired_returns,
    current_price,
    shares,
    strategies,
    hedge_beta,
    valuation_date,
    hedge_horizon_days,
    risk_free_rate,
    dividend_yield,
    hedge_ratio=1.0,
    n_simulations=1000,
    block_size=10,
    random_seed=42
):
    """Compare unhedge, cross-hedge, and option strategies on shared paths
    
    Input:
    Price history, paired stock/hedge returns, position, option strategies, beta/ratio, dates, horizon, rates, and block-resampling settings. The cross hedge uses daily-rebalanced returns.

    Output:
    Dictionary with simulation_df of strategy results and summary_df of median outcomes, lower 5% and tail-average final returns
    """
    required_columns = ["Stock Return", "Hedge Return"]

    paired = paired_returns[required_columns].copy()

    paired = paired_returns[required_columns].copy()

    paired = paired.apply(
        pd.to_numeric,
        errors="coerce"
    )

    paired = paired.astype("float64").dropna()

    historical_data = paired.to_numpy(
        dtype=np.float64
    )

    if not np.isfinite(historical_data).all():
        raise ValueError("Non-finite paired returns.")

    if not (
        1 <= block_size <= len(paired)
        and n_simulations >= 1
        and hedge_horizon_days >= 1
        and current_price > 0
        and shares > 0
        and np.isfinite(hedge_beta)
        and np.isfinite(hedge_ratio)
    ):
        raise ValueError("Invalid unified simulation inputs.")

    if (
        paired["Stock Return"].le(-1).any()
        or paired["Hedge Return"].le(-1).any()
    ):
        raise ValueError("Invalid historical daily returns.")

    prices = pd.to_numeric(
        pd.Series(historical_prices),
        errors="coerce"
    ).dropna()

    rng = np.random.default_rng(random_seed)

    n_observations = len(historical_data)

    strategy_results = []

    for simulation_id in range(n_simulations):
        sampled_blocks = []
        sampled_length = 0

        while sampled_length < hedge_horizon_days:
            start = rng.integers(
                0,
                n_observations - block_size + 1
            )

            block = historical_data[
                start:start + block_size
            ]

            sampled_blocks.append(block)
            sampled_length += len(block)

        sampled = np.vstack(
            sampled_blocks
        )[:hedge_horizon_days]

        stock_returns = sampled[:, 0]
        etf_returns = sampled[:, 1]

        # Same stock path for every strategy.
        stock_growth = np.r_[
            1.0,
            np.cumprod(1.0 + stock_returns)
        ]

        initial_value = current_price * shares
        unhedged_values = initial_value * stock_growth

        # Daily-rebalanced return-based cross hedge.
        cross_returns = (
            stock_returns
            - hedge_ratio * hedge_beta * etf_returns
        )

        if np.any(cross_returns <= -1):
            raise ValueError(
                "Cross-hedge daily return is -100% or lower."
            )

        cross_growth = np.r_[
            1.0,
            np.cumprod(1.0 + cross_returns)
        ]

        cross_values = initial_value * cross_growth

        portfolio_paths = {
            "Unhedged": unhedged_values,
            "Cross Hedge": cross_values
        }

        # Evaluate option strategies on identical AAPL returns.
        for strategy_name, legs in strategies.items():
            simulation = option_strategy_path_simulation(
                historical_prices=prices,
                current_price=current_price,
                shares=shares,
                legs=legs,
                valuation_date=valuation_date,
                hedge_horizon_days=hedge_horizon_days,
                risk_free_rate=risk_free_rate,
                dividend_yield=dividend_yield,
                strategy_name=strategy_name,
                scenario_returns=stock_returns
            )

            hedged_values = (
                simulation["backtest_df"]["Hedged Value"]
                .to_numpy(dtype=float)
            )

            portfolio_paths[strategy_name] = hedged_values

        for strategy_name, values in portfolio_paths.items():
            if (
                len(values) != hedge_horizon_days + 1
                or not np.isfinite(values).all()
                or np.any(values <= 0)
            ):
                raise ValueError(
                    f"Invalid portfolio path: {strategy_name}"
                )

            final_return = values[-1] / values[0] - 1.0

            strategy_results.append({
                "Simulation": simulation_id,
                "Strategy": strategy_name,
                "Maximum Drawdown":
                    calculate_max_drawdown(values),
                "Final Return": float(final_return)
            })

    simulation_df = pd.DataFrame(strategy_results)

    def calculate_cvar(series, alpha=0.05):
        """Average returns at or below the selected lower-tail quantile (*cut point that divides a sorted dataset into equal sizes) 
        
        Input: 
        Return Series and tail fraction alpha (default: 0.05)

        Output: 
        Mean tail return as a float; NaN if no returns are selcted 
        """
        cutoff = series.quantile(alpha)
        return float(series[series <= cutoff].mean())

    summary_df = (
        simulation_df
        .groupby("Strategy")
        .agg(
            Median_Drawdown=(
                "Maximum Drawdown", "median"
            ),
            Worst_5pct_Drawdown=(
                "Maximum Drawdown",
                lambda x: x.quantile(0.05)
            ),
            Median_Final_Return=(
                "Final Return", "median"
            ),
            VaR_5pct=(
                "Final Return",
                lambda x: x.quantile(0.05)
            ),
            CVaR_5pct=(
                "Final Return",
                calculate_cvar
            )
        )
        .reset_index()
    )

    return {
        "simulation_df": simulation_df,
        "summary_df": summary_df
    }
