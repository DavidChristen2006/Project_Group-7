import pandas as pd

from analytics.ml_dataset import build_ml_rows
from analytics.risk_metrics import calculate_return_metrics
from analytics.strategy_backtest import (
    calculate_rolling_returns,
    calculate_rolling_compounded_returns,
    protective_put_backtest,
    collar_backtest,
    put_spread_backtest
)

from pipeline.options_analysis import (
    prepare_option_universe,
    select_protective_put,
    select_collar,
    select_put_spread
)

from pipeline.strategy_candidates import (
    generate_protective_put_configs,
    generate_collar_configs,
    generate_put_spread_configs,
    generate_cross_hedge_configs
)

from pipeline.analysis import analyze_position

from lseg_api.equity import get_equity_history

from pipeline.cross_hedge_analysis import analyze_cross_hedges

def build_strategy_metrics_row(
    strategy_name,
    risk_metrics,
    hedge_cost_pct):
    return pd.DataFrame([
        {
            "Strategy": strategy_name,

            "Annualized Volatility":
                risk_metrics[
                    "annualized_volatility"
                ],

            "VaR":
                risk_metrics["var"],

            "CVaR":
                risk_metrics["cvar"],

            "Maximum Drawdown":
                risk_metrics[
                    "max_drawdown"
                ],

            "Worst Return":
                risk_metrics[
                    "worst_return"
                ],

            "Average Return":
                risk_metrics[
                    "average_return"
                ],

            "Hedge Cost %":
                hedge_cost_pct,

            "Correlation": None,
            "Beta": None,
            "R Squared": None
        }
    ])

def evaluate_strategy_configs(
    position,
    metadata,
    market,
    history,
    historical_risk,
    options,
    cross_analysis
):
    all_ml_rows = []

    historical_horizon_returns = (
        calculate_rolling_returns(
            history=history,
            horizon_days=(
                position.hedge_horizon_days
            )
        )
    )

    periods_per_year = (
        252
        / position.hedge_horizon_days
    )

    protective_put_configs = (
        generate_protective_put_configs()
    )

    for config in protective_put_configs:

        try:
            result = select_protective_put(
                position=position,
                market=market,
                options=options,

                target_put_strike_pct=(
                    config[
                        "put_strike_pct"
                    ]
                ),

                target_put_delta=(
                    config[
                        "put_delta"
                    ]
                ),

                selection_mode=(
                    config[
                        "put_selection_mode"
                    ]
                )
            )

            selected_put = (
                result["selected_put"]
            )

            put_returns = (
                protective_put_backtest(
                    stock_returns=(
                        historical_horizon_returns
                    ),

                    current_price=(
                        market.current_price
                    ),

                    put_strike=(
                        selected_put[
                            "StrikePrice"
                        ]
                    ),

                    put_premium_per_share=(
                        selected_put["ASK"]
                    )
                )
            )

            risk_metrics = (
                calculate_return_metrics(
                    put_returns,
                    periods_per_year=(
                        periods_per_year
                    ),
                    calculate_drawdown=False
                )
            )

            hedge_cost_pct = (
                selected_put["ASK"]
                / market.current_price
            )

            strategy_metrics_df = (
                build_strategy_metrics_row(
                    strategy_name=(
                        "Protective Put"
                    ),
                    risk_metrics=(
                        risk_metrics
                    ),
                    hedge_cost_pct=(
                        hedge_cost_pct
                    )
                )
            )

            ml_rows = build_ml_rows(
                position=position,
                metadata=metadata,
                market=market,
                historical_risk=(
                    historical_risk
                ),
                strategy_metrics_df=(
                    strategy_metrics_df
                ),
                hedge_config=config,
                protective_put=(
                    selected_put
                )
            )

            all_ml_rows.append(
                ml_rows
            )

        except Exception as error:
            print(
                "Skipping Protective Put "
                f"config {config}: {error}"
            )
    collar_configs = (
        generate_collar_configs()
    )

    for config in collar_configs:

        try:
            result = select_collar(
                position=position,
                market=market,
                options=options,

                target_put_strike_pct=(
                    config[
                        "put_strike_pct"
                    ]
                ),

                target_put_delta=(
                    config["put_delta"]
                ),

                target_call_strike_pct=(
                    config[
                        "call_strike_pct"
                    ]
                ),

                target_call_delta=(
                    config["call_delta"]
                ),

                put_selection_mode=(
                    config[
                        "put_selection_mode"
                    ]
                ),

                call_selection_mode=(
                    config[
                        "call_selection_mode"
                    ]
                )
            )

            selected_put = (
                result["selected_put"]
            )

            selected_call = (
                result["selected_call"]
            )

            collar_returns = (
                collar_backtest(
                    stock_returns=(
                        historical_horizon_returns
                    ),

                    current_price=(
                        market.current_price
                    ),

                    put_strike=(
                        selected_put[
                            "StrikePrice"
                        ]
                    ),

                    put_premium_per_share=(
                        selected_put["ASK"]
                    ),

                    call_strike=(
                        selected_call[
                            "StrikePrice"
                        ]
                    ),

                    call_premium_per_share=(
                        selected_call["BID"]
                    )
                )
            )

            risk_metrics = (
                calculate_return_metrics(
                    collar_returns,
                    periods_per_year=(
                        periods_per_year
                    ),
                    calculate_drawdown=False
                )
            )

            hedge_cost_pct = (
                (
                    selected_put["ASK"]
                    - selected_call["BID"]
                )
                / market.current_price
            )

            strategy_metrics_df = (
                build_strategy_metrics_row(
                    strategy_name="Collar",
                    risk_metrics=(
                        risk_metrics
                    ),
                    hedge_cost_pct=(
                        hedge_cost_pct
                    )
                )
            )

            ml_rows = build_ml_rows(
                position=position,
                metadata=metadata,
                market=market,
                historical_risk=(
                    historical_risk
                ),
                strategy_metrics_df=(
                    strategy_metrics_df
                ),
                hedge_config=config,
                collar_put=selected_put,
                collar_call=selected_call
            )

            all_ml_rows.append(
                ml_rows
            )

        except Exception as error:
            print(
                "Skipping Collar "
                f"config {config}: {error}"
            )

    put_spread_configs = (
        generate_put_spread_configs()
    )

    for config in put_spread_configs:

        try:
            result = select_put_spread(
                position=position,
                market=market,
                options=options,

                target_long_put_strike_pct=(
                    config[
                        "put_strike_pct"
                    ]
                ),

                target_long_put_delta=(
                    config["put_delta"]
                ),

                target_short_put_strike_pct=(
                    config[
                        "short_put_strike_pct"
                    ]
                ),

                target_short_put_delta=(
                    config[
                        "short_put_delta"
                    ]
                ),

                long_put_selection_mode=(
                    config[
                        "put_selection_mode"
                    ]
                ),

                short_put_selection_mode=(
                    config[
                        "short_put_selection_mode"
                    ]
                )
            )

            long_put = (
                result[
                    "selected_long_put"
                ]
            )

            short_put = (
                result[
                    "selected_short_put"
                ]
            )

            spread_returns = (
                put_spread_backtest(
                    stock_returns=(
                        historical_horizon_returns
                    ),

                    current_price=(
                        market.current_price
                    ),

                    long_put_strike=(
                        long_put[
                            "StrikePrice"
                        ]
                    ),

                    long_put_premium_per_share=(
                        long_put["ASK"]
                    ),

                    short_put_strike=(
                        short_put[
                            "StrikePrice"
                        ]
                    ),

                    short_put_premium_per_share=(
                        short_put["BID"]
                    )
                )
            )

            risk_metrics = (
                calculate_return_metrics(
                    spread_returns,
                    periods_per_year=(
                        periods_per_year
                    ),
                    calculate_drawdown=False
                )
            )

            hedge_cost_pct = (
                (
                    long_put["ASK"]
                    - short_put["BID"]
                )
                / market.current_price
            )

            strategy_metrics_df = (
                build_strategy_metrics_row(
                    strategy_name=(
                        "Put Spread"
                    ),
                    risk_metrics=(
                        risk_metrics
                    ),
                    hedge_cost_pct=(
                        hedge_cost_pct
                    )
                )
            )

            ml_rows = build_ml_rows(
                position=position,
                metadata=metadata,
                market=market,
                historical_risk=(
                    historical_risk
                ),
                strategy_metrics_df=(
                    strategy_metrics_df
                ),
                hedge_config=config,
                spread_long_put=(
                    long_put
                ),
                spread_short_put=(
                    short_put
                )
            )

            all_ml_rows.append(
                ml_rows
            )

        except Exception as error:
            print(
                "Skipping Put Spread "
                f"config {config}: {error}"
            )

    cross_hedge_configs = (
        generate_cross_hedge_configs()
    )

    base_cross_hedge_returns = (
        cross_analysis[
            "cross_hedge_returns"
        ]
    )
    
    cross_hedge_ric = cross_analysis["ric"]

    cross_hedge_beta = (
        cross_analysis["beta"]
    )

    cross_hedge_correlation = (
        cross_analysis["correlation"]
    )

    cross_hedge_r_squared = (
        cross_analysis["r_squared"]
    )

    for config in cross_hedge_configs:

        try:
            hedge_ratio = (
                config[
                    "cross_hedge_ratio"
                ]
            )

            cross_hedge_returns = (
                base_cross_hedge_returns.copy()
            )

            cross_hedge_returns[
                "Hedged Return"
            ] = (
                cross_hedge_returns[
                    "Stock Return"
                ]
                - hedge_ratio
                * cross_hedge_beta
                * cross_hedge_returns[
                    "Hedge Return"
                ]
            )

            cross_hedge_horizon_returns = (
                calculate_rolling_compounded_returns(
                    returns=(
                        cross_hedge_returns[
                            "Hedged Return"
                        ]
                    ),
                    horizon_days=(
                        position.hedge_horizon_days
                    )
                )
            )

            risk_metrics = (
                calculate_return_metrics(
                    cross_hedge_horizon_returns,
                    periods_per_year=(
                        periods_per_year
                    ),
                    calculate_drawdown=False
                )
            )

            strategy_metrics_df = (
                build_strategy_metrics_row(
                    strategy_name=(
                        "Cross Hedge"
                    ),
                    risk_metrics=(
                        risk_metrics
                    ),
                    hedge_cost_pct=0.0
                )
            )

            strategy_metrics_df.loc[
                0,
                "Correlation"
            ] = cross_hedge_correlation

            strategy_metrics_df.loc[
                0,
                "Beta"
            ] = cross_hedge_beta

            strategy_metrics_df.loc[
                0,
                "R Squared"
            ] = cross_hedge_r_squared

            ml_rows = build_ml_rows(
                position=position,
                metadata=metadata,
                market=market,
                historical_risk=historical_risk,
                strategy_metrics_df=strategy_metrics_df,
                hedge_config=config,
                cross_hedge_ric=cross_hedge_ric)

            all_ml_rows.append(
                ml_rows
            )

        except Exception as error:
            print(
                "Skipping Cross Hedge "
                f"config {config}: {error}"
            )

    if not all_ml_rows:
        return pd.DataFrame()

    valid_frames = [
        frame
        for frame in all_ml_rows
        if frame is not None and not frame.empty
    ]

    if not valid_frames:
        return pd.DataFrame()

    all_columns = list(
        dict.fromkeys(
            column
            for frame in valid_frames
            for column in frame.columns
        )
    )

    configuration_dataset = pd.concat(
        [
            frame.dropna(axis=1, how="all")
            for frame in valid_frames
        ],
        ignore_index=True
    ).reindex(columns=all_columns)

    return configuration_dataset

def analyze_position_configurations(position):
    analysis = analyze_position(
        position
    )

    metadata = analysis[
        "metadata"
    ]

    market = analysis[
        "market"
    ]

    history = analysis[
        "history"
    ]

    historical_risk = analysis[
        "historical_risk"
    ]

    start_date = analysis[
        "start_date"
    ]

    end_date = analysis[
        "end_date"
    ]

    cross_analysis = (
        analyze_cross_hedges(
            position=position,
            metadata=metadata,
            history=history,
            start_date=start_date,
            end_date=end_date,
            history_loader=(
                get_equity_history
            )
        )
    )

    options = (
        prepare_option_universe(
            position=position,
            market=market
        )
    )

    configuration_dataset = (
        evaluate_strategy_configs(
            position=position,
            metadata=metadata,
            market=market,
            history=history,
            historical_risk=(
                historical_risk
            ),
            options=options,
            cross_analysis=(
                cross_analysis
            )
        )
    )

    return configuration_dataset

def build_multi_stock_dataset(
    positions
):
    dataset_parts = []

    for position in positions:

        try:
            stock_dataset = (
                analyze_position_configurations(
                    position=position
                )
            )

            if stock_dataset.empty:
                print(
                    f"No configurations: "
                    f"{position.ric}"
                )
                continue

            dataset_parts.append(
                stock_dataset
            )

            print(
                f"Completed: "
                f"{position.ric} "
                f"({len(stock_dataset)} rows)"
            )

        except Exception as error:
            print(
                f"Skipping "
                f"{position.ric}: "
                f"{error}"
            )

    if not dataset_parts:
        return pd.DataFrame()

    return pd.concat(
        dataset_parts,
        ignore_index=True
    )