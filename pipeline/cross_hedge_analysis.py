from lseg_api.hedge_universe import get_cross_hedge_universe, extract_etf_rics
from analytics.cross_hedge import load_cross_hedge_histories, compare_cross_hedge_candidates
from hedging.cross_hedge import calculate_cross_hedge_returns, calculate_cross_hedge_metrics


def analyze_cross_hedges(
    position,
    metadata,
    history,
    start_date,
    end_date,
    history_loader,
    hedge_ratio=1.0):
    hedge_universe = get_cross_hedge_universe(metadata)

    etf_rics = extract_etf_rics(hedge_universe, max_candidates=10)

    cross_hedge_rics = hedge_universe["indices"] + hedge_universe["broad_market"] + etf_rics

    cross_hedge_rics = list(dict.fromkeys(cross_hedge_rics))

    hedge_histories = (
        load_cross_hedge_histories(
            hedge_rics=cross_hedge_rics,
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d"),
            history_loader=history_loader))

    if not hedge_histories:
        raise ValueError("No valid cross-hedge histories were found.")

    cross_hedge_comparison = (
        compare_cross_hedge_candidates(
            stock_history=history,
            hedge_histories=(hedge_histories)))

    if cross_hedge_comparison.empty:
        raise ValueError("No valid cross-hedge candidates were found.")

    selected_cross_hedge = cross_hedge_comparison.iloc[0]
    cross_hedge_ric = selected_cross_hedge["Hedge"]
    cross_hedge_beta = selected_cross_hedge["Beta"]
    cross_hedge_correlation = selected_cross_hedge["Correlation"]
    cross_hedge_r_squared = selected_cross_hedge["R Squared"]
    selected_hedge_history = hedge_histories[cross_hedge_ric]

    cross_hedge_returns = (
        calculate_cross_hedge_returns(
            stock_history=history,
            hedge_history=(selected_hedge_history),
            beta=cross_hedge_beta,
            hedge_ratio=hedge_ratio))
    cross_hedge_metrics = calculate_cross_hedge_metrics(cross_hedge_returns)

    return {
        "hedge_universe": hedge_universe,
        "cross_hedge_rics": cross_hedge_rics,
        "hedge_histories": hedge_histories,
        "comparison": cross_hedge_comparison,
        "selected_cross_hedge": selected_cross_hedge,
        "selected_hedge_history": selected_hedge_history,
        "cross_hedge_returns": cross_hedge_returns,
        "cross_hedge_metrics": cross_hedge_metrics,
        "ric": cross_hedge_ric,
        "beta": cross_hedge_beta,
        "correlation": cross_hedge_correlation,
        "r_squared": cross_hedge_r_squared,
        "hedge_ratio": hedge_ratio}