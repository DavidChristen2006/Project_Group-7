import pandas as pd

def compare_strategies(
    put_df,
    collar_df,
    put_spread_df):
    comparison = pd.DataFrame({
        "Scenario": put_df["Scenario"],
        "Scenario Price": put_df["Scenario Price"],
        "Unhedged P&L": put_df["Unhedged P&L"],
        "Protective Put P&L": put_df["Hedged P&L"],
        "Collar P&L": collar_df["Collar P&L"],
        "Put Spread P&L": put_spread_df["Put Spread P&L"]})

    comparison["Protective Put Improvement"] = comparison["Protective Put P&L"] - comparison["Unhedged P&L"]

    comparison["Collar Improvement"] = comparison["Collar P&L"] - comparison["Unhedged P&L"]

    comparison["Put Spread Improvement"] = comparison["Put Spread P&L"] - comparison["Unhedged P&L"]

    return comparison