
import numpy as np
import pandas as pd

#Turns raw prices into a training DataFrame for the Ml Model  
from pipeline.historical_data_pipeline import ( 
    load_price_history
)
from pipeline.historical_option_pipeline import (
    load_option_quote_history
)

#List & Definition of the features 
def build_historical_option_features(
    underlying_ric,
    option_contracts,
    volatility_window=20,
    momentum_window=20
):
    stock = load_price_history(underlying_ric).copy()  #Copies data from the Api  

    stock = stock.sort_index()          #Creates a DataFrame and sorts them from oldest to newest 
    stock["Close"] = pd.to_numeric(
        stock["Close"],
        errors="coerce"
    ).astype("float64")

    # Features use only information through the decision date.
    stock["Daily Return"] = stock["Close"].pct_change(
        fill_method=None
    )

    stock["Historical Volatility"] = (     #Measures the volatility of each stock on a rolling basis (20d), then computes the standard deviation on that 
                                              #, and converts that into a yearly value, 252 trading days in a year. 
                                                                                    
        stock["Daily Return"]
        .rolling(volatility_window)
        .std()
        * np.sqrt(252)
    )

    stock["Momentum"] = (              # % Amount of change of a stock between 20 days ago and today's prices 
        stock["Close"].pct_change(
            periods=momentum_window,
            fill_method=None            #If there isn't change don't enter a value 
        )
    )

    feature_frames = []

    for contract in option_contracts:        #load the options price history 
        option_ric = contract["RIC"]

        quotes = load_option_quote_history(
            option_ric
        ).copy()

        quotes = quotes.sort_index()

        features = quotes.join(               #joins stock and option data and puts the columns next to eachother 
            stock[
                [
                    "Close",
                    "Daily Return",
                    "Historical Volatility",
                    "Momentum"
                ]
            ],
            how="inner"                  #innerjoins them to keep dates only where there is data from both stocks and options 
        )

        features["Underlying RIC"] = underlying_ric        #Creates all of the different rows and assigns the value to them 
        features["Option RIC"] = option_ric

        features["Option Type"] = contract["Type"]
        features["Strike"] = float(contract["Strike"])

        expiry = pd.Timestamp(         #Gives a clear ending date, and normalizes the Df to midnight so that we only account for the day 
            contract["Expiry"]
        ).normalize() 

        features["Expiry"] = expiry

        features["Days to Expiry"] = (
            expiry - features.index
        ).days

        features = features[
            features["Days to Expiry"] > 0
        ].copy()

        features["Moneyness"] = (         #Calculates how far is the option from it's strike 
            features["Close"] / features["Strike"]
        )

        features["Strike to Spot"] = (    #Calculates how far is the strike to the stock price 
            features["Strike"] / features["Close"]
        )

        features["Relative Spread"] = (   #Calculates how expensive it is to trade the option = Spread
            features["SPREAD"] / features["MID"]
        )

        features["Decision Date"] = features.index  #Saves Decision date into a column 

        feature_frames.append(
            features.reset_index(drop=True)
        )

    if not feature_frames:            #Check stops program if there is no value
        raise ValueError("No option contracts supplied.")

    dataset = pd.concat(        #Stacks all of the per option tables into one single table 
        feature_frames,
        ignore_index=True
    )

    dataset = dataset.sort_values(    #Rearrange values in the dataset by Decision dat and Option Ric 
        ["Decision Date", "Option RIC"]
    ).reset_index(drop=True)

    required_features = [       #Values get dropped from the dataset if they don't present the followning features  
        "BID",
        "ASK",
        "Close",
        "Historical Volatility",
        "Momentum",
        "Moneyness",
        "Days to Expiry"
    ]

    dataset = dataset.dropna(
        subset=required_features
    )

    if dataset.empty:         #Program stops running if no value is stored in the dataset 
        raise ValueError(
            "No complete historical feature observations."
        )

    return dataset
