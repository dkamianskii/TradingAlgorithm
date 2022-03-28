import numpy as np
import pandas as pd
from AbstractTradeAlgorithm import AbstractTradeAlgorithm
from MACDSuperTrendTradeAlgorithm import MACDSuperTrendTradeAlgorithm
from typing import Optional, List, Dict


class TradeManager:
    trade_algorithms_list: List[str] = ["MACD_SuperTrend", "Indicators_council", "Price_prediction"]

    def __init__(self, trade_algorithm_to_use: str = "MACD_SuperTrend"):
        """
        Control:
         - stock market data,
         - currently managed assets (open long and short bids) and their limits
        Init:
         - initiate selected TradeAlgorithm
         - receive primary data
        BackTest:
         calculate efficiency of TradeAlgorithm for provided primary data
         - receive starting date for back test to separate train data from test data
         procedure of back test mainly runs as main work pipeline
        Start:
         - train TradeAlgorithm on full primary data
         - switch manager to the working pipeline mode
        Pipeline:
         - receive new day data
         - watch at currently managed assets:
            * if a bid reaches its stop loss or take profit level manager handles that event and calculate result of the deal
         - provide a new day data to the TradeAlgorithm and wait for information about what actions should be made
         - if there is any active action it operates selling or buying of currently managed mentioned assets
           then open corresponding long or short bids, evaluate stop loss and take profit for them and put them into the
           currently managed assets
         - save information about profitability of made deals and current finance assets
        """
        self.__data: Optional[pd.DataFrame] = None
        self.__last_day_data = None
        self.portfolio = [] # currently managed assets
        self.__trade_algorithm: Optional[AbstractTradeAlgorithm] = None
        self.__trade_algorithm_name = trade_algorithm_to_use
        if (trade_algorithm_to_use not in TradeManager.trade_algorithms_list):
           raise ValueError("name of trade algorithm must be one from trade algorithms names list")
        if (trade_algorithm_to_use == "MACD_SuperTrend"):
            self.__trade_algorithm = MACDSuperTrendTradeAlgorithm()
        self.default_params_fit_grid: List[dict] = []

    def __set_default_params_fit_grid(self):

        if(self.__trade_algorithm_name == "MACD_SuperTrend"):
            lookback_periods = [8,9,10]
            multiplier = [1,2,3]

    def __add_to_portfolio(self, stock_name: str, price: float, amount: int, action: str, final_sum: float):
        take_profit_lvl = self.__evaluate_take_profit(price, action)
        stop_loss_lvl = self.__evaluate_stop_loss(price, action)
        bid_type = "short bid"
        if ((action == "buy") or (action == "actively buy")):
            bid_type = "long bid"
        self.portfolio.append({"Name": stock_name,
                               "Price": price,
                               "Type": bid_type,
                               "Amount": amount,
                               "Final Sum": final_sum,
                               "Take Profit Level": take_profit_lvl,
                               "Stop Loss Level": stop_loss_lvl})

    def __evaluate_take_profit(self, price: float, action: str) -> float:
        """IN PROGRESS"""
        return price*1.04
    def __evaluate_stop_loss(self, price: float, action: str) -> float:
        """IN PROGRESS"""
        return price*0.98

    def backtest(self, use_auto_fit: Optional[bool] = True, custom_fit_frid: Optional[List[dict]] = None):
        params_fit_grid: List[dict]
        # if (self.__trade_algorithm_name == TradeManager.trade_algorithms_list[0]):
        #
        # if custom_fit_frid is None:
        #     pass
        # else:
        #     pass
        # if use_auto_fit:
        #     pass
