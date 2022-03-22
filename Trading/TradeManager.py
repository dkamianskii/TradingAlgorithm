import numpy as np
import pandas as pd
from typing import Optional


class TradeManager:
    def __init__(self):
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
        self.data: Optional[pd.DataFrame] = None
        self.last_day_data = None
        self.portfolio = [] # currently managed assets

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
