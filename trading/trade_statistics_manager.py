import pandas as pd
from typing import Dict, Optional

from trading.trade_manager import BidType


class TradeStatisticsManager:

    def __init__(self):
        self.trade_result: pd.DataFrame = pd.DataFrame([{"Stock Name": "Total",
                                                         "Earned Profit": 0.0,
                                                         "Wins": 0,
                                                         "Loses": 0,
                                                         "Draws": 0}]).set_index("Stock Name")
        self.stocks_statistics: Dict[str, Dict[str, pd.DataFrame]] = {}

    def set_tracked_stock(self, stock_name: str):
        if stock_name in self.stocks_statistics.keys():
            self.trade_result.loc[stock_name] = {"Earned Profit": 0.0, "Wins": 0, "Loses": 0, "Draws": 0}
        else:
            new_stock = pd.DataFrame([{"Stock Name": stock_name,
                                       "Earned Profit": 0.0,
                                       "Wins": 0,
                                       "Loses": 0,
                                       "Draws": 0}]).set_index("Stock Name")
            self.trade_result = pd.concat([new_stock, self.trade_result])

        self.stocks_statistics[stock_name]["earnings history"] = pd.DataFrame(columns=["Date", "Value"]).set_index(
            "Date")
        self.stocks_statistics[stock_name]["bids history"] = pd.DataFrame(columns=["Date Open",
                                                                                   "Open Price",
                                                                                   "Date Close",
                                                                                   "Close Price",
                                                                                   "Type",
                                                                                   "Result color"]).set_index(
            "Date Open")

    def clear_history(self):
        self.trade_result[self.trade_result.columns] = 0
        for stock_name, stock_stat in self.stocks_statistics.items():
            stock_stat["earnings history"] = pd.DataFrame(columns=["Date", "Value"]).set_index("Date")
            stock_stat["bids history"] = pd.DataFrame(columns=["Date Open",
                                                               "Open Price",
                                                               "Date Close",
                                                               "Close Price",
                                                               "Type",
                                                               "Result color"]).set_index("Date Open")

    def update_trade_result(self, stock_name: str, profit: Optional[float], Draw=False):
        if Draw:
            self.trade_result.loc[[stock_name, "Total"], "Draw"] += 1
        else:
            if profit > 0:
                self.trade_result.loc[[stock_name, "Total"], ["Earned Profit", "Wins"]] += [profit, 1]
            else:
                self.trade_result.loc[[stock_name, "Total"], ["Earned Profit", "Loses"]] += [profit, 1]

    def add_earnings(self, stock_name: str, earnings: float, date: pd.Timestamp):
        self.stocks_statistics[stock_name]["earnings history"].loc[date] = [earnings]

    def open_bid(self, stock_name: str, date_open: pd.Timestamp, open_price: float, bid_type: BidType):
        self.stocks_statistics[stock_name]["bids history"].loc[date_open] = {"Open Price": open_price, "Type": bid_type}

    def close_bid(self, stock_name: str, date_open: pd.Timestamp, date_close: pd.Timestamp, close_price: float):
        bid = self.stocks_statistics[stock_name]["bids history"].loc[date_open]

        if (bid["Type"] == BidType.LONG and bid["Open Price"] < close_price) or (bid["Type"] == BidType.SHORT and bid["Open Price"] > close_price):
            result_color = "green"
        else:
            result_color = "red"

        self.stocks_statistics[stock_name]["bids history"].loc[date_open, ["Date Close",
                                                                           "Close Price",
                                                                           "Result Color"]] = [date_close, close_price, result_color]


