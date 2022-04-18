import pandas as pd
from typing import Dict, Optional

from trading.trading_enums import *


class TradeStatisticsManager:

    def __init__(self):
        self.trade_result: pd.DataFrame = pd.DataFrame([{TradeResultColumn.STOCK_NAME: TradeResultColumn.TOTAL,
                                                         TradeResultColumn.EARNED_PROFIT: 0.0,
                                                         TradeResultColumn.WINS: 0,
                                                         TradeResultColumn.LOSES: 0,
                                                         TradeResultColumn.DRAWS: 0}]).set_index(
            TradeResultColumn.STOCK_NAME)
        self.stocks_statistics: Dict[str, Dict[StocksStatisticsType, pd.DataFrame]] = {}
        self.total_earnings_history: pd.DataFrame = pd.DataFrame(columns=[EarningsHistoryColumn.DATE,
                                                                          EarningsHistoryColumn.VALUE]).set_index(
            EarningsHistoryColumn.DATE)

    def set_tracked_stock(self, stock_name: str):
        if stock_name in self.stocks_statistics.keys():
            self.trade_result.loc[stock_name] = {TradeResultColumn.EARNED_PROFIT: 0.0, TradeResultColumn.WINS: 0,
                                                 TradeResultColumn.LOSES: 0, TradeResultColumn.DRAWS: 0}
        else:
            new_stock = pd.DataFrame([{TradeResultColumn.STOCK_NAME: stock_name,
                                       TradeResultColumn.EARNED_PROFIT: 0.0,
                                       TradeResultColumn.WINS: 0,
                                       TradeResultColumn.LOSES: 0,
                                       TradeResultColumn.DRAWS: 0}]).set_index(TradeResultColumn.STOCK_NAME)
            self.trade_result = pd.concat([new_stock, self.trade_result])

        self.stocks_statistics[stock_name] = {}
        self.stocks_statistics[stock_name][StocksStatisticsType.EARNINGS_HISTORY] = pd.DataFrame(
            columns=[EarningsHistoryColumn.DATE,
                     EarningsHistoryColumn.VALUE]).set_index(EarningsHistoryColumn.DATE)
        self.stocks_statistics[stock_name][StocksStatisticsType.BIDS_HISTORY] = pd.DataFrame(
            columns=[BidsHistoryColumn.DATE_OPEN,
                     BidsHistoryColumn.OPEN_PRICE,
                     BidsHistoryColumn.DATE_CLOSE,
                     BidsHistoryColumn.CLOSE_PRICE,
                     BidsHistoryColumn.TYPE,
                     BidsHistoryColumn.RESULT_COLOR]).set_index(BidsHistoryColumn.DATE_OPEN)

    def clear_history(self):
        self.trade_result[self.trade_result.columns] = 0
        for stock_name, stock_stat in self.stocks_statistics.items():
            stock_stat[StocksStatisticsType.EARNINGS_HISTORY] = pd.DataFrame(
                columns=[EarningsHistoryColumn.DATE,
                         EarningsHistoryColumn.VALUE]).set_index(EarningsHistoryColumn.DATE)
            stock_stat[StocksStatisticsType.BIDS_HISTORY] = pd.DataFrame(
                columns=[BidsHistoryColumn.DATE_OPEN,
                         BidsHistoryColumn.OPEN_PRICE,
                         BidsHistoryColumn.DATE_CLOSE,
                         BidsHistoryColumn.CLOSE_PRICE,
                         BidsHistoryColumn.TYPE,
                         BidsHistoryColumn.RESULT_COLOR]).set_index(BidsHistoryColumn.DATE_OPEN)

    def update_trade_result(self, stock_name: str, profit: Optional[float], Draw=False):
        if Draw:
            self.trade_result.loc[[stock_name, TradeResultColumn.TOTAL], TradeResultColumn.DRAWS] += 1
        else:
            if profit > 0:
                self.trade_result.loc[
                    [stock_name, TradeResultColumn.TOTAL], [TradeResultColumn.EARNED_PROFIT, TradeResultColumn.WINS]] += [profit, 1]
            else:
                self.trade_result.loc[
                    [stock_name, TradeResultColumn.TOTAL], [TradeResultColumn.EARNED_PROFIT, TradeResultColumn.LOSES]] += [profit, 1]

    def add_earnings(self, stock_name: str, earnings: float, date: pd.Timestamp):
        self.stocks_statistics[stock_name][StocksStatisticsType.EARNINGS_HISTORY].loc[date] = [earnings]
        self.total_earnings_history.loc[date] = [earnings]

    def open_bid(self, stock_name: str, date_open: pd.Timestamp, open_price: float, bid_type: BidType):
        self.stocks_statistics[stock_name][StocksStatisticsType.BIDS_HISTORY].loc[date_open] = {
            BidsHistoryColumn.OPEN_PRICE: open_price,
            BidsHistoryColumn.TYPE: bid_type}

    def close_bid(self, stock_name: str, date_open: pd.Timestamp, date_close: pd.Timestamp, close_price: float):
        bid = self.stocks_statistics[stock_name][StocksStatisticsType.BIDS_HISTORY].loc[date_open]

        if (bid[BidsHistoryColumn.TYPE] == BidType.LONG and bid[BidsHistoryColumn.OPEN_PRICE] < close_price) or (
                bid[BidsHistoryColumn.TYPE] == BidType.SHORT and bid[BidsHistoryColumn.OPEN_PRICE] > close_price):
            result_color = "green"
        else:
            result_color = "red"

        self.stocks_statistics[stock_name][StocksStatisticsType.BIDS_HISTORY].loc[
            date_open, [BidsHistoryColumn.DATE_CLOSE,
                        BidsHistoryColumn.CLOSE_PRICE,
                        BidsHistoryColumn.RESULT_COLOR]] = [date_close, close_price,
                                                            result_color]
