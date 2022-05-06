import pandas as pd
from typing import Dict, Optional

from indicators.abstract_indicator import TradeAction
from trading.trade_manager_enums import BidType, BidResult
from trading.trade_statistics_manager_enums import *


class TradeStatisticsManager:  # todo Sharpe Ratio, Sortino Ratio

    def __init__(self):
        self.trade_result: pd.DataFrame = pd.DataFrame([{TradeResultColumn.STOCK_NAME: TradeResultColumn.TOTAL,
                                                         TradeResultColumn.EARNED_PROFIT: 0.0,
                                                         TradeResultColumn.WINS: 0,
                                                         TradeResultColumn.LOSES: 0,
                                                         TradeResultColumn.DRAWS: 0}]).set_index(
            TradeResultColumn.STOCK_NAME)
        self.bids_history: pd.DataFrame = pd.DataFrame(columns=BidsHistoryColumn.get_elements_list())
        self.earnings_history: Dict[str, pd.DataFrame] = {str(EarningsHistoryColumn.TOTAL): pd.DataFrame(
            columns=[EarningsHistoryColumn.DATE,
                     EarningsHistoryColumn.VALUE]).set_index(EarningsHistoryColumn.DATE)}

    def set_tracked_stock(self, stock_name: str):
        if stock_name in self.trade_result.index:
            self.trade_result.loc[stock_name] = {TradeResultColumn.EARNED_PROFIT: 0.0, TradeResultColumn.WINS: 0,
                                                 TradeResultColumn.LOSES: 0, TradeResultColumn.DRAWS: 0}
        else:
            new_stock = pd.DataFrame([{TradeResultColumn.STOCK_NAME: stock_name,
                                       TradeResultColumn.EARNED_PROFIT: 0.0,
                                       TradeResultColumn.WINS: 0,
                                       TradeResultColumn.LOSES: 0,
                                       TradeResultColumn.DRAWS: 0}]).set_index(TradeResultColumn.STOCK_NAME)
            self.trade_result = pd.concat([new_stock, self.trade_result])

        self.earnings_history[stock_name] = pd.DataFrame(columns=[EarningsHistoryColumn.DATE,
                                                                  EarningsHistoryColumn.VALUE]).set_index(
            EarningsHistoryColumn.DATE)

    def get_bids_history(self, stock_name: Optional[str] = None) -> pd.DataFrame:
        if stock_name is None:
            return self.bids_history
        bids_history = self.bids_history[self.bids_history[BidsHistoryColumn.NAME] == stock_name]
        bids_history = bids_history[~bids_history[BidsHistoryColumn.DATE_CLOSE].isna()]
        return bids_history

    def get_earnings_history(self, stock_name: Optional[str] = None) -> pd.DataFrame:
        if stock_name is None:
            return self.earnings_history[str(EarningsHistoryColumn.TOTAL)]
        return self.earnings_history[stock_name]

    def clear_history(self):
        self.trade_result[self.trade_result.columns] = 0
        self.bids_history = pd.DataFrame(columns=BidsHistoryColumn.get_elements_list())
        for stock_name in self.earnings_history.keys():
            self.earnings_history[stock_name] = pd.DataFrame(columns=[EarningsHistoryColumn.DATE,
                                                                      EarningsHistoryColumn.VALUE]).set_index(
                EarningsHistoryColumn.DATE)

    def update_trade_result(self, stock_name: str, profit: Optional[float], result: BidResult):
        if result == BidResult.DRAW:
            self.trade_result.loc[[stock_name, TradeResultColumn.TOTAL],
                                  [TradeResultColumn.EARNED_PROFIT, TradeResultColumn.DRAWS]] += [profit, 1]
        elif result == BidResult.WIN:
            self.trade_result.loc[[stock_name, TradeResultColumn.TOTAL],
                                  [TradeResultColumn.EARNED_PROFIT, TradeResultColumn.WINS]] += [profit, 1]
        else:
            self.trade_result.loc[[stock_name, TradeResultColumn.TOTAL],
                                  [TradeResultColumn.EARNED_PROFIT, TradeResultColumn.LOSES]] += [profit, 1]

    def add_earnings(self, stock_name: str, earnings: float, date: pd.Timestamp):
        self.earnings_history[stock_name].loc[date] = [earnings]
        if date in self.earnings_history[str(EarningsHistoryColumn.TOTAL)].index:
            self.earnings_history[str(EarningsHistoryColumn.TOTAL)].loc[date] += [earnings]
        else:
            self.earnings_history[str(EarningsHistoryColumn.TOTAL)].loc[date] = [earnings]

    def open_bid(self, stock_name: str,
                 date_open: pd.Timestamp,
                 open_price: float,
                 bid_type: BidType,
                 take_profit: float,
                 stop_loss: float,
                 amount: int,
                 trade_action: TradeAction,
                 prolongation: bool):
        self.bids_history.loc[self.bids_history.shape[0]] = {BidsHistoryColumn.NAME: stock_name,
                                                             BidsHistoryColumn.DATE_OPEN: date_open,
                                                             BidsHistoryColumn.OPEN_PRICE: open_price,
                                                             BidsHistoryColumn.AMOUNT: amount,
                                                             BidsHistoryColumn.TYPE: bid_type,
                                                             BidsHistoryColumn.TRADE_ACTION: trade_action,
                                                             BidsHistoryColumn.TAKE_PROFIT: take_profit,
                                                             BidsHistoryColumn.STOP_LOSS: stop_loss,
                                                             BidsHistoryColumn.PROLONGATION: prolongation}

    def close_bid(self, stock_name: str,
                  date_open: pd.Timestamp,
                  date_close: pd.Timestamp,
                  close_price: float,
                  result: BidResult):
        self.bids_history.loc[(self.bids_history[BidsHistoryColumn.DATE_OPEN] == date_open) & (
                self.bids_history[BidsHistoryColumn.NAME] == stock_name),
                              [BidsHistoryColumn.DATE_CLOSE,
                               BidsHistoryColumn.CLOSE_PRICE,
                               BidsHistoryColumn.RESULT]] = [date_close, close_price, result]
