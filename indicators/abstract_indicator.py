import pandas as pd
from abc import ABC, abstractmethod
from typing import Optional, Union, Hashable

from helping.base_enum import BaseEnum


class TradeAction(BaseEnum):
    NONE = 0
    BUY = 1
    ACTIVELY_BUY = 2
    SELL = 3
    ACTIVELY_SELL = 4


class TradePointColumn(BaseEnum):
    DATE = 1,
    PRICE = 2,
    ACTION = 3


class AbstractIndicator(ABC):
    """
    Abstract base class for complex indicators
    Abstract Methods: calculate, print_trade_points, plot

    Every indicator contains in itself data it is working with.
    So it has constructor that receives data, and method set_data for saving it.
    """

    def __init__(self, data: Optional[pd.DataFrame] = None):
        """
        :param data: dataframe of stock prices. Dataframe has to be in format of yahoo finance.
        It means that dataframe should contain columns "Close" and "Open".
        """
        self.data: Optional[pd.DataFrame] = None
        self.trade_points: Optional[pd.DataFrame] = None
        if data is not None:
            self.set_data(data)

    def set_data(self, data: pd.DataFrame):
        """
        :param data: dataframe of stock prices. Dataframe has to be in format of yahoo finance.
        It means that it should contain columns "Close" and "Open".
        """
        self.data = data
        self.clear_vars()
        return self

    @abstractmethod
    def clear_vars(self):
        self.trade_points = pd.DataFrame(columns=TradePointColumn.get_elements_list()).set_index(TradePointColumn.DATE)

    @abstractmethod
    def calculate(self, data: Optional[pd.DataFrame] = None):
        if data is None:
            if self.data is None:
                raise ValueError("calculate method could not be used without setted data")
        else:
            self.set_data(data)

    @abstractmethod
    def find_trade_points(self) -> pd.DataFrame:
        pass

    @abstractmethod
    def evaluate_new_point(self, new_point: pd.Series, date: Union[str, pd.Timestamp], special_params: Optional = None,
                           update_data: bool = True) -> TradeAction:
        pass

    @abstractmethod
    def plot(self, start_date: Optional[pd.Timestamp] = None, end_date: Optional[pd.Timestamp] = None):
        pass

    def add_trade_point(self, date: Union[pd.Timestamp, Hashable], price: float, action: TradeAction):
        """
        :param date: date for trade action
        :param price: price of purchase
        :param action: action to make at the moment
        """
        self.trade_points.loc[date] = {TradePointColumn.PRICE: price, TradePointColumn.ACTION: action}

    def select_action_trade_points(self, start_date: Optional[pd.Timestamp] = None,
                                   end_date: Optional[pd.Timestamp] = None) -> pd.DataFrame:
        """
        Select trade action points starting from the start date and to the end date.
        Actions are "actively buy", "buy", "sell", "actively sell"
        """
        if self.trade_points.shape[0] == 0:
            return self.trade_points.copy()
        if start_date is None:
            start_date = self.trade_points.index[0]
        if end_date is None:
            end_date = self.trade_points.index[-1]

        action_points = self.trade_points[start_date:end_date]
        return action_points[action_points[TradePointColumn.ACTION] != TradeAction.NONE].copy()
