import pandas as pd
from abc import ABC, abstractmethod
from typing import Optional, Union, Hashable


class AbstractIndicator(ABC):
    """
    Abstract base class for complex indicators
    Abstract Methods: calculate, print_trade_points, plot

    Every indicator contains in itself data it is working with.
    So it has constructor that receives data, and method set_data for saving it.
    """
    actions = {"actively buy", "buy", "none", "sell", "actively sell"}

    def __init__(self, data: Optional[pd.DataFrame] = None):
        """
        :param data: dataframe of stock prices. Dataframe has to be in format of yahoo finance.
        It means that dataframe should contain columns "Close" and "Open".
        """
        self.data: Optional[pd.DataFrame] = None
        self.price: Optional[pd.Series] = None
        self.trade_points: Optional[pd.DataFrame] = None
        if (data is not None):
            self.set_data(data)

    def set_data(self, data: pd.DataFrame):
        """
        :param data: dataframe of stock prices. Dataframe has to be in format of yahoo finance.
        It means that it should contain columns "Close" and "Open".
        """
        self.data = data
        self.price = data["Close"]
        self.clear_trade_points()
        return self

    def clear_trade_points(self):
        self.trade_points = pd.DataFrame()
        self.trade_points["This Day Close"] = self.data[:-1]["Close"]
        self.trade_points["Next Day Open"] = self.data[1:]["Open"].to_numpy()
        self.trade_points["Action"] = "none"

    @abstractmethod
    def calculate(self, data: Optional[pd.DataFrame] = None):
        if (data is None):
            if (self.data is None):
                raise ValueError("calculate method could not be used without setted data")
        else:
            self.set_data(data)

    @abstractmethod
    def plot(self, start_date: Optional[pd.Timestamp] = None, end_date: Optional[pd.Timestamp] = None):
        pass

    def add_trade_point(self, date: Union[pd.Timestamp, Hashable], action: str):
        """
        :param date: date for trade action
        :param action: "actively buy", "buy", "none", "sell", "actively sell" (one of the actions)
        """
        if (action not in AbstractIndicator.actions):
            raise ValueError("unknown action was translated")

        if (date > self.trade_points.index[-1]):
            raise ValueError("irrelevant date was translated")
        # next_date_to_action_index = self.trade_points.index.tolist().index(date) + 1
        # if(next_date_to_action_index >= self.trade_points.shape[0]):
        #     return
        #next_date_to_action = self.trade_points.index[next_date_to_action_index]
        self.trade_points.at[date, "Action"] = action

    def select_action_trade_points(self, start_date: Optional[pd.Timestamp] = None,
                                   end_date: Optional[pd.Timestamp] = None) -> pd.DataFrame:
        """
        Select trade action points starting from the start date and to the end date.
        Actions are "actively buy", "buy", "sell", "actively sell"
        """
        if(start_date is None):
            start_date = self.trade_points.index[0]
        if (end_date is None):
            end_date = self.trade_points.index[-1]

        action_points = self.trade_points[start_date:end_date]
        return action_points.copy()[action_points["Action"] != "none"]
