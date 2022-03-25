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
        self.trade_points: Optional[pd.DataFrame] = None
        if (data is not None):
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
        self.trade_points = pd.DataFrame(columns=["date", "Price", "Action"])
        self.trade_points["date"] = pd.to_datetime(self.trade_points["date"])
        self.trade_points["Price"] = self.trade_points["Price"].astype("float")
        self.trade_points["Action"] = self.trade_points["Action"].astype("str")
        self.trade_points = self.trade_points.set_index("date")

    @abstractmethod
    def calculate(self, data: Optional[pd.DataFrame] = None):
        if (data is None):
            if (self.data is None):
                raise ValueError("calculate method could not be used without setted data")
        else:
            self.set_data(data)

    @abstractmethod
    def find_trade_points(self) -> pd.DataFrame:
        pass

    @abstractmethod
    def evaluate_new_point(self, new_point: pd.Series, date: Union[str, pd.Timestamp], special_params: Optional = None):
        pass

    @abstractmethod
    def plot(self, start_date: Optional[pd.Timestamp] = None, end_date: Optional[pd.Timestamp] = None):
        pass

    def add_trade_point(self, date: Union[pd.Timestamp, Hashable], price: float, action: str):
        """
        :param date: date for trade action
        :param price: price of purchase
        :param action: "actively buy", "buy", "none", "sell", "actively sell" (one of the actions)
        """
        if (action not in AbstractIndicator.actions):
            raise ValueError("unknown action was translated")

        self.trade_points.loc[date] = {"Price": price, "Action": action}

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
