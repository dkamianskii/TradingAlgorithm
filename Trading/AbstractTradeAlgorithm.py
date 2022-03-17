from abc import ABC, abstractmethod
import pandas as pd
from typing import Optional

class AbstractTradeAlgorithm(ABC):

    def __init__(self):
        self.train_data: Optional[pd.DataFrame] = None
        self.trade_data: Optional[pd.DataFrame] = None
        self.trade_points: Optional[pd.DataFrame] = None

    def __clear_trade_points(self, data: pd.DataFrame):
        self.trade_points = pd.DataFrame()
        self.trade_points["This Day Close"] = data[:-1]["Close"]
        self.trade_points["Next Day Open"] = data[1:]["Open"].to_numpy()
        self.trade_points["Action"] = "none"

    @abstractmethod
    def train(self, data: pd.DataFrame):
        self.train_data = data
        pass

    @abstractmethod
    def trade(self, data: pd.DataFrame) -> pd.DataFrame:
        self.__clear_trade_points(data)
        self.trade_data = data
        pass
