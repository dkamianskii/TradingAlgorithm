from abc import ABC, abstractmethod
import pandas as pd
from typing import Optional

class AbstractTradeAlgorithm(ABC):

    def __init__(self):
        self.train_data: Optional[pd.DataFrame] = None
        self.whole_data: Optional[pd.DataFrame] = None
        self.trade_start_date: Optional[pd.Timestamp] = None

    def __clear_trade_points(self, data: pd.DataFrame):
        self.trade_points = pd.DataFrame()
        self.trade_points["This Day Close"] = data[:-1]["Close"]
        self.trade_points["Next Day Open"] = data[1:]["Open"].to_numpy()
        self.trade_points["Action"] = "none"

    @abstractmethod
    def train(self, data: pd.DataFrame):
        self.train_data = data
        self.trade_start_date = data.index[-1]
        self.whole_data = data.copy()
        pass

    @abstractmethod
    def day_analysis(self, new_day_data: pd.Series) -> pd.DataFrame:
        self.whole_data.append(new_day_data)
        pass
