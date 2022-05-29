from abc import ABC, abstractmethod
import pandas as pd
from typing import Optional, Union, Dict, List

from indicators.abstract_indicator import TradeAction


class AbstractTradeAlgorithm(ABC):

    def __init__(self):
        self.data: Optional[pd.DataFrame] = None

    @staticmethod
    @abstractmethod
    def get_algorithm_name() -> str:
        pass

    @staticmethod
    @abstractmethod
    def get_default_hyperparameters_grid() -> List[Dict]:
        pass

    @abstractmethod
    def train(self, data: pd.DataFrame, hyperparameters: Dict):
        self.data = data.copy()
        pass

    @abstractmethod
    def evaluate_new_point(self, new_point: pd.Series,
                           date: Union[str, pd.Timestamp],
                           special_params: Optional[Dict] = None) -> TradeAction:
        pass

    @abstractmethod
    def plot(self, img_dir: str, start_date: Optional[pd.Timestamp] = None, end_date: Optional[pd.Timestamp] = None,
             show_full: bool = False):
        pass
