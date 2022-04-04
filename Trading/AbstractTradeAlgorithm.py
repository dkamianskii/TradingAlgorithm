from abc import ABC, abstractmethod
import pandas as pd
from typing import Optional, Union, Dict


class AbstractTradeAlgorithm(ABC):

    def __init__(self):
        self.train_data: Optional[pd.DataFrame] = None
        self.whole_data: Optional[pd.DataFrame] = None
        self.trade_start_date: Optional[pd.Timestamp] = None

    @abstractmethod
    def get_algorithm_name(self) -> str:
        pass

    @abstractmethod
    def train(self, data: pd.DataFrame,
              hyperparameters_autofit: Optional[bool] = True,
              params_train_grid: Optional[Dict] = None):
        self.train_data = data
        self.trade_start_date = data.index[-1]
        self.whole_data = data.copy()
        pass

    @abstractmethod
    def evaluate_new_point(self, new_point: pd.Series,
                           date: Union[str, pd.Timestamp],
                           special_params: Optional[Dict] = None) -> str:
        pass
