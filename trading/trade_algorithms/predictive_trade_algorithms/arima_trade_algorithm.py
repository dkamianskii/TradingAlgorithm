from typing import Optional, Union, Dict, List

import pandas as pd

from indicators.abstract_indicator import TradeAction
from trading.trade_algorithms.abstract_trade_algorithm import AbstractTradeAlgorithm


class ARIMATradeAlgorithm(AbstractTradeAlgorithm):

    def __init__(self):
        super().__init__()

    @staticmethod
    def get_algorithm_name() -> str:
        pass

    @staticmethod
    def get_default_hyperparameters_grid() -> List[Dict]:
        pass

    def train(self, data: pd.DataFrame, hyperparameters: Dict):
        pass

    def evaluate_new_point(self, new_point: pd.Series, date: Union[str, pd.Timestamp],
                           special_params: Optional[Dict] = None) -> TradeAction:
        pass

    def plot(self, start_date: Optional[pd.Timestamp] = None, end_date: Optional[pd.Timestamp] = None):
        pass
