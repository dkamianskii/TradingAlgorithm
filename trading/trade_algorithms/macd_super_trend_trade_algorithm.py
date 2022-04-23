import pandas as pd
from typing import Optional, Union, Dict, List

from trading.trade_algorithms.abstract_trade_algorithm import AbstractTradeAlgorithm
from indicators.super_trend import SuperTrend
from indicators.macd import MACD


class MACDSuperTrendTradeAlgorithm(AbstractTradeAlgorithm):

    @staticmethod
    def get_default_hyperparameters_grid() -> List[Dict]:
        pass

    def __init__(self):
        super().__init__()
        self.__super_trend: SuperTrend = SuperTrend()
        self.__MACD: MACD = MACD()

    def train(self, data: pd.DataFrame, hyperparameters: Dict):
        super().train(data, hyperparameters)
        pass

    def evaluate_new_point(self, new_point: pd.Series,
                           date: Union[str, pd.Timestamp],
                           special_params: Optional[Dict] = None):
        pass

    def plot(self, start_date: Optional[pd.Timestamp] = None, end_date: Optional[pd.Timestamp] = None):
        pass