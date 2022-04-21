import pandas as pd
from typing import Optional, Union, Dict

from trading.trade_algorithms.abstract_trade_algorithm import AbstractTradeAlgorithm
from indicators.super_trend import SuperTrend
from indicators.macd import MACD


class MACDSuperTrendTradeAlgorithm(AbstractTradeAlgorithm):

    def __init__(self):
        super().__init__()
        self.__super_trend: SuperTrend = SuperTrend()
        self.__MACD: MACD = MACD()

    def train(self, data: pd.DataFrame,
              hyperparameters_autofit: Optional[bool] = True,
              params_train_grid: Optional[Dict] = None):
        super().train(data)
        if hyperparameters_autofit:
            pass
        pass

    def evaluate_new_point(self, new_point: pd.Series,
                           date: Union[str, pd.Timestamp],
                           special_params: Optional[Dict] = None):
        pass
