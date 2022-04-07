import pandas as pd
from typing import Optional, Union, Dict

from Trading.AbstractTradeAlgorithm import AbstractTradeAlgorithm
from Indicators.SuperTrend import SuperTrend
from Indicators.MACD import MACD


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
