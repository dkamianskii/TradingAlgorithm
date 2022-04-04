import pandas as pd
from typing import Optional, Union, Dict

from Trading.AbstractTradeAlgorithm import AbstractTradeAlgorithm
from indicators.AbstractIndicator import AbstractIndicator


class OneIndicatorTradeAlgorithm(AbstractTradeAlgorithm):

    def get_algorithm_name(self) -> str:
        return self._indicator_name

    def __init__(self, indicator: AbstractIndicator, indicator_name: str):
        super().__init__()
        self._indicator: AbstractIndicator = indicator
        self._indicator_name: str = indicator_name

    def train(self, data: pd.DataFrame,
              hyperparameters_autofit: Optional[bool] = True,
              params_train_grid: Optional[Dict] = None):
        super().train(data)
        self._indicator.calculate(self.train_data)

    def evaluate_new_point(self, new_point: pd.Series,
                           date: Union[str, pd.Timestamp],
                           special_params: Optional[Dict] = None) -> str:
        self._indicator.evaluate_new_point(new_point, date, special_params)
        return self._indicator.trade_points.iloc[-1]["Action"]
