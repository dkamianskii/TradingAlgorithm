import pandas as pd
from typing import Optional, Union, Dict

from Trading.AbstractTradeAlgorithm import AbstractTradeAlgorithm
from indicators.RSI import RSI


class RSITradeAlgorithm(AbstractTradeAlgorithm):
    algorithm_name = "RSI indicator algorithm"

    def __init__(self):
        super().__init__()
        self._indicator: RSI = RSI()

    def get_algorithm_name(self) -> str:
        return RSITradeAlgorithm.algorithm_name

    def train(self, data: pd.DataFrame, hyperparameters: Dict):
        super().train(data, hyperparameters)
        self._indicator.set_N(hyperparameters["N"])
        self._indicator.clear_vars()
        self._indicator.calculate(self.data)

    def evaluate_new_point(self, new_point: pd.Series,
                           date: Union[str, pd.Timestamp],
                           special_params: Optional[Dict] = None) -> str:
        self._indicator.evaluate_new_point(new_point, date, special_params)
        return self._indicator.trade_points.iloc[-1]["Action"]
