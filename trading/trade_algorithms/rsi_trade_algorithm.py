import pandas as pd
from typing import Optional, Union, Dict, List

from indicators.abstract_indicator import TradeAction
from trading.trade_algorithms.abstract_trade_algorithm import AbstractTradeAlgorithm
from indicators.rsi import RSI


class RSITradeAlgorithm(AbstractTradeAlgorithm):

    def __init__(self):
        super().__init__()
        self._indicator: RSI = RSI()

    def get_default_hyperparameters_grid(self) -> List[Dict]:
        return [{"N": 8}, {"N": 10}, {"N": 12}, {"N": 14}]

    def train(self, data: pd.DataFrame, hyperparameters: Dict):
        super().train(data, hyperparameters)
        self._indicator.set_N(hyperparameters["N"])
        self._indicator.clear_vars()
        self._indicator.calculate(self.data)

    def evaluate_new_point(self, new_point: pd.Series,
                           date: Union[str, pd.Timestamp],
                           special_params: Optional[Dict] = None) -> TradeAction:
        self._indicator.evaluate_new_point(new_point, date, special_params)
        return self._indicator.trade_points.iloc[-1]["Action"]

    def plot(self, start_date: Optional[pd.Timestamp] = None, end_date: Optional[pd.Timestamp] = None):
        self._indicator.plot(start_date, end_date)
