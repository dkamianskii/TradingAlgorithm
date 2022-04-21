import pandas as pd
from typing import Optional, Union, Dict, List

from indicators.abstract_indicator import TradeAction
from trading.trade_algorithms.abstract_trade_algorithm import AbstractTradeAlgorithm
from indicators.macd import MACD


class MACDTradeAlgorithm(AbstractTradeAlgorithm):

    def __init__(self):
        super().__init__()
        self._indicator: MACD = MACD()

    def get_default_hyperparameters_grid(self) -> List[Dict]:
        return [{"short period": 8, "long period": 16, "signal period": 8, "trade strategy": "classic"},
                {"short period": 10, "long period": 20, "signal period": 9, "trade strategy": "classic"},
                {"short period": 12, "long period": 26, "signal period": 9, "trade strategy": "classic"},
                {"short period": 9, "long period": 18, "signal period": 8, "trade strategy": "convergence"},
                {"short period": 10, "long period": 22, "signal period": 9, "trade strategy": "convergence"},
                {"short period": 12, "long period": 26, "signal period": 8, "trade strategy": "convergence"}]

    def train(self, data: pd.DataFrame, hyperparameters: Dict):
        super().train(data, hyperparameters)
        self._indicator.set_ma_periods(hyperparameters["short period"],
                                       hyperparameters["long period"],
                                       hyperparameters["signal period"])
        self._indicator.set_trade_strategy(hyperparameters["trade strategy"])
        self._indicator.clear_vars()
        self._indicator.calculate(self.data)

    def evaluate_new_point(self, new_point: pd.Series, date: Union[str, pd.Timestamp],
                           special_params: Optional[Dict] = None) -> TradeAction:
        self._indicator.evaluate_new_point(new_point, date, special_params)
        return self._indicator.trade_points.iloc[-1]["Action"]

    def plot(self, start_date: Optional[pd.Timestamp] = None, end_date: Optional[pd.Timestamp] = None):
        self._indicator.plot(start_date, end_date)
