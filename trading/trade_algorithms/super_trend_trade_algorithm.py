import pandas as pd
from typing import Optional, Union, Dict, List

from indicators.abstract_indicator import TradeAction, TradePointColumn
from trading.trade_algorithms.abstract_trade_algorithm import AbstractTradeAlgorithm
from indicators.super_trend import SuperTrend


class SuperTrendTradeAlgorithm(AbstractTradeAlgorithm):
    def __init__(self):
        super().__init__()
        self._indicator: SuperTrend = SuperTrend()

    @staticmethod
    def create_hyperparameters_dict(lookback_period: int = 10, multiplier: Union[float, int] = 3):
        return {"lookback period": lookback_period, "multiplier": multiplier}

    @staticmethod
    def get_default_hyperparameters_grid() -> List[Dict]:
        return [{"lookback period": 8, "multiplier": 3},
                {"lookback period": 9, "multiplier": 3},
                {"lookback period": 10, "multiplier": 3},
                {"lookback period": 10, "multiplier": 2},
                {"lookback period": 10, "multiplier": 1.5},
                {"lookback period": 8, "multiplier": 1.5}]

    def train(self, data: pd.DataFrame, hyperparameters: Dict):
        super().train(data, hyperparameters)
        self._indicator.set_params(lookback_period=hyperparameters["lookback period"],
                                   multiplier=hyperparameters["multiplier"])
        self._indicator.clear_vars()
        self._indicator.calculate(self.data)

    def evaluate_new_point(self, new_point: pd.Series, date: Union[str, pd.Timestamp],
                           special_params: Optional[Dict] = None) -> TradeAction:
        self._indicator.evaluate_new_point(new_point, date, special_params)
        return self._indicator.trade_points.iloc[-1][TradePointColumn.ACTION]

    def plot(self, start_date: Optional[pd.Timestamp] = None, end_date: Optional[pd.Timestamp] = None):
        self._indicator.plot(start_date, end_date)
