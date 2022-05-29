import pandas as pd
from typing import Optional, Union, Dict, List

from indicators.abstract_indicator import TradeAction, TradePointColumn
from trading.trade_algorithms.abstract_trade_algorithm import AbstractTradeAlgorithm
from indicators.super_trend import SuperTrend, SuperTrendHyperparam


class SuperTrendTradeAlgorithm(AbstractTradeAlgorithm):
    name = "SuperTrend trade algorithm"

    def __init__(self):
        super().__init__()
        self._stock_name = ""
        self._indicator: SuperTrend = SuperTrend()

    @staticmethod
    def get_algorithm_name() -> str:
        return SuperTrendTradeAlgorithm.name

    @staticmethod
    def create_hyperparameters_dict(lookback_period: int = 10, multiplier: Union[float, int] = 3) -> Dict:
        return {SuperTrendHyperparam.LOOKBACK_PERIOD: lookback_period, SuperTrendHyperparam.MULTIPLIER: multiplier}

    @staticmethod
    def get_default_hyperparameters_grid() -> List[Dict]:
        return [{SuperTrendHyperparam.LOOKBACK_PERIOD: 8, SuperTrendHyperparam.MULTIPLIER: 3},
                {SuperTrendHyperparam.LOOKBACK_PERIOD: 9, SuperTrendHyperparam.MULTIPLIER: 3},
                {SuperTrendHyperparam.LOOKBACK_PERIOD: 10, SuperTrendHyperparam.MULTIPLIER: 3},
                {SuperTrendHyperparam.LOOKBACK_PERIOD: 10, SuperTrendHyperparam.MULTIPLIER: 2},
                {SuperTrendHyperparam.LOOKBACK_PERIOD: 10, SuperTrendHyperparam.MULTIPLIER: 1.5},
                {SuperTrendHyperparam.LOOKBACK_PERIOD: 8, SuperTrendHyperparam.MULTIPLIER: 1.5}]

    def train(self, data: pd.DataFrame, hyperparameters: Dict):
        super().train(data, hyperparameters)
        self._stock_name = hyperparameters["DATA_NAME"]
        self._indicator.set_params(lookback_period=hyperparameters[SuperTrendHyperparam.LOOKBACK_PERIOD],
                                   multiplier=hyperparameters[SuperTrendHyperparam.MULTIPLIER])
        self._indicator.clear_vars()
        self._indicator.calculate(self.data)

    def evaluate_new_point(self, new_point: pd.Series, date: Union[str, pd.Timestamp],
                           special_params: Optional[Dict] = None) -> TradeAction:
        return self._indicator.evaluate_new_point(new_point, date, special_params)

    def plot(self, img_dir: str, start_date: Optional[pd.Timestamp] = None,
             end_date: Optional[pd.Timestamp] = None, show_full: bool = False):
        if show_full:
            self._indicator.plot(img_dir, (self._stock_name + " full history"))
        else:
            self._indicator.plot(img_dir, self._stock_name, start_date, end_date)
