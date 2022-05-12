import pandas as pd
from typing import Optional, Union, Dict, List

from indicators.abstract_indicator import TradeAction, TradePointColumn
from trading.trade_algorithms.abstract_trade_algorithm import AbstractTradeAlgorithm
from indicators.macd import MACD, MACDTradeStrategy, MACDHyperparam


class MACDTradeAlgorithm(AbstractTradeAlgorithm):
    name = "MACD trade algorithm"

    def __init__(self):
        super().__init__()
        self._indicator: MACD = MACD()

    @staticmethod
    def get_algorithm_name() -> str:
        return MACDTradeAlgorithm.name

    @staticmethod
    def create_hyperparameters_dict(short_period: int = 12,
                                    long_period: int = 26,
                                    signal_period: int = 9,
                                    trade_strategy: MACDTradeStrategy = MACDTradeStrategy.CLASSIC) -> Dict:
        return {MACDHyperparam.SHORT_PERIOD: short_period,
                MACDHyperparam.LONG_PERIOD: long_period,
                MACDHyperparam.SIGNAL_PERIOD: signal_period,
                MACDHyperparam.TRADE_STRATEGY: trade_strategy}

    @staticmethod
    def get_default_hyperparameters_grid() -> List[Dict]:
        return [{MACDHyperparam.SHORT_PERIOD: 8, MACDHyperparam.LONG_PERIOD: 16,
                 MACDHyperparam.SIGNAL_PERIOD: 8, MACDHyperparam.TRADE_STRATEGY: MACDTradeStrategy.CLASSIC},
                {MACDHyperparam.SHORT_PERIOD: 10, MACDHyperparam.LONG_PERIOD: 20,
                 MACDHyperparam.SIGNAL_PERIOD: 9, MACDHyperparam.TRADE_STRATEGY: MACDTradeStrategy.CLASSIC},
                {MACDHyperparam.SHORT_PERIOD: 12, MACDHyperparam.LONG_PERIOD: 26,
                 MACDHyperparam.SIGNAL_PERIOD: 9, MACDHyperparam.TRADE_STRATEGY: MACDTradeStrategy.CLASSIC},
                {MACDHyperparam.SHORT_PERIOD: 9, MACDHyperparam.LONG_PERIOD: 18,
                 MACDHyperparam.SIGNAL_PERIOD: 8, MACDHyperparam.TRADE_STRATEGY: MACDTradeStrategy.CONVERGENCE},
                {MACDHyperparam.SHORT_PERIOD: 10, MACDHyperparam.LONG_PERIOD: 22,
                 MACDHyperparam.SIGNAL_PERIOD: 9, MACDHyperparam.TRADE_STRATEGY: MACDTradeStrategy.CONVERGENCE},
                {MACDHyperparam.SHORT_PERIOD: 12, MACDHyperparam.LONG_PERIOD: 26,
                 MACDHyperparam.SIGNAL_PERIOD: 9, MACDHyperparam.TRADE_STRATEGY: MACDTradeStrategy.CONVERGENCE}]

    def train(self, data: pd.DataFrame, hyperparameters: Dict):
        super().train(data, hyperparameters)
        self._indicator.set_ma_periods(hyperparameters[MACDHyperparam.SHORT_PERIOD],
                                       hyperparameters[MACDHyperparam.LONG_PERIOD],
                                       hyperparameters[MACDHyperparam.SIGNAL_PERIOD])
        self._indicator.set_trade_strategy(hyperparameters[MACDHyperparam.TRADE_STRATEGY])
        self._indicator.clear_vars()
        self._indicator.calculate(self.data)

    def evaluate_new_point(self, new_point: pd.Series, date: Union[str, pd.Timestamp],
                           special_params: Optional[Dict] = None) -> TradeAction:
        return self._indicator.evaluate_new_point(new_point, date, special_params)

    def plot(self, start_date: Optional[pd.Timestamp] = None, end_date: Optional[pd.Timestamp] = None):
        self._indicator.plot(start_date, end_date)
