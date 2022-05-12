import pandas as pd
from typing import Optional, Union, Dict, List

from indicators.abstract_indicator import TradeAction, TradePointColumn
from trading.trade_algorithms.abstract_trade_algorithm import AbstractTradeAlgorithm
from indicators.bollinger_bands import BollingerBands, BollingerBandsHyperparam


class BollingerBandsTradeAlgorithm(AbstractTradeAlgorithm):
    name = "Bollinger bands trade algorithm"

    def __init__(self):
        super().__init__()
        self._indicator: BollingerBands = BollingerBands()

    @staticmethod
    def get_algorithm_name() -> str:
        return BollingerBandsTradeAlgorithm.name

    @staticmethod
    def create_hyperparameters_dict(N: int = 20,
                                    K: Union[float, int] = 2) -> Dict:
        return {BollingerBandsHyperparam.N: N,
                BollingerBandsHyperparam.K: K}

    @staticmethod
    def get_default_hyperparameters_grid() -> List[Dict]:
        return [BollingerBandsTradeAlgorithm.create_hyperparameters_dict(),
                BollingerBandsTradeAlgorithm.create_hyperparameters_dict(N=18),
                BollingerBandsTradeAlgorithm.create_hyperparameters_dict(N=22),
                BollingerBandsTradeAlgorithm.create_hyperparameters_dict(K=1.5),
                BollingerBandsTradeAlgorithm.create_hyperparameters_dict(K=2.5)]

    def train(self, data: pd.DataFrame, hyperparameters: Dict):
        super().train(data, hyperparameters)
        self._indicator.set_params(N=hyperparameters[BollingerBandsHyperparam.N],
                                   K=hyperparameters[BollingerBandsHyperparam.K])
        self._indicator.clear_vars()
        self._indicator.calculate(self.data)

    def evaluate_new_point(self, new_point: pd.Series, date: Union[str, pd.Timestamp],
                           special_params: Optional[Dict] = None) -> TradeAction:
        return self._indicator.evaluate_new_point(new_point, date, special_params)

    def plot(self, start_date: Optional[pd.Timestamp] = None, end_date: Optional[pd.Timestamp] = None):
        self._indicator.plot(start_date, end_date)
