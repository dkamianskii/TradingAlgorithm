import pandas as pd
from typing import Optional, Union, Dict, List

from indicators.abstract_indicator import TradeAction, TradePointColumn
from trading.trade_algorithms.abstract_trade_algorithm import AbstractTradeAlgorithm
from indicators.rsi import RSI


class RSITradeAlgorithm(AbstractTradeAlgorithm):
    name = "RSI trade algorithm"

    def __init__(self):
        super().__init__()
        self._indicator: RSI = RSI()
        self._stock_name = ""

    @staticmethod
    def get_algorithm_name() -> str:
        return RSITradeAlgorithm.name

    @staticmethod
    def create_hyperparameters_dict(N: int = 14) -> Dict:
        return {"N": N}

    @staticmethod
    def get_default_hyperparameters_grid() -> List[Dict]:
        return [{"N": 8}, {"N": 10}, {"N": 12}, {"N": 14}]

    def train(self, data: pd.DataFrame, hyperparameters: Dict):
        super().train(data, hyperparameters)
        self._stock_name = hyperparameters["DATA_NAME"]
        self._indicator.set_N(hyperparameters["N"])
        self._indicator.clear_vars()
        self._indicator.calculate(self.data)

    def evaluate_new_point(self, new_point: pd.Series,
                           date: Union[str, pd.Timestamp],
                           special_params: Optional[Dict] = None) -> TradeAction:
        return self._indicator.evaluate_new_point(new_point, date, special_params)

    def plot(self, img_dir: str, start_date: Optional[pd.Timestamp] = None,
             end_date: Optional[pd.Timestamp] = None, show_full: bool = False):
        if show_full:
            self._indicator.plot(img_dir, (self._stock_name + " full history"))
        else:
            self._indicator.plot(img_dir, self._stock_name, start_date, end_date)

