from typing import Optional, Union, Dict, List

import pandas as pd

from indicators.abstract_indicator import TradeAction
from indicators.cci import CCI
from trading.trade_algorithms.abstract_trade_algorithm import AbstractTradeAlgorithm


class CCITrradeAlgorithm(AbstractTradeAlgorithm):
    name = "CCi trade algorithm"

    def __init__(self):
        super().__init__()
        self._stock_name = ""
        self._indicator: CCI = CCI()

    @staticmethod
    def get_algorithm_name() -> str:
        return CCITrradeAlgorithm.name

    @staticmethod
    def create_hyperparameters_dict(N: int = 20) -> Dict:
        return {"N": N}

    @staticmethod
    def get_default_hyperparameters_grid() -> List[Dict]:
        return [CCITrradeAlgorithm.create_hyperparameters_dict(),
                CCITrradeAlgorithm.create_hyperparameters_dict(22),
                CCITrradeAlgorithm.create_hyperparameters_dict(18),
                CCITrradeAlgorithm.create_hyperparameters_dict(16)]

    def train(self, data: pd.DataFrame, hyperparameters: Dict):
        super().train(data, hyperparameters)
        self._stock_name = hyperparameters["DATA_NAME"]
        self._indicator.set_params(hyperparameters["N"])
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
