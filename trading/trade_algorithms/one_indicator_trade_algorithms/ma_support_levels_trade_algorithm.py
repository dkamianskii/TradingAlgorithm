import pandas as pd
from typing import Optional, Union, Dict, List

from indicators.abstract_indicator import TradeAction, TradePointColumn
from trading.trade_algorithms.abstract_trade_algorithm import AbstractTradeAlgorithm
from indicators.ma_support_levels import MASupportLevels


class MASupportLevelsTradeAlgorithm(AbstractTradeAlgorithm):
    name = "MA Support Levels trade algorithm"

    def __init__(self):
        super().__init__()
        self._stock_name = ""
        self._indicator: MASupportLevels = MASupportLevels()

    @staticmethod
    def get_algorithm_name() -> str:
        return MASupportLevelsTradeAlgorithm.name

    @staticmethod
    def create_hyperparameters_dict(ma_periods: List[int] = MASupportLevels.default_ma_periods_for_test
                                    ) -> Dict:
        return {"ma periods": ma_periods}

    @staticmethod
    def get_default_hyperparameters_grid() -> List[Dict]:
        return [MASupportLevelsTradeAlgorithm.create_hyperparameters_dict(),
                MASupportLevelsTradeAlgorithm.create_hyperparameters_dict([20, 50]),
                MASupportLevelsTradeAlgorithm.create_hyperparameters_dict([50, 100, 200])]

    def train(self, data: pd.DataFrame, hyperparameters: Dict):
        super().train(data, hyperparameters)
        self._stock_name = hyperparameters["DATA_NAME"]
        self._indicator.set_ma_periods(hyperparameters["ma periods"])
        self._indicator.set_tested_MAs_usage(use_tested_MAs=True)
        self._indicator.clear_vars()
        self._indicator.calculate(self.data)
        self._indicator.test_MAs_for_data()

    def evaluate_new_point(self, new_point: pd.Series, date: Union[str, pd.Timestamp],
                           special_params: Optional[Dict] = None) -> TradeAction:
        return self._indicator.evaluate_new_point(new_point, date, special_params)

    def plot(self, img_dir: str, start_date: Optional[pd.Timestamp] = None,
             end_date: Optional[pd.Timestamp] = None, show_full: bool = False):
        if show_full:
            self._indicator.plot(img_dir, (self._stock_name + " full history"))
        else:
            self._indicator.plot(img_dir, self._stock_name, start_date, end_date)
