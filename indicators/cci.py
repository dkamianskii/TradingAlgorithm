from typing import Optional, Union

import pandas as pd

from indicators.abstract_indicator import AbstractIndicator, TradeAction


class CCI(AbstractIndicator):  # todo сделать индикатор CCI (low priority)
    def calculate(self, data: Optional[pd.DataFrame] = None):
        pass

    def find_trade_points(self) -> pd.DataFrame:
        pass

    def evaluate_new_point(self, new_point: pd.Series, date: Union[str, pd.Timestamp], special_params: Optional = None,
                           update_data: bool = True) -> TradeAction:
        pass

    def plot(self, start_date: Optional[pd.Timestamp] = None, end_date: Optional[pd.Timestamp] = None):
        pass

    def clear_vars(self):
        pass
