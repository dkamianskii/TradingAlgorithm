from abc import ABC

import pandas as pd
from typing import Optional

from Trading.AbstractTradeAlgorithm import AbstractTradeAlgorithm
from indicators.SuperTrend import SuperTrend


class MACDSuperTrendTradeAlgorithm(AbstractTradeAlgorithm):

    def __init__(self):
        super().__init__()
        self.__super_trend: SuperTrend = SuperTrend()

    def train(self, data: pd.DataFrame):
        super().train(data)
        pass

    def day_analysis(self, new_day_data: pd.Series) -> pd.DataFrame:
        super().day_analysis(new_day_data)
        pass