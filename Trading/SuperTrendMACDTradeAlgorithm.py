import pandas as pd
from typing import Optional

from Trading.AbstractTradeAlgorithm import AbstractTradeAlgorithm


class SuperTrendMACDTradeAlgorithm(AbstractTradeAlgorithm):

    def __init__(self):
        super().__init__()

    def train(self, data: pd.DataFrame):
        pass

    def trade(self, data: pd.DataFrame) -> pd.DataFrame:
        pass
