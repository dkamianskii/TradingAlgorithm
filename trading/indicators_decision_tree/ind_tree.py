import random

import pandas as pd
from typing import List, Union, Dict

from indicators.abstract_indicator import TradeAction
from trading.indicators_decision_tree.ind_brunch import IndBrunch


class IndTree:

    def __init__(self, data: pd.DataFrame, indicators: List[str], label: str = "label"):
        random.seed(42)
        self.label: str = label
        self.indicators: List[str] = indicators
        self.root: IndBrunch = IndBrunch(data, indicators, 0, label, [], None)

    def get_trade_action(self, indicators_values: Union[Dict[str, TradeAction], pd.Series]) -> TradeAction:
        return self.root.get_trade_action(indicators_values)

    def print_tree(self, file_path: str):
        self.root.print([], file_path)

