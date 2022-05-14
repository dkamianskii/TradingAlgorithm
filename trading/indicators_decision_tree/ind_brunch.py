import pandas as pd
from typing import List, Union, Dict, Optional

from indicators.abstract_indicator import TradeAction
from trading.indicators_decision_tree.ind_leaf import IndLeaf


class IndBrunch:

    def __init__(self, data: pd.DataFrame,
                 indicators: List[str],
                 indicator_index: int,
                 label: str,
                 parents_brunch_direction: List[TradeAction],
                 brunch_action: Optional[TradeAction]):
        self.direction: List[TradeAction] = parents_brunch_direction.copy()
        if brunch_action is not None:
            self.direction.append(brunch_action)
        self.action = brunch_action
        self.indicator = indicators[indicator_index]
        self.children: Dict[TradeAction, Union[IndBrunch, IndLeaf]] = {}

        for action in TradeAction:
            child_data = data[data[self.indicator] == action]
            if child_data.shape[0] == 0:
                self.children[action] = IndLeaf(None, label, self.direction, action)
            elif indicator_index == (len(indicators) - 1):
                self.children[action] = IndLeaf(child_data, label, self.direction, action)
            elif len(child_data[label].unique()) == 1:
                self.children[action] = IndLeaf(child_data, label, self.direction, action)
            else:
                self.children[action] = IndBrunch(child_data, indicators, indicator_index + 1, label, self.direction, action)

    def get_trade_action(self, indicators_values: Union[Dict[str, TradeAction], pd.Series]) -> TradeAction:
        indicator_action = indicators_values[self.indicator]
        return self.children[indicator_action].get_trade_action(indicators_values)

    def print(self, prev_brunch_to_print: List[str]):
        for action, child in self.children.items():
            brunch_to_print = prev_brunch_to_print.copy()
            brunch_to_print.append(f"{self.indicator}|{action} -> ")
            child.print(brunch_to_print)
