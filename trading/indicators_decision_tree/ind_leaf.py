import random
from typing import Optional, List, Tuple, Dict

import pandas as pd

from indicators.abstract_indicator import TradeAction


class IndLeaf:
    def __init__(self, data: Optional[pd.DataFrame],
                 label: str,
                 parents_brunch_direction: List[TradeAction],
                 leaf_action: TradeAction):
        self._determinate_trade_action: Optional[TradeAction] = None
        self._stochastic_trade_action: Optional[Dict[TradeAction, Tuple[int, int]]] = None
        self._stochastic_base: int = 0

        if data is None:
            trade_direction = parents_brunch_direction.copy()
            trade_direction.append(leaf_action)
            self._determinate_trade_action = IndLeaf.evaluate_possible_trade_action(trade_direction)
        else:
            possible_trade_actions = data[label].unique()
            if len(possible_trade_actions) == 1:
                self._determinate_trade_action = possible_trade_actions[0]
            else:
                self._stochastic_trade_action = {}
                self._stochastic_base = data.shape[0]
                lower_bound = 0
                for trade_action in possible_trade_actions:
                    trade_action_data = data[data[label] == trade_action]
                    trade_action_points = trade_action_data.shape[0]
                    self._stochastic_trade_action[trade_action] = (lower_bound + 1, lower_bound + trade_action_points)
                    lower_bound += trade_action_points

    @staticmethod
    def evaluate_possible_trade_action(trade_direction: List[TradeAction]) -> TradeAction:
        buys, sells = 0, 0
        for action in trade_direction:
            if (action == TradeAction.BUY) or (action == TradeAction.ACTIVELY_BUY):
                buys += 1
            elif (action == TradeAction.SELL) or (action == TradeAction.ACTIVELY_SELL):
                sells += 1

        if (sells == 0) and (buys / len(trade_direction) >= 0.3):
            return TradeAction.BUY
        elif (buys == 0) and (sells / len(trade_direction) >= 0.3):
            return TradeAction.SELL
        return TradeAction.NONE

    def get_trade_action(self, indicators_values) -> TradeAction:
        if self._determinate_trade_action is not None:
            return self._determinate_trade_action

        stochastic_result = random.randint(1, self._stochastic_base,)
        for trade_action, boounds in self._stochastic_trade_action.items():
            if (stochastic_result >= boounds[0]) and (stochastic_result <= boounds[1]):
                return trade_action

    def print(self, prev_brunch_to_print: List[str], file_path: str):
        brunch_to_print = prev_brunch_to_print.copy()
        if self._determinate_trade_action is not None:
            brunch_to_print.append(f"|{self._determinate_trade_action}|")
        else:
            brunch_to_print.append("|")
            for action, bounds in self._stochastic_trade_action.items():
                brunch_to_print.append(f"P({action})={(bounds[1] - bounds[0] + 1) / self._stochastic_base}|")
        str_to_write = "".join(brunch_to_print)
        with open(file_path, "a") as f:
            f.write(str_to_write)
            f.write("\n")
