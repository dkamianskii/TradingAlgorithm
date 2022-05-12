from typing import Tuple

import numpy as np
import pandas as pd
from indicators.abstract_indicator import TradeAction


class RiskManager:

    def __init__(self,
                 use_limited_money: bool = False,
                 money_for_a_bid: float = 10000,
                 start_capital: float = 0,
                 equity_risk_rate: float = 0.02,
                 bid_risk_rate: float = 0.03,
                 take_profit_multiplier: float = 2,
                 active_action_multiplier: float = 1.5,
                 use_atr: bool = False
                 ):
        self.available_money: float = start_capital
        self.account_money: float = start_capital
        self.start_capital: float = start_capital

        self._use_limited_money: bool = use_limited_money
        self._money_for_a_bid: float = money_for_a_bid

        self._equity_risk_rate: float = equity_risk_rate
        self._bid_risk_rate: float = bid_risk_rate
        self._take_profit_multiplier: float = take_profit_multiplier
        self._active_action_multiplier: float = active_action_multiplier
        self._use_atr: bool = use_atr

    def set_manager_params(self,
                           use_limited_money: bool = False,
                           money_for_a_bid: float = 10000,
                           start_capital: float = 0,
                           equity_risk_rate: float = 0.02,
                           bid_risk_rate: float = 0.03,
                           take_profit_multiplier: float = 2,
                           active_action_multiplier: float = 1.5,
                           use_atr: bool = False):
        self.available_money: float = start_capital
        self.account_money: float = start_capital
        self.start_capital: float = start_capital

        self._use_limited_money: bool = use_limited_money
        self._money_for_a_bid: float = money_for_a_bid

        self._equity_risk_rate: float = equity_risk_rate
        self._bid_risk_rate: float = bid_risk_rate
        self._take_profit_multiplier: float = take_profit_multiplier
        self._active_action_multiplier: float = active_action_multiplier
        self._use_atr: bool = use_atr

    def reset_money(self):
        self.available_money = self.start_capital
        self.available_money = self.start_capital

    def set_manager_params_dict(self, params_dict):
        for param, value in params_dict.items():
            if param not in self.__dict__:
                raise ValueError("Uknown parameter was provided")
            setattr(self, "_" + param, value)

    def set_money_for_bid(self, bid_money_value: float):
        if self._use_limited_money:
            self.available_money -= bid_money_value

    def set_bid_returns(self, cashback: float, profit: float):
        if self._use_limited_money:
            self.available_money += cashback
            self.account_money += profit

    def evaluate_stop_loss_and_take_profit(self,
                                           new_point: pd.Series,
                                           action: TradeAction,
                                           atr: float) -> Tuple[float, float]:
        price = new_point["Close"]
        if self._use_atr:
            if action == TradeAction.BUY:
                stop_loss = price - atr
                take_profit = price + atr * self._take_profit_multiplier
            elif action == TradeAction.ACTIVELY_BUY:
                stop_loss = price - atr
                take_profit = price + atr * self._take_profit_multiplier * self._active_action_multiplier
            elif action == TradeAction.SELL:
                stop_loss = price + atr
                take_profit = price - atr * self._take_profit_multiplier
            else:  # TradeAction.ACTIVELY_SELL
                stop_loss = price + atr
                take_profit = price - atr * self._take_profit_multiplier * self._active_action_multiplier
        else:
            if action == TradeAction.BUY:
                stop_loss = price * (1 - self._bid_risk_rate)
                take_profit = price * (1 + self._bid_risk_rate * self._take_profit_multiplier)
            elif action == TradeAction.ACTIVELY_BUY:
                stop_loss = price * (1 - self._bid_risk_rate)
                take_profit = price * (
                        1 + self._bid_risk_rate * self._take_profit_multiplier * self._active_action_multiplier)
            elif action == TradeAction.SELL:
                stop_loss = price * (1 + self._bid_risk_rate)
                take_profit = price * (
                        1 - self._bid_risk_rate * self._take_profit_multiplier)
            else:  # TradeAction.ACTIVELY_SELL
                stop_loss = price * (1 + self._bid_risk_rate)
                take_profit = price * (
                        1 - self._bid_risk_rate * self._take_profit_multiplier * self._active_action_multiplier)
        return stop_loss, take_profit

    def evaluate_shares_amount_to_bid(self, price: float) -> int:
        shares_to_buy = 0
        if self._use_limited_money:
            money_to_risk = self.account_money * self._equity_risk_rate
            if self.available_money > money_to_risk:
                shares_to_buy = np.floor(money_to_risk / price)
        else:
            shares_to_buy = np.floor(self._money_for_a_bid / price)
        return shares_to_buy
