from typing import Tuple

import numpy as np
import pandas as pd

from helping.base_enum import BaseEnum
from indicators.abstract_indicator import TradeAction
from trading.trade_statistics_manager_enums import EarningsHistoryColumn


class RiskManagerHyperparam(BaseEnum):
    USE_LIMITED_MONEY = 1
    MONEY_FOR_A_BID = 2
    START_CAPITAL = 3
    EQUITY_RISK_RATE = 4
    BID_RISK_RATE = 5
    TAKE_PROFIT_MULTIPLIER = 6
    ACTIVE_ACTION_MULTIPLIER = 7
    USE_ATR = 8


class RiskManager:
    risk_free_rate = 0.0154

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
        self.account_money = self.start_capital
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
                shares_to_buy = np.floor(self.available_money / price)
        else:
            shares_to_buy = np.floor(self._money_for_a_bid / price)
        return shares_to_buy

    def evaluate_sharpe_ratio(self, earnings_history: pd.DataFrame) -> float:
        earnings: pd.Series = earnings_history[earnings_history[EarningsHistoryColumn.VALUE] != 0][EarningsHistoryColumn.VALUE]
        return_rates: pd.Series = earnings / self.start_capital
        expected_return = return_rates.mean()
        std_of_return = return_rates.std()
        if expected_return >= 0:
            sharpe_ratio = expected_return / (1 + std_of_return)
        else:
            sharpe_ratio = expected_return * (1 + std_of_return)
        return sharpe_ratio

    def evaluate_sortino_ratio(self, earnings_history: pd.DataFrame) -> float:
        downsides = earnings_history[earnings_history[EarningsHistoryColumn.VALUE] < 0][EarningsHistoryColumn.VALUE]
        earnings: pd.Series = earnings_history[earnings_history[EarningsHistoryColumn.VALUE] != 0][
            EarningsHistoryColumn.VALUE]
        return_rates: pd.Series = earnings / self.start_capital
        expected_return = return_rates.mean()
        std_of_downside = downsides.std()
        if expected_return >= 0:
            sharpe_ratio = expected_return / (1 + std_of_downside)
        else:
            sharpe_ratio = expected_return * (1 + std_of_downside)
        return sharpe_ratio

    def evaluate_calmar_ratio(self, earnings_history: pd.DataFrame) -> float:
        earnings = earnings_history[earnings_history[EarningsHistoryColumn.VALUE] != 0][EarningsHistoryColumn.VALUE]
        drowdown = self.evaluate_max_drowdown(earnings_history)
        return_rates: pd.Series = earnings / self.start_capital
        expected_return = return_rates.mean()
        if expected_return > 0:
            return expected_return / (1 + drowdown / self.start_capital)
        else:
            return expected_return * (1 + drowdown / self.start_capital)

    @staticmethod
    def evaluate_max_drowdown(earnings_history: pd.DataFrame) -> float:
        earnings = earnings_history[earnings_history[EarningsHistoryColumn.VALUE] != 0][EarningsHistoryColumn.VALUE]
        drowdown, max_drowdown = 0, 0
        for earning in earnings:
            drowdown -= earning
            if drowdown < 0:
                drowdown = 0
            elif drowdown > max_drowdown:
                max_drowdown = drowdown
        return max_drowdown



