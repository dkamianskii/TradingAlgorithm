from typing import Optional, Union, Dict, List, Hashable

import pandas as pd
import numpy as np

from helping.base_enum import BaseEnum
from indicators.abstract_indicator import AbstractIndicator, TradePointColumn
from indicators.rsi import RSI
from indicators.macd import MACD, MACDTradeStrategy, MACDHyperparam
from indicators.super_trend import SuperTrend, SuperTrendHyperparam
from indicators.cci import CCI
from indicators.bollinger_bands import BollingerBands, BollingerBandsHyperparam
from indicators.ma_support_levels import MASupportLevels
from trading.trade_algorithms.abstract_trade_algorithm import AbstractTradeAlgorithm, TradeAction

import cufflinks as cf
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from trading.trade_algorithms.indicators_summary_trade_algorithms.indicators_summary_enums import \
    IndicatorPermittedToVote

cf.go_offline()


class IndicatorsVotingTradeAlgorithmHyperparam(BaseEnum):
    RSI_HYPERPARAMS = 1
    MACD_HYPERPARAMS = 2
    SUPER_TREND_HYPERPARAMS = 3
    CCI_HYPERPARAMS = 4
    BOLLINGER_BANDS_HYPERPARAMS = 5
    MA_SUPPORT_LEVELS_HYPERPARAMS = 6
    RETROSPECTIVE_PERIOD = 7
    QUALIFICATION_BARRIER = 8


class IndicatorsVotingTradeAlgorithm(AbstractTradeAlgorithm):
    name = "Indicators Voting trade algorithm"
    color_map = {TradeAction.ACTIVELY_BUY: "green",
                 TradeAction.BUY: "lightgreen",
                 TradeAction.NONE: "grey",
                 TradeAction.SELL: "red",
                 TradeAction.ACTIVELY_SELL: "maroon"}

    def __init__(self, indicators_permitted_to_vote: Optional[List[IndicatorPermittedToVote]] = None):
        super().__init__()
        self._indicators: List[AbstractIndicator] = []
        self._MACD: MACD = MACD()
        if (indicators_permitted_to_vote is None) or (IndicatorPermittedToVote.MACD in indicators_permitted_to_vote):
            self._indicators.append(self._MACD)
        self._RSI: RSI = RSI()
        if (indicators_permitted_to_vote is None) or (IndicatorPermittedToVote.RSI in indicators_permitted_to_vote):
            self._indicators.append(self._RSI)
        self._super_trend: SuperTrend = SuperTrend()
        if (indicators_permitted_to_vote is None) or (IndicatorPermittedToVote.SUPER_TREND in indicators_permitted_to_vote):
            self._indicators.append(self._super_trend)
        self._CCI: CCI = CCI()
        if (indicators_permitted_to_vote is None) or (IndicatorPermittedToVote.CCI in indicators_permitted_to_vote):
            self._indicators.append(self._CCI)
        self._bollinger_bands: BollingerBands = BollingerBands()
        if (indicators_permitted_to_vote is None) or (IndicatorPermittedToVote.BOLLINGER_BANDS in indicators_permitted_to_vote):
            self._indicators.append(self._bollinger_bands)
        self._ma_support_levels: MASupportLevels = MASupportLevels()
        if (indicators_permitted_to_vote is None) or (IndicatorPermittedToVote.MA_SUPPORT_LEVELS in indicators_permitted_to_vote):
            self._indicators.append(self._ma_support_levels)

        self._retrospective_period: int = 3
        self._qualification_barrier: float = 0.3
        self._polls: List[Dict[str, TradeAction]] = []
        self.polls_history: Optional[pd.DataFrame] = None
        self.trade_points: Optional[pd.DataFrame] = None

    @staticmethod
    def get_algorithm_name() -> str:
        return IndicatorsVotingTradeAlgorithm.name

    @staticmethod
    def create_hyperparameters_dict(retrospective_period: int = 3, qualification_barrier: float = 0.3,
                                    rsi_N: int = 14, macd_short_period: int = 12,
                                    macd_long_period: int = 26, macd_signal_period: int = 9,
                                    macd_trade_strategy: MACDTradeStrategy = MACDTradeStrategy.CLASSIC,
                                    super_trend_lookback_period: int = 10, super_trend_multiplier: float = 3,
                                    cci_N: int = 20, bollinger_bands_N: int = 20, bollinger_bands_K: float = 2,
                                    ma_support_levels_periods: List[
                                        int] = MASupportLevels.default_ma_periods_for_test) -> Dict:
        return {
            IndicatorsVotingTradeAlgorithmHyperparam.RETROSPECTIVE_PERIOD: retrospective_period,
            IndicatorsVotingTradeAlgorithmHyperparam.QUALIFICATION_BARRIER: qualification_barrier,
            IndicatorsVotingTradeAlgorithmHyperparam.RSI_HYPERPARAMS: {
                "N": rsi_N
            },
            IndicatorsVotingTradeAlgorithmHyperparam.MACD_HYPERPARAMS: {
                MACDHyperparam.SHORT_PERIOD: macd_short_period,
                MACDHyperparam.LONG_PERIOD: macd_long_period,
                MACDHyperparam.SIGNAL_PERIOD: macd_signal_period,
                MACDHyperparam.TRADE_STRATEGY: macd_trade_strategy
            },
            IndicatorsVotingTradeAlgorithmHyperparam.SUPER_TREND_HYPERPARAMS: {
                SuperTrendHyperparam.LOOKBACK_PERIOD: super_trend_lookback_period,
                SuperTrendHyperparam.MULTIPLIER: super_trend_multiplier
            },
            IndicatorsVotingTradeAlgorithmHyperparam.CCI_HYPERPARAMS: {
                "N": cci_N
            },
            IndicatorsVotingTradeAlgorithmHyperparam.BOLLINGER_BANDS_HYPERPARAMS: {
                BollingerBandsHyperparam.N: bollinger_bands_N,
                BollingerBandsHyperparam.K: bollinger_bands_K
            },
            IndicatorsVotingTradeAlgorithmHyperparam.MA_SUPPORT_LEVELS_HYPERPARAMS: {
                "ma_periods": ma_support_levels_periods
            }}

    @staticmethod
    def get_default_hyperparameters_grid() -> List[Dict]:
        return [IndicatorsVotingTradeAlgorithm.create_hyperparameters_dict(),
                IndicatorsVotingTradeAlgorithm.create_hyperparameters_dict(macd_short_period=10, macd_long_period=22,
                                                                           macd_trade_strategy=MACDTradeStrategy.CONVERGENCE),
                IndicatorsVotingTradeAlgorithm.create_hyperparameters_dict(macd_short_period=10, macd_long_period=20,
                                                                           super_trend_multiplier=2,
                                                                           super_trend_lookback_period=8),
                IndicatorsVotingTradeAlgorithm.create_hyperparameters_dict(super_trend_multiplier=2.5,
                                                                           super_trend_lookback_period=8,
                                                                           retrospective_period=5),
                IndicatorsVotingTradeAlgorithm.create_hyperparameters_dict(rsi_N=8, cci_N=14,
                                                                           retrospective_period=2),
                IndicatorsVotingTradeAlgorithm.create_hyperparameters_dict(rsi_N=8, macd_short_period=10,
                                                                           macd_long_period=22,
                                                                           macd_trade_strategy=MACDTradeStrategy.CONVERGENCE,
                                                                           cci_N=18, bollinger_bands_N=14,
                                                                           bollinger_bands_K=1.5)]

    def __clear_vars(self):
        self.trade_points = pd.DataFrame(columns=TradePointColumn.get_elements_list()).set_index(TradePointColumn.DATE)
        columns = [indicator.name for indicator in self._indicators]
        columns.append("Date")
        self._polls = []
        self.polls_history = pd.DataFrame(columns=columns).set_index("Date")
        for indicator in self._indicators:
            indicator.clear_vars()

    def train(self, data: pd.DataFrame, hyperparameters: Dict):
        if data.shape[0] > 2000:
            super().train(data[-2000:], hyperparameters)
        else:
            super().train(data, hyperparameters)
        self._RSI.set_N(hyperparameters[IndicatorsVotingTradeAlgorithmHyperparam.RSI_HYPERPARAMS]["N"])
        self._MACD.set_ma_periods(
            hyperparameters[IndicatorsVotingTradeAlgorithmHyperparam.MACD_HYPERPARAMS][MACDHyperparam.SHORT_PERIOD],
            hyperparameters[IndicatorsVotingTradeAlgorithmHyperparam.MACD_HYPERPARAMS][MACDHyperparam.LONG_PERIOD],
            hyperparameters[IndicatorsVotingTradeAlgorithmHyperparam.MACD_HYPERPARAMS][MACDHyperparam.SIGNAL_PERIOD])
        self._MACD.set_trade_strategy(hyperparameters[IndicatorsVotingTradeAlgorithmHyperparam.MACD_HYPERPARAMS][
                                          MACDHyperparam.TRADE_STRATEGY])
        self._super_trend.set_params(
            lookback_period=hyperparameters[IndicatorsVotingTradeAlgorithmHyperparam.SUPER_TREND_HYPERPARAMS][
                SuperTrendHyperparam.LOOKBACK_PERIOD],
            multiplier=hyperparameters[IndicatorsVotingTradeAlgorithmHyperparam.SUPER_TREND_HYPERPARAMS][
                SuperTrendHyperparam.MULTIPLIER])
        self._CCI.set_params(hyperparameters[IndicatorsVotingTradeAlgorithmHyperparam.CCI_HYPERPARAMS]["N"])
        self._bollinger_bands.set_params(
            hyperparameters[IndicatorsVotingTradeAlgorithmHyperparam.BOLLINGER_BANDS_HYPERPARAMS][
                BollingerBandsHyperparam.N],
            hyperparameters[IndicatorsVotingTradeAlgorithmHyperparam.BOLLINGER_BANDS_HYPERPARAMS][
                BollingerBandsHyperparam.K])
        self._ma_support_levels.set_ma_periods(
            hyperparameters[IndicatorsVotingTradeAlgorithmHyperparam.MA_SUPPORT_LEVELS_HYPERPARAMS]["ma_periods"])
        self._ma_support_levels.set_tested_MAs_usage(use_tested_MAs=True)

        self.__clear_vars()
        for indicator in self._indicators:
            indicator.calculate(self.data)
        self._ma_support_levels.test_MAs_for_data()

    def evaluate_new_point(self, new_point: pd.Series, date: Union[str, pd.Timestamp],
                           special_params: Optional[Dict] = None) -> TradeAction:
        poll: Dict[str, TradeAction] = {}
        for indicator in self._indicators:
            poll[indicator.name] = indicator.evaluate_new_point(new_point, date, special_params, False)
        self.data.loc[date] = new_point

        if len(self._polls) == self._retrospective_period:
            self._polls.pop(0)
        self._polls.append(poll)

        cumulative_poll: Dict[str, TradeAction] = {}

        for indicator in self._indicators:
            indicator_name = indicator.name
            cumulative_poll[indicator_name] = TradeAction.NONE
            for poll in self._polls:
                if poll[indicator_name] != TradeAction.NONE:
                    cumulative_poll[indicator_name] = poll[indicator_name]

        poll_results: Dict[TradeAction, int] = {action: 0 for action in TradeAction.get_elements_list()}
        buy_dir, sell_dir = 0, 0
        for _, action in cumulative_poll.items():
            poll_results[action] += 1
            if (action == TradeAction.BUY) or (action == TradeAction.ACTIVELY_BUY):
                buy_dir += 1
            elif (action == TradeAction.SELL) or (action == TradeAction.ACTIVELY_SELL):
                sell_dir += 1

        final_action = TradeAction.NONE
        if (buy_dir / len(self._indicators) >= self._qualification_barrier) or (
                sell_dir / len(self._indicators) >= self._qualification_barrier):
            if (poll_results[TradeAction.NONE] >= buy_dir) and (poll_results[TradeAction.NONE] >= sell_dir):
                if buy_dir == 0:
                    final_action = TradeAction.SELL
                elif sell_dir == 0:
                    final_action = TradeAction.BUY
            elif buy_dir > sell_dir:
                final_action = TradeAction.BUY
            elif sell_dir > buy_dir:
                final_action = TradeAction.SELL

        if (final_action == TradeAction.BUY) and (
                poll_results[TradeAction.ACTIVELY_BUY] > poll_results[TradeAction.BUY]):
            final_action = TradeAction.ACTIVELY_BUY
        elif final_action == TradeAction.SELL and (
                poll_results[TradeAction.ACTIVELY_SELL] > poll_results[TradeAction.SELL]):
            final_action = TradeAction.ACTIVELY_SELL

        self.polls_history.loc[date] = cumulative_poll
        if final_action != TradeAction.NONE:
            self.__add_trade_point(date, new_point["Close"], final_action)
        return final_action

    def __add_trade_point(self, date: Union[pd.Timestamp, Hashable], price: float, action: TradeAction):
        self.trade_points.loc[date] = {TradePointColumn.PRICE: price, TradePointColumn.ACTION: action}

    def plot(self, start_date: Optional[pd.Timestamp] = None, end_date: Optional[pd.Timestamp] = None):
        if (start_date is None) or (start_date < self.data.index[0]):
            start_date = self.data.index[0]
        if (end_date is None) or (end_date > self.data.index[-1]):
            end_date = self.data.index[-1]

        selected_data = self.data[start_date:end_date]
        selected_trade_points = self.trade_points[start_date:end_date]

        fig = make_subplots(rows=2, cols=1,
                            shared_xaxes=True,
                            subplot_titles=[
                                "Price with Indicators votes",
                                "Indicators polls"],
                            vertical_spacing=0.25)
        fig.add_candlestick(x=selected_data.index,
                            open=selected_data["Open"],
                            close=selected_data["Close"],
                            high=selected_data["High"],
                            low=selected_data["Low"],
                            name="Price",
                            row=1, col=1)

        buy_actions = [TradeAction.BUY, TradeAction.ACTIVELY_BUY]
        active_actions = [TradeAction.ACTIVELY_BUY, TradeAction.ACTIVELY_SELL]
        bool_buys = selected_trade_points[TradePointColumn.ACTION].isin(buy_actions)
        bool_actives = selected_trade_points[TradePointColumn.ACTION].isin(active_actions)

        fig.add_trace(go.Scatter(x=selected_trade_points.index,
                                 y=selected_trade_points[TradePointColumn.PRICE],
                                 mode="markers",
                                 marker=dict(
                                     color=np.where(bool_buys, "green", "red"),
                                     size=np.where(bool_actives, 15, 10),
                                     symbol=np.where(bool_buys, "triangle-up", "triangle-down")),
                                 name="Action points"),
                      row=1, col=1)

        for action, color in IndicatorsVotingTradeAlgorithm.color_map.items():
            action_dates = []
            for indicator in self._indicators:
                temp = self.polls_history[self.polls_history[indicator.name] == action]
                action_dates.append(pd.DataFrame(data={"date": temp.index,
                                                       "indicator": [indicator.name]*temp.shape[0]}))
            action_full = pd.concat(action_dates)
            fig.add_trace(go.Scatter(x=action_full["date"],
                                     y=action_full["indicator"],
                                     mode="markers",
                                     marker=dict(
                                         color=color,
                                         size=10,
                                         symbol="square"),
                                     name=action.name),
                          row=2, col=1)
        fig.show()
