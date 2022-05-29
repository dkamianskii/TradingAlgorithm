from typing import Optional, Union, Dict, List

import pandas as pd
import numpy as np

from helping.base_enum import BaseEnum
from indicators.abstract_indicator import TradeAction, AbstractIndicator, TradePointColumn
from indicators.atr import ATR
from indicators.bollinger_bands import BollingerBands, BollingerBandsHyperparam
from indicators.cci import CCI
from indicators.ma_support_levels import MASupportLevels
from indicators.macd import MACD, MACDHyperparam, MACDTradeStrategy
from indicators.rsi import RSI
from indicators.super_trend import SuperTrend, SuperTrendHyperparam
from trading.indicators_decision_tree.ind_tree import IndTree
from trading.trade_algorithms.abstract_trade_algorithm import AbstractTradeAlgorithm
from trading.risk_manager import RiskManager, RiskManagerHyperparam
from trading.trade_algorithms.indicators_summary_trade_algorithms.indicators_summary_enums import \
    IndicatorPermittedToVote

import cufflinks as cf
import plotly.graph_objects as go

cf.go_offline()


class DecisionTreeTradeAlgorithmHyperparam(BaseEnum):
    RSI_HYPERPARAMS = 1
    MACD_HYPERPARAMS = 2
    SUPER_TREND_HYPERPARAMS = 3
    CCI_HYPERPARAMS = 4
    BOLLINGER_BANDS_HYPERPARAMS = 5
    MA_SUPPORT_LEVELS_HYPERPARAMS = 6
    RISK_MANAGER_HYPERPARAMS = 7
    DAYS_TO_KEEP_LIMIT = 8
    ATR_PERIOD = 9


class DecisionTreeTradeAlgorithm(AbstractTradeAlgorithm):
    name = "Decision Tree trade algorithm"

    def __init__(self, indicators_permitted_to_vote: Optional[List[IndicatorPermittedToVote]] = None):
        super().__init__()
        self._risk_manager = RiskManager()
        self._stock_name = ""
        self._decision_tree: Optional[IndTree] = None
        self._indicators: List[AbstractIndicator] = []
        self._indicators_trade_points: Dict[str, pd.DataFrame] = {}
        self._dataframe: pd.DataFrame = pd.DataFrame()
        self._days_to_keep_limit: int = 0
        self._atr_period: int = 14
        self._atr: Optional[pd.Series] = None
        self._MACD: MACD = MACD()
        if (indicators_permitted_to_vote is None) or (IndicatorPermittedToVote.MACD in indicators_permitted_to_vote):
            self._indicators.append(self._MACD)
        self._RSI: RSI = RSI()
        if (indicators_permitted_to_vote is None) or (IndicatorPermittedToVote.RSI in indicators_permitted_to_vote):
            self._indicators.append(self._RSI)
        self._super_trend: SuperTrend = SuperTrend()
        if (indicators_permitted_to_vote is None) or (
                IndicatorPermittedToVote.SUPER_TREND in indicators_permitted_to_vote):
            self._indicators.append(self._super_trend)
        self._CCI: CCI = CCI()
        if (indicators_permitted_to_vote is None) or (IndicatorPermittedToVote.CCI in indicators_permitted_to_vote):
            self._indicators.append(self._CCI)
        self._bollinger_bands: BollingerBands = BollingerBands()
        if (indicators_permitted_to_vote is None) or (
                IndicatorPermittedToVote.BOLLINGER_BANDS in indicators_permitted_to_vote):
            self._indicators.append(self._bollinger_bands)
        self._ma_support_levels: MASupportLevels = MASupportLevels()
        if (indicators_permitted_to_vote is None) or (
                IndicatorPermittedToVote.MA_SUPPORT_LEVELS in indicators_permitted_to_vote):
            self._indicators.append(self._ma_support_levels)

    @staticmethod
    def get_algorithm_name() -> str:
        return DecisionTreeTradeAlgorithm.name

    @staticmethod
    def create_hyperparameters_dict(bid_risk_rate: float = 0.03, take_profit_multiplier: float = 2,
                                    active_action_multiplier: float = 1.5, use_atr: bool = False,
                                    days_to_keep_limit: int = 14, atr_period: int = 14,
                                    rsi_N: int = 14, macd_short_period: int = 12,
                                    macd_long_period: int = 26, macd_signal_period: int = 9,
                                    macd_trade_strategy: MACDTradeStrategy = MACDTradeStrategy.CLASSIC,
                                    super_trend_lookback_period: int = 10, super_trend_multiplier: float = 3,
                                    cci_N: int = 20, bollinger_bands_N: int = 20, bollinger_bands_K: float = 2,
                                    ma_support_levels_periods: List[
                                        int] = MASupportLevels.default_ma_periods_for_test) -> Dict:
        return {
            DecisionTreeTradeAlgorithmHyperparam.DAYS_TO_KEEP_LIMIT: days_to_keep_limit,
            DecisionTreeTradeAlgorithmHyperparam.ATR_PERIOD: atr_period,
            DecisionTreeTradeAlgorithmHyperparam.RISK_MANAGER_HYPERPARAMS: {
                RiskManagerHyperparam.BID_RISK_RATE: bid_risk_rate,
                RiskManagerHyperparam.TAKE_PROFIT_MULTIPLIER: take_profit_multiplier,
                RiskManagerHyperparam.ACTIVE_ACTION_MULTIPLIER: active_action_multiplier,
                RiskManagerHyperparam.USE_ATR: use_atr
            },
            DecisionTreeTradeAlgorithmHyperparam.RSI_HYPERPARAMS: {
                "N": rsi_N
            },
            DecisionTreeTradeAlgorithmHyperparam.MACD_HYPERPARAMS: {
                MACDHyperparam.SHORT_PERIOD: macd_short_period,
                MACDHyperparam.LONG_PERIOD: macd_long_period,
                MACDHyperparam.SIGNAL_PERIOD: macd_signal_period,
                MACDHyperparam.TRADE_STRATEGY: macd_trade_strategy
            },
            DecisionTreeTradeAlgorithmHyperparam.SUPER_TREND_HYPERPARAMS: {
                SuperTrendHyperparam.LOOKBACK_PERIOD: super_trend_lookback_period,
                SuperTrendHyperparam.MULTIPLIER: super_trend_multiplier
            },
            DecisionTreeTradeAlgorithmHyperparam.CCI_HYPERPARAMS: {
                "N": cci_N
            },
            DecisionTreeTradeAlgorithmHyperparam.BOLLINGER_BANDS_HYPERPARAMS: {
                BollingerBandsHyperparam.N: bollinger_bands_N,
                BollingerBandsHyperparam.K: bollinger_bands_K
            },
            DecisionTreeTradeAlgorithmHyperparam.MA_SUPPORT_LEVELS_HYPERPARAMS: {
                "ma_periods": ma_support_levels_periods
            }}

    @staticmethod
    def get_default_hyperparameters_grid() -> List[Dict]:
        return [DecisionTreeTradeAlgorithm.create_hyperparameters_dict(),
                DecisionTreeTradeAlgorithm.create_hyperparameters_dict(macd_short_period=10, macd_long_period=22,
                                                                       macd_trade_strategy=MACDTradeStrategy.CONVERGENCE),
                DecisionTreeTradeAlgorithm.create_hyperparameters_dict(macd_short_period=10, macd_long_period=20,
                                                                       super_trend_multiplier=2,
                                                                       super_trend_lookback_period=8),
                DecisionTreeTradeAlgorithm.create_hyperparameters_dict(bid_risk_rate=0.015, take_profit_multiplier=3),
                DecisionTreeTradeAlgorithm.create_hyperparameters_dict(bid_risk_rate=0.02, active_action_multiplier=2),
                DecisionTreeTradeAlgorithm.create_hyperparameters_dict(use_atr=True, take_profit_multiplier=1.5),
                DecisionTreeTradeAlgorithm.create_hyperparameters_dict(rsi_N=8, macd_short_period=10,
                                                                       macd_long_period=22,
                                                                       macd_trade_strategy=MACDTradeStrategy.CONVERGENCE,
                                                                       cci_N=18, bollinger_bands_N=14,
                                                                       bollinger_bands_K=1.5)]

    def __clear_vars(self):
        # self.trade_points = pd.DataFrame(columns=TradePointColumn.get_elements_list()).set_index(TradePointColumn.DATE)
        self._indicators_trade_points = {}
        self._dataframe = pd.DataFrame()
        for indicator in self._indicators:
            indicator.clear_vars()

    def __create_train_dataset(self):
        index = 0
        self._dataframe["label"] = TradeAction.NONE
        self._dataframe["take profit"] = 0.
        self._dataframe["stop loss"] = 0.
        self._dataframe["exit date"] = self.data.index
        for date, point in self.data.iterrows():
            label = TradeAction.NONE
            for indicator_name, trade_points in self._indicators_trade_points.items():
                if date < trade_points.index[0]:
                    continue
                action = trade_points.loc[date][TradePointColumn.ACTION]
                if action != TradeAction.NONE:
                    self._dataframe.loc[date, indicator_name] = action
                    if label == TradeAction.NONE:
                        if date < self._atr.index[0]:
                            atr = self._atr[0]
                        else:
                            atr = self._atr.loc[date]
                        right_action, take_profit, stop_loss, exit_date = self.__evaluate_right_action(point, index,
                                                                                                       atr)
                        self._dataframe.loc[date, ["label", "price", "take profit", "stop loss", "exit date"]] = [
                            right_action,
                            point["Close"],
                            take_profit,
                            stop_loss,
                            exit_date]
            index += 1

        not_all_none = None
        for indicator in self._indicators:
            not_none = self._dataframe[indicator.name] != TradeAction.NONE
            if not_all_none is None:
                not_all_none = not_none
            else:
                not_all_none = np.logical_or(not_all_none, not_none)
        self._dataframe = self._dataframe[not_all_none]

    def __evaluate_right_action(self, start_point: pd.Series, start_index: int, atr: float) -> (
            TradeAction, float, float, pd.Timestamp):
        """
        return trade action, take profit, stop loss, exit date
        """
        buy_stop_loss, buy_take_profit = self._risk_manager.evaluate_stop_loss_and_take_profit(start_point,
                                                                                               TradeAction.BUY,
                                                                                               atr)
        _, actively_buy_take_profit = self._risk_manager.evaluate_stop_loss_and_take_profit(start_point,
                                                                                            TradeAction.ACTIVELY_BUY,
                                                                                            atr)
        sell_stop_loss, sell_take_profit = self._risk_manager.evaluate_stop_loss_and_take_profit(start_point,
                                                                                                 TradeAction.SELL,
                                                                                                 atr)
        _, actively_sell_take_profit = self._risk_manager.evaluate_stop_loss_and_take_profit(start_point,
                                                                                             TradeAction.ACTIVELY_SELL,
                                                                                             atr)
        first_action_happened = TradeAction.NONE
        buy_stop_loss_triggered = False
        sell_stop_loss_triggered = False
        right_action = TradeAction.NONE
        date_shift, final_take_profit, final_stop_loss = 0, 0., 0.
        for j in range(1, self._days_to_keep_limit):
            if (start_index + j) == self.data.shape[0]:
                break
            date_shift += 1
            cur_point = self.data.iloc[start_index + j]
            cur_price = cur_point["Close"]
            if cur_price <= buy_stop_loss:
                buy_stop_loss_triggered = True
            if cur_price >= sell_stop_loss:
                sell_stop_loss_triggered = True
            if buy_stop_loss_triggered and sell_stop_loss_triggered:
                break
            if first_action_happened == TradeAction.NONE:
                if not buy_stop_loss_triggered and (cur_price >= buy_take_profit):
                    first_action_happened = TradeAction.BUY
                    right_action = TradeAction.BUY
                    final_take_profit, final_stop_loss = buy_take_profit, buy_stop_loss
                elif not sell_stop_loss_triggered and (cur_price <= sell_take_profit):
                    first_action_happened = TradeAction.SELL
                    right_action = TradeAction.SELL
                    final_take_profit, final_stop_loss = sell_take_profit, sell_stop_loss
            if first_action_happened == TradeAction.BUY:
                if buy_stop_loss_triggered:
                    break
                if cur_price >= actively_buy_take_profit:
                    right_action = TradeAction.ACTIVELY_BUY
                    final_take_profit = actively_buy_take_profit
                    break
            elif first_action_happened == TradeAction.SELL:
                if sell_stop_loss_triggered:
                    break
                if cur_price <= actively_sell_take_profit:
                    right_action = TradeAction.ACTIVELY_SELL
                    final_take_profit = actively_sell_take_profit
                    break

        exit_date = self.data.index[start_index + date_shift]
        return right_action, final_take_profit, final_stop_loss, exit_date

    def train(self, data: pd.DataFrame, hyperparameters: Dict):
        super().train(data, hyperparameters)
        self._risk_manager.set_manager_params(
            bid_risk_rate=hyperparameters[DecisionTreeTradeAlgorithmHyperparam.RISK_MANAGER_HYPERPARAMS][
                RiskManagerHyperparam.BID_RISK_RATE],
            take_profit_multiplier=hyperparameters[DecisionTreeTradeAlgorithmHyperparam.RISK_MANAGER_HYPERPARAMS][
                RiskManagerHyperparam.TAKE_PROFIT_MULTIPLIER],
            active_action_multiplier=hyperparameters[DecisionTreeTradeAlgorithmHyperparam.RISK_MANAGER_HYPERPARAMS][
                RiskManagerHyperparam.ACTIVE_ACTION_MULTIPLIER],
            use_atr=hyperparameters[DecisionTreeTradeAlgorithmHyperparam.RISK_MANAGER_HYPERPARAMS][
                RiskManagerHyperparam.USE_ATR])
        self._RSI.set_N(hyperparameters[DecisionTreeTradeAlgorithmHyperparam.RSI_HYPERPARAMS]["N"])
        self._MACD.set_ma_periods(
            hyperparameters[DecisionTreeTradeAlgorithmHyperparam.MACD_HYPERPARAMS][MACDHyperparam.SHORT_PERIOD],
            hyperparameters[DecisionTreeTradeAlgorithmHyperparam.MACD_HYPERPARAMS][MACDHyperparam.LONG_PERIOD],
            hyperparameters[DecisionTreeTradeAlgorithmHyperparam.MACD_HYPERPARAMS][MACDHyperparam.SIGNAL_PERIOD])
        self._MACD.set_trade_strategy(hyperparameters[DecisionTreeTradeAlgorithmHyperparam.MACD_HYPERPARAMS][
                                          MACDHyperparam.TRADE_STRATEGY])
        self._super_trend.set_params(
            lookback_period=hyperparameters[DecisionTreeTradeAlgorithmHyperparam.SUPER_TREND_HYPERPARAMS][
                SuperTrendHyperparam.LOOKBACK_PERIOD],
            multiplier=hyperparameters[DecisionTreeTradeAlgorithmHyperparam.SUPER_TREND_HYPERPARAMS][
                SuperTrendHyperparam.MULTIPLIER])
        self._CCI.set_params(hyperparameters[DecisionTreeTradeAlgorithmHyperparam.CCI_HYPERPARAMS]["N"])
        self._bollinger_bands.set_params(
            hyperparameters[DecisionTreeTradeAlgorithmHyperparam.BOLLINGER_BANDS_HYPERPARAMS][
                BollingerBandsHyperparam.N],
            hyperparameters[DecisionTreeTradeAlgorithmHyperparam.BOLLINGER_BANDS_HYPERPARAMS][
                BollingerBandsHyperparam.K])
        self._ma_support_levels.set_ma_periods(
            hyperparameters[DecisionTreeTradeAlgorithmHyperparam.MA_SUPPORT_LEVELS_HYPERPARAMS]["ma_periods"])
        self._ma_support_levels.set_tested_MAs_usage(use_tested_MAs=True)
        self._days_to_keep_limit = hyperparameters[DecisionTreeTradeAlgorithmHyperparam.DAYS_TO_KEEP_LIMIT]
        self._atr_period = hyperparameters[DecisionTreeTradeAlgorithmHyperparam.ATR_PERIOD]
        self._stock_name = hyperparameters["DATA_NAME"]

        self.__clear_vars()

        self._dataframe["Date"] = self.data.index
        self._dataframe = self._dataframe.set_index("Date")
        self._atr = ATR(self.data, self._atr_period)
        for indicator in self._indicators:
            self._dataframe[indicator.name] = TradeAction.NONE
            indicator.calculate(self.data)
            if type(indicator) is MASupportLevels:
                self._ma_support_levels.test_MAs_for_data()
            self._indicators_trade_points[indicator.name] = indicator.find_trade_points()

        self.__create_train_dataset()
        self._decision_tree = IndTree(self._dataframe, [indicator.name for indicator in self._indicators])

    def evaluate_new_point(self, new_point: pd.Series, date: Union[str, pd.Timestamp],
                           special_params: Optional[Dict] = None) -> TradeAction:
        indicators_results: Dict[str, TradeAction] = {}
        for indicator in self._indicators:
            indicators_results[indicator.name] = indicator.evaluate_new_point(new_point, date, special_params, False)
        self.data.loc[date] = new_point

        final_action = self._decision_tree.get_trade_action(pd.Series(data=indicators_results))
        return final_action

    def plot(self, img_dir: str, start_date: Optional[pd.Timestamp] = None,
             end_date: Optional[pd.Timestamp] = None, show_full: bool = False):
        tree_file = img_dir
        tree_file += f"/Decision tree on {self._stock_name}.txt"
        self._decision_tree.print_tree(tree_file)

        # labels_to_plot = self._dataframe[self._dataframe["label"] != TradeAction.NONE]
        # data_to_plot = self.data[:labels_to_plot.iloc[-1]["exit date"]]
        #
        # fig = go.Figure()
        #
        # fig.add_candlestick(x=data_to_plot.index,
        #                     open=data_to_plot["Open"],
        #                     close=data_to_plot["Close"],
        #                     high=data_to_plot["High"],
        #                     low=data_to_plot["Low"],
        #                     name="Price")
        #
        # for date, row in labels_to_plot.iterrows():
        #     fig.add_shape(type="rect",
        #                   x0=date, y0=row["price"],
        #                   x1=row["exit date"], y1=row["take profit"],
        #                   opacity=0.2,
        #                   fillcolor="green",
        #                   line_color="green")
        #     fig.add_shape(type="rect",
        #                   x0=date, y0=row["price"],
        #                   x1=row["exit date"], y1=row["stop loss"],
        #                   opacity=0.2,
        #                   fillcolor="red",
        #                   line_color="red")
        #
        # buy_points = labels_to_plot["label"].isin([TradeAction.BUY, TradeAction.ACTIVELY_BUY])
        # fig.add_trace(go.Scatter(x=labels_to_plot.index,
        #                          y=labels_to_plot["price"],
        #                          mode="markers",
        #                          marker=dict(
        #                              color=np.where(buy_points, "green", "red"),
        #                              size=np.where(labels_to_plot["label"].isin([TradeAction.BUY, TradeAction.SELL]),
        #                                            8, 14),
        #                              symbol=np.where(buy_points, "triangle-up", "triangle-down")),
        #                          name="marked action labels"))
        #
        # fig.update_layout(
        #     title="Decision tree dataset's evaluated labels",
        #     xaxis_title="Date",
        #     yaxis_title="Price")
        #
        # fig.show()
