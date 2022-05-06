import pandas as pd
import numpy as np
from typing import Optional, Union, Dict, List, Hashable

from helping.base_enum import BaseEnum
from indicators.abstract_indicator import TradePointColumn
from trading.trade_algorithms.abstract_trade_algorithm import AbstractTradeAlgorithm, TradeAction
from indicators.super_trend import SuperTrend, SuperTrendHyperparam
from indicators.macd import MACD, MACDHyperparam
from indicators.moving_averages import EMA, EMA_one_point

import cufflinks as cf
from plotly.subplots import make_subplots
import plotly.graph_objects as go

cf.go_offline()


class MACDSuperTrendTradeAlgorithmHyperparam(BaseEnum):
    STRICT_MACD = 1,
    DAYS_TO_WAIT_FOR_ST = 2,
    MACD_HYPERPARAMS = 3,
    SUPER_TREND_HYPERPARAMS = 4


class MACDSuperTrendTradeAlgorithm(AbstractTradeAlgorithm):
    name = "MACD+SuperTrend trade algorithm"

    def __init__(self):
        super().__init__()
        self._super_trend: SuperTrend = SuperTrend()
        self._MACD: MACD = MACD()
        self._strict_macd: bool = False
        self._days_to_wait_for_st: int = 0
        self._days_from_macd_crossing: int = 0
        self._macd_crossing_flag: bool = False
        self._macd_saved_action: TradeAction = TradeAction.NONE
        self.trade_points: Optional[pd.DataFrame] = None
        # self._EMA200: Optional[pd.Series] = None

    @staticmethod
    def get_algorithm_name() -> str:
        return MACDSuperTrendTradeAlgorithm.name

    def __clear_vars(self):
        self.trade_points = pd.DataFrame(columns=TradePointColumn.get_elements_list()).set_index(TradePointColumn.DATE)
        self._MACD.clear_vars()
        self._super_trend.clear_vars()

    @staticmethod
    def create_hyperparameters_dict(macd_short_period: int = 12, macd_long_period: int = 26, macd_signal_period: int = 9,
                                    super_trend_lookback_period: int = 10, super_trend_multiplier: float = 3,
                                    strict_macd: bool = False, days_to_wait_for_st: int = 3) -> Dict:
        return {MACDSuperTrendTradeAlgorithmHyperparam.STRICT_MACD: strict_macd,
                MACDSuperTrendTradeAlgorithmHyperparam.DAYS_TO_WAIT_FOR_ST: days_to_wait_for_st,
                MACDSuperTrendTradeAlgorithmHyperparam.MACD_HYPERPARAMS: {
                    MACDHyperparam.SHORT_PERIOD: macd_short_period,
                    MACDHyperparam.LONG_PERIOD: macd_long_period,
                    MACDHyperparam.SIGNAL_PERIOD: macd_signal_period
                },
                MACDSuperTrendTradeAlgorithmHyperparam.SUPER_TREND_HYPERPARAMS: {
                    SuperTrendHyperparam.LOOKBACK_PERIOD: super_trend_lookback_period,
                    SuperTrendHyperparam.MULTIPLIER: super_trend_multiplier
                }}

    @staticmethod
    def get_default_hyperparameters_grid() -> List[Dict]:
        return [MACDSuperTrendTradeAlgorithm.create_hyperparameters_dict(),
                MACDSuperTrendTradeAlgorithm.create_hyperparameters_dict(super_trend_multiplier=2),
                MACDSuperTrendTradeAlgorithm.create_hyperparameters_dict(days_to_wait_for_st=0),
                MACDSuperTrendTradeAlgorithm.create_hyperparameters_dict(days_to_wait_for_st=5),
                MACDSuperTrendTradeAlgorithm.create_hyperparameters_dict(strict_macd=True),
                MACDSuperTrendTradeAlgorithm.create_hyperparameters_dict(super_trend_multiplier=2, strict_macd=True),
                MACDSuperTrendTradeAlgorithm.create_hyperparameters_dict(macd_short_period=10, macd_long_period=20,
                                                                         super_trend_multiplier=2)]

    def train(self, data: pd.DataFrame, hyperparameters: Dict):
        super().train(data, hyperparameters)
        self._strict_macd = hyperparameters[MACDSuperTrendTradeAlgorithmHyperparam.STRICT_MACD]
        self._days_to_wait_for_st = hyperparameters[MACDSuperTrendTradeAlgorithmHyperparam.DAYS_TO_WAIT_FOR_ST]

        self._MACD.set_ma_periods(
            hyperparameters[MACDSuperTrendTradeAlgorithmHyperparam.MACD_HYPERPARAMS][MACDHyperparam.SHORT_PERIOD],
            hyperparameters[MACDSuperTrendTradeAlgorithmHyperparam.MACD_HYPERPARAMS][MACDHyperparam.LONG_PERIOD],
            hyperparameters[MACDSuperTrendTradeAlgorithmHyperparam.MACD_HYPERPARAMS][MACDHyperparam.SIGNAL_PERIOD])
        self._super_trend.set_params(
            lookback_period=hyperparameters[MACDSuperTrendTradeAlgorithmHyperparam.SUPER_TREND_HYPERPARAMS][
                SuperTrendHyperparam.LOOKBACK_PERIOD],
            multiplier=hyperparameters[MACDSuperTrendTradeAlgorithmHyperparam.SUPER_TREND_HYPERPARAMS][
                SuperTrendHyperparam.MULTIPLIER])

        self.__clear_vars()
        self._MACD.calculate(self.data)
        self._super_trend.calculate(self.data)
        # self._EMA200 = EMA(self.data["Close"], 200)

    def evaluate_new_point(self, new_point: pd.Series,
                           date: Union[str, pd.Timestamp],
                           special_params: Optional[Dict] = None) -> TradeAction:
        macd_action = self._MACD.evaluate_new_point(new_point, date, special_params, False)
        super_trend_action = self._super_trend.evaluate_new_point(new_point, date, special_params, False)
        # self._EMA200[date] = EMA_one_point(self._EMA200[-1], new_point["Close"], 200)
        super_trend_color = self._super_trend.super_trend_value.iloc[-1]["Color"]
        if (not self._strict_macd and (macd_action != TradeAction.NONE)) or (self._strict_macd and (
                (macd_action == TradeAction.ACTIVELY_BUY) or (macd_action == TradeAction.ACTIVELY_SELL))):
            if ((super_trend_color == "green") and (
                    (macd_action == TradeAction.ACTIVELY_BUY) or (macd_action == TradeAction.BUY))) or (
                    (super_trend_color == "red") and (
                    (macd_action == TradeAction.ACTIVELY_SELL) or (macd_action == TradeAction.SELL))):
                self._macd_crossing_flag = False
                self._days_from_macd_crossing = 0
                final_action = macd_action
            else:
                self._macd_crossing_flag = True
                self._macd_saved_action = macd_action
                final_action = TradeAction.NONE
        elif self._macd_crossing_flag:
            if super_trend_action != TradeAction.NONE:
                self._macd_crossing_flag = False
                self._days_from_macd_crossing = 0
                final_action = self._macd_saved_action
            else:
                final_action = TradeAction.NONE
                self._days_from_macd_crossing += 1
                if self._days_from_macd_crossing > self._days_to_wait_for_st:
                    self._macd_crossing_flag = False
                    self._days_from_macd_crossing = 0
        else:
            final_action = TradeAction.NONE
        self.data.loc[date] = new_point
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
        selected_macd = self._MACD.MACD_val[start_date:end_date]
        selected_super_trend = self._super_trend.super_trend_value[start_date:end_date]

        fig = make_subplots(rows=2, cols=1,
                            shared_xaxes=True,
                            subplot_titles=[f"Price with SuperTrend {self._super_trend._lookback_period}, {self._super_trend._multiplier}",
                                            f"MACD {self._MACD._short_period}, {self._MACD._long_period}, {self._MACD._signal_period}"],
                            vertical_spacing=0.2)

        fig.add_candlestick(x=selected_data.index,
                            open=selected_data["Open"],
                            close=selected_data["Close"],
                            high=selected_data["High"],
                            low=selected_data["Low"],
                            name="Price",
                            row=1, col=1)

        sub_lines = []
        line = {"dates": [selected_super_trend.index[0]],
                "values": [selected_super_trend["Value"][0]],
                "color": selected_super_trend["Color"][0]}
        for date, row in selected_super_trend[1:].iterrows():
            if line["color"] == row["Color"]:
                line["dates"].append(date)
                line["values"].append(row["Value"])
            else:
                sub_lines.append(line)
                line = {"dates": [date],
                        "values": [row["Value"]],
                        "color": row["Color"]}
        sub_lines.append(line)

        for line in sub_lines:
            fig.add_trace(go.Scatter(
                    x=line["dates"],
                    y=line["values"],
                    mode='lines',
                    line_color=line["color"],
                    showlegend=False,
                ), row=1, col=1)

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

        fig.add_trace(go.Scatter(x=selected_macd.index, y=selected_macd["MACD"], mode='lines',
                                 line=dict(width=1, color="blue"), name="MACD"),
                      row=2, col=1)
        fig.add_trace(go.Scatter(x=selected_macd.index, y=selected_macd["signal"], mode='lines',
                                 line=dict(width=1, color="orange"), name="signal line"),
                      row=2, col=1)

        fig.add_bar(x=selected_macd.index, y=selected_macd["histogram"], marker=dict(
            color=np.where(selected_macd["histogram"] > 0, "green", "red")
        ), name="MACD signal difference", row=2, col=1)

        fig.show()
