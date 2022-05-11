from typing import Optional, Union

import pandas as pd
import numpy as np

from helping.base_enum import BaseEnum
from indicators.abstract_indicator import AbstractIndicator, TradeAction, TradePointColumn
from indicators.moving_averages import SMA, RMAD

import cufflinks as cf
from plotly.subplots import make_subplots
import plotly.graph_objects as go


class CCI(AbstractIndicator):

    name = "CCI"

    def __init__(self, data: Optional[pd.DataFrame] = None,
                 N: int = 20):
        super().__init__(data)
        self._N = N
        self._scale_const = 0.015
        self._hit_low: bool = False
        self._hit_high: bool = False
        self.CCI_value: Optional[pd.Series] = None

    def set_params(self, N: int = 20):
        """
        :param N: days period for calculation of simple moving average
        """
        self.clear_vars()
        self._N = N
        return self

    def clear_vars(self):
        super().clear_vars()
        self.CCI_value: Optional[pd.Series] = None

    def evaluate_new_point(self, new_point: pd.Series, date: Union[str, pd.Timestamp], special_params: Optional = None,
                           update_data: bool = True) -> TradeAction:
        last_N_sample = self.data[-(self._N - 1):]
        last_N_sample.loc[date] = new_point
        typical_price: pd.Series = (last_N_sample["Low"] + last_N_sample["High"] + last_N_sample["Close"]) / 3
        ma = SMA(typical_price, self._N)
        mad = self._scale_const * RMAD(typical_price, self._N)
        cci = (typical_price[-1] - ma[0]) / mad[0]
        if update_data:
            self.data.loc[date] = new_point
        self.CCI_value = pd.concat([self.CCI_value, pd.Series({date: cci})])
        return self.__make_trade_decision(new_point, date, cci)

    def __make_trade_decision(self, new_point, date, cci) -> TradeAction:
        if cci <= -200:
            trade_action = TradeAction.ACTIVELY_BUY
            self._hit_low = False
        elif cci >= 200:
            trade_action = TradeAction.ACTIVELY_SELL
            self._hit_high = False
        elif self._hit_low and (cci >= -100):
            trade_action = TradeAction.BUY
            self._hit_low = False
        elif self._hit_high and (cci <= 100):
            trade_action = TradeAction.SELL
            self._hit_high = False
        else:
            trade_action = TradeAction.NONE
            if cci > 100:
                self._hit_high = True
            elif cci < -100:
                self._hit_low = True

        self.add_trade_point(date, new_point["Close"], trade_action)
        return trade_action

    def calculate(self, data: Optional[pd.DataFrame] = None):
        super().calculate(data)
        typical_price: pd.Series = (self.data["Low"] + self.data["High"] + self.data["Close"]) / 3
        ma = SMA(typical_price, self._N)
        mad = self._scale_const * RMAD(typical_price, self._N)  # mean absolute deviation
        self.CCI_value = (typical_price[(self._N - 1):] - ma) / mad

        return self

    def find_trade_points(self) -> pd.DataFrame:
        if self.CCI_value is None:
            raise SyntaxError("find_trade_points can be called only after calculate method was called at least once")
        for i in range(len(self.CCI_value)):
            date = self.CCI_value.index[i]
            cci = self.CCI_value[i]
            point = self.data.loc[date]
            self.__make_trade_decision(point, date, cci)
        return self.trade_points

    def plot(self, start_date: Optional[pd.Timestamp] = None, end_date: Optional[pd.Timestamp] = None):
        if (start_date is None) or (start_date < self.CCI_value.index[0]):
            start_date = self.CCI_value.index[0]
        if (end_date is None) or (end_date > self.CCI_value.index[-1]):
            end_date = self.CCI_value.index[-1]

        selected_data = self.data[start_date:end_date]
        selected_cci = self.CCI_value[start_date:end_date]
        selected_trade_points = self.select_action_trade_points(start_date=start_date, end_date=end_date)

        fig = make_subplots(rows=2, cols=1,
                            shared_xaxes=True,
                            vertical_spacing=0.2)

        fig.add_candlestick(x=selected_data.index,
                            open=selected_data["Open"],
                            close=selected_data["Close"],
                            high=selected_data["High"],
                            low=selected_data["Low"],
                            name="Price",
                            row=1, col=1)

        bool_buys = selected_trade_points[TradePointColumn.ACTION].isin(self.buy_actions)
        bool_actives = selected_trade_points[TradePointColumn.ACTION].isin(self.active_actions)
        fig.add_trace(go.Scatter(x=selected_trade_points.index,
                                 y=selected_trade_points[TradePointColumn.PRICE],
                                 mode="markers",
                                 marker=dict(
                                     color=np.where(bool_buys, "green", "red"),
                                     size=np.where(bool_actives, 15, 10),
                                     symbol=np.where(bool_buys, "triangle-up", "triangle-down")),
                                 name="Action points"),
                      row=1, col=1)

        fig.add_trace(go.Scatter(x=selected_cci.index, y=selected_cci, mode='lines',
                                 line=dict(width=1, color="blue"), name="RSI"),
                      row=2, col=1)
        fig.add_trace(go.Scatter(x=selected_cci.index, y=np.full(len(selected_cci), 100), mode='lines',
                                 line=dict(width=1, dash='dash', color="black"), showlegend=False),
                      row=2, col=1)
        fig.add_trace(go.Scatter(x=selected_cci.index, y=np.full(len(selected_cci), -100), mode='lines',
                                 line=dict(width=1, dash='dash', color="black"), showlegend=False),
                      row=2, col=1)

        fig.update_layout(title=f"Price with CCI {self._N}",
                          xaxis_title="Date")

        fig.show()
