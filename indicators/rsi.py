# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 13:13:05 2021

@author: aspod
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Union

import indicators.moving_averages as ma
from indicators.abstract_indicator import AbstractIndicator, TradeAction, TradePointColumn

import cufflinks as cf
from plotly.subplots import make_subplots
import plotly.graph_objects as go

cf.go_offline()


# RSI - Relative strength index
class RSI(AbstractIndicator):
    """
    Class for calculation of Relative Strength Index

    The relative strength index (RSI) is a momentum indicator
    that measures the magnitude of recent price changes
    to evaluate overbought or oversold conditions in the price of a stock or other asset.

    Methods: set_N, set_data, calculate, print_trade_points, plot
    Attributes: RSI_val, trade_points
    """

    name = "RSI"

    def __init__(self, data: Optional[pd.DataFrame] = None,
                 N: Optional[int] = 14):
        """
        :param N: Parameter that determines number of points in MA, used for calculating index.
         Must be > 0. By default, equals 14
        :param data: time series for computing RSI. Could be not specified with instantiating,
         but must be set before calculating
        """
        super(RSI, self).__init__(data)
        self._N: int = 0
        self.RSI_val: Optional[pd.Series] = None
        self._last_average_U: float = 0
        self._last_average_D: float = 0
        self._prev_RSI: Optional[float] = None
        self.set_N(N)

    def set_N(self, N: int):
        """
        :param N: Parameter that determines number of points in MA, used for calculating index. Must be > 0.
        """
        if N <= 0:
            raise ValueError("N parameter must be > 0 and less then length of the time_series")
        self._N = N
        self.clear_vars()
        return self

    def clear_vars(self):
        super().clear_vars()
        self.RSI_val: Optional[pd.Series] = None
        self._last_average_U = 0
        self._last_average_D = 0
        self._prev_RSI = None

    def evaluate_new_point(self, new_point: pd.Series, date: Union[str, pd.Timestamp], special_params: Optional = None,
                           update_data: bool = True) -> TradeAction:
        """
        Calculates RSI for provided date point

        Formula: RSI = 100 - 100 / (1 + (average gain / average loss) )
        """
        date = pd.Timestamp(ts_input=date)
        prev_Close = self.data.iloc[-1]["Close"]
        if new_point["Close"] > prev_Close:
            val_U = new_point["Close"] - prev_Close
            val_D = 0
        else:
            val_D = prev_Close - new_point["Close"]
            val_U = 0
        self._last_average_U = ma.SMMA_one_point(prev_smma=self._last_average_U, new_point=val_U, N=self._N)
        self._last_average_D = ma.SMMA_one_point(prev_smma=self._last_average_D, new_point=val_D, N=self._N)
        RS = np.divide(self._last_average_U, self._last_average_D)
        RSI = 100 - 100 / (1 + RS)
        if update_data:
            self.data.loc[date] = new_point
        self.RSI_val = pd.concat([self.RSI_val, pd.Series({date: RSI})])
        return self.__make_trade_decision(new_point, date, RSI)

    def __make_trade_decision(self, new_point, date, RSI) -> TradeAction:
        """
                Trade strategy explanation:
                Key index values are 70 and 30.
                When index is >= 70 we consider that stock is overbought and for <= 30 is oversold.
                We describe strategy for sell/open short case, buy/open long is symmetrically opposite.

                1. While RSI is in (70,30) boundaries we consider there is no signal to sell or buy.
                2. Once RSI hits boundary we start track it's dynamic.
                3. If RSI achieves level of 80, it is a strong sign that stock is overbought and soon will go down - actively sell.
                4. In other cases:
                    - if we see a rapid index change (>= 5 in one day) - sell.
                    - if we see RSI goes down and gets close to the boundary - sell.
                    - if current RSI is less than 70 but previous day it was over it - sell.
                """
        if (RSI < 70) and (RSI > 30):
            if self._prev_RSI is None:
                trade_action = TradeAction.NONE
            else:
                if (self._prev_RSI >= 70) and (RSI >= 67.5):
                    trade_action = TradeAction.SELL
                elif (self._prev_RSI <= 30) and (RSI <= 32.5):
                    trade_action = TradeAction.BUY
                else:
                    trade_action = TradeAction.NONE
                self._prev_RSI = None
        elif RSI > 80:
            trade_action = TradeAction.ACTIVELY_SELL
            self._prev_RSI = None
        elif RSI < 20:
            trade_action = TradeAction.ACTIVELY_BUY
            self._prev_RSI = None
        elif RSI >= 70:
            if self._prev_RSI is None:
                self._prev_RSI = RSI
                trade_action = TradeAction.NONE
            elif (abs(RSI - self._prev_RSI) >= 5) or ((RSI < self._prev_RSI) and (RSI < 70.5)):
                trade_action = TradeAction.SELL
                self._prev_RSI = None
            else:
                self._prev_RSI = RSI
                trade_action = TradeAction.NONE
        elif RSI <= 30:
            if self._prev_RSI is None:
                self._prev_RSI = RSI
                trade_action = TradeAction.NONE
            elif (abs(RSI - self._prev_RSI) >= 5) or ((RSI > self._prev_RSI) and (RSI > 29.5)):
                trade_action = TradeAction.BUY
                self._prev_RSI = None
            else:
                self._prev_RSI = RSI
                trade_action = TradeAction.NONE
        else:
            self._prev_RSI = RSI
            trade_action = TradeAction.NONE
        self.add_trade_point(date, new_point["Close"], trade_action)
        return trade_action

    def calculate(self, data: Optional[pd.DataFrame] = None):
        """
        Calculates RSI for provided data

        Formula: RSI = 100 - 100 / (1 + (average gain / average loss) )

        :param data: Default None. User has to provide data before or inside calculate method.
        """
        super().calculate(data)
        # calculating U and D
        data_len = self.data.shape[0]
        days_U_D = {'U': np.zeros((data_len - 1), dtype=float), 'D': np.zeros((data_len - 1), dtype=float)}
        Closes = self.data["Close"]
        for i in range(1, data_len):
            if Closes[i] > Closes[i - 1]:
                days_U_D['U'][i - 1] = Closes[i] - Closes[i - 1]
            else:
                days_U_D['D'][i - 1] = Closes[i - 1] - Closes[i]

        average_U = ma.SMMA(days_U_D['U'], self._N)
        average_D = ma.SMMA(days_U_D['D'], self._N)
        self._last_average_U = average_U[-1]
        self._last_average_D = average_D[-1]
        # calculating RSI
        RS = np.divide(average_U, average_D)
        RSI = 100 - 100 / (1 + RS)
        self.RSI_val = pd.Series(data=RSI, index=self.data.index[self._N:])

        return self

    def find_trade_points(self) -> pd.DataFrame:
        """
        Finds trade points in provided data using previously calculated indicator values
        """
        if self.RSI_val is None:
            raise SyntaxError("find_trade_points can be called only after calculate method was called at least once")
        self._prev_RSI = None
        for i in range(len(self.RSI_val)):
            date = self.RSI_val.index[i]
            RSI = self.RSI_val[i]
            point = self.data.loc[date]
            self.__make_trade_decision(point, date, RSI)
        return self.trade_points

    def plot(self, img_dir: str, name: str, start_date: Optional[pd.Timestamp] = None,
             end_date: Optional[pd.Timestamp] = None):
        """
        Plots the RSI graphic in specified time diapason
        """
        if (start_date is None) or (start_date < self.RSI_val.index[0]):
            start_date = self.RSI_val.index[0]
        if (end_date is None) or (end_date > self.RSI_val.index[-1]):
            end_date = self.RSI_val.index[-1]

        selected_data = self.data[start_date:end_date]
        selected_rsi = self.RSI_val[start_date:end_date]
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

        fig.add_trace(go.Scatter(x=selected_rsi.index, y=selected_rsi, mode='lines',
                                 line=dict(width=1, color="blue"), name="RSI"),
                      row=2, col=1)
        fig.add_trace(go.Scatter(x=selected_rsi.index, y=np.full(len(selected_rsi), 70), mode='lines',
                                 line=dict(width=1, dash='dash', color="black"), showlegend=False),
                      row=2, col=1)
        fig.add_trace(go.Scatter(x=selected_rsi.index, y=np.full(len(selected_rsi), 30), mode='lines',
                                 line=dict(width=1, dash='dash', color="black"), showlegend=False),
                      row=2, col=1)

        fig.update_layout(title=f"{name} with RSI {self._N}",
                          xaxis_title="Date")

        # fig.show()
        fig.write_image(f"{img_dir}/{name}.png", scale=1, width=1400, height=900)
