from typing import Optional, Union

import pandas as pd
import numpy as np

from helping.base_enum import BaseEnum
from indicators.abstract_indicator import AbstractIndicator, TradeAction, TradePointColumn
from indicators.moving_averages import SMA

import cufflinks as cf
import plotly.graph_objects as go

cf.go_offline()


class BollingerBandsHyperparam(BaseEnum):
    N = 1,
    K = 2


class BollingerBands(AbstractIndicator):

    name = "BollingerBands"

    def __init__(self, data: Optional[pd.DataFrame] = None,
                 N: int = 20,
                 K: Union[float, int] = 2):
        super().__init__(data)
        self._N = N
        self._K = K
        self.bollinger_bands_value: Optional[pd.DataFrame] = None

    def set_params(self, N: int = 20, K: Union[float, int] = 2):
        """
        :param N: days period for calculation of simple moving average
        :param K: multiplier on standard deviation of SMA for calculation of bands
        """
        self.clear_vars()
        self._N = N
        self._K = K
        return self

    def clear_vars(self):
        super().clear_vars()
        self.bollinger_bands_value: Optional[pd.DataFrame] = None

    def evaluate_new_point(self, new_point: pd.Series, date: Union[str, pd.Timestamp], special_params: Optional = None,
                           update_data: bool = True) -> TradeAction:
        last_N_sample = pd.concat([self.data["Close"][-(self._N - 1):],
                                   pd.Series(data=[new_point["Close"]], index=[date])])
        ma = SMA(last_N_sample, self._N)[0]
        k_sigma = self._K * last_N_sample.rolling(self._N).std()[self._N - 1]
        upper_band = ma + k_sigma
        lower_band = ma - k_sigma

        dict_for_df = {"Simple moving average": ma, "Upper band": upper_band, "Lower band": lower_band}
        self.bollinger_bands_value.loc[date] = dict_for_df
        if update_data:
            self.data.loc[date] = new_point
        return self.__make_trade_decision(new_point, date, upper_band, lower_band)

    def __make_trade_decision(self, new_point, date, upper_band, lower_band) -> TradeAction:
        if (new_point["Open"] > upper_band) and (new_point["Close"] > upper_band):
            trade_action = TradeAction.ACTIVELY_SELL
        elif (new_point["Open"] < lower_band) and (new_point["Close"] < lower_band):
            trade_action = TradeAction.ACTIVELY_BUY
        elif (new_point["Open"] >= upper_band * 0.9985) and (new_point["Close"] > upper_band):
            trade_action = TradeAction.SELL
        elif (new_point["Open"] <= lower_band * 1.0015) and (new_point["Close"] < lower_band):
            trade_action = TradeAction.BUY
        else:
            trade_action = TradeAction.NONE

        self.add_trade_point(date, new_point["Close"], trade_action)
        return trade_action

    def calculate(self, data: Optional[pd.DataFrame] = None):
        super().calculate(data)
        ma = SMA(self.data["Close"], self._N)
        k_sigma = self._K * self.data["Close"].rolling(self._N).std()[(self._N - 1):]
        upper_band = ma + k_sigma
        lower_band = ma - k_sigma

        dict_for_df = {"Simple moving average": ma, "Upper band": upper_band, "Lower band": lower_band}
        self.bollinger_bands_value = pd.DataFrame(data=dict_for_df, index=self.data.index[(self._N - 1):])

        return self

    def find_trade_points(self) -> pd.DataFrame:
        if self.bollinger_bands_value is None:
            raise SyntaxError("find_trade_points can be called only after calculate method was called at least once")
        for i in range(self.bollinger_bands_value.shape[0]):
            date = self.bollinger_bands_value.index[i]
            point = self.data.loc[date]
            upper_band, lower_band = self.bollinger_bands_value.iloc[i][["Upper band", "Lower band"]]
            self.__make_trade_decision(point, date, upper_band, lower_band)
        return self.trade_points

    def plot(self, start_date: Optional[pd.Timestamp] = None, end_date: Optional[pd.Timestamp] = None):
        """
        Plots the candle graph for data and Bollinger bands with highlighted trade points in specified time diapason
        """
        if (start_date is None) or (start_date < self.bollinger_bands_value.index[0]):
            start_date = self.bollinger_bands_value.index[0]
        if (end_date is None) or (end_date > self.data.index[-1]):
            end_date = self.data.index[-1]

        selected_data = self.data[start_date:end_date]
        selected_bollinger_bands = self.bollinger_bands_value[start_date:end_date]

        fig = go.Figure()
        fig.add_candlestick(x=selected_data.index,
                            open=selected_data["Open"],
                            close=selected_data["Close"],
                            high=selected_data["High"],
                            low=selected_data["Low"],
                            name="Price")

        fig.add_trace(go.Scatter(x=selected_bollinger_bands.index, y=selected_bollinger_bands["Lower band"],
                                 mode='lines', line=dict(width=1, color="orange"), name="Lower band"))

        fig.add_trace(go.Scatter(x=selected_bollinger_bands.index, y=selected_bollinger_bands["Upper band"],
                                 mode='lines', line=dict(width=1, color="blue"), name="Upper band", fill='tonexty'))

        selected_trade_points = self.select_action_trade_points(start_date=start_date, end_date=end_date)
        bool_buys = selected_trade_points[TradePointColumn.ACTION].isin(self.buy_actions)
        bool_actives = selected_trade_points[TradePointColumn.ACTION].isin(self.active_actions)
        fig.add_trace(go.Scatter(x=selected_trade_points.index,
                                 y=selected_trade_points[TradePointColumn.PRICE],
                                 mode="markers",
                                 marker=dict(
                                     color=np.where(bool_buys, "green", "red"),
                                     size=np.where(bool_actives, 15, 10),
                                     symbol=np.where(bool_buys, "triangle-up", "triangle-down")),
                                 name="Action points"))

        fig.update_layout(title=f"Price with Bollinger bands {self._N},{self._K}",
                          xaxis_title="Date",
                          yaxis_title="Price")

        fig.show()
