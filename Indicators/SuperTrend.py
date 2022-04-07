import Indicators.moving_averages as ma
import numpy as np
import pandas as pd
from typing import Optional, List, Tuple, Dict, Union

from Indicators.ATR import ATR, ATR_one_point
from Indicators.AbstractIndicator import AbstractIndicator, TradeAction

import cufflinks as cf
import plotly.graph_objects as go

cf.go_offline()


class SuperTrend(AbstractIndicator):

    def __init__(self, data: Optional[pd.DataFrame] = None,
                 lookback_period: Optional[int] = 10,
                 multiplier: Optional[int] = 3):
        """
        :param lookback_period: lookback period is the number of data points to take into account for the calculation
        :param multiplier: _multiplier is the value used to multiply the ATR
        """
        super().__init__(data)
        self._multiplier: int = multiplier
        self._lookback_period: int = lookback_period
        self.super_trend_value: Optional[pd.DataFrame] = None
        self._prev_atr: float = 0
        self._prev_fub: float = 0
        self._prev_flb: float = 0
        self._prev_color: Optional[str] = None

    def set_params(self, lookback_period: Optional[int] = 10, multiplier: Optional[int] = 3):
        """
        :param lookback_period: lookback period is the number of data points to take into account for the calculation
        :param multiplier: multiplier is the value used to multiply the ATR
        """
        self.clear_vars()
        self._multiplier = multiplier
        self._lookback_period = lookback_period

    def clear_vars(self):
        super().clear_vars()
        self.super_trend_value: Optional[pd.DataFrame] = None
        self._prev_atr: float = 0
        self._prev_fub: float = 0
        self._prev_flb: float = 0
        self._prev_color: Optional[str] = None

    def evaluate_new_point(self, new_point: pd.Series, date: Union[str, pd.Timestamp], special_params: Optional = None):
        date = pd.Timestamp(ts_input=date)
        prev_close = self.data["Close"][-1]
        atr = ATR_one_point(self._prev_atr, prev_close, new_point, self._lookback_period)
        hla = (new_point["High"] + new_point["Low"]) / 2
        bub = hla + self._multiplier * atr
        blb = hla - self._multiplier * atr

        if (bub < self._prev_fub) or (prev_close > self._prev_fub):
            fub = bub
        else:
            fub = self._prev_fub

        if (blb > self._prev_flb) or (prev_close < self._prev_flb):
            flb = blb
        else:
            flb = self._prev_flb

        prev_st = self.super_trend_value["Value"][-1]
        cur_st = 0
        if (prev_st == self._prev_fub) and (new_point["Close"] <= fub):
            cur_st = fub
        elif (prev_st == self._prev_fub) and (new_point["Close"] >= fub):
            cur_st = flb
        elif (prev_st == self._prev_flb) and (new_point["Close"] >= flb):
            cur_st = flb
        elif (prev_st == self._prev_flb) and (new_point["Close"] <= flb):
            cur_st = fub

        if cur_st >= new_point["Close"]:
            color = "red"
        else:
            color = "green"

        self.__make_trade_decision(new_point, date, color)
        self.data.loc[date] = new_point
        self.super_trend_value.loc[date] = {"Value": cur_st, "Color": color}

    def __make_trade_decision(self, new_point, date, super_trend_color):
        if self._prev_color is None:
            self._prev_color = super_trend_color
            self.add_trade_point(date, new_point["Close"], TradeAction.NONE)
            return
        if super_trend_color != self._prev_color:
            self._prev_color = super_trend_color
            if super_trend_color == "green":
                self.add_trade_point(date, new_point["Close"], TradeAction.BUY)
            else:
                self.add_trade_point(date, new_point["Close"], TradeAction.SELL)
        else:
            self.add_trade_point(date, new_point["Close"], TradeAction.NONE)

    def calculate(self, data: Optional[pd.DataFrame] = None):
        super().calculate(data)
        atr = ATR(self.data, self._lookback_period)
        # high low average
        hla = (self.data["High"][self._lookback_period:] + self.data["Low"][self._lookback_period:]) / 2
        # basic upper band
        bub = hla + self._multiplier * atr
        # basic lower band
        blb = hla - self._multiplier * atr
        # calculating final bands
        close = self.data["Close"][self._lookback_period:]
        arr_len = len(close)
        # final upper band
        fub = np.zeros(arr_len)
        fub[0] = bub[0]
        for i in range(1, arr_len):
            if (bub[i] < fub[i - 1]) or (close[i - 1] > fub[i - 1]):
                fub[i] = bub[i]
            else:
                fub[i] = fub[i - 1]
        # final lower band
        flb = np.zeros(arr_len)
        flb[0] = blb[0]
        for i in range(1, arr_len):
            if (blb[i] > flb[i - 1]) or (close[i - 1] < flb[i - 1]):
                flb[i] = blb[i]
            else:
                flb[i] = flb[i - 1]
        # calculating SuperTrend value
        self.super_trend_value = pd.DataFrame(index=self.data.index[self._lookback_period:], columns=["Value", "Color"])
        self.super_trend_value.loc[self.super_trend_value.index[0], ["Value", "Color"]] = [fub[0], "red"]
        prev_st = fub[0]
        cur_st = 0
        for i in range(1, arr_len):
            if (prev_st == fub[i - 1]) and (close[i] <= fub[i]):
                cur_st = fub[i]
            elif (prev_st == fub[i - 1]) and (close[i] >= fub[i]):
                cur_st = flb[i]
            elif (prev_st == flb[i - 1]) and (close[i] >= flb[i]):
                cur_st = flb[i]
            elif (prev_st == flb[i - 1]) and (close[i] <= flb[i]):
                cur_st = fub[i]

            if cur_st >= close[i]:
                self.super_trend_value.loc[self.super_trend_value.index[i], ["Value", "Color"]] = [cur_st, "red"]
            else:
                self.super_trend_value.loc[self.super_trend_value.index[i], ["Value", "Color"]] = [cur_st, "green"]
            prev_st = cur_st

        self._prev_atr = atr[-1]
        self._prev_fub = fub[-1]
        self._prev_flb = flb[-1]
        return self

    def find_trade_points(self):
        if self.super_trend_value is None:
            raise SyntaxError("find_trade_points can be called only after calculate method was called at least once")
        for i in range(len(self.super_trend_value)):
            date = self.super_trend_value.index[i]
            color = self.super_trend_value["Color"][i]
            point = self.data.loc[date]
            self.__make_trade_decision(point, date, color)
        return self.trade_points

    def plot(self, start_date: Optional[pd.Timestamp] = None, end_date: Optional[pd.Timestamp] = None):
        """
        Plots the candle graph for data and Super Trend line with highlighted trade points in specified time diapason
        """
        if (start_date is None) or (start_date < self.super_trend_value.index[0]):
            start_date = self.super_trend_value.index[0]
        if (end_date is None) or (end_date > self.data.index[-1]):
            end_date = self.data.index[-1]

        selected_data = self.data[start_date:end_date]
        selected_super_trend = self.super_trend_value[start_date:end_date]

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

        fig = go.Figure(
            [
                go.Scatter(
                    x=line["dates"],
                    y=line["values"],
                    mode='lines',
                    line_color=line["color"],
                    showlegend=False,
                )
                for line in sub_lines
            ]
        )
        fig.add_candlestick(x=selected_data.index,
                            open=selected_data["Open"],
                            close=selected_data["Close"],
                            high=selected_data["High"],
                            low=selected_data["Low"],
                            name="Price")

        selected_trade_points = self.select_action_trade_points(start_date=start_date, end_date=end_date)

        fig.add_trace(go.Scatter(x=selected_trade_points.index,
                                 y=selected_trade_points["Price"],
                                 mode="markers",
                                 marker=dict(color=np.where(selected_trade_points["Action"] == TradeAction.BUY, "green", "red"),
                                             size=7,
                                             symbol=np.where(selected_trade_points["Action"] == TradeAction.BUY, "triangle-up", "triangle-down")),
                                 name="Action points"))

        fig.update_layout(title=f"Price with SuperTrend {self._lookback_period},{self._multiplier}",
                          xaxis_title="Date",
                          yaxis_title="Price")

        fig.show()
