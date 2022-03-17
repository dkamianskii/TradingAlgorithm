import indicators.moving_averages as ma
import numpy as np
import pandas as pd
from typing import Optional, List, Tuple, Dict

from indicators.ATR import ATR
from indicators.AbstractIndicator import AbstractIndicator

import cufflinks as cf
import plotly.graph_objects as go

cf.go_offline()


class SuperTrend(AbstractIndicator):

    def __init__(self, data: Optional[pd.DataFrame] = None,
                 lookback_period: Optional[int] = 10,
                 multiplier: Optional[int] = 3):
        """
        :param lookback_period: lookback period is the number of data points to take into account for the calculation
        :param multiplier: multiplier is the value used to multiply the ATR
        """
        super().__init__(data)
        self.multiplier: int = multiplier
        self.lookback_period: int = lookback_period
        self.super_trend_value: Optional[pd.DataFrame] = None

    def set_params(self, lookback_period: Optional[int] = 10, multiplier: Optional[int] = 3):
        """
        :param lookback_period: lookback period is the number of data points to take into account for the calculation
        :param multiplier: multiplier is the value used to multiply the ATR
        """
        self.multiplier = multiplier
        self.lookback_period = lookback_period

    def calculate(self, data: Optional[pd.DataFrame] = None):
        super().calculate(data)
        atr = ATR(self.data, self.lookback_period)
        # high low average
        hla = (self.data["High"][self.lookback_period:] + self.data["Low"][self.lookback_period:]) / 2
        # basic upper band
        bub = hla + self.multiplier * atr
        # basic lower band
        blb = hla - self.multiplier * atr
        # calculating final bands
        close = self.data["Close"][self.lookback_period:]
        arr_len = len(close)
        # final upper band
        fub = np.zeros(arr_len)
        fub[0] = bub[0]
        for i in range(1, arr_len):
            if ((bub[i] < fub[i - 1]) or (close[i - 1] > fub[i - 1])):
                fub[i] = bub[i]
            else:
                fub[i] = fub[i - 1]
        # final lower band
        flb = np.zeros(arr_len)
        flb[0] = blb[0]
        for i in range(1, arr_len):
            if ((blb[i] > flb[i - 1]) or (close[i - 1] < flb[i - 1])):
                flb[i] = blb[i]
            else:
                flb[i] = flb[i - 1]
        # calculating SuperTrend value
        self.super_trend_value = pd.DataFrame(index=self.data.index[self.lookback_period:], columns=["Value", "Color"])
        self.super_trend_value.loc[self.super_trend_value.index[0], ["Value", "Color"]] = [fub[0], "red"]
        prev_st = fub[0]
        cur_st = 0
        for i in range(1, arr_len):
            if ((prev_st == fub[i - 1]) and (close[i] <= fub[i])):
                cur_st = fub[i]
            elif ((prev_st == fub[i - 1]) and (close[i] >= fub[i])):
                cur_st = flb[i]
            elif ((prev_st == flb[i - 1]) and (close[i] >= flb[i])):
                cur_st = flb[i]
            elif ((prev_st == flb[i - 1]) and (close[i] <= flb[i])):
                cur_st = fub[i]

            if (cur_st >= close[i]):
                self.super_trend_value.loc[self.super_trend_value.index[i], ["Value", "Color"]] = [cur_st, "red"]
            else:
                self.super_trend_value.loc[self.super_trend_value.index[i], ["Value", "Color"]] = [cur_st, "green"]
            prev_st = cur_st
        return self

    def find_trade_points(self):
        self.clear_trade_points()
        if (self.super_trend_value is None):
            raise SyntaxError("find_trade_points can be called only after calculate method was called at least once")

        prev_color = self.super_trend_value.loc[self.super_trend_value.index[0], "Color"]
        for date, row in self.super_trend_value[1:].iterrows():
            if (row["Color"] != prev_color):
                prev_color = row["Color"]
                if (row["Color"] == "green"):
                    self.add_trade_point(date, "buy")
                else:
                    self.add_trade_point(date, "sell")

    def plot(self, start_date: Optional[pd.Timestamp] = None, end_date: Optional[pd.Timestamp] = None):
        """
        Plots the candle graph for data and Super Trend line with highlighted trade points in specified time diapason
        """
        if ((start_date is None) or (start_date < self.super_trend_value.index[0])):
            start_date = self.super_trend_value.index[0]
        if ((end_date is None) or (end_date > self.data.index[-1])):
            end_date = self.data.index[-1]

        selected_data = self.data[start_date:end_date]
        selected_super_trend = self.super_trend_value[start_date:end_date]

        sub_lines = []
        line = {"dates": [selected_super_trend.index[0]],
                "values": [selected_super_trend["Value"][0]],
                "color": selected_super_trend["Color"][0]}
        for date, row in selected_super_trend[1:].iterrows():
            if (line["color"] == row["Color"]):
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
                                 y=selected_trade_points["This Day Close"],
                                 mode="markers",
                                 marker=dict(color=np.where(selected_trade_points["Action"] == "buy", "green", "red"),
                                             size=5),
                                 name="Action points"))

        fig.update_layout(title=f"Price with SuperTrend {self.lookback_period},{self.multiplier}",
                          xaxis_title="Date",
                          yaxis_title="Price")

        fig.show()


