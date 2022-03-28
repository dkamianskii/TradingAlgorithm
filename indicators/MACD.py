import numpy as np
import pandas as pd
from typing import Optional, Union

import indicators.moving_averages as ma
from indicators.AbstractIndicator import AbstractIndicator

import cufflinks as cf
from plotly.subplots import make_subplots
import plotly.graph_objects as go

cf.go_offline()


class MACD(AbstractIndicator):
    """
    Class for MACD calculation

    Moving average convergence divergence (MACD) is a trend-following momentum indicator that shows the relationship
    between two moving averages of a security’s price. The classical MACD is calculated by subtracting the 26-period
    exponential moving average (EMA) from the 12-period EMA.
    The result of that calculation is the MACD line. A nine-day EMA of the MACD called the "signal line".

    Trade strategy explanation:
        * classic
        Buy / Open long, when index line cross up through the signal, actively buy when MACD is below zero.
        Sell / Open short, when index line cross down through the signal, actively sell when MACD is above zero.

        * convergence
        On the first step detecting whether there is a convergence in MACD lines.
        We say that MACD lines converging if there are 2 sequential MACD hist downsizing.
        Like cur hist < prev hist < pre-prev hist. By abs value.
        Secondly, after convergence was established we take pre-prev hist value as a starting peak,
        and waiting for a moment when hist value will be less then 15% of the starting peak or it will be different sign.
        When it comes to the trade point by convergence and not by sign changing, it's a sign for active trade action,
        because it is playing on the expectations and not just following trend when it might be already late.
    """
    supported_trade_strategies = ["classic", "convergence"]

    def __init__(self, data: Optional[pd.DataFrame] = None,
                 short_period: Optional[int] = 12,
                 long_period: Optional[int] = 26,
                 signal_period: Optional[int] = 9,
                 trade_strategy: Optional[str] = "classic"):
        """
        :param data: time series for computing MACD. Could be not specified with instantiating,
         but must be set before calculating
        """
        super().__init__(data)
        self._short_period: int = 0
        self._long_period: int = 0
        self._signal_period: int = 0
        self.set_ma_periods(short_period, long_period, signal_period)
        self._trade_strategy: str = ""
        self.set_trade_strategy(trade_strategy)
        self.MACD_val: Optional[pd.DataFrame] = None
        self._last_short_ma: float = 0
        self._last_long_ma: float = 0
        self._last_signal_ma: float = 0
        self._prev_hist: float = 0
        self._pre_prev_hist: float = 0
        self._hist_peak: float = 0
        self._convergence_flag: bool = False

    def set_ma_periods(self, short_period: Optional[int] = 12,
                       long_period: Optional[int] = 26,
                       signal_period: Optional[int] = 9):
        if short_period == long_period:
            raise ValueError("short period and long period can't be equal")
        if short_period > long_period:
            t = long_period
            long_period = short_period
            short_period = t
        self.clear_vars()
        self._short_period = short_period
        self._long_period = long_period
        self._signal_period = signal_period

    def set_trade_strategy(self, trade_strategy: str):
        self.clear_vars()
        if trade_strategy not in MACD.supported_trade_strategies:
            raise ValueError("trade strategy must be one of supported trade strategies")
        self._trade_strategy = trade_strategy

    def clear_vars(self):
        super().clear_vars()
        self.MACD_val: Optional[pd.DataFrame] = None
        self._last_short_ma: float = 0
        self._last_long_ma: float = 0
        self._last_signal_ma: float = 0
        self._prev_hist: float = 0
        self._pre_prev_hist: float = 0
        self._hist_peak: float = 0
        self._convergence_flag: bool = False

    def evaluate_new_point(self, new_point: pd.Series, date: Union[str, pd.Timestamp], special_params: Optional = None):
        date = pd.Timestamp(ts_input=date)
        self._last_short_ma = ma.EMA_one_point(prev_ema=self._last_short_ma, new_point=new_point["Close"],
                                               N=self._short_period)
        self._last_long_ma = ma.EMA_one_point(prev_ema=self._last_long_ma, new_point=new_point["Close"],
                                              N=self._long_period)
        MACD = self._last_short_ma - self._last_long_ma
        signal = ma.EMA_one_point(prev_ema=self.MACD_val["signal"][-1], new_point=MACD, N=self._signal_period)
        histogram = MACD - self._last_signal_ma
        self.data.loc[date] = new_point
        self.MACD_val.loc[date] = {"MACD": MACD, "signal": signal, "histogram": histogram}
        self.__make_trade_decision(new_point, date, MACD, histogram)

    def __make_trade_decision(self, new_point, date, MACD, histogram):
        """
        Trade strategy explanation:
        * classic
        Buy / Open long, when index line cross up through the signal, actively buy when MACD is below zero.
        Sell / Open short, when index line cross down through the signal, actively sell when MACD is above zero.

        * convergence
        On the first step detecting whether there is a convergence in MACD lines.
        We say that MACD lines converging if there are 2 sequential MACD hist downsizing.
        Like cur hist < prev hist < pre-prev hist. By abs value.
        Secondly, after convergence was established we take pre-prev hist value as a starting peak,
        and waiting for a moment when hist value will be less then 15% of the starting peak or it will be different sign.
        When it comes to the trade point by convergence and not by sign changing, it's a sign for active trade action,
        because it is playing on the expectations and not just following trend when it might be already late.
        """
        if self._trade_strategy == "classic":
            if (self._prev_hist is not None) and ((histogram == 0) or (np.sign(self._prev_hist) != np.sign(histogram))):
                if np.sign(self._prev_hist) > 0:
                    if MACD > 0:
                        self.add_trade_point(date, new_point["Close"], "actively sell")
                    else:
                        self.add_trade_point(date, new_point["Close"], "sell")
                    return
                elif np.sign(self._prev_hist) < 0:
                    if MACD < 0:
                        self.add_trade_point(date, new_point["Close"], "actively buy")
                    else:
                        self.add_trade_point(date, new_point["Close"], "buy")
                    return
            self._prev_hist = histogram
            self.add_trade_point(date, new_point["Close"], "none")

        if self._trade_strategy == "convergence":
            if (self._prev_hist is not None) and (self._pre_prev_hist is not None):
                sign_diff_flag = np.sign(self._prev_hist) != np.sign(histogram)
                if not self._convergence_flag:
                    if (np.abs(histogram) < np.abs(self._prev_hist)) and (
                            np.abs(self._prev_hist) < np.abs(self._pre_prev_hist)):
                        self._convergence_flag = True
                        self._hist_peak = self._pre_prev_hist
                elif (np.abs(histogram) <= 0.15 * np.abs(self._hist_peak)) or sign_diff_flag:
                    if np.sign(self._prev_hist) > 0:
                        if sign_diff_flag:
                            self.add_trade_point(date, new_point["Close"], "sell")
                        else:
                            self.add_trade_point(date, new_point["Close"], "actively sell")
                    elif np.sign(self._prev_hist) < 0:
                        if sign_diff_flag:
                            self.add_trade_point(date, new_point["Close"], "buy")
                        else:
                            self.add_trade_point(date, new_point["Close"], "actively buy")
                    self._convergence_flag = False
                    self._pre_prev_hist = self._prev_hist
                    self._prev_hist = histogram
                    return
            self._pre_prev_hist = self._prev_hist
            self._prev_hist = histogram
            self.add_trade_point(date, new_point["Close"], "none")

    def calculate(self, data: Optional[pd.DataFrame] = None):
        """
        Calculates MACD and signal line for provided data

        :param data: Default None. User has to provide data before or inside calculate method.
        """
        super().calculate(data)
        # calculating short and long moving averages
        short_ma = ma.EMA(time_series=self.data["Close"], N=self._short_period)
        long_ma = ma.EMA(time_series=self.data["Close"], N=self._long_period)
        # macd = MA_short - MA_long
        MACD = short_ma - long_ma
        # singal is MA on macd
        signal = ma.EMA(MACD, self._signal_period)
        histogram = MACD - signal
        df_data = {"date": self.data.index, "MACD": MACD, "signal": signal, "histogram": histogram}
        self.MACD_val = pd.DataFrame(data=df_data).set_index("date")

        return self

    def find_trade_points(self) -> pd.DataFrame:
        """
        Finds trade points in provided data using previously calculated indicator value
        """
        if self.MACD_val is None:
            raise SyntaxError("find_trade_points can be called only after calculate method was called at least once")
        for i in range(len(self.MACD_val)):
            date = self.MACD_val.index[i]
            MACD = self.MACD_val["MACD"][i]
            histogram = self.MACD_val["histogram"][i]
            point = self.data.loc[date]
            self.__make_trade_decision(point, date, MACD, histogram)
        return self.trade_points

    def plot(self, start_date: Optional[pd.Timestamp] = None, end_date: Optional[pd.Timestamp] = None):
        """
        Plots the MACD graphic with highlighted trade points in specified time diapason
        """
        if (start_date is None) or (start_date < self.MACD_val.index[0]):
            start_date = self.MACD_val.index[0]
        if (end_date is None) or (end_date > self.MACD_val.index[-1]):
            end_date = self.MACD_val.index[-1]

        selected_data = self.data[start_date:end_date]
        selected_trade_points = self.select_action_trade_points(start_date=start_date, end_date=end_date)
        selected_macd = self.MACD_val[start_date:end_date]

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

        buy_actions = ["buy", "actively buy"]
        fig.add_trace(go.Scatter(x=selected_trade_points.index,
                                 y=selected_trade_points["Price"],
                                 mode="markers",
                                 marker=dict(
                                     color=np.where(selected_trade_points["Action"].isin(buy_actions), "green", "red"),
                                     size=5),
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
        ), row=2, col=1)

        fig.update_layout(title=f"Price with MACD",
                          xaxis_title="Date")

        fig.show()
