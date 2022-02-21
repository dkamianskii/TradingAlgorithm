import indicators.moving_averages as ma
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional
from indicators.AbstractIndicator import AbstractIndicator


class MACD(AbstractIndicator):
    """
    Class for MACD calculation

    Moving average convergence divergence (MACD) is a trend-following momentum indicator that shows the relationship
    between two moving averages of a security’s price. The classical MACD is calculated by subtracting the 26-period
    exponential moving average (EMA) from the 12-period EMA.
    The result of that calculation is the MACD line. A nine-day EMA of the MACD called the "signal line".
    """

    def __init__(self, data: Optional[pd.DataFrame] = None,
                 short_period: Optional[int] = 12,
                 long_period: Optional[int] = 26,
                 signal_period: Optional[int] = 9):
        """
        :param data: time series for computing MACD. Could be not specified with instantiating,
         but must be set before calculating
        """
        super().__init__(data)
        self._short_period: int = 0
        self._long_period: int = 0
        self._signal_period: int = 0
        self.set_ma_periods(short_period, long_period, signal_period)
        self.MACD_val: Optional[pd.DataFrame] = None

    def set_ma_periods(self, short_period: Optional[int] = 12,
                       long_period: Optional[int] = 26,
                       signal_period: Optional[int] = 9):
        if (short_period == long_period):
            raise ValueError("short period and long period can't be equal")
        if (short_period > long_period):
            t = long_period
            long_period = short_period
            short_period = t
        self._short_period = short_period
        self._long_period = long_period
        self._signal_period = signal_period

    def calculate(self, data: Optional[pd.DataFrame] = None):
        """
        Calculates MACD and signal line for provided data

        :param data: Default None. User has to provide data before or inside calculate method.
        """
        super().calculate(data)
        # calculating short and long moving averages
        short_ma = ma.EMA(time_series=self.price, N=self._short_period)
        long_ma = ma.EMA(time_series=self.price, N=self._long_period)
        # macd = MA_short - MA_long
        macd = short_ma - long_ma
        # singal is MA on macd
        signal = ma.EMA(macd, self._signal_period)
        histogram = macd - signal
        df_data = {"date": self.data.index, "MACD": macd, "signal": signal, "histogram": histogram}
        self.MACD_val = pd.DataFrame(data=df_data).set_index("date")

        return self

    def find_trade_points(self, strategy: str = "classic") -> pd.DataFrame:
        """
        Finds trade points in provided data using previously calculated indicator value
        """
        self.clear_trade_points()
        if(self.MACD_val is None):
            raise SyntaxError("find_trade_points can be called only after calculate method was called at least once")
        elif(strategy == "classic"):
            self.__classic_strategy()
        elif(strategy == "convergence"):
            self.__convergence_strategy()
        return self.trade_points

    def __classic_strategy(self):
        """
        Trade strategy explanation:
        Buy / Open long, when index line cross up through the signal, actively buy when MACD is below zero.
        Sell / Open short, when index line cross down through the signal, actively sell when MACD is above zero.
        """
        prev_hist = None
        for date, row in self.MACD_val.iterrows():
            if (prev_hist is None):
                prev_hist = row["histogram"]
                continue

            if ((row["histogram"] == 0) or (np.sign(prev_hist) != np.sign(row["histogram"]))):
                if (np.sign(prev_hist) > 0):
                    if(row["MACD"] > 0):
                        self.add_trade_point(date=date, action="actively sell")
                    else:
                        self.add_trade_point(date=date, action="sell")
                elif (np.sign(prev_hist) < 0):
                    if(row["MACD"] < 0):
                        self.add_trade_point(date=date, action="actively buy")
                    else:
                        self.add_trade_point(date=date, action="buy")

            prev_hist = row["histogram"]

    def __convergence_strategy(self):
        """
        Trade strategy explanation:
        On the first step detecting whether there is a convergence in MACD lines.
        We say that MACD lines converging if there are 2 sequential MACD hist downsizing.
        Like cur hist < prev hist < pre-prev hist. By abs value.
        Secondly, after convergence was established we take pre-prev hist value as a starting peak,
        and waiting for a moment when hist value will be less then 15% of the starting peak or it will be different sign.
        When it comes to the trade point by convergence and not by sign changing, it's a sign for active trade action,
        because it is playing on the expectations and not just following trend when it might be already late.
        """
        prev_hist, prev_hist_2 = None, None
        convergence_flag, sign_diff_flag = False, False
        hist_peak = None
        for date, row in self.MACD_val.iterrows():
            if (prev_hist_2 is None):
                prev_hist_2 = row["histogram"]
                continue
            if (prev_hist is None):
                prev_hist = row["histogram"]
                continue

            sign_diff_flag = np.sign(prev_hist) != np.sign(row["histogram"])
            if not convergence_flag:
                if ((np.abs(row["histogram"]) < np.abs(prev_hist)) and (np.abs(prev_hist) < np.abs(prev_hist_2))):
                    convergence_flag = True
                    hist_peak = prev_hist_2
            elif ((np.abs(row["histogram"]) <= 0.15*np.abs(hist_peak)) or sign_diff_flag):
                if (np.sign(prev_hist) > 0):
                    if sign_diff_flag:
                        self.add_trade_point(date=date, action="sell")
                    else:
                        self.add_trade_point(date=date, action="actively sell")
                elif (np.sign(prev_hist) < 0):
                    if sign_diff_flag:
                        self.add_trade_point(date=date, action="buy")
                    else:
                        self.add_trade_point(date=date, action="actively buy")
                convergence_flag = False

            prev_hist_2 = prev_hist
            prev_hist = row["histogram"]

    def plot(self, start_date: Optional[pd.Timestamp] = None, end_date: Optional[pd.Timestamp] = None):
        """
        Plots the MACD graphic with highlighted trade points in specified time diapason
        """
        if((start_date is None) or (start_date < self.MACD_val.index[0])):
            start_date = self.MACD_val.index[0]
        if((end_date is None) or (end_date > self.MACD_val.index[-1])):
            end_date = self.MACD_val.index[-1]

        day_diff = (end_date - start_date).days
        selected_macd = self.MACD_val[start_date:end_date]

        fig, ax = plt.subplots(figsize=(int(day_diff / 3), 8))
        ax.plot(selected_macd.index, selected_macd["MACD"], color="blue")
        ax.plot(selected_macd.index, selected_macd["signal"], color="orange")

        positive_hist = selected_macd[selected_macd["histogram"] >= 0]
        negative_hist = selected_macd[selected_macd["histogram"] < 0]
        ax.bar(positive_hist.index, positive_hist["histogram"], color="green")
        ax.bar(negative_hist.index, negative_hist["histogram"], color="red")

        selected_trade_points = self.select_action_trade_points(start_date=start_date, end_date=end_date)
        ax.scatter(selected_trade_points.index,
                   self.MACD_val[self.MACD_val.index.isin(selected_trade_points.index)]["MACD"],
                   facecolors='none', linewidths=2, marker="o", color="green")
        ax.grid(which='both')
        plt.show()
