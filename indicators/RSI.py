# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 13:13:05 2021

@author: aspod
"""

import indicators.moving_averages as ma
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, Dict
from indicators.AbstractIndicator import AbstractIndicator

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
    def __init__(self, data: Optional[pd.DataFrame] = None, N: Optional[int] = 14):
        """
        :param N: Parameter that determines number of points in MA, used for calculating index.
         Must be > 0. By default, equals 14
        :param data: time series for computing RSI. Could be not specified with instantiating,
         but must be set before calculating
        """
        super(RSI, self).__init__(data)
        self._N: int
        self.set_N(N)
        self.RSI_val: Optional[pd.Series] = None

    def set_N(self, N: int):
        """
        :param N: Parameter that determines number of points in MA, used for calculating index. Must be > 0.
        """
        if (N <= 0):
            raise ValueError("N parameter must be > 0 and less then length of the time_series")
        self._N = N
        return self

    def calculate(self, data: Optional[pd.DataFrame] = None):
        """
        Calculates RSI for provided data

        Formula: RSI = 100 - 100 / (1 + (average gain / average loss) )

        :param data: Default None. User has to provide data before or inside calculate method.
        """
        super().calculate(data)
        # calculating U and D
        data_len = len(self.price)
        days_U_D = {'U': np.zeros((data_len - 1), dtype=float), 'D': np.zeros((data_len - 1), dtype=float)}
        for i in range(1, data_len):
            if (self.price[i] > self.price[i - 1]):
                days_U_D['U'][i - 1] = self.price[i] - self.price[i - 1]
            else:
                days_U_D['D'][i - 1] = self.price[i - 1] - self.price[i]

        # calculating RSI
        RS = np.divide(ma.SMMA(days_U_D['U'], self._N), ma.SMMA(days_U_D['D'], self._N))
        RSI = 100 - 100 / (1 + RS)
        self.RSI_val = pd.Series(data=RSI, index=self.data.index[self._N:])

        return self

    def find_trade_points(self) -> pd.DataFrame:
        """
        Finds trade points in provided data using previously calculated indicator value

        Trade strategy explanation:
        Key index values are 70 and 30.
        When index is >= 70 we consider that stock is overbought and for <= 30 is oversold.
        The strategy for sell/open short case will be described bellow, buy/open long is symmetrically opposite.

        1. While RSI is in (70,30) boundaries we consider there is no signal to sell or buy.
        2. Once RSI hits boundary we start track it's dynamic.
        3. If RSI achieves level of 80, it is a strong sign that stock is overbought and soon will go down - actively sell.
        4. In other cases:
            - if we see a rapid index change (>= 5 in one day) - sell.
            - if we see RSI goes down and gets close to the boundary - sell.
            - if current RSI is less than 70 but previous day it was over it - sell.
        """
        self.clear_trade_points()
        self.__trade_rule()
        return self.trade_points

    def __trade_rule(self):
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
        dates = self.RSI_val.index
        prev = None
        for i, rsi in enumerate(self.RSI_val):
            if ((rsi < 70) and (rsi > 30)):
                if (prev is None):
                    continue
                else:
                    if ((prev >= 70) and (rsi >= 67.5)):
                        self.add_trade_point(dates[i], "sell")
                    elif ((prev <= 30) and (rsi <= 32.5)):
                        self.add_trade_point(dates[i], "buy")
                    prev = None
                    continue

            if (rsi > 80):
                self.add_trade_point(dates[i], "actively sell")
                continue
            if (rsi < 20):
                self.add_trade_point(dates[i], "actively buy")
                continue

            if (rsi >= 70):
                if (prev is None):
                    prev = rsi
                    continue
                if ((abs(rsi - prev) >= 5) or ((rsi < prev) and (rsi < 70.5))):
                    self.add_trade_point(dates[i], "sell")
                    prev = rsi
                    continue
            if (rsi <= 30):
                if (prev is None):
                    prev = rsi
                    continue
                if ((abs(rsi - prev) >= 5) or ((rsi > prev) and (rsi > 30.5))):
                    self.add_trade_point(dates[i], "buy")
                    prev = rsi
                    continue

    def plot(self, start_date: Optional[pd.Timestamp] = None, end_date: Optional[pd.Timestamp] = None):
        """
        Plots the RSI graphic in specified time diapason
        """
        if((start_date is None) or (start_date < self.RSI_val.index[0])):
            start_date = self.RSI_val.index[0]
        if((end_date is None) or (end_date > self.RSI_val.index[-1])):
            end_date = self.RSI_val.index[-1]

        day_diff = (end_date - start_date).days
        selected_rsi = self.RSI_val[start_date:end_date]

        fig, ax = plt.subplots(figsize=(int(day_diff / 3), 8))
        ax.plot(selected_rsi.index, selected_rsi, color="blue")
        ax.plot(selected_rsi.index, np.full(len(selected_rsi), 70), linestyle="--", color="red")
        ax.plot(selected_rsi.index, np.full(len(selected_rsi), 30), linestyle="--", color="red")

        selected_trade_points = self.select_action_trade_points(start_date=start_date, end_date=end_date)

        ax.scatter(selected_trade_points.index,
                   self.RSI_val[self.RSI_val.index.isin(selected_trade_points.index)],
                   facecolors='none', linewidths=2, marker="o", color="green")
        ax.grid(which='both')
        plt.show()
