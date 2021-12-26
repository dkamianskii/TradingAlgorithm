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

#RSI - Relative strength index 
class RSI:
    """
    Class for calculation of Relative Strength Index

    The relative strength index (RSI) is a momentum indicator
    that measures the magnitude of recent price changes
    to evaluate overbought or oversold conditions in the price of a stock or other asset.

    Methods: set_N, set_data, calculate, print_trade_points, plot
    Attributes: RSI_val, trade_points
    """
    strategies = {"open long", "open short"}
    
    def __init__(self, N: Optional[int] = 14, data: Optional[pd.Series] = None):
        """
        :param N: Parameter that determines number of points in MA, used for calculating index.
         Must be > 0. By default, equals 14
        :param data: time series for computing RSI. Could be not specified with instantiating,
         but must be set before calculating
        """
        self.set_N(N)
        self.set_data(data)
        self.RSI_val: pd.Series = None
        #dict['date', 'price', 'RSI', 'strategy']
        self.trade_points: Dict = None
    
    def set_N(self, N: int):
        """
        :param N: Parameter that determines number of points in MA, used for calculating index. Must be > 0.
        """
        if (N <= 0):
            raise ValueError("N parameter must be > 0 and less then length of the time_series")
        self.N = N
        return self
    
    def set_data(self, data: pd.Series):
        """
        :param data: time series for computing RSI.
        """
        self.data = data
        return self
               
    def calculate(self) -> pd.Series:
        """
        Calculates RSI for provided data and selects trade points

        Formula: RSI = 100 - 100 / (1 + (average gain / average loss) )

        Trade points can be got from attribute trade_points
        """
        #calculating U and D
        data_len = len(self.data)
        days_U_D = {'U': np.zeros((data_len - 1), dtype=float), 'D': np.zeros((data_len - 1), dtype=float)}
        for i in range(1, data_len):
            if(self.data[i] > self.data[i - 1]):
                days_U_D['U'][i - 1] = self.data[i] - self.data[i - 1]
            else:
                days_U_D['D'][i - 1] = self.data[i - 1] - self.data[i]
                
        #calculating RSI        
        RS = np.divide(ma.SMMA(days_U_D['U'], self.N), ma.SMMA(days_U_D['D'], self.N))
        RSI = 100 - 100/(1 + RS)
        self.RSI_val = pd.Series(data=RSI, index=self.data.index[self.N:])
        
        self.__trade_rule()
        
        return self.RSI_val
    
    def __trade_rule(self):
        """
        Trade strategy explanation:
        Key index values are 70 and 30.
        When index is >= 70 we consider that stock is overbought and for <= 30 is oversold.
        We describe strategy for sell/open short case, buy/open long is symmetrically opposite.

        1. While RSI is in (70,30) boundaries we consider there is no signal to sell or buy.
        2. Once RSI hits boundary we start track it's dynamic.
        3. If RSI achieves level of 80, it is a strong sign that stock is overbought and soon will go down, so we open short.
        4. In other cases:
            - if we see a rapid index growth (>= 5 in one day) - open short.
            - if we see RSI goes down and gets close to the boundary - open short.
            - if current RSI is less than 70 but previous day it was over it - open short.
        """
        prev = None
        for i, rsi in enumerate(self.RSI_val):
            if((rsi < 70) and (rsi > 30)):
                if(prev is None):
                    continue
                else:
                    if((prev >= 70) and (rsi >= 67.5)):
                        self.__add_trade_point(i, "open_short")
                    elif((prev <= 30) and (rsi <= 32.5)):
                        self.__add_trade_point(i, "open_long")
                    prev = None
            
            if(rsi > 80):
                self.__add_trade_point(i, "open_short")
                continue
            if(rsi < 20):
                self.__add_trade_point(i, "open_long")
                continue
            
            if(rsi >= 70):
                if(prev is None):
                    prev = rsi
                    continue
                if((abs(rsi - prev) >= 5) or ((rsi < prev) and (rsi < 70.5))):
                    self.__add_trade_point(i, "open_short")
                    prev = rsi
                    continue
            if(rsi <= 30):
                if(prev is None):
                    prev = rsi
                    continue 
                if((abs(rsi - prev) >= 5) or ((rsi > prev) and (rsi > 30.5))):
                    self.__add_trade_point(i, "open_long")
                    prev = rsi
                    continue
                                          
    def __add_trade_point(self, i: int, action: str):
        """
        :param i: position in RSI series array
        :param action: 'open_long' or 'open_short'
        """
        if(self.trade_points is None):
            self.trade_points = {"date": [], "price": [], "RSI": [], "strategy": []}
        
        self.trade_points["date"].append(self.RSI_val.index[i])
        self.trade_points["price"].append(self.data[i + self.N])
        self.trade_points["RSI"].append(self.RSI_val[i])
        if(action == "open_long"):
            self.trade_points["strategy"].append("open_long")
        elif(action == "open_short"):
            self.trade_points["strategy"].append("open_short")
            
    def print_trade_points(self):
        """
        Prints trade points
        """
        for i in range(len(self.trade_points['date'])):
            print(f"date: {self.trade_points['date'][i]}, price: {self.trade_points['price'][i]}, RSI:  {self.trade_points['RSI'][i]}, strategy: {self.trade_points['strategy'][i]}")
            
    def __select_trade_points(self, first_date: pd.Timestamp):
        i = 0
        for date in self.trade_points["date"]:
            if(date >= first_date):
                break
            i += 1
        
        return {"date": self.trade_points["date"][i:],
                "price": self.trade_points["price"][i:],
                "RSI": self.trade_points["RSI"][i:],
                "strategy": self.trade_points["strategy"][i:]}
        
    def plot(self, showing_size: Optional[int] = 30):
        """
        Plots the RSI graphic

        :param showing_size: Specifies how much trade points from the last date to show. By default, equals 30
        """
        fig, ax = plt.subplots(figsize=(int(showing_size/3), 5))
        ax.plot(self.RSI_val.index[-showing_size:], self.RSI_val[-showing_size:], color="blue")
        ax.plot(self.RSI_val.index[-showing_size:], np.full(showing_size,70), linestyle="--", color="red")
        ax.plot(self.RSI_val.index[-showing_size:], np.full(showing_size,30), linestyle="--", color="red")
        selected_trade_points = self.__select_trade_points(self.RSI_val.index[-showing_size])
        ax.scatter(selected_trade_points["date"], selected_trade_points["RSI"], facecolors='none', linewidths=2, marker="o", color="green")
        ax.grid(which='both')
        plt.show()


