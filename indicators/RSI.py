# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 13:13:05 2021

@author: aspod
"""

import indicators.moving_averages as ma 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#RSI - Relative strength index 
class RSI:
    """
    Class for calculation of Relative Strength Index
    """
    strategies = {"open long", "open short"}
    
    def __init__(self,
                 N: "int > 0, period of RSI" = 14,
                 data: "pandas series or dataframe with column 'Close'" = None):
        self.set_N(N)
        self.set_data(data)
        self.RSI_val = None
        self.trade_points = None
    
    def set_N(self, N: "int > 0, period of RSI"):
        self.N = N
        return self
    
    def set_data(self, data: "pandas series or dataframe with column 'Close'"):
        if(isinstance(data, pd.core.series.Series)):
            self.data = data
            return self
        if(isinstance(data, pd.core.frame.DataFrame)):
            self.data = data["Close"]
            return self
        else:
            raise TypeError("data parameter must be pandas series or dataframe with column 'Close'")
               
    def calculate(self) -> "pandas Series with calculated RSI for provided time series starting with N+1 time point":
        """
        Calculate RSI for provided data and select trade points
        """
        #calculating U and D
        data_len = len(self.data)
        days_U_D = {'U': np.ndarray((data_len - 1), dtype=float), 'D': np.ndarray((data_len - 1), dtype=float)}
        for i in range(1, data_len):
            if(self.data[i] > self.data[i - 1]):
                days_U_D['U'][i - 1] = self.data[i] - self.data[i - 1]
                days_U_D['D'][i - 1] = 0
            else:
                days_U_D['U'][i - 1] = 0
                days_U_D['D'][i - 1] = self.data[i - 1] - self.data[i]
                
        #calculating RSI        
        RS = np.divide(ma.SMMA(days_U_D['U'], self.N), ma.SMMA(days_U_D['D'], self.N))
        RSI = 100 - 100/(1 + RS)
        self.RSI_val = pd.Series(data=RSI, index=self.data.index[self.N:])
        
        self.__trade_rule()
        
        return self.RSI_val
    
    def __trade_rule(self):
        """
        Если кар. РСА > 80 : опен шорт, если < 20 : опен лонг.
        Если флаг пустой и кар. РСА - овербот : ставим флаг на овербот, - оверсолд : флаг на оверсолд. Контин.
        Если кар. РСА - овербот и флаг овербот :
            Если кар. РСА > прев. РСА : Контин.,
            Иначе если (прев. РСА - кар. РСА > 5) или (кар. РСА < 70.5) : опен шорт.,
        Иначе если кар. РСА > 67.5 : опен шорт
        Для лонга стратегия зеркальная
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
                                          
    def __add_trade_point(self,
                          i: "position in RSI series array",
                          action: "str : 'open_long' or 'open_short'"):
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
        for i in range(len(self.trade_points['date'])):
            print(f"date: {self.trade_points['date'][i]}, price: {self.trade_points['price'][i]}, RSI:  {self.trade_points['RSI'][i]}, strategy: {self.trade_points['strategy'][i]}")
            
    def __select_trade_points(self, first_date):
        i = 0
        for date in self.trade_points["date"]:
            if(date >= first_date):
                break
            i += 1
        
        return {"date": self.trade_points["date"][i:],
                "price": self.trade_points["price"][i:],
                "RSI": self.trade_points["RSI"][i:],
                "strategy": self.trade_points["strategy"][i:]}
        
    def plot(self, showing_size: "int > 0, how much data points will be shown" = 30):
        fig, ax = plt.subplots(figsize=(int(showing_size/3), 5))
        ax.plot(self.RSI_val.index[-showing_size:], self.RSI_val[-showing_size:], color="blue")
        ax.plot(self.RSI_val.index[-showing_size:], np.full(showing_size,70), linestyle="--", color="red")
        ax.plot(self.RSI_val.index[-showing_size:], np.full(showing_size,30), linestyle="--", color="red")
        selected_trade_points = self.__select_trade_points(self.RSI_val.index[-showing_size])
        ax.scatter(selected_trade_points["date"], selected_trade_points["RSI"], facecolors='none', linewidths=2, marker="o", color="green")
        ax.grid(which='both')
        plt.show()


