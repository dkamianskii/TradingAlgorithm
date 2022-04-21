from datetime import datetime

import yfinance as yf
import pandas as pd
import numpy as np

from indicators import moving_averages as ma
from indicators.macd import MACD
from indicators.rsi import RSI
from indicators.atr import ATR
from indicators.super_trend import SuperTrend

import cufflinks as cf
import plotly.graph_objects as go

cf.go_offline()

data = yf.download("AMD", start="2021-01-01", end="2021-02-21")

start_test = datetime.strptime("2021-01-18", "%Y-%m-%d")
end_test = datetime.strptime("2021-01-26", "%Y-%m-%d")
dates_test = pd.date_range(start_test, end_test)

# start_test = datetime.strptime("2021-01-10", "%Y-%m-%d")
# ee = data[start_test:]
# new_point = data.loc[new_date]
# train_data = data.iloc[0:-1]

# start_date = pd.Timestamp(ts_input="2021-02-19")
# data.loc[start_date] += np.ones(data.shape[1])
# print(data["Close"][start_date])
# test_df = pd.DataFrame(index=data.index[20:], columns=["Value", "Color"])
# print(test_df)

# support_levels = MASupportLevels(data=data)
# support_levels.calculate()
# support_levels.test_MAs_for_data()
# support_levels.find_trade_points(use_tested_MAs=True)
# tp = support_levels.trade_points

# action_trade_points = support_levels.select_action_trade_points()
# print(action_trade_points)
# support_levels.plot()

# RSI

# rsi_test = RSI(data=data)
# rsi_test.calculate()
# rsi_test.find_trade_points()
# b1 = rsi_test.select_action_trade_points()
# rsi_test.plot()
#
# rsi_apple = RSI(data=train_data)
# rsi_apple.calculate()
# rsi_apple.find_trade_points()
# rsi_apple.evaluate_new_point(new_point, new_date)
# b2 = rsi_apple.select_action_trade_points()
# rsi_apple.plot()

# MACD

# macd_test = MACD(data=data)
# macd_test.calculate()
# macd_test.find_trade_points()
# macd_test.plot()
#
# macd_apple = MACD(data=train_data, trade_strategy=MACD.supported_trade_strategies[1])
# macd_apple.calculate()
# macd_apple.find_trade_points()
# macd_apple.evaluate_new_point(new_point, new_date)
# b3 = macd_apple.select_action_trade_points()
# print(b3)
# macd_apple.plot()

# Super Trend

# super_trend = SuperTrend(data=data)
# super_trend.calculate()
# super_trend.find_trade_points()
# super_trend.plot()
#
# super_trend = SuperTrend(data=train_data)
# super_trend.calculate()
# super_trend.find_trade_points()
# super_trend.evaluate_new_point(new_point, new_date)
# super_trend.plot()

dd = 1
