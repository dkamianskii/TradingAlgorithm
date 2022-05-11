from datetime import datetime

import yfinance as yf
import pandas as pd
import numpy as np

from indicators import moving_averages as ma
from indicators.ma_support_levels import MASupportLevels
from indicators.macd import MACD, MACDTradeStrategy
from indicators.rsi import RSI
from indicators.atr import ATR
from indicators.super_trend import SuperTrend
from indicators.bollinger_bands import BollingerBands
from indicators.cci import CCI

import cufflinks as cf
import plotly.graph_objects as go

cf.go_offline()

start_date = "2000-01-01"
end_date = "2021-12-31"
test_start_date = "2020-01-01"
back_test_start_date = "2019-01-01"
test_start_date_ts = pd.Timestamp(ts_input=test_start_date)
start_test = datetime.strptime(test_start_date, "%Y-%m-%d")
end_test = datetime.strptime(end_date, "%Y-%m-%d")
dates_test = pd.date_range(start_test, end_test)

data = yf.download("XOM", start=start_date, end=end_date)

# new_date = data.index[-1]
# new_point = data.loc[new_date]
# train_data = data.iloc[0:-1]

train_data = data.loc[:start_test]
test_data = data[start_test:]

# MA Support Levels

# support_levels_test = MASupportLevels(data=data)
# support_levels_test.calculate()
# support_levels_test.set_tested_MAs_usage(True)
# support_levels_test.test_MAs_for_data()
# support_levels_test.find_trade_points()
# tp_1 = support_levels_test.select_action_trade_points()
# support_levels_test.plot()
#
# support_levels = MASupportLevels(data=train_data)
# # support_levels.set_ma_periods([20, 50, 100, 200])
# support_levels.calculate()
# support_levels.set_tested_MAs_usage(True)
# mas_test_result = support_levels.test_MAs_for_data()
# # print(mas_test_result)
# # support_levels.find_trade_points()
# for date, point in test_data.iterrows():
#     support_levels.evaluate_new_point(point, date, update_data=False)
#     train_data.loc[date] = point
# support_levels.plot(pd.Timestamp(start_test), pd.Timestamp(end_test))

# RSI

# rsi_test = RSI(data=data)
# rsi_test.calculate()
# rsi_test.find_trade_points()
# b1 = rsi_test.select_action_trade_points()
# rsi_test.plot()
#
# rsi = RSI(data=train_data)
# rsi.set_N(8)
# rsi.calculate()
# # rsi.find_trade_points()
# for date, point in test_data.iterrows():
#     rsi.evaluate_new_point(point, date, update_data=False)
#     train_data.loc[date] = point
# rsi.plot(pd.Timestamp(start_test), pd.Timestamp(end_test))

# MACD

# macd_test = MACD(data=data)
# macd_test.calculate()
# macd_test.find_trade_points()
# macd_test.plot(pd.Timestamp(start_test), pd.Timestamp(end_test))

# macd = MACD(data=train_data, trade_strategy=MACDTradeStrategy.CLASSIC)
# macd.set_ma_periods(10, 22, 9)
# macd.set_trade_strategy(MACDTradeStrategy.CONVERGENCE)
# macd.calculate()
# # macd.find_trade_points()
# for date, point in test_data.iterrows():
#     macd.evaluate_new_point(point, date, update_data=False)
#     train_data.loc[date] = point
# macd.plot(pd.Timestamp(start_test), pd.Timestamp(end_test))

# Super Trend

# super_trend_test = SuperTrend(data=data)
# super_trend_test.set_params(10, 3)
# super_trend_test.calculate()
# super_trend_test.find_trade_points()
# super_trend_test.plot()
#
super_trend = SuperTrend(data=train_data)
super_trend.set_params(10, 3)
super_trend.calculate()
# super_trend.find_trade_points()
for date, point in test_data.iterrows():
    super_trend.evaluate_new_point(point, date, update_data=False)
    train_data.loc[date] = point
super_trend.plot(pd.Timestamp(start_test), pd.Timestamp(end_test))

# Bollinger bands

# bollinger_bands_test = BollingerBands(data=data)
# bollinger_bands_test.calculate()
# bollinger_bands_test.find_trade_points()
# bollinger_bands_test.plot()
#
# bollinger_bands = BollingerBands(data=train_data)
# bollinger_bands.calculate()
# # bollinger_bands.find_trade_points()
# for date, point in test_data.iterrows():
#     bollinger_bands.evaluate_new_point(point, date, update_data=False)
# bollinger_bands.plot(pd.Timestamp(start_test), pd.Timestamp(end_test))

# CCI

# cci_test = CCI(data=data)
# cci_test.calculate()
# cci_test.find_trade_points()
# cci_test.plot()
#
# cci_test = CCI(data=train_data)
# cci_test.calculate()
# # cci_test.find_trade_points()
# for date, point in test_data.iterrows():
#     cci_test.evaluate_new_point(point, date, update_data=False)
# cci_test.plot(pd.Timestamp(start_test), pd.Timestamp(end_test))



