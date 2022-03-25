from indicators import moving_averages as ma
from indicators.ATR import ATR
import yfinance as yf
import pandas as pd
import typing

from indicators.MACD import MACD
from indicators.RSI import RSI
from indicators.MASupportLevels import MASupportLevels
from indicators.SuperTrend import SuperTrend

data = yf.download("AMD", start="2021-05-14", end="2021-12-21")
new_date = pd.Timestamp(ts_input="2021-12-20")
new_point = data.loc[new_date]
train_date = data.iloc[0:-1]
# start_date = pd.Timestamp(ts_input="2021-10-01")
# print(data["Close"][start_date])
# test_df = pd.DataFrame(index=data.index[20:], columns=["Value", "Color"])
# print(test_df)

# support_levels = MASupportLevels(data=data)
# support_levels.calculate()
# support_levels.test_MAs_for_data()
# support_levels.find_trade_points(use_tested_MAs=True)
# tp = support_levels.trade_points
# super_trend = SuperTrend(data=data)
# super_trend.calculate()
# super_trend.find_trade_points()
# tp = super_trend.select_action_trade_points()
# super_trend.plot()

# action_trade_points = support_levels.select_action_trade_points()
# print(action_trade_points)
# support_levels.plot()
rsi_test = RSI(data=data)
rsi_test.calculate()
rsi_test.find_trade_points()
b1 = rsi_test.select_action_trade_points()
rsi_test.plot()

rsi_apple = RSI(data=train_date)
rsi_apple.calculate()
rsi_apple.find_trade_points()
rsi_apple.evaluate_new_point(new_point, new_date)
b2 = rsi_apple.select_action_trade_points()

m1 = rsi_test.RSI_val[-1]
m2 = rsi_apple.RSI_val[-1]

rsi_apple.plot()

dd = 1


# macd_apple = MACD(data=data["Close"])
# macd_apple.calculate()




