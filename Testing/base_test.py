from indicators import moving_averages as ma
from indicators.ATR import ATR
import yfinance as yf
import pandas as pd
import typing

from indicators.MACD import MACD
from indicators.RSI import RSI
from indicators.MASupportLevels import MASupportLevels

data = yf.download("AAPL", start="2021-01-01", end="2021-12-30")
#start_date = pd.Timestamp(ts_input="2021-10-01")

support_levels = MASupportLevels(data=data)
support_levels.calculate()
support_levels.test_MAs_for_data()
support_levels.find_trade_points(use_tested_MAs=True)
print(data.shape[0])
action_trade_points = support_levels.select_action_trade_points()
print(action_trade_points)
support_levels.plot()

#rsi_apple = RSI(data=data)
#rsi_apple.calculate()
#b = rsi_apple.select_action_trade_points()


# macd_apple = MACD(data=data["Close"])
# macd_apple.calculate()




