from datetime import datetime

import yfinance as yf
import pandas as pd
import numpy as np

from indicators.abstract_indicator import TradeAction
from indicators.atr import ATR
from trading.trade_algorithms.one_indicator_trade_algorithms.rsi_trade_algorithm import RSITradeAlgorithm

import cufflinks as cf
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from helping.base_enum import BaseEnum

cf.go_offline()

start_date = "2021-12-01"
end_date = "2021-12-31"
test_start_date = "2015-01-03"
back_test_start_date = "2019-01-01"
test_start_date_ts = pd.Timestamp(ts_input=test_start_date)
start_test = datetime.strptime(test_start_date, "%Y-%m-%d")
end_test = datetime.strptime("2024-02-01", "%Y-%m-%d")
dates_test = pd.date_range(start_test, end_test)



data = yf.download("XOM", start=start_date, end=end_date)

m = pd.DataFrame(data={"Date": data.index}).set_index("Date")
m["E"] = TradeAction.NONE
print(m)
# e = pd.DataFrame(data={"A": [1, 2, 6, 5]})
# e.iloc[0, "A"] = 4
#
# print(e)

