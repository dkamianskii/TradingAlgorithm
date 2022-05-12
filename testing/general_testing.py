from datetime import datetime

import yfinance as yf
import pandas as pd
import numpy as np
from trading.trade_algorithms.one_indicator_trade_algorithms.rsi_trade_algorithm import RSITradeAlgorithm
from trading.trade_manager import TradeManager

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

cls_rsi = RSITradeAlgorithm

m1 = TradeManager()
m2 = TradeManager()

e1 = cls_rsi()
e2 = cls_rsi()
m1.set_tracked_stock("XOM", data[:test_start_date_ts], cls_rsi())
print(m1.get_tracked_stocks())
m2.set_tracked_stock("ZIM", data, cls_rsi())
print(m2.get_tracked_stocks())


