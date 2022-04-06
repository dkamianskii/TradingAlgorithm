from datetime import datetime

import yfinance as yf
import pandas as pd
from Trading.TradeManager import TradeManager
from Trading.RSITradeAlgorithm import RSITradeAlgorithm

start_date = "2000-01-01"
end_date = "2021-12-31"
test_start_date = "2020-01-01"
back_test_start_date = "2019-01-01"
test_start_date_ts = pd.Timestamp(ts_input="2021-12-20")

# Mega corps pack
data_wmt = yf.download("WMT", start=start_date, end=end_date)
data_aapl = yf.download("AAPL", start=start_date, end=end_date)
data_msft = yf.download("MSFT", start=start_date, end=end_date)
data_jpm = yf.download("JPM", start=start_date, end=end_date)
data_ko = yf.download("KO", start=start_date, end=end_date)
data_pg = yf.download("PG", start=start_date, end=end_date)
data_xom = yf.download("XOM", start=start_date, end=end_date)

manager = TradeManager()

rsi_trade_alg = RSITradeAlgorithm()

manager.set_tracked_stock("WMT", data_wmt[:test_start_date_ts], rsi_trade_alg)
manager.set_tracked_stock("AAPL", data_aapl[:test_start_date_ts], rsi_trade_alg)
manager.set_tracked_stock("XOM", data_xom[:test_start_date_ts], rsi_trade_alg)

train_result = manager.train(back_test_start_date, test_start_date)

start_test = datetime.strptime(test_start_date, "%Y-%m-%d")
end_test = datetime.strptime(end_date, "%Y-%m-%d")
dates_test = pd.date_range(start_test, end_test)
for date in dates_test:
    if date in data_wmt[start_test:].index:
        point_wmt = data_wmt.loc[date]
        manager.evaluate_new_point("WMT", point_wmt, date)
    if date in data_aapl[start_test:].index:
        point_aapl = data_wmt.loc[date]
        manager.evaluate_new_point("AAPL", point_aapl, date)
    if date in data_xom[start_test:].index:
        point_xom = data_xom.loc[date]
        manager.evaluate_new_point("XOM", point_xom, date)

print(manager.trade_result)

