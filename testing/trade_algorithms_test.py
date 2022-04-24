from datetime import datetime

import yfinance as yf
import pandas as pd
from trading.trade_manager import TradeManager
from trading.trade_algorithms.rsi_trade_algorithm import RSITradeAlgorithm
from trading.trade_algorithms.macd_trade_algorithm import MACDTradeAlgorithm
from trading.trade_algorithms.super_trend_trade_algorithm import SuperTrendTradeAlgorithm

start_date = "2000-01-01"
end_date = "2021-12-31"
test_start_date = "2020-01-01"
back_test_start_date = "2019-01-01"
test_start_date_ts = pd.Timestamp(ts_input=test_start_date)
start_test = datetime.strptime(test_start_date, "%Y-%m-%d")
end_test = datetime.strptime(end_date, "%Y-%m-%d")
dates_test = pd.date_range(start_test, end_test)

# Mega corps pack
data_wmt = yf.download("WMT", start=start_date, end=end_date)
# data_aapl = yf.download("AAPL", start=start_date, end=end_date)
# data_msft = yf.download("MSFT", start=start_date, end=end_date)
# data_jpm = yf.download("JPM", start=start_date, end=end_date)
# data_ko = yf.download("KO", start=start_date, end=end_date)
# data_pg = yf.download("PG", start=start_date, end=end_date)
data_xom = yf.download("XOM", start=start_date, end=end_date)

print("RSI INDICATOR TRADING")

manager_1 = TradeManager(days_to_chill=5)

manager_1.set_tracked_stock("WMT", data_wmt[:test_start_date_ts], RSITradeAlgorithm())
# manager_1.set_tracked_stock("JPM", data_jpm[:test_start_date_ts], RSITradeAlgorithm())
manager_1.set_tracked_stock("XOM", data_xom[:test_start_date_ts], RSITradeAlgorithm())

train_result_1 = manager_1.train(back_test_start_date)
print(manager_1.get_chosen_params())

for date in dates_test:
    if date in data_wmt[start_test:].index:
        point_wmt = data_wmt.loc[date]
        manager_1.evaluate_new_point("WMT", point_wmt, date)
    # if date in data_jpm[start_test:].index:
    #     point_jpm = data_jpm.loc[date]
    #     manager_1.evaluate_new_point("JPM", point_jpm, date)
    if date in data_xom[start_test:].index:
        point_xom = data_xom.loc[date]
        manager_1.evaluate_new_point("XOM", point_xom, date)

print(manager_1.get_trade_results())
print(manager_1.get_bids_history())
manager_1.plot_earnings_curve()
manager_1.plot_stock_history("WMT")
# manager_1.plot_stock_history("JPM")
manager_1.plot_stock_history("XOM")

print("MACD INDICATOR TRADING")

manager_2 = TradeManager(days_to_chill=5)

manager_2.set_tracked_stock("WMT", data_wmt[:test_start_date_ts], MACDTradeAlgorithm())
# manager_2.set_tracked_stock("JPM", data_jpm[:test_start_date_ts], MACDTradeAlgorithm())
manager_2.set_tracked_stock("XOM", data_xom[:test_start_date_ts], MACDTradeAlgorithm())

train_result_2 = manager_2.train(back_test_start_date)
print(manager_2.get_chosen_params())

for date in dates_test:
    if date in data_wmt[start_test:].index:
        point_wmt = data_wmt.loc[date]
        manager_2.evaluate_new_point("WMT", point_wmt, date)
    # if date in data_jpm[start_test:].index:
    #     point_jpm = data_jpm.loc[date]
    #     manager_2.evaluate_new_point("JPM", point_jpm, date)
    if date in data_xom[start_test:].index:
        point_xom = data_xom.loc[date]
        manager_2.evaluate_new_point("XOM", point_xom, date)

print(manager_2.get_trade_results())
print(manager_2.get_bids_history())
manager_2.plot_earnings_curve()
manager_2.plot_stock_history("WMT")
# manager_2.plot_stock_history("JPM")
manager_2.plot_stock_history("XOM")

print("SUPER TREND INDICATOR TRADING")

manager_3 = TradeManager(days_to_chill=5)

manager_3.set_tracked_stock("WMT", data_wmt[:test_start_date_ts], SuperTrendTradeAlgorithm())
# manager_3.set_tracked_stock("JPM", data_jpm[:test_start_date_ts], SuperTrendTradeAlgorithm())
manager_3.set_tracked_stock("XOM", data_xom[:test_start_date_ts], SuperTrendTradeAlgorithm())

train_result_3 = manager_3.train(back_test_start_date)
print(manager_3.get_chosen_params())

for date in dates_test:
    if date in data_wmt[start_test:].index:
        point_wmt = data_wmt.loc[date]
        manager_3.evaluate_new_point("WMT", point_wmt, date)
    # if date in data_jpm[start_test:].index:
    #     point_jpm = data_jpm.loc[date]
    #     manager_3.evaluate_new_point("JPM", point_jpm, date)
    if date in data_xom[start_test:].index:
        point_xom = data_xom.loc[date]
        manager_3.evaluate_new_point("XOM", point_xom, date)

print(manager_3.get_trade_results())
print(manager_3.get_bids_history())
manager_3.plot_earnings_curve()
manager_3.plot_stock_history("WMT")
# manager_3.plot_stock_history("JPM")
manager_3.plot_stock_history("XOM")
