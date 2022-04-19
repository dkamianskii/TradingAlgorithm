from datetime import datetime

import yfinance as yf
import pandas as pd
from trading.trade_manager import TradeManager
from trading.rsi_trade_algorithm import RSITradeAlgorithm

start_date = "2000-01-01"
end_date = "2021-12-31"
test_start_date = "2020-01-01"
back_test_start_date = "2019-01-01"
test_start_date_ts = pd.Timestamp(ts_input=test_start_date)

# Mega corps pack
data_wmt = yf.download("WMT", start=start_date, end=end_date)
# data_aapl = yf.download("AAPL", start=start_date, end=end_date)
# data_msft = yf.download("MSFT", start=start_date, end=end_date)
data_jpm = yf.download("JPM", start=start_date, end=end_date)
# data_ko = yf.download("KO", start=start_date, end=end_date)
# data_pg = yf.download("PG", start=start_date, end=end_date)
data_xom = yf.download("XOM", start=start_date, end=end_date)

manager = TradeManager(days_to_chill=5)

manager.set_tracked_stock("WMT", data_wmt[:test_start_date_ts], RSITradeAlgorithm())
manager.set_tracked_stock("JPM", data_jpm[:test_start_date_ts], RSITradeAlgorithm())
manager.set_tracked_stock("XOM", data_xom[:test_start_date_ts], RSITradeAlgorithm())

train_result = manager.train(back_test_start_date)

start_test = datetime.strptime(test_start_date, "%Y-%m-%d")
end_test = datetime.strptime(end_date, "%Y-%m-%d")
dates_test = pd.date_range(start_test, end_test)
for date in dates_test:
    if date in data_wmt[start_test:].index:
        point_wmt = data_wmt.loc[date]
        manager.evaluate_new_point("WMT", point_wmt, date)
    if date in data_jpm[start_test:].index:
        point_jpm = data_jpm.loc[date]
        manager.evaluate_new_point("JPM", point_jpm, date)
    if date in data_xom[start_test:].index:
        point_xom = data_xom.loc[date]
        manager.evaluate_new_point("XOM", point_xom, date)

print(manager.get_trade_results())
print(manager.get_bids_history())
manager.plot_earnings_curve()
manager.plot_stock_history("WMT")
manager.plot_stock_history("JPM")
manager.plot_stock_history("XOM")

