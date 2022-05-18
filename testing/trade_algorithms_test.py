from datetime import datetime
from typing import List

import yfinance as yf
import pandas as pd
from trading.trade_manager import TradeManager
from trading.trade_algorithms.abstract_trade_algorithm import AbstractTradeAlgorithm
from trading.trade_algorithms.one_indicator_trade_algorithms.rsi_trade_algorithm import RSITradeAlgorithm
from trading.trade_algorithms.one_indicator_trade_algorithms.macd_trade_algorithm import MACDTradeAlgorithm
from trading.trade_algorithms.one_indicator_trade_algorithms.super_trend_trade_algorithm import SuperTrendTradeAlgorithm
from trading.trade_algorithms.one_indicator_trade_algorithms.cci_trade_algorithm import CCITrradeAlgorithm
from trading.trade_algorithms.one_indicator_trade_algorithms.bollinger_bands_trade_algorithm import BollingerBandsTradeAlgorithm
from trading.trade_algorithms.one_indicator_trade_algorithms.ma_support_levels_trade_algorithm import MASupportLevelsTradeAlgorithm

start_date = "2000-01-01"
end_date = "2021-12-31"
test_start_date = "2020-01-01"
back_test_start_date = "2018-01-01"
test_start_date_ts = pd.Timestamp(ts_input=test_start_date)
start_test = datetime.strptime(test_start_date, "%Y-%m-%d")
end_test = datetime.strptime(end_date, "%Y-%m-%d")
dates_test = pd.date_range(start_test, end_test)

# Mega corps pack
data_wmt = yf.download("WMT", start=start_date, end=end_date)
data_aapl = yf.download("AAPL", start=start_date, end=end_date)
# data_msft = yf.download("MSFT", start=start_date, end=end_date)
data_jpm = yf.download("JPM", start=start_date, end=end_date)
# data_ko = yf.download("KO", start=start_date, end=end_date)
# data_pg = yf.download("PG", start=start_date, end=end_date)
# data_xom = yf.download("XOM", start=start_date, end=end_date)

# one_indicator_trade_algorithms: List = [RSITradeAlgorithm, MACDTradeAlgorithm, SuperTrendTradeAlgorithm,
#                                         CCITrradeAlgorithm, BollingerBandsTradeAlgorithm, MASupportLevelsTradeAlgorithm]
one_indicator_trade_algorithms: List = [RSITradeAlgorithm]

for indicator_trade_algorithm in one_indicator_trade_algorithms:
    print(indicator_trade_algorithm.get_algorithm_name())
    manager = TradeManager(days_to_keep_limit=10, days_to_chill=5, use_limited_money=True, start_capital=200000, equity_risk_rate=0.04)
    manager.set_tracked_stock("WMT", data_wmt[:test_start_date_ts], indicator_trade_algorithm())
    manager.set_tracked_stock("AAPL", data_aapl[:test_start_date_ts], indicator_trade_algorithm())
    manager.set_tracked_stock("JPM", data_jpm[:test_start_date_ts], indicator_trade_algorithm())
    # manager.set_tracked_stock("XOM", data_xom[:test_start_date_ts], indicator_trade_algorithm())
    train_result_1 = manager.train(back_test_start_date)
    print(manager.get_chosen_params())
    for date in dates_test:
        if date in data_wmt[start_test:].index:
            point_wmt = data_wmt.loc[date]
            manager.evaluate_new_point("WMT", point_wmt, date)
        if date in data_aapl[start_test:].index:
            point_xom = data_aapl.loc[date]
            manager.evaluate_new_point("AAPL", point_xom, date)
        if date in data_jpm[start_test:].index:
            point_jpm = data_jpm.loc[date]
            manager.evaluate_new_point("JPM", point_jpm, date)
        # if date in data_xom[start_test:].index:
        #     point_xom = data_xom.loc[date]
        #     manager.evaluate_new_point("XOM", point_xom, date)

    print(manager.get_equity_info())
    print(manager.get_trade_results())
    print(manager.get_bids_history())
    manager.plot_earnings_curve()
    # manager_1.plot_stock_history("WMT")
    # manager_1.plot_stock_history("JPM")
    manager.plot_stock_history("WMT", plot_algorithm_graph=True)

