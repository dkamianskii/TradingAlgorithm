from datetime import datetime

import yfinance as yf
import pandas as pd

from helping.base_enum import BaseEnum


class TradeManagerGrid(BaseEnum):
    DAYS_TO_KEEP_LIMIT = 1
    USE_ATR = 2
    BID_RISK_RATE = 3
    TAKE_PROFIT_ACTIVE_ACTION = 4
    KEEP_HOLDING_RATE = 5


img_dir = "images"

start_date = "2000-01-01"
end_date = "2021-12-31"
test_start_date = "2020-01-01"
back_test_start_date = "2019-01-01"
test_start_date_ts = pd.Timestamp(ts_input=test_start_date)
start_test = datetime.strptime(test_start_date, "%Y-%m-%d")
end_test = datetime.strptime(end_date, "%Y-%m-%d")
dates_test = pd.date_range(start_test, end_test)

start_capital = 200000

trade_manager_grid = {TradeManagerGrid.DAYS_TO_KEEP_LIMIT: [7, 14],
                      TradeManagerGrid.USE_ATR: [True, False],
                      TradeManagerGrid.BID_RISK_RATE: [0.025, 0.05, 0.075],
                      TradeManagerGrid.TAKE_PROFIT_ACTIVE_ACTION: [(2, 1), (2, 1.5), (1.5, 2)],
                      TradeManagerGrid.KEEP_HOLDING_RATE: [0, 0.25, 0.5]}

random_grid_search_attempts = 12

companies_names = ["WMT", "AAPL", "MSFT", "JPM", "KO", "PG", "XOM"]
companies_data = {}

for company in companies_names:
    full_data = yf.download(company, start=start_date, end=end_date)
    train_data = full_data[:test_start_date_ts]
    trade_data = full_data[test_start_date_ts:]
    companies_data[company] = {"full data": full_data,
                               "train data": train_data,
                               "trade data": trade_data}
