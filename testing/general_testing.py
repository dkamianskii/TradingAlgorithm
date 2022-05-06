from datetime import datetime

import yfinance as yf
import pandas as pd
import numpy as np

start_date = "2015-01-01"
end_date = "2021-12-31"
test_start_date = "2015-01-03"
back_test_start_date = "2019-01-01"
test_start_date_ts = pd.Timestamp(ts_input=test_start_date)
start_test = datetime.strptime(test_start_date, "%Y-%m-%d")
end_test = datetime.strptime("2024-02-01", "%Y-%m-%d")
dates_test = pd.date_range(start_test, end_test)

data = yf.download("XOM", start=start_date, end=end_date)

print(data.index[:30])
selected_data = data["Close"][pd.Timestamp(start_test):pd.Timestamp(end_test)]
print(selected_data)
