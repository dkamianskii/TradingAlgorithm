from datetime import datetime
from typing import List

import yfinance as yf
import pandas as pd
import numpy as np
from pmdarima.arima import AutoARIMA, ARIMA, auto_arima
from trading.trade_algorithms.predictive_trade_algorithms.arima_trade_algorithm import ARIMATradeAlgorithm

start_date = "2000-01-01"
end_date = "2021-12-31"
test_start_date = "2020-01-01"
back_test_start_date = "2019-01-01"
test_start_date_ts = pd.Timestamp(ts_input=test_start_date)
start_test = datetime.strptime(test_start_date, "%Y-%m-%d")
end_test = datetime.strptime(end_date, "%Y-%m-%d")
dates_test = pd.date_range(start_test, end_test)

data = yf.download("WMT", start=start_date, end=end_date)
train_data = data.loc[:start_test]
test_data = data[start_test:]

alg = ARIMATradeAlgorithm()
alg.train(train_data, ARIMATradeAlgorithm.create_hyperparameters_dict(use_refit=True, fit_size=70, refit_add_size=8))
print(alg.best_model_order)
print(alg.mean_absolute_prediction_error)

for date, point in test_data.iterrows():
    alg.evaluate_new_point(point, date)

alg.plot()


