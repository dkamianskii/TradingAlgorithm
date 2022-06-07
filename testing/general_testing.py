from datetime import datetime

import pylab as pl
import yfinance as yf
import pandas as pd
import numpy as np
import json
from matplotlib import pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf

from indicators.abstract_indicator import TradeAction
from indicators.atr import ATR
from trading.indicators_decision_tree.ind_tree import IndTree
from trading.trade_algorithms.one_indicator_trade_algorithms.rsi_trade_algorithm import RSITradeAlgorithm
from trading.trade_algorithms.indicators_summary_trade_algorithms.decision_tree_trade_algorithm import \
    DecisionTreeTradeAlgorithm
from trading.trade_algorithms.predictive_trade_algorithms.ffn_trade_algorithm import ModelGridColumns, FFNTradeAlgorithm
from trading.trade_algorithms.predictive_trade_algorithms.lstm_trade_algorithm import LSTMTradeAlgorithm
from trading.trade_algorithms.predictive_trade_algorithms.lstm_diff_trade_algorithm import LSTMDiffTradeAlgorithm
from trading.trade_algorithms.predictive_trade_algorithms.arima_trade_algorithm import ARIMATradeAlgorithm

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import MinMaxScaler

import cufflinks as cf
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from helping.base_enum import BaseEnum

cf.go_offline()

start_date = "2000-01-01"
end_date = "2021-12-31"
test_start_date = "2019-01-01"
back_test_start_date = "2019-01-01"
test_start_date_ts = pd.Timestamp(ts_input=test_start_date)
start_test = datetime.strptime(test_start_date, "%Y-%m-%d")


data = yf.download("MSFT", start=start_date, end=end_date)

p = plot_acf(data["Close"].diff()[1:])
plt.title("MSFT Autocorrelation of Close Price diff")
plt.show()

# h = LSTMDiffTradeAlgorithm.create_hyperparameters_dict()
# h["DATA_NAME"] = "TEST"
# k = LSTMDiffTradeAlgorithm()
# k.train(data[:-2], h)
# k.evaluate_new_point(data.iloc[-2], data.index[-2])
# k.evaluate_new_point(data.iloc[-1], data.index[-1])
# k.plot(img_dir="images")

train_data = data.loc[:start_test]
test_data = data[start_test:]

# m = ARIMATradeAlgorithm()
# h = ARIMATradeAlgorithm.create_hyperparameters_dict(use_refit=True)
# h["DATA_NAME"] = "MSFT"
# m.train(train_data, hyperparameters=h)
# for date, point in test_data.iterrows():
#     m.evaluate_new_point(point, date)
# m.plot(img_dir="images", show_full=False)




