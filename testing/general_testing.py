from datetime import datetime

import yfinance as yf
import pandas as pd
import numpy as np
import json

from indicators.abstract_indicator import TradeAction
from indicators.atr import ATR
from trading.indicators_decision_tree.ind_tree import IndTree
from trading.trade_algorithms.one_indicator_trade_algorithms.rsi_trade_algorithm import RSITradeAlgorithm
from trading.trade_algorithms.indicators_summary_trade_algorithms.decision_tree_trade_algorithm import \
    DecisionTreeTradeAlgorithm
from trading.trade_algorithms.predictive_trade_algorithms.ffn_trade_algorithm import ModelGridColumns, FFNTradeAlgorithm
from trading.trade_algorithms.predictive_trade_algorithms.lstm_trade_algorithm import LSTMTradeAlgorithm

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import MinMaxScaler

import cufflinks as cf
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from helping.base_enum import BaseEnum

cf.go_offline()

start_date = "2008-01-01"
end_date = "2021-12-31"
test_start_date = "2019-01-01"
back_test_start_date = "2019-01-01"
test_start_date_ts = pd.Timestamp(ts_input=test_start_date)
start_test = datetime.strptime(test_start_date, "%Y-%m-%d")

a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
b = [5, 7, 1, 2, 3, 4, 5, 6, 7, 8]

cor_1 = np.corrcoef(a, b)
print(cor_1)
print(a[:-2])
print(b[2:])
cor_2 = np.corrcoef(a[:-2], b[2:])
print(cor_2)

# data = yf.download("WMT", start=start_date, end=end_date)
#
#
# train_data = data.loc[:start_test]
# test_data = data[start_test:]
#
# m = FFNTradeAlgorithm()
# h = FFNTradeAlgorithm.create_hyperparameters_dict()
# h["DATA_NAME"] = "TEST"
# m.train(train_data, hyperparameters=h)
# for date, point in test_data.iterrows():
#     m.evaluate_new_point(point, date)
# m.plot(show_full=True)




