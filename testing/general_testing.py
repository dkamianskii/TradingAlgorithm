from datetime import datetime

import yfinance as yf
import pandas as pd
import numpy as np

from indicators.abstract_indicator import TradeAction
from indicators.atr import ATR
from trading.indicators_decision_tree.ind_tree import IndTree
from trading.trade_algorithms.one_indicator_trade_algorithms.rsi_trade_algorithm import RSITradeAlgorithm
from trading.trade_algorithms.indicators_summary_trade_algorithms.decision_tree_trade_algorithm import \
    DecisionTreeTradeAlgorithm

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

import cufflinks as cf
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from helping.base_enum import BaseEnum

cf.go_offline()

start_date = "2021-06-01"
end_date = "2021-12-31"
test_start_date = "2015-01-03"
back_test_start_date = "2019-01-01"
test_start_date_ts = pd.Timestamp(ts_input=test_start_date)
start_test = datetime.strptime(test_start_date, "%Y-%m-%d")
end_test = datetime.strptime("2024-02-01", "%Y-%m-%d")
dates_test = pd.date_range(start_test, end_test)

data = yf.download("XOM", start=start_date, end=end_date)

e = (0, 1, 4)
print(e[0])

# df = pd.DataFrame(data={"Date": data.index})
# df = df.set_index("Date")
# df["A"] = data["Close"]
# df["B"] = data[10:]["Open"]
# print(df)
# df = df.dropna()
# print(df)
# df = pd.DataFrame()
# df["A"] = np.random.randint(1, 101, 1000)
# df["B"] = np.random.randint(1, 101, 1000)
# df["C"] = np.random.randint(1, 101, 1000)
# tr = TradeAction.get_elements_list()
# y = [str(tr[i]) for i in np.random.randint(0, 5, 1000)]
#
# clf = RandomForestClassifier(max_depth=2, random_state=0)
# clf.fit(df, y)
# p = clf.predict(pd.DataFrame({"A":[15],"B": [63],"C": [81]}))
# print(p)

# dstree = DecisionTreeTradeAlgorithm()
# hyperparams = dstree.get_default_hyperparameters_grid()
# dstree.train(data=data, hyperparameters=hyperparams[0])
# dstree.plot()

# p = {"A": [TradeAction.BUY, TradeAction.NONE, TradeAction.NONE, TradeAction.SELL, TradeAction.BUY,
#            TradeAction.NONE, TradeAction.ACTIVELY_BUY, TradeAction.ACTIVELY_SELL, TradeAction.NONE,
#            TradeAction.SELL, TradeAction.NONE, TradeAction.BUY, TradeAction.NONE, TradeAction.SELL,
#            TradeAction.ACTIVELY_SELL, TradeAction.NONE, TradeAction.NONE, TradeAction.ACTIVELY_SELL,
#            TradeAction.SELL, TradeAction.ACTIVELY_SELL, TradeAction.BUY, TradeAction.NONE, TradeAction.NONE,
#            TradeAction.SELL, TradeAction.NONE, TradeAction.NONE, TradeAction.ACTIVELY_BUY,
#            TradeAction.ACTIVELY_BUY, TradeAction.NONE, TradeAction.NONE, TradeAction.NONE],
#      "B": [TradeAction.NONE, TradeAction.NONE, TradeAction.NONE, TradeAction.SELL, TradeAction.BUY,
#            TradeAction.NONE, TradeAction.ACTIVELY_SELL, TradeAction.SELL, TradeAction.NONE,
#            TradeAction.NONE, TradeAction.NONE, TradeAction.SELL, TradeAction.NONE, TradeAction.BUY,
#            TradeAction.ACTIVELY_SELL, TradeAction.NONE, TradeAction.BUY, TradeAction.ACTIVELY_SELL,
#            TradeAction.SELL, TradeAction.BUY, TradeAction.BUY, TradeAction.NONE, TradeAction.NONE,
#            TradeAction.SELL, TradeAction.NONE, TradeAction.NONE, TradeAction.ACTIVELY_BUY,
#            TradeAction.ACTIVELY_SELL, TradeAction.SELL, TradeAction.NONE, TradeAction.BUY],
#      "C": [TradeAction.ACTIVELY_SELL, TradeAction.SELL, TradeAction.NONE, TradeAction.NONE, TradeAction.NONE,
#            TradeAction.NONE, TradeAction.ACTIVELY_BUY, TradeAction.BUY, TradeAction.NONE,
#            TradeAction.NONE, TradeAction.BUY, TradeAction.NONE, TradeAction.NONE, TradeAction.BUY,
#            TradeAction.ACTIVELY_SELL, TradeAction.NONE, TradeAction.ACTIVELY_BUY, TradeAction.NONE,
#            TradeAction.SELL, TradeAction.NONE, TradeAction.NONE, TradeAction.SELL, TradeAction.NONE,
#            TradeAction.SELL, TradeAction.ACTIVELY_SELL, TradeAction.NONE, TradeAction.NONE,
#            TradeAction.NONE, TradeAction.ACTIVELY_SELL, TradeAction.SELL, TradeAction.NONE],
#      "label": [TradeAction.NONE, TradeAction.SELL, TradeAction.NONE, TradeAction.NONE, TradeAction.BUY,
#                TradeAction.ACTIVELY_BUY, TradeAction.NONE, TradeAction.NONE, TradeAction.NONE,
#                TradeAction.NONE, TradeAction.SELL, TradeAction.ACTIVELY_SELL, TradeAction.NONE, TradeAction.NONE,
#                TradeAction.SELL, TradeAction.NONE, TradeAction.BUY, TradeAction.ACTIVELY_BUY,
#                TradeAction.SELL, TradeAction.SELL, TradeAction.NONE, TradeAction.BUY, TradeAction.NONE,
#                TradeAction.NONE, TradeAction.ACTIVELY_BUY, TradeAction.NONE, TradeAction.BUY,
#                TradeAction.NONE, TradeAction.SELL, TradeAction.SELL, TradeAction.NONE]
#      }
#
# df = pd.DataFrame(data=p)
# aa = ["A", "B", "C"]
# cum_bool = None
# for a in aa:
#     a_bool = df[a] != TradeAction.NONE
#     if cum_bool is None:
#         cum_bool = a_bool
#     else:
#         cum_bool = np.logical_or(cum_bool, a_bool)
#
#
# df = df[cum_bool]
#
# tree = IndTree(data=df, indicators=["A", "B", "C"])
# tree.print_tree()
# for i in range(10):
#     a = tree.get_trade_action(pd.Series(data={"A": TradeAction.BUY, "B": TradeAction.BUY, "C": TradeAction.NONE}))
#     print(a)
