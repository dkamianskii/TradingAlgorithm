from datetime import datetime

import yfinance as yf
import pandas as pd
import numpy as np

import cufflinks as cf
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from helping.base_enum import BaseEnum

cf.go_offline()

start_date = "2021-12-01"
end_date = "2021-12-31"
test_start_date = "2015-01-03"
back_test_start_date = "2019-01-01"
test_start_date_ts = pd.Timestamp(ts_input=test_start_date)
start_test = datetime.strptime(test_start_date, "%Y-%m-%d")
end_test = datetime.strptime("2024-02-01", "%Y-%m-%d")
dates_test = pd.date_range(start_test, end_test)


class TestColumn(BaseEnum):
    A = 1
    B = 2
    C = 3


data = yf.download("XOM", start=start_date, end=end_date)
pp = [TestColumn.A, TestColumn.B, TestColumn.C]
cc = ["E", "F", "S"]
color_map = {TestColumn.A: "red", TestColumn.B: "blue", TestColumn.C: "yellow"}
a = {"date": data.index,
     "E": [pp[i - 1] for i in np.random.randint(0, 4, data.shape[0])],
     "F": [pp[i - 1] for i in np.random.randint(0, 4, data.shape[0])],
     "S": [pp[i - 1] for i in np.random.randint(0, 4, data.shape[0])]}
df = pd.DataFrame(data=a).set_index("date")
print(df)
for_plot = {}
for test in pp:
    test_temp = []
    for c in cc:
        k = df[df[c] == test]
        temp_dates = k.index
        c_test = {"date": temp_dates, "c": [c]*temp_dates.shape[0]}
        test_temp.append(pd.DataFrame(data=c_test))
    for_plot[test] = pd.concat(test_temp)
fig = go.Figure()

for test, plot_data in for_plot.items():
    fig.add_trace(go.Scatter(x=plot_data["date"], y=plot_data["c"], mode="markers", marker=dict(
        color=color_map[test],
        size=15,
        symbol="square"),
                             hoverinfo="name",
                             name=test.name))

fig.show()
