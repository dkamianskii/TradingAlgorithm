from datetime import datetime

import yfinance as yf
import pandas as pd

from helping.base_enum import BaseEnum
from trading.trade_manager import TradeManager
from trading.trade_statistics_manager_enums import TradeResultColumn


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


def print_best_manager_results(best_trade_manager: TradeManager, best_trade_manager_params: dict):
    print("BEST TRADE MANAGER PARAMS")
    print(best_trade_manager_params)

    print("CHOSEN PARAMS")
    print(best_trade_manager.get_chosen_params())

    trade_results = best_trade_manager.get_trade_results()
    tr_total = trade_results.loc[TradeResultColumn.TOTAL]

    print(trade_results)

    win_rate = tr_total[TradeResultColumn.WINS] / (
                tr_total[TradeResultColumn.WINS] + tr_total[TradeResultColumn.LOSES] +
                tr_total[TradeResultColumn.DRAWS])
    lose_rate = tr_total[TradeResultColumn.LOSES] / (
                tr_total[TradeResultColumn.WINS] + tr_total[TradeResultColumn.LOSES] +
                tr_total[TradeResultColumn.DRAWS])

    print(f"WIN RATE = {win_rate}")
    print(f"LOSE RATE = {lose_rate}")

    print(f"RETURN ON START CAPITAL = {best_trade_manager.get_traiding_gain()}")

    print(f"MAX DROWDOWN = {best_trade_manager.get_max_drowdown()}")

    print(f"AVG WIN / AVG LOSE = {best_trade_manager.get_avg_win_to_avg_lose()}")

    print(f"Sharpe ratio = {best_trade_manager.get_sharpe_ratio()}")
    # print(f"Sortino ratio = {best_trade_manager.get_sortino_ratio()}")
    print(f"Calmar ratio = {best_trade_manager.get_calmar_ratio()}")

    best_trade_manager.plot_earnings_curve(img_dir=img_dir)

    for company in companies_names:
        best_trade_manager.plot_stock_history(company, plot_algorithm_graph=True, img_dir=img_dir)
        best_trade_manager.plot_stock_history(company, plot_algorithm_graph=True, plot_algorithm_graph_full=True,
                                              img_dir=img_dir)

    print("With respect to 2020 corona crisis")

    trade_results = best_trade_manager.get_trade_results(ignore_crisis=False)
    tr_total = trade_results.loc[TradeResultColumn.TOTAL]

    print(trade_results)

    win_rate = tr_total[TradeResultColumn.WINS] / (
                tr_total[TradeResultColumn.WINS] + tr_total[TradeResultColumn.LOSES] +
                tr_total[TradeResultColumn.DRAWS])
    lose_rate = tr_total[TradeResultColumn.LOSES] / (
                tr_total[TradeResultColumn.WINS] + tr_total[TradeResultColumn.LOSES] +
                tr_total[TradeResultColumn.DRAWS])

    print(f"WIN RATE = {win_rate}")
    print(f"LOSE RATE = {lose_rate}")

    print(f"RETURN ON START CAPITAL = {best_trade_manager.get_traiding_gain(ignore_crisis=False)}")

    print(f"MAX DROWDOWN = {best_trade_manager.get_max_drowdown(ignore_crisis=False)}")

    print(f"AVG WIN / AVG LOSE = {best_trade_manager.get_avg_win_to_avg_lose(ignore_crisis=False)}")

    print(f"Sharpe ratio = {best_trade_manager.get_sharpe_ratio(ignore_crisis=False)}")
    # print(f"Sortino ratio = {best_trade_manager.get_sortino_ratio(ignore_crisis=False)}")
    print(f"Calmar ratio = {best_trade_manager.get_calmar_ratio(ignore_crisis=False)}")

    best_trade_manager.plot_earnings_curve(img_dir=img_dir, ignore_crisis=False)
