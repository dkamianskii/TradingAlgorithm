from datetime import datetime

import yfinance as yf
import pandas as pd
from trading.trade_manager import TradeManager
from trading.trade_algorithms.predictive_trade_algorithms.lstm_trade_algorithm import LSTMTradeAlgorithm, \
    LSTMTradeAlgorithmHyperparam

img_dir = "images"

start_date = "2000-01-01"
end_date = "2021-12-31"
test_start_date = "2020-01-01"
back_test_start_date = "2019-01-01"
test_start_date_ts = pd.Timestamp(ts_input=test_start_date)
start_test = datetime.strptime(test_start_date, "%Y-%m-%d")
end_test = datetime.strptime(end_date, "%Y-%m-%d")
dates_test = pd.date_range(start_test, end_test)

# Mega corps pack
companies_names = ["WMT", "AAPL", "MSFT", "JPM", "KO", "PG", "XOM"]
companies_data = {}

for company in companies_names[3:4]:
    companies_data[company] = yf.download(company, start=start_date, end=end_date)
# data_wmt = yf.download("WMT", start=start_date, end=end_date)
# data_aapl = yf.download("AAPL", start=start_date, end=end_date)
# data_msft = yf.download("MSFT", start=start_date, end=end_date)
# data_jpm = yf.download("JPM", start=start_date, end=end_date)
# data_ko = yf.download("KO", start=start_date, end=end_date)
# data_pg = yf.download("PG", start=start_date, end=end_date)
# data_xom = yf.download("XOM", start=start_date, end=end_date)

manager = TradeManager(days_to_keep_limit=14, keep_holding_rate=0,
                       take_profit_multiplier=2, active_action_multiplier=1, bid_risk_rate=0.025)

lstm_hyperparams_grid = LSTMTradeAlgorithm.get_default_hyperparameters_grid()
for lstm_hyperparams in lstm_hyperparams_grid:
    lstm_hyperparams[LSTMTradeAlgorithmHyperparam.TAKE_ACTION_BARRIER] = 0.025 * 0.1
    lstm_hyperparams[LSTMTradeAlgorithmHyperparam.ACTIVE_ACTION_MULTIPLIER] = 1

for company in companies_names[3:4]:
    manager.set_tracked_stock(company, companies_data[company][:test_start_date_ts], LSTMTradeAlgorithm(), lstm_hyperparams_grid)

# manager.set_tracked_stock("WMT", data_wmt[:test_start_date_ts], FFNTradeAlgorithm())
# manager.set_tracked_stock("JPM", data_jpm[:test_start_date_ts], FFNTradeAlgorithm())
# manager.set_tracked_stock("XOM", data_xom[:test_start_date_ts], FFNTradeAlgorithm())

train_result = manager.train(back_test_start_date, plot_test=False)
print(manager.get_chosen_params())

for date in dates_test:
    for company in companies_names[3:4]:
        data = companies_data[company]
        if date in data[start_test:].index:
            point = data.loc[date]
            manager.evaluate_new_point(company, point, date)
    # if date in data_wmt[start_test:].index:
    #     point_wmt = data_wmt.loc[date]
    #     manager.evaluate_new_point("WMT", point_wmt, date)
    # if date in data_jpm[start_test:].index:
    #     point_jpm = data_jpm.loc[date]
    #     manager.evaluate_new_point("JPM", point_jpm, date)
    # if date in data_xom[start_test:].index:
    #     point_xom = data_xom.loc[date]
    #     manager.evaluate_new_point("XOM", point_xom, date)

print(manager.get_trade_results())
# print(manager.get_bids_history())
manager.plot_earnings_curve(img_dir)
for company in companies_names[3:4]:
    manager.plot_stock_history(company, img_dir, plot_algorithm_graph=True)
    manager.plot_stock_history(company, img_dir, plot_algorithm_graph=True, plot_algorithm_graph_full=True)
# manager.plot_stock_history("WMT", plot_algorithm_graph=True)
# manager.plot_stock_history("WMT", plot_algorithm_graph=True, plot_algorithm_graph_full=True)
# manager.plot_stock_history("JPM", plot_algorithm_graph=True)
# manager.plot_stock_history("JPM", plot_algorithm_graph=True, plot_algorithm_graph_full=True)
# manager.plot_stock_history("XOM", plot_algorithm_graph=True)
# manager.plot_stock_history("XOM", plot_algorithm_graph=True, plot_algorithm_graph_full=True)
