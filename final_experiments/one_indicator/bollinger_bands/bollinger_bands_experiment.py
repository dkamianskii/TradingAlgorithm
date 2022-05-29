import random
from typing import Optional

from trading.trade_manager import TradeManager

from trading.trade_algorithms.one_indicator_trade_algorithms.bollinger_bands_trade_algorithm import BollingerBandsTradeAlgorithm
from final_experiments import experiment_base as eb
from final_experiments.experiment_base import TradeManagerGrid

random.seed(4466)

best_trade_manager: Optional[TradeManager] = None
best_trade_manager_result = 0
best_trade_manager_params = None

trade_manager_params_used = []
for i in range(eb.random_grid_search_attempts):
    trade_manager_params = {}
    days_to_keep_limit = eb.trade_manager_grid[TradeManagerGrid.DAYS_TO_KEEP_LIMIT][
        random.randint(0, len(eb.trade_manager_grid[TradeManagerGrid.DAYS_TO_KEEP_LIMIT]) - 1)]
    trade_manager_params[TradeManagerGrid.DAYS_TO_KEEP_LIMIT.name] = days_to_keep_limit
    keep_holding_rate = eb.trade_manager_grid[TradeManagerGrid.KEEP_HOLDING_RATE][
        random.randint(0, len(eb.trade_manager_grid[TradeManagerGrid.KEEP_HOLDING_RATE]) - 1)]
    trade_manager_params[TradeManagerGrid.KEEP_HOLDING_RATE.name] = keep_holding_rate
    take_profit_multiplier, active_action_multiplier = \
        eb.trade_manager_grid[TradeManagerGrid.TAKE_PROFIT_ACTIVE_ACTION][
            random.randint(0, len(eb.trade_manager_grid[TradeManagerGrid.TAKE_PROFIT_ACTIVE_ACTION]) - 1)]
    trade_manager_params[TradeManagerGrid.TAKE_PROFIT_ACTIVE_ACTION.name] = (
        take_profit_multiplier, active_action_multiplier)

    use_atr = eb.trade_manager_grid[TradeManagerGrid.USE_ATR][random.randint(0, 1)]
    bid_risk_rate = 0
    if use_atr:
        trade_manager_params[TradeManagerGrid.USE_ATR.name] = True
    else:
        bid_risk_rate = eb.trade_manager_grid[TradeManagerGrid.BID_RISK_RATE][
            random.randint(0, len(eb.trade_manager_grid[TradeManagerGrid.BID_RISK_RATE]) - 1)]
        trade_manager_params[TradeManagerGrid.BID_RISK_RATE.name] = bid_risk_rate

    if trade_manager_params in trade_manager_params_used:
        i -= 1
        continue
    trade_manager_params_used.append(trade_manager_params)
    print("TRADE MANAGER PARAMS")
    print(trade_manager_params)

    manager = TradeManager(days_to_keep_limit=days_to_keep_limit,
                           use_limited_money=True,
                           start_capital=eb.start_capital,
                           bid_risk_rate=bid_risk_rate,
                           take_profit_multiplier=take_profit_multiplier,
                           active_action_multiplier=active_action_multiplier,
                           use_atr=use_atr,
                           keep_holding_rate=keep_holding_rate)
    for company in eb.companies_names:
        manager.set_tracked_stock(company, eb.companies_data[company]["train data"][-900:], BollingerBandsTradeAlgorithm())

    print("TRAINING PHASE")

    train_result = manager.train(eb.back_test_start_date, plot_test=False)

    print("TRADING PHASE")

    for date in eb.dates_test:
        for company in eb.companies_names:
            data = eb.companies_data[company]["trade data"]
            if date in data.index:
                point = data.loc[date]
                manager.evaluate_new_point(company, point, date)

    result = manager.get_equity_info()["account money"]
    print(f"Final account money = {result}")
    if best_trade_manager is None or result > best_trade_manager_result:
        best_trade_manager = manager
        best_trade_manager_result = result
        best_trade_manager_params = trade_manager_params

eb.print_best_manager_results(best_trade_manager, best_trade_manager_params)


