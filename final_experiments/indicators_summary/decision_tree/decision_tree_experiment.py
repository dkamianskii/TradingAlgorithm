import random
from typing import Optional

from trading.trade_manager import TradeManager

from trading.trade_algorithms.indicators_summary_trade_algorithms.decision_tree_trade_algorithm import DecisionTreeTradeAlgorithm, DecisionTreeTradeAlgorithmHyperparam, RiskManagerHyperparam
from final_experiments import experiment_base as eb
from final_experiments.experiment_base import TradeManagerGrid

random.seed(7744)

best_trade_manager: Optional[TradeManager] = None
best_trade_manager_result = 0
best_trade_manager_params = None

trade_manager_params_used = []
for i in range(eb.random_grid_search_attempts - 2):
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

    ds_tree_hyperparams_grid = DecisionTreeTradeAlgorithm.get_default_hyperparameters_grid()
    for ds_tree_hyperparams in ds_tree_hyperparams_grid:
        ds_tree_hyperparams[DecisionTreeTradeAlgorithmHyperparam.DAYS_TO_KEEP_LIMIT] = days_to_keep_limit
        rs_params = {RiskManagerHyperparam.BID_RISK_RATE: bid_risk_rate,
                     RiskManagerHyperparam.USE_ATR: use_atr,
                     RiskManagerHyperparam.ACTIVE_ACTION_MULTIPLIER: active_action_multiplier,
                     RiskManagerHyperparam.TAKE_PROFIT_MULTIPLIER: take_profit_multiplier}
        ds_tree_hyperparams[DecisionTreeTradeAlgorithmHyperparam.RISK_MANAGER_HYPERPARAMS] = rs_params

    manager = TradeManager(days_to_keep_limit=days_to_keep_limit,
                           use_limited_money=True,
                           start_capital=eb.start_capital,
                           bid_risk_rate=bid_risk_rate,
                           take_profit_multiplier=take_profit_multiplier,
                           active_action_multiplier=active_action_multiplier,
                           use_atr=use_atr,
                           keep_holding_rate=keep_holding_rate)
    for company in eb.companies_names:
        manager.set_tracked_stock(company, eb.companies_data[company]["train data"],
                                  DecisionTreeTradeAlgorithm(), ds_tree_hyperparams_grid)

    print("TRAINING PHASE")

    train_result = manager.train(eb.back_test_start_date, plot_test=False)

    print(f"Final train result = {train_result}")
    if best_trade_manager is None or train_result > best_trade_manager_result:
        best_trade_manager = manager
        best_trade_manager_result = train_result
        best_trade_manager_params = trade_manager_params

print("TRADING PHASE")

for date in eb.dates_test:
    for company in eb.companies_names:
        data = eb.companies_data[company]["trade data"]
        if date in data.index:
            point = data.loc[date]
            best_trade_manager.evaluate_new_point(company, point, date)

eb.print_best_manager_results(best_trade_manager, best_trade_manager_params)