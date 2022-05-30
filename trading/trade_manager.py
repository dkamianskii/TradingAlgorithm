import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Union, Tuple

from indicators.abstract_indicator import TradeAction
from indicators.atr import ATR, ATR_one_point
from trading.trade_statistics_manager_enums import *
from trading.trade_manager_enums import *
from trading.trade_algorithms.abstract_trade_algorithm import AbstractTradeAlgorithm
from trading.trade_algorithms.indicators_summary_trade_algorithms.macd_super_trend_trade_algorithm import \
    MACDSuperTrendTradeAlgorithm
from trading.trade_statistics_manager import TradeStatisticsManager
from trading.risk_manager import RiskManager

import cufflinks as cf
import plotly.graph_objects as go

cf.go_offline()


class TradeManager:

    def __init__(self, days_to_keep_limit: int = 14,
                 days_to_chill: int = 5,
                 use_limited_money: bool = False,
                 money_for_a_bid: float = 10000,
                 start_capital: float = 0,
                 equity_risk_rate: float = 0.025,
                 bid_risk_rate: float = 0.03,
                 take_profit_multiplier: float = 2,
                 active_action_multiplier: float = 1.5,
                 use_atr: bool = False,
                 atr_period: int = 14,
                 keep_holding_rate: float = 0.5):
        """
        Control:
         - stock market data,
         - currently managed assets (open long and short bids) and their limits
        Init:
        Set tracked stocks with their primary stock data and trade algorithms with params grid for each
        BackTest:
         calculate efficiency of TradeAlgorithm for provided primary data
         - receive starting date for back test to separate train data from test data
         procedure of back test mainly runs as main work pipeline
        Start:
         - train TradeAlgorithm on full primary data
         - switch manager to the working pipeline mode
        Pipeline:
         - receive new day data
         - watch at currently managed assets:
            * if a bid reaches its stop loss or take profit level manager handles that event and calculate result of the deal
         - provide a new day data to the TradeAlgorithm and wait for information about what actions should be made
         - if there is any active action it operates selling or buying of currently managed mentioned assets
           then open corresponding long or short bids, evaluate stop loss and take profit for them and put them into the
           currently managed assets
         - save information about profitability of made deals and current finance assets
        """
        self.portfolio: pd.DataFrame = pd.DataFrame(columns=PortfolioColumn.get_elements_list())
        self._last_bid_index: int = 0
        self.portfolio[PortfolioColumn.DATE] = pd.to_datetime(self.portfolio[PortfolioColumn.DATE])
        self._tracked_stocks: Dict[str, Dict] = {}
        self._train_results: float = 0

        self._atr_period: int = atr_period
        self._days_to_keep_limit: pd.Timedelta = pd.Timedelta(days=days_to_keep_limit)
        self._days_to_chill: pd.Timedelta = pd.Timedelta(days=days_to_chill)
        self._keep_holding_rate = keep_holding_rate

        self._statistics_manager: TradeStatisticsManager = TradeStatisticsManager()
        self._risk_manager: RiskManager = RiskManager(use_limited_money=use_limited_money,
                                                      money_for_a_bid=money_for_a_bid,
                                                      start_capital=start_capital,
                                                      equity_risk_rate=equity_risk_rate,
                                                      bid_risk_rate=bid_risk_rate,
                                                      take_profit_multiplier=take_profit_multiplier,
                                                      active_action_multiplier=active_action_multiplier,
                                                      use_atr=use_atr)

    def get_tracked_stocks(self) -> List[str]:
        *stock_names, = self._tracked_stocks.keys()
        return stock_names

    def get_train_results(self, stock_name: str) -> List[Tuple[str, pd.DataFrame]]:
        return [(params, result) for params, result in
                self._train_results[stock_name].items()]

    def get_chosen_params(self) -> List[Tuple[str, dict]]:
        return [(stock_name, stock[TrackedStocksColumn.CHOSEN_PARAMS]) for stock_name, stock in
                self._tracked_stocks.items()]

    def get_trade_results(self, ignore_crisis: bool = True) -> pd.DataFrame:
        return self._statistics_manager.get_trade_results(ignore_crisis)

    def get_traiding_gain(self, ignore_crisis: bool = True):
        trade_results = self._statistics_manager.get_trade_results(ignore_crisis)
        return trade_results.loc[TradeResultColumn.TOTAL][TradeResultColumn.EARNED_PROFIT] / self._risk_manager.start_capital

    def get_max_drowdown(self, ignore_crisis: bool = True):
        earnings_history = self._statistics_manager.get_earnings_history(ignore_crisis=ignore_crisis)
        return self._risk_manager.evaluate_max_drowdown(earnings_history)

    def get_bids_history(self, stock_name: Optional[str] = None):
        return self._statistics_manager.get_bids_history(stock_name)

    def get_portfolio(self):
        return self.portfolio

    def get_equity_info(self):
        return {"start capital": self._risk_manager.start_capital,
                "account money": self._risk_manager.account_money,
                "available money": self._risk_manager.available_money}

    def get_sharpe_ratio(self, ignore_crisis: bool = True):
        earnings_history = self._statistics_manager.get_earnings_history(ignore_crisis=ignore_crisis)
        return self._risk_manager.evaluate_sharpe_ratio(earnings_history)

    def get_sortino_ratio(self, ignore_crisis: bool = True):
        earnings_history = self._statistics_manager.get_earnings_history(ignore_crisis=ignore_crisis)
        return self._risk_manager.evaluate_sortino_ratio(earnings_history)

    def get_calmar_ratio(self, ignore_crisis: bool = True):
        earnings_history = self._statistics_manager.get_earnings_history(ignore_crisis=ignore_crisis)
        return self._risk_manager.evaluate_calmar_ratio(earnings_history)

    def get_avg_win_to_avg_lose(self, ignore_crisis: bool = True):
        earnings_history = self._statistics_manager.get_earnings_history(ignore_crisis=ignore_crisis)
        avg_win = earnings_history[earnings_history[EarningsHistoryColumn.VALUE] > 0][
            EarningsHistoryColumn.VALUE].mean()
        avg_lose = -earnings_history[earnings_history[EarningsHistoryColumn.VALUE] < 0][
            EarningsHistoryColumn.VALUE].mean()
        return avg_win / avg_lose

    def clear_history(self):
        self.portfolio = pd.DataFrame(columns=PortfolioColumn.get_elements_list())
        self._last_bid_index = 0
        self.portfolio[PortfolioColumn.DATE] = pd.to_datetime(self.portfolio[PortfolioColumn.DATE])
        self._statistics_manager.clear_history()
        self._risk_manager.reset_money()
        self._train_results = 0
        for _, stock in self._tracked_stocks.items():
            stock[TrackedStocksColumn.TRADING_START_DATE] = None

    def clear_tracked_stocks_list(self):
        self._tracked_stocks = {}
        self._statistics_manager = TradeStatisticsManager()
        self.clear_history()

    def set_manager_params(self, days_to_keep_limit: int = 14,
                           days_to_chill: int = 5,
                           use_limited_money: bool = False,
                           money_for_a_bid: float = 10000,
                           start_capital: float = 0,
                           equity_risk_rate: float = 0.025,
                           bid_risk_rate: float = 0.03,
                           take_profit_multiplier: float = 2,
                           active_action_multiplier: float = 1.5,
                           use_atr: bool = False,
                           atr_period: int = 14,
                           keep_holding_rate: float = 0.5):
        self._days_to_keep_limit: pd.Timedelta = pd.Timedelta(days=days_to_keep_limit)
        self._days_to_chill: pd.Timedelta = pd.Timedelta(days=days_to_chill)
        self._atr_period: int = atr_period
        self._keep_holding_rate = keep_holding_rate
        self._risk_manager.set_manager_params(use_limited_money=use_limited_money,
                                              money_for_a_bid=money_for_a_bid,
                                              start_capital=start_capital,
                                              equity_risk_rate=equity_risk_rate,
                                              bid_risk_rate=bid_risk_rate,
                                              take_profit_multiplier=take_profit_multiplier,
                                              active_action_multiplier=active_action_multiplier,
                                              use_atr=use_atr)

    # def set_manager_params_dict(self, params_dict):
    #     for param, value in params_dict.items():
    #         if param not in self.__dict__:
    #             raise ValueError("Uknown parameter was provided")
    #         setattr(self, "_" + param, value)

    def set_tracked_stock(self, stock_name: str,
                          stock_data: pd.DataFrame,
                          trade_algorithm: Union[DefaultTradeAlgorithmType, AbstractTradeAlgorithm],
                          custom_params_grid: Optional[List[Dict]] = None):
        if type(trade_algorithm) is DefaultTradeAlgorithmType:
            if trade_algorithm == DefaultTradeAlgorithmType.MACD_SuperTrend:
                algorithm = MACDSuperTrendTradeAlgorithm()
            else:
                raise ValueError(
                    "algorithm must be one of default trade algorithms or user should provide custom trade algorithm")
        else:
            algorithm = trade_algorithm

        self._statistics_manager.set_tracked_stock(stock_name)
        self._tracked_stocks[stock_name] = {
            TrackedStocksColumn.DATA: stock_data.copy(),
            TrackedStocksColumn.TRADE_ALGORITHM: algorithm,
            TrackedStocksColumn.PARAMS_GRID: custom_params_grid,
            TrackedStocksColumn.CHOSEN_PARAMS: None,
            TrackedStocksColumn.TRADING_START_DATE: None,
            TrackedStocksColumn.ATR: ATR(stock_data, self._atr_period)}

    def __add_to_portfolio(self, stock_name: str,
                           new_point: pd.Series,
                           date: pd.Timestamp,
                           amount: int,
                           action: TradeAction,
                           bid_type: BidType,
                           prolongation: bool,
                           forced_stop_loss_lvl: float = 0):
        atr: pd.Series = self._tracked_stocks[stock_name][TrackedStocksColumn.ATR]
        if date < atr.index[0]:
            curr_atr = atr[0]
        else:
            curr_atr = atr.loc[date]
        if prolongation:
            _, take_profit_lvl = self._risk_manager.evaluate_stop_loss_and_take_profit(new_point, action, curr_atr)
            stop_loss_lvl = forced_stop_loss_lvl
        else:
            stop_loss_lvl, take_profit_lvl = self._risk_manager.evaluate_stop_loss_and_take_profit(new_point, action,
                                                                                                   curr_atr)
        self._risk_manager.set_money_for_bid(new_point["Close"] * amount)
        self.portfolio.loc[self._last_bid_index] = {PortfolioColumn.STOCK_NAME: stock_name,
                                                    PortfolioColumn.PRICE: new_point["Close"],
                                                    PortfolioColumn.TYPE: bid_type,
                                                    PortfolioColumn.TRADE_ACTION: action,
                                                    PortfolioColumn.AMOUNT: amount,
                                                    PortfolioColumn.TAKE_PROFIT_LEVEL: take_profit_lvl,
                                                    PortfolioColumn.STOP_LOSS_LEVEL: stop_loss_lvl,
                                                    PortfolioColumn.DATE: date}
        self._last_bid_index += 1
        self._statistics_manager.open_bid(stock_name, date, new_point["Close"], bid_type, take_profit_lvl,
                                          stop_loss_lvl, amount, action, prolongation)

    def __close_bid(self, stock_name: str, close_price: float,
                    open_date: pd.Timestamp, close_date: pd.Timestamp,
                    cashback: float, profit: float, result: BidResult):
        self._statistics_manager.update_trade_result(stock_name, profit, result)
        self._statistics_manager.close_bid(stock_name, open_date, close_date, close_price, result)
        self._statistics_manager.add_earnings(stock_name, profit, close_date)
        self._risk_manager.set_bid_returns(cashback, profit)

    def __prolongation(self, assets_for_prolongation: List, new_point: pd.Series, date: pd.Timestamp):
        for asset in assets_for_prolongation:
            if asset[PortfolioColumn.TYPE] == BidType.LONG:
                action = TradeAction.BUY
            else:
                action = TradeAction.SELL
            self.__add_to_portfolio(asset[PortfolioColumn.STOCK_NAME],
                                    new_point,
                                    date,
                                    int(asset[PortfolioColumn.AMOUNT] * self._keep_holding_rate),
                                    action,
                                    asset[PortfolioColumn.TYPE],
                                    prolongation=True,
                                    forced_stop_loss_lvl=(new_point["Close"] + asset[PortfolioColumn.PRICE]) / 2)

    def __manage_portfolio_assets(self, stock_name: str, new_point: pd.Series,
                                  date: pd.Timestamp, new_action: TradeAction):
        indexes_to_drop = []
        assets_for_prolongation = []
        assets_in_portfolio = self.portfolio[self.portfolio[PortfolioColumn.STOCK_NAME] == stock_name]
        for index, asset in assets_in_portfolio.iterrows():
            close_flag = False
            price_diff = 0
            cashback = 0
            if asset[PortfolioColumn.TYPE] == BidType.LONG:
                if (new_point["Close"] >= asset[PortfolioColumn.TAKE_PROFIT_LEVEL]) or (
                        new_point["Close"] <= asset[PortfolioColumn.STOP_LOSS_LEVEL]) or (
                        (date - asset[PortfolioColumn.DATE]) > self._days_to_keep_limit) or (
                        (new_action == TradeAction.ACTIVELY_SELL) or (new_action == TradeAction.SELL)):
                    close_flag = True
                    price_diff = new_point["Close"] - asset[PortfolioColumn.PRICE]
                    cashback = new_point["Close"] * asset[PortfolioColumn.AMOUNT]
                    if (new_point["Close"] >= asset[PortfolioColumn.TAKE_PROFIT_LEVEL]) and (
                            (new_action != TradeAction.ACTIVELY_SELL) and (new_action != TradeAction.SELL)):
                        assets_for_prolongation.append(asset)
            else:
                if (new_point["Close"] <= asset[PortfolioColumn.TAKE_PROFIT_LEVEL]) or (
                        new_point["Close"] >= asset[PortfolioColumn.STOP_LOSS_LEVEL]) or (
                        (date - asset[PortfolioColumn.DATE]) > self._days_to_keep_limit) or (
                        (new_action == TradeAction.ACTIVELY_BUY) or (new_action == TradeAction.BUY)):
                    close_flag = True
                    price_diff = asset[PortfolioColumn.PRICE] - new_point["Close"]
                    cashback = (asset[PortfolioColumn.PRICE] + price_diff) * asset[PortfolioColumn.AMOUNT]
                    if (new_point["Close"] <= asset[PortfolioColumn.TAKE_PROFIT_LEVEL]) and (
                            (new_action != TradeAction.ACTIVELY_BUY) and (new_action != TradeAction.BUY)):
                        assets_for_prolongation.append(asset)
            if close_flag:
                if (date - asset[PortfolioColumn.DATE]) > self._days_to_keep_limit:
                    result = BidResult.DRAW
                elif price_diff > 0:
                    result = BidResult.WIN
                else:
                    result = BidResult.LOSE
                self.__close_bid(stock_name,
                                 new_point["Close"],
                                 asset[PortfolioColumn.DATE],
                                 date,
                                 cashback,
                                 price_diff * asset[PortfolioColumn.AMOUNT],
                                 result)
                indexes_to_drop.append(index)
        if len(indexes_to_drop) > 0:
            self.__prolongation(assets_for_prolongation, new_point, date)
            self.portfolio.drop(labels=indexes_to_drop, inplace=True)
        else:
            self._statistics_manager.add_earnings(stock_name, 0, date)

    def evaluate_new_point(self, stock_name: str,
                           new_point: pd.Series,
                           date: Union[str, pd.Timestamp],
                           add_point_to_the_data: bool = True):
        date = pd.Timestamp(ts_input=date)

        stock = self._tracked_stocks[stock_name]

        atr = ATR_one_point(stock[TrackedStocksColumn.ATR][-1],
                            stock[TrackedStocksColumn.DATA]["Close"][-1],
                            new_point,
                            self._atr_period)
        self._tracked_stocks[stock_name][TrackedStocksColumn.ATR].loc[date] = atr
        action = stock[TrackedStocksColumn.TRADE_ALGORITHM].evaluate_new_point(new_point, date)
        self.__manage_portfolio_assets(stock_name, new_point, date, action)
        if action != TradeAction.NONE:
            if (action == TradeAction.BUY) or (action == TradeAction.ACTIVELY_BUY):
                bid_type = BidType.LONG
            else:
                bid_type = BidType.SHORT

            assets_in_portfolio = self.portfolio[self.portfolio[PortfolioColumn.STOCK_NAME] == stock_name]
            last_bid_date = assets_in_portfolio[assets_in_portfolio[PortfolioColumn.TYPE] == bid_type][
                PortfolioColumn.DATE].max()
            if not ((last_bid_date is not pd.NaT) and (date - last_bid_date <= self._days_to_chill)):
                amount = self._risk_manager.evaluate_shares_amount_to_bid(new_point["Close"])
                if amount == 0:
                    pass
                    # print(f"Not enough money to purchase {stock_name} at {date} by price {new_point['Close']}")
                else:
                    self.__add_to_portfolio(stock_name, new_point, date, amount, action, bid_type, prolongation=False)
        if add_point_to_the_data:
            stock[TrackedStocksColumn.DATA].loc[date] = new_point
            if stock[TrackedStocksColumn.TRADING_START_DATE] is None:
                stock[TrackedStocksColumn.TRADING_START_DATE] = date

    def train(self, test_start_date: Union[str, pd.Timestamp],
              test_end_date: Optional[Union[str, pd.Timestamp]] = None,
              plot_test: bool = False) -> float:
        self._train_results = 0
        test_start_date = pd.Timestamp(ts_input=test_start_date)
        for stock_name, stock in self._tracked_stocks.items():
            train_data: pd.DataFrame = stock[TrackedStocksColumn.DATA][:test_start_date]
            if test_end_date is not None:
                test_end_date = pd.Timestamp(ts_input=test_end_date)
                test_data: pd.DataFrame = stock[TrackedStocksColumn.DATA][test_start_date:test_end_date]
            else:
                test_data: pd.DataFrame = stock[TrackedStocksColumn.DATA][test_start_date:]

            algorithm: AbstractTradeAlgorithm = stock[TrackedStocksColumn.TRADE_ALGORITHM]
            if stock[TrackedStocksColumn.PARAMS_GRID] is None:
                params_grid = algorithm.get_default_hyperparameters_grid()
            else:
                params_grid = stock[TrackedStocksColumn.PARAMS_GRID]

            best_params = None
            max_earnings = None
            print(f"Train for {stock_name}")
            for params in params_grid:
                params["DATA_NAME"] = stock_name
                print(params)
                algorithm.train(train_data, params)
                for date, point in test_data.iterrows():
                    self.evaluate_new_point(stock_name, point, date, False)
                earnings = self._statistics_manager.trade_result.at[TradeResultColumn.TOTAL,
                                                                    TradeResultColumn.EARNED_PROFIT]
                print(f"Earnings = {earnings}")
                if (max_earnings is None) or (earnings > max_earnings):
                    max_earnings = earnings
                    self._train_results += earnings
                    best_params = params
                if plot_test:
                    algorithm.plot(test_start_date, test_end_date)
                self.clear_history()
            stock[TrackedStocksColumn.CHOSEN_PARAMS] = best_params
            algorithm.train(stock[TrackedStocksColumn.DATA], best_params)

        return self._train_results

    def plot_stock_history(self,
                           stock_name: str,
                           img_dir: str,
                           show_full_stock_history: bool = False,
                           plot_algorithm_graph: bool = False,
                           plot_algorithm_graph_full: bool = False):
        bids_history = self._statistics_manager.get_bids_history(stock_name)
        stock = self._tracked_stocks[stock_name]
        color_map: Dict[BidResult, str] = {BidResult.WIN: "green",
                                           BidResult.LOSE: "red",
                                           BidResult.DRAW: "grey"}
        fig = go.Figure(
            [
                go.Scatter(
                    x=[bid[BidsHistoryColumn.DATE_OPEN], bid[BidsHistoryColumn.DATE_CLOSE]],
                    y=[bid[BidsHistoryColumn.OPEN_PRICE], bid[BidsHistoryColumn.CLOSE_PRICE]],
                    mode='lines',
                    line_color=color_map[bid[BidsHistoryColumn.RESULT]],
                    line=dict(width=3),
                    showlegend=False,
                )
                for _, bid in bids_history.iterrows()
            ]
        )

        for _, bid in bids_history.iterrows():
            fig.add_shape(type="rect",
                          x0=bid[BidsHistoryColumn.DATE_OPEN], y0=bid[BidsHistoryColumn.OPEN_PRICE],
                          x1=bid[BidsHistoryColumn.DATE_CLOSE], y1=bid[BidsHistoryColumn.TAKE_PROFIT],
                          opacity=0.2,
                          fillcolor="green",
                          line_color="green")
            fig.add_shape(type="rect",
                          x0=bid[BidsHistoryColumn.DATE_OPEN], y0=bid[BidsHistoryColumn.OPEN_PRICE],
                          x1=bid[BidsHistoryColumn.DATE_CLOSE], y1=bid[BidsHistoryColumn.STOP_LOSS],
                          opacity=0.2,
                          fillcolor="red",
                          line_color="red")

        trading_start_date: pd.Timestamp = stock[TrackedStocksColumn.TRADING_START_DATE]
        if plot_algorithm_graph:
            stock[TrackedStocksColumn.TRADE_ALGORITHM].plot(img_dir=img_dir, start_date=trading_start_date,
                                                            show_full=plot_algorithm_graph_full)
        if show_full_stock_history:
            stock_data: pd.DataFrame = stock[TrackedStocksColumn.DATA]
            fig.add_vline(x=trading_start_date, line_width=3, line_dash="dash", line_color="red",
                          name="Start of trading")
        else:
            stock_data: pd.DataFrame = stock[TrackedStocksColumn.DATA][trading_start_date:]

        fig.add_candlestick(x=stock_data.index,
                            open=stock_data["Open"],
                            close=stock_data["Close"],
                            high=stock_data["High"],
                            low=stock_data["Low"],
                            name="Price")

        fig.add_trace(go.Scatter(x=bids_history[BidsHistoryColumn.DATE_OPEN],
                                 y=bids_history[BidsHistoryColumn.OPEN_PRICE],
                                 mode="markers",
                                 marker=dict(
                                     color=np.where(bids_history[BidsHistoryColumn.TYPE] == BidType.LONG, "green",
                                                    "red"),
                                     size=np.where(bids_history[BidsHistoryColumn.TRADE_ACTION].isin([TradeAction.BUY,
                                                                                                      TradeAction.SELL]),
                                                   8, 14),
                                     symbol=np.where(bids_history[BidsHistoryColumn.TYPE] == BidType.LONG,
                                                     "triangle-up",
                                                     "triangle-down")),
                                 name="Bids openings"))

        fig.add_trace(go.Scatter(x=bids_history[BidsHistoryColumn.DATE_CLOSE],
                                 y=bids_history[BidsHistoryColumn.CLOSE_PRICE],
                                 mode="markers",
                                 marker=dict(
                                     color=np.where(bids_history[BidsHistoryColumn.TYPE] == BidType.LONG, "green",
                                                    "red"),
                                     size=5,
                                     symbol="square"),
                                 name="Bids closures"))

        fig.add_bar(x=[stock_data.index[0]], y=[stock_data["Close"][0]],
                    marker=dict(opacity=0.2, color="red"),
                    name="Stop loss level", visible="legendonly")
        fig.add_bar(x=[stock_data.index[0]], y=[stock_data["Close"][0]],
                    marker=dict(opacity=0.2, color="green"),
                    name="Take profit level", visible="legendonly")

        fig.update_layout(
            title=f"Trading activity on {stock_name} with {stock[TrackedStocksColumn.TRADE_ALGORITHM].get_algorithm_name()}",
            xaxis_title="Date",
            yaxis_title="Price")

        # fig.show()
        fig.write_image(f"{img_dir}/{stock_name}_trading.png", scale=1, width=1400, height=900)

    def plot_earnings_curve(self, img_dir: str, stock_name: Optional[str] = None, ignore_crisis: bool = True):
        if stock_name is None:
            name = "Total Earnings"
        else:
            name = f"Earnings on {stock_name} with {self._tracked_stocks[stock_name][TrackedStocksColumn.TRADE_ALGORITHM].get_algorithm_name()}"

        if not ignore_crisis:
            name += " Out of crisis"

        earnings_history = self._statistics_manager.get_earnings_history(stock_name, ignore_crisis)

        fig = go.Figure(go.Scatter(
            x=earnings_history.index,
            y=earnings_history.cumsum()[EarningsHistoryColumn.VALUE],
            mode='lines',
            line_color="blue",
            showlegend=False,
        ))

        fig.update_layout(title=name,
                          xaxis_title="Date",
                          yaxis_title="Total Profit")

        # fig.show()
        fig.write_image(f"{img_dir}/{name}.png", scale=1, width=1400, height=900)
