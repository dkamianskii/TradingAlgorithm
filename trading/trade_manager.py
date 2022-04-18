import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Union, Tuple

from helping.base_enum import BaseEnum
from indicators.abstract_indicator import TradeAction
from trading.trading_enums import BidType
from trading.abstract_trade_algorithm import AbstractTradeAlgorithm
from trading.macd_super_trend_trade_algorithm import MACDSuperTrendTradeAlgorithm
from trading.trade_statistics_manager import TradeStatisticsManager

import cufflinks as cf
import plotly.graph_objects as go

cf.go_offline()


class TradeManager:
    class DefaultTradeAlgorithm(BaseEnum):
        MACD_SuperTrend = 1,
        Indicators_council = 2,
        Price_prediction = 3

    def __init__(self, days_to_keep_limit: int = 14,
                 days_to_chill: int = 4,
                 use_limited_money: bool = False,
                 money_for_a_bid: float = 10000,
                 start_capital: float = 0,
                 risk_rate: float = 0.05):
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
        self.portfolio: pd.DataFrame = pd.DataFrame(columns=["Stock Name", "Price", "Type", "Amount",
                                                             "Take Profit Level", "Stop Loss Level", "Date"])

        self.available_money: float = start_capital
        self.account_money: float = start_capital
        self.start_capital: float = start_capital

        self._tracked_stocks: Dict[str, Dict] = {}
        self._statistics_manager: TradeStatisticsManager = TradeStatisticsManager()
        self._train_results: Dict[str, Dict[str, pd.DataFrame]] = {}
        self._days_to_keep_limit: pd.Timedelta = pd.Timedelta(days=days_to_keep_limit)
        self._days_to_chill: pd.Timedelta = pd.Timedelta(days=days_to_chill)
        self._use_limited_money: bool = use_limited_money
        self._risk_rate: float = risk_rate
        self._money_for_a_bid: float = money_for_a_bid

    def clear_history(self):
        self.portfolio = pd.DataFrame(columns=["Stock Name", "Price", "Type", "Amount",
                                               "Take Profit Level", "Stop Loss Level", "Date"])
        self._statistics_manager.clear_history()
        self.available_money = self.start_capital
        self.account_money = self.start_capital
        for _, stock in self._tracked_stocks.items():
            stock["trading start date"] = None

    def clear_tracked_stocks_list(self):
        self._tracked_stocks = {}
        self._statistics_manager = TradeStatisticsManager()

    def set_manager_params(self, days_to_keep_limit: int = 14,
                           days_to_chill: int = 4,
                           use_limited_money: bool = False,
                           money_for_a_bid: float = 10000,
                           start_capital: float = 0,
                           risk_rate: float = 0.05):
        self._days_to_keep_limit: pd.Timedelta = pd.Timedelta(days=days_to_keep_limit)
        self._days_to_chill: pd.Timedelta = pd.Timedelta(days=days_to_chill)
        self._use_limited_money: bool = use_limited_money
        self._risk_rate: float = risk_rate
        self._money_for_a_bid: float = money_for_a_bid
        self.available_money: float = start_capital
        self.account_money: float = start_capital
        self.start_capital: float = start_capital

    def set_tracked_stock(self, stock_name: str,
                          stock_data: pd.DataFrame,
                          trade_algorithm: Union[DefaultTradeAlgorithm, AbstractTradeAlgorithm],
                          custom_params_grid: Optional[List[Dict]] = None):
        if type(trade_algorithm) is TradeManager.DefaultTradeAlgorithm:
            if trade_algorithm == TradeManager.DefaultTradeAlgorithm.MACD_SuperTrend:
                algorithm = MACDSuperTrendTradeAlgorithm()
            else:
                raise ValueError(
                    "algorithm must be one of default trade algorithms or user should provide custom trade algorithm")
        else:
            algorithm = trade_algorithm

        self._statistics_manager.set_tracked_stock(stock_name)
        self._tracked_stocks[stock_name] = {
            "data": stock_data,
            "trade algorithm": algorithm,
            "params grid": custom_params_grid,
            "chosen params": None,
            "trading start date": None}

    def __add_to_portfolio(self, stock_name: str,
                           price: float,
                           date: pd.Timestamp,
                           amount: int,
                           action: TradeAction,
                           bid_type: BidType):
        stop_loss_lvl, take_profit_lvl = self.__evaluate_stop_loss_and_take_profit(price, action)
        if self._use_limited_money:
            self.available_money -= price * amount
        bid_to_append = pd.DataFrame([{"Stock Name": stock_name,
                                       "Price": price,
                                       "Type": bid_type,
                                       "Amount": amount,
                                       "Take Profit Level": take_profit_lvl,
                                       "Stop Loss Level": stop_loss_lvl,
                                       "Date": date}])
        self.portfolio = pd.concat([self.portfolio, bid_to_append])
        self._statistics_manager.open_bid(stock_name, date, price, bid_type)

    def __evaluate_stop_loss_and_take_profit(self, price: float, action: TradeAction) -> Tuple[
        float, float]:  # todo Переработать систему стоп лосс - тейк профит
        """IN PROGRESS"""
        if action == TradeAction.BUY:
            return price * 0.985, price * 1.025
        elif action == TradeAction.ACTIVELY_BUY:
            return price * 0.985, price * 1.05
        elif action == TradeAction.SELL:
            return price * 1.015, price * 0.975
        elif action == TradeAction.ACTIVELY_SELL:
            return price * 1.015, price * 0.95

    def __close_bid(self, stock_name: str, close_price: float,
                    open_date: pd.Timestamp, close_date: pd.Timestamp,
                    cashback: float, profit: float = 0, draw: bool = False):
        self._statistics_manager.update_trade_result(stock_name, profit, draw)
        self._statistics_manager.close_bid(stock_name, open_date, close_date, close_price)
        self._statistics_manager.add_earnings(stock_name, profit, close_date)
        if self._use_limited_money:
            self.available_money += cashback
            self.account_money += profit

    def __manage_portfolio_assets(self, stock_name: str, assets_in_portfolio: pd.DataFrame,
                                  new_point: pd.Series, date: pd.Timestamp):
        """IN PROGRESS"""
        indexes_to_drop = []
        for index, asset in assets_in_portfolio.iterrows():
            if asset["Type"] == BidType.LONG:
                if (new_point["Close"] >= asset["Take Profit Level"]) or (
                        new_point["Close"] <= asset["Stop Loss Level"]):
                    price_diff = new_point["Close"] - asset["Price"]
                    self.__close_bid(stock_name, asset["Date"], date, new_point["Close"],
                                     new_point["Close"] * asset["Amount"], price_diff * asset["Amount"])
                    indexes_to_drop.append(index)
                    continue
            else:
                if (new_point["Close"] <= asset["Take Profit Level"]) or (
                        new_point["Close"] >= asset["Stop Loss Level"]):
                    price_diff = asset["Price"] - new_point["Close"]
                    self.__close_bid(stock_name, new_point["Close"], asset["Date"], date,
                                     (asset["Price"] + price_diff) * asset["Amount"], price_diff * asset["Amount"])
                    indexes_to_drop.append(index)
                    continue
            if (date - asset["Date"]) > self._days_to_keep_limit:
                self.__close_bid(stock_name, new_point["Close"], asset["Date"], date,
                                 new_point["Close"] * asset["Amount"], draw=True)
                indexes_to_drop.append(index)
        if len(indexes_to_drop) > 0:
            self.portfolio.drop(labels=indexes_to_drop, inplace=True)
            assets_in_portfolio.drop(labels=indexes_to_drop, inplace=True)
        else:
            self._statistics_manager.add_earnings(stock_name, 0, date)

    def __evaluate_shares_amount_to_bid(self, price: float) -> int:
        shares_to_buy = 0
        if self._use_limited_money:
            money_to_risk = self.account_money * self._risk_rate
            if self.available_money > money_to_risk:
                shares_to_buy = np.floor(money_to_risk / price)
        else:
            shares_to_buy = np.floor(self._money_for_a_bid / price)
        return shares_to_buy

    def evaluate_new_point(self, stock_name: str,
                           new_point: pd.Series,
                           date: Union[str, pd.Timestamp],
                           add_point_to_the_data: bool = True):
        date = pd.Timestamp(ts_input=date)
        assets_in_portfolio = self.portfolio[self.portfolio["Stock Name"] == stock_name]
        self.__manage_portfolio_assets(stock_name, assets_in_portfolio, new_point, date)

        stock = self._tracked_stocks[stock_name]
        action = stock["trade algorithm"].evaluate_new_point(new_point, date)
        if action != TradeAction.NONE:
            if (action == TradeAction.BUY) or (action == TradeAction.ACTIVELY_BUY):
                bid_type = BidType.LONG
            else:
                bid_type = BidType.SHORT

            if assets_in_portfolio.shape[0] != 0:
                last_bid_date = assets_in_portfolio[assets_in_portfolio["Type"] == bid_type]["Date"].max()
                if date - last_bid_date < self._days_to_chill:
                    if add_point_to_the_data:
                        stock["data"].loc[date] = new_point
                    return
            amount = self.__evaluate_shares_amount_to_bid(new_point["Close"])
            if amount == 0:
                print(f"Not enough money to purchase {stock_name} at {date} by price {new_point['Close']}")
            else:
                self.__add_to_portfolio(stock_name, new_point["Close"], date, amount, action, bid_type)
        if add_point_to_the_data:
            stock["data"].loc[date] = new_point
            if stock["trading start date"] is None:
                stock["trading start date"] = date

    def train(self, test_start_date: Union[str, pd.Timestamp],
              test_end_date: Optional[Union[str, pd.Timestamp]] = None) -> Dict[str, Dict[str, pd.DataFrame]]:
        self._train_results = {}
        test_start_date = pd.Timestamp(ts_input=test_start_date)
        for stock_name, stock in self._tracked_stocks.items():
            train_data: pd.DataFrame = stock["data"][:test_start_date]
            if test_end_date is not None:
                test_end_date = pd.Timestamp(ts_input=test_end_date)
                test_data: pd.DataFrame = stock["data"][test_start_date:test_end_date]
            else:
                test_data: pd.DataFrame = stock["data"][test_start_date:]

            algorithm: AbstractTradeAlgorithm = stock["trade algorithm"]
            if stock["params grid"] is None:
                params_grid = algorithm.get_default_hyperparameters_grid()
            else:
                params_grid = stock["params grid"]

            best_params = params_grid[0]
            max_earnings = None
            for params in params_grid:
                self.clear_history()
                algorithm.train(train_data, params)
                for date, point in test_data.iterrows():
                    self.evaluate_new_point(stock_name, point, date, False)
                self._train_results[stock_name] = {str(params): self._statistics_manager.trade_result.copy()}
                earnings = self._statistics_manager.trade_result.at["Total", "Earned Profit"]
                if (max_earnings is None) and (earnings > max_earnings):
                    max_earnings = earnings
                    best_params = params
            algorithm.train(stock["data"], best_params)

        return self._train_results

    def get_trade_results(self) -> pd.DataFrame:
        return self._statistics_manager.trade_result

    def plot_stock_history(self,
                           stock_name: str,
                           show_full_stock_history: bool = False):
        bids_history = self._statistics_manager.stocks_statistics[stock_name]["bids history"]
        bids_history = bids_history[~bids_history["Date Close"].isna()]
        fig = go.Figure(
            [
                go.Scatter(
                    x=[date_open, bid["Date Close"]],
                    y=[bid["Open Price"], bid["Close Price"]],
                    mode='lines',
                    line_color=bid["Result color"],
                    line=dict(width=3),
                    showlegend=False,
                )
                for date_open, bid in bids_history.iterrows()
            ]
        )

        trading_start_date: pd.Timestamp = self._tracked_stocks[stock_name]["trading start date"]
        if show_full_stock_history:
            stock_data: pd.DataFrame = self._tracked_stocks[stock_name]["data"]
            max = stock_data["High"].max()
            min = stock_data["Low"].min()
            fig.add_trace(go.Scatter(x=[trading_start_date, trading_start_date],
                                     y=[min, max],
                                     mode='lines',
                                     line_color="red",
                                     line=dict(width=2),
                                     name="Start of trading"))
        else:
            stock_data: pd.DataFrame = self._tracked_stocks[stock_name]["data"][trading_start_date:]

        fig.add_candlestick(x=stock_data.index,
                            open=stock_data["Open"],
                            close=stock_data["Close"],
                            high=stock_data["High"],
                            low=stock_data["Low"],
                            name="Price")

        fig.add_trace(go.Scatter(x=bids_history.index,
                                 y=bids_history["Open Price"],
                                 mode="markers",
                                 marker=dict(
                                     color=np.where(bids_history["Type"] == BidType.LONG, "green", "red"),
                                     size=7,
                                     symbol=np.where(bids_history["Type"] == BidType.LONG, "triangle-up",
                                                     "triangle-down")),
                                 name="Bids openings"))

        fig.add_trace(go.Scatter(x=bids_history["Date Close"],
                                 y=bids_history["Close Price"],
                                 mode="markers",
                                 marker=dict(
                                     color=np.where(bids_history["Type"] == BidType.LONG, "green", "red"),
                                     size=7,
                                     symbol="square"),
                                 name="Bids closures"))

        fig.update_layout(title=f"{stock_name} Trading activity",
                          xaxis_title="Date",
                          yaxis_title="Price")

        fig.show()

    def plot_earnings_curve(self, stock_name: Optional[str] = None):
        if stock_name is None:
            earnings_history = self._statistics_manager.total_earnings_history
            name = "Total Earnings"
        else:
            earnings_history = self._statistics_manager.stocks_statistics[stock_name]["earnings history"]
            name = f"Earnings on {stock_name}"

        fig = go.Figure(go.Scatter(
            x=earnings_history["Date"],
            y=earnings_history.cumsum(),
            mode='lines',
            line_color="blue",
            showlegend=False,
        ))

        fig.update_layout(title=name,
                          xaxis_title="Date",
                          yaxis_title="Total Profit")

        fig.show()
