from typing import Optional, Union, Dict, List, Any

import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

from helping.base_enum import BaseEnum
from indicators.abstract_indicator import TradeAction, AbstractIndicator, TradePointColumn
from indicators.atr import ATR, ATR_one_point
from indicators.bollinger_bands import BollingerBands, BollingerBandsHyperparam
from indicators.cci import CCI
from indicators.macd import MACD, MACDHyperparam, MACDTradeStrategy
from indicators.rsi import RSI
from indicators.super_trend import SuperTrend, SuperTrendHyperparam
from trading.trade_algorithms.abstract_trade_algorithm import AbstractTradeAlgorithm
from trading.risk_manager import RiskManager, RiskManagerHyperparam


class RandomForestTradeAlgorithmHyperparam(BaseEnum):
    RSI_HYPERPARAMS = 1
    MACD_HYPERPARAMS = 2
    SUPER_TREND_HYPERPARAMS = 3
    CCI_HYPERPARAMS = 4
    BOLLINGER_BANDS_HYPERPARAMS = 5
    RISK_MANAGER_HYPERPARAMS = 6
    DAYS_TO_KEEP_LIMIT = 7
    ATR_PERIOD = 8


class RandomForestTradeAlgorithm(AbstractTradeAlgorithm):
    name = "Random Forest trade algorithm"
    random_forest_grid = {'n_estimators': [100, 200, 300, 400],
                          'max_features': [4, 6, 8, 10],
                          'max_depth': [4, 5, 6, 7, 8],
                          'random_state': [42]}

    def __init__(self):
        super().__init__()
        self._risk_manager = RiskManager()
        # self.trade_points: Optional[pd.DataFrame] = None
        self._random_forest_cls: Optional[RandomForestClassifier] = None
        self._indicators: List[AbstractIndicator] = []
        self._dataframe: pd.DataFrame = pd.DataFrame()
        self._trade_actions_train: pd.Series = pd.Series(dtype='float64')
        self._best_params: Optional[Dict] = None
        self._best_score: Any = None

        self._days_to_keep_limit: int = 0
        self._atr_period: int = 10
        self._atr: Optional[pd.Series] = None
        self._MACD: MACD = MACD()
        self._indicators.append(self._MACD)
        self._RSI: RSI = RSI()
        self._indicators.append(self._RSI)
        self._super_trend: SuperTrend = SuperTrend()
        self._indicators.append(self._super_trend)
        self._CCI: CCI = CCI()
        self._indicators.append(self._CCI)
        self._bollinger_bands: BollingerBands = BollingerBands()
        self._indicators.append(self._bollinger_bands)

    @staticmethod
    def get_algorithm_name() -> str:
        return RandomForestTradeAlgorithm.name

    @staticmethod
    def create_hyperparameters_dict(bid_risk_rate: float = 0.03, take_profit_multiplier: float = 2,
                                    active_action_multiplier: float = 1.5, use_atr: bool = False,
                                    days_to_keep_limit: int = 14, atr_period: int = 14,
                                    rsi_N: int = 14, macd_short_period: int = 12,
                                    macd_long_period: int = 26, macd_signal_period: int = 9,
                                    macd_trade_strategy: MACDTradeStrategy = MACDTradeStrategy.CLASSIC,
                                    super_trend_lookback_period: int = 10, super_trend_multiplier: float = 3,
                                    cci_N: int = 20, bollinger_bands_N: int = 20, bollinger_bands_K: float = 2) -> Dict:
        return {
            RandomForestTradeAlgorithmHyperparam.DAYS_TO_KEEP_LIMIT: days_to_keep_limit,
            RandomForestTradeAlgorithmHyperparam.ATR_PERIOD: atr_period,
            RandomForestTradeAlgorithmHyperparam.RISK_MANAGER_HYPERPARAMS: {
                RiskManagerHyperparam.BID_RISK_RATE: bid_risk_rate,
                RiskManagerHyperparam.TAKE_PROFIT_MULTIPLIER: take_profit_multiplier,
                RiskManagerHyperparam.ACTIVE_ACTION_MULTIPLIER: active_action_multiplier,
                RiskManagerHyperparam.USE_ATR: use_atr
            },
            RandomForestTradeAlgorithmHyperparam.RSI_HYPERPARAMS: {
                "N": rsi_N
            },
            RandomForestTradeAlgorithmHyperparam.MACD_HYPERPARAMS: {
                MACDHyperparam.SHORT_PERIOD: macd_short_period,
                MACDHyperparam.LONG_PERIOD: macd_long_period,
                MACDHyperparam.SIGNAL_PERIOD: macd_signal_period,
                MACDHyperparam.TRADE_STRATEGY: macd_trade_strategy
            },
            RandomForestTradeAlgorithmHyperparam.SUPER_TREND_HYPERPARAMS: {
                SuperTrendHyperparam.LOOKBACK_PERIOD: super_trend_lookback_period,
                SuperTrendHyperparam.MULTIPLIER: super_trend_multiplier
            },
            RandomForestTradeAlgorithmHyperparam.CCI_HYPERPARAMS: {
                "N": cci_N
            },
            RandomForestTradeAlgorithmHyperparam.BOLLINGER_BANDS_HYPERPARAMS: {
                BollingerBandsHyperparam.N: bollinger_bands_N,
                BollingerBandsHyperparam.K: bollinger_bands_K
            }}

    @staticmethod
    def get_default_hyperparameters_grid() -> List[Dict]:
        return [RandomForestTradeAlgorithm.create_hyperparameters_dict(),
                RandomForestTradeAlgorithm.create_hyperparameters_dict(macd_short_period=10, macd_long_period=22,
                                                                       macd_trade_strategy=MACDTradeStrategy.CONVERGENCE),
                RandomForestTradeAlgorithm.create_hyperparameters_dict(macd_short_period=10, macd_long_period=20,
                                                                       super_trend_multiplier=2,
                                                                       super_trend_lookback_period=8),
                RandomForestTradeAlgorithm.create_hyperparameters_dict(rsi_N=8, macd_short_period=10,
                                                                       macd_long_period=22,
                                                                       macd_trade_strategy=MACDTradeStrategy.CONVERGENCE,
                                                                       cci_N=18, bollinger_bands_N=14,
                                                                       bollinger_bands_K=1.5)]

    def __clear_vars(self):
        self.trade_points = pd.DataFrame(columns=TradePointColumn.get_elements_list()).set_index(TradePointColumn.DATE)
        self._indicators_trade_points = {}
        self._dataframe = self.data[["Close", "Open", "Low", "High"]]
        self._trade_actions_train = pd.Series(dtype='float64')
        for indicator in self._indicators:
            indicator.clear_vars()

    def __create_train_dataset(self):
        self._dataframe["RSI"] = self._RSI.RSI_val
        self._dataframe["MACD hist"] = self._MACD.MACD_val["histogram"]
        self._dataframe["SuperTrend value"] = self._super_trend.super_trend_value["Value"]
        self._dataframe["SuperTrend color"] = self._super_trend.super_trend_value["Color"]
        self._dataframe["CCI"] = self._CCI.CCI_value
        self._dataframe["Bollinger Upper Band"] = self._bollinger_bands.bollinger_bands_value["Upper band"]
        self._dataframe["Bollinger Lower Band"] = self._bollinger_bands.bollinger_bands_value["Lower band"]
        self._dataframe["ATR"] = self._atr

        self._dataframe = self._dataframe.dropna()
        super_trend_color_dummy = np.where(self._dataframe["SuperTrend color"] == "green", 1, 0)
        self._dataframe["SuperTrend color"] = super_trend_color_dummy

        index = 0
        for date, point in self._dataframe.iterrows():
            right_action = self.__evaluate_right_action(point, index, point["ATR"])
            self._trade_actions_train.loc[date] = right_action.name
            index += 1

    def __evaluate_right_action(self, start_point: pd.Series, start_index: int, atr: float) -> TradeAction:
        buy_stop_loss, buy_take_profit = self._risk_manager.evaluate_stop_loss_and_take_profit(start_point,
                                                                                               TradeAction.BUY,
                                                                                               atr)
        _, actively_buy_take_profit = self._risk_manager.evaluate_stop_loss_and_take_profit(start_point,
                                                                                            TradeAction.ACTIVELY_BUY,
                                                                                            atr)
        sell_stop_loss, sell_take_profit = self._risk_manager.evaluate_stop_loss_and_take_profit(start_point,
                                                                                                 TradeAction.SELL,
                                                                                                 atr)
        _, actively_sell_take_profit = self._risk_manager.evaluate_stop_loss_and_take_profit(start_point,
                                                                                             TradeAction.ACTIVELY_SELL,
                                                                                             atr)
        first_action_happened = TradeAction.NONE
        buy_stop_loss_triggered = False
        sell_stop_loss_triggered = False
        right_action = TradeAction.NONE
        date_shift, final_take_profit, final_stop_loss = 0, 0., 0.
        for j in range(1, self._days_to_keep_limit):
            if (start_index + j) == self.data.shape[0]:
                break
            date_shift += 1
            cur_point = self.data.iloc[start_index + j]
            cur_price = cur_point["Close"]
            if cur_price <= buy_stop_loss:
                buy_stop_loss_triggered = True
            if cur_price >= sell_stop_loss:
                sell_stop_loss_triggered = True
            if buy_stop_loss_triggered and sell_stop_loss_triggered:
                break
            if first_action_happened == TradeAction.NONE:
                if not buy_stop_loss_triggered and (cur_price >= buy_take_profit):
                    first_action_happened = TradeAction.BUY
                    right_action = TradeAction.BUY
                elif not sell_stop_loss_triggered and (cur_price <= sell_take_profit):
                    first_action_happened = TradeAction.SELL
                    right_action = TradeAction.SELL
            if first_action_happened == TradeAction.BUY:
                if buy_stop_loss_triggered:
                    break
                if cur_price >= actively_buy_take_profit:
                    right_action = TradeAction.ACTIVELY_BUY
                    break
            elif first_action_happened == TradeAction.SELL:
                if sell_stop_loss_triggered:
                    break
                if cur_price <= actively_sell_take_profit:
                    right_action = TradeAction.ACTIVELY_SELL
                    break

        return right_action

    def train(self, data: pd.DataFrame, hyperparameters: Dict):
        # print(f"Start train with hyperparameters {hyperparameters}")
        super().train(data, hyperparameters)
        self._risk_manager.set_manager_params(
            bid_risk_rate=hyperparameters[RandomForestTradeAlgorithmHyperparam.RISK_MANAGER_HYPERPARAMS][
                RiskManagerHyperparam.BID_RISK_RATE],
            take_profit_multiplier=hyperparameters[RandomForestTradeAlgorithmHyperparam.RISK_MANAGER_HYPERPARAMS][
                RiskManagerHyperparam.TAKE_PROFIT_MULTIPLIER],
            active_action_multiplier=hyperparameters[RandomForestTradeAlgorithmHyperparam.RISK_MANAGER_HYPERPARAMS][
                RiskManagerHyperparam.ACTIVE_ACTION_MULTIPLIER],
            use_atr=hyperparameters[RandomForestTradeAlgorithmHyperparam.RISK_MANAGER_HYPERPARAMS][
                RiskManagerHyperparam.USE_ATR])
        self._RSI.set_N(hyperparameters[RandomForestTradeAlgorithmHyperparam.RSI_HYPERPARAMS]["N"])
        self._MACD.set_ma_periods(
            hyperparameters[RandomForestTradeAlgorithmHyperparam.MACD_HYPERPARAMS][MACDHyperparam.SHORT_PERIOD],
            hyperparameters[RandomForestTradeAlgorithmHyperparam.MACD_HYPERPARAMS][MACDHyperparam.LONG_PERIOD],
            hyperparameters[RandomForestTradeAlgorithmHyperparam.MACD_HYPERPARAMS][MACDHyperparam.SIGNAL_PERIOD])
        self._MACD.set_trade_strategy(hyperparameters[RandomForestTradeAlgorithmHyperparam.MACD_HYPERPARAMS][
                                          MACDHyperparam.TRADE_STRATEGY])
        self._super_trend.set_params(
            lookback_period=hyperparameters[RandomForestTradeAlgorithmHyperparam.SUPER_TREND_HYPERPARAMS][
                SuperTrendHyperparam.LOOKBACK_PERIOD],
            multiplier=hyperparameters[RandomForestTradeAlgorithmHyperparam.SUPER_TREND_HYPERPARAMS][
                SuperTrendHyperparam.MULTIPLIER])
        self._CCI.set_params(hyperparameters[RandomForestTradeAlgorithmHyperparam.CCI_HYPERPARAMS]["N"])
        self._bollinger_bands.set_params(
            hyperparameters[RandomForestTradeAlgorithmHyperparam.BOLLINGER_BANDS_HYPERPARAMS][
                BollingerBandsHyperparam.N],
            hyperparameters[RandomForestTradeAlgorithmHyperparam.BOLLINGER_BANDS_HYPERPARAMS][
                BollingerBandsHyperparam.K])

        self.__clear_vars()

        self._days_to_keep_limit = hyperparameters[RandomForestTradeAlgorithmHyperparam.DAYS_TO_KEEP_LIMIT]
        self._atr_period = hyperparameters[RandomForestTradeAlgorithmHyperparam.ATR_PERIOD]
        self._atr = ATR(self.data, self._atr_period)
        for indicator in self._indicators:
            indicator.calculate(self.data)

        # print("finished indicators calculation")

        self.__create_train_dataset()
        # print("created dataset")
        grid_search_cross_val = GridSearchCV(estimator=RandomForestClassifier(),
                                             param_grid=RandomForestTradeAlgorithm.random_forest_grid,
                                             cv=5)
        grid_search_cross_val.fit(self._dataframe, self._trade_actions_train)
        # print("cross validation finished")
        self._best_score = grid_search_cross_val.best_score_
        self._best_params = grid_search_cross_val.best_params_
        self._random_forest_cls = RandomForestClassifier(n_estimators=self._best_params['n_estimators'],
                                                         max_depth=self._best_params['max_depth'],
                                                         max_features=self._best_params['max_features'],
                                                         random_state=self._best_params['random_state'])
        self._random_forest_cls.fit(self._dataframe, self._trade_actions_train)

    def evaluate_new_point(self, new_point: pd.Series, date: Union[str, pd.Timestamp],
                           special_params: Optional[Dict] = None) -> TradeAction:
        for indicator in self._indicators:
            indicator.evaluate_new_point(new_point, date, special_params, False)
        atr_new_point = ATR_one_point(self._atr[-1], self.data.iloc[-1]["Close"], new_point, self._atr_period)
        self._atr.loc[date] = atr_new_point
        self.data.loc[date] = new_point

        super_trend_new_point = self._super_trend.super_trend_value.iloc[-1]
        if super_trend_new_point["Color"] == "green":
            super_trend_new_point_color = 1
        else:
            super_trend_new_point_color = 0
        bollinger_bands_new_point = self._bollinger_bands.bollinger_bands_value.iloc[-1]
        data_for_prediction = {"Close": [new_point["Close"]], "Open": [new_point["Open"]], "Low": [new_point["Low"]],
                               "High": [new_point["High"]], "RSI": [self._RSI.RSI_val[-1]],
                               "MACD hist": [self._MACD.MACD_val.iloc[-1]["histogram"]],
                               "SuperTrend value": [super_trend_new_point["Value"]],
                               "SuperTrend color": [super_trend_new_point_color],
                               "CCI": [self._CCI.CCI_value[-1]],
                               "Bollinger Upper Band": [bollinger_bands_new_point["Upper band"]],
                               "Bollinger Lower Band": [bollinger_bands_new_point["Lower band"]],
                               "ATR": [atr_new_point]}
        trade_action = self._random_forest_cls.predict(pd.DataFrame(data_for_prediction))
        final_action = TradeAction[trade_action[0]]
        return final_action

    def plot(self, start_date: Optional[pd.Timestamp] = None, end_date: Optional[pd.Timestamp] = None):
        print(self._best_params)
        print(self._best_score)

