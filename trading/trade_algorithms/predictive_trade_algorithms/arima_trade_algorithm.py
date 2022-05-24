from typing import Optional, Union, Dict, List, Iterable, Set, Hashable
from helping.base_enum import BaseEnum

import pandas as pd
import numpy as np
from pmdarima.arima import ARIMA, auto_arima

from indicators.abstract_indicator import TradeAction, TradePointColumn
from trading.trade_algorithms.abstract_trade_algorithm import AbstractTradeAlgorithm

import cufflinks as cf
import plotly.graph_objects as go

cf.go_offline()


class ARIMATradeAlgorithmHyperparam(BaseEnum):
    TEST_PERIOD_SIZE = 1
    PREDICTION_PERIOD = 2
    TAKE_ACTION_BARRIER = 3
    ACTIVE_ACTION_MULTIPLIER = 4
    USE_REFIT = 5
    FIT_SIZE = 6
    REFIT_ADD_SIZE = 7


class ARIMATradeAlgorithm(AbstractTradeAlgorithm):
    name = "ARIMA trade algorithm"

    def __init__(self):
        super().__init__()
        self._prediction_period: int = 0
        self._take_action_barrier: float = 0
        self._active_action_multiplier: float = 0
        self._test_size: int = 0

        self._use_refit: bool = False
        self._fit_size: int = 0
        self._refit_add_size: int = 0

        self._train_data: Optional[pd.DataFrame] = None
        self._test_data: Optional[pd.DataFrame] = None

        self._model: Optional[ARIMA] = None
        self.best_model_order: Optional[Set] = None
        self.mean_absolute_prediction_error: float = 0

        self._last_train_date: Optional[pd.Timestamp] = None
        self._last_train_date_index: int = 0
        self._refit_upload: int = 0
        self.predictions: Optional[List] = None
        self.trade_points: Optional[pd.DataFrame] = None
        self.refit_points: Optional[pd.DataFrame] = None

    @staticmethod
    def get_algorithm_name() -> str:
        return ARIMATradeAlgorithm.name

    @staticmethod
    def create_hyperparameters_dict(prediction_period: int = 3, test_period_size=50,
                                    take_action_barrier: float = 0.01, active_action_multiplier: float = 1.5,
                                    use_refit: bool = False, fit_size: int = 100, refit_add_size: int = 14):
        return {
            ARIMATradeAlgorithmHyperparam.TEST_PERIOD_SIZE: test_period_size,
            ARIMATradeAlgorithmHyperparam.PREDICTION_PERIOD: prediction_period,
            ARIMATradeAlgorithmHyperparam.TAKE_ACTION_BARRIER: take_action_barrier,
            ARIMATradeAlgorithmHyperparam.ACTIVE_ACTION_MULTIPLIER: active_action_multiplier,
            ARIMATradeAlgorithmHyperparam.USE_REFIT: use_refit,
            ARIMATradeAlgorithmHyperparam.FIT_SIZE: fit_size,
            ARIMATradeAlgorithmHyperparam.REFIT_ADD_SIZE: refit_add_size
        }

    @staticmethod
    def get_default_hyperparameters_grid() -> List[Dict]:
        return [ARIMATradeAlgorithm.create_hyperparameters_dict(),
                ARIMATradeAlgorithm.create_hyperparameters_dict(prediction_period=2),
                ARIMATradeAlgorithm.create_hyperparameters_dict(prediction_period=4),
                ARIMATradeAlgorithm.create_hyperparameters_dict(use_refit=True),
                ARIMATradeAlgorithm.create_hyperparameters_dict(use_refit=True, fit_size=60, refit_add_size=5)]

    def __clear_vars(self):
        self.trade_points = pd.DataFrame(columns=TradePointColumn.get_elements_list()).set_index(TradePointColumn.DATE)
        self.refit_points = pd.DataFrame(columns=["Date", "price"]).set_index("Date")
        self._last_train_date = None
        self._last_train_date_index = 0
        self.predictions = None
        self._refit_upload: int = 0

    def train(self, data: pd.DataFrame, hyperparameters: Dict):
        super().train(data, hyperparameters)
        self.__clear_vars()
        self._prediction_period = hyperparameters[ARIMATradeAlgorithmHyperparam.PREDICTION_PERIOD]
        self._take_action_barrier = hyperparameters[ARIMATradeAlgorithmHyperparam.TAKE_ACTION_BARRIER]
        self._active_action_multiplier = hyperparameters[ARIMATradeAlgorithmHyperparam.ACTIVE_ACTION_MULTIPLIER]
        self._use_refit = hyperparameters[ARIMATradeAlgorithmHyperparam.USE_REFIT]
        self._fit_size = hyperparameters[ARIMATradeAlgorithmHyperparam.FIT_SIZE]
        self._refit_add_size = hyperparameters[ARIMATradeAlgorithmHyperparam.REFIT_ADD_SIZE]
        self._test_size = hyperparameters[ARIMATradeAlgorithmHyperparam.TEST_PERIOD_SIZE]

        self._train_data = self.data[:-self._test_size]
        self._test_data = self.data[-self._test_size:]

        if self._use_refit:
            self._train_data = self.data[-self._fit_size:]

        train_closes = self._train_data["Close"]
        self._model: ARIMA = auto_arima(y=train_closes, start_p=6, start_q=6, max_p=10, max_q=10,
                                        seasonal=False, max_order=21, max_d=1, error_action="ignore")
        self.best_model_order = self._model.order

        test_closes = self._test_data["Close"]
        cum_sum = 0
        cur_close = train_closes[-1]
        for close in test_closes:
            prediction = self._model.predict(n_periods=self._prediction_period)[-1]
            abs_relative_diff = np.abs((prediction - close) / cur_close)
            cum_sum += abs_relative_diff
            cur_close = close
            self._model.update(y=[close])
        self.mean_absolute_prediction_error = cum_sum / self._test_size

        if self._use_refit:
            self._model.fit(y=self.data["Close"][-self._fit_size:])
        else:
            self._model.fit(y=self.data["Close"])
        self.predictions = self._model.predict_in_sample().tolist()
        self._last_train_date = self.data.index[-1]
        self._last_train_date_index = self.data.shape[0] - 1
        for prediction in self._model.predict(n_periods=self._prediction_period):
            self.predictions.append(prediction)

    def __refit_model(self):
        train_closes = self.data[-(self._fit_size + self._test_size):-self._test_size]["Close"]
        test_closes = self.data[-self._test_size:]["Close"]

        self._model: ARIMA = auto_arima(y=train_closes, start_p=6, start_q=6, max_p=10, max_q=10,
                                        seasonal=False, max_order=21, max_d=1, error_action="ignore")
        self.best_model_order = self._model.order 

        cum_sum = 0
        cur_close = train_closes[-1]
        for close in test_closes:
            prediction = self._model.predict(n_periods=self._prediction_period)[-1]
            abs_relative_diff = np.abs((prediction - close) / cur_close)
            cum_sum += abs_relative_diff
            cur_close = close
            self._model.update(y=[close])
        self.mean_absolute_prediction_error = cum_sum / self._test_size
        self._model.fit(y=self.data["Close"][-self._fit_size:])

    def evaluate_new_point(self, new_point: pd.Series, date: Union[str, pd.Timestamp],
                           special_params: Optional[Dict] = None) -> TradeAction:
        final_action = TradeAction.NONE
        self.data.loc[date] = new_point
        last_close = new_point["Close"]

        if self._use_refit:
            self._refit_upload += 1
            if self._refit_upload == self._refit_add_size:
                self.__refit_model()
                self._refit_upload = 0
            else:
                self._model.update(y=[last_close])
                self.__add_refit_point(date, last_close)
        else:
            self._model.update(y=[last_close])

        prediction = self._model.predict(n_periods=self._prediction_period)[-1]
        relative_diff = (prediction - last_close) / last_close
        if relative_diff > 0:
            relative_diff -= self.mean_absolute_prediction_error
            if relative_diff >= self._take_action_barrier:
                if relative_diff >= self._take_action_barrier * self._active_action_multiplier:
                    final_action = TradeAction.ACTIVELY_BUY
                else:
                    final_action = TradeAction.BUY
        else:
            relative_diff += self.mean_absolute_prediction_error
            if relative_diff <= -self._take_action_barrier:
                if relative_diff <= -self._take_action_barrier * self._active_action_multiplier:
                    final_action = TradeAction.ACTIVELY_SELL
                else:
                    final_action = TradeAction.SELL

        self.predictions.append(prediction)
        if final_action != TradeAction.NONE:
            self.__add_trade_point(date, new_point["Close"], final_action)
        return final_action

    def __add_trade_point(self, date: Union[pd.Timestamp, Hashable], price: float, action: TradeAction):
        self.trade_points.loc[date] = {TradePointColumn.PRICE: price, TradePointColumn.ACTION: action}

    def __add_refit_point(self, date: Union[pd.Timestamp, Hashable], price: float):
        self.refit_points.loc[date] = {"price": price}

    def plot(self, start_date: Optional[pd.Timestamp] = None, end_date: Optional[pd.Timestamp] = None):
        start_index = self._last_train_date_index - self._test_size
        selected_data = self.data[start_index:]
        if self._use_refit:
            selected_predictions = self.predictions[(self._fit_size - self._test_size):-self._prediction_period]
        else:
            selected_predictions = self.predictions[start_index:-self._prediction_period]

        fig = go.Figure()

        fig.add_trace(go.Scatter(x=selected_data.index, y=selected_data["Close"], mode='lines',
                                 line=dict(width=2, color="blue"), name="Real close price"))

        fig.add_trace(go.Scatter(x=selected_data.index, y=selected_predictions, mode='lines',
                                 line=dict(width=2, color="orange"), name="Predicted close price"))

        buy_actions = [TradeAction.BUY, TradeAction.ACTIVELY_BUY]
        active_actions = [TradeAction.ACTIVELY_BUY, TradeAction.ACTIVELY_SELL]
        bool_buys = self.trade_points[TradePointColumn.ACTION].isin(buy_actions)
        bool_actives = self.trade_points[TradePointColumn.ACTION].isin(active_actions)

        fig.add_trace(go.Scatter(x=self.trade_points.index,
                                 y=self.trade_points[TradePointColumn.PRICE],
                                 mode="markers",
                                 marker=dict(
                                     color=np.where(bool_buys, "green", "red"),
                                     size=np.where(bool_actives, 15, 10),
                                     symbol=np.where(bool_buys, "triangle-up", "triangle-down")),
                                 name="Action points"))

        if self._use_refit:
            fig.add_trace(go.Scatter(x=self.refit_points.index,
                                     y=self.refit_points["price"],
                                     mode="markers",
                                     marker=dict(
                                         color="yellow",
                                         size=3,
                                         symbol="square"),
                                     name="Reevaluation points"))

        d_max, d_min = selected_data["Close"].max(), selected_data["Close"].min()
        fig.add_trace(go.Scatter(x=[self._last_train_date, self._last_train_date], y=[d_max, d_min], mode='lines',
                                 line=dict(width=1, dash="dash", color="red"), name="Start of trading"))

        if self._use_refit:
            title = f"ARIMA model with reevaluation | prediction period {self._prediction_period}"
        else:
            title = f"ARIMA model ({self.best_model_order[0]}, {self.best_model_order[1]}, {self.best_model_order[2]}) | prediction period {self._prediction_period}"
        fig.update_layout(
            title=title,
            xaxis_title="Date")

        fig.show()
