from typing import Optional, Union, Dict, List, Hashable, Tuple

import numpy as np
import pandas as pd
import json
from os.path import exists
import random

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error

import keras
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers import LSTM
from keras.optimizers import Adam
from keras.losses import MeanAbsoluteError, MeanSquaredError
from keras.initializers.initializers_v2 import GlorotUniform, Orthogonal, RandomNormal, Zeros
from keras.regularizers import L2, L1
from keras import metrics
from keras.callbacks import EarlyStopping
from keras.models import load_model

from helping.base_enum import BaseEnum
from indicators.abstract_indicator import TradeAction, TradePointColumn
from trading.trade_algorithms.abstract_trade_algorithm import AbstractTradeAlgorithm

import cufflinks as cf
import plotly.graph_objects as go

cf.go_offline()


class LSTMTradeAlgorithmHyperparam(BaseEnum):
    TEST_PERIOD_SIZE = 1
    PREDICTION_PERIOD = 2
    TAKE_ACTION_BARRIER = 3
    ACTIVE_ACTION_MULTIPLIER = 4


class ModelGridColumns(BaseEnum):
    LEARNING_RATE = 1
    RANDOM_SEED = 2
    HIDDEN = 3
    WINDOW_SIZE = 4
    DROPOUT = 5


class LSTMTradeAlgorithm(AbstractTradeAlgorithm):
    name = "LSTM trade algorithm"
    model_directory = "../../../models/lstm/"
    #model_directory = "../models/lstm/"

    model_grid = {ModelGridColumns.WINDOW_SIZE: [20, 40, 60],
                  ModelGridColumns.LEARNING_RATE: [0.01, 0.001, 0.0001],
                  ModelGridColumns.DROPOUT: [0.0, 0.01, 0.05],
                  ModelGridColumns.HIDDEN: [2, 2.5, 3, 3.5, 4],
                  ModelGridColumns.RANDOM_SEED: [42, 766, 1144, 5555]}
    epochs = 500
    random_grid_search_attempts = 10

    def __init__(self):
        super().__init__()
        self._prediction_period: int = 0
        self._test_period_size: int = 0
        self._take_action_barrier: float = 0
        self._active_action_multiplier: float = 0

        self.mean_absolute_prediction_error: float = 0

        self._scaled_data: Optional[np.ndarray] = None
        self._input_scaler: Optional[MinMaxScaler] = None
        self._model: Optional[Sequential] = None
        self._model_params: Optional[Dict] = None
        self._data_name: str = ""
        self._model_name: str = ""

        self.predictions: Optional[List] = None
        self.trade_points: Optional[pd.DataFrame] = None
        self._last_train_date: Optional[pd.Timestamp] = None
        self._last_train_date_index: int = 0

    @staticmethod
    def get_algorithm_name() -> str:
        return LSTMTradeAlgorithm.name

    @staticmethod
    def create_hyperparameters_dict(test_period_size: int = 60, prediction_period: int = 3,
                                    take_action_barrier: float = 0.01, active_action_multiplier: float = 2):
        return {
            LSTMTradeAlgorithmHyperparam.TEST_PERIOD_SIZE: test_period_size,
            LSTMTradeAlgorithmHyperparam.PREDICTION_PERIOD: prediction_period,
            LSTMTradeAlgorithmHyperparam.TAKE_ACTION_BARRIER: take_action_barrier,
            LSTMTradeAlgorithmHyperparam.ACTIVE_ACTION_MULTIPLIER: active_action_multiplier
        }

    @staticmethod
    def get_default_hyperparameters_grid() -> List[Dict]:
        return [LSTMTradeAlgorithm.create_hyperparameters_dict(),
                LSTMTradeAlgorithm.create_hyperparameters_dict(prediction_period=5)]

    def __clear_vars(self):
        self._dataframe = pd.DataFrame()
        self._input_scaler = MinMaxScaler(feature_range=(0, 1))
        self.trade_points = pd.DataFrame(columns=TradePointColumn.get_elements_list()).set_index(TradePointColumn.DATE)
        self._last_train_date = None
        self._last_train_date_index = 0
        self.predictions = None

    def __define_model_name(self):
        base_name = f"{self._data_name} PP={self._prediction_period} TPS={self._test_period_size}"
        self._model_name = LSTMTradeAlgorithm.model_directory + base_name

    def __save_model(self):
        m = self._model_name + ".h5"
        self._model.save(m)
        p = self._model_name + " params.txt"
        t = json.dumps(self._model_params)
        with open(p, "w") as out_file:
            out_file.write(t)

    def __load_model(self):
        m = self._model_name + ".h5"
        self._model = load_model(m)
        p = self._model_name + " params.txt"
        with open(p, "r") as out_file:
            t = out_file.read()
        self._model_params = json.loads(t)

    def __create_train_dataset(self, window_size: int = 20) -> Tuple[np.ndarray, np.ndarray]:
        dataframe = []
        y = []
        for i in range(self.data.shape[0] - (self._prediction_period + window_size - 1)):
            window = self._scaled_data[i:(i + window_size)]
            dataframe.append(window)
            y.append(self.data["Close"][(i + window_size + self._prediction_period - 1)])
        dataframe = np.array(dataframe, dtype=np.float32)
        y = np.array(y, dtype=np.float32)
        return dataframe, y

    def __choose_best_model(self):
        model_grid = LSTMTradeAlgorithm.model_grid
        best_model: Optional[Sequential] = None
        best_model_params: Optional[Dict] = None
        best_val_loss = 0
        es = EarlyStopping(min_delta=1e-8, patience=15, verbose=0)
        random.seed(4117)

        for window_size in model_grid[ModelGridColumns.WINDOW_SIZE]:
            x, y = self.__create_train_dataset(window_size)
            x_train = x[:-self._test_period_size]
            y_train = y[:-self._test_period_size]
            for i in range(0, LSTMTradeAlgorithm.random_grid_search_attempts):
                model_params = {ModelGridColumns.WINDOW_SIZE.name: window_size}
                learning_rate = model_grid[ModelGridColumns.LEARNING_RATE][
                    random.randint(0, len(model_grid[ModelGridColumns.LEARNING_RATE]) - 1)]
                model_params[ModelGridColumns.LEARNING_RATE.name] = learning_rate
                hidden_coef = model_grid[ModelGridColumns.HIDDEN][
                    random.randint(0, len(model_grid[ModelGridColumns.HIDDEN]) - 1)]
                hidden = int(hidden_coef * 5)
                model_params[ModelGridColumns.HIDDEN.name] = hidden
                dropout = model_grid[ModelGridColumns.DROPOUT][
                    random.randint(0, len(model_grid[ModelGridColumns.DROPOUT]) - 1)]
                model_params[ModelGridColumns.DROPOUT.name] = dropout
                random_seed = model_grid[ModelGridColumns.RANDOM_SEED][
                    random.randint(0, len(model_grid[ModelGridColumns.RANDOM_SEED]) - 1)]

                optimizer = Adam(learning_rate=learning_rate)
                kernel_initializer = GlorotUniform(seed=random_seed)
                recurrent_initializer = Orthogonal(seed=random_seed)
                weight_initializer = RandomNormal(seed=random_seed)
                model = Sequential()
                model.add(LSTM(units=hidden, kernel_initializer=kernel_initializer,
                               recurrent_initializer=recurrent_initializer,
                               activation="relu",
                               kernel_regularizer=L2(1e-4),
                               recurrent_regularizer=L2(1e-4),
                               bias_regularizer=L2(1e-4),
                               dropout=dropout))
                model.add(Dense(1, kernel_initializer=weight_initializer, kernel_regularizer=L2(1e-4),
                                bias_regularizer=L2(1e-4)))
                model.compile(optimizer=optimizer, loss=MeanSquaredError(),
                              metrics=["mse", "mae"])
                history = model.fit(x=x_train, y=y_train, epochs=LSTMTradeAlgorithm.epochs, shuffle=False,
                                    validation_split=0.15, callbacks=[es], verbose=0)
                model_loss = history.history['val_mse'][-1]
                if (best_model is None) or model_loss < best_val_loss:
                    best_model = model
                    best_val_loss = model_loss
                    best_model_params = model_params

        self._model = best_model
        self._model_params = best_model_params

    def train(self, data: pd.DataFrame, hyperparameters: Dict):
        super().train(data, hyperparameters)
        self.__clear_vars()
        self._prediction_period = hyperparameters[LSTMTradeAlgorithmHyperparam.PREDICTION_PERIOD]
        self._test_period_size = hyperparameters[LSTMTradeAlgorithmHyperparam.TEST_PERIOD_SIZE]
        self._take_action_barrier = hyperparameters[LSTMTradeAlgorithmHyperparam.TAKE_ACTION_BARRIER]
        self._active_action_multiplier = hyperparameters[LSTMTradeAlgorithmHyperparam.ACTIVE_ACTION_MULTIPLIER]
        self._data_name = hyperparameters["DATA_NAME"]
        self.__define_model_name()

        selected_data: pd.DataFrame = self.data.drop("Adj Close", axis=1)
        self._scaled_data = self._input_scaler.fit_transform(selected_data)

        if exists(self._model_name + ".h5"):
            print("loading")
            self.__load_model()
        else:
            self.__choose_best_model()
            self.__save_model()

        print("Model params")
        print(self._model_params)

        x, y = self.__create_train_dataset(self._model_params[ModelGridColumns.WINDOW_SIZE.name])
        x_test = x[-self._test_period_size:]
        y_test = y[-self._test_period_size:]
        predictions = self._model.predict(x=x_test)
        closes_test = self.data["Close"][
                      -(self._test_period_size + self._prediction_period): -self._prediction_period]
        cum_sum = 0
        for i in range(0, len(predictions)):
            cur_close = closes_test[i]
            abs_relative_diff = np.abs((predictions[i][0] - y_test[i]) / cur_close)
            cum_sum += abs_relative_diff
        self.mean_absolute_prediction_error = cum_sum / len(predictions)

        print(f"MADRC = {self.mean_absolute_prediction_error}")
        es = EarlyStopping(min_delta=1e-8, patience=5, verbose=0)
        history = self._model.fit(x=x, y=y, epochs=LSTMTradeAlgorithm.epochs, validation_split=0.05,
                                  shuffle=False, callbacks=[es], verbose=0)
        print(f"Best model MSE = {history.history['mse'][-1]}, Val MSE = {history.history['val_mse'][-1]}")

        predictions = self._model.predict(x=x, verbose=0)
        self.predictions = predictions.flatten().tolist()
        self._last_train_date = self.data.index[-(self._prediction_period + 1)]
        self._last_train_date_index = self.data.shape[0] - (self._prediction_period + 1)

        for i in range(self._prediction_period - 1, -1, -1):
            x = self.__transform_point(i)
            prediction = self._model.predict(x=x, verbose=0)
            self.predictions.append(prediction[0][0])

    def __transform_point(self, index: int = 0) -> np.ndarray:
        window_size = self._model_params[ModelGridColumns.WINDOW_SIZE.name]
        if index == 0:
            window = self.data[-window_size:]
        else:
            window = self.data[-(window_size + index): -index]
        window = window.drop("Adj Close", axis=1)
        window = self._input_scaler.transform(window)
        return np.array([window], dtype=np.float32)

    def evaluate_new_point(self, new_point: pd.Series, date: Union[str, pd.Timestamp],
                           special_params: Optional[Dict] = None) -> TradeAction:
        self.data.loc[date] = new_point
        last_close = new_point["Close"]
        final_action = TradeAction.NONE
        new_point_scaled = self.__transform_point()
        prediction = self._model.predict(new_point_scaled, verbose=0)[0][0]
        self.predictions.append(prediction)
        relative_diff = (prediction - last_close) / last_close

        if relative_diff > 0:
            # relative_diff -= self.mean_absolute_prediction_error
            if np.abs(relative_diff) >= 0.15:
                return final_action
            if relative_diff >= self._take_action_barrier:
                if relative_diff >= self._take_action_barrier * self._active_action_multiplier:
                    final_action = TradeAction.ACTIVELY_BUY
                else:
                    final_action = TradeAction.BUY
        else:
            # relative_diff += self.mean_absolute_prediction_error
            if np.abs(relative_diff) >= 0.15:
                return final_action
            if relative_diff <= -self._take_action_barrier:
                if relative_diff <= -self._take_action_barrier * self._active_action_multiplier:
                    final_action = TradeAction.ACTIVELY_SELL
                else:
                    final_action = TradeAction.SELL

        if final_action != TradeAction.NONE:
            self.__add_trade_point(date, new_point["Close"], final_action)
        return final_action

    def __add_trade_point(self, date: Union[pd.Timestamp, Hashable], price: float, action: TradeAction):
        self.trade_points.loc[date] = {TradePointColumn.PRICE: price, TradePointColumn.ACTION: action}

    def plot(self, img_dir: str, start_date: Optional[pd.Timestamp] = None,
             end_date: Optional[pd.Timestamp] = None, show_full: bool = False):
        intend = self._prediction_period + self._model_params[ModelGridColumns.WINDOW_SIZE.name] - 1
        selected_data = self.data[intend:]
        selected_predictions = self.predictions[:-self._prediction_period]
        base_file_name = f"{img_dir}/{self._data_name}"
        print(f"MADRC = {self.mean_absolute_prediction_error}")
        if not show_full:
            watch_intend = (self._last_train_date_index - intend) - 150
            selected_data = selected_data[watch_intend:]
            selected_predictions = selected_predictions[watch_intend:]
            corr = np.corrcoef(selected_data[:-self._prediction_period]["Close"], selected_predictions[self._prediction_period:])
            print(f"{self._data_name} Correlation last known close and prediction on train = {corr[0][1]}")
            auto_corr = np.corrcoef(selected_data[:-self._prediction_period]["Close"], selected_data[self._prediction_period:]["Close"])
            print(f"{self._data_name} Autocorrelation last known close and prediction on train = {auto_corr[0][1]}")
        else:
            base_file_name += " full data"
            corr = np.corrcoef(selected_data[:-self._prediction_period]["Close"], selected_predictions[self._prediction_period:])
            print(f"{self._data_name} Correlation last known close and prediction on full = {corr[0][1]}")
            auto_corr = np.corrcoef(selected_data[:-self._prediction_period]["Close"],
                                    selected_data[self._prediction_period:]["Close"])
            print(f"{self._data_name} Autocorrelation last known close and prediction on full = {auto_corr[0][1]}")

        fig = go.Figure()

        title = f"{self._data_name} LSTM model with params LR={self._model_params[ModelGridColumns.LEARNING_RATE.name]}, HL={self._model_params[ModelGridColumns.HIDDEN.name]}, WINDOW={self._model_params[ModelGridColumns.WINDOW_SIZE.name]}"
        fig.update_layout(
            title=title,
            xaxis_title="Date")

        fig.add_trace(go.Scatter(x=selected_data.index, y=selected_data["Close"], mode='lines',
                                 line=dict(width=2, color="blue"), name="Real close price"))

        fig.add_trace(go.Scatter(x=selected_data.index, y=selected_predictions, mode='lines',
                                 line=dict(width=2, color="orange"), name="Predicted close price"))

        d_max, d_min = selected_data["Close"].max(), selected_data["Close"].min()
        fig.add_trace(go.Scatter(x=[self._last_train_date, self._last_train_date], y=[d_max, d_min], mode='lines',
                                 line=dict(width=1, dash="dash", color="red"), name="Start of trading"))

        # fig.show()
        fig.write_image((base_file_name + " without marks.png"), scale=1, width=1400, height=900)

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

        trade_point_index = 0
        for i in range(0, selected_data.shape[0]):
            if (trade_point_index == self.trade_points.shape[0]) or (
                    i + self._prediction_period == len(selected_predictions)):
                break
            if selected_data.index[i] == self.trade_points.index[trade_point_index]:
                prediction = selected_predictions[i + self._prediction_period]
                prediction_date = selected_data.index[i + self._prediction_period]
                trade_point = self.trade_points.iloc[trade_point_index]
                if prediction - trade_point[TradePointColumn.PRICE] >= 0:
                    color = "green"
                else:
                    color = "red"
                fig.add_trace(go.Scatter(x=[self.trade_points.index[trade_point_index], prediction_date],
                                         y=[trade_point[TradePointColumn.PRICE], prediction], mode='lines',
                                         line=dict(width=1, color=color), showlegend=False))
                trade_point_index += 1

        # fig.show()
        fig.write_image((base_file_name + ".png"), scale=1, width=1400, height=900)
