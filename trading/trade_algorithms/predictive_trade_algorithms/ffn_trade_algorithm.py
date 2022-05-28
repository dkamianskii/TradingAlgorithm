from typing import Optional, Union, Dict, List, Hashable

import numpy as np
import pandas as pd
import json
from os.path import exists
import random

from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error

import keras
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import Adam
from keras.losses import MeanAbsoluteError, MeanSquaredError
from keras.initializers.initializers_v2 import RandomNormal, Zeros
from keras.regularizers import L2, L1
from keras import metrics
from keras.models import load_model

from helping.base_enum import BaseEnum
from indicators.abstract_indicator import TradeAction, TradePointColumn
from trading.trade_algorithms.abstract_trade_algorithm import AbstractTradeAlgorithm

import cufflinks as cf
import plotly.graph_objects as go

cf.go_offline()


class FFNTradeAlgorithmHyperparam(BaseEnum):
    TEST_PERIOD_SIZE = 1
    PREDICTION_PERIOD = 2
    TAKE_ACTION_BARRIER = 3
    ACTIVE_ACTION_MULTIPLIER = 4


class ModelGridColumns(BaseEnum):
    LEARNING_RATE = 1
    RANDOM_SEED = 2
    LAYER_1 = 3
    NUM_OF_ADD_LAYERS = 4
    LAYER_2 = 5
    LAYER_3 = 6


class FFNTradeAlgorithm(AbstractTradeAlgorithm):
    name = "FFN trade algorithm"
    model_directory = "../models/ffn/"

    look_back_period = 3
    min_max_periods = [20, 30, 40, 50]
    min_max_window = 10

    model_grid = {ModelGridColumns.LEARNING_RATE: [0.1, 0.01, 0.001, 0.0001],
                  ModelGridColumns.RANDOM_SEED: [42, 766, 1144, 5555],
                  ModelGridColumns.LAYER_1: [1, 1.25, 1.5, 1.75, 2, 2.25, 2.5],
                  ModelGridColumns.NUM_OF_ADD_LAYERS: [0, 1, 2],
                  ModelGridColumns.LAYER_2: [0.5, 1, 1.25, 1.5],
                  ModelGridColumns.LAYER_3: [.33, 0.5, 0.66, 1]}
    random_grid_search_attempts = 20
    batch_size = 100
    epochs = 500

    def __init__(self):
        super().__init__()
        self._prediction_period: int = 0
        self._test_period_size: int = 0
        self._take_action_barrier: float = 0
        self._active_action_multiplier: float = 0

        self.mean_absolute_prediction_error: float = 0

        self._dataframe: pd.DataFrame = pd.DataFrame()
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
        return FFNTradeAlgorithm.name

    @staticmethod
    def create_hyperparameters_dict(test_period_size: int = 60, prediction_period: int = 3,
                                    take_action_barrier: float = 0.01, active_action_multiplier: float = 1.5):
        return {
            FFNTradeAlgorithmHyperparam.TEST_PERIOD_SIZE: test_period_size,
            FFNTradeAlgorithmHyperparam.PREDICTION_PERIOD: prediction_period,
            FFNTradeAlgorithmHyperparam.TAKE_ACTION_BARRIER: take_action_barrier,
            FFNTradeAlgorithmHyperparam.ACTIVE_ACTION_MULTIPLIER: active_action_multiplier
        }

    @staticmethod
    def get_default_hyperparameters_grid() -> List[Dict]:
        return [FFNTradeAlgorithm.create_hyperparameters_dict(),
                FFNTradeAlgorithm.create_hyperparameters_dict(prediction_period=2),
                FFNTradeAlgorithm.create_hyperparameters_dict(prediction_period=4)]

    def __clear_vars(self):
        self._dataframe = pd.DataFrame()
        self._input_scaler = MinMaxScaler(feature_range=(0, 1))
        self.trade_points = pd.DataFrame(columns=TradePointColumn.get_elements_list()).set_index(TradePointColumn.DATE)
        self._last_train_date = None
        self._last_train_date_index = 0
        self.predictions = None

    def __define_model_name(self):
        base_name = f"{self._data_name} PP={self._prediction_period} TPS={self._test_period_size}"
        self._model_name = FFNTradeAlgorithm.model_directory + base_name

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

    def __create_train_dataset(self):
        for i in range(0, FFNTradeAlgorithm.look_back_period):
            self._dataframe[f"Close t - {i}"] = self.data["Close"].shift(i)
            self._dataframe[f"Open t - {i}"] = self.data["Open"].shift(i)
            self._dataframe[f"High t - {i}"] = self.data["High"].shift(i)
            self._dataframe[f"Low t - {i}"] = self.data["Low"].shift(i)
            self._dataframe[f"Volume t - {i}"] = self.data["Volume"].shift(i)

        max_close = self.data["Close"].rolling(FFNTradeAlgorithm.min_max_window).max()
        min_close = self.data["Close"].rolling(FFNTradeAlgorithm.min_max_window).min()
        for min_max_period in FFNTradeAlgorithm.min_max_periods:
            self._dataframe[f"Max Close t - {min_max_period}"] = max_close.shift(min_max_period)
            self._dataframe[f"Min Close t - {min_max_period}"] = min_close.shift(min_max_period)

        self._dataframe[f"target t + {self._prediction_period}"] = self.data["Close"].shift(-self._prediction_period)
        self._dataframe.dropna(inplace=True)

    def __choose_best_model(self, x_train, y_train):
        input_shape = self._dataframe.shape[1] - 1
        model_grid = FFNTradeAlgorithm.model_grid
        bias_initializer = Zeros()
        es = EarlyStopping(min_delta=1e-8, patience=15, verbose=0)
        best_model: Optional[Sequential] = None
        best_model_params: Optional[Dict] = None
        best_val_loss = 0
        random.seed(1666)

        for i in range(0, FFNTradeAlgorithm.random_grid_search_attempts):
            model_params = {}
            learning_rate = model_grid[ModelGridColumns.LEARNING_RATE][
                random.randint(0, len(model_grid[ModelGridColumns.LEARNING_RATE]) - 1)]
            model_params[ModelGridColumns.LEARNING_RATE.name] = learning_rate
            layer_1_coef = model_grid[ModelGridColumns.LAYER_1][
                random.randint(0, len(model_grid[ModelGridColumns.LAYER_1]) - 1)]
            layer_1_n = int(input_shape * layer_1_coef)
            layer_2_n = 0
            layer_3_n = 0
            model_params[ModelGridColumns.LAYER_1.name] = layer_1_n
            num_of_add_layers = model_grid[ModelGridColumns.NUM_OF_ADD_LAYERS][
                random.randint(0, len(model_grid[ModelGridColumns.NUM_OF_ADD_LAYERS]) - 1)]
            if num_of_add_layers >= 1:
                layer_2_coef = model_grid[ModelGridColumns.LAYER_2][
                    random.randint(0, len(model_grid[ModelGridColumns.LAYER_2]) - 1)]
                layer_2_n = int(layer_1_n * layer_2_coef)
                if num_of_add_layers == 2:
                    layer_3_coef = model_grid[ModelGridColumns.LAYER_3][
                        random.randint(0, len(model_grid[ModelGridColumns.LAYER_3]) - 1)]
                    layer_3_n = int(layer_2_n * layer_3_coef)

            model_params[ModelGridColumns.NUM_OF_ADD_LAYERS.name] = num_of_add_layers
            model_params[ModelGridColumns.LAYER_2.name] = layer_2_n
            model_params[ModelGridColumns.LAYER_3.name] = layer_3_n
            print(model_params)

            for random_seed in model_grid[ModelGridColumns.RANDOM_SEED]:
                optimizer = Adam(learning_rate=learning_rate)
                weight_initializer = RandomNormal(seed=random_seed)
                model = Sequential()
                model.add(Dense(layer_1_n, input_shape=(input_shape,), activation="relu",
                                kernel_initializer=weight_initializer,
                                bias_initializer=bias_initializer,
                                kernel_regularizer=L2(1e-4), bias_regularizer=L2(1e-4)))
                if num_of_add_layers >= 1:
                    model.add(Dense(layer_2_n, activation="relu", kernel_initializer=weight_initializer,
                                    bias_initializer=bias_initializer, kernel_regularizer=L2(1e-4),
                                    bias_regularizer=L2(1e-4)))
                    if num_of_add_layers == 2:
                        model.add(Dense(layer_3_n, activation="relu", kernel_initializer=weight_initializer,
                                        bias_initializer=bias_initializer, kernel_regularizer=L2(1e-4),
                                        bias_regularizer=L2(1e-4)))
                model.add(Dense(1, kernel_initializer=weight_initializer,
                                bias_initializer=bias_initializer, kernel_regularizer=L2(1e-4),
                                bias_regularizer=L2(1e-4)))
                model.compile(optimizer=optimizer, loss=MeanSquaredError(),
                              metrics=["mse", "mae"])
                history = model.fit(x=x_train, y=y_train, epochs=FFNTradeAlgorithm.epochs,
                                    batch_size=FFNTradeAlgorithm.batch_size, shuffle=False,
                                    validation_split=0.15, callbacks=[es], verbose=0)
                model_loss = history.history['val_mse'][-1]

                print(f"MSE = {history.history['mse'][-1]} Val MSE ={model_loss}")
                if (best_model is None) or model_loss < best_val_loss:
                    best_model = model
                    best_val_loss = model_loss
                    best_model_params = model_params

        self._model = best_model
        self._model_params = best_model_params

    def train(self, data: pd.DataFrame, hyperparameters: Dict):
        super().train(data, hyperparameters)
        self.__clear_vars()
        self._prediction_period = hyperparameters[FFNTradeAlgorithmHyperparam.PREDICTION_PERIOD]
        self._test_period_size = hyperparameters[FFNTradeAlgorithmHyperparam.TEST_PERIOD_SIZE]
        self._take_action_barrier = hyperparameters[FFNTradeAlgorithmHyperparam.TAKE_ACTION_BARRIER]
        self._active_action_multiplier = hyperparameters[FFNTradeAlgorithmHyperparam.ACTIVE_ACTION_MULTIPLIER]
        self._data_name = hyperparameters["DATA_NAME"]
        self.__define_model_name()

        self.__create_train_dataset()
        train_dataframe = self._dataframe[:-self._test_period_size]
        test_dataframe = self._dataframe[-self._test_period_size:]
        y_train = train_dataframe[f"target t + {self._prediction_period}"]
        x_train = train_dataframe.drop(f"target t + {self._prediction_period}", axis=1)
        y_test = test_dataframe[f"target t + {self._prediction_period}"]
        x_test = test_dataframe.drop(f"target t + {self._prediction_period}", axis=1)
        y_train = y_train.values
        x_train = self._input_scaler.fit_transform(x_train)

        if exists(self._model_name + ".h5"):
            self.__load_model()
        else:
            self.__choose_best_model(x_train, y_train)

        print("Model params")
        print(self._model_params)

        closes_test = x_test["Close t - 0"]
        x_test = self._input_scaler.transform(x_test)
        predictions = self._model.predict(x=x_test)
        cum_sum = 0
        for i in range(0, len(predictions)):
            cur_close = closes_test[i]
            abs_relative_diff = np.abs((predictions[i][0] - y_test[i]) / cur_close)
            cum_sum += abs_relative_diff
        self.mean_absolute_prediction_error = cum_sum / len(predictions)

        print(f"MARGE = {self.mean_absolute_prediction_error}")

        x = self._dataframe.drop(f"target t + {self._prediction_period}", axis=1)
        x = self._input_scaler.transform(x)
        y = self._dataframe[f"target t + {self._prediction_period}"]
        es = EarlyStopping(min_delta=1e-8, patience=10, verbose=0)
        history = self._model.fit(x=x, y=y, epochs=FFNTradeAlgorithm.epochs, batch_size=FFNTradeAlgorithm.batch_size,
                                  validation_split=0.15, shuffle=False, callbacks=[es], verbose=0)
        print(f"Loss of best model {history.history['val_mse'][-1]}")
        self.__save_model()

        print(self._model.weights[0][0])

        predictions = self._model.predict(x=x, batch_size=FFNTradeAlgorithm.batch_size)
        self.predictions = predictions.flatten().tolist()
        self._last_train_date = self.data.index[-(self._prediction_period + 1)]
        self._last_train_date_index = self.data.shape[0] - (self._prediction_period + 1)

        for i in range(self._prediction_period, 0, -1):
            x = self.__transform_point(i)
            prediction = self._model.predict(x=x, verbose=0)
            self.predictions.append(prediction[0][0])

    def __transform_point(self, index: int = 1) -> np.ndarray:
        new_point_dict = {}

        for i in range(0, FFNTradeAlgorithm.look_back_period):
            new_point_dict[f"Close t - {i}"] = self.data["Close"][-(i + index)]
            new_point_dict[f"Open t - {i}"] = self.data["Open"][-(i + index)]
            new_point_dict[f"High t - {i}"] = self.data["High"][-(i + index)]
            new_point_dict[f"Low t - {i}"] = self.data["Low"][-(i + index)]
            new_point_dict[f"Volume t - {i}"] = self.data["Volume"][-(i + index)]

        for min_max_period in FFNTradeAlgorithm.min_max_periods:
            new_point_dict[f"Max Close t - {min_max_period}"] = \
                self.data[-(min_max_period + FFNTradeAlgorithm.min_max_window + index - 1):-min_max_period][
                    "Close"].max()
            new_point_dict[f"Min Close t - {min_max_period}"] = \
                self.data[-(min_max_period + FFNTradeAlgorithm.min_max_window + index - 1):-min_max_period][
                    "Close"].min()

        new_point_dataframe = pd.DataFrame([new_point_dict])
        new_point_scaled = self._input_scaler.transform(new_point_dataframe)
        return new_point_scaled

    def evaluate_new_point(self, new_point: pd.Series, date: Union[str, pd.Timestamp],
                           special_params: Optional[Dict] = None) -> TradeAction:
        self.data.loc[date] = new_point
        last_close = new_point["Close"]
        final_action = TradeAction.NONE
        new_point_scaled = self.__transform_point()
        prediction = self._model.predict(new_point_scaled, verbose=0)[0][0]
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

    def plot(self, start_date: Optional[pd.Timestamp] = None, end_date: Optional[pd.Timestamp] = None,
             show_full: bool = False):
        intend = FFNTradeAlgorithm.min_max_periods[-1] + FFNTradeAlgorithm.min_max_window + self._prediction_period - 1
        selected_data = self.data[intend:]
        selected_predictions = self.predictions[:-self._prediction_period - 1]
        if not show_full:
            watch_intend = selected_data.shape[0] - 600
            selected_data = selected_data[watch_intend:]
            selected_predictions = selected_predictions[watch_intend:]

        fig = go.Figure()

        title = f"FFN model with params LR={self._model_params[ModelGridColumns.LEARNING_RATE.name]}, L1={self._model_params[ModelGridColumns.LAYER_1.name]}"
        if self._model_params[ModelGridColumns.NUM_OF_ADD_LAYERS.name] > 0:
            title += f", L2={self._model_params[ModelGridColumns.LAYER_2.name]}"
            if self._model_params[ModelGridColumns.NUM_OF_ADD_LAYERS.name] == 2:
                title += f", L2={self._model_params[ModelGridColumns.LAYER_3.name]}"
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

        fig.show()

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

        fig.show()
