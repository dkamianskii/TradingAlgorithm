from plotly.subplots import make_subplots

import indicators.moving_averages as ma
import numpy as np
import pandas as pd
from typing import Optional, List, Tuple, Dict, Union
from indicators.abstract_indicator import AbstractIndicator, TradeAction, TradePointColumn
from helping.base_enum import BaseEnum

import cufflinks as cf
import plotly.graph_objects as go

cf.go_offline()


class MAsColumns(BaseEnum):
    TOP_BORDER = 1,
    MAIN = 2,
    BOTTOM_BORDER = 3


class MASupportLevels(AbstractIndicator):

    default_ma_periods_for_test = [20, 30, 50, 75, 100, 150, 200]

    def __init__(self, data: Optional[pd.DataFrame] = None,
                 ma_periods: Optional[List] = None,
                 use_tested_MAs: bool = False):
        super().__init__(data)
        self.ma_periods: Optional[List] = None
        self._use_tested_MAs: bool = use_tested_MAs
        if ma_periods is not None:
            self.set_ma_periods(ma_periods)
        self.tested_MAs: Optional[Dict] = None
        self.MAs: Dict[int, pd.DataFrame] = {}
        self.MA_test_results: Optional[List] = None

    def set_ma_periods(self, ma_periods: List[int]):
        self.ma_periods = ma_periods
        return self

    def set_tested_MAs_usage(self, use_tested_MAs: bool):
        self._use_tested_MAs = use_tested_MAs

    def clear_vars(self):
        super().clear_vars()

    def evaluate_new_point(self, new_point: pd.Series, date: Union[str, pd.Timestamp], special_params: Optional = None,
                           update_data: bool = True) -> TradeAction:
        if self._use_tested_MAs:
            if self.tested_MAs is None:
                self.test_MAs_for_data()
        resistance, support = 0, 0
        activation_flag = False
        for period, MA in self.MAs.items():
            new_ma = ma.EMA_one_point(MA[MAsColumns.MAIN][-1], new_point["Close"], period)
            new_MA_point = {MAsColumns.TOP_BORDER: new_ma * 1.002,
                            MAsColumns.MAIN: new_ma,
                            MAsColumns.BOTTOM_BORDER: new_ma * 0.998}
            is_activation, activation_type = self.__activation_point_check(new_point, new_MA_point)
            if (not self._use_tested_MAs) or (period in self.tested_MAs.keys()):
                if is_activation:
                    activation_flag = True
                    if activation_type == "resistance":
                        resistance += 1
                    else:
                        support += 1
            self.MAs[period].loc[date] = new_MA_point
        if update_data:
            self.data.loc[date] = new_point
        return self.__make_trade_decision(activation_flag, resistance, support, new_point, date)

    @staticmethod
    def __activation_point_check(point, MA_point) -> Tuple[bool, Optional[str]]:
        """
        Checks if provided data point is activation point for provided MA and support or resistance it is

        :param point: data point with fields "Open", "Close", "High", "Low"
        :param MA_point: dict with fields MAsColumns.MAIN, MAsColumns.TOP_BORDER, MAsColumns.BOTTOM_BORDER. All fields are float numbers
        :return: tuple(is point activation:bool, activation type: "support" or "resistance")
        """
        if (point["Open"] > MA_point[MAsColumns.MAIN]) and (point["Open"] > point["Close"]):  # support
            if (point["Close"] <= MA_point[MAsColumns.TOP_BORDER]) or (point["Low"] <= MA_point[MAsColumns.TOP_BORDER]):
                return True, "support"
        elif (point["Open"] < MA_point[MAsColumns.MAIN]) and (point["Open"] < point["Close"]):  # resistance
            if (point["Close"] >= MA_point[MAsColumns.BOTTOM_BORDER]) or (point["High"] >= MA_point[MAsColumns.BOTTOM_BORDER]):
                return True, "resistance"

        return False, None

    def __make_trade_decision(self, activation_flag: bool, resistance: int, support: int,
                              new_point: pd.Series, date: Union[str, pd.Timestamp]) -> TradeAction:
        if activation_flag:
            if (resistance == 1) and (support == 0):
                trade_action = TradeAction.SELL
            elif (resistance == 0) and (support == 1):
                trade_action = TradeAction.BUY
            elif ((resistance == 0) and (support == 0)) or (np.abs(resistance - support) <= 1):
                trade_action = TradeAction.NONE
            elif resistance - support == 2:
                trade_action = TradeAction.SELL
            elif support - resistance == 2:
                trade_action = TradeAction.BUY
            elif resistance - support >= 3:
                trade_action = TradeAction.ACTIVELY_SELL
            elif support - resistance >= 3:
                trade_action = TradeAction.ACTIVELY_BUY
            else:
                trade_action = TradeAction.NONE
        else:
            trade_action = TradeAction.NONE
        self.add_trade_point(date, new_point["Close"], trade_action)
        return trade_action

    def calculate(self, data: Optional[pd.DataFrame] = None):
        super().calculate(data)
        if self.ma_periods is None:
            self.ma_periods = MASupportLevels.default_ma_periods_for_test
        for period in self.ma_periods:
            if self.data.shape[0] <= period:
                continue
            m_a = ma.EMA(self.data["Close"], N=period)
            MA_data = {MAsColumns.TOP_BORDER: m_a * 1.002,
                       MAsColumns.MAIN: m_a,
                       MAsColumns.BOTTOM_BORDER: m_a * 0.998}
            self.MAs[period] = pd.DataFrame(data=MA_data, index=self.data.index)
        return self

    def test_MAs_for_data(self, days_for_bounce: int = 4) -> List:
        """
        Tests provided MAs for price response to crossing support/resistance borders
        and saves MAs that got more than 50% positive response (price bounce) to those cross points

        :param days_for_bounce: days range (from day when price cross MA line) in which price have to bounce
        """
        if self.MAs is None:
            raise SyntaxError("test_MAs_for_data can be called only after calculate method was called at least once")
        self.MA_test_results = []
        self.tested_MAs = {}
        for period in self.MAs.keys():
            result = self.__test_MA(period, self.MAs[period], days_for_bounce)
            self.MA_test_results.append((period, result))
            if (result["activations"] <= 3) or (result["successes"] / result["activations"] >= 0.45):
                self.tested_MAs[period] = self.MAs[period]

        return self.MA_test_results

    def __test_MA(self, period: int, MA: pd.DataFrame, days_for_bounce: int) -> Dict:
        """
        Проходим по всем точкам:
        1) Сначала определяем, является ли в данный момент МА линией поддержки или сопротивления.
        Для этого смотрим на значение Open в данной точке. Если оно над МА, то МА - поддержка. Иначе сопротивление.
        Далее случай для поддержки:
        2) Если в точке Close или Low пересекают верхнюю границу МА, то такое событие считается Активацией.
        3) После Активации смотрятся следующие ближайшие точки (4):
         Если Close не опускается ниже нижней границы МА, но меньше Open или меньше Close в момент Активации, то смотрим следующую.
         Если Close выше Close в момент Активации и выше Open, то Активация считается Успешной.
         Если Close опускается ниже нижней границы МА, то Активация считается неуспешной.
        4) Считается общее количество Активаций и Успехов (для тестирования записываются индексы активации и тип активации в dict)
        """
        if days_for_bounce < 2:
            raise ValueError("days for bounce can't be less then 2")

        activation_points = {"Dates": [], "activation types": [], "results": []}
        activations, successes = 0, 0
        activation_point = None
        activation_flag: bool = False
        activation_type: Optional[str] = None
        days_counter = 0
        for date, MA_point in MA[period:].iterrows():
            point = self.data.loc[date]
            if activation_flag:
                if days_counter > days_for_bounce:
                    activation_points["results"].append("fail")
                    activation_flag = False
                elif activation_type == "support":
                    if point["Close"] < activation_point["Close"]:
                        activation_points["results"].append("fail")
                        activation_flag = False
                    elif (point["Close"] > activation_point["Close"]) and (point["Close"] > point["Open"]):
                        successes += 1
                        activation_points["results"].append("success")
                        activation_flag = False
                else:
                    if point["Close"] > activation_point["Close"]:
                        activation_points["results"].append("fail")
                        activation_flag = False
                    elif (point["Close"] < activation_point["Close"]) and (point["Close"] < point["Open"]):
                        successes += 1
                        activation_points["results"].append("success")
                        activation_flag = False
                days_counter += 1

            if not activation_flag:
                activation_flag, activation_type = self.__activation_point_check(point, MA_point)
                if activation_flag:
                    activations += 1
                    days_counter = 1
                    activation_points["Dates"].append(date)
                    activation_points["activation types"].append(activation_type)
                    activation_point = point

        if len(activation_points["Dates"]) != len(activation_points["results"]):
            activation_points["results"].append("none")
        return {"activations": activations, "successes": successes,
                "activation_points": pd.DataFrame(activation_points)}

    def find_trade_points(self) -> pd.DataFrame:
        intend = np.max([*self.MAs.keys()])
        for date, point in self.data[intend:].iterrows():
            resistance, support = 0, 0
            activation_flag = False
            for period, MA in self.MAs.items():
                MA_point = MA.loc[date]
                is_activation, activation_type = self.__activation_point_check(point, MA_point)
                if (not self._use_tested_MAs) or (period in self.tested_MAs.keys()):
                    if is_activation:
                        activation_flag = True
                        if activation_type == "resistance":
                            resistance += 1
                        else:
                            support += 1
            self.__make_trade_decision(activation_flag, resistance, support, point, date)
        return self.trade_points

    def plot(self, start_date: Optional[pd.Timestamp] = None, end_date: Optional[pd.Timestamp] = None):
        """
        Plots the candle graph for data and MAs with highlighted trade points in specified time diapason
        """
        if (start_date is None) or (start_date < self.data.index[0]):
            start_date = self.data.index[0]
        if (end_date is None) or (end_date > self.data.index[-1]):
            end_date = self.data.index[-1]

        selected_data = self.data[start_date:end_date]

        fig = go.Figure()

        fig.add_candlestick(x=selected_data.index,
                            open=selected_data["Open"],
                            close=selected_data["Close"],
                            high=selected_data["High"],
                            low=selected_data["Low"],
                            name="Price")

        for period, MA in self.MAs.items():
            if ((self.tested_MAs is not None) and (len(self.tested_MAs.keys()) != 0) and (
                    period in self.tested_MAs.keys())):
                fig.add_trace(go.Scatter(x=MA.index, y=MA[MAsColumns.MAIN], mode='lines',
                                         line=dict(width=2), name=f"{period} EMA tested"))
            else:
                fig.add_trace(go.Scatter(x=MA.index, y=MA[MAsColumns.MAIN], mode='lines',
                                         line=dict(width=1, dash='dash'), name=f"{period} EMA"))

        selected_trade_points = self.select_action_trade_points(start_date=start_date, end_date=end_date)

        bool_buys = selected_trade_points[TradePointColumn.ACTION].isin(self.buy_actions)
        bool_actives = selected_trade_points[TradePointColumn.ACTION].isin(self.active_actions)
        fig.add_trace(go.Scatter(x=selected_trade_points.index,
                                 y=selected_trade_points[TradePointColumn.PRICE],
                                 mode="markers",
                                 marker=dict(
                                     color=np.where(bool_buys, "green", "red"),
                                     size=np.where(bool_actives, 15, 10),
                                     symbol=np.where(bool_buys, "triangle-up", "triangle-down")),
                                 name="Action points"))

        fig.update_layout(title="Price with Moving Averages",
                          xaxis_title="Date")

        fig.show()
