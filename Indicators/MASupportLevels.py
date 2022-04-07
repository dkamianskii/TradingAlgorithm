import Indicators.moving_averages as ma
import numpy as np
import pandas as pd
from typing import Optional, List, Tuple, Dict
from Indicators.AbstractIndicator import AbstractIndicator, TradeAction

import plotly as py
import plotly.express as px
import cufflinks as cf
import plotly.graph_objects as go

cf.go_offline()


class MASupportLevels(AbstractIndicator):

    def __init__(self, data: Optional[pd.DataFrame] = None,
                 ma_periods: List = None):
        super().__init__(data)
        self.ma_periods: Optional[List] = None
        if ma_periods is not None:
            self.set_ma_periods(ma_periods)
        self.tested_MAs: Optional[Dict] = None
        self.MAs: Optional[Dict[int, pd.DataFrame]] = None
        self.MA_test_results: Optional[List] = None

    def set_ma_periods(self, ma_periods: List[int]):
        self.ma_periods = ma_periods
        return self

    default_ma_periods_for_test = [20, 30, 50, 75, 100, 150, 200]

    def __calculate_MAs(self, ma_periods) -> Dict:
        MAs = {period: ma.EMA(self.data["Close"], N=period) for period in ma_periods if self.data.shape[0] > period + 1}
        for period in MAs.keys():
            m_a = MAs[period]
            MA_data = {"top border": m_a * 1.002, "main": m_a,
                       "bottom border": m_a * 0.998}
            MAs[period] = pd.DataFrame(data=MA_data, index=self.data.index)
        return MAs

    def calculate(self, data: Optional[pd.DataFrame] = None):
        super().calculate(data)
        if self.ma_periods is None:
            self.ma_periods = MASupportLevels.default_ma_periods_for_test
        self.MAs = self.__calculate_MAs(self.ma_periods)
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
            result = self.__test_MA(self.MAs[period], days_for_bounce)
            self.MA_test_results.append((period, result))
            if (result["activations"] <= 3) or (result["successes"] / result["activations"] >= 0.45):
                self.tested_MAs[period] = self.MAs[period]

        return self.MA_test_results

    def __test_MA(self, MA: pd.DataFrame, days_for_bounce: int = 4) -> Dict:
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
        if (days_for_bounce < 2):
            raise ValueError("days for bounce can't be less then 2")

        activation_points = {"Dates": [], "activation types": [], "results": []}
        activations, successes = 0, 0
        activation_point = None
        activation_flag: bool = False
        activation_type: Optional[str] = None
        days_counter = 0
        for date, MA_point in MA.iterrows():
            point = self.data.loc[date]
            if activation_flag:
                if days_counter == days_for_bounce:
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

        return {"activations": activations, "successes": successes,
                "activation_points": pd.DataFrame(activation_points)}

    def __activation_point_check(self, point, MA_point) -> Tuple[bool, Optional[str]]:
        """
        Checks if provided data point is activation point for provided MA and support or resistance it is

        :param point: data point with fields "Open", "Close", "High", "Low"
        :param MA_point: dict with fields "main", "top border", "bottom border". All fields are float numbers
        :return: tuple(is point activation:bool, activation type: "support" or "resistance")
        """
        if (point["Open"] > MA_point["main"]) and (point["Open"] > point["Close"]):  # support
            if (point["Close"] <= MA_point["top border"]) or (point["Low"] <= MA_point["top border"]):
                return True, "support"
        elif (point["Open"] < MA_point["main"]) and (point["Open"] < point["Close"]):  # resistance
            if (point["Close"] >= MA_point["bottom border"]) or (point["High"] >= MA_point["bottom border"]):
                return True, "resistance"

        return False, None

    def find_trade_points(self, use_tested_MAs: bool = False) -> pd.DataFrame:
        if use_tested_MAs:
            if self.tested_MAs is None:
                self.test_MAs_for_data()
            MAs_in_use = self.tested_MAs
        else:
            MAs_in_use = self.MAs

        for date, point in self.data[1:].iterrows():
            resistance, support = 0, 0
            activation_flag = False
            if date == pd.Timestamp(ts_input="2021-01-19"):
                i = 1
            for period, MA in MAs_in_use.items():
                MA_point = MA.loc[date]
                is_activation, activation_type = self.__activation_point_check(point, MA_point)
                if is_activation:
                    activation_flag = True
                    if activation_type == "resistance":
                        resistance += 1
                    else:
                        support += 1

            if activation_flag:
                if (resistance == 1) and (support == 0):
                    self.add_trade_point(date, TradeAction.SELL)
                    continue
                if (resistance == 0) and (support == 1):
                    self.add_trade_point(date, TradeAction.BUY)
                    continue
                if ((resistance == 0) and (support == 0)) or (np.abs(resistance - support) <= 1):
                    continue
                if resistance - support == 2:
                    self.add_trade_point(date, TradeAction.SELL)
                    continue
                if support - resistance == 2:
                    self.add_trade_point(date, TradeAction.BUY)
                    continue
                if resistance - support >= 3:
                    self.add_trade_point(date, TradeAction.ACTIVELY_SELL)
                    continue
                if support - resistance >= 3:
                    self.add_trade_point(date, TradeAction.ACTIVELY_BUY)
                    continue

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
                fig.add_trace(go.Scatter(x=MA.index, y=MA["main"], mode='lines',
                                         line=dict(width=2), name=f"{period} EMA tested"))
            else:
                fig.add_trace(go.Scatter(x=MA.index, y=MA["main"], mode='lines',
                                         line=dict(width=1, dash='dash'), name=f"{period} EMA"))

        selected_trade_points = self.select_action_trade_points(start_date=start_date, end_date=end_date)

        buy_actions = [TradeAction.BUY, TradeAction.ACTIVELY_BUY]
        bool_arr = selected_trade_points["Action"].isin(buy_actions)
        fig.add_trace(go.Scatter(x=selected_trade_points.index,
                                 y=selected_trade_points["Price"],
                                 mode="markers",
                                 marker=dict(
                                     color=np.where(bool_arr, "green", "red"),
                                     size=7,
                                     symbol=np.where(bool_arr, "triangle-up", "triangle-down")),
                                 name="Action points"),
                      row=1, col=1)

        fig.update_layout(title="Price with Moving Averages",
                          xaxis_title="Date")

        fig.show()
