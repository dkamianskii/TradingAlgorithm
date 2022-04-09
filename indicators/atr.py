import pandas as pd
import numpy as np
from typing import Union

from numpy import double

import indicators.moving_averages as ma


def ATR(data: pd.DataFrame, N: int = 14) -> pd.Series:
    """
    Average true range (ATR) is a technical analysis volatility indicator.
    The average true range is an N-period smoothed moving average (SMMA) of the true range values.
    Wilder recommended a 14-period smoothing.

    The true range is the largest of the:
        * Most recent period's high minus the most recent period's low
        * Absolute value of the most recent period's high minus the previous close
        * Absolute value of the most recent period's low minus the previous close

    :param data: dataframe of a stock data in format of yahoo finance dataframe with columns "High", "Low", "Close".
    :param N: smoothing period for SMMA calculation.
    """
    high = data["High"][1:].to_numpy()
    low = data["Low"][1:].to_numpy()
    close_prev = data["Close"][:-1].to_numpy()

    tr = np.maximum(high, close_prev) - np.minimum(low, close_prev)
    atr = ma.SMMA(tr, N)
    atr = pd.Series(data=atr, index=data.index[N:])
    return atr


def ATR_one_point(prev_atr: Union[float, double, int],
                  prev_close: Union[float, double, int],
                  new_point: pd.Series,
                  N: int) -> float:
    high = new_point["High"]
    low = new_point["Low"]

    tr = np.maximum(high, prev_close) - np.minimum(low, prev_close)
    return ma.SMMA_one_point(prev_atr, tr, N)
