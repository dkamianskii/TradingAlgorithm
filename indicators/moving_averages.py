# -*- coding: utf-8 -*-
"""
Library for moving averages
Created on Mon Dec 13 13:05:20 2021

@author: aspod
"""

import pandas as pd
import numpy as np
from typing import Union, Optional, Sequence

# Typing
from numpy import double

Realization = Sequence[float]


# Simple moving average

def SMA(time_series: Union[pd.Series, Realization], N: int) -> np.ndarray:
    """
        Compute Simple Moving Average for time series

        :param time_series: Input time series or a sequence of floats
        :param N: Number of points used in averaging
    """
    if isinstance(time_series, pd.Series):
        ts = time_series
    else:
        ts = pd.Series(time_series)

    if (N > len(time_series)) or (N <= 0):
        raise ValueError("N parameter must be > 0 and less then length of the time_series")

    return ts.rolling(N).mean().to_numpy()[(N - 1):]


# EMA - Exponential moving average

def EMA(time_series: Union[pd.Series, Realization], N: int,
        alpha: Optional[float] = None,
        ema_0_init: Optional[str] = "first") -> np.ndarray:
    """
        Compute Exponential Moving Average for time series
        Calculation Formula EMA(i) = alpha*time_series[i] + (1 - alpha)*EMA(i - 1)

        :param time_series: Input time series or a sequence of floats
        :param N: Determines alpha if it's not specified and number of points in SMA,
         using for calculating first EMA
        :param alpha: Smoothing coefficient, must be in (0,1) range. The bigger it is
         the more weight last element in averaging gains. By default, calculates as 2 / (N + 1)
        :param ema_0_init: Determines the way first EMA is calculated. Could be 'first' or 'mean N'.
         If 'first' then EMA(0) = time_series[0], if 'mean N' then EMA(0) = SMA(time_series[0:N - 1]).
         Equal 'first' by default.
    """
    if (N > len(time_series)) or (N <= 0):
        raise ValueError("N parameter must be > 0 and less then length of the time_series")

    if alpha is None:
        alpha = 2 / (N + 1)
    elif (alpha <= 0) or (alpha > 1):
        raise ValueError("alpha parameter must be in (0,1) boundaries")

    indent = 0

    if ema_0_init == "mean N":
        final_length = len(time_series) - N + 1
        ema = np.empty(final_length)
        ema[0] = np.mean(time_series[0:N])
        indent = N - 1
    elif ema_0_init == "first":
        final_length = len(time_series)
        ema = np.empty(final_length)
        ema[0] = time_series[0]
    else:
        raise ValueError("ema_0_init parameter could be only 'mean N' or 'first'")

    for i in range(1, final_length):
        ema[i] = alpha * time_series[indent + i] + (1 - alpha) * ema[i - 1]

    return ema


def EMA_one_point(prev_ema: Union[float, double, int],
                  new_point: Union[float, double, int],
                  N: int, alpha: Optional[float] = None) -> float:
    if alpha is None:
        alpha = 2 / (N + 1)
    return alpha * new_point + (1 - alpha) * prev_ema


# SMMA - Smoothed moving average

def SMMA(time_series: Union[pd.Series, Realization], N: int) -> np.ndarray:
    """
        Compute Smoothed Moving Average for time series
        Equals EMA with parameter alpha = 1 / N, and EMA(0) = SMA(time_series[0:N - 1])

        :param time_series: Input time series or a sequence of floats
        :param N: Determines alpha (alpha = 1 / N) and number of points in SMA,
         using for calculating first EMA
    """
    return EMA(time_series, N, alpha=(1 / N), ema_0_init="mean N")


def SMMA_one_point(prev_smma: Union[float, double, int],
                   new_point: Union[float, double, int],
                   N: int) -> float:
    return EMA_one_point(prev_smma, new_point, N, alpha=(1 / N))
