# -*- coding: utf-8 -*-
"""
Library for moving averages
Created on Mon Dec 13 13:05:20 2021

@author: aspod
"""

import pandas as pd
import numpy as np


# Simple moving average

def SMA(time_series, N):
    """
        Compute Simple Moving Average for time_series
    """
    if (isinstance(time_series, pd.core.series.Series)):
        ts = time_series
    elif (isinstance(time_series, np.ndarray) or isinstance(time_series, list)):
        ts = pd.Series(time_series)
    else:
        raise TypeError("time_series parameter must be pandas Series, list or numpy ndarray")

    if (not isinstance(N, int)):
        raise TypeError("N parameter must be int")
    if ((N >= len(time_series)) or (N < 0)):
        raise ValueError("N parameter must be >= 0 and less then lenght of the time_series")

    return ts.rolling(N).mean().to_numpy()[(N - 1):]


# EMA - Exponential moving average

def EMA(time_series: "pandas Series, list or numpy ndarray",
        N: "int >= 0 ,number of previous data points for averaging",
        alpha: "float in (0,1) interval, smoothing factor" = None,
        ema_0_init: "str: 'mean N' or 'first', definied the way of initialization of the first ema:" +
                    "first value can be chosen or can be calculated as mean of first N values" = "first"
        ) -> "numpy ndarray of averaged values, shorter then original time_series by size of N":
    """
        Compute Exponential Moving Average for time_series
    """
    if (not (isinstance(time_series, pd.core.series.Series) or
             isinstance(time_series, list) or
             isinstance(time_series, np.ndarray))):
        raise TypeError("time_series parameter must be pandas Series, list or numpy ndarray")

    if (not isinstance(N, int)):
        raise TypeError("N parameter must be int")
    if ((N >= len(time_series)) or (N < 0)):
        raise ValueError("N parameter must be >= 0 and less then lenght of the time_series")

    if (alpha is None):
        alpha = 2 / (N + 1)
    elif (not isinstance(alpha, float)):
        raise TypeError("alpha parameter must be float")
    elif ((alpha <= 0) or (alpha >= 1)):
        raise ValueError("alpha parameter must be in (0,1) boundaries")

    ema = None
    final_lenght = 0
    indent = 0

    if (ema_0_init == "mean N"):
        final_lenght = len(time_series) - N + 1
        ema = np.ndarray(final_lenght)
        ema[0] = np.mean(time_series[0:N])
        indent = N - 1
    elif (ema_0_init == "first"):
        final_lenght = len(time_series)
        ema = np.ndarray(final_lenght)
        ema[0] = time_series[0]
    else:
        raise TypeError("ema_0_init parameter must be str: 'mean N' or 'first'")

    for i in range(1, final_lenght):
        ema[i] = alpha * time_series[indent + i] + (1 - alpha) * ema[i - 1]
    return ema


# SMMA - Smoothed moving average

def SMMA(time_series: "pandas Series, list or numpy ndarray",
         N: "int >= 0 ,number of previous data points for averaging") -> "numpy ndarray of averaged values, shorter then original time_series by size of N":
    """
        Compute Smoothed Moving Average for time_series
    """
    return EMA(time_series, N, (1 / N), ema_0_init="mean N")
