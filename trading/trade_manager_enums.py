from helping.base_enum import BaseEnum


class BidType(BaseEnum):
    LONG = 1
    SHORT = 2


class BidResult(BaseEnum):
    WIN = 1
    LOSE = 2
    DRAW = 3


class DefaultTradeAlgorithmType(BaseEnum):
    MACD_SuperTrend = 1
    Indicators_council = 2
    Price_prediction = 3


class PortfolioColumn(BaseEnum):
    STOCK_NAME = 1
    PRICE = 2
    TYPE = 3
    AMOUNT = 4
    TAKE_PROFIT_LEVEL = 5
    STOP_LOSS_LEVEL = 6
    DATE = 7
    TRADE_ACTION = 8


class TrackedStocksColumn(BaseEnum):
    DATA = 1
    TRADE_ALGORITHM = 2
    PARAMS_GRID = 3
    CHOSEN_PARAMS = 4
    TRADING_START_DATE = 5
    LAST_ATR = 6
