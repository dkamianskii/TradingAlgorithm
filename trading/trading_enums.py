from helping.base_enum import BaseEnum


class BidType(BaseEnum):
    LONG = 1,
    SHORT = 2


class StocksStatisticsType(BaseEnum):
    EARNINGS_HISTORY = 1,
    BIDS_HISTORY = 2


class TradeResultColumn(BaseEnum):
    TOTAL = 0,
    STOCK_NAME = 1,
    EARNED_PROFIT = 2,
    WINS = 3,
    LOSES = 4,
    DRAWS = 5


class EarningsHistoryColumn(BaseEnum):
    DATE = 1,
    VALUE = 2


class BidsHistoryColumn(BaseEnum):
    DATE_OPEN = 1,
    OPEN_PRICE = 2,
    DATE_CLOSE = 3,
    CLOSE_PRICE = 4,
    TYPE = 5,
    RESULT_COLOR = 6