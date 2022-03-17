import numpy as np
import pandas as pd
from typing import Optional

class TradeManager:

    def __init__(self):
        self.data: Optional[pd.DataFrame] = None
        self.trade_algorithm