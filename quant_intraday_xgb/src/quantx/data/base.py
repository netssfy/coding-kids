from abc import ABC, abstractmethod
import pandas as pd

class DataProvider(ABC):
    @abstractmethod
    def load_bars(self, symbol: str) -> pd.DataFrame:
        """Return df indexed by datetime with columns: open, high, low, close, volume"""
        raise NotImplementedError
