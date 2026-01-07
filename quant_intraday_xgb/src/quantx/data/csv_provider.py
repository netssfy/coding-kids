import pandas as pd
from .base import DataProvider

class CSVProvider(DataProvider):
    def __init__(self, path: str):
        self.path = path

    def load_bars(self, symbol: str) -> pd.DataFrame:
        df = pd.read_csv(self.path, parse_dates=["datetime"])
        df = df.sort_values("datetime").set_index("datetime")
        cols = ["open", "high", "low", "close", "volume"]
        df = df[cols].astype(float)
        df = df[~df.index.duplicated(keep="last")]
        return df