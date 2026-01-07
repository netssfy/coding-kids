import pandas as pd
import yfinance as yf
from .base import DataProvider

class YFinanceProvider(DataProvider):
    """
    Yahoo Finance via yfinance.
    For intraday, history is usually limited (e.g., 1m~60m with short lookback).
    """
    def __init__(self, interval: str = "5m", period: str = "60d"):
        self.interval = interval
        self.period = period

    def load_bars(self, symbol: str) -> pd.DataFrame:
        t = yf.Ticker(symbol)
        df = t.history(interval=self.interval, period=self.period)

        if df is None or df.empty:
            raise ValueError(f"yfinance returned empty data for {symbol} interval={self.interval} period={self.period}")

        # yfinance columns: Open High Low Close Volume
        df = df.rename(columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        })

        # Keep only needed
        df = df[["open", "high", "low", "close", "volume"]].copy()

        # index is DatetimeIndex already
        df.index.name = "datetime"
        df = df.sort_index()
        df = df[~df.index.duplicated(keep="last")]

        # Ensure float dtype
        for c in ["open", "high", "low", "close", "volume"]:
            df[c] = df[c].astype(float)

        return df
