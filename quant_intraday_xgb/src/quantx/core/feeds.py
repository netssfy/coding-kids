# src/quantx/core/feeds.py
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from typing import Iterator, Optional
import pandas as pd
from .types import Bar

@dataclass
class ReplayFeed:
    symbol: str
    bars_df: pd.DataFrame   # index=datetime tz-aware, cols=open/high/low/close/volume

    def __iter__(self) -> Iterator[Bar]:
        df = self.bars_df.sort_index()
        for ts, row in df.iterrows():
            yield Bar(
                symbol=self.symbol,
                dt=ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts,
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row.get("volume", 0.0)),
            )

class LiveFeed:
    """
    每次轮询 provider.load_bars 拿到最新的一段分钟线，增量吐出新 bar。
    """
    def __init__(
        self,
        provider,
        symbol_hk: str,          # akshare 用 "01810"
        symbol_name: str,        # 你的内部标识 "1810.HK" 或其它
        window: pd.Timedelta,
        poll_seconds: int = 60,
    ):
        self.provider = provider
        self.symbol_hk = symbol_hk
        self.symbol_name = symbol_name
        self.window = window
        self.poll_seconds = poll_seconds
        self._last_ts: Optional[pd.Timestamp] = None

    def iter_bars(self, now_func) -> Iterator[Bar]:
        import time

        while True:
            now = now_func()
            end = now.floor("min")  # 取到当前分钟开头（不代表闭合）
            start = end - self.window

            df = self.provider.load_bars(self.symbol_hk, start=start, end=end)
            if df is None or df.empty:
                time.sleep(self.poll_seconds)
                continue

            df = df.sort_index()

            if self._last_ts is not None:
                df = df[df.index > self._last_ts]

            for ts, row in df.iterrows():
                self._last_ts = ts
                yield Bar(
                    symbol=self.symbol_name,
                    dt=ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts,
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=float(row.get("volume", 0.0)),
                )

            time.sleep(self.poll_seconds)
