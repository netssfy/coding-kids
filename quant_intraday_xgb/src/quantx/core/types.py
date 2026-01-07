# src/quantx/core/types.py
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime

@dataclass(frozen=True)
class Bar:
    symbol: str
    dt: datetime          # tz-aware
    open: float
    high: float
    low: float
    close: float
    volume: float

@dataclass(frozen=True)
class Signal:
    symbol: str
    dt: datetime          # 使用的 ts_use（t-1）
    side: int             # 1=BUY, -1=SELL, 0=HOLD
    score: float          # p_reversal
    reason: str
