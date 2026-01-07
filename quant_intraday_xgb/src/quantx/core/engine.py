# src/quantx/core/engine.py
from __future__ import annotations
from typing import Iterable, List, Optional
from .types import Bar, Signal

class TradingEngine:
    def __init__(self, strategy, sinks: list):
        self.strategy = strategy
        self.sinks = sinks

    def run(self, bars: Iterable[Bar]) -> None:
        try:
            for bar in bars:
                sig: Optional[Signal] = self.strategy.on_bar(bar)
                if sig is not None:
                    for s in self.sinks:
                        if hasattr(s, "on_signal"):
                            s.on_signal(sig)
        finally:
            for s in self.sinks:
                if hasattr(s, "close"):
                    s.close()
