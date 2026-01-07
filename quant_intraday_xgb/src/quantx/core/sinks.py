# src/quantx/core/sinks.py
from __future__ import annotations
import csv
from pathlib import Path
from typing import Optional
from .types import Signal

class ConsoleSink:
    def on_signal(self, s: Signal) -> None:
        action = "BUY" if s.side == 1 else ("SELL" if s.side == -1 else "HOLD")
        print(f"[{s.dt:%Y-%m-%d %H:%M}] {s.symbol} {action} p={s.score:.4f} {s.reason}")

class CsvSink:
    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fp = open(self.path, "w", newline="")
        self.w = csv.writer(self.fp)
        self.w.writerow(["datetime", "symbol", "p_reversal", "side", "action", "reason"])
        self.fp.flush()

    def on_signal(self, s: Signal) -> None:
        action = "BUY" if s.side == 1 else ("SELL" if s.side == -1 else "HOLD")
        self.w.writerow([s.dt.isoformat(), s.symbol, f"{s.score:.6f}", s.side, action, s.reason])
        self.fp.flush()

    def close(self) -> None:
        try:
            self.fp.close()
        except Exception:
            pass
