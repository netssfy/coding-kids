# src/quantx/strategies/reversal_xgb.py
from __future__ import annotations
from dataclasses import dataclass
from datetime import timedelta
import numpy as np
import pandas as pd
from typing import Callable, Optional
from quantx.core.types import Bar, Signal

# 复用你的逻辑
from daily_train_reversal import RevConfig, make_reversal_features

def latest_signal_from_bars_tminus1(
    bars: pd.DataFrame,
    model,
    feature_cols: list[str],
    rev_cfg: RevConfig,
    p_threshold: float = 0.6,
):
    if len(bars) < max(rev_cfg.vwap_win, rev_cfg.vol_win, rev_cfg.K) + 5:
        return None, np.nan, 0, "not_enough_bars"

    ts_use = bars.index[-1]   # bars 的最后一根视为“已闭合 bar”
    bars_upto = bars.loc[:ts_use]

    X = make_reversal_features(bars_upto, rev_cfg)
    if ts_use not in X.index:
        return ts_use, np.nan, 0, "feature_ts_missing"

    x_last = X.loc[[ts_use], feature_cols].copy()
    if not np.isfinite(x_last.values).all():
        return ts_use, np.nan, 0, "features_not_ready"

    close = bars_upto["close"].astype(float)
    mom = close.loc[ts_use] - close.shift(rev_cfg.K).loc[ts_use]
    if not np.isfinite(mom) or mom == 0:
        direction = 0
    else:
        direction = 1 if mom > 0 else -1

    p = float(model.predict_proba(x_last)[:, 1][0])

    if p >= p_threshold and direction != 0:
        side = -direction
        return ts_use, p, side, f"p>={p_threshold} & dir={direction} => side={side}"

    return ts_use, p, 0, "below_threshold_or_no_dir"


class ReversalXgbStrategy:
    """
    通用策略：吃 Bar，内部维护bars_df，输出 Signal（基于 t-1 已闭合 bar）。
    支持模型按“前一交易日”加载（避免数据泄漏）。
    """
    def __init__(
        self,
        feature_cols: list[str],
        rev_cfg: RevConfig,
        p_threshold: float,
        model_loader: Callable[[pd.Timestamp], object],   # 输入当前时刻 -> 返回模型（建议前一交易日模型）
        max_lookback_bars: int = 2000,                    # 控制计算量
        tz: str = "Asia/Hong_Kong",
    ):
        self.feature_cols = feature_cols
        self.rev_cfg = rev_cfg
        self.p_threshold = p_threshold
        self.model_loader = model_loader
        self.max_lookback_bars = max_lookback_bars
        self.tz = tz

        self._bars = pd.DataFrame(columns=["open","high","low","close","volume"])
        self._last_signal_ts: Optional[pd.Timestamp] = None
        self._model = None
        self._model_for_day: Optional[pd.Timestamp] = None  # day ts for caching

    def _append_bar(self, bar: Bar) -> pd.Timestamp:
        ts = pd.Timestamp(bar.dt).tz_convert(self.tz)
        row = pd.DataFrame(
            {"open":[bar.open], "high":[bar.high], "low":[bar.low], "close":[bar.close], "volume":[bar.volume]},
            index=[ts],
        )
        self._bars = pd.concat([self._bars, row])
        self._bars = self._bars[~self._bars.index.duplicated(keep="last")].sort_index()
        if len(self._bars) > self.max_lookback_bars:
            self._bars = self._bars.tail(self.max_lookback_bars)
        return ts

    def _ensure_model(self, now_ts: pd.Timestamp):
        # “交易日级”缓存模型：同一天内不重复 load
        day_key = now_ts.normalize()
        if self._model is None or self._model_for_day is None or self._model_for_day != day_key:
            self._model = self.model_loader(now_ts)
            self._model_for_day = day_key

    def on_bar(self, bar: Bar) -> Optional[Signal]:
        now_ts = self._append_bar(bar)

        # 统一：只用 t-1 已闭合bar
        ts_last_closed = now_ts.floor("min") - pd.Timedelta(minutes=1)
        bars_upto = self._bars.loc[:ts_last_closed]

        if bars_upto.empty:
            return None

        # 避免重复发同一个 timestamp 的信号
        if self._last_signal_ts is not None and bars_upto.index[-1] <= self._last_signal_ts:
            return None

        self._ensure_model(now_ts)

        ts_use, p, side, reason = latest_signal_from_bars_tminus1(
            bars_upto,
            self._model,
            self.feature_cols,
            self.rev_cfg,
            p_threshold=self.p_threshold,
        )
        if ts_use is None:
            return None

        self._last_signal_ts = ts_use

        return Signal(
            symbol=bar.symbol,
            dt=ts_use.to_pydatetime() if hasattr(ts_use, "to_pydatetime") else ts_use,
            side=int(side),
            score=float(p) if p == p else float("nan"),
            reason=str(reason),
        )
