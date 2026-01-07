#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from quantx.config import Config
from daily_train_reversal import RevConfig, make_reversal_features
from quantx.data.akshare_provider import AkShareHKProvider

HK_TZ = "Asia/Hong_Kong"


# -----------------------------
# 2) 时间对齐：取 t-1 完整bar
# -----------------------------
def now_hk() -> pd.Timestamp:
    return pd.Timestamp.now(tz=HK_TZ)


def last_closed_minute(now: pd.Timestamp) -> pd.Timestamp:
    # t时刻运行，用 t-1 的完整bar
    return now.floor("min") - pd.Timedelta(minutes=1)


def sleep_until_next_minute(buffer_seconds: int = 2):
    n = now_hk()
    nxt = (n + pd.Timedelta(minutes=1)).floor("min") + pd.Timedelta(seconds=buffer_seconds)
    secs = (nxt - n).total_seconds()
    if secs > 0:
        time.sleep(secs)


# -----------------------------
# 3) 载入模型bundle（你 daily_train_reversal 保存的）
# -----------------------------
def load_model_bundle(model_path: Path):
    bundle = joblib.load(model_path)
    model = bundle["model"]
    feature_cols = bundle["feature_columns"]
    rev_cfg = RevConfig(**bundle["rev_cfg"]) if isinstance(bundle["rev_cfg"], dict) else bundle["rev_cfg"]
    return model, feature_cols, rev_cfg, bundle


# -----------------------------
# 4) 基于 t-1 bar 输出信号
# -----------------------------
def latest_signal_from_bars_tminus1(
    bars: pd.DataFrame,
    model,
    feature_cols: list[str],
    rev_cfg: RevConfig,
    p_threshold: float = 0.6,
):
    """
    bars: 必须包含 open/high/low/close/volume，index 为分钟级 datetime
    返回：ts_used(用于预测的bar时间=t-1), p, side, reason
    """
    if len(bars) < max(rev_cfg.vwap_win, rev_cfg.vol_win, rev_cfg.K) + 5:
        return None, np.nan, 0, "not_enough_bars"

    # 只用到“最后一个已闭合bar”
    ts_use = bars.index[-1]  # 我们外部已经确保 bars 截止到 last_closed_minute
    bars_upto = bars.loc[:ts_use]

    # 特征全量算（数据量不大时最稳）
    X = make_reversal_features(bars_upto, rev_cfg)
    if ts_use not in X.index:
        return ts_use, np.nan, 0, "feature_ts_missing"

    x_last = X.loc[[ts_use], feature_cols].copy()
    if not np.isfinite(x_last.values).all():
        return ts_use, np.nan, 0, "features_not_ready"

    # 方向（用 close[t] - close[t-K]）
    close = bars_upto["close"].astype(float)
    mom = close.loc[ts_use] - close.shift(rev_cfg.K).loc[ts_use]
    if not np.isfinite(mom) or mom == 0:
        direction = 0
    else:
        direction = 1 if mom > 0 else -1

    p = float(model.predict_proba(x_last)[:, 1][0])

    if p >= p_threshold and direction != 0:
        side = -direction  # 反转方向
        return ts_use, p, side, f"p>={p_threshold} & dir={direction} => side={side}"

    return ts_use, p, 0, "below_threshold_or_no_dir"


def emit_signal(log_path: Path, ts: pd.Timestamp, p: float, side: int, reason: str):
    log_path.parent.mkdir(parents=True, exist_ok=True)

    row = {
        "datetime_str": ts.strftime("%Y-%m-%d %H:%M:%S"),
        "p_reversal": p,
        "side": side,
        "action": "BUY" if side == 1 else ("SELL" if side == -1 else "HOLD"),
        "reason": reason,
    }
    print(row)

    df = pd.DataFrame([row])
    header = not log_path.exists()
    df.to_csv(log_path, mode="a", header=header, index=False)


# -----------------------------
# 5) 主循环：每分钟拉取数据并预测 t-1
# -----------------------------
def main():
    cfg = Config()
    model_N = 15

    project_root = Path(__file__).resolve().parents[1]

    today = now_hk().date().strftime("%Y-%m-%d")
    model_path = project_root / "models" / f"{cfg.symbol}_1m_revN{model_N}_{today}.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"model not found: {model_path}\n请先跑 daily_train_reversal.py 生成当天模型")

    model, feature_cols, rev_cfg, _ = load_model_bundle(model_path)
    print(f"[model] loaded: {model_path}")
    print(f"[model] rev_cfg={rev_cfg}, feature_cols={len(feature_cols)}")

    provider = AkShareHKProvider(period="1", adjust="")  # 1分钟
    p_threshold = 0.6

    log_path = project_root / "data" / "live_signals" / f"{cfg.symbol}_1m_{today}_ak.csv"

    last_emitted_ts = None

    while True:
        sleep_until_next_minute(buffer_seconds=2)

        now = now_hk()
        print(f"running at {now}")
        ts_last = last_closed_minute(now)  # 预测用的 bar 时间

        # 取最近窗口（建议至少覆盖：vwap_win/vol_win/K + 一些buffer）
        # 例如取最近 3 天分钟数据（按你机器/接口速度可调）
        start = ts_last - pd.Timedelta(days=3)
        end = ts_last

        bars = provider.load_bars(cfg.symbol, start=start, end=end)

        # 防呆：确保 bars 的最后一根就是 ts_last（否则说明数据延迟/缺bar）
        if len(bars) == 0:
            continue

        # 严格对齐到 ts_last：只用已闭合bar
        bars = bars.loc[:ts_last]
        print(f"last bar {bars.iloc[-1]}")

        if last_emitted_ts is not None and ts_last <= last_emitted_ts:
            continue

        ts_use, p, side, reason = latest_signal_from_bars_tminus1(
            bars, model, feature_cols, rev_cfg, p_threshold=p_threshold
        )

        if ts_use is not None:
            emit_signal(log_path, ts_use, p, side, reason)
            last_emitted_ts = ts_last


if __name__ == "__main__":
    main()
