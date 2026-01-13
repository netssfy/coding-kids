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

    cls_model = bundle["model"]
    feature_cols = bundle["feature_columns"]
    rev_cfg = RevConfig(**bundle["rev_cfg"]) if isinstance(bundle["rev_cfg"], dict) else bundle["rev_cfg"]

    reg_model = bundle.get("reg_model", None)
    reg_target = bundle.get("reg_target", None)

    return cls_model, feature_cols, rev_cfg, reg_model, reg_target, bundle



# -----------------------------
# 4) 基于 t-1 bar 输出信号
# -----------------------------
def latest_signal_from_bars_tminus1(
    bars: pd.DataFrame,
    cls_model,
    feature_cols: list[str],
    rev_cfg: RevConfig,
    reg_model=None,
    reg_target: str | None = None,
    p_threshold: float = 0.6,
    amp_threshold: float = 0.006,   # 例如：预测未来N分钟振幅至少 0.6% 才做
):
    if len(bars) < max(rev_cfg.vwap_win, rev_cfg.vol_win, rev_cfg.K) + 5:
        return None, np.nan, np.nan, 0, "not_enough_bars"

    ts_use = bars.index[-1]
    bars_upto = bars.loc[:ts_use]

    X = make_reversal_features(bars_upto, rev_cfg)
    if ts_use not in X.index:
        return ts_use, np.nan, np.nan, 0, "feature_ts_missing"

    x_last = X.loc[[ts_use], feature_cols].copy()
    if not np.isfinite(x_last.values).all():
        return ts_use, np.nan, np.nan, 0, "features_not_ready"

    # ---- 分类概率：是否反转 ----
    p = float(cls_model.predict_proba(x_last)[:, 1][0])

    # ---- 回归幅度：未来N分钟振幅/上行/下行等 ----
    amp_pred = np.nan
    if reg_model is not None:
        amp_pred = float(reg_model.predict(x_last)[0])   # pred_range_amp / pred_up_amp ...

    # ---- 方向：用 close[t]-close[t-K] ----
    close = bars_upto["close"].astype(float)
    mom = close.loc[ts_use] - close.shift(rev_cfg.K).loc[ts_use]
    direction = 0 if (not np.isfinite(mom) or mom == 0) else (1 if mom > 0 else -1)

    # ---- 交易决策：p 过阈值 + 幅度也够大 ----
    if direction != 0 and p >= p_threshold:
        if np.isfinite(amp_pred) and amp_pred < amp_threshold:
            return ts_use, p, amp_pred, 0, f"p ok but amp_pred={amp_pred:.6f} < {amp_threshold:.6f}"

        side = -direction
        return ts_use, p, amp_pred, side, f"OPEN side={side} p={p:.3f} amp_pred={amp_pred:.6f}"

    return ts_use, p, amp_pred, 0, "below_threshold_or_no_dir"



def emit_signal(log_path: Path, ts: pd.Timestamp, p: float, amp_pred: float, side: int, reason: str, reg_target: str | None):
    row = {
        "datetime_str": ts.strftime("%Y-%m-%d %H:%M:%S"),
        "p_reversal": p,
        "amp_pred": amp_pred,
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
    theta = 0.005

    project_root = Path(__file__).resolve().parents[1]

    today = now_hk().date().strftime("%Y-%m-%d")
    model_path = project_root / "models" / f"{cfg.symbol}_1m_revN{model_N}_t{theta}_{today}.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"model not found: {model_path}\n请先跑 daily_train_reversal.py 生成当天模型")

    cls_model, feature_cols, rev_cfg, reg_model, reg_target, _ = load_model_bundle(model_path)
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
        print(f"last bar at {bars.iloc[-1].name}")

        if last_emitted_ts is not None and ts_last <= last_emitted_ts:
            continue

        ts_use, p, amp_pred, side, reason = latest_signal_from_bars_tminus1(
            bars,
            cls_model,
            feature_cols,
            rev_cfg,
            reg_model=reg_model,
            reg_target=reg_target,
            p_threshold=p_threshold,
            amp_threshold=0.006,  # 你可以用训练集 top20_mean/阈值来定
        )

        if ts_use is not None:
            emit_signal(log_path, ts_use, p, amp_pred, side, reason, reg_target)


if __name__ == "__main__":
    main()
