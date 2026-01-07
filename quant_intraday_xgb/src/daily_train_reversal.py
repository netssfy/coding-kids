#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

from quantx.config import Config
from quantx.data.akshare_provider import AkShareHKProvider

HK_TZ = "Asia/Hong_Kong"
# ============================================================
# 0) 复用：你昨天版本的核心函数（原样/微调）
# ============================================================

@dataclass
class RevConfig:
    N: int = 15
    K: int = 15
    theta: float = 0.005
    ema_fast: int = 12
    ema_slow: int = 26
    ema_signal: int = 9
    vwap_win: int = 20
    vol_win: int = 20


def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def macd_components(close: pd.Series, fast: int, slow: int, signal: int):
    dif = ema(close, fast) - ema(close, slow)
    dea = ema(dif, signal)
    hist = dif - dea
    return dif, dea, hist


def vwap_rolling(tp: pd.Series, vol: pd.Series, win: int) -> pd.Series:
    # 兜底：如果滚动成交量为0（很多1m数据会这样），用tp rolling mean 替代，避免全NaN
    num = (tp * vol).rolling(win).sum()
    den = vol.rolling(win).sum()
    vwap = num / den.replace(0, np.nan)
    fallback = tp.rolling(win).mean()
    return vwap.fillna(fallback)


def make_reversal_features(bars: pd.DataFrame, cfg: RevConfig) -> pd.DataFrame:
    df = bars.copy()
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    vol = df["volume"].astype(float)

    tp = (high + low + close) / 3.0

    vwap = vwap_rolling(tp, vol, cfg.vwap_win)
    vwap_dev = (close / (vwap + 1e-12)) - 1.0

    dif, dea, hist = macd_components(close, cfg.ema_fast, cfg.ema_slow, cfg.ema_signal)
    macd_ratio = (dif.abs()) / (hist.abs() + 1e-12)
    hist_slope = hist.diff()

    # 消 warning：fill_method=None
    ret1 = close.pct_change(fill_method=None)
    rv = ret1.rolling(cfg.vol_win).std()

    hl_range = (high / (low.replace(0, np.nan) + 1e-12)) - 1.0
    vol_z = (vol - vol.rolling(cfg.vol_win).mean()) / (vol.rolling(cfg.vol_win).std() + 1e-12)

    X = pd.DataFrame({
        "vwap_dev": vwap_dev,
        "dif": dif,
        "dea": dea,
        "hist": hist,
        "macd_ratio": macd_ratio,
        "hist_slope": hist_slope,
        "rv": rv,
        "hl_range": hl_range,
        "vol_z": vol_z,
    }, index=df.index)

    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    return X


def make_reversal_label(bars: pd.DataFrame, cfg: RevConfig) -> pd.Series:
    close = bars["close"].astype(float)
    hi = bars["high"].astype(float)
    lo = bars["low"].astype(float)

    mom = close - close.shift(cfg.K)

    future_min = lo.shift(-1).rolling(cfg.N, min_periods=cfg.N).min()
    future_max = hi.shift(-1).rolling(cfg.N, min_periods=cfg.N).max()

    thr = cfg.theta * close

    hit_down = (mom > 0) & (future_min <= (close - thr))
    hit_up   = (mom < 0) & (future_max >= (close + thr))

    y = (hit_down | hit_up).astype(int)
    y.name = "y_reversal"
    return y


def train_reversal_model(X: pd.DataFrame, y: pd.Series):
    import xgboost as xgb

    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        min_child_weight=20,
        objective="binary:logistic",
        eval_metric="logloss",
        n_jobs=4,
        random_state=42,
        tree_method="hist",
        missing=np.nan
    )
    model.fit(X, y, verbose=False)
    return model


def predict_reversal_proba(model, X: pd.DataFrame) -> pd.Series:
    p = model.predict_proba(X)[:, 1]
    return pd.Series(p, index=X.index, name="p_reversal")


def avg_future_reversal_path(full_df: pd.DataFrame, start_mask: pd.Series, K: int, horizon: int = 15) -> np.ndarray:
    close = full_df["close"].astype(float)
    mom = close - close.shift(K)
    direction = np.sign(mom).replace(0, np.nan)

    mask = start_mask.reindex(full_df.index).fillna(False)

    paths = []
    n = len(full_df)
    for t in range(n - horizon):
        if not mask.iloc[t]:
            continue
        dir_t = direction.iloc[t]
        if not np.isfinite(dir_t):
            continue

        p0 = close.iloc[t]
        fut = close.iloc[t+1:t+1+horizon] / p0 - 1.0
        fut_rev = (-dir_t) * fut.values

        if np.any(~np.isfinite(fut_rev)):
            continue
        paths.append(fut_rev)

    if not paths:
        return np.full(horizon, np.nan)

    return np.nanmean(np.vstack(paths), axis=0)


def avg_future_reversal_extreme_path(full_df: pd.DataFrame, start_mask: pd.Series, K: int, horizon: int = 15) -> np.ndarray:
    """
    用 high/low 画“到第k分钟为止的反转方向最大有利触达（极值）路径”：
    - 先用过去K根判断起点方向 dir = sign(close[t] - close[t-K])
    - 若 dir=+1（之前上涨），反转方向是下：用未来 low 的累计最小值
    - 若 dir=-1（之前下跌），反转方向是上：用未来 high 的累计最大值
    - 对齐：乘以 (-dir)，让“反转方向收益”为正

    返回：长度 horizon 的数组，第k个元素表示“在k分钟内的最大反转有利触达”的平均值
    """
    close = full_df["close"].astype(float)
    high = full_df["high"].astype(float)
    low  = full_df["low"].astype(float)

    mom = close - close.shift(K)
    direction = np.sign(mom).replace(0, np.nan)  # +1 上行段, -1 下行段

    mask = start_mask.reindex(full_df.index).fillna(False)

    # 收集每个样本的 (horizon,) 极值路径，然后求平均
    paths = []
    n = len(full_df)

    for t in range(n - horizon):
        if not mask.iloc[t]:
            continue

        dir_t = direction.iloc[t]
        if not np.isfinite(dir_t):
            continue

        p0 = close.iloc[t]
        if not np.isfinite(p0) or p0 == 0:
            continue

        # 未来窗口（t+1..t+horizon）
        fut_high = high.iloc[t+1:t+1+horizon].values
        fut_low  = low.iloc[t+1:t+1+horizon].values

        if np.any(~np.isfinite(fut_high)) or np.any(~np.isfinite(fut_low)):
            continue

        if dir_t > 0:
            # 上行段：反转方向向下，用未来 low 的累计最小值
            cum_extreme = np.minimum.accumulate(fut_low)
            ret_extreme = cum_extreme / p0 - 1.0
        else:
            # 下行段：反转方向向上，用未来 high 的累计最大值
            cum_extreme = np.maximum.accumulate(fut_high)
            ret_extreme = cum_extreme / p0 - 1.0

        # 方向对齐：让反转方向统一为正
        ret_aligned = (-dir_t) * ret_extreme

        if np.any(~np.isfinite(ret_aligned)):
            continue

        paths.append(ret_aligned)

    if not paths:
        return np.full(horizon, np.nan)

    return np.nanmean(np.vstack(paths), axis=0)


# ============================================================
# 1) Daily：数据下载 + 快照落盘（文件名带日期string）+ master累积
# ============================================================

def run_date_str(tz: str = "Asia/Hong_Kong") -> str:
    return pd.Timestamp.now(tz=tz).date().strftime("%Y-%m-%d")  # ✅ string


def save_snapshot_with_datetime_str(bars: pd.DataFrame, out_path: Path):
    """
    保存快照：额外加一列 datetime_str = 'YYYY-MM-DD HH:mm:ss'
    同时把 datetime 转成无tz（便于你直接查看文件）
    """
    df = bars.copy()
    df = df.reset_index()  # datetime column
    df["datetime_str"] = df["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")
    # 你要求“保存文件时时间保存成yyyy-MM-dd HH:mm:ss”，这里保留字符串列；
    # 同时把 datetime 去掉 tz，避免有些工具显示复杂
    df["datetime"] = df["datetime"].dt.tz_localize(None)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)


def load_or_download_latest(cfg: Config, root: Path, date_tag: str) -> pd.DataFrame:
    """
    每次运行都下载，并保存到 data/raw 里（文件名包含日期tag）
    同时也保存一份“无日期”的 csv 以兼容你现有 CSVProvider（可选）
    """
    raw_dir = root / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # dated snapshot parquet
    snap_path = raw_dir / f"{cfg.symbol}_{cfg.bar_size}_{date_tag}.parquet"

    # provider = YFinanceProvider(interval=cfg.bar_size, period=cfg.yfinance_period)
    # bars = provider.load_bars(cfg.symbol).sort_index()
    now = pd.Timestamp.now(tz=HK_TZ)
    start = now - pd.Timedelta(days=7)
    provider = AkShareHKProvider(period=cfg.bar_size)
    bars = provider.load_bars(cfg.symbol, start=start, end=now).sort_index()

    bars = bars[~bars.index.duplicated(keep="last")]

    # 保存快照（含 datetime_str）
    save_snapshot_with_datetime_str(bars, snap_path)

    # 兼容旧逻辑：保存一份无日期 raw csv（可选）
    raw_csv = raw_dir / f"{cfg.symbol}_{cfg.bar_size}.csv"
    bars.reset_index().to_csv(raw_csv, index=False)

    print(f"[data] downloaded and saved snapshot: {snap_path}")
    return bars


def append_to_master(root: Path, symbol: str, bar_size: str, bars_new: pd.DataFrame) -> Path:
    """
    master: 保持 DatetimeIndex（用于计算）
    同时新增 datetime_str 列（用于展示阅读）
    """
    master_dir = root / "data" / "master"
    master_dir.mkdir(parents=True, exist_ok=True)
    master_path = master_dir / f"{symbol}_{bar_size}_master.parquet"

    def with_datetime_str(bars: pd.DataFrame) -> pd.DataFrame:
        df = bars.copy()
        # index 是 datetime（可能带 tz），用于计算
        idx = df.index
        if idx.tz is not None:
            idx_local = idx.tz_convert("Asia/Hong_Kong")
        else:
            idx_local = idx
        df["datetime_str"] = idx_local.strftime("%Y-%m-%d %H:%M:%S")
        return df

    new_df = with_datetime_str(bars_new)

    if master_path.exists():
        old = pd.read_parquet(master_path)
        # 如果老数据没有 datetime_str，也补上（兼容历史）
        if "datetime_str" not in old.columns:
            old = with_datetime_str(old)

        merged = pd.concat([old, new_df]).sort_index()
        merged = merged[~merged.index.duplicated(keep="last")]
    else:
        merged = new_df.sort_index()
        merged = merged[~merged.index.duplicated(keep="last")]

    merged.to_parquet(master_path)
    print(f"[data] master updated: {master_path} (rows={len(merged)})")
    return master_path


# ============================================================
# 2) Daily：训练 + 评估 + 保存模型/报告（日期string）
# ============================================================

def time_split_df(df: pd.DataFrame, ratio: float = 0.8):
    split = int(len(df) * ratio)
    return df.iloc[:split], df.iloc[split:]


def evaluate_metrics(eval_df: pd.DataFrame) -> pd.DataFrame:
    y = eval_df["y_reversal"].astype(int).values
    p = eval_df["p_reversal"].astype(float).values

    base_rate = float(np.mean(y))
    top = eval_df[eval_df["p_reversal"] >= eval_df["p_reversal"].quantile(0.8)]
    top20_rate = float(top["y_reversal"].mean()) if len(top) else np.nan
    lift_top20 = (top20_rate / base_rate) if base_rate > 0 else np.nan

    if len(np.unique(y)) == 2:
        auc_roc = float(roc_auc_score(y, p))
        auc_pr = float(average_precision_score(y, p))
    else:
        auc_roc = np.nan
        auc_pr = np.nan

    brier = float(brier_score_loss(y, p))

    return pd.DataFrame([{
        "base_rate": base_rate,
        "top20_rate": top20_rate,
        "lift_top20": lift_top20,
        "auc_pr": auc_pr,
        "auc_roc": auc_roc,
        "brier": brier,
    }])


def bucket_table(eval_df: pd.DataFrame) -> pd.DataFrame:
    tmp = eval_df.copy()
    tmp["bucket"] = pd.qcut(tmp["p_reversal"], 10, duplicates="drop")
    bt = (
        tmp.groupby("bucket", observed=True)["y_reversal"]
        .agg(count="count", real_reversal_rate="mean")
        .reset_index()
    )
    return bt


def plot_bucket_bar(bt: pd.DataFrame, out_png: Path, title: str):
    out_png.parent.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(bt))
    y = bt["real_reversal_rate"].astype(float).values

    plt.figure(figsize=(10, 4))
    plt.bar(x, y)
    plt.xticks(x, [str(b) for b in bt["bucket"]], rotation=45, ha="right")
    plt.ylabel("Real reversal rate")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def plot_aligned_path(df_with_close_and_p: pd.DataFrame, out_png: Path, K: int, horizon: int, title: str):
    out_png.parent.mkdir(parents=True, exist_ok=True)

    q20 = df_with_close_and_p["p_reversal"].quantile(0.2)
    q80 = df_with_close_and_p["p_reversal"].quantile(0.8)

    mask_low = df_with_close_and_p["p_reversal"] < q20
    mask_high = df_with_close_and_p["p_reversal"] > q80

    # 1) 原版：用 close 的方向对齐路径
    path_low = avg_future_reversal_path(df_with_close_and_p, mask_low, K=K, horizon=horizon)
    path_high = avg_future_reversal_path(df_with_close_and_p, mask_high, K=K, horizon=horizon)

    # 2) 新增：用 high/low 的累计极值方向对齐路径（更贴近触达型标签）
    path_low_ext = avg_future_reversal_extreme_path(df_with_close_and_p, mask_low, K=K, horizon=horizon)
    path_high_ext = avg_future_reversal_extreme_path(df_with_close_and_p, mask_high, K=K, horizon=horizon)

    x = np.arange(1, horizon + 1)
    plt.figure(figsize=(10, 5.2))

    # close-path
    plt.plot(x, path_low * 100, marker="o", label="Close-path: Low p (bottom 20%)")
    plt.plot(x, path_high * 100, marker="o", label="Close-path: High p (top 20%)")

    # extreme-path (high/low)
    plt.plot(x, path_low_ext * 100, marker="x", linestyle="--", label="Extreme(H/L): Low p (bottom 20%)")
    plt.plot(x, path_high_ext * 100, marker="x", linestyle="--", label="Extreme(H/L): High p (top 20%)")

    plt.axhline(0, linewidth=1)
    plt.xlabel("Minutes ahead (k)")
    plt.ylabel("Avg 'reversal-direction' return (%)")
    plt.title(title + " (Close vs High/Low Extreme)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def decision_line(metrics: pd.DataFrame) -> tuple[str, str]:
    m = metrics.iloc[0]
    base = m["base_rate"]
    pr = m["auc_pr"]
    brier = m["brier"]
    lift = m["lift_top20"]
    top20 = m["top20_rate"]

    pr_base = base
    brier_rand = base * (1 - base)

    cond = (lift >= 2.0) and (top20 >= 0.55) and (np.isfinite(pr) and pr >= 1.3 * pr_base) and (brier < brier_rand)
    if cond:
        return "USE ✅", f"lift={lift:.2f}, top20={top20:.3f}, PR-AUC={pr:.3f}, Brier={brier:.4f}"
    return "SKIP ❌", f"lift={lift:.2f}, top20={top20:.3f}, PR-AUC={pr:.3f}, Brier={brier:.4f}"


def save_html_report(report_path: Path, cfg: Config, rev_cfg: RevConfig, date_tag: str,
                     metrics: pd.DataFrame, bt: pd.DataFrame, bucket_png: Path, aligned_png: Path,
                     decision: tuple[str, str]):
    report_path.parent.mkdir(parents=True, exist_ok=True)
    dec, reason = decision

    styled = metrics.style.format({
        "base_rate": "{:.3f}",
        "top20_rate": "{:.3f}",
        "lift_top20": "{:.2f}",
        "auc_pr": "{:.3f}",
        "auc_roc": "{:.3f}",
        "brier": "{:.4f}",
    }).hide(axis="index")

    html = []
    html.append("<html><head><meta charset='utf-8'><title>Daily Reversal Report</title></head><body>")
    html.append(f"<h2>Daily Reversal Model Report</h2>")
    html.append(f"<p><b>Date:</b> {date_tag} &nbsp; <b>Symbol:</b> {cfg.symbol} &nbsp; <b>Bar:</b> {cfg.bar_size}</p>")
    html.append(f"<p><b>RevCfg:</b> N={rev_cfg.N}, K={rev_cfg.K}, theta={rev_cfg.theta}</p>")

    html.append(f"<h3>Decision: {dec}</h3>")
    html.append(f"<p><code>{reason}</code></p>")

    html.append("<h3>Metrics (test split)</h3>")
    html.append(styled.to_html())

    html.append("<h3>Decile bucket (test split)</h3>")
    html.append(bt.to_html(index=False))
    html.append(f"<p><img src='{bucket_png.name}' style='max-width: 100%; height: auto;'></p>")

    html.append("<h3>Aligned path (test split)</h3>")
    html.append(f"<p><img src='{aligned_png.name}' style='max-width: 100%; height: auto;'></p>")

    html.append("<h3>Configs (json)</h3>")
    html.append("<pre>")
    html.append(json.dumps({
        "date": date_tag,
        "config": cfg.__dict__,
        "rev_config": asdict(rev_cfg),
    }, ensure_ascii=False, indent=2))
    html.append("</pre>")

    html.append("</body></html>")
    report_path.write_text("\n".join(html), encoding="utf-8")


# ============================================================
# 3) Main
# ============================================================

def main():
    # 1) 读你的 Config（沿用你项目结构）
    cfg = Config()

    root = Path(__file__).resolve().parents[1]
    date_tag = run_date_str("Asia/Hong_Kong")

    # 2) 下载并保存快照（文件名带日期）
    bars_new = load_or_download_latest(cfg, root, date_tag)

    # 3) 追加 master（长期）
    master_path = append_to_master(root, cfg.symbol, cfg.bar_size, bars_new)

    # 4) 用 master 训练
    bars = pd.read_parquet(master_path).sort_index()
    bars = bars[~bars.index.duplicated(keep="last")]

    rev_cfg = RevConfig(N=15, K=15, theta=0.005)

    # 特征 & 标签（复用昨天版本）
    X = make_reversal_features(bars, rev_cfg)
    y = make_reversal_label(bars, rev_cfg)

    # 对齐：和你昨天一样 dropna（现在 vwap 有 fallback，不至于全空）
    df = X.join(y.rename("y")).dropna()
    if len(df) < 500:
        raise RuntimeError(f"Not enough samples after dropna: {len(df)}. "
                           f"Try accumulating more master data or reduce windows.")

    X2 = df.drop(columns=["y"])
    y2 = df["y"].astype(int)

    # 5) 时间切分
    tr, te = time_split_df(df, ratio=0.8)
    X_tr, y_tr = tr.drop(columns=["y"]), tr["y"].astype(int)
    X_te, y_te = te.drop(columns=["y"]), te["y"].astype(int)

    # 6) 训练 & 评估
    model = train_reversal_model(X_tr, y_tr)
    p_te = predict_reversal_proba(model, X_te)
    eval_df = pd.DataFrame({"y_reversal": y_te, "p_reversal": p_te}, index=X_te.index)

    metrics = evaluate_metrics(eval_df)
    bt = bucket_table(eval_df)
    dec = decision_line(metrics)

    # 7) 画图（bucket + aligned path）
    reports_dir = root / "reports" / date_tag
    reports_dir.mkdir(parents=True, exist_ok=True)

    bucket_png = reports_dir / f"{cfg.symbol}_{cfg.bar_size}_revN{rev_cfg.N}_{date_tag}_bucket.png"
    aligned_png = reports_dir / f"{cfg.symbol}_{cfg.bar_size}_revN{rev_cfg.N}_{date_tag}_aligned.png"

    plot_bucket_bar(bt, bucket_png, title=f"Decile bucket: {cfg.symbol} {cfg.bar_size} (test split)")

    # aligned path 需要 close + p_reversal（用测试段）
    te_with_close = eval_df.join(bars[["high", "low", "close"]], how="left").dropna(subset=["close", "high", "low"])
    plot_aligned_path(te_with_close, aligned_png, K=rev_cfg.K, horizon=rev_cfg.N,
                      title=f"Aligned path: {cfg.symbol} {cfg.bar_size} (test split)")

    # 8) 保存模型（文件名带日期 string）
    models_dir = root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / f"{cfg.symbol}_{cfg.bar_size}_revN{rev_cfg.N}_{date_tag}.joblib"
    joblib.dump({
        "date": date_tag,
        "symbol": cfg.symbol,
        "bar_size": cfg.bar_size,
        "rev_cfg": asdict(rev_cfg),
        "feature_columns": list(X2.columns),
        "model": model,
    }, model_path)

    # 9) 保存报告（HTML）
    report_path = reports_dir / f"{cfg.symbol}_{cfg.bar_size}_revN{rev_cfg.N}_{date_tag}.html"
    save_html_report(report_path, cfg, rev_cfg, date_tag, metrics, bt, bucket_png, aligned_png, dec)

    # 10) 控制台输出（简洁、决策友好）
    print("\n=== Daily Train Summary ===")
    print("date:", date_tag)
    print("master:", master_path)
    print("model:", model_path)
    print("report:", report_path)
    print("decision:", dec[0], "-", dec[1])
    print("\nmetrics(test):")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
