#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from quantx.config import Config
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import argparse  # [新增] 用于接收命令行参数 n

# 你已有的
from daily_train_reversal import RevConfig, make_reversal_features

HK_TZ = "Asia/Hong_Kong"


# -----------------------------
# 1) Provider 接口（可换 AkShare / yfinance / master parquet）
# -----------------------------
class BarProvider:
    def load_1m_bars(self, symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        raise NotImplementedError


class MasterParquetProvider(BarProvider):
    """
    从 data/master 读一份大表，然后按时间切片
    要求 master index 是 datetime（你现在就是）
    """

    def __init__(self, master_path: Path):
        self.master_path = master_path
        self._cache = None

    def _load_all(self) -> pd.DataFrame:
        if self._cache is None:
            df = pd.read_parquet(self.master_path).sort_index()
            df = df[~df.index.duplicated(keep="last")]
            # 确保 tz
            if df.index.tz is None:
                df.index = df.index.tz_localize(HK_TZ)
            else:
                df.index = df.index.tz_convert(HK_TZ)
            self._cache = df
        return self._cache

    def load_1m_bars(self, symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        df = self._load_all()
        out = df.loc[start:end].copy()
        return out


# -----------------------------
# 2) 模型bundle读取
# -----------------------------
@dataclass
class ModelBundle:
    model: object
    feature_cols: list[str]
    rev_cfg: RevConfig
    meta: dict


def load_model_bundle(model_path: Path) -> ModelBundle:
    bundle = joblib.load(model_path)
    model = bundle["model"]
    feature_cols = bundle["feature_columns"]
    rev_cfg = RevConfig(**bundle["rev_cfg"]) if isinstance(bundle["rev_cfg"], dict) else bundle["rev_cfg"]
    return ModelBundle(model=model, feature_cols=feature_cols, rev_cfg=rev_cfg, meta=bundle)


# -----------------------------
# 3) 回放配置 & 执行器（paper trading）
# -----------------------------
@dataclass
class ReplayConfig:
    symbol: str = "1810.HK"
    p_threshold: float = 0.6  # 触发信号阈值（建议可用 top20 分位替代）
    hold_minutes: int = 10  # 简单策略：入场后持有 N 分钟
    cooldown_minutes: int = 0  # 平仓后冷却 N 分钟再允许开新仓
    n: int = 1  # [新增] 连续 n 次超过阈值才开仓


@dataclass
class PositionState:
    side: int = 0  # +1 long, -1 short, 0 flat
    entry_time: pd.Timestamp | None = None
    entry_price: float | None = None
    hold_until: pd.Timestamp | None = None
    cooldown_until: pd.Timestamp | None = None


def calc_direction(close: pd.Series, t: pd.Timestamp, K: int) -> int:
    """
    用 close[t] - close[t-K] 判断起点前方向：
    +1 上行段；-1 下行段；0 无法判断
    """
    try:
        mom = float(close.loc[t] - close.shift(K).loc[t])
    except Exception:
        return 0
    if not np.isfinite(mom) or mom == 0:
        return 0
    return 1 if mom > 0 else -1


def predict_p_at(model_bundle: ModelBundle, bars_upto: pd.DataFrame, t: pd.Timestamp) -> float:
    """
    用截至 t 的 bars 计算特征，并取 t 时刻那行特征做预测。
    """
    X = make_reversal_features(bars_upto, model_bundle.rev_cfg)
    if t not in X.index:
        return np.nan
    x = X.loc[[t], model_bundle.feature_cols]
    if not np.isfinite(x.values).all():
        return np.nan
    p = float(model_bundle.model.predict_proba(x)[:, 1][0])
    return p


def maybe_open_position(
    pos: PositionState,
    t: pd.Timestamp,
    price: float,
    p: float,
    direction: int,
    cfg: ReplayConfig,
) -> tuple[PositionState, str]:
    """
    规则：
    - 仅当 flat 且未在 cooldown
    - 且 p >= threshold
    - 且 direction!=0
    开“反转方向”的仓：side = -direction
    """
    if pos.side != 0:
        return pos, "already_in_position"

    if pos.cooldown_until is not None and t < pos.cooldown_until:
        return pos, "cooldown"

    if not np.isfinite(p) or p < cfg.p_threshold or direction == 0:
        return pos, "no_signal"

    side = -direction
    pos.side = side
    pos.entry_time = t
    pos.entry_price = price
    pos.hold_until = t + pd.Timedelta(minutes=cfg.hold_minutes)
    return pos, f"OPEN side={side} p={p:.3f}"


def maybe_close_position(
    pos: PositionState,
    t: pd.Timestamp,
    price: float,
    cfg: ReplayConfig,
) -> tuple[PositionState, str, float]:
    """
    简单平仓：到 hold_until 就平仓
    返回 (pos, action, pnl)
    """
    if pos.side == 0:
        return pos, "flat", 0.0

    if pos.hold_until is not None and t >= pos.hold_until:
        # pnl：按方向计算
        pnl = pos.side * (price / pos.entry_price - 1.0)
        pos.side = 0
        pos.entry_time = None
        pos.entry_price = None
        pos.hold_until = None
        if cfg.cooldown_minutes > 0:
            pos.cooldown_until = t + pd.Timedelta(minutes=cfg.cooldown_minutes)
        return pos, f"CLOSE pnl={pnl:.5f}", pnl

    return pos, "HOLD", 0.0


# -----------------------------
# 4) 回放主函数：回放 T 日（含 T-1 数据）
# -----------------------------
def replay_one_day(
    provider: BarProvider,
    model_bundle: ModelBundle,
    day_T: str,
    cfg: ReplayConfig,
    out_csv: Path,
):
    """
    day_T: 'YYYY-MM-DD'
    逻辑：
    - 加载 [T-1 00:00, T 23:59] 的 bars（你也可以只取交易时段）
    - 回放区间只在 T 日分钟上迭代
    - 在 T 日的每一分钟 t（bar 形成完成后），用 t-1 的已闭合bar做预测并发信号
    """
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    T = pd.Timestamp(day_T, tz=HK_TZ)
    start = (T - pd.Timedelta(days=1)).normalize()  # T-1 00:00
    end = (T + pd.Timedelta(days=1)).normalize() - pd.Timedelta(seconds=1)  # T 23:59:59

    bars = provider.load_1m_bars(cfg.symbol, start, end).copy()
    if len(bars) == 0:
        raise RuntimeError("No bars loaded for T-1..T range")

    # 确保必要列
    need = {"open", "high", "low", "close", "volume"}
    missing = need - set(bars.columns)
    if missing:
        raise RuntimeError(f"Bars missing columns: {missing}")

    bars = bars.sort_index()
    bars = bars[~bars.index.duplicated(keep="last")]

    # 回放分钟索引：只取 T 日（你也可换成港股交易时段过滤）
    bars_T = bars.loc[T.normalize() : (T.normalize() + pd.Timedelta(days=1) - pd.Timedelta(minutes=1))]
    if len(bars_T) == 0:
        raise RuntimeError("No bars for T day in loaded data")

    close = bars["close"].astype(float)

    pos = PositionState()
    rows = []
    cum_pnl = 0.0

    # [新增] 信号计数器变量
    signal_counter = 0
    pending_side = 0  # 正在等待确认的方向 (1 for long, -1 for short)

    for t in bars_T.index:
        # 用 t-1 的完整bar 做预测
        t_pred = t - pd.Timedelta(minutes=1)
        if t_pred not in bars.index:
            continue

        # 截止到 t_pred 的数据（包含 T-1）
        bars_upto = bars.loc[:t_pred]

        p = predict_p_at(model_bundle, bars_upto, t_pred)

        dir_t = calc_direction(close.loc[:t_pred], t_pred, model_bundle.rev_cfg.K)

        price_now = float(bars.loc[t, "close"])  # 用 t 的 close 当作执行价格（简单处理）

        # 先检查是否需要平仓（到期平）
        pos, action_close, pnl = maybe_close_position(pos, t, price_now, cfg)
        cum_pnl += pnl

        # [新增] 逻辑：计算信号是否连续出现 n 次
        # 1. 判断当前时刻是否有原始信号
        raw_signal_side = 0
        if np.isfinite(p) and p >= cfg.p_threshold and dir_t != 0:
            raw_signal_side = -dir_t  # 策略逻辑：反转方向

        # 2. 更新计数器
        if raw_signal_side != 0:
            if raw_signal_side == pending_side:
                signal_counter += 1
            else:
                # 方向变了（或者之前无信号），重置并计数为1
                pending_side = raw_signal_side
                signal_counter = 1
        else:
            # 信号消失
            signal_counter = 0
            pending_side = 0

        # 3. 再检查是否开仓（仅当 count >= n 时调用开仓逻辑）
        # 注意：maybe_open_position 内部还会检查 p >= threshold，这没问题，只要这里传进去的 p 是合格的
        action_open = "wait_signal"
        if signal_counter >= cfg.n:
            price_entry = price_now
            # 这里的 p 和 dir_t 肯定是满足条件的，所以 maybe_open_position 会通过校验
            pos, action_open = maybe_open_position(pos, t, price_entry, p, dir_t, cfg)
        else:
            if raw_signal_side != 0:
                action_open = f"accumulating({signal_counter}/{cfg.n})"
            else:
                action_open = "no_signal"

        side = pos.side
        rows.append({
            "t_exec": t.strftime("%Y-%m-%d %H:%M:%S"),
            "t_pred": t_pred.strftime("%Y-%m-%d %H:%M:%S"),
            "p_reversal": p,
            "direction": dir_t,  # +1 上行段，-1 下行段
            "signal_side": (-dir_t if (np.isfinite(p) and p >= cfg.p_threshold and dir_t != 0) else 0),
            "signal_count": signal_counter,  # [新增] 记录当前的计数
            "pos_side": side,
            "action_close": action_close,
            "action_open": action_open,
            "price": price_now,
            "cum_pnl": cum_pnl,
        })

    df_out = pd.DataFrame(rows)
    df_out.to_csv(out_csv, index=False)
    return df_out


# -----------------------------
# 5) CLI入口
# -----------------------------
def main():
    # [新增] 参数解析
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--n", type=int, default=1, help="连续 n 次超过阈值才开仓")
    args = parser.parse_args()

    cfg = Config()
    rev_cfg = RevConfig(N=15, K=15, theta=0.005)

    project_root = Path(__file__).resolve().parents[1]

    # 你要回放哪一天 T
    day_T = "2026-01-08"  # 改成你要的日期
    args.n=1
    args.pt=80

    # 模型：通常用 T-1 收盘后训练的模型预测 T
    model_path = project_root / "models" / f"{cfg.symbol}_{cfg.bar_size}_revN{rev_cfg.N}_t{rev_cfg.theta}_{day_T}.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"model not found: {model_path}")

    model_bundle = load_model_bundle(model_path)

    master_path = project_root / "data" / "master" / f"{cfg.symbol}_{cfg.bar_size}_master.parquet"
    provider = MasterParquetProvider(master_path)

    # [修改] 传入 n 参数
    replayCfg = ReplayConfig(symbol=cfg.symbol, p_threshold=args.pt / 100, hold_minutes=10, n=args.n)

    # [修改] 文件名加入 n
    out_csv = project_root / "data" / "replay" / f"{cfg.symbol}_{cfg.bar_size}_replay_{day_T}_n{args.n}_pt{args.pt}.csv"

    df = replay_one_day(provider, model_bundle, day_T, replayCfg, out_csv)
    print("Saved replay:", out_csv)
    print(df.tail(5))


if __name__ == "__main__":
    main()