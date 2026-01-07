import pandas as pd
from .base import DataProvider
import akshare as ak

HK_TZ = "Asia/Hong_Kong"

# -----------------------------
# 1) AkShare Provider (HK 1m)
# -----------------------------
class AkShareHKProvider(DataProvider):
    """
    使用 ak.stock_hk_hist_min_em 获取港股分钟K线
    返回统一格式：index=datetime(tz-aware), columns=open/high/low/close/volume(+amount可选)
    """

    # AkShare 常见中文列名 → 英文列名
    _COL_MAP = {
        "开盘": "open",
        "收盘": "close",
        "最高": "high",
        "最低": "low",
        "成交量": "volume",
        "成交额": "amount",
        "时间": "datetime",
    }

    def __init__(self, period: str = "1", adjust: str = ""):
        self.period = period[0]  # "1","5","15","30","60"
        self.adjust = adjust  # "", "qfq", "hfq"

    def load_bars(self, symbol_hk: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        """
        symbol_hk: 港股代码，通常是5位字符串，如小米 "01810"
        start/end: timezone-aware timestamp
        """
        start_str = start.strftime("%Y-%m-%d %H:%M:%S")
        end_str = end.strftime("%Y-%m-%d %H:%M:%S")

        df = ak.stock_hk_hist_min_em(
            symbol=symbol_hk,
            period=self.period,
            adjust=self.adjust,
            # start_date=start_str,
            # end_date=end_str,
        )

        if df is None or len(df) == 0:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        # 统一列名
        df = df.rename(columns=self._COL_MAP)

        # 时间列解析
        df["datetime"] = pd.to_datetime(df["datetime"])
        # AkShare 返回一般是本地交易所时间（不带tz），这里显式标注为 HK 时区
        df["datetime"] = df["datetime"].dt.tz_localize(HK_TZ, nonexistent="shift_forward", ambiguous="NaT")
        df = df.set_index("datetime").sort_index()

        # 统一数值类型
        for c in ["open", "high", "low", "close", "volume"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        df = df[["open", "high", "low", "close", "volume"]].dropna()
        df = df[~df.index.duplicated(keep="last")]
        return df