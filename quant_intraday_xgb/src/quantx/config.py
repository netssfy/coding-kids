from dataclasses import dataclass

@dataclass
class Config:
    symbol: str = "01810"
    bar_size: str = "1m"  # "5m" or "15m"
    horizon_bars: int = 3  # 预测未来N根bar收益
    hold_bars: int = 2     # 策略持有N根bar（示例）
    prob_long: float = 0.60
    prob_flat: float = 0.50  # <= 0.50 视为不做多（示例）

    # 成本（港股：佣金/平台费/印花税等你可以细化；先用bps近似）
    fee_bps: float = 8.0      # 单边交易成本，bps
    slippage_bps: float = 2.0 # 滑点，bps

    # Walk-forward
    train_days: int = 45
    test_days: int = 15

    # data source switch
    data_source: str = "yfinance"  # "yfinance" or "csv"
    yfinance_period: str = "8d"  # e.g. "30d", "60d", "1y"
