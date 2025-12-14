from pathlib import Path

import numpy as np
import pandas as pd
import pandas_ta as ta


DATA_DIR = Path(__file__).resolve().parents[1] / "processed"


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes a DataFrame with columns:
    close, high, low, volume, btc_close, eth_close, ...
    and adds technical indicators + returns.
    """
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    # 1-hour and 24-hour returns
    df["ret_1h"] = close.pct_change()
    df["log_ret_1h"] = np.log(close / close.shift(1))
    df["ret_24h"] = close.pct_change(24)

    # RSI
    df["rsi_14"] = ta.rsi(close, length=14)

    # MACD
    macd = ta.macd(close)
    df["macd"] = macd["MACD_12_26_9"]
    df["macd_signal"] = macd["MACDs_12_26_9"]

    # Bollinger Bands
    bb = ta.bbands(close, length=20, std=2)

    # Different pandas_ta versions use slightly different column names.
    # We pick columns by pattern instead of hardcoding exact names.
    upper_candidates = [c for c in bb.columns if "BBU" in c or "upper" in c.lower()]
    lower_candidates = [c for c in bb.columns if "BBL" in c or "lower" in c.lower()]

    if not upper_candidates or not lower_candidates:
        raise ValueError(f"Could not find Bollinger Band columns in: {list(bb.columns)}")

    upper_col = upper_candidates[0]
    lower_col = lower_candidates[0]

    df["bb_upper"] = bb[upper_col]
    df["bb_lower"] = bb[lower_col]
    df["bb_pct"] = (close - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])

    # Average True Range (volatility)
    df["atr_14"] = ta.atr(high, low, close, length=14)

    # On-Balance Volume and VWAP
    df["obv"] = ta.obv(close, volume)
    df["vwap"] = ta.vwap(high, low, close, volume)

    # Cross-asset returns (BTC, ETH)
    for sym in ["btc", "eth"]:
        c = df[f"{sym}_close"]
        df[f"{sym}_ret_1h"] = c.pct_change()
        df[f"{sym}_ret_24h"] = c.pct_change(24)

    # Time-of-day / day-of-week features
    df["hour"] = df.index.hour
    df["dayofweek"] = df.index.dayofweek

    return df


def build_feature_dataset(timeframe: str = "1h"):
    """
    Load the raw parquet, add features, drop NaNs from warmup periods, save again.
    """
    raw_path = DATA_DIR / f"xrp_raw_{timeframe}.parquet"
    df = pd.read_parquet(raw_path)

    df = add_technical_indicators(df)
    df = df.dropna().copy()

    out_path = DATA_DIR / f"xrp_features_{timeframe}.parquet"
    df.to_parquet(out_path)
    print(f"Saved feature dataset to {out_path}")


if __name__ == "__main__":
    build_feature_dataset()