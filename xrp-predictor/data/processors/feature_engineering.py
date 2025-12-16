from pathlib import Path

import numpy as np
import pandas as pd
import pandas_ta as ta


# Folder where raw & processed parquet files are stored
DATA_DIR = Path(__file__).resolve().parents[1] / "processed"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical indicators and future-return targets to the dataframe.

    Assumes df has at least:
        - 'open', 'high', 'low', 'close', 'volume'
    Index is a DatetimeIndex (UTC).
    """
    close = df["close"]

    # === Core technical indicators ===

    # 1) RSI (momentum)
    df["rsi_14"] = ta.rsi(close, length=14)

    # 2) MACD (trend + momentum)
    macd = ta.macd(close, fast=12, slow=26, signal=9)
    if macd is not None and not macd.empty:
        # column order from pandas_ta is typically:
        # [MACD, MACDh, MACDs] but we check by name just in case
        macd_cols = list(macd.columns)
        # Try to map by name; fall back to positional if needed
        try:
            df["macd"] = macd[[c for c in macd_cols if "MACD_" in c][0]]
            df["macd_signal"] = macd[[c for c in macd_cols if "MACDs" in c][0]]
            df["macd_hist"] = macd[[c for c in macd_cols if "MACDh" in c][0]]
        except Exception:
            df["macd"] = macd.iloc[:, 0]
            if macd.shape[1] > 1:
                df["macd_signal"] = macd.iloc[:, 1]
            if macd.shape[1] > 2:
                df["macd_hist"] = macd.iloc[:, 2]

    # 3) Bollinger Bands (volatility)
    bb = ta.bbands(close, length=20, std=2.0)
    if bb is not None and not bb.empty:
        # pandas_ta usually returns 3 cols: lower, middle, upper
        # We DO NOT rely on exact column names, just column order
        # (lower, middle, upper) = (0, 1, 2)
        if bb.shape[1] >= 3:
            df["bb_lower"] = bb.iloc[:, 0]
            df["bb_middle"] = bb.iloc[:, 1]
            df["bb_upper"] = bb.iloc[:, 2]

    # 4) EMAs (short / medium / long trend)
    for length in [9, 21, 50, 200]:
        df[f"ema_{length}"] = ta.ema(close, length=length)

    # 5) ATR (Average True Range → volatility)
    df["atr_14"] = ta.atr(df["high"], df["low"], close, length=14)

    # 6) Volume-based (On-Balance Volume)
    df["obv"] = ta.obv(close, df["volume"])

    # === Targets: future returns ===
    # 1-hour ahead return (keep as auxiliary target / extra info)
    df["ret_1h"] = df["close"].pct_change(periods=1).shift(-1)
    df["log_ret_1h"] = np.log(df["close"].shift(-1) / df["close"])

    # 4-hour ahead return (MAIN TARGET NOW)
    df["ret_4h"] = df["close"].pct_change(periods=4).shift(-4)
    df["log_ret_4h"] = np.log(df["close"].shift(-4) / df["close"])

    # 24-hour ahead (optional extra horizon)
    df["ret_24h"] = df["close"].pct_change(periods=24).shift(-24)

    # Drop rows where we don't have enough history for indicators
    # or not enough future for targets
    df = df.dropna()

    return df


def build_feature_dataset(timeframe: str = "1h") -> None:
    """
    Load raw combined data (from price_collector.py),
    compute indicators + targets, and save as xrp_features_1h.parquet.
    """
    raw_path = DATA_DIR / f"xrp_raw_{timeframe}.parquet"
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Raw file not found at {raw_path}. "
            f"Run data/collectors/price_collector.py first."
        )

    df = pd.read_parquet(raw_path)

    # Ensure sorted by time
    df = df.sort_index()

    # Add indicators + targets
    df = add_technical_indicators(df)

    out_path = DATA_DIR / f"xrp_features_{timeframe}.parquet"
    df.to_parquet(out_path)
    print(f"Saved feature dataset to {out_path}")


if __name__ == "__main__":
    build_feature_dataset()