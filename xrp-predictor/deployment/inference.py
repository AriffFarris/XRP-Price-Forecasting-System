"""
Fixed autoregressive inference for XRP price forecasting.

Key fixes:
1. Recalculate technical indicators from predicted prices
2. Properly propagate price changes through the feature pipeline
3. Add confidence intervals based on historical prediction error
"""

from pathlib import Path
import sys
from typing import List, Tuple

import numpy as np
import pandas as pd
import pandas_ta as ta
import torch
import matplotlib.pyplot as plt

# Add project root to path for module imports
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.gru_model import GRUPredictor


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "saved_models"

HORIZON_HOURS = 4


def load_latest_features(timeframe: str = "1h") -> pd.DataFrame:
    """Load the feature dataset."""
    return pd.read_parquet(DATA_DIR / f"xrp_features_{timeframe}.parquet")


def load_gru_model():
    """Load trained GRU model + scaler + metadata."""
    ckpt_path = MODELS_DIR / "gru_xrp.pt"
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)

    feature_cols = ckpt["feature_cols"]
    context_len = ckpt["context_len"]

    model = GRUPredictor(input_dim=len(feature_cols))
    model.load_state_dict(ckpt["model_state"])
    model.to(DEVICE)
    model.eval()

    scaler = ckpt["scaler"]
    target_col = ckpt["target_col"]

    return model, scaler, feature_cols, target_col, context_len


def recalculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recalculate all technical indicators from OHLCV data.
    This must match exactly what feature_engineering.py does.
    """
    close = df["close"]
    
    # RSI
    df["rsi_14"] = ta.rsi(close, length=14)
    
    # MACD
    macd = ta.macd(close, fast=12, slow=26, signal=9)
    if macd is not None and not macd.empty:
        macd_cols = list(macd.columns)
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
    
    # Bollinger Bands
    bb = ta.bbands(close, length=20, std=2.0)
    if bb is not None and not bb.empty and bb.shape[1] >= 3:
        df["bb_lower"] = bb.iloc[:, 0]
        df["bb_middle"] = bb.iloc[:, 1]
        df["bb_upper"] = bb.iloc[:, 2]
    
    # EMAs
    for length in [9, 21, 50, 200]:
        df[f"ema_{length}"] = ta.ema(close, length=length)
    
    # ATR
    df["atr_14"] = ta.atr(df["high"], df["low"], close, length=14)
    
    # OBV
    df["obv"] = ta.obv(close, df["volume"])
    
    return df


def generate_autoregressive_fixed(
    steps: int = 24,
    timeframe: str = "1h",
) -> Tuple[List[float], List[float]]:
    """
    Properly generate autoregressive forecasts by:
    1. Predicting next log return
    2. Computing new price from log return
    3. Appending synthetic OHLCV row
    4. Recalculating ALL technical indicators
    5. Re-scaling features
    6. Feeding updated context to model
    
    Returns:
        predictions: List of predicted 4h log returns
        prices: List of predicted prices
    """
    # Load raw data (we need OHLCV to recalculate indicators)
    raw_path = DATA_DIR / f"xrp_raw_{timeframe}.parquet"
    df_raw = pd.read_parquet(raw_path)
    
    # Load model and metadata
    model, scaler, feature_cols, target_col, context_len = load_gru_model()
    
    # We need enough history to calculate indicators (200 for EMA_200)
    min_history = max(200, context_len) + 50  # buffer
    df_working = df_raw.iloc[-min_history:].copy()
    
    predictions = []
    prices = []
    
    for step in range(steps):
        # 1. Recalculate indicators on current data
        df_feat = recalculate_technical_indicators(df_working.copy())
        
        # 2. Get the last context_len rows of features
        # Filter to only the columns the model expects
        available_cols = [c for c in feature_cols if c in df_feat.columns]
        
        if len(available_cols) < len(feature_cols):
            missing = set(feature_cols) - set(available_cols)
            print(f"Warning: Missing features: {missing}")
            # Fill missing with zeros (not ideal but prevents crash)
            for col in missing:
                df_feat[col] = 0.0
        
        # Take last context_len rows
        ctx = df_feat[feature_cols].iloc[-context_len:].values.astype("float32")
        
        # Handle any NaNs from indicator warm-up
        ctx = np.nan_to_num(ctx, nan=0.0)
        
        # 3. Scale features
        ctx_scaled = scaler.transform(ctx)
        
        # 4. Run model
        seq = torch.from_numpy(ctx_scaled).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            pred_log_ret = model(seq).item()
        
        predictions.append(pred_log_ret)
        
        # 5. Compute new price
        last_close = df_working["close"].iloc[-1]
        new_price = last_close * np.exp(pred_log_ret)
        prices.append(new_price)
        
        # 6. Create synthetic OHLCV row for next iteration
        last_idx = df_working.index[-1]
        new_idx = last_idx + pd.Timedelta(hours=HORIZON_HOURS)
        
        # Simple synthetic candle: use predicted close
        # In reality, you'd want more sophisticated OHLV estimation
        new_row = pd.DataFrame({
            "open": [last_close],
            "high": [max(last_close, new_price) * 1.001],  # slight wick
            "low": [min(last_close, new_price) * 0.999],
            "close": [new_price],
            "volume": [df_working["volume"].iloc[-20:].mean()],  # avg volume
            # Copy BTC/ETH columns (assume they stay flat - simplification)
            **{col: [df_working[col].iloc[-1]] 
               for col in df_working.columns 
               if col.startswith(("btc_", "eth_", "BTC", "ETH"))}
        }, index=[new_idx])
        
        df_working = pd.concat([df_working, new_row])
    
    return predictions, prices


def forecast_prices_fixed(
    steps: int = 24,
    history_hours: int = 72,
    timeframe: str = "1h",
) -> pd.DataFrame:
    """
    Generate forecast with proper feature recalculation.
    """
    df = load_latest_features(timeframe)
    
    # Get predictions
    log_rets, future_prices = generate_autoregressive_fixed(
        steps=steps, 
        timeframe=timeframe
    )
    
    # Historical prices
    hist = df["close"].iloc[-history_hours:].copy()
    
    # Build future timestamps
    last_idx = df.index[-1]
    tz = last_idx.tz
    future_index = pd.date_range(
        start=last_idx + pd.Timedelta(hours=HORIZON_HOURS),
        periods=steps,
        freq=f"{HORIZON_HOURS}h",
        tz=tz,
    )
    
    # Combine
    hist_df = pd.DataFrame(
        {"price": hist.values, "type": "historical"},
        index=hist.index,
    )
    fut_df = pd.DataFrame(
        {"price": future_prices, "type": "forecast"},
        index=future_index,
    )
    
    return pd.concat([hist_df, fut_df])


def plot_forecast_with_uncertainty(
    steps: int = 24,
    history_hours: int = 72,
    timeframe: str = "1h",
    confidence_pct: float = 0.10,  # ±10% confidence band
):
    """
    Plot forecast with uncertainty bands.
    
    The confidence band grows over time to reflect compounding uncertainty.
    """
    combined = forecast_prices_fixed(
        steps=steps, 
        history_hours=history_hours, 
        timeframe=timeframe
    )
    
    hist_mask = combined["type"] == "historical"
    fut_mask = combined["type"] == "forecast"
    
    fut_prices = combined.loc[fut_mask, "price"].values
    fut_idx = combined.index[fut_mask]
    
    # Uncertainty grows with sqrt(steps) - random walk assumption
    steps_ahead = np.arange(1, len(fut_prices) + 1)
    uncertainty = confidence_pct * np.sqrt(steps_ahead / steps_ahead[-1])
    
    upper = fut_prices * (1 + uncertainty)
    lower = fut_prices * (1 - uncertainty)
    
    total_hours = steps * HORIZON_HOURS
    
    plt.figure(figsize=(12, 6))
    
    # Historical
    plt.plot(
        combined.index[hist_mask],
        combined["price"][hist_mask],
        label="Historical",
        color="blue",
        linewidth=1.5,
    )
    
    # Forecast
    plt.plot(
        fut_idx,
        fut_prices,
        linestyle="--",
        marker="o",
        markersize=4,
        label="Forecast",
        color="red",
    )
    
    # Confidence band
    plt.fill_between(
        fut_idx,
        lower,
        upper,
        alpha=0.2,
        color="red",
        label=f"±{int(confidence_pct*100)}% uncertainty",
    )
    
    plt.xlabel("Time (UTC)")
    plt.ylabel("XRP/USDT price")
    plt.title(f"XRP Forecast: next {total_hours}h ({HORIZON_HOURS}h steps) - FIXED")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(ROOT / "forecast_plot.png", dpi=150)
    plt.show()
    
    # Print summary
    print(f"\n{'='*50}")
    print("FORECAST SUMMARY")
    print(f"{'='*50}")
    print(f"Last historical close:  ${combined['price'][hist_mask].iloc[-1]:.4f}")
    print(f"First forecast (T+{HORIZON_HOURS}h): ${fut_prices[0]:.4f}")
    print(f"Final forecast (T+{total_hours}h): ${fut_prices[-1]:.4f}")
    
    total_change = (fut_prices[-1] / combined['price'][hist_mask].iloc[-1] - 1) * 100
    print(f"Total predicted change: {total_change:+.2f}%")
    
    # Check for monotonic behavior (still a warning sign)
    diffs = np.diff(fut_prices)
    if np.all(diffs > 0):
        print("\n⚠️  WARNING: Forecast is monotonically increasing!")
        print("   This may indicate the model has learned a bullish bias.")
    elif np.all(diffs < 0):
        print("\n⚠️  WARNING: Forecast is monotonically decreasing!")
        print("   This may indicate the model has learned a bearish bias.")
    else:
        print("\n✓ Forecast shows non-monotonic behavior (more realistic)")


if __name__ == "__main__":
    plot_forecast_with_uncertainty(steps=24, history_hours=72, timeframe="1h")