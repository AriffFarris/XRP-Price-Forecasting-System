from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from models.gru_model import GRUPredictor


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "saved_models"


def load_latest_features(timeframe: str = "1h") -> pd.DataFrame:
    """
    Load the feature dataset we built earlier.
    This still contains the original 'close' column for XRP.
    """
    return pd.read_parquet(DATA_DIR / f"xrp_features_{timeframe}.parquet")


def load_gru_model():
    """
    Load the trained GRU model + scaler + metadata from checkpoint.
    NOTE: We set weights_only=False because our checkpoint includes
    a scikit-learn StandardScaler object, not just raw weights.
    """
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


def generate_autoregressive(
    steps: int = 24,
    timeframe: str = "1h",
) -> List[float]:
    """
    Predict the next `steps` log returns autoregressively:

    - Start from the last `context_len` rows of the dataset.
    - Each step:
        * Run the model to get next log-return prediction.
        * Append that prediction into the feature sequence (in place of target feature).
        * Slide the window and repeat.

    Returns a list of predicted log-returns.
    """
    df = load_latest_features(timeframe)
    model, scaler, feature_cols, target_col, context_len = load_gru_model()

    # Take last context_len rows as our starting context
    ctx_df = df.iloc[-context_len:].copy()

    feats = ctx_df[feature_cols].values.astype("float32")
    feats_scaled = scaler.transform(feats)

    # seq shape: (1, context_len, num_features)
    seq = torch.from_numpy(feats_scaled).unsqueeze(0).to(DEVICE)

    preds = []

    for _ in range(steps):
        # Model expects shape (batch, seq_len, input_dim)
        with torch.no_grad():
            y_next = model(seq).item()
        preds.append(y_next)

        # Convert back to numpy to manipulate
        seq_np = seq.squeeze(0).cpu().numpy()  # (context_len, num_features)

        # Roll sequence up by 1: drop oldest row, shift everything up
        seq_np = np.roll(seq_np, -1, axis=0)

        # For the newest row (last position), start by copying previous row
        seq_np[-1] = seq_np[-2]

        # If target is one of the features, update that position with our prediction
        if target_col in feature_cols:
            idx = feature_cols.index(target_col)
            seq_np[-1, idx] = y_next

        # Back to torch tensor for next iteration
        seq = torch.from_numpy(seq_np).unsqueeze(0).to(DEVICE)

    return preds


def forecast_prices(
    steps: int = 24,
    history_hours: int = 72,
    timeframe: str = "1h",
) -> pd.DataFrame:
    """
    Convert predicted log-returns into a future price path and
    return a DataFrame combining historical and forecast prices.

    Columns:
        - price: XRP/USDT price
        - type: 'historical' or 'forecast'
    Index:
        - datetime (UTC)
    """
    # 1. Load full feature dataset (includes 'close')
    df = load_latest_features(timeframe)

    # Last actual XRP close price
    last_close = df["close"].iloc[-1]

    # Recent history for plotting (last N hours)
    hist = df["close"].iloc[-history_hours:].copy()

    # 2. Get future log-return predictions from GRU
    future_log_rets = generate_autoregressive(steps=steps, timeframe=timeframe)

    # 3. Reconstruct future prices from last_close and log-returns
    future_prices = []
    price = last_close
    for r in future_log_rets:
        price = price * float(np.exp(r))
        future_prices.append(price)

    # 4. Build future timestamps (assuming 1h intervals)
    last_idx = df.index[-1]
    # Ensure timezone-aware future index matches last_idx's tz
    tz = last_idx.tz
    future_index = pd.date_range(
        start=last_idx + pd.Timedelta(hours=1),
        periods=steps,
        freq="H",
        tz=tz,
    )

    # 5. Build combined DataFrame
    hist_df = pd.DataFrame(
        {"price": hist.values, "type": "historical"},
        index=hist.index,
    )
    fut_df = pd.DataFrame(
        {"price": future_prices, "type": "forecast"},
        index=future_index,
    )

    combined = pd.concat([hist_df, fut_df])
    return combined


def plot_forecast(
    steps: int = 24,
    history_hours: int = 72,
    timeframe: str = "1h",
):
    """
    Convenience function: compute forecast prices and plot
    historical vs forecast on the same chart.
    """
    combined = forecast_prices(steps=steps, history_hours=history_hours, timeframe=timeframe)

    hist_mask = combined["type"] == "historical"
    fut_mask = combined["type"] == "forecast"

    plt.figure(figsize=(10, 5))
    plt.plot(
        combined.index[hist_mask],
        combined["price"][hist_mask],
        label="Historical",
    )
    plt.plot(
        combined.index[fut_mask],
        combined["price"][fut_mask],
        linestyle="--",
        marker="o",
        label="Forecast",
    )
    plt.xlabel("Time (UTC)")
    plt.ylabel("XRP/USDT price")
    plt.title(f"XRP Forecast: next {steps} hours (GRU, autoregressive)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print(f"Last real close: {combined['price'][hist_mask].iloc[-1]:.4f}")
    print(f"First forecast price: {combined['price'][fut_mask].iloc[0]:.4f}")
    print(f"Last forecast price:  {combined['price'][fut_mask].iloc[-1]:.4f}")


if __name__ == "__main__":
    # Example: show last 72 hours and next 24 hours
    plot_forecast(steps=24, history_hours=72, timeframe="1h")