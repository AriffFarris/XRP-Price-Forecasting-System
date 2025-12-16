from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training.train_reversal_detector import ReversalDetector, create_reversal_features

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "saved_models"


def load_lstm_ckpt():
    ckpt_path = MODELS_DIR / "reversal_detector.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing {ckpt_path}. Train first: python -m training.train_reversal_detector")

    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model = ReversalDetector(input_dim=len(ckpt["feature_cols"]))
    model.load_state_dict(ckpt["model_state"])
    model.to(DEVICE)
    model.eval()
    return model, ckpt


def choose_expected_moves(df_raw: pd.DataFrame, horizon: int):
    # Typical move sizes from history (robust median)
    rets = df_raw["close"].pct_change(horizon).dropna().values
    pos = rets[rets > 0]
    neg = rets[rets < 0]

    up_move = float(np.median(pos)) if len(pos) else 0.003
    down_move = float(np.median(neg)) if len(neg) else -0.003

    # Cap to avoid silly spikes
    up_move = float(np.clip(up_move, 0.0, 0.03))
    down_move = float(np.clip(down_move, -0.03, 0.0))

    return up_move, down_move


def backtest_plot(n_points: int = 800):
    # Load data
    feat_path = DATA_DIR / "xrp_features_1h.parquet"
    raw_path = DATA_DIR / "xrp_raw_1h.parquet"
    if not feat_path.exists():
        raise FileNotFoundError(f"Missing {feat_path}. Run feature pipeline first.")
    if not raw_path.exists():
        raise FileNotFoundError(f"Missing {raw_path}. Run data collector first.")

    df_raw = pd.read_parquet(raw_path)
    df_feat = pd.read_parquet(feat_path)

    # Load model + metadata
    model, ckpt = load_lstm_ckpt()
    scaler = ckpt["scaler"]
    feature_cols = ckpt["feature_cols"]
    context_len = int(ckpt["context_len"])
    horizon = int(ckpt["horizon"])
    threshold = float(ckpt.get("best_threshold", 0.5))

    # Expected move sizes for converting direction -> price
    up_move, down_move = choose_expected_moves(df_raw, horizon=horizon)
    print(f"Using expected moves (median {horizon}h): UP={up_move*100:.2f}%  DOWN={down_move*100:.2f}%")

    # Rebuild reversal features exactly like training
    df, built_cols = create_reversal_features(df_feat, horizon=horizon)

    # Ensure all expected feature columns exist
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0

    # Scale features
    df_scaled = df.copy()
    df_scaled[feature_cols] = scaler.transform(df_scaled[feature_cols])

    # We'll evaluate on the tail to keep the plot readable
    # Need enough room for context window
    end = len(df_scaled) - 1
    start = max(context_len, end - n_points)

    times = []
    pred_prices = []
    actual_prices = []
    pred_dirs = []
    actual_dirs = []

    close = df["close"].values  # unscaled close
    idx = df.index

    # actual future close at t+h is close shifted by -horizon (1h steps)
    future_close = df["close"].shift(-horizon).values

    with torch.no_grad():
        for i in range(start, end):
            # Need future price available
            if np.isnan(future_close[i]):
                continue

            # context window ends at i (inclusive)
            ctx = df_scaled[feature_cols].iloc[i - context_len + 1 : i + 1].values.astype("float32")
            ctx = np.nan_to_num(ctx, nan=0.0)

            seq = torch.from_numpy(ctx).unsqueeze(0).to(DEVICE)
            rev_prob = torch.sigmoid(model(seq)).item()

            curr_dir = 1.0 if df["close"].iloc[i] > df["close"].iloc[i - 1] else 0.0  # UP=1, DOWN=0
            will_reverse = rev_prob > threshold
            pred_dir = (1.0 - curr_dir) if will_reverse else curr_dir

            move = up_move if pred_dir == 1.0 else down_move
            pred_future_price = float(close[i] * (1.0 + move))
            act_future_price = float(future_close[i])

            # For plotting, align at prediction target time (t + horizon hours)
            times.append(idx[i] + pd.Timedelta(hours=horizon))
            pred_prices.append(pred_future_price)
            actual_prices.append(act_future_price)

            pred_dirs.append(int(pred_dir))
            actual_dirs.append(int(act_future_price > close[i]))  # actual direction over horizon

    pred_prices = np.array(pred_prices)
    actual_prices = np.array(actual_prices)

    # Metrics
    mae = np.mean(np.abs(pred_prices - actual_prices))
    mape = np.mean(np.abs((pred_prices - actual_prices) / actual_prices)) * 100
    dir_acc = (np.array(pred_dirs) == np.array(actual_dirs)).mean() * 100

    print(f"MAE:  {mae:.6f}")
    print(f"MAPE: {mape:.2f}%")
    print(f"Directional accuracy ({horizon}h): {dir_acc:.1f}%")

    # Plot
    plt.figure(figsize=(14, 6))
    plt.plot(times, actual_prices, label="Actual (t + horizon)")
    plt.plot(times, pred_prices, label="Predicted (LSTM direction + expected move)", linestyle="--")
    plt.title(f"LSTM Predicted vs Actual Price (horizon={horizon}h)")
    plt.xlabel("Time")
    plt.ylabel("XRP/USDT Price")
    plt.grid(True, alpha=0.3)
    plt.legend()
    out = ROOT / "diagnostics_lstm_pred_vs_actual.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    print(f"Saved: {out}")
    plt.show()


if __name__ == "__main__":
    backtest_plot(n_points=800)