"""
Quick XRP Prediction - Simple one-line output

Usage:
    python -m deployment.quick_predict
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training.train_reversal_detector import ReversalDetector, create_reversal_features

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "saved_models"


def quick_predict():
    """Get a quick prediction."""
    
    # Load model
    ckpt = torch.load(MODELS_DIR / "reversal_detector.pt", map_location=DEVICE, weights_only=False)
    model = ReversalDetector(input_dim=len(ckpt["feature_cols"]))
    model.load_state_dict(ckpt["model_state"])
    model.to(DEVICE)
    model.eval()
    
    # Load data
    df_raw = pd.read_parquet(DATA_DIR / "xrp_features_1h.parquet")
    
    # Create features
    df, feature_cols = create_reversal_features(df_raw, horizon=ckpt["horizon"])
    
    # Scale and predict
    scaler = ckpt["scaler"]
    context_len = ckpt["context_len"]
    
    df_scaled = df.copy()
    df_scaled[feature_cols] = scaler.transform(df[feature_cols])
    
    ctx = df_scaled[feature_cols].iloc[-context_len:].values.astype("float32")
    seq = torch.from_numpy(ctx).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        reversal_prob = torch.sigmoid(model(seq)).item()
    
    # Get current state
    current_price = df["close"].iloc[-1]
    current_dir = "UP" if df["close"].iloc[-1] > df["close"].iloc[-2] else "DOWN"
    
    # Predict direction
    threshold = ckpt.get("best_threshold", 0.5)
    will_reverse = reversal_prob > threshold
    
    if will_reverse:
        pred_dir = "DOWN" if current_dir == "UP" else "UP"
        confidence = reversal_prob
    else:
        pred_dir = current_dir
        confidence = 1 - reversal_prob
    
    # Format output
    arrow = "⬆️" if pred_dir == "UP" else "⬇️"
    
    print(f"\n{'='*50}")
    print(f"XRP: ${current_price:.4f} → {pred_dir} {arrow} ({confidence:.0%} confidence)")
    print(f"{'='*50}")
    print(f"Prediction for next {ckpt['horizon']} hours")
    print(f"Data as of: {df.index[-1]}")
    print(f"{'='*50}\n")
    
    return {
        "price": current_price,
        "direction": pred_dir,
        "confidence": confidence,
    }


if __name__ == "__main__":
    quick_predict()