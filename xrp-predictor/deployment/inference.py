from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch

from models.gru_model import GRUPredictor


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "saved_models"


def load_latest_features(timeframe: str = "1h") -> pd.DataFrame:
    return pd.read_parquet(DATA_DIR / f"xrp_features_{timeframe}.parquet")


def load_gru_model():
    ckpt_path = MODELS_DIR / "gru_xrp.pt"

    # We created this checkpoint ourselves in train_gru.py,
    # so it's safe to load full objects (including the StandardScaler).
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
    - Use last context_len rows as starting context.
    - Each time, feed the model, get a prediction,
      then slide window and insert that prediction.
    """
    df = load_latest_features(timeframe)
    model, scaler, feature_cols, target_col, context_len = load_gru_model()

    # Take last context_len rows
    ctx_df = df.iloc[-context_len:].copy()

    feats = ctx_df[feature_cols].values.astype("float32")
    feats_scaled = scaler.transform(feats)

    seq = torch.from_numpy(feats_scaled).unsqueeze(0).to(DEVICE)  # (1, context_len, num_features)

    preds = []

    for _ in range(steps):
        with torch.no_grad():
            y_next = model(seq).item()
        preds.append(y_next)

        # Update the sequence: roll and insert new prediction
        seq_np = seq.squeeze(0).cpu().numpy()
        seq_np = np.roll(seq_np, -1, axis=0)
        seq_np[-1] = seq_np[-2]   # copy previous row as base

        if target_col in feature_cols:
            idx = feature_cols.index(target_col)
            seq_np[-1, idx] = y_next

        seq = torch.from_numpy(seq_np).unsqueeze(0).to(DEVICE)

    return preds


if __name__ == "__main__":
    future_log_rets = generate_autoregressive(steps=24)
    print("Next 24 predicted log returns:")
    print(future_log_rets)