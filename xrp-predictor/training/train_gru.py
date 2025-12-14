from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler

from training.dataset import SequenceDataset
from models.gru_model import GRUPredictor


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "saved_models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def train_gru(
    timeframe: str = "1h",
    context_len: int = 72,
    batch_size: int = 64,
    max_epochs: int = 100,
):
    # 1. Load feature dataset
    df = pd.read_parquet(DATA_DIR / f"xrp_features_{timeframe}.parquet")

    target_col = "log_ret_1h"

    # Features: all columns except these
    ignore_cols = ["ret_1h", "ret_24h", "log_ret_1h"]
    feature_cols = [c for c in df.columns if c not in ignore_cols]

    # 2. Time-based split
    n = len(df)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)

    df_train = df.iloc[:train_end].copy()
    df_val = df.iloc[train_end:val_end].copy()
    df_test = df.iloc[val_end:].copy()  # currently unused, but you can evaluate later

    # 3. Scale features (fit on train only to avoid leakage)
    scaler = StandardScaler()
    scaler.fit(df_train[feature_cols].values)

    for d in (df_train, df_val, df_test):
        d[feature_cols] = scaler.transform(d[feature_cols].values)

    # 4. Build datasets & loaders
    train_ds = SequenceDataset(df_train, feature_cols, target_col, context_len)
    val_ds = SequenceDataset(df_val, feature_cols, target_col, context_len)
    test_ds = SequenceDataset(df_test, feature_cols, target_col, context_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # 5. Init model
    model = GRUPredictor(input_dim=len(feature_cols), hidden_dim=128, num_layers=2, dropout=0.2)
    model.to(DEVICE)

    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=1e-3)

    best_val_loss = np.inf
    patience = 10
    patience_counter = 0

    # 6. Training loop
    for epoch in range(max_epochs):
        model.train()
        train_loss = 0.0

        for xb, yb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * xb.size(0)

        train_loss /= len(train_ds)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(DEVICE)
                yb = yb.to(DEVICE)
                preds = model(xb)
                loss = criterion(preds, yb)
                val_loss += loss.item() * xb.size(0)
        val_loss /= len(val_ds)

        print(f"Epoch {epoch+1:03d} | Train {train_loss:.6f} | Val {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            ckpt = {
                "model_state": model.state_dict(),
                "scaler": scaler,
                "feature_cols": feature_cols,
                "target_col": target_col,
                "context_len": context_len,
            }
            torch.save(ckpt, MODELS_DIR / "gru_xrp.pt")
            print("  -> Saved new best model")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    # You can add test evaluation here if you want


if __name__ == "__main__":
    train_gru()