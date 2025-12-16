"""
Train LSTM Price Regressor

Predicts future PRICE by predicting future log-return:
    y = log(close[t+h] / close[t])

Then:
    pred_close[t+h] = close[t] * exp(y_hat)

Why log-return?
- more stationary than price
- avoids needing to scale y
- easy to convert to price for plots

Usage:
  python -m training.train_price_regressor --timeframe 1h --horizon 4 --context 48 --epochs 30
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import RobustScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "saved_models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


class PriceRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)                 # [B, T, 2H]
        yhat = self.head(out[:, -1, :])       # [B, 1]
        return yhat.squeeze(-1)               # [B]


class PriceSequenceDataset(Dataset):
    """
    Sequences ending at time k = idx + context_len - 1.
    Target y is aligned to that SAME k:
        y[k] = log(close[k+h] / close[k])

    Returns: (X_seq, y, base_close, future_close)
    """
    def __init__(
        self,
        df_scaled: pd.DataFrame,
        df_raw: pd.DataFrame,
        feature_cols: list[str],
        y_col: str,
        context_len: int,
        horizon: int,
        close_col: str = "close",
    ):
        self.feature_cols = feature_cols
        self.context_len = context_len
        self.horizon = horizon

        self.X = df_scaled[feature_cols].values.astype("float32")
        self.y = df_raw[y_col].values.astype("float32")

        self.base_close = df_raw[close_col].values.astype("float32")
        self.future_close = df_raw[close_col].shift(-horizon).values.astype("float32")

        # Drop tail rows that can't form (base_close -> future_close)
        # (y already removed most of these, but keep safe)
        valid = ~np.isnan(self.future_close)
        self.X = self.X[valid]
        self.y = self.y[valid]
        self.base_close = self.base_close[valid]
        self.future_close = self.future_close[valid]

    def __len__(self) -> int:
        return max(0, len(self.y) - self.context_len + 1)

    def __getitem__(self, idx: int):
        x = self.X[idx : idx + self.context_len]
        k = idx + self.context_len - 1

        y = self.y[k]
        base = self.base_close[k]
        fut = self.future_close[k]

        return (
            torch.from_numpy(x),
            torch.tensor(y, dtype=torch.float32),
            torch.tensor(base, dtype=torch.float32),
            torch.tensor(fut, dtype=torch.float32),
        )


@torch.no_grad()
def eval_price_metrics(model: nn.Module, loader: DataLoader) -> dict:
    model.eval()
    abs_errs = []
    ape = []
    for xb, yb, base, fut in loader:
        xb = xb.to(DEVICE)
        yb = yb.to(DEVICE)
        base = base.to(DEVICE)
        fut = fut.to(DEVICE)

        yhat = model(xb)  # predicted log-return
        pred_price = base * torch.exp(yhat)
        actual_price = fut

        abs_err = torch.abs(pred_price - actual_price)
        abs_errs.append(abs_err.cpu().numpy())

        ape.append((abs_err / torch.clamp(actual_price, min=1e-12)).cpu().numpy())

    abs_errs = np.concatenate(abs_errs) if abs_errs else np.array([])
    ape = np.concatenate(ape) if ape else np.array([])

    return {
        "MAE": float(abs_errs.mean()) if abs_errs.size else float("nan"),
        "MAPE_pct": float(ape.mean() * 100) if ape.size else float("nan"),
    }


def make_target_logret(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    df = df.copy()
    df["y_logret"] = np.log(df["close"].shift(-horizon) / df["close"])
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df


def select_feature_cols(df: pd.DataFrame) -> list[str]:
    # IMPORTANT: these are FUTURE-leaking in your feature_engineering.py
    # because they are shifted negative (future info).
    drop_future = [c for c in df.columns if c.startswith("ret_") or c.startswith("log_ret_")]
    keep = [c for c in df.columns if c not in drop_future and c != "y_logret"]
    return keep


def main(timeframe: str = "1h", horizon: int = 4, context_len: int = 48, epochs: int = 30, batch: int = 64):
    feat_path = DATA_DIR / f"xrp_features_{timeframe}.parquet"
    if not feat_path.exists():
        raise FileNotFoundError(f"Missing {feat_path}. Run collector + feature pipeline first.")

    df = pd.read_parquet(feat_path).sort_index()
    df = make_target_logret(df, horizon=horizon)

    feature_cols = select_feature_cols(df)

    # Time split
    n = len(df)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)
    df_train = df.iloc[:train_end].copy()
    df_val = df.iloc[train_end:val_end].copy()
    df_test = df.iloc[val_end:].copy()

    # Scale X only (fit on train only)
    scaler = RobustScaler()
    scaler.fit(df_train[feature_cols])

    def scaled_copy(d: pd.DataFrame) -> pd.DataFrame:
        out = d.copy()
        out[feature_cols] = scaler.transform(out[feature_cols])
        return out

    df_train_s = scaled_copy(df_train)
    df_val_s = scaled_copy(df_val)
    df_test_s = scaled_copy(df_test)

    # Datasets
    train_ds = PriceSequenceDataset(df_train_s, df_train, feature_cols, "y_logret", context_len, horizon)
    val_ds = PriceSequenceDataset(df_val_s, df_val, feature_cols, "y_logret", context_len, horizon)
    test_ds = PriceSequenceDataset(df_test_s, df_test, feature_cols, "y_logret", context_len, horizon)

    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch, shuffle=False)

    model = PriceRegressor(input_dim=len(feature_cols)).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.SmoothL1Loss()  # Huber

    best_val = float("inf")
    best_state = None

    print(f"\nDevice: {DEVICE}")
    print(f"Samples: train={len(df_train):,}, val={len(df_val):,}, test={len(df_test):,}")
    print(f"Features: {len(feature_cols)} | Horizon={horizon}h | Context={context_len}")

    for ep in range(1, epochs + 1):
        model.train()
        total = 0.0
        count = 0

        for xb, yb, _, _ in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            optimizer.zero_grad()
            yhat = model(xb)
            loss = loss_fn(yhat, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total += float(loss.item()) * len(yb)
            count += len(yb)

        train_loss = total / max(1, count)

        # validation loss (on y space)
        model.eval()
        vtotal = 0.0
        vcount = 0
        with torch.no_grad():
            for xb, yb, _, _ in val_loader:
                xb = xb.to(DEVICE)
                yb = yb.to(DEVICE)
                yhat = model(xb)
                vloss = loss_fn(yhat, yb)
                vtotal += float(vloss.item()) * len(yb)
                vcount += len(yb)

        val_loss = vtotal / max(1, vcount)

        metrics = eval_price_metrics(model, val_loader)
        print(f"Epoch {ep:03d} | train_loss={train_loss:.5f} | val_loss={val_loss:.5f} | val_MAE=${metrics['MAE']:.6f} | val_MAPE={metrics['MAPE_pct']:.2f}%")

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics = eval_price_metrics(model, test_loader)
    print(f"\nTEST: MAE=${test_metrics['MAE']:.6f} | MAPE={test_metrics['MAPE_pct']:.2f}%")

    ckpt_path = MODELS_DIR / f"lstm_price_regressor_{timeframe}_h{horizon}_c{context_len}.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "scaler": scaler,
            "feature_cols": feature_cols,
            "context_len": context_len,
            "horizon": horizon,
            "timeframe": timeframe,
            "target": "y_logret",
            "test_metrics": test_metrics,
        },
        ckpt_path,
    )
    print(f"Saved: {ckpt_path}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--timeframe", "-t", default="1h")
    p.add_argument("--horizon", "-H", type=int, default=4)
    p.add_argument("--context", "-c", type=int, default=48)
    p.add_argument("--epochs", "-e", type=int, default=30)
    p.add_argument("--batch", "-b", type=int, default=64)
    args = p.parse_args()

    main(timeframe=args.timeframe, horizon=args.horizon, context_len=args.context, epochs=args.epochs, batch=args.batch)