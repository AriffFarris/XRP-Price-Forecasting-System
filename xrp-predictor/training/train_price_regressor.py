"""
Train LSTM Price Regressor (log-return regression)

Predicts future PRICE by predicting future log-return:
    y = log(close[t+h] / close[t])

Then:
    pred_close[t+h] = close[t] * exp(y_hat)

Usage:
  python -m training.train_price_regressor --timeframe 1h --horizon 4 --context 48 --epochs 30 --batch 64
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import List, Dict, Tuple

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
        out, _ = self.lstm(x)           # [B, T, 2H]
        yhat = self.head(out[:, -1, :]) # [B, 1]
        return yhat.squeeze(-1)         # [B]


class PriceSequenceDataset(Dataset):
    """
    Sequences ending at time k = idx + context_len - 1.
    Target y is aligned to that SAME k:
        y[k] = log(close[k+h] / close[k])

    Returns: (X_seq, y_logret, base_close, future_close)
    """
    def __init__(
        self,
        df_scaled: pd.DataFrame,
        df_raw: pd.DataFrame,
        feature_cols: List[str],
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


def make_target_logret(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    df = df.copy()
    df["y_logret"] = np.log(df["close"].shift(-horizon) / df["close"])
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df


def select_feature_cols(df: pd.DataFrame) -> List[str]:
    # Drop any future-leaking columns from earlier pipelines (shift(-k))
    drop_future = [c for c in df.columns if c.startswith("ret_") or c.startswith("log_ret_")]
    keep = [c for c in df.columns if c not in drop_future and c != "y_logret"]
    return keep


def regression_metrics(pred: np.ndarray, actual: np.ndarray) -> Dict[str, float]:
    err = pred - actual
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))

    mape = float(np.mean(np.abs(err) / np.clip(np.abs(actual), 1e-12, None)) * 100)
    smape = float(np.mean(2 * np.abs(err) / np.clip(np.abs(pred) + np.abs(actual), 1e-12, None)) * 100)

    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((actual - np.mean(actual)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return {"MAE": mae, "RMSE": rmse, "MAPE_pct": mape, "sMAPE_pct": smape, "R2": r2}


def directional_accuracy(pred: np.ndarray, base: np.ndarray, actual: np.ndarray) -> float:
    pred_dir = (pred > base).astype(int)
    true_dir = (actual > base).astype(int)
    return float((pred_dir == true_dir).mean() * 100)


@torch.no_grad()
def eval_price_metrics(model: nn.Module, loader: DataLoader) -> Dict[str, float]:
    model.eval()
    pred_prices, act_prices, base_prices = [], [], []

    for xb, _, base, fut in loader:
        xb = xb.to(DEVICE)
        base = base.to(DEVICE)
        fut = fut.to(DEVICE)

        yhat = model(xb)  # predicted log-return
        pred = base * torch.exp(yhat)

        pred_prices.append(pred.detach().cpu().numpy())
        act_prices.append(fut.detach().cpu().numpy())
        base_prices.append(base.detach().cpu().numpy())

    pred_prices = np.concatenate(pred_prices) if pred_prices else np.array([])
    act_prices = np.concatenate(act_prices) if act_prices else np.array([])
    base_prices = np.concatenate(base_prices) if base_prices else np.array([])

    if pred_prices.size == 0:
        return {"MAE": float("nan"), "RMSE": float("nan"), "MAPE_pct": float("nan"), "sMAPE_pct": float("nan"), "R2": float("nan"), "DIR_ACC": float("nan")}

    m = regression_metrics(pred_prices, act_prices)
    m["DIR_ACC"] = directional_accuracy(pred_prices, base_prices, act_prices)
    return m


def main(
    timeframe: str = "1h",
    horizon: int = 4,
    context_len: int = 48,
    epochs: int = 30,
    batch: int = 64,
    hidden: int = 64,
    layers: int = 2,
    dropout: float = 0.3,
    lr: float = 1e-3,
):
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

    model = PriceRegressor(input_dim=len(feature_cols), hidden_dim=hidden, num_layers=layers, dropout=dropout).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
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

        # val loss (log-return space)
        model.eval()
        vtotal, vcount = 0.0, 0
        with torch.no_grad():
            for xb, yb, _, _ in val_loader:
                xb = xb.to(DEVICE)
                yb = yb.to(DEVICE)
                yhat = model(xb)
                vloss = loss_fn(yhat, yb)
                vtotal += float(vloss.item()) * len(yb)
                vcount += len(yb)
        val_loss = vtotal / max(1, vcount)

        vm = eval_price_metrics(model, val_loader)
        print(
            f"Epoch {ep:03d} | train_loss={train_loss:.5f} | val_loss={val_loss:.5f} | "
            f"val_MAE=${vm['MAE']:.6f} | val_RMSE=${vm['RMSE']:.6f} | val_MAPE={vm['MAPE_pct']:.2f}% | "
            f"val_R2={vm['R2']:.4f} | val_DIR_ACC={vm['DIR_ACC']:.2f}%"
        )

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    tm = eval_price_metrics(model, test_loader)

    # baseline: no-change future = base
    # (use test_loader arrays by re-walking once)
    base_list, fut_list = [], []
    for _, _, base, fut in test_loader:
        base_list.append(base.numpy())
        fut_list.append(fut.numpy())
    base_np = np.concatenate(base_list)
    fut_np = np.concatenate(fut_list)
    baseline_pred = base_np.copy()
    bm = regression_metrics(baseline_pred, fut_np)

    print("\nTEST METRICS")
    print(
        f"MODEL | MAE=${tm['MAE']:.6f} RMSE=${tm['RMSE']:.6f} MAPE={tm['MAPE_pct']:.2f}% "
        f"sMAPE={tm['sMAPE_pct']:.2f}% R2={tm['R2']:.4f} DIR_ACC={tm['DIR_ACC']:.2f}%"
    )
    print(
        f"BASELINE(no-change) | MAE=${bm['MAE']:.6f} RMSE=${bm['RMSE']:.6f} MAPE={bm['MAPE_pct']:.2f}% "
        f"sMAPE={bm['sMAPE_pct']:.2f}% R2={bm['R2']:.4f}"
    )

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
            "test_metrics": tm,
            "baseline_metrics": bm,
            "arch": {"hidden_dim": hidden, "num_layers": layers, "dropout": dropout},
        },
        ckpt_path,
    )
    print(f"\nSaved: {ckpt_path}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--timeframe", "-t", default="1h")
    p.add_argument("--horizon", "-H", type=int, default=4)
    p.add_argument("--context", "-c", type=int, default=48)
    p.add_argument("--epochs", "-e", type=int, default=30)
    p.add_argument("--batch", "-b", type=int, default=64)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--lr", type=float, default=1e-3)
    args = p.parse_args()

    main(
        timeframe=args.timeframe,
        horizon=args.horizon,
        context_len=args.context,
        epochs=args.epochs,
        batch=args.batch,
        hidden=args.hidden,
        layers=args.layers,
        dropout=args.dropout,
        lr=args.lr,
    )