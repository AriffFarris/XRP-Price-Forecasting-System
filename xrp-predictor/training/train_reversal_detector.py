"""
Train Momentum Reversal Detector (Option A direction model)

Goal:
- Predict whether momentum will CONTINUE or REVERSE over horizon h.
- Output: P(reversal) (binary classification)

Convert to direction:
- Get current direction curr_dir (based on last 1-step return)
- If reversal_prob > threshold -> flip curr_dir
- Else -> keep curr_dir

Usage:
  python -m training.train_reversal_detector --timeframe 1h --horizon 4 --context 48 --epochs 100
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


class ReversalDetector(nn.Module):
    """Binary classifier: reversal(1) vs continue(0)"""
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
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        logits = self.head(out[:, -1, :]).squeeze(-1)
        return logits


class SeqDataset(Dataset):
    """
    Produces (X_seq, y_reversal) where y is aligned to the END of the window.
    Window ends at k, target is y[k] (computed from close[k] and close[k+h]).
    """
    def __init__(self, X: np.ndarray, y: np.ndarray, context_len: int):
        self.X = X.astype("float32")
        self.y = y.astype("float32")
        self.context_len = context_len

    def __len__(self) -> int:
        return max(0, len(self.y) - self.context_len + 1)

    def __getitem__(self, idx: int):
        x = self.X[idx : idx + self.context_len]
        k = idx + self.context_len - 1
        return torch.from_numpy(x), torch.tensor(self.y[k], dtype=torch.float32)


def create_reversal_features(df: pd.DataFrame, horizon: int) -> tuple[pd.DataFrame, List[str]]:
    """
    Builds reversal-focused features from OHLCV.

    Target:
      current_dir = sign(close[t] - close[t-1])
      future_dir  = sign(close[t+h] - close[t])
      reversal = (future_dir != current_dir)

    IMPORTANT:
    - Only uses past/current info at time t (no shift(-k) features).
    """
    df = df.copy()

    # Basic series
    close = df["close"]
    high = df["high"]
    low = df["low"]
    open_ = df["open"]
    volume = df["volume"]

    # Current momentum
    df["curr_ret_1"] = close.pct_change(1)
    df["curr_ret_4"] = close.pct_change(4)
    df["curr_dir"] = (df["curr_ret_1"] > 0).astype(float)  # 1 = up, 0 = down

    # Momentum strength / acceleration
    df["momentum_strength"] = df["curr_ret_4"].abs()
    df["momentum_accel"] = df["curr_ret_1"] - df["curr_ret_1"].shift(1)

    # Streak length in current direction
    direction = (close.pct_change(1) > 0).astype(int)
    streak_groups = (direction != direction.shift()).cumsum()
    df["streak_len"] = direction.groupby(streak_groups).cumcount() + 1
    df["streak_extended"] = (df["streak_len"] > 5).astype(float)

    # RSI
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))

    df["rsi_exhaustion"] = np.where(
        df["curr_dir"] == 1,
        (df["rsi"] > 70).astype(float),
        (df["rsi"] < 30).astype(float),
    )

    # MA distance
    for p in [10, 20, 50]:
        ma = close.rolling(p).mean()
        df[f"dist_ma{p}"] = (close - ma) / ma

    df["ma_exhaustion"] = (df["dist_ma20"].abs() > df["dist_ma20"].rolling(100).std() * 2).astype(float)

    # Bollinger band position
    ma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    df["bb_position"] = (close - ma20) / (2 * std20).replace(0, np.nan)
    df["bb_extreme"] = (df["bb_position"].abs() > 1).astype(float)

    # Volume signals
    df["vol_ratio"] = volume / volume.rolling(20).mean()
    df["vol_spike"] = (df["vol_ratio"] > 2).astype(float)
    df["vol_trend"] = volume.rolling(5).mean() / volume.rolling(20).mean()
    df["vol_divergence"] = (df["vol_trend"] < 0.8).astype(float)

    # Candle patterns
    body = (close - open_).abs()
    full_range = (high - low).replace(0, np.nan)
    df["body_ratio"] = body / full_range
    df["indecision"] = (df["body_ratio"] < 0.2).astype(float)

    upper_wick = high - close.clip(lower=open_)
    lower_wick = close.clip(upper=open_) - low
    df["upper_rejection"] = (upper_wick / full_range > 0.6).astype(float)
    df["lower_rejection"] = (lower_wick / full_range > 0.6).astype(float)

    # Volatility
    df["volatility"] = close.pct_change().rolling(12).std()
    df["vol_expanding"] = (df["volatility"] > df["volatility"].shift(1)).astype(float)

    # Targets
    future_dir = (close.shift(-horizon) > close).astype(float)
    current_dir = (close.pct_change(1) > 0).astype(float)
    df["target_reversal"] = (future_dir != current_dir).astype(float)
    df["target_direction"] = future_dir

    # Cleanup
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    exclude = [
        "open", "high", "low", "close", "volume",
        "target_reversal", "target_direction",
        # we keep curr_dir for conversion later, but exclude from features
        "curr_dir",
    ]
    feature_cols = [c for c in df.columns if c not in exclude]

    return df, feature_cols


def accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()
    return float((preds == y).float().mean().item())


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_n = 0
    correct = 0
    for xb, yb in loader:
        xb = xb.to(DEVICE)
        yb = yb.to(DEVICE)
        logits = model(xb)
        loss = criterion(logits, yb)
        total_loss += float(loss.item()) * len(yb)
        total_n += len(yb)

        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()
        correct += int((preds == yb).sum().item())

    return total_loss / max(1, total_n), correct / max(1, total_n)


def convert_reversal_to_direction(reversal_probs: np.ndarray, curr_dir: np.ndarray, threshold: float) -> np.ndarray:
    rev = (reversal_probs > threshold).astype(int)
    return np.where(rev == 1, 1 - curr_dir.astype(int), curr_dir.astype(int))


def main(
    timeframe: str = "1h",
    horizon: int = 4,
    context_len: int = 48,
    epochs: int = 100,
    batch: int = 64,
    hidden: int = 64,
    layers: int = 2,
    dropout: float = 0.3,
    lr: float = 1e-3,
    patience: int = 15,
):
    feat_path = DATA_DIR / f"xrp_features_{timeframe}.parquet"
    if not feat_path.exists():
        raise FileNotFoundError(f"Missing {feat_path}. Run collector + feature pipeline first.")

    df_raw = pd.read_parquet(feat_path).sort_index()
    df, feature_cols = create_reversal_features(df_raw, horizon=horizon)

    n = len(df)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)

    df_train = df.iloc[:train_end].copy()
    df_val = df.iloc[train_end:val_end].copy()
    df_test = df.iloc[val_end:].copy()

    reversal_rate = float(df["target_reversal"].mean())
    persistence_acc = 1.0 - reversal_rate

    print(f"\nDevice: {DEVICE}")
    print(f"Samples: train={len(df_train):,}, val={len(df_val):,}, test={len(df_test):,}")
    print(f"Features: {len(feature_cols)} | Horizon={horizon}h | Context={context_len}")
    print(f"Reversal rate: {reversal_rate:.2%} | Persistence baseline (continue): {persistence_acc:.2%}")

    # Scale features on train only
    scaler = RobustScaler()
    scaler.fit(df_train[feature_cols])

    def scale_inplace(d: pd.DataFrame) -> None:
        d.loc[:, feature_cols] = scaler.transform(d[feature_cols])

    scale_inplace(df_train)
    scale_inplace(df_val)
    scale_inplace(df_test)

    X_train = df_train[feature_cols].values
    y_train = df_train["target_reversal"].values
    X_val = df_val[feature_cols].values
    y_val = df_val["target_reversal"].values
    X_test = df_test[feature_cols].values
    y_test = df_test["target_reversal"].values

    train_ds = SeqDataset(X_train, y_train, context_len=context_len)
    val_ds = SeqDataset(X_val, y_val, context_len=context_len)
    test_ds = SeqDataset(X_test, y_test, context_len=context_len)

    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch, shuffle=False)

    # Class weighting (reversals often rarer)
    pos_weight = torch.tensor([(1 - reversal_rate) / max(1e-9, reversal_rate)], device=DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    model = ReversalDetector(input_dim=len(feature_cols), hidden_dim=hidden, num_layers=layers, dropout=dropout).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_val_acc = 0.0
    best_state = None
    epochs_no_improve = 0

    print(f"\n{'Epoch':<7}{'TrainAcc':<12}{'ValAcc':<12}{'ValLoss':<12}")
    print("-" * 45)

    for ep in range(1, epochs + 1):
        model.train()
        correct = 0
        total = 0
        running_loss = 0.0

        for xb, yb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            opt.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            running_loss += float(loss.item()) * len(yb)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            correct += int((preds == yb).sum().item())
            total += len(yb)

        sched.step()

        train_acc = correct / max(1, total)
        val_loss, val_acc = evaluate(model, val_loader, criterion)

        marker = ""
        if val_acc > best_val_acc + 0.001:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
            marker = " <- best"
        else:
            epochs_no_improve += 1

        print(f"{ep:<7}{train_acc*100:<12.2f}{val_acc*100:<12.2f}{val_loss:<12.4f}{marker}")

        if epochs_no_improve >= patience:
            print(f"\nEarly stopping at epoch {ep} (no val improvement for {patience} epochs).")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    # --- Test evaluation (reversal accuracy) ---
    test_loss, test_rev_acc = evaluate(model, test_loader, criterion)

    # --- Threshold tuning on VAL for best DIRECTION accuracy ---
    # Align curr_dir and direction targets with sequence outputs:
    # sequences end at k >= context_len-1, so labels are effectively df_* rows from (context_len-1) onward.
    # Our SeqDataset returns y[k], so we align using those same indices.
    with torch.no_grad():
        model.eval()
        # compute reversal probs for VAL windows in order
        probs_val = []
        for xb, _ in val_loader:
            xb = xb.to(DEVICE)
            logits = model(xb)
            probs_val.append(torch.sigmoid(logits).cpu().numpy())
        probs_val = np.concatenate(probs_val) if probs_val else np.array([])

    # build aligned curr_dir and true direction for val
    val_curr_dir = df_val["curr_dir"].values[context_len - 1 :]
    val_true_dir = df_val["target_direction"].values[context_len - 1 :]

    min_len = min(len(probs_val), len(val_curr_dir), len(val_true_dir))
    probs_val = probs_val[:min_len]
    val_curr_dir = val_curr_dir[:min_len]
    val_true_dir = val_true_dir[:min_len]

    best_thresh = 0.5
    best_dir_acc = -1.0
    for thresh in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
        dir_pred = convert_reversal_to_direction(probs_val, val_curr_dir, thresh)
        dir_acc = float((dir_pred == val_true_dir.astype(int)).mean())
        if dir_acc > best_dir_acc:
            best_dir_acc = dir_acc
            best_thresh = thresh

    # --- Test direction accuracy using best_thresh ---
    with torch.no_grad():
        probs_test = []
        for xb, _ in test_loader:
            xb = xb.to(DEVICE)
            logits = model(xb)
            probs_test.append(torch.sigmoid(logits).cpu().numpy())
        probs_test = np.concatenate(probs_test) if probs_test else np.array([])

    test_curr_dir = df_test["curr_dir"].values[context_len - 1 :]
    test_true_dir = df_test["target_direction"].values[context_len - 1 :]

    min_len = min(len(probs_test), len(test_curr_dir), len(test_true_dir))
    probs_test = probs_test[:min_len]
    test_curr_dir = test_curr_dir[:min_len]
    test_true_dir = test_true_dir[:min_len]

    test_dir_pred = convert_reversal_to_direction(probs_test, test_curr_dir, best_thresh)
    test_dir_acc = float((test_dir_pred == test_true_dir.astype(int)).mean())

    # Persistence baseline on test
    persistence_pred = test_curr_dir.astype(int)
    persistence_test_acc = float((persistence_pred == test_true_dir.astype(int)).mean())

    print("\nTEST RESULTS")
    print(f"Reversal accuracy: {test_rev_acc*100:.2f}% (loss={test_loss:.4f})")
    print(f"Direction (converted) acc: {test_dir_acc*100:.2f}% | persistence baseline: {persistence_test_acc*100:.2f}%")
    print(f"Best threshold (from VAL): {best_thresh:.2f} | VAL dir acc: {best_dir_acc*100:.2f}%")

    ckpt_path = MODELS_DIR / f"reversal_detector_{timeframe}_h{horizon}_c{context_len}.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "scaler": scaler,
            "feature_cols": feature_cols,
            "context_len": context_len,
            "horizon": horizon,
            "timeframe": timeframe,
            "best_threshold": best_thresh,
            "test_reversal_acc": test_rev_acc,
            "test_direction_acc": test_dir_acc,
            "persistence_test_acc": persistence_test_acc,
            "reversal_rate": reversal_rate,
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
    p.add_argument("--epochs", "-e", type=int, default=100)
    p.add_argument("--batch", "-b", type=int, default=64)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=15)
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
        patience=args.patience,
    )