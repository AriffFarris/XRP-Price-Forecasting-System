"""
Improved GRU training with fixes for:
1. Model predicting too conservatively (narrow predictions)
2. Data drift between train/test
3. Better regularization and loss functions

Key changes:
- Huber loss instead of MSE (less sensitive to outliers)
- Learning rate scheduler
- Gradient clipping
- Larger context window
- Weight decay (L2 regularization)
- Optional: train on returns instead of log returns
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler, RobustScaler

from training.dataset import SequenceDataset
from models.gru_model import GRUPredictor


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "saved_models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def train_one_epoch(model, loader, criterion, optimizer, clip_grad=1.0):
    model.train()
    total_loss = 0.0
    n_batches = 0

    for xb, yb in loader:
        xb = xb.to(DEVICE)
        yb = yb.to(DEVICE)

        optimizer.zero_grad()
        preds = model(xb).squeeze(-1)
        loss = criterion(preds, yb)
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        if clip_grad:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(1, n_batches)


def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_true = []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            preds = model(xb).squeeze(-1)
            loss = criterion(preds, yb)

            total_loss += loss.item()
            all_preds.extend(preds.cpu().numpy())
            all_true.extend(yb.cpu().numpy())

    avg_loss = total_loss / max(1, len(loader))
    
    # Calculate directional accuracy
    all_preds = np.array(all_preds)
    all_true = np.array(all_true)
    
    mask = all_true != 0
    if mask.sum() > 0:
        dir_acc = (np.sign(all_preds[mask]) == np.sign(all_true[mask])).mean()
    else:
        dir_acc = 0.5
    
    return avg_loss, dir_acc


def add_mean_reversion_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add features that help the model learn mean reversion.
    These are crucial for preventing monotonic forecasts.
    """
    close = df["close"]
    
    # Distance from moving averages (as percentage)
    for period in [20, 50, 100]:
        ma = close.rolling(period).mean()
        df[f"dist_ma_{period}"] = (close - ma) / ma
    
    # RSI extremes (already have RSI, but add binary signals)
    if "rsi_14" in df.columns:
        df["rsi_overbought"] = (df["rsi_14"] > 70).astype(float)
        df["rsi_oversold"] = (df["rsi_14"] < 30).astype(float)
    
    # Bollinger Band position (where is price within the bands)
    if "bb_upper" in df.columns and "bb_lower" in df.columns:
        bb_range = df["bb_upper"] - df["bb_lower"]
        df["bb_position"] = (close - df["bb_lower"]) / bb_range.replace(0, np.nan)
    
    # Consecutive up/down days (momentum exhaustion signal)
    returns = close.pct_change()
    df["consec_up"] = (returns > 0).astype(int)
    df["consec_down"] = (returns < 0).astype(int)
    
    # Rolling count of consecutive moves
    df["consec_up_count"] = df["consec_up"].groupby(
        (df["consec_up"] != df["consec_up"].shift()).cumsum()
    ).cumsum() * df["consec_up"]
    
    df["consec_down_count"] = df["consec_down"].groupby(
        (df["consec_down"] != df["consec_down"].shift()).cumsum()
    ).cumsum() * df["consec_down"]
    
    # Clean up intermediate columns
    df.drop(columns=["consec_up", "consec_down"], inplace=True)
    
    return df


def main(
    timeframe: str = "1h",
    target: str = "log_ret_4h",  # or "log_ret_1h" for shorter horizon
    context_len: int = 48,       # increased from 24
    hidden_dim: int = 128,
    num_layers: int = 2,
    dropout: float = 0.3,        # increased from 0.2
    lr: float = 5e-4,            # reduced from 1e-3
    weight_decay: float = 1e-4,  # L2 regularization
    batch_size: int = 64,
    max_epochs: int = 100,
    patience: int = 15,          # increased from 10
):
    print(f"\n{'='*60}")
    print("IMPROVED GRU TRAINING")
    print(f"{'='*60}")
    print(f"Target: {target}")
    print(f"Context length: {context_len}")
    print(f"Hidden dim: {hidden_dim}")
    print(f"Dropout: {dropout}")
    print(f"Learning rate: {lr}")
    print(f"Weight decay: {weight_decay}")
    print(f"{'='*60}\n")

    # -------------------------
    # 1. Load features
    # -------------------------
    df = pd.read_parquet(DATA_DIR / f"xrp_features_{timeframe}.parquet")
    print(f"Loaded {len(df):,} samples")

    # -------------------------
    # 2. Add mean reversion features
    # -------------------------
    df = add_mean_reversion_features(df)
    df = df.dropna()
    print(f"After adding features: {len(df):,} samples")

    # -------------------------
    # 3. Define target & features
    # -------------------------
    target_col = target

    # Exclude all return/target columns from the feature set
    ignore_cols = [
        "ret_1h", "log_ret_1h",
        "ret_4h", "log_ret_4h",
        "ret_24h",
    ]

    feature_cols = [c for c in df.columns if c not in ignore_cols]
    print(f"Features: {len(feature_cols)} columns")

    # -------------------------
    # 4. Time-based split
    # -------------------------
    n = len(df)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)

    df_train = df.iloc[:train_end].copy()
    df_val = df.iloc[train_end:val_end].copy()
    df_test = df.iloc[val_end:].copy()

    print(f"Split: Train={len(df_train)}, Val={len(df_val)}, Test={len(df_test)}")

    # Check for data drift warning
    train_mean = df_train[target_col].mean()
    test_mean = df_test[target_col].mean()
    print(f"Train target mean: {train_mean:.6f}")
    print(f"Test target mean:  {test_mean:.6f}")
    if np.sign(train_mean) != np.sign(test_mean):
        print("⚠️  WARNING: Train and test have opposite bias!")

    # -------------------------
    # 5. Scale features (RobustScaler is less sensitive to outliers)
    # -------------------------
    scaler = RobustScaler()
    scaler.fit(df_train[feature_cols].values)

    for d in (df_train, df_val, df_test):
        d[feature_cols] = scaler.transform(d[feature_cols].values)

    # -------------------------
    # 6. Build datasets & loaders
    # -------------------------
    train_ds = SequenceDataset(df_train, feature_cols, target_col, context_len)
    val_ds = SequenceDataset(df_val, feature_cols, target_col, context_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # -------------------------
    # 7. Define model, loss, optimizer
    # -------------------------
    input_dim = len(feature_cols)

    model = GRUPredictor(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    ).to(DEVICE)

    # Huber loss is more robust to outliers than MSE
    criterion = nn.HuberLoss(delta=0.01)  # delta controls transition from L2 to L1
    
    # AdamW with weight decay for regularization
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=lr, 
        weight_decay=weight_decay
    )
    
    # Learning rate scheduler - reduce LR when validation loss plateaus
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=5,
        verbose=True
    )

    # -------------------------
    # 8. Training loop
    # -------------------------
    best_val_loss = float("inf")
    best_dir_acc = 0.0
    epochs_no_improve = 0

    print(f"\n{'='*60}")
    print("TRAINING")
    print(f"{'='*60}")

    for epoch in range(1, max_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_dir_acc = evaluate(model, val_loader, criterion)
        
        # Step the scheduler
        scheduler.step(val_loss)
        
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch:03d} | Train {train_loss:.6f} | Val {val_loss:.6f} | "
              f"Dir Acc {val_dir_acc*100:.1f}% | LR {current_lr:.2e}")

        # Check for improvement (use validation loss as primary metric)
        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_dir_acc = val_dir_acc
            epochs_no_improve = 0

            # Save checkpoint
            ckpt = {
                "model_state": model.state_dict(),
                "scaler": scaler,
                "feature_cols": feature_cols,
                "target_col": target_col,
                "context_len": context_len,
                "best_val_loss": best_val_loss,
                "best_dir_acc": best_dir_acc,
            }
            torch.save(ckpt, MODELS_DIR / "gru_xrp.pt")
            print("  -> Saved new best model")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print("Early stopping triggered.")
            break

    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Best directional accuracy: {best_dir_acc*100:.1f}%")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", "-t", default="log_ret_4h", 
                        choices=["log_ret_1h", "log_ret_4h"],
                        help="Target variable")
    parser.add_argument("--context", "-c", type=int, default=48,
                        help="Context window length")
    parser.add_argument("--epochs", "-e", type=int, default=100)
    args = parser.parse_args()
    
    main(
        target=args.target,
        context_len=args.context,
        max_epochs=args.epochs,
    )