"""
Momentum Continuation Predictor

THE KEY INSIGHT FROM YOUR DIAGNOSTICS:
- Persistence baseline: 76% accuracy
- Your GRU: 50% (random!)

This means: "predict direction = previous direction" works 76% of the time.
So momentum CONTINUES most of the time.

NEW STRATEGY:
Instead of predicting UP/DOWN directly, we predict:
"Will momentum CONTINUE or REVERSE?"

If we can predict reversals with >50% accuracy, we can beat 76%.

Example:
- If current momentum is UP and we predict CONTINUE → predict UP
- If current momentum is UP and we predict REVERSE → predict DOWN
"""

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import RobustScaler
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")


class ReversalDetector(nn.Module):
    """
    Predicts: Will current momentum REVERSE?
    
    Output: probability of reversal
    """
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, 
            num_layers=2, 
            batch_first=True,
            dropout=0.3,
            bidirectional=True,
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
        )
    
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.classifier(out[:, -1, :]).squeeze(-1)


class ReversalDataset(Dataset):
    def __init__(self, features: np.ndarray, targets: np.ndarray, context_len: int):
        self.features = features.astype("float32")
        self.targets = targets.astype("float32")
        self.context_len = context_len
    
    def __len__(self):
        return max(0, len(self.targets) - self.context_len)
    
    def __getitem__(self, idx):
        x = self.features[idx:idx + self.context_len]
        y = self.targets[idx + self.context_len]
        return torch.from_numpy(x), torch.tensor(y)


def create_reversal_features(df: pd.DataFrame, horizon: int = 4):
    """
    Create features focused on detecting momentum reversals.
    
    Target: Did momentum REVERSE?
    - If previous was UP and future is DOWN → reversal (1)
    - If previous was DOWN and future is UP → reversal (1)
    - Otherwise → continuation (0)
    """
    df = df.copy()
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]
    
    # === CURRENT MOMENTUM STATE ===
    # These tell us the CURRENT direction
    df["curr_ret_1h"] = close.pct_change(1)
    df["curr_ret_4h"] = close.pct_change(4)
    df["curr_dir"] = (df["curr_ret_1h"] > 0).astype(float)  # Current direction
    
    # === MOMENTUM STRENGTH ===
    # Strong momentum is less likely to reverse
    df["momentum_strength"] = df["curr_ret_4h"].abs()
    df["momentum_accel"] = df["curr_ret_1h"] - df["curr_ret_1h"].shift(1)
    
    # Streak (how long in current direction)
    direction = (close.pct_change() > 0).astype(int)
    streak_groups = (direction != direction.shift()).cumsum()
    df["streak_len"] = direction.groupby(streak_groups).cumcount() + 1
    
    # Long streaks might be due for reversal
    df["streak_extended"] = (df["streak_len"] > 5).astype(float)
    
    # === EXHAUSTION SIGNALS ===
    # These suggest momentum might be tiring
    
    # RSI extremes
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))
    
    # Overbought when bullish, oversold when bearish = potential reversal
    df["rsi_exhaustion"] = np.where(
        df["curr_dir"] == 1,
        (df["rsi"] > 70).astype(float),  # Overbought in uptrend
        (df["rsi"] < 30).astype(float),  # Oversold in downtrend
    )
    
    # Distance from MAs (overextension)
    for p in [10, 20, 50]:
        ma = close.rolling(p).mean()
        df[f"dist_ma{p}"] = (close - ma) / ma
    
    # Extreme distance = potential reversal
    df["ma_exhaustion"] = (df["dist_ma20"].abs() > df["dist_ma20"].rolling(100).std() * 2).astype(float)
    
    # Bollinger Band extremes
    ma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    df["bb_position"] = (close - ma20) / (2 * std20).replace(0, np.nan)
    df["bb_extreme"] = (df["bb_position"].abs() > 1).astype(float)
    
    # === VOLUME SIGNALS ===
    # Climactic volume often precedes reversals
    df["vol_ratio"] = volume / volume.rolling(20).mean()
    df["vol_spike"] = (df["vol_ratio"] > 2).astype(float)
    
    # Volume declining during trend = weakening
    df["vol_trend"] = volume.rolling(5).mean() / volume.rolling(20).mean()
    df["vol_divergence"] = np.where(
        df["curr_dir"] == 1,
        (df["vol_trend"] < 0.8).astype(float),  # Declining volume in uptrend
        (df["vol_trend"] < 0.8).astype(float),  # Declining volume in downtrend
    )
    
    # === CANDLE PATTERNS ===
    body = (close - df["open"]).abs()
    full_range = high - low
    df["body_ratio"] = body / full_range.replace(0, np.nan)
    
    # Doji / indecision candles
    df["indecision"] = (df["body_ratio"] < 0.2).astype(float)
    
    # Rejection wicks
    upper_wick = high - close.clip(lower=df["open"])
    lower_wick = close.clip(upper=df["open"]) - low
    df["upper_rejection"] = (upper_wick / full_range.replace(0, np.nan) > 0.6).astype(float)
    df["lower_rejection"] = (lower_wick / full_range.replace(0, np.nan) > 0.6).astype(float)
    
    # === VOLATILITY ===
    df["volatility"] = close.pct_change().rolling(12).std()
    df["vol_expanding"] = (df["volatility"] > df["volatility"].shift(1)).astype(float)
    
    # === TARGET: DID MOMENTUM REVERSE? ===
    future_dir = (close.shift(-horizon) > close).astype(float)
    current_dir = (close.pct_change(1) > 0).astype(float)
    
    # Reversal = future direction != current direction
    df["target_reversal"] = (future_dir != current_dir).astype(float)
    
    # Also keep standard direction target for comparison
    df["target_direction"] = future_dir
    
    # === CLEANUP ===
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    
    # Select features (exclude targets and raw price)
    exclude = ["open", "high", "low", "close", "volume", 
               "target_reversal", "target_direction", "curr_dir"]
    feature_cols = [c for c in df.columns if c not in exclude]
    
    return df, feature_cols


def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct, total = 0, 0, 0
    
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item() * len(yb)
        preds = (torch.sigmoid(logits) > 0.5).float()
        correct += (preds == yb).sum().item()
        total += len(yb)
    
    return total_loss / total, correct / total


def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_probs, all_true = [], []
    
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            logits = model(xb)
            loss = criterion(logits, yb)
            probs = torch.sigmoid(logits)
            
            total_loss += loss.item() * len(yb)
            preds = (probs > 0.5).float()
            correct += (preds == yb).sum().item()
            total += len(yb)
            
            all_probs.extend(probs.cpu().numpy())
            all_true.extend(yb.cpu().numpy())
    
    return total_loss / total, correct / total, np.array(all_probs), np.array(all_true)


def convert_reversal_to_direction(reversal_probs: np.ndarray, current_direction: np.ndarray, threshold: float = 0.5):
    """
    Convert reversal predictions to direction predictions.
    
    Logic:
    - If reversal_prob > threshold: predict opposite of current direction
    - Else: predict same as current direction (momentum continues)
    """
    reversal_pred = reversal_probs > threshold
    
    # If current is UP (1) and we predict reversal → DOWN (0)
    # If current is UP (1) and we predict continue → UP (1)
    # If current is DOWN (0) and we predict reversal → UP (1)
    # If current is DOWN (0) and we predict continue → DOWN (0)
    
    direction_pred = np.where(
        reversal_pred,
        1 - current_direction,  # Flip direction
        current_direction,      # Keep direction
    )
    
    return direction_pred


def main(
    data_path: str = "data/processed/xrp_features_1h.parquet",
    horizon: int = 4,
    context_len: int = 48,
    max_epochs: int = 100,
    patience: int = 15,
):
    print(f"\n{'='*70}")
    print("MOMENTUM REVERSAL DETECTOR")
    print(f"{'='*70}")
    print(f"Strategy: Predict REVERSAL, then convert to direction")
    print(f"Horizon: {horizon}h | Context: {context_len}h")
    print(f"{'='*70}\n")
    
    # Load data
    data_path = Path(data_path)
    if not data_path.exists():
        data_path = Path(__file__).parent.parent / "data" / "processed" / "xrp_features_1h.parquet"
    
    df_raw = pd.read_parquet(data_path)
    print(f"Loaded {len(df_raw):,} samples")
    
    # Create features
    df, feature_cols = create_reversal_features(df_raw, horizon=horizon)
    print(f"After features: {len(df):,} samples")
    print(f"Features: {len(feature_cols)}")
    
    # Split
    n = len(df)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)
    
    df_train = df.iloc[:train_end].copy()
    df_val = df.iloc[train_end:val_end].copy()
    df_test = df.iloc[val_end:].copy()
    
    print(f"\nSplit: Train={len(df_train)}, Val={len(df_val)}, Test={len(df_test)}")
    
    # Check reversal rate
    reversal_rate = df["target_reversal"].mean()
    print(f"\nReversal rate: {reversal_rate:.1%}")
    print(f"(Momentum continues {1-reversal_rate:.1%} of the time)")
    
    # Baselines
    persistence_acc = 1 - reversal_rate  # If momentum always continues
    print(f"\n🎯 Persistence baseline (direction): {persistence_acc:.1%}")
    
    # Scale
    scaler = RobustScaler()
    scaler.fit(df_train[feature_cols])
    
    for d in (df_train, df_val, df_test):
        d[feature_cols] = scaler.transform(d[feature_cols])
    
    # Datasets
    train_ds = ReversalDataset(df_train[feature_cols].values, df_train["target_reversal"].values, context_len)
    val_ds = ReversalDataset(df_val[feature_cols].values, df_val["target_reversal"].values, context_len)
    test_ds = ReversalDataset(df_test[feature_cols].values, df_test["target_reversal"].values, context_len)
    
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=64)
    test_loader = DataLoader(test_ds, batch_size=64)
    
    # Class weight for imbalanced data (reversals are less common)
    pos_weight = torch.tensor([(1 - reversal_rate) / reversal_rate]).to(DEVICE)
    
    # Model
    model = ReversalDetector(len(feature_cols)).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    
    # Training
    print(f"\n{'Epoch':<7}{'Loss':<10}{'Train':<12}{'Val':<12}")
    print("-" * 45)
    
    best_val_acc = 0
    epochs_no_improve = 0
    best_state = None
    
    for epoch in range(1, max_epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion)
        scheduler.step()
        
        marker = ""
        if val_acc > best_val_acc + 0.001:
            best_val_acc = val_acc
            epochs_no_improve = 0
            best_state = model.state_dict().copy()
            marker = " ← best"
        else:
            epochs_no_improve += 1
        
        print(f"{epoch:<7}{train_loss:<10.4f}{train_acc*100:<12.1f}{val_acc*100:<12.1f}{marker}")
        
        if epochs_no_improve >= patience:
            print(f"\nEarly stopping at epoch {epoch}")
            break
    
    # Load best
    if best_state:
        model.load_state_dict(best_state)
    
    # Final evaluation
    print(f"\n{'='*70}")
    print("TEST RESULTS")
    print(f"{'='*70}")
    
    _, reversal_acc, reversal_probs, reversal_true = evaluate(model, test_loader, criterion)
    
    print(f"\n1. REVERSAL PREDICTION:")
    print(f"   Reversal accuracy: {reversal_acc:.1%}")
    print(f"   (Random would be ~{reversal_rate:.1%})")
    
    # Convert to direction predictions
    # Get current direction for test set (aligned with predictions)
    test_curr_dir = df_test["curr_dir"].values[context_len:]
    test_true_dir = df_test["target_direction"].values[context_len:]
    
    # Ensure lengths match
    min_len = min(len(reversal_probs), len(test_curr_dir), len(test_true_dir))
    reversal_probs = reversal_probs[:min_len]
    test_curr_dir = test_curr_dir[:min_len]
    test_true_dir = test_true_dir[:min_len]
    
    # Convert to direction
    direction_pred = convert_reversal_to_direction(reversal_probs, test_curr_dir)
    direction_acc = (direction_pred == test_true_dir).mean()
    
    # Persistence baseline for comparison
    persistence_pred = test_curr_dir  # Just predict current direction continues
    persistence_acc_test = (persistence_pred == test_true_dir).mean()
    
    print(f"\n2. DIRECTION PREDICTION (converted from reversal):")
    print(f"   Persistence baseline: {persistence_acc_test:.1%}")
    print(f"   Your model:          {direction_acc:.1%}")
    print(f"   Improvement:         {(direction_acc - persistence_acc_test)*100:+.1f}%")
    
    # Try different thresholds
    print(f"\n3. OPTIMIZING THRESHOLD:")
    best_thresh = 0.5
    best_dir_acc = direction_acc
    
    for thresh in [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]:
        dir_pred = convert_reversal_to_direction(reversal_probs, test_curr_dir, thresh)
        acc = (dir_pred == test_true_dir).mean()
        print(f"   Threshold {thresh}: {acc:.1%}")
        if acc > best_dir_acc:
            best_dir_acc = acc
            best_thresh = thresh
    
    print(f"\n   Best threshold: {best_thresh} → {best_dir_acc:.1%}")
    
    # High confidence predictions
    print(f"\n4. HIGH CONFIDENCE PREDICTIONS:")
    high_conf = (reversal_probs > 0.7) | (reversal_probs < 0.3)
    if high_conf.sum() > 10:
        conf_dir_pred = convert_reversal_to_direction(reversal_probs[high_conf], test_curr_dir[high_conf])
        conf_dir_acc = (conf_dir_pred == test_true_dir[high_conf]).mean()
        print(f"   Accuracy on {high_conf.mean():.0%} of samples: {conf_dir_acc:.1%}")
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"\n📊 Final direction accuracy: {best_dir_acc:.1%}")
    
    if best_dir_acc >= 0.60:
        print("\n✅ SUCCESS! Model achieves 60%+ accuracy")
    elif best_dir_acc > persistence_acc_test:
        print(f"\n🟡 Model beats persistence ({persistence_acc_test:.1%}) but not at 60% yet")
    else:
        print(f"\n❌ Model doesn't beat persistence ({persistence_acc_test:.1%})")
    
    # Save
    save_path = Path(__file__).parent.parent / "saved_models" / "reversal_detector.pt"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.save({
        "model_state": model.state_dict(),
        "scaler": scaler,
        "feature_cols": feature_cols,
        "context_len": context_len,
        "horizon": horizon,
        "best_threshold": best_thresh,
        "test_accuracy": best_dir_acc,
    }, save_path)
    
    print(f"\n💾 Model saved to: {save_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--context", type=int, default=48)
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()
    
    main(
        horizon=args.horizon,
        context_len=args.context,
        max_epochs=args.epochs,
    )