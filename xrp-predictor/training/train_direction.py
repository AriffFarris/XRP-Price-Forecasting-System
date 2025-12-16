"""
Direction Classifier V2 - Momentum Focused

Key insight: Persistence model gets 76% accuracy using just:
    "Was last return positive? Predict positive."

So we BUILD ON momentum, not replace it.

Strategy:
1. Use raw momentum as PRIMARY signal
2. Use indicators only to FILTER (when to trust momentum)
3. Predict: "Will momentum CONTINUE or REVERSE?"
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import RobustScaler
from sklearn.utils.class_weight import compute_class_weight


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "saved_models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


class GRUClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(
            input_dim, hidden_dim, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, x):
        out, _ = self.gru(x)
        return self.classifier(out[:, -1, :]).squeeze(-1)


class DirectionDataset(Dataset):
    def __init__(self, df, feature_cols, target_col, context_len):
        self.context_len = context_len
        self.features = df[feature_cols].values.astype("float32")
        self.targets = (df[target_col].values > 0).astype("float32")
    
    def __len__(self):
        return max(0, len(self.targets) - self.context_len)
    
    def __getitem__(self, idx):
        x = self.features[idx : idx + self.context_len]
        y = self.targets[idx + self.context_len]
        return torch.from_numpy(x), torch.tensor(y)


def create_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create features that COMPLEMENT momentum, not replace it.
    
    The key insight: we want to predict when momentum CONTINUES vs REVERSES.
    """
    close = df["close"].copy()
    high = df["high"].copy()
    low = df["low"].copy()
    volume = df["volume"].copy()
    
    # =================================================================
    # PRIMARY: RAW MOMENTUM (this is what persistence uses!)
    # =================================================================
    
    # Recent returns - MOST IMPORTANT FEATURES
    for h in [1, 2, 3, 4, 6, 8, 12, 24]:
        df[f"ret_{h}h"] = close.pct_change(h)
    
    # Direction of recent returns (binary)
    for h in [1, 2, 4]:
        df[f"dir_{h}h"] = (close.pct_change(h) > 0).astype(float)
    
    # Momentum strength (absolute return)
    df["momentum_strength"] = close.pct_change(4).abs()
    
    # =================================================================
    # MOMENTUM CONTINUATION SIGNALS
    # =================================================================
    
    # Consecutive same-direction candles
    ret_1h = close.pct_change()
    direction = (ret_1h > 0).astype(int)
    
    # Streak length
    streak_groups = (direction != direction.shift()).cumsum()
    df["streak_len"] = direction.groupby(streak_groups).cumcount() + 1
    
    # Positive or negative streak
    df["streak_sign"] = direction * 2 - 1  # +1 for up, -1 for down
    df["signed_streak"] = df["streak_len"] * df["streak_sign"]
    
    # =================================================================
    # REVERSAL SIGNALS (when momentum might stop)
    # =================================================================
    
    # RSI extremes (overbought/oversold)
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))
    
    # RSI zones
    df["rsi_overbought"] = (df["rsi"] > 70).astype(float)
    df["rsi_oversold"] = (df["rsi"] < 30).astype(float)
    df["rsi_extreme"] = df["rsi_overbought"] + df["rsi_oversold"]
    
    # Distance from moving averages (mean reversion signal)
    for period in [10, 20, 50]:
        ma = close.rolling(period).mean()
        df[f"dist_ma{period}"] = (close - ma) / ma
    
    # Bollinger Band position
    ma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    df["bb_upper"] = ma20 + 2 * std20
    df["bb_lower"] = ma20 - 2 * std20
    df["bb_position"] = (close - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])
    df["bb_outside"] = ((close > df["bb_upper"]) | (close < df["bb_lower"])).astype(float)
    
    # =================================================================
    # VOLUME CONFIRMATION
    # =================================================================
    
    # Volume relative to average
    df["vol_ratio"] = volume / volume.rolling(20).mean()
    
    # Volume trend
    df["vol_change"] = volume.pct_change()
    
    # Price-volume divergence
    df["pv_divergence"] = df["ret_1h"] * df["vol_change"]  # Should be positive if healthy
    
    # =================================================================
    # VOLATILITY (affects prediction confidence)
    # =================================================================
    
    df["volatility"] = ret_1h.rolling(12).std()
    df["volatility_ratio"] = df["volatility"] / ret_1h.rolling(48).std()
    
    # ATR
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14).mean()
    df["atr_ratio"] = df["atr"] / close
    
    # =================================================================
    # PATTERN FEATURES
    # =================================================================
    
    # Higher highs / lower lows
    df["higher_high"] = (high > high.shift(1)).astype(float)
    df["lower_low"] = (low < low.shift(1)).astype(float)
    
    # Candle body ratio
    body = (close - df["open"]).abs()
    wick = high - low
    df["body_ratio"] = body / wick.replace(0, np.nan)
    
    # Drop intermediate columns
    drop_cols = ["bb_upper", "bb_lower", "streak_sign"]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)
    
    return df


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
    
    all_probs = np.array(all_probs)
    all_true = np.array(all_true)
    
    # High confidence accuracy
    confident = (all_probs > 0.6) | (all_probs < 0.4)
    if confident.sum() > 0:
        conf_acc = ((all_probs[confident] > 0.5) == all_true[confident]).mean()
        conf_pct = confident.mean()
    else:
        conf_acc, conf_pct = correct/total, 0
    
    return total_loss/total, correct/total, conf_acc, conf_pct


def main(horizon=4, context_len=24, max_epochs=100, patience=15):
    print(f"\n{'='*60}")
    print("MOMENTUM-FOCUSED DIRECTION CLASSIFIER")
    print(f"{'='*60}")
    print(f"Predicting: UP/DOWN in {horizon}h")
    print(f"Context: {context_len}h")
    print(f"{'='*60}\n")
    
    # Load data
    df = pd.read_parquet(DATA_DIR / "xrp_features_1h.parquet")
    print(f"Loaded {len(df):,} samples")
    
    # Create momentum features
    df = create_momentum_features(df)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    print(f"After features: {len(df):,} samples")
    
    # Target
    target_col = f"log_ret_{horizon}h"
    if target_col not in df.columns:
        df[target_col] = np.log(df["close"].shift(-horizon) / df["close"])
        df = df.dropna()
    
    # Select features - MOMENTUM FIRST
    exclude = ["open", "high", "low", "close", "volume", 
               "ret_1h", "log_ret_1h", "ret_4h", "log_ret_4h", "ret_24h"]
    feature_cols = [c for c in df.columns if c not in exclude]
    
    # Prioritize momentum features
    momentum_cols = [c for c in feature_cols if any(x in c for x in ["ret_", "dir_", "streak", "momentum"])]
    other_cols = [c for c in feature_cols if c not in momentum_cols]
    feature_cols = momentum_cols + other_cols
    
    print(f"Momentum features: {len(momentum_cols)}")
    print(f"Other features: {len(other_cols)}")
    print(f"Total features: {len(feature_cols)}")
    
    # Split
    n = len(df)
    train_end, val_end = int(n * 0.7), int(n * 0.85)
    
    df_train = df.iloc[:train_end].copy()
    df_val = df.iloc[train_end:val_end].copy()
    df_test = df.iloc[val_end:].copy()
    
    print(f"Split: Train={len(df_train)}, Val={len(df_val)}, Test={len(df_test)}")
    
    # Scale
    scaler = RobustScaler()
    scaler.fit(df_train[feature_cols])
    for d in (df_train, df_val, df_test):
        d[feature_cols] = scaler.transform(d[feature_cols])
    
    # Datasets
    train_ds = DirectionDataset(df_train, feature_cols, target_col, context_len)
    val_ds = DirectionDataset(df_val, feature_cols, target_col, context_len)
    test_ds = DirectionDataset(df_test, feature_cols, target_col, context_len)
    
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=64)
    test_loader = DataLoader(test_ds, batch_size=64)
    
    # Class weights
    train_targets = (df_train[target_col].values > 0).astype(int)[context_len:]
    weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=train_targets)
    pos_weight = torch.tensor([weights[1] / weights[0]]).to(DEVICE)
    
    # Model
    model = GRUClassifier(len(feature_cols), hidden_dim=64, num_layers=2, dropout=0.3).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    # Training
    print(f"\n{'Epoch':<7}{'Loss':<10}{'Train':<10}{'Val':<10}{'Conf Acc':<12}{'Conf %':<10}")
    print("-" * 60)
    
    best_val_acc = 0
    epochs_no_improve = 0
    
    for epoch in range(1, max_epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc, conf_acc, conf_pct = evaluate(model, val_loader, criterion)
        scheduler.step(val_acc)
        
        print(f"{epoch:<7}{train_loss:<10.4f}{train_acc*100:<10.1f}{val_acc*100:<10.1f}{conf_acc*100:<12.1f}{conf_pct*100:<10.1f}")
        
        if val_acc > best_val_acc + 0.002:
            best_val_acc = val_acc
            epochs_no_improve = 0
            torch.save({
                "model_state": model.state_dict(),
                "scaler": scaler,
                "feature_cols": feature_cols,
                "target_col": target_col,
                "context_len": context_len,
            }, MODELS_DIR / "gru_direction_v2.pt")
            print(f"        → Saved best model")
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve >= patience:
            print("\nEarly stopping.")
            break
    
    # Final test
    print(f"\n{'='*60}")
    print("TEST SET RESULTS")
    print(f"{'='*60}")
    
    ckpt = torch.load(MODELS_DIR / "gru_direction_v2.pt", map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    
    _, test_acc, test_conf_acc, test_conf_pct = evaluate(model, test_loader, criterion)
    
    # Persistence baseline
    test_returns = df_test[target_col].values[context_len:]
    prev_returns = df_test["ret_4h"].values[context_len-1:-1] if "ret_4h" in df_test.columns else df_test[target_col].shift(1).values[context_len:]
    persistence_acc = ((prev_returns > 0) == (test_returns > 0)).mean()
    
    print(f"\nPersistence baseline:      {persistence_acc*100:.1f}%")
    print(f"Your model (overall):      {test_acc*100:.1f}%")
    print(f"Your model (high conf):    {test_conf_acc*100:.1f}% (on {test_conf_pct*100:.0f}% of samples)")
    print(f"\nTarget:                    60-65%")
    
    if test_acc >= 0.60:
        print("✅ SUCCESS!")
    elif test_acc >= 0.55:
        print("⚠️  Close! Try: --horizon 1 or --context 48")
    else:
        print("❌ Try shorter horizon: python -m training.train_direction_v2 --horizon 1")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--context", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()
    
    main(horizon=args.horizon, context_len=args.context, max_epochs=args.epochs)