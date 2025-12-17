"""
Simple Trade Signal (BUY/SELL/HOLD)

Uses Option A (reversal detector) to predict direction:
- Model predicts P(reversal) over horizon h.
- Compute curr_dir from last 1-step return.
- If P(reversal) > threshold => flip curr_dir else keep curr_dir.

Optionally uses price regressor to estimate expected move magnitude:
- price model predicts y_hat = log(close[t+h]/close[t])
- expected_move_pct = abs(exp(y_hat) - 1) * 100

Signal rule (simple + explicit):
- Define "high confidence" as:
    reversal_prob <= conf_low  (high-confidence CONTINUE)
    reversal_prob >= conf_high (high-confidence REVERSE)
  otherwise low confidence band.

- If expected_move_pct < min_move_pct => HOLD
- Else if high confidence:
    predicted_dir UP   => BUY
    predicted_dir DOWN => SELL
  Else => HOLD

Usage:
  python -m inference.trade_signal --timeframe 1h --horizon 4 --context 48 --min_move_pct 0.25 --conf_low 0.30 --conf_high 0.70
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.preprocessing import RobustScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "saved_models"


# -----------------------
# Models (must match training)
# -----------------------

class ReversalDetector(nn.Module):
    """Binary classifier: reversal(1) vs continue(0) -> outputs logits"""
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
        return self.head(out[:, -1, :]).squeeze(-1)


class PriceRegressor(nn.Module):
    """Regresses log-return y_hat = log(close[t+h]/close[t])"""
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
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


# -----------------------
# Feature building (must match Option A training)
# -----------------------

def create_reversal_features(df: pd.DataFrame, horizon: int) -> Tuple[pd.DataFrame, list[str]]:
    """
    Rebuild the same reversal-focused features from OHLCV (NO future-leaking shift(-k) features).
    Also computes:
      - curr_dir at time t
      - target_direction at t (for training; inference only uses curr_dir)
    """
    df = df.copy()

    close = df["close"]
    high = df["high"]
    low = df["low"]
    open_ = df["open"]
    volume = df["volume"]

    # Current momentum
    df["curr_ret_1"] = close.pct_change(1)
    df["curr_ret_4"] = close.pct_change(4)
    df["curr_dir"] = (df["curr_ret_1"] > 0).astype(float)  # 1=up, 0=down

    # Momentum strength / acceleration
    df["momentum_strength"] = df["curr_ret_4"].abs()
    df["momentum_accel"] = df["curr_ret_1"] - df["curr_ret_1"].shift(1)

    # Streak
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

    # Bollinger position
    ma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    df["bb_position"] = (close - ma20) / (2 * std20).replace(0, np.nan)
    df["bb_extreme"] = (df["bb_position"].abs() > 1).astype(float)

    # Volume
    df["vol_ratio"] = volume / volume.rolling(20).mean()
    df["vol_spike"] = (df["vol_ratio"] > 2).astype(float)
    df["vol_trend"] = volume.rolling(5).mean() / volume.rolling(20).mean()
    df["vol_divergence"] = (df["vol_trend"] < 0.8).astype(float)

    # Candles
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

    # Targets (not needed for inference but keeps alignment consistent)
    future_dir = (close.shift(-horizon) > close).astype(float)
    current_dir = (close.pct_change(1) > 0).astype(float)
    df["target_reversal"] = (future_dir != current_dir).astype(float)
    df["target_direction"] = future_dir

    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    exclude = [
        "open", "high", "low", "close", "volume",
        "target_reversal", "target_direction",
        "curr_dir",
    ]
    feature_cols = [c for c in df.columns if c not in exclude]
    return df, feature_cols


# -----------------------
# Helpers
# -----------------------

def try_load_price_ckpt(timeframe: str, horizon: int, context_len: int) -> Optional[Path]:
    """
    Your project has used a couple ckpt name patterns.
    Try both.
    """
    candidates = [
        MODELS_DIR / f"lstm_price_regressor_{timeframe}_h{horizon}_c{context_len}.pt",
        MODELS_DIR / f"lstm_price_regressor_mt_{timeframe}_h{horizon}_c{context_len}.pt",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def direction_str(d: int) -> str:
    return "UP" if int(d) == 1 else "DOWN"


# -----------------------
# Main inference
# -----------------------

@torch.no_grad()
def main(
    timeframe: str = "1h",
    horizon: int = 4,
    context_len: int = 48,
    min_move_pct: float = 0.25,
    conf_low: float = 0.30,
    conf_high: float = 0.70,
    threshold: Optional[float] = None,
    use_price_model: bool = True,
):
    feat_path = DATA_DIR / f"xrp_features_{timeframe}.parquet"
    if not feat_path.exists():
        raise FileNotFoundError(f"Missing {feat_path}. Run collector + feature pipeline first.")

    df_raw = pd.read_parquet(feat_path).sort_index()

    # --- Load reversal detector checkpoint ---
    rev_ckpt_path = MODELS_DIR / f"reversal_detector_{timeframe}_h{horizon}_c{context_len}.pt"
    if not rev_ckpt_path.exists():
        raise FileNotFoundError(
            f"Missing {rev_ckpt_path}. Train first:\n"
            f"  python -m training.train_reversal_detector --timeframe {timeframe} --horizon {horizon} --context {context_len}"
        )

    rev_ckpt = torch.load(rev_ckpt_path, map_location=DEVICE, weights_only=False)
    rev_feature_cols = rev_ckpt["feature_cols"]
    rev_scaler: RobustScaler = rev_ckpt["scaler"]
    best_threshold = float(rev_ckpt.get("best_threshold", 0.5))
    if threshold is None:
        threshold = best_threshold

    arch = rev_ckpt.get("arch", {})
    hidden_dim = int(arch.get("hidden_dim", 64))
    num_layers = int(arch.get("num_layers", 2))
    dropout = float(arch.get("dropout", 0.3))

    # rebuild features the same way as training
    df_rev, _ = create_reversal_features(df_raw, horizon=horizon)

    if len(df_rev) < context_len:
        raise ValueError(f"Not enough rows after feature creation. Have {len(df_rev)}, need >= {context_len}.")

    # scale using the checkpoint scaler (trained on train split)
    X_rev = df_rev[rev_feature_cols].copy()
    X_rev.loc[:, :] = rev_scaler.transform(X_rev.values)

    # last window
    xwin = X_rev.values.astype("float32")[-context_len:]
    curr_dir = int(df_rev["curr_dir"].values[-1])
    last_close = float(df_rev["close"].values[-1]) if "close" in df_rev.columns else float(df_raw["close"].values[-1])
    last_time = df_rev.index[-1]

    model_rev = ReversalDetector(
        input_dim=len(rev_feature_cols),
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    ).to(DEVICE)
    model_rev.load_state_dict(rev_ckpt["model_state"])
    model_rev.eval()

    xb = torch.from_numpy(xwin).unsqueeze(0).to(DEVICE)
    logits = model_rev(xb)
    reversal_prob = float(torch.sigmoid(logits).item())

    # convert to direction
    flip = reversal_prob > float(threshold)
    pred_dir = int(1 - curr_dir) if flip else int(curr_dir)

    # confidence band (based on reversal prob)
    high_conf = (reversal_prob <= conf_low) or (reversal_prob >= conf_high)

    # --- Optional: load price regressor to estimate move magnitude ---
    expected_move_pct = None
    pred_price = None

    if use_price_model:
        price_ckpt_path = try_load_price_ckpt(timeframe, horizon, context_len)
        if price_ckpt_path is not None:
            price_ckpt = torch.load(price_ckpt_path, map_location=DEVICE, weights_only=False)
            price_cols = price_ckpt["feature_cols"]
            price_scaler: RobustScaler = price_ckpt["scaler"]

            # We use the original feature parquet columns for the price model window
            df_p = df_raw.copy().sort_index()
            # Ensure needed columns exist
            missing = [c for c in price_cols if c not in df_p.columns]
            if missing:
                print(f"⚠️ Price model skipped: missing columns in parquet: {missing[:5]}{'...' if len(missing)>5 else ''}")
            else:
                Xp = df_p[price_cols].copy()
                Xp.loc[:, :] = price_scaler.transform(Xp.values)
                if len(Xp) >= context_len:
                    xwin_p = Xp.values.astype("float32")[-context_len:]
                    base_price = float(df_p["close"].values[-1])

                    model_p = PriceRegressor(input_dim=len(price_cols)).to(DEVICE)
                    model_p.load_state_dict(price_ckpt["model_state"])
                    model_p.eval()

                    yhat = float(model_p(torch.from_numpy(xwin_p).unsqueeze(0).to(DEVICE)).item())
                    pred_price = float(base_price * np.exp(yhat))
                    expected_move_pct = float(abs(np.exp(yhat) - 1.0) * 100.0)
        else:
            print("⚠️ Price model ckpt not found (skipping expected move filter).")

    # move filter
    move_ok = True
    if expected_move_pct is not None:
        move_ok = expected_move_pct >= float(min_move_pct)

    # final signal
    if (high_conf and move_ok):
        signal = "BUY" if pred_dir == 1 else "SELL"
    else:
        signal = "HOLD"

    # Pretty print
    print("\n" + "=" * 70)
    print("TRADE SIGNAL (simple)")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"Timeframe={timeframe} | Horizon={horizon}h | Context={context_len}")
    print(f"Last timestamp: {last_time}")
    print(f"Last close: {last_close:.6f}")
    print("-" * 70)
    print(f"Current dir (from last 1-step return): {direction_str(curr_dir)}")
    print(f"P(reversal): {reversal_prob:.4f}")
    print(f"Threshold used: {threshold:.2f} (ckpt best={best_threshold:.2f}) -> flip={flip}")
    print(f"Predicted dir: {direction_str(pred_dir)} | HighConfBand={high_conf} (<= {conf_low} or >= {conf_high})")

    if expected_move_pct is not None:
        print(f"Expected move (price model): {expected_move_pct:.3f}% | min_move_pct={min_move_pct:.3f}% | move_ok={move_ok}")
    else:
        print(f"Expected move (price model): N/A | min_move_pct={min_move_pct:.3f}% (skipped)")

    if pred_price is not None:
        print(f"Predicted price (price model): {pred_price:.6f}")

    print("-" * 70)
    print(f"SIGNAL: {signal}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--timeframe", "-t", default="1h")
    p.add_argument("--horizon", "-H", type=int, default=4)
    p.add_argument("--context", "-c", type=int, default=48)
    p.add_argument("--min_move_pct", type=float, default=0.25)
    p.add_argument("--conf_low", type=float, default=0.30)
    p.add_argument("--conf_high", type=float, default=0.70)
    p.add_argument("--threshold", type=float, default=None, help="Override ckpt best_threshold for reversal->direction conversion.")
    p.add_argument("--no_price_model", action="store_true", help="Disable price regressor (no expected-move filter).")
    args = p.parse_args()

    main(
        timeframe=args.timeframe,
        horizon=args.horizon,
        context_len=args.context,
        min_move_pct=args.min_move_pct,
        conf_low=args.conf_low,
        conf_high=args.conf_high,
        threshold=args.threshold,
        use_price_model=(not args.no_price_model),
    )