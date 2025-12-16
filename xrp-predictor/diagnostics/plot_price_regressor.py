"""
Plot Predicted vs Actual price for the price regressor.

Usage:
  python -m diagnostics.plot_price_regressor --timeframe 1h --horizon 4 --context 48 --n_points 800
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "saved_models"


class PriceRegressor(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.lstm = torch.nn.LSTM(
            input_size=input_dim,              # ✅ FIX (was input_dim=...)
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.head = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim * 2, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        yhat = self.head(out[:, -1, :])
        return yhat.squeeze(-1)


def make_target_logret(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    df = df.copy()
    df["y_logret"] = np.log(df["close"].shift(-horizon) / df["close"])
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df


def select_feature_cols(df: pd.DataFrame) -> list[str]:
    drop_future = [c for c in df.columns if c.startswith("ret_") or c.startswith("log_ret_")]
    keep = [c for c in df.columns if c not in drop_future and c != "y_logret"]
    return keep


@torch.no_grad()
def main(timeframe: str = "1h", horizon: int = 4, context_len: int = 48, n_points: int = 800):
    feat_path = DATA_DIR / f"xrp_features_{timeframe}.parquet"
    df = pd.read_parquet(feat_path).sort_index()
    df = make_target_logret(df, horizon=horizon)
    feature_cols = select_feature_cols(df)

    # Same split rule as training script
    n = len(df)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)
    df_train = df.iloc[:train_end].copy()
    df_test = df.iloc[val_end:].copy()

    # Fit scaler on train only
    scaler = RobustScaler()
    scaler.fit(df_train[feature_cols])

    df_test_s = df_test.copy()
    df_test_s[feature_cols] = scaler.transform(df_test_s[feature_cols])

    ckpt_path = MODELS_DIR / f"lstm_price_regressor_{timeframe}_h{horizon}_c{context_len}.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing {ckpt_path}. Train first.")

    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)

    model = PriceRegressor(input_dim=len(ckpt["feature_cols"])).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    X = df_test_s[ckpt["feature_cols"]].values.astype("float32")
    close = df_test["close"].values.astype("float32")
    fut_close = df_test["close"].shift(-horizon).values.astype("float32")
    idx = df_test.index

    # walk-forward predict last n_points windows
    end = len(df_test) - horizon - 1

    # if n_points is -1 or 0, plot the full test range
    if n_points is None or n_points <= 0:
        start = context_len - 1
    else:
        start = max(context_len - 1, end - n_points)

    times = []
    pred_prices = []
    act_prices = []

    for k in range(start, end):
        # window ends at k
        xwin = X[k - context_len + 1 : k + 1]
        base = close[k]
        actual = fut_close[k]
        if np.isnan(actual):
            continue

        xb = torch.from_numpy(xwin).unsqueeze(0).to(DEVICE)
        yhat = model(xb).item()  # logret

        pred = float(base * np.exp(yhat))

        times.append(idx[k] + pd.Timedelta(hours=horizon))
        pred_prices.append(pred)
        act_prices.append(float(actual))

    pred_prices = np.array(pred_prices)
    act_prices = np.array(act_prices)

    mae = float(np.mean(np.abs(pred_prices - act_prices)))
    mape = float(np.mean(np.abs((pred_prices - act_prices) / act_prices)) * 100)

    print(f"MAE=${mae:.6f} | MAPE={mape:.2f}% | points={len(pred_prices)}")

    plt.figure(figsize=(14, 6))
    plt.plot(times, act_prices, label="Actual (t + horizon)")
    plt.plot(times, pred_prices, label="Predicted (LSTM regressor)", linestyle="--")
    plt.title(f"Predicted vs Actual XRP Price (horizon={horizon}h)")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.grid(True, alpha=0.3)
    plt.legend()
    out = ROOT / "diagnostics_lstm_regressor_pred_vs_actual.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    print(f"Saved: {out}")
    plt.show()


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--timeframe", "-t", default="1h")
    p.add_argument("--horizon", "-H", type=int, default=4)
    p.add_argument("--context", "-c", type=int, default=48)
    p.add_argument("--n_points", "-n", type=int, default=-1,
               help="How many points to plot from the end. Use -1 for full test set.")
    args = p.parse_args()
    main(timeframe=args.timeframe, horizon=args.horizon, context_len=args.context, n_points=args.n_points)