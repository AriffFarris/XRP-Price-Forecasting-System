"""
Plot Predicted vs Actual price for the price regressor.

Usage:
  # Full test set
  python -m diagnostics.plot_price_regressor --timeframe 1h --horizon 4 --context 48 --n_points -1

  # Only last N points
  python -m diagnostics.plot_price_regressor --timeframe 1h --horizon 4 --context 48 --n_points 800

  # If using the multi-task checkpoint (direction head), also show prob(up):
  python -m diagnostics.plot_price_regressor --timeframe 1h --horizon 4 --context 48 --n_points -1 --show_dir
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


# -------------------------
# Models (old + multi-task)
# -------------------------
class PriceRegressor(torch.nn.Module):
    """Old single-head regressor (predicts logret only)."""
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.lstm = torch.nn.LSTM(
            input_size=input_dim,
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


class PriceRegressorMulti(torch.nn.Module):
    """Multi-task model: (logret, direction_logit)."""
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.lstm = torch.nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )

        trunk_dim = hidden_dim * 2

        self.reg_head = torch.nn.Sequential(
            torch.nn.Linear(trunk_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(hidden_dim, 1),
        )

        self.dir_head = torch.nn.Sequential(
            torch.nn.Linear(trunk_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor):
        out, _ = self.lstm(x)
        h = out[:, -1, :]
        yhat_logret = self.reg_head(h).squeeze(-1)
        yhat_dir_logit = self.dir_head(h).squeeze(-1)
        return yhat_logret, yhat_dir_logit


# -------------------------
# Data helpers
# -------------------------
def make_target_logret(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    df = df.copy()
    df["y_logret"] = np.log(df["close"].shift(-horizon) / df["close"])
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df


def select_feature_cols(df: pd.DataFrame) -> list[str]:
    drop_future = [c for c in df.columns if c.startswith("ret_") or c.startswith("log_ret_")]
    keep = [c for c in df.columns if c not in drop_future and c != "y_logret"]
    return keep


# -------------------------
# Metrics
# -------------------------
def regression_metrics(pred: np.ndarray, actual: np.ndarray) -> dict:
    err = pred - actual
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))

    mape = float(np.mean(np.abs(err) / np.clip(np.abs(actual), 1e-12, None)) * 100)
    smape = float(np.mean(2 * np.abs(err) / np.clip(np.abs(pred) + np.abs(actual), 1e-12, None)) * 100)

    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((actual - np.mean(actual)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return {"MAE": mae, "RMSE": rmse, "MAPE_pct": mape, "sMAPE_pct": smape, "R2": r2}


def directional_accuracy(pred: np.ndarray, base: np.ndarray, actual: np.ndarray) -> float:
    pred_dir = (pred > base).astype(int)
    true_dir = (actual > base).astype(int)
    return float((pred_dir == true_dir).mean() * 100)


# -------------------------
# Main
# -------------------------
@torch.no_grad()
def main(
    timeframe: str = "1h",
    horizon: int = 4,
    context_len: int = 48,
    n_points: int = -1,
    show_dir: bool = False,
    enforce_sign: bool = False,
    enforce_conf: float = 0.70,
):
    feat_path = DATA_DIR / f"xrp_features_{timeframe}.parquet"
    if not feat_path.exists():
        raise FileNotFoundError(f"Missing {feat_path}. Run collector + feature pipeline first.")

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

    # Prefer multi-task checkpoint, fall back to old single-head
    ckpt_mt = MODELS_DIR / f"lstm_price_regressor_mt_{timeframe}_h{horizon}_c{context_len}.pt"
    ckpt_old = MODELS_DIR / f"lstm_price_regressor_{timeframe}_h{horizon}_c{context_len}.pt"

    if ckpt_mt.exists():
        ckpt_path = ckpt_mt
        is_multi = True
    elif ckpt_old.exists():
        ckpt_path = ckpt_old
        is_multi = False
    else:
        raise FileNotFoundError(
            f"Missing checkpoint.\nTried:\n  {ckpt_mt}\n  {ckpt_old}\n"
            "Train first: python -m training.train_price_regressor ..."
        )

    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    ckpt_features = ckpt["feature_cols"]

    # Build correct model
    if is_multi:
        model = PriceRegressorMulti(input_dim=len(ckpt_features)).to(DEVICE)
    else:
        model = PriceRegressor(input_dim=len(ckpt_features)).to(DEVICE)

    model.load_state_dict(ckpt["model_state"])
    model.eval()

    print(f"\nLoaded checkpoint: {ckpt_path.name}")
    print(f"Model type: {'multi-task (reg+dir)' if is_multi else 'single-head reg'}")

    # Arrays
    X = df_test_s[ckpt_features].values.astype("float32")
    close = df_test["close"].values.astype("float32")
    fut_close = df_test["close"].shift(-horizon).values.astype("float32")
    idx = df_test.index

    end = len(df_test) - horizon - 1
    if end <= (context_len - 1):
        raise ValueError(
            f"Not enough test data for context_len={context_len} and horizon={horizon}. "
            f"Test rows={len(df_test)}, need at least {context_len + horizon + 1}."
        )

    if n_points is None or n_points <= 0:
        start = context_len - 1
    else:
        start = max(context_len - 1, end - n_points)

    times: list[pd.Timestamp] = []
    pred_prices: list[float] = []
    act_prices: list[float] = []
    base_prices: list[float] = []
    prob_up_list: list[float] = []

    for k in range(start, end):
        xwin = X[k - context_len + 1 : k + 1]
        base = float(close[k])
        actual = fut_close[k]
        if np.isnan(actual):
            continue

        xb = torch.from_numpy(xwin).unsqueeze(0).to(DEVICE)

        if is_multi:
            yhat_logret, yhat_dir_logit = model(xb)
            prob_up = float(torch.sigmoid(yhat_dir_logit).item())

            # Optional sign enforcement (only meaningful for multi-task)
            if enforce_sign:
                conf = max(prob_up, 1 - prob_up)
                y = float(yhat_logret.item())
                if conf >= enforce_conf:
                    if prob_up >= 0.5:
                        y = abs(y)
                    else:
                        y = -abs(y)
                yhat = y
            else:
                yhat = float(yhat_logret.item())

            prob_up_list.append(prob_up)
        else:
            yhat = float(model(xb).item())

        pred = float(base * np.exp(yhat))

        times.append(idx[k] + pd.Timedelta(hours=horizon))
        pred_prices.append(pred)
        act_prices.append(float(actual))
        base_prices.append(base)

    pred_prices_np = np.array(pred_prices, dtype=float)
    act_prices_np = np.array(act_prices, dtype=float)
    base_prices_np = np.array(base_prices, dtype=float)
    prob_up_np = np.array(prob_up_list, dtype=float) if prob_up_list else None

    if pred_prices_np.size == 0:
        raise RuntimeError("No points generated for plot/metrics (check horizon/context and NaNs).")

    # Model metrics
    model_m = regression_metrics(pred_prices_np, act_prices_np)
    model_dir = directional_accuracy(pred_prices_np, base_prices_np, act_prices_np)

    # Baseline: no-change
    baseline_pred = base_prices_np.copy()
    base_m = regression_metrics(baseline_pred, act_prices_np)
    base_dir = directional_accuracy(baseline_pred, base_prices_np, act_prices_np)

    print(
        f"\nMODEL | MAE=${model_m['MAE']:.6f} RMSE=${model_m['RMSE']:.6f} "
        f"MAPE={model_m['MAPE_pct']:.2f}% sMAPE={model_m['sMAPE_pct']:.2f}% R2={model_m['R2']:.3f} "
        f"DIR_ACC={model_dir:.2f}% | points={len(pred_prices_np)}"
    )
    print(
        f"BASELINE(no-change) | MAE=${base_m['MAE']:.6f} RMSE=${base_m['RMSE']:.6f} "
        f"MAPE={base_m['MAPE_pct']:.2f}% sMAPE={base_m['sMAPE_pct']:.2f}% R2={base_m['R2']:.3f} "
        f"DIR_ACC={base_dir:.2f}%"
    )
    print(
        f"IMPROVEMENT | ΔMAE=${(base_m['MAE'] - model_m['MAE']):+.6f} "
        f"ΔRMSE=${(base_m['RMSE'] - model_m['RMSE']):+.6f} "
        f"ΔDIR_ACC={(model_dir - base_dir):+.2f}%"
    )

    # Plot price
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(times, act_prices_np, label="Actual (t + horizon)")
    ax.plot(times, pred_prices_np, label="Predicted", linestyle="--")
    ax.set_title(f"Predicted vs Actual XRP Price (horizon={horizon}h) — Test Set")
    ax.set_xlabel("Time")
    ax.set_ylabel("Price")
    ax.grid(True, alpha=0.3)

    # Optional: plot direction probability on right axis (multi-task only)
    if show_dir and is_multi and prob_up_np is not None and len(prob_up_np) == len(times):
        ax2 = ax.twinx()
        ax2.plot(times, prob_up_np, linestyle=":", label="P(UP) (dir head)")
        ax2.set_ylabel("P(UP)")
        ax2.set_ylim(-0.05, 1.05)

        # merge legends
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="best")
    else:
        ax.legend(loc="best")

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
    p.add_argument(
        "--n_points",
        "-n",
        type=int,
        default=-1,
        help="How many points to plot from the end. Use -1 for full test set.",
    )
    p.add_argument("--show_dir", action="store_true", help="If multi-task ckpt, plot P(UP) as well")
    p.add_argument("--enforce_sign", action="store_true", help="If multi-task ckpt, enforce sign when confident")
    p.add_argument("--enforce_conf", type=float, default=0.70, help="Confidence for sign enforcement")
    args = p.parse_args()

    main(
        timeframe=args.timeframe,
        horizon=args.horizon,
        context_len=args.context,
        n_points=args.n_points,
        show_dir=args.show_dir,
        enforce_sign=args.enforce_sign,
        enforce_conf=args.enforce_conf,
    )