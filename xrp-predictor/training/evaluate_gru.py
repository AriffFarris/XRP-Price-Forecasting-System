from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from training.dataset import SequenceDataset
from models.gru_model import GRUPredictor


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "saved_models"


def load_checkpoint():
    """
    Load the trained GRU checkpoint (model weights + scaler + metadata).
    We explicitly set weights_only=False because the checkpoint
    includes a scikit-learn StandardScaler object.
    """
    ckpt_path = MODELS_DIR / "gru_xrp.pt"
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)

    feature_cols = ckpt["feature_cols"]
    target_col = ckpt["target_col"]
    context_len = ckpt["context_len"]
    scaler = ckpt["scaler"]

    model = GRUPredictor(input_dim=len(feature_cols))
    model.load_state_dict(ckpt["model_state"])
    model.to(DEVICE)
    model.eval()

    return model, scaler, feature_cols, target_col, context_len


def build_test_dataset(
    timeframe: str = "1h",
    context_len: int = 24,
    feature_cols=None,
    target_col: str = "log_ret_1h",
    scaler=None,
):
    """
    Recreate the same train/val/test split as in train_gru.py,
    and return a SequenceDataset + DataLoader for the test set.
    """
    df = pd.read_parquet(DATA_DIR / f"xrp_features_{timeframe}.parquet")

    # If feature_cols not provided, infer them like in train_gru.py
    if feature_cols is None:
        ignore_cols = ["ret_1h", "ret_24h", "log_ret_1h"]
        feature_cols = [c for c in df.columns if c not in ignore_cols]

    # Time-based split: 70% train, 15% val, 15% test
    n = len(df)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)

    df_train = df.iloc[:train_end].copy()
    df_val = df.iloc[train_end:val_end].copy()
    df_test = df.iloc[val_end:].copy()

    # Use the same scaler as in training (fitted on train)
    if scaler is None:
        raise ValueError("Scaler must be provided from checkpoint.")

    for d in (df_train, df_val, df_test):
        d[feature_cols] = scaler.transform(d[feature_cols].values)

    test_ds = SequenceDataset(df_test, feature_cols, target_col, context_len)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)
    return test_loader, df_test.index, feature_cols


def safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute R², but return NaN if not well-defined (len < 2)."""
    if len(y_true) < 2:
        return float("nan")
    return r2_score(y_true, y_pred)


def evaluate_gru_on_test(timeframe: str = "1h"):
    """
    Evaluate the GRU model on the held-out test set.
    Prints regression metrics, directional accuracy,
    and compares against simple baselines.
    """
    model, scaler, feature_cols, target_col, context_len = load_checkpoint()
    test_loader, test_index, feature_cols = build_test_dataset(
        timeframe=timeframe,
        context_len=context_len,
        feature_cols=feature_cols,
        target_col=target_col,
        scaler=scaler,
    )

    all_preds = []
    all_true = []

    # ---- Run model on test set ----
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            preds = model(xb)
            all_preds.append(preds.cpu().numpy())
            all_true.append(yb.cpu().numpy())

    if len(all_true) == 0:
        print("No test examples available (sequence window too long vs test set length).")
        return

    # Concatenate list of arrays into 1D arrays
    all_preds = np.concatenate(all_preds)
    all_true = np.concatenate(all_true)

    # --- GRU regression metrics (on log returns) ---
    mse = mean_squared_error(all_true, all_preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(all_true, all_preds)
    r2 = safe_r2(all_true, all_preds)

    # --- GRU directional accuracy ---
    sign_true = np.sign(all_true)
    sign_pred = np.sign(all_preds)
    mask = sign_true != 0  # ignore truly flat moves

    if mask.sum() > 0:
        correct_dir = (sign_true[mask] == sign_pred[mask]).mean()
    else:
        correct_dir = float("nan")

    print("=== GRU Test Set Evaluation (1-step ahead log returns) ===")
    print(f"Number of test examples: {len(all_true)}")
    print()
    print(f"MSE  : {mse:.8f}")
    print(f"RMSE : {rmse:.6f}")
    print(f"MAE  : {mae:.6f}")
    print(f"R^2  : {r2:.4f}" if not np.isnan(r2) else "R^2  : NaN (not enough samples)")
    print()
    if not np.isnan(correct_dir):
        print(f"Directional accuracy (up/down, ignoring flat): {correct_dir * 100:.2f}%")
    else:
        print("Directional accuracy: not defined (no non-zero moves in test set)")

    # ============================
    # Baseline 1: always predict 0
    # ============================
    zero_preds = np.zeros_like(all_true)
    mse_zero = mean_squared_error(all_true, zero_preds)
    rmse_zero = np.sqrt(mse_zero)
    mae_zero = mean_absolute_error(all_true, zero_preds)
    r2_zero = safe_r2(all_true, zero_preds)

    print("\n=== Baseline 1: Always predict 0 (no move) ===")
    print(f"MSE  : {mse_zero:.8f}")
    print(f"RMSE : {rmse_zero:.6f}")
    print(f"MAE  : {mae_zero:.6f}")
    print(f"R^2  : {r2_zero:.4f}" if not np.isnan(r2_zero) else "R^2  : NaN (not enough samples)")

    # ==========================================
    # Baseline 2: persistence (previous log-ret)
    # ==========================================
    prev_preds = np.zeros_like(all_true)
    if len(all_true) > 1:
        # For i > 0: pred[i] = true[i-1]; for i = 0: use 0
        prev_preds[1:] = all_true[:-1]

        mse_prev = mean_squared_error(all_true, prev_preds)
        rmse_prev = np.sqrt(mse_prev)
        mae_prev = mean_absolute_error(all_true, prev_preds)
        r2_prev = safe_r2(all_true, prev_preds)

        sign_pred_prev = np.sign(prev_preds)
        if mask.sum() > 0:
            correct_dir_prev = (sign_true[mask] == sign_pred_prev[mask]).mean()
        else:
            correct_dir_prev = float("nan")

        print("\n=== Baseline 2: Predict previous log-return (persistence) ===")
        print(f"MSE  : {mse_prev:.8f}")
        print(f"RMSE : {rmse_prev:.6f}")
        print(f"MAE  : {mae_prev:.6f}")
        print(f"R^2  : {r2_prev:.4f}" if not np.isnan(r2_prev) else "R^2  : NaN (not enough samples)")
        if not np.isnan(correct_dir_prev):
            print(f"Directional accuracy (up/down, ignoring flat): {correct_dir_prev * 100:.2f}%")
        else:
            print("Directional accuracy: not defined (no non-zero moves in test set)")
    else:
        print("\n=== Baseline 2: Predict previous log-return (persistence) ===")
        print("Not enough samples to compute persistence baseline.")


if __name__ == "__main__":
    evaluate_gru_on_test()