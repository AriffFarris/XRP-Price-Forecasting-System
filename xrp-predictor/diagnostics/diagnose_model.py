"""
Diagnostic script to identify why forecasts may be biased.

Run this to understand your model's behavior and data characteristics.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from scipy import stats

# Add project root to path for module imports
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.gru_model import GRUPredictor


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "saved_models"


def load_model_and_data():
    """Load everything we need for diagnostics."""
    # Load model
    ckpt = torch.load(MODELS_DIR / "gru_xrp.pt", map_location=DEVICE, weights_only=False)
    model = GRUPredictor(input_dim=len(ckpt["feature_cols"]))
    model.load_state_dict(ckpt["model_state"])
    model.to(DEVICE)
    model.eval()
    
    # Load data
    df = pd.read_parquet(DATA_DIR / "xrp_features_1h.parquet")
    
    return model, ckpt, df


def diagnose_target_distribution(df: pd.DataFrame, target_col: str = "log_ret_4h"):
    """
    Check if the target variable is biased toward positive or negative.
    """
    print("=" * 60)
    print("1. TARGET DISTRIBUTION ANALYSIS")
    print("=" * 60)
    
    target = df[target_col].dropna()
    
    print(f"\nTarget column: {target_col}")
    print(f"Number of samples: {len(target)}")
    print(f"\nStatistics:")
    print(f"  Mean:     {target.mean():.6f}")
    print(f"  Median:   {target.median():.6f}")
    print(f"  Std:      {target.std():.6f}")
    print(f"  Skewness: {stats.skew(target):.4f}")
    print(f"  Kurtosis: {stats.kurtosis(target):.4f}")
    
    # Count positive vs negative
    n_positive = (target > 0).sum()
    n_negative = (target < 0).sum()
    n_zero = (target == 0).sum()
    
    print(f"\nDirection breakdown:")
    print(f"  Positive: {n_positive} ({100*n_positive/len(target):.1f}%)")
    print(f"  Negative: {n_negative} ({100*n_negative/len(target):.1f}%)")
    print(f"  Zero:     {n_zero} ({100*n_zero/len(target):.1f}%)")
    
    # Statistical test for mean != 0
    t_stat, p_value = stats.ttest_1samp(target, 0)
    print(f"\nT-test (H0: mean = 0):")
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  p-value:     {p_value:.4f}")
    
    if p_value < 0.05:
        bias_direction = "BULLISH" if target.mean() > 0 else "BEARISH"
        print(f"  ⚠️  Significant bias detected: {bias_direction}")
        print(f"     Your training data has a {bias_direction.lower()} tilt!")
    else:
        print(f"  ✓ No significant directional bias in training data")
    
    # Plot distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].hist(target, bins=100, edgecolor='black', alpha=0.7)
    axes[0].axvline(target.mean(), color='red', linestyle='--', label=f'Mean: {target.mean():.4f}')
    axes[0].axvline(0, color='green', linestyle='-', label='Zero')
    axes[0].set_xlabel('4-hour log return')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Distribution of Target Variable')
    axes[0].legend()
    
    # Q-Q plot
    stats.probplot(target, dist="norm", plot=axes[1])
    axes[1].set_title('Q-Q Plot (vs Normal)')
    
    plt.tight_layout()
    plt.savefig(ROOT / "diagnostics_target_distribution.png", dpi=150)
    plt.show()


def diagnose_train_val_test_split(df: pd.DataFrame, target_col: str = "log_ret_4h"):
    """
    Check if train/val/test have different characteristics (data drift).
    """
    print("\n" + "=" * 60)
    print("2. TRAIN/VAL/TEST DISTRIBUTION ANALYSIS")
    print("=" * 60)
    
    n = len(df)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)
    
    train = df.iloc[:train_end][target_col]
    val = df.iloc[train_end:val_end][target_col]
    test = df.iloc[val_end:][target_col]
    
    print(f"\nSplit sizes: Train={len(train)}, Val={len(val)}, Test={len(test)}")
    
    for name, data in [("Train", train), ("Val", val), ("Test", test)]:
        print(f"\n{name}:")
        print(f"  Mean:   {data.mean():.6f}")
        print(f"  Std:    {data.std():.6f}")
        print(f"  % Positive: {100*(data > 0).mean():.1f}%")
    
    # Check if distributions are different
    ks_train_val = stats.ks_2samp(train, val)
    ks_train_test = stats.ks_2samp(train, test)
    
    print(f"\nKolmogorov-Smirnov tests:")
    print(f"  Train vs Val:  p={ks_train_val.pvalue:.4f}")
    print(f"  Train vs Test: p={ks_train_test.pvalue:.4f}")
    
    if ks_train_test.pvalue < 0.05:
        print(f"  ⚠️  Train and Test distributions are significantly different!")
        print(f"     This suggests data drift - model may not generalize well.")
    
    # Plot time series of target
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    # Rolling mean of target
    rolling_mean = df[target_col].rolling(window=168).mean()  # 1 week rolling
    
    axes[0].plot(df.index, rolling_mean, label='7-day rolling mean of target')
    axes[0].axhline(0, color='black', linestyle='--', alpha=0.5)
    axes[0].axvline(df.index[train_end], color='red', linestyle='--', label='Train/Val split')
    axes[0].axvline(df.index[val_end], color='orange', linestyle='--', label='Val/Test split')
    axes[0].set_ylabel('Rolling mean of log_ret_4h')
    axes[0].set_title('Target Variable Over Time (shows market regime changes)')
    axes[0].legend()
    
    # Price for context
    if 'close' in df.columns:
        axes[1].plot(df.index, df['close'], label='XRP price')
        axes[1].axvline(df.index[train_end], color='red', linestyle='--')
        axes[1].axvline(df.index[val_end], color='orange', linestyle='--')
        axes[1].set_ylabel('Price')
        axes[1].set_title('XRP Price (for context)')
        axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(ROOT / "diagnostics_split_analysis.png", dpi=150)
    plt.show()


def diagnose_model_predictions(model, ckpt, df: pd.DataFrame):
    """
    Analyze what the model predicts on different inputs.
    """
    print("\n" + "=" * 60)
    print("3. MODEL PREDICTION ANALYSIS")
    print("=" * 60)
    
    feature_cols = ckpt["feature_cols"]
    scaler = ckpt["scaler"]
    context_len = ckpt["context_len"]
    target_col = ckpt["target_col"]
    
    # Get predictions on last 500 samples
    n_test = min(500, len(df) - context_len)
    
    all_preds = []
    all_true = []
    
    for i in range(len(df) - context_len - n_test, len(df) - context_len):
        ctx = df[feature_cols].iloc[i:i+context_len].values.astype("float32")
        ctx_scaled = scaler.transform(ctx)
        
        seq = torch.from_numpy(ctx_scaled).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            pred = model(seq).item()
        
        true = df[target_col].iloc[i + context_len]
        
        all_preds.append(pred)
        all_true.append(true)
    
    all_preds = np.array(all_preds)
    all_true = np.array(all_true)
    
    print(f"\nPrediction statistics (last {n_test} samples):")
    print(f"  Mean prediction: {all_preds.mean():.6f}")
    print(f"  Std prediction:  {all_preds.std():.6f}")
    print(f"  Min prediction:  {all_preds.min():.6f}")
    print(f"  Max prediction:  {all_preds.max():.6f}")
    
    print(f"\nActual statistics:")
    print(f"  Mean actual:     {all_true.mean():.6f}")
    print(f"  Std actual:      {all_true.std():.6f}")
    
    # Check prediction diversity
    unique_bins = len(np.unique(np.round(all_preds, 5)))
    print(f"\nPrediction diversity:")
    print(f"  Unique prediction values (5 decimal places): {unique_bins}")
    
    if unique_bins < 20:
        print(f"  ⚠️  Very low prediction diversity - model may be collapsing!")
    
    # Direction accuracy
    sign_match = (np.sign(all_preds) == np.sign(all_true)).mean()
    print(f"\nDirectional accuracy: {100*sign_match:.1f}%")
    print(f"Random baseline would be ~50%")
    
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Predictions vs actual scatter
    axes[0, 0].scatter(all_true, all_preds, alpha=0.3, s=10)
    lims = [min(all_true.min(), all_preds.min()), max(all_true.max(), all_preds.max())]
    axes[0, 0].plot(lims, lims, 'r--', label='Perfect prediction')
    axes[0, 0].set_xlabel('Actual')
    axes[0, 0].set_ylabel('Predicted')
    axes[0, 0].set_title('Predictions vs Actual')
    axes[0, 0].legend()
    
    # Prediction histogram
    axes[0, 1].hist(all_preds, bins=50, alpha=0.7, label='Predictions', edgecolor='black')
    axes[0, 1].hist(all_true, bins=50, alpha=0.5, label='Actual', edgecolor='black')
    axes[0, 1].set_xlabel('Value')
    axes[0, 1].set_title('Distribution of Predictions vs Actual')
    axes[0, 1].legend()
    
    # Residuals
    residuals = all_preds - all_true
    axes[1, 0].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    axes[1, 0].axvline(residuals.mean(), color='red', linestyle='--', 
                       label=f'Mean error: {residuals.mean():.5f}')
    axes[1, 0].set_xlabel('Prediction Error')
    axes[1, 0].set_title('Residual Distribution')
    axes[1, 0].legend()
    
    # Time series of predictions
    axes[1, 1].plot(all_preds, label='Predicted', alpha=0.7)
    axes[1, 1].plot(all_true, label='Actual', alpha=0.7)
    axes[1, 1].set_xlabel('Sample')
    axes[1, 1].set_ylabel('Log return')
    axes[1, 1].set_title('Predictions Over Time')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(ROOT / "diagnostics_predictions.png", dpi=150)
    plt.show()
    
    return all_preds, all_true


def diagnose_autoregressive_drift():
    """
    Show how predictions drift when using the old vs new inference method.
    """
    print("\n" + "=" * 60)
    print("4. AUTOREGRESSIVE DRIFT ANALYSIS")
    print("=" * 60)
    
    # This would compare old vs new inference
    # For now, just explain the issue
    
    print("""
The key insight is that your original inference code does this:

    seq_np[-1] = seq_np[-2]  # Copy previous row
    if target_col in feature_cols:  # THIS IS ALWAYS FALSE
        seq_np[-1, idx] = y_next

Since target_col ('log_ret_4h') is NOT in feature_cols, the features
never update. The model sees nearly identical inputs each step.

If the model has learned even a tiny positive bias (e.g., mean prediction
of +0.0001 instead of 0), this compounds:

    Step 1: predict +0.0001 → price goes up 0.01%
    Step 2: predict +0.0001 → price goes up another 0.01%
    ...
    Step 24: cumulative +0.24% → monotonic upward forecast

SOLUTIONS:
1. Use inference_fixed.py which recalculates indicators
2. Train a model that doesn't need indicators (just price history)
3. Add noise/uncertainty to break the deterministic drift
4. Ensemble multiple models to reduce individual biases
    """)


def run_all_diagnostics():
    """Run all diagnostic checks."""
    model, ckpt, df = load_model_and_data()
    target_col = ckpt["target_col"]
    
    diagnose_target_distribution(df, target_col)
    diagnose_train_val_test_split(df, target_col)
    diagnose_model_predictions(model, ckpt, df)
    diagnose_autoregressive_drift()
    
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)
    print("""
Based on typical issues with time series forecasting:

1. USE inference_fixed.py for autoregressive forecasting

2. CONSIDER training with regularization to reduce prediction magnitude:
   - Add L2 penalty
   - Use dropout more aggressively
   - Train with noise injection

3. BALANCE your training data:
   - Oversample periods of decline if data is mostly bullish
   - Or undersample bull market periods

4. ADD MEAN REVERSION signal:
   - Include distance from moving average as strong feature
   - The model should learn: "price far above MA → expect pullback"

5. USE ENSEMBLE of models trained on different time periods

6. CONSIDER shorter prediction horizons (1h instead of 4h)
   - Longer horizons are harder to predict
   - Compounding errors get worse with longer horizons
    """)


if __name__ == "__main__":
    run_all_diagnostics()
