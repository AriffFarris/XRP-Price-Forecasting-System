"""
Diagnostic script for the Reversal Detector model.

Run this to understand your reversal detector's behavior and verify performance.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

# Add project root to path for module imports
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Import the model class from training script
from training.train_reversal_detector import ReversalDetector, create_reversal_features

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "saved_models"


def load_reversal_model():
    """Load the trained reversal detector model."""
    ckpt_path = MODELS_DIR / "reversal_detector.pt"
    
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Reversal detector not found at {ckpt_path}\n"
            "Run: python -m training.train_reversal_detector"
        )
    
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    
    model = ReversalDetector(input_dim=len(ckpt["feature_cols"]))
    model.load_state_dict(ckpt["model_state"])
    model.to(DEVICE)
    model.eval()
    
    return model, ckpt


def diagnose_reversal_model():
    """Run comprehensive diagnostics on the reversal detector."""
    
    print("=" * 70)
    print("REVERSAL DETECTOR DIAGNOSTICS")
    print("=" * 70)
    
    # Load model
    print("\n1. Loading model...")
    try:
        model, ckpt = load_reversal_model()
        print(f"   ✓ Model loaded successfully")
        print(f"   - Features: {len(ckpt['feature_cols'])}")
        print(f"   - Context length: {ckpt['context_len']}")
        print(f"   - Horizon: {ckpt['horizon']}h")
        print(f"   - Best threshold: {ckpt['best_threshold']}")
        print(f"   - Saved test accuracy: {ckpt['test_accuracy']:.1%}")
    except Exception as e:
        print(f"   ❌ Failed to load model: {e}")
        return
    
    # Load and prepare data
    print("\n2. Loading data...")
    df_raw = pd.read_parquet(DATA_DIR / "xrp_features_1h.parquet")
    print(f"   ✓ Loaded {len(df_raw):,} samples")
    
    # Recreate features
    df, feature_cols = create_reversal_features(df_raw, horizon=ckpt["horizon"])
    print(f"   ✓ Created {len(feature_cols)} features")
    
    # Split (same as training)
    n = len(df)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)
    
    df_test = df.iloc[val_end:].copy()
    print(f"   ✓ Test set: {len(df_test):,} samples")
    
    # Scale features
    scaler = ckpt["scaler"]
    df_test[feature_cols] = scaler.transform(df_test[feature_cols])
    
    # Get predictions
    print("\n3. Running predictions on test set...")
    context_len = ckpt["context_len"]
    
    all_probs = []
    all_reversal_true = []
    all_direction_true = []
    all_curr_dir = []
    
    model.eval()
    with torch.no_grad():
        for i in range(len(df_test) - context_len):
            # Get context window
            ctx = df_test[feature_cols].iloc[i:i + context_len].values.astype("float32")
            seq = torch.from_numpy(ctx).unsqueeze(0).to(DEVICE)
            
            # Predict
            logit = model(seq)
            prob = torch.sigmoid(logit).item()
            
            # Get true values
            idx = i + context_len
            reversal_true = df_test["target_reversal"].iloc[idx]
            direction_true = df_test["target_direction"].iloc[idx]
            curr_dir = df_test["curr_dir"].iloc[idx]
            
            all_probs.append(prob)
            all_reversal_true.append(reversal_true)
            all_direction_true.append(direction_true)
            all_curr_dir.append(curr_dir)
    
    all_probs = np.array(all_probs)
    all_reversal_true = np.array(all_reversal_true)
    all_direction_true = np.array(all_direction_true)
    all_curr_dir = np.array(all_curr_dir)
    
    print(f"   ✓ Generated {len(all_probs):,} predictions")
    
    # Analyze reversal predictions
    print("\n" + "=" * 70)
    print("4. REVERSAL PREDICTION ANALYSIS")
    print("=" * 70)
    
    threshold = ckpt["best_threshold"]
    reversal_pred = (all_probs > threshold).astype(int)
    reversal_acc = (reversal_pred == all_reversal_true).mean()
    
    print(f"\n   Threshold: {threshold}")
    print(f"   Reversal accuracy: {reversal_acc:.1%}")
    
    # Confusion matrix for reversals
    cm = confusion_matrix(all_reversal_true, reversal_pred)
    print(f"\n   Confusion Matrix (Reversal):")
    print(f"                  Predicted")
    print(f"                  Continue  Reverse")
    print(f"   Actual Continue    {cm[0,0]:5d}    {cm[0,1]:5d}")
    print(f"   Actual Reverse     {cm[1,0]:5d}    {cm[1,1]:5d}")
    
    # Precision/Recall
    if cm[0,0] + cm[1,0] > 0:
        precision_continue = cm[0,0] / (cm[0,0] + cm[1,0])
    else:
        precision_continue = 0
    if cm[1,1] + cm[0,1] > 0:
        precision_reverse = cm[1,1] / (cm[1,1] + cm[0,1])
    else:
        precision_reverse = 0
    
    print(f"\n   Precision (Continue): {precision_continue:.1%}")
    print(f"   Precision (Reverse):  {precision_reverse:.1%}")
    
    # Convert to direction predictions
    print("\n" + "=" * 70)
    print("5. DIRECTION PREDICTION ANALYSIS")
    print("=" * 70)
    
    # Convert reversal to direction
    direction_pred = np.where(
        reversal_pred == 1,
        1 - all_curr_dir,  # Flip if reversal
        all_curr_dir,      # Keep if continue
    )
    
    direction_acc = (direction_pred == all_direction_true).mean()
    persistence_acc = (all_curr_dir == all_direction_true).mean()
    
    print(f"\n   Persistence baseline: {persistence_acc:.1%}")
    print(f"   Your model:           {direction_acc:.1%}")
    print(f"   Improvement:          {(direction_acc - persistence_acc)*100:+.1f}%")
    
    # Direction confusion matrix
    cm_dir = confusion_matrix(all_direction_true, direction_pred)
    print(f"\n   Confusion Matrix (Direction):")
    print(f"                  Predicted")
    print(f"                  DOWN      UP")
    print(f"   Actual DOWN    {cm_dir[0,0]:5d}    {cm_dir[0,1]:5d}")
    print(f"   Actual UP      {cm_dir[1,0]:5d}    {cm_dir[1,1]:5d}")
    
    # Analyze prediction confidence
    print("\n" + "=" * 70)
    print("6. CONFIDENCE ANALYSIS")
    print("=" * 70)
    
    print(f"\n   Prediction probability distribution:")
    print(f"   - Mean:   {all_probs.mean():.3f}")
    print(f"   - Std:    {all_probs.std():.3f}")
    print(f"   - Min:    {all_probs.min():.3f}")
    print(f"   - Max:    {all_probs.max():.3f}")
    
    # Accuracy by confidence level
    print(f"\n   Accuracy by confidence level:")
    
    confidence_thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    for conf in confidence_thresholds:
        high_conf = (all_probs > conf) | (all_probs < (1 - conf))
        if high_conf.sum() > 10:
            conf_dir_pred = np.where(
                all_probs[high_conf] > 0.5,
                1 - all_curr_dir[high_conf],
                all_curr_dir[high_conf],
            )
            conf_acc = (conf_dir_pred == all_direction_true[high_conf]).mean()
            pct = high_conf.mean() * 100
            print(f"   - Confidence >{conf:.0%}: {conf_acc:.1%} accuracy on {pct:.0f}% of samples")
    
    # Plot diagnostics
    print("\n" + "=" * 70)
    print("7. GENERATING PLOTS")
    print("=" * 70)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Probability distribution
    axes[0, 0].hist(all_probs, bins=50, edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(threshold, color='red', linestyle='--', label=f'Threshold: {threshold}')
    axes[0, 0].axvline(0.5, color='green', linestyle='--', alpha=0.5, label='0.5')
    axes[0, 0].set_xlabel('Reversal Probability')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distribution of Reversal Probabilities')
    axes[0, 0].legend()
    
    # 2. Accuracy over time (rolling)
    window = 100
    rolling_acc = pd.Series(direction_pred == all_direction_true).rolling(window).mean()
    axes[0, 1].plot(rolling_acc.values, label=f'Model ({window}-sample rolling)')
    axes[0, 1].axhline(persistence_acc, color='red', linestyle='--', label=f'Persistence: {persistence_acc:.1%}')
    axes[0, 1].axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='Random: 50%')
    axes[0, 1].set_xlabel('Sample')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Rolling Accuracy Over Time')
    axes[0, 1].legend()
    axes[0, 1].set_ylim(0.3, 1.0)
    
    # 3. Calibration plot
    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    calibration_acc = []
    calibration_count = []
    for i in range(n_bins):
        mask = (all_probs >= bin_edges[i]) & (all_probs < bin_edges[i+1])
        if mask.sum() > 0:
            calibration_acc.append(all_reversal_true[mask].mean())
            calibration_count.append(mask.sum())
        else:
            calibration_acc.append(np.nan)
            calibration_count.append(0)
    
    axes[1, 0].plot(bin_centers, calibration_acc, 'bo-', label='Model')
    axes[1, 0].plot([0, 1], [0, 1], 'r--', label='Perfect calibration')
    axes[1, 0].set_xlabel('Predicted Probability')
    axes[1, 0].set_ylabel('Actual Reversal Rate')
    axes[1, 0].set_title('Calibration Plot')
    axes[1, 0].legend()
    axes[1, 0].set_xlim(0, 1)
    axes[1, 0].set_ylim(0, 1)
    
    # 4. Accuracy vs threshold
    thresholds = np.linspace(0.3, 0.7, 21)
    accuracies = []
    for t in thresholds:
        rev_pred = (all_probs > t).astype(int)
        dir_pred = np.where(rev_pred == 1, 1 - all_curr_dir, all_curr_dir)
        acc = (dir_pred == all_direction_true).mean()
        accuracies.append(acc)
    
    axes[1, 1].plot(thresholds, accuracies, 'b-', linewidth=2)
    axes[1, 1].axvline(threshold, color='red', linestyle='--', label=f'Current: {threshold}')
    axes[1, 1].axhline(persistence_acc, color='green', linestyle='--', label=f'Baseline: {persistence_acc:.1%}')
    axes[1, 1].set_xlabel('Reversal Threshold')
    axes[1, 1].set_ylabel('Direction Accuracy')
    axes[1, 1].set_title('Accuracy vs Threshold')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(ROOT / "diagnostics_reversal_detector.png", dpi=150)
    print(f"   ✓ Saved plot to: {ROOT / 'diagnostics_reversal_detector.png'}")
    plt.show()
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    print(f"""
    Model Performance:
    ├── Reversal accuracy:     {reversal_acc:.1%}
    ├── Direction accuracy:    {direction_acc:.1%}
    ├── Persistence baseline:  {persistence_acc:.1%}
    └── Improvement:           {(direction_acc - persistence_acc)*100:+.1f}%
    
    Recommendation:
    """)
    
    if direction_acc >= 0.60:
        print("    ✅ Model is performing well (≥60% accuracy)")
        print("    Consider paper trading to validate on live data.")
    elif direction_acc > persistence_acc + 0.02:
        print("    🟡 Model beats baseline but below 60%")
        print("    Try: different horizon, more data, or ensemble approach.")
    else:
        print("    ❌ Model doesn't significantly beat baseline")
        print("    The market may be too efficient at this timeframe.")


if __name__ == "__main__":
    diagnose_reversal_model()