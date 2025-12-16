"""
Live Inference for Reversal Detector

Makes predictions using your trained reversal detector model.
Outputs: Direction (UP/DOWN) + Confidence level

Usage:
    python -m deployment.inference_reversal
    
    # Or with options:
    python -m deployment.inference_reversal --refresh  # Fetch new data first
"""

from pathlib import Path
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training.train_reversal_detector import ReversalDetector, create_reversal_features

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "saved_models"


def load_model():
    """Load the trained reversal detector."""
    ckpt_path = MODELS_DIR / "reversal_detector.pt"
    
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Model not found at {ckpt_path}\n"
            "Run: python -m training.train_reversal_detector"
        )
    
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    
    model = ReversalDetector(input_dim=len(ckpt["feature_cols"]))
    model.load_state_dict(ckpt["model_state"])
    model.to(DEVICE)
    model.eval()
    
    return model, ckpt


def get_latest_data(refresh: bool = False):
    """Load the latest price data."""
    if refresh:
        print("Refreshing data...")
        try:
            from data.collectors.price_collector import PriceCollector
            collector = PriceCollector()
            collector.run(timeframe="1h", years=2)
            
            from data.processors.feature_engineering import build_feature_dataset
            build_feature_dataset(timeframe="1h")
            print("Data refreshed!\n")
        except Exception as e:
            print(f"Warning: Could not refresh data: {e}")
            print("Using existing data.\n")
    
    # Load raw data for feature creation
    raw_path = DATA_DIR / "xrp_features_1h.parquet"
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Data not found at {raw_path}\n"
            "Run: python data/collectors/price_collector.py"
        )
    
    return pd.read_parquet(raw_path)


def predict(model, ckpt, df_raw):
    """
    Make a prediction using the latest data.
    
    Returns:
        dict with prediction details
    """
    # Create features (same as training)
    df, feature_cols = create_reversal_features(df_raw, horizon=ckpt["horizon"])
    
    # Get the scaler from checkpoint
    scaler = ckpt["scaler"]
    context_len = ckpt["context_len"]
    threshold = ckpt.get("best_threshold", 0.5)
    
    # Scale features
    df_scaled = df.copy()
    df_scaled[feature_cols] = scaler.transform(df[feature_cols])
    
    # Get the last context_len rows
    if len(df_scaled) < context_len:
        raise ValueError(f"Not enough data. Need {context_len} rows, have {len(df_scaled)}")
    
    ctx = df_scaled[feature_cols].iloc[-context_len:].values.astype("float32")
    
    # Get current momentum direction
    current_price = df["close"].iloc[-1]
    prev_price = df["close"].iloc[-2]
    current_direction = "UP" if current_price > prev_price else "DOWN"
    current_ret = (current_price - prev_price) / prev_price * 100
    
    # Make prediction
    seq = torch.from_numpy(ctx).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        logit = model(seq)
        reversal_prob = torch.sigmoid(logit).item()
    
    # Convert reversal probability to direction
    will_reverse = reversal_prob > threshold
    
    if will_reverse:
        predicted_direction = "DOWN" if current_direction == "UP" else "UP"
        confidence = reversal_prob
        reasoning = f"Momentum likely to REVERSE (prob: {reversal_prob:.1%})"
    else:
        predicted_direction = current_direction
        confidence = 1 - reversal_prob
        reasoning = f"Momentum likely to CONTINUE (prob: {1-reversal_prob:.1%})"
    
    # Get timestamp
    last_timestamp = df.index[-1]
    prediction_time = last_timestamp + pd.Timedelta(hours=ckpt["horizon"])
    
    return {
        "current_price": current_price,
        "current_direction": current_direction,
        "current_return": current_ret,
        "predicted_direction": predicted_direction,
        "confidence": confidence,
        "reversal_probability": reversal_prob,
        "will_reverse": will_reverse,
        "reasoning": reasoning,
        "horizon_hours": ckpt["horizon"],
        "data_timestamp": last_timestamp,
        "prediction_for": prediction_time,
        "threshold": threshold,
    }


def format_prediction(pred: dict) -> str:
    """Format prediction for display."""
    
    # Direction arrow and color hint
    if pred["predicted_direction"] == "UP":
        arrow = "⬆️"
        direction_str = "UP"
    else:
        arrow = "⬇️"
        direction_str = "DOWN"
    
    # Confidence level description
    conf = pred["confidence"]
    if conf >= 0.9:
        conf_level = "VERY HIGH"
    elif conf >= 0.8:
        conf_level = "HIGH"
    elif conf >= 0.7:
        conf_level = "MODERATE"
    elif conf >= 0.6:
        conf_level = "LOW"
    else:
        conf_level = "VERY LOW"
    
    # Current momentum
    curr_arrow = "⬆️" if pred["current_direction"] == "UP" else "⬇️"
    
    output = f"""
{'='*50}
XRP PRICE PREDICTION
{'='*50}

📊 CURRENT STATE
   Price:          ${pred['current_price']:.4f}
   Momentum:       {pred['current_direction']} {curr_arrow} ({pred['current_return']:+.2f}%)
   Data as of:     {pred['data_timestamp']}

🎯 PREDICTION (Next {pred['horizon_hours']} hours)
   Direction:      {direction_str} {arrow}
   Confidence:     {pred['confidence']:.1%} ({conf_level})
   
   {pred['reasoning']}

📈 PREDICTION DETAILS
   Reversal Prob:  {pred['reversal_probability']:.1%}
   Threshold:      {pred['threshold']}
   Will Reverse:   {'Yes' if pred['will_reverse'] else 'No'}
   Valid until:    {pred['prediction_for']}

{'='*50}
"""
    
    # Add trading suggestion based on confidence
    if conf >= 0.8:
        if pred["predicted_direction"] == "UP":
            output += "\n💡 SUGGESTION: Strong bullish signal. Consider LONG position.\n"
        else:
            output += "\n💡 SUGGESTION: Strong bearish signal. Consider SHORT position.\n"
    elif conf >= 0.6:
        output += "\n💡 SUGGESTION: Moderate signal. Use caution, consider smaller position.\n"
    else:
        output += "\n💡 SUGGESTION: Low confidence. Consider waiting for clearer signal.\n"
    
    output += "\n⚠️  DISCLAIMER: This is not financial advice. Always do your own research.\n"
    
    return output


def run_inference(refresh: bool = False, json_output: bool = False):
    """Main inference function."""
    
    print(f"\n🔄 Loading model...")
    model, ckpt = load_model()
    print(f"   ✓ Model loaded (accuracy: {ckpt['test_accuracy']:.1%})")
    
    print(f"\n📊 Loading data...")
    df_raw = get_latest_data(refresh=refresh)
    print(f"   ✓ Loaded {len(df_raw):,} samples")
    print(f"   ✓ Latest: {df_raw.index[-1]}")
    
    print(f"\n🎯 Making prediction...")
    prediction = predict(model, ckpt, df_raw)
    
    if json_output:
        import json
        # Convert timestamps to strings for JSON
        pred_json = prediction.copy()
        pred_json["data_timestamp"] = str(pred_json["data_timestamp"])
        pred_json["prediction_for"] = str(pred_json["prediction_for"])
        print(json.dumps(pred_json, indent=2))
    else:
        print(format_prediction(prediction))
    
    return prediction


def predict_multiple_horizons(refresh: bool = False):
    """Make predictions for multiple time horizons."""
    
    print(f"\n🔄 Loading model...")
    model, ckpt = load_model()
    
    print(f"\n📊 Loading data...")
    df_raw = get_latest_data(refresh=refresh)
    
    # Note: This uses the same model trained for 4h horizon
    # For true multi-horizon, you'd need models trained for each horizon
    prediction = predict(model, ckpt, df_raw)
    
    print(f"""
{'='*50}
XRP MULTI-TIMEFRAME ANALYSIS
{'='*50}

Current Price: ${prediction['current_price']:.4f}
Current Momentum: {prediction['current_direction']}

Prediction ({ckpt['horizon']}h horizon):
  Direction:  {prediction['predicted_direction']}
  Confidence: {prediction['confidence']:.1%}

Note: For multiple horizons, train separate models with --horizon 1, 2, 4, etc.
{'='*50}
""")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="XRP Price Prediction")
    parser.add_argument("--refresh", "-r", action="store_true", 
                        help="Refresh data before predicting")
    parser.add_argument("--json", "-j", action="store_true",
                        help="Output as JSON")
    parser.add_argument("--multi", "-m", action="store_true",
                        help="Multi-timeframe analysis")
    args = parser.parse_args()
    
    if args.multi:
        predict_multiple_horizons(refresh=args.refresh)
    else:
        run_inference(refresh=args.refresh, json_output=args.json)