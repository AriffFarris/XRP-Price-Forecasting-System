"""
Realistic Weekly Forecast for XRP

Fixes from previous version:
1. Uses actual historical move distribution (not inflated)
2. Adds mean reversion pressure (price can't go up forever)
3. Direction-only predictions (magnitude from historical data)
4. Growing uncertainty bands
5. Clear reliability warnings

Usage:
    python -m deployment.forecast_week
"""

from pathlib import Path
import sys
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

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
        raise FileNotFoundError(f"Model not found at {ckpt_path}")
    
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    
    model = ReversalDetector(input_dim=len(ckpt["feature_cols"]))
    model.load_state_dict(ckpt["model_state"])
    model.to(DEVICE)
    model.eval()
    
    return model, ckpt


def get_realistic_move(
    direction: str,
    historical_returns: np.ndarray,
    cumulative_change: float,
    mean_reversion_strength: float = 0.3,
) -> float:
    """
    Get a realistic price move based on:
    1. Historical return distribution (sample from actual moves)
    2. Mean reversion pressure (large cumulative moves get pulled back)
    
    Args:
        direction: "UP" or "DOWN"
        historical_returns: Array of historical 4h returns
        cumulative_change: How much price has already moved (as ratio, e.g., 0.05 = 5%)
        mean_reversion_strength: How strongly to apply mean reversion (0-1)
    """
    # Separate positive and negative returns
    pos_returns = historical_returns[historical_returns > 0]
    neg_returns = historical_returns[historical_returns < 0]
    
    if direction == "UP":
        if len(pos_returns) > 0:
            # Sample from actual positive returns
            base_move = np.random.choice(pos_returns)
        else:
            base_move = 0.005  # Default 0.5%
    else:
        if len(neg_returns) > 0:
            # Sample from actual negative returns
            base_move = np.random.choice(neg_returns)
        else:
            base_move = -0.005  # Default -0.5%
    
    # Apply mean reversion pressure
    # If price has gone up a lot, reduce UP moves and increase DOWN moves
    if cumulative_change > 0.05:  # More than 5% up
        reversion_factor = 1 - (cumulative_change * mean_reversion_strength)
        if direction == "UP":
            base_move *= max(0.3, reversion_factor)  # Reduce UP moves
        else:
            base_move *= min(1.5, 2 - reversion_factor)  # Increase DOWN moves
    elif cumulative_change < -0.05:  # More than 5% down
        reversion_factor = 1 + (abs(cumulative_change) * mean_reversion_strength)
        if direction == "DOWN":
            base_move *= max(0.3, 2 - reversion_factor)  # Reduce DOWN moves
        else:
            base_move *= min(1.5, reversion_factor)  # Increase UP moves
    
    # Cap individual moves at ±3% (realistic for 4h)
    base_move = np.clip(base_move, -0.03, 0.03)
    
    return base_move


def generate_forecast(
    model,
    ckpt: dict,
    df_raw: pd.DataFrame,
    days: int = 7,
    history_hours: int = 72,
    n_simulations: int = 50,  # Monte Carlo simulations for uncertainty
) -> Tuple[pd.DataFrame, List[Dict], pd.DataFrame]:
    """
    Generate realistic forecast with uncertainty bands.
    
    Uses Monte Carlo simulation to show range of possible outcomes.
    """
    horizon = ckpt["horizon"]
    context_len = ckpt["context_len"]
    scaler = ckpt["scaler"]
    feature_cols = ckpt["feature_cols"]
    threshold = ckpt.get("best_threshold", 0.5)
    
    n_steps = (days * 24) // horizon
    
    print(f"\n📈 Generating {days}-day forecast ({n_steps} steps of {horizon}h)")
    print(f"   Running {n_simulations} Monte Carlo simulations...")
    
    # Get historical returns for realistic move sampling
    historical_returns = df_raw["close"].pct_change(horizon).dropna().values
    print(f"   Historical {horizon}h return: mean={historical_returns.mean()*100:.2f}%, std={historical_returns.std()*100:.2f}%")
    
    # Store all simulation paths
    all_paths = []
    all_directions = []  # Store predicted directions
    
    min_history = max(200, context_len) + 50
    
    for sim in range(n_simulations):
        df_working = df_raw.iloc[-min_history:].copy()
        
        path_prices = [df_working["close"].iloc[-1]]
        path_directions = []
        cumulative_change = 0.0
        
        for step in range(n_steps):
            # Create features
            df_feat, _ = create_reversal_features(df_working.copy(), horizon=horizon)
            
            for col in feature_cols:
                if col not in df_feat.columns:
                    df_feat[col] = 0.0
            
            # Scale
            df_scaled = df_feat.copy()
            df_scaled[feature_cols] = scaler.transform(df_feat[feature_cols])
            
            # Get context
            ctx = df_scaled[feature_cols].iloc[-context_len:].values.astype("float32")
            ctx = np.nan_to_num(ctx, nan=0.0)
            
            # Predict
            seq = torch.from_numpy(ctx).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                reversal_prob = torch.sigmoid(model(seq)).item()
            
            # Determine direction
            current_price = df_working["close"].iloc[-1]
            prev_price = df_working["close"].iloc[-2]
            current_dir = "UP" if current_price > prev_price else "DOWN"
            
            will_reverse = reversal_prob > threshold
            pred_direction = ("DOWN" if current_dir == "UP" else "UP") if will_reverse else current_dir
            
            path_directions.append(pred_direction)
            
            # Get realistic move
            move = get_realistic_move(
                direction=pred_direction,
                historical_returns=historical_returns,
                cumulative_change=cumulative_change,
                mean_reversion_strength=0.4,
            )
            
            # Add small random noise for simulation variety
            noise = np.random.normal(0, historical_returns.std() * 0.3)
            move += noise
            
            # Update price
            new_price = current_price * (1 + move)
            cumulative_change = (new_price / path_prices[0]) - 1
            
            path_prices.append(new_price)
            
            # Create synthetic candle
            last_timestamp = df_working.index[-1]
            new_timestamp = last_timestamp + pd.Timedelta(hours=horizon)
            
            new_row = pd.DataFrame({
                "open": [current_price],
                "high": [max(current_price, new_price) * 1.001],
                "low": [min(current_price, new_price) * 0.999],
                "close": [new_price],
                "volume": [df_working["volume"].iloc[-20:].mean()],
            }, index=[new_timestamp])
            
            # Copy other columns
            for col in df_working.columns:
                if col not in new_row.columns and col.startswith(("btc_", "eth_", "BTC", "ETH")):
                    new_row[col] = df_working[col].iloc[-1]
            
            df_working = pd.concat([df_working, new_row])
        
        all_paths.append(path_prices)
        all_directions.append(path_directions)
        
        if (sim + 1) % 10 == 0:
            print(f"   Completed {sim + 1}/{n_simulations} simulations")
    
    # Convert to array for statistics
    all_paths = np.array(all_paths)
    
    # Calculate statistics across simulations
    median_path = np.median(all_paths, axis=0)
    percentile_10 = np.percentile(all_paths, 10, axis=0)
    percentile_90 = np.percentile(all_paths, 90, axis=0)
    percentile_25 = np.percentile(all_paths, 25, axis=0)
    percentile_75 = np.percentile(all_paths, 75, axis=0)
    
    # Build timestamps
    last_timestamp = df_raw.index[-1]
    future_timestamps = [last_timestamp + pd.Timedelta(hours=horizon * (i + 1)) for i in range(n_steps)]
    
    # Historical data
    hist_df = pd.DataFrame({
        "price": df_raw["close"].iloc[-history_hours:].values,
        "type": "historical",
    }, index=df_raw.index[-history_hours:])
    
    # Forecast data (median + bands)
    pred_df = pd.DataFrame({
        "price": median_path[1:],  # Skip first (it's the last historical)
        "p10": percentile_10[1:],
        "p90": percentile_90[1:],
        "p25": percentile_25[1:],
        "p75": percentile_75[1:],
        "type": "forecast",
    }, index=future_timestamps)
    
    combined = pd.concat([hist_df, pred_df])
    
    # Get consensus direction for each step
    consensus_directions = []
    for step in range(n_steps):
        step_directions = [all_directions[sim][step] for sim in range(n_simulations)]
        up_pct = sum(1 for d in step_directions if d == "UP") / n_simulations
        consensus_directions.append({
            "timestamp": future_timestamps[step],
            "up_probability": up_pct,
            "consensus": "UP" if up_pct > 0.5 else "DOWN",
        })
    
    return combined, consensus_directions, pd.DataFrame(all_paths.T)


def plot_forecast(
    combined: pd.DataFrame,
    directions: List[Dict],
    save_path: Path = None,
):
    """Plot forecast with uncertainty bands."""
    
    hist_mask = combined["type"] == "historical"
    pred_mask = combined["type"] == "forecast"
    
    hist_data = combined[hist_mask]
    pred_data = combined[pred_mask]
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})
    
    # === Price Chart ===
    ax1 = axes[0]
    
    # Historical
    ax1.plot(hist_data.index, hist_data["price"], color="blue", linewidth=1.5, label="Historical")
    
    # Forecast median
    ax1.plot(pred_data.index, pred_data["price"], color="red", linewidth=2, label="Forecast (median)")
    
    # 80% confidence band (10th-90th percentile)
    ax1.fill_between(
        pred_data.index,
        pred_data["p10"],
        pred_data["p90"],
        alpha=0.15,
        color="red",
        label="80% confidence band",
    )
    
    # 50% confidence band (25th-75th percentile)
    ax1.fill_between(
        pred_data.index,
        pred_data["p25"],
        pred_data["p75"],
        alpha=0.25,
        color="red",
        label="50% confidence band",
    )
    
    # Mark transition
    ax1.axvline(hist_data.index[-1], color="gray", linestyle=":", alpha=0.7)
    
    ax1.set_ylabel("XRP/USDT Price", fontsize=12)
    ax1.set_title("XRP Price Forecast (Monte Carlo Simulation)", fontsize=14, fontweight="bold")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
    ax1.xaxis.set_major_locator(mdates.DayLocator())
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # === Direction Probability Chart ===
    ax2 = axes[1]
    
    timestamps = [d["timestamp"] for d in directions]
    up_probs = [d["up_probability"] for d in directions]
    
    colors = ["green" if p > 0.6 else "red" if p < 0.4 else "gray" for p in up_probs]
    
    ax2.bar(timestamps, up_probs, color=colors, alpha=0.7, width=0.12)
    ax2.axhline(0.5, color="black", linestyle="--", alpha=0.5)
    ax2.axhline(0.6, color="green", linestyle=":", alpha=0.3)
    ax2.axhline(0.4, color="red", linestyle=":", alpha=0.3)
    
    ax2.set_ylabel("P(UP)", fontsize=12)
    ax2.set_ylim(0, 1)
    ax2.set_title("Direction Probability (Green=Bullish, Red=Bearish)", fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
    ax2.xaxis.set_major_locator(mdates.DayLocator())
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n📊 Plot saved to: {save_path}")
    
    plt.show()


def print_summary(combined: pd.DataFrame, directions: List[Dict]):
    """Print forecast summary."""
    
    hist_mask = combined["type"] == "historical"
    pred_mask = combined["type"] == "forecast"
    
    start_price = combined[hist_mask]["price"].iloc[-1]
    pred_data = combined[pred_mask]
    
    median_end = pred_data["price"].iloc[-1]
    low_end = pred_data["p10"].iloc[-1]
    high_end = pred_data["p90"].iloc[-1]
    
    median_change = (median_end / start_price - 1) * 100
    low_change = (low_end / start_price - 1) * 100
    high_change = (high_end / start_price - 1) * 100
    
    # Count bullish consensus
    bullish_steps = sum(1 for d in directions if d["consensus"] == "UP")
    
    print(f"""
{'='*60}
WEEKLY FORECAST SUMMARY
{'='*60}

📊 PRICE PROJECTION (with uncertainty)
   Current Price:     ${start_price:.4f}
   
   End of Week:
   ├── Pessimistic (10%):  ${low_end:.4f} ({low_change:+.1f}%)
   ├── Median (50%):       ${median_end:.4f} ({median_change:+.1f}%)
   └── Optimistic (90%):   ${high_end:.4f} ({high_change:+.1f}%)

📈 DIRECTION CONSENSUS
   Bullish periods:   {bullish_steps}/{len(directions)} ({100*bullish_steps/len(directions):.0f}%)
   Bearish periods:   {len(directions)-bullish_steps}/{len(directions)} ({100*(len(directions)-bullish_steps)/len(directions):.0f}%)

⚠️  RELIABILITY GUIDE
   • First 24h:  Most reliable
   • 24-48h:     Moderately reliable  
   • 48h+:       Use with caution
   • 5-7 days:   Directional bias only

💡 HOW TO USE
   • Trade on the DIRECTION, not exact prices
   • Focus on high-probability periods (>60% or <40%)
   • Re-run daily with fresh data
   • Use proper risk management

{'='*60}
""")


def main(days: int = 7, history_hours: int = 72, refresh: bool = False):
    """Main forecast function."""
    
    print(f"\n{'='*60}")
    print("XRP REALISTIC WEEKLY FORECAST")
    print(f"{'='*60}")
    
    if refresh:
        print("\n🔄 Refreshing data...")
        try:
            from data.collectors.price_collector import PriceCollector
            collector = PriceCollector()
            collector.run(timeframe="1h", years=2)
            from data.processors.feature_engineering import build_feature_dataset
            build_feature_dataset(timeframe="1h")
        except Exception as e:
            print(f"   Warning: {e}")
    
    print("\n🔄 Loading model...")
    model, ckpt = load_model()
    print(f"   ✓ Loaded (accuracy: {ckpt['test_accuracy']:.1%})")
    
    print("\n📊 Loading data...")
    df_raw = pd.read_parquet(DATA_DIR / "xrp_features_1h.parquet")
    print(f"   ✓ {len(df_raw):,} samples, latest: {df_raw.index[-1]}")
    print(f"   ✓ Current price: ${df_raw['close'].iloc[-1]:.4f}")
    
    # Generate forecast
    combined, directions, all_paths = generate_forecast(
        model=model,
        ckpt=ckpt,
        df_raw=df_raw,
        days=days,
        history_hours=history_hours,
        n_simulations=50,
    )
    
    # Print and plot
    print_summary(combined, directions)
    plot_forecast(combined, directions, save_path=ROOT / "forecast_weekly.png")
    
    return combined, directions


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", "-d", type=int, default=7)
    parser.add_argument("--history", "-H", type=int, default=72)
    parser.add_argument("--refresh", "-r", action="store_true")
    args = parser.parse_args()
    
    main(days=args.days, history_hours=args.history, refresh=args.refresh)