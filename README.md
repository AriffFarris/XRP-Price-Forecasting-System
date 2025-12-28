# XRP Price Predictor (Python / PyTorch)

End-to-end XRP forecasting + direction pipeline:
- **Price model:** LSTM regressor that predicts **future log-returns**, then converts back to price for evaluation/plots.
- **Direction model:** LSTM-based **momentum reversal detector** that predicts whether momentum will **continue vs reverse**, then converts that into an UP/DOWN direction call.
- **Deployment scripts:** quick direction inference, simple trade signal, and a realistic weekly forecast with Monte Carlo uncertainty bands.

>  Disclaimer: This project is for learning/experimentation and is **not financial advice**.

---

## Repository Structure

```text
xrp-predictor/
  data/
    collectors/
      price_collector.py
    processors/
      feature_engineering.py
    processed/
      xrp_raw_1h.parquet
      xrp_features_1h.parquet

  training/
    dataset.py
    train_price_regressor.py
    train_reversal_detector.py

  deployment/
    quick_predict.py
    inference_reversal.py
    forecast_week.py

  inference/
    trade_signal.py

  diagnostics/
    diagnose_reversal_detector.py
    plot_price_regressor.py
    graph.py

  saved_models/
    lstm_price_regressor_1h_h4_c48.pt
    reversal_detector_1h_h4_c48.pt
    ... (other checkpoints)

  *.png    # diagnostic + forecast plots
Setup
1) Create environment
bash
Copy code
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Mac/Linux:
# source .venv/bin/activate
2) Install dependencies
bash
Copy code
pip install -r requirements.txt
If you don’t have a requirements.txt yet, you’ll need at least:

torch

pandas, numpy

scikit-learn

matplotlib

ccxt

yfinance

pyarrow

pandas_ta

Data Pipeline
Step A — Collect OHLCV (Binance via ccxt) + Market Proxy (yfinance)
Downloads historical candles with pagination and saves:

data/processed/xrp_raw_<timeframe>.parquet

bash
Copy code
python data/collectors/price_collector.py --timeframe 1h --years 2
# or with a fixed start date:
python data/collectors/price_collector.py --timeframe 1h --start 2020-01-01
Step B — Feature Engineering
Builds technical indicators (RSI, MACD, Bollinger Bands, EMAs, ATR, OBV, etc.) and saves:

data/processed/xrp_features_<timeframe>.parquet

bash
Copy code
python data/processors/feature_engineering.py
Models
1) Price Forecasting (Log-return Regression)
Goal: predict

text
Copy code
y = log(close[t+h] / close[t])
pred_close[t+h] = close[t] * exp(y_hat)
Key details:

Time-based split: train/val/test = 70/15/15 (no leakage)

RobustScaler fit on train only

Sequence windowing via context length

Metrics: MAE, RMSE, MAPE, sMAPE, R² + directional accuracy vs persistence baseline

Train:

bash
Copy code
python -m training.train_price_regressor --timeframe 1h --horizon 4 --context 48 --epochs 30 --batch 64
Checkpoint output:

saved_models/lstm_price_regressor_1h_h4_c48.pt

2) Direction Forecasting (Momentum Reversal Detector)
Goal: predict P(reversal) over horizon h.

Convert to direction:

determine current direction from last 1-step return

if P(reversal) > threshold → flip direction

else → continue direction

Key details:

Reversal-focused features built from OHLCV only (no future-leaking shift(-k) features)

Class imbalance handled with pos_weight (BCEWithLogitsLoss)

Threshold tuned on validation set to maximise direction accuracy

Reports test direction accuracy vs persistence baseline

Train:

bash
Copy code
python -m training.train_reversal_detector --timeframe 1h --horizon 4 --context 48 --epochs 100
Checkpoint output:

saved_models/reversal_detector_1h_h4_c48.pt

Inference / Deployment
Quick one-line prediction
bash
Copy code
python -m deployment.quick_predict
Detailed direction prediction (+ confidence)
bash
Copy code
python -m deployment.inference_reversal
# Refresh data first:
python -m deployment.inference_reversal --refresh
# JSON output:
python -m deployment.inference_reversal --json
Simple Trade Signal (BUY / SELL / HOLD)
Uses the reversal detector direction + optional expected move magnitude from the price regressor.

bash
Copy code
python -m inference.trade_signal --timeframe 1h --horizon 4 --context 48 --min_move_pct 0.25 --conf_low 0.30 --conf_high 0.70
# Disable the price regressor magnitude filter:
python -m inference.trade_signal --no_price_model
Weekly Forecast (Monte Carlo, uncertainty bands)
Direction-driven simulation with:

historical move sampling

mean reversion pressure

uncertainty bands from multiple simulation paths

bash
Copy code
python -m deployment.forecast_week
# Refresh data first:
python -m deployment.forecast_week --refresh
Outputs:

forecast_weekly.png (and other forecast plots)

Diagnostics
Scripts and plots live under:

diagnostics/

diagnostics_*.png

Notes / Limitations
This is not a trading system (no slippage, fees, execution logic, or proper risk management).

Performance depends on regime and market conditions; direction accuracy can vary over time.

For stronger evaluation: walk-forward validation, feature ablations, and calibration tests.

Next Improvements (Ideas)
Walk-forward / rolling retraining

Probabilistic forecasting (quantiles) for price model

Feature ablation (which indicators actually help)

Experiment tracking (MLflow / W&B)

CI checks + basic unit tests for the data pipeline
