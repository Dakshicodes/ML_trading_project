# Project Engineering Notes & Run Instructions

## 1. Quick Start Guide

### Step 1: Activate Virtual Environment
```powershell
# In PowerShell from project root
.venv\Scripts\Activate.ps1
```

### Step 2: Run Unit Tests
```bash
pytest tests/ -v
```
Verifies all 8 unit tests:
- Zero look-ahead bias mutation test
- Feature calculation completeness
- Label generation logic
- SignalPredictor output contracts
- CandleStore zero-repainting enforcement

### Step 3: Run the Full Pipeline
```bash
# 1. Download & Clean 24 months of ETHUSDT 5m candles
python src/data.py

# 2. Train model & Calibrate confidence threshold (70/15/15 chronological split)
python src/train.py

# 3. Evaluate out-of-sample holdout test set vs baselines
python src/evaluate.py
```

### Step 4: Launch the Live Dashboard
```bash
# Start FastAPI web server on http://localhost:8000
python -m uvicorn src.server:app --host 127.0.0.1 --port 8000
```
Open [http://localhost:8000](http://localhost:8000) in your web browser to view the live TradingView chart with streaming signals.

---

## 2. Key Architecture Decisions

* **Asset**: `ETHUSDT` (Ethereum perpetual/spot on Binance)
* **Timeframe**: `5m` (300,000 ms per candle)
* **Horizon**: `3 bars` (15 minutes forward)
* **Model**: `HistGradientBoostingClassifier` (LightGBM-style decision trees)
* **Threshold**: $\tau = 0.60$ (requires $\ge 60\%$ probability to fire a directional signal)
* **Data Sources**:
  - Historical: Binance Public REST API (`https://data-api.binance.vision`)
  - Live Stream: Binance Public WebSocket (`wss://stream.binance.com:9443/ws/ethusdt@kline_5m`)
* **Cost Model**: $0.10\%$ taker fee $+ 0.02\%$ slippage per side ($0.24\%$ round-trip hurdle)
