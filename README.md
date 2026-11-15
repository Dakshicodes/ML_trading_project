# AI-Powered Trading Indicator (ETH/USDT 5m)

An end-to-end machine learning trading indicator for **ETHUSDT 5-minute candlesticks**, predicting price direction over a **15-minute horizon (3 candles)** with real-time calibrated confidence and **zero repainting**.

Live Chart Web App featuring a FastAPI backend and TradingView's Lightweight Charts frontend.

---

## Features
- **Asset**: `ETHUSDT` (Ethereum / Tether)
- **Timeframe & Horizon**: `5m` candles | `3 bars` forward (15 minutes)
- **Data Scope**: 24 months of high-frequency data (**207,360 candles**) from Binance public API.
- **22 Scale-Invariant Features**: Log returns, RSI-14, MACD, Moving Average spreads (EMA 9, 21, 50), Normalized ATR, Bollinger Bands %B, Volume z-scores, Candle anatomy, and Cyclical session time.
- **Strict Zero Lookahead Bias**: Verified by automated mutation unit tests.
- **Calibrated Probabilistic Model**: `HistGradientBoostingClassifier` with confidence threshold gating ($\tau = 0.60$).
- **No Repainting**: Signals locked on closed candles are permanent and immutable.
- **Live Streaming Web App**: FastAPI + WebSockets + TradingView Lightweight Charts.

---

## Directory Layout
```
ML_TRADING_PROJECT/
├── config.py           # Centralized configuration (ETHUSDT, 5m, fees, paths)
├── requirements.txt    # Project dependencies
├── README.md           # Project documentation
├── .gitignore          # Ignored files (data, models, logs, venv)
├── docs/
│   ├── brief.pdf       # Student project brief
│   ├── report.md       # Comprehensive technical evaluation report
│   ├── user_guide.md   # Practical guide for traders & risk rules
│   └── notes.md        # Architecture & run notes
├── data/
│   ├── raw/            # ETHUSDT_5m.csv (raw download)
│   └── processed/      # ETHUSDT_5m_clean.parquet / CSV (validated, 0 gaps)
├── models/
│   └── model.joblib    # Serialized model artifact bundle
├── reports/
│   ├── metrics.json    # Out-of-sample benchmark metrics vs baselines
│   └── evaluation_summary.txt
├── src/
│   ├── __init__.py
│   ├── data.py         # Binance historical REST ingestion & validation
│   ├── features.py     # 22 stationary technical indicators (single source of truth)
│   ├── train.py        # Chronological 70/15/15 time-series split & training
│   ├── evaluate.py     # Benchmark evaluation vs Always Up & RSI rules
│   ├── predictor.py    # Real-time sliding window inference
│   ├── store.py        # In-memory candle store & no-repainting enforcement
│   ├── live.py         # Binance live WebSocket client
│   └── server.py       # FastAPI web server & WebSocket broadcaster
├── tests/
│   ├── test_features.py # Unit tests for features & zero lookahead bias
│   └── test_store.py    # Unit tests for predictor & zero repainting
└── web/
    ├── index.html      # Live chart dashboard
    ├── app.js          # TradingView Lightweight Charts & WebSocket handler
    └── style.css       # Dark institutional trading terminal theme
```

---

## How to Run

### 1. Activate Environment & Run Tests
```powershell
.venv\Scripts\Activate.ps1
pytest tests/ -v
```
*All 8 unit tests must pass (verifying zero lookahead bias and no repainting).*

### 2. Reproduce Pipeline
```bash
# Download 24 months of historical candles (if not already present)
python src/data.py

# Train the model with chronological time-series splitting
python src/train.py

# Run benchmark evaluation against Always Up and RSI rules
python src/evaluate.py
```

### 3. Launch Live Chart Web Dashboard
```bash
python -m uvicorn src.server:app --host 127.0.0.1 --port 8000
```
Open **[http://localhost:8000](http://localhost:8000)** in your browser.
- Displays live streaming candlesticks from Binance WebSockets.
- Overlays **`▲ UP`** and **`▼ DOWN`** signal markers with confidence percentages.
- Automatically updates in real time.
