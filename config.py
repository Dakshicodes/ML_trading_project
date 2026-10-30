"""
config.py – Central configuration for the AI-Powered Trading Indicator project.

All constants and path objects used across data download, training, live feed,
and the web server are defined here.  Import from this module; do NOT hard-code
values elsewhere.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Project root – everything is relative to this
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Asset & timeframe
# ---------------------------------------------------------------------------
SYMBOL: str = "ETHUSDT"
INTERVAL: str = "5m"          # Binance kline interval string
INTERVAL_MS: int = 5 * 60 * 1000   # Duration of one candle in milliseconds (300 000 ms)
HORIZON_BARS: int = 3         # Prediction horizon in candles (3 × 5 min = 15 min)

# ---------------------------------------------------------------------------
# Download settings
# ---------------------------------------------------------------------------
HISTORY_MONTHS: int = 24      # How many months of history to download
REST_BASE_URL: str = "https://data-api.binance.vision"   # Public data endpoint – no API key needed

# ---------------------------------------------------------------------------
# Model / experiment settings
# ---------------------------------------------------------------------------
RANDOM_SEED: int = 42

# ---------------------------------------------------------------------------
# Paper-trading cost model (placeholders – review before backtesting)
# ---------------------------------------------------------------------------
FEE_PER_SIDE: float = 0.001        # 0.10 % taker fee each side
SLIPPAGE_PER_SIDE: float = 0.0002  # 0.02 % assumed slippage each side

# ---------------------------------------------------------------------------
# Directory paths (created on demand by individual scripts)
# ---------------------------------------------------------------------------
DATA_DIR       = PROJECT_ROOT / "data"
RAW_DIR        = DATA_DIR / "raw"
PROCESSED_DIR  = DATA_DIR / "processed"
MODELS_DIR     = PROJECT_ROOT / "models"
REPORTS_DIR    = PROJECT_ROOT / "reports"
LOGS_DIR       = PROJECT_ROOT / "logs"

# ---------------------------------------------------------------------------
# File paths
# ---------------------------------------------------------------------------
RAW_CSV       = RAW_DIR       / f"{SYMBOL}_{INTERVAL}.csv"
CLEAN_CSV     = PROCESSED_DIR / f"{SYMBOL}_{INTERVAL}_clean.csv"
MODEL_PATH    = MODELS_DIR    / "model.joblib"
SIGNAL_LOG    = LOGS_DIR      / "signals.jsonl"
