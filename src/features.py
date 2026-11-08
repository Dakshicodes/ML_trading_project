"""
src/features.py – Feature engineering and target labeling for the AI Trading Indicator.

Design Rules:
1. ONE public function, `compute_features(df)`, is the single source of truth for
   all derived columns. It is called identically by training (train.py) and
   live inference (predictor.py).
2. Strict zero look-ahead bias: all features for candle t are calculated strictly
   using data up to and including candle t.
3. Target labeling is computed separately via `create_labels(df, horizon_bars, fee_hurdle)`
   so inference never accesses future labels.
"""

import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

# Add project root to sys.path if not present
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config

# List of all engineered feature column names for model training & inference
FEATURE_COLUMNS: List[str] = [
    # Returns & Momentum
    "return_1",
    "return_3",
    "return_6",
    "return_12",
    "rsi_14",
    "macd",
    "macd_signal",    "macd_hist",
    # Moving Average Spreads (percentage difference from close)
    "ema_9_diff",
    "ema_21_diff",
    "ema_50_diff",
    "ema_9_21_spread",
    # Volatility & Bands
    "natr_14",
    "bb_percent_b",
    "bb_bandwidth",
    # Volume Dynamics
    "volume_ratio_20",
    "volume_return_3",
    # Candle Structure & Geometry
    "body_size_ratio",
    "upper_wick_ratio",
    "lower_wick_ratio",
    "is_bullish",
    # Intraday Time Features (Cyclical time-of-day)
    "hour_sin",
    "hour_cos",
]


def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index (RSI) using Wilder's smoothing."""
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    # Wilder's exponential moving average
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate True Range and Average True Range (ATR)."""
    prev_close = df["close"].shift(1)
    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - prev_close).abs()
    tr3 = (df["low"] - prev_close).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    return atr


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute technical indicator features from raw OHLCV candles.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: open, high, low, close, volume, open_time.
        Must be sorted in ascending open_time order.

    Returns
    -------
    pd.DataFrame
        Copy of input DataFrame with all FEATURE_COLUMNS added.
        Rows where rolling warm-up windows produce NaNs are preserved or dropped
        consistently based on dropna parameter.
    """
    required_cols = {"open", "high", "low", "close", "volume", "open_time"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Input DataFrame is missing required columns: {missing}")

    feat = df.copy()

    # 1. Price Momentum & Returns (log returns to preserve symmetry)
    close = feat["close"]
    feat["return_1"] = np.log(close / close.shift(1))
    feat["return_3"] = np.log(close / close.shift(3))
    feat["return_6"] = np.log(close / close.shift(6))
    feat["return_12"] = np.log(close / close.shift(12))

    # RSI
    feat["rsi_14"] = calculate_rsi(close, period=14)

    # MACD (12, 26, 9)
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    feat["macd"] = (ema_12 - ema_26) / (close + 1e-10)  # normalized by close
    feat["macd_signal"] = feat["macd"].ewm(span=9, adjust=False).mean()
    feat["macd_hist"] = feat["macd"] - feat["macd_signal"]

    # 2. Moving Average Spreads
    ema_9 = close.ewm(span=9, adjust=False).mean()
    ema_21 = close.ewm(span=21, adjust=False).mean()
    ema_50 = close.ewm(span=50, adjust=False).mean()

    feat["ema_9_diff"] = (close - ema_9) / (close + 1e-10)
    feat["ema_21_diff"] = (close - ema_21) / (close + 1e-10)
    feat["ema_50_diff"] = (close - ema_50) / (close + 1e-10)
    feat["ema_9_21_spread"] = (ema_9 - ema_21) / (close + 1e-10)

    # 3. Volatility & Bollinger Bands (20 periods, 2 std)
    rolling_20_mean = close.rolling(window=20).mean()
    rolling_20_std = close.rolling(window=20).std()
    bb_upper = rolling_20_mean + 2.0 * rolling_20_std
    bb_lower = rolling_20_mean - 2.0 * rolling_20_std

    # %B: 1.0 = upper band, 0.0 = lower band, 0.5 = middle
    band_span = bb_upper - bb_lower + 1e-10
    feat["bb_percent_b"] = (close - bb_lower) / band_span
    feat["bb_bandwidth"] = band_span / (rolling_20_mean + 1e-10)

    # Normalized ATR (NATR = ATR / Close)
    atr_14 = calculate_atr(feat, period=14)
    feat["natr_14"] = atr_14 / (close + 1e-10)

    # 4. Volume Dynamics
    vol = feat["volume"]
    vol_ma_20 = vol.rolling(window=20).mean() + 1e-10
    feat["volume_ratio_20"] = vol / vol_ma_20
    feat["volume_return_3"] = np.log((vol + 1.0) / (vol.shift(3) + 1.0))

    # 5. Candlestick Anatomy & Geometry
    candle_range = feat["high"] - feat["low"] + 1e-10
    body = (close - feat["open"]).abs()
    feat["body_size_ratio"] = body / candle_range

    upper_shadow = feat["high"] - np.maximum(feat["open"], close)
    lower_shadow = np.minimum(feat["open"], close) - feat["low"]
    feat["upper_wick_ratio"] = upper_shadow / candle_range
    feat["lower_wick_ratio"] = lower_shadow / candle_range
    feat["is_bullish"] = (close >= feat["open"]).astype(np.float64)

    # 6. Cyclical Time-of-Day Features (UTC)
    datetimes = pd.to_datetime(feat["open_time"], unit="ms", utc=True)
    minute_of_day = datetimes.dt.hour * 60 + datetimes.dt.minute
    total_minutes = 24 * 60
    feat["hour_sin"] = np.sin(2 * np.pi * minute_of_day / total_minutes)
    feat["hour_cos"] = np.cos(2 * np.pi * minute_of_day / total_minutes)

    return feat


def create_labels(
    df: pd.DataFrame,
    horizon_bars: int = config.HORIZON_BARS,
    fee_per_side: float = config.FEE_PER_SIDE,
    slippage_per_side: float = config.SLIPPAGE_PER_SIDE,
    binary: bool = True,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Generate target labels based on forward return over `horizon_bars`.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with at least 'close' column.
    horizon_bars : int
        Number of future bars to compute return (default: config.HORIZON_BARS).
    fee_per_side : float
        Taker fee per side (default: 0.10%).
    slippage_per_side : float
        Assumed slippage per side (default: 0.02%).
    binary : bool
        If True (recommended for probability calibration):
            1: Up (forward return > 0)
            0: Down (forward return <= 0)
        If False:
            1: Up (> hurdle)
            0: Neutral (within [-hurdle, +hurdle])
           -1: Down (< -hurdle)

    Returns
    -------
    (df_with_target, target_series)
    """
    hurdle = 2.0 * (fee_per_side + slippage_per_side)

    # Forward price shift: close at t + horizon_bars
    future_close = df["close"].shift(-horizon_bars)
    forward_return = (future_close - df["close"]) / df["close"]

    df_labeled = df.copy()
    df_labeled["forward_return"] = forward_return

    # Binary directional target (1: Up, 0: Down)
    target_binary = pd.Series(0, index=df.index, dtype=np.int64)
    target_binary[forward_return > 0.0] = 1
    df_labeled["target_binary"] = target_binary

    # Ternary hurdle target (1: Up, 0: Neutral, -1: Down)
    target_ternary = pd.Series(0, index=df.index, dtype=np.int64)
    target_ternary[forward_return > hurdle] = 1
    target_ternary[forward_return < -hurdle] = -1
    df_labeled["target_ternary"] = target_ternary

    if binary:
        df_labeled["target"] = target_binary
        return df_labeled, target_binary
    else:
        df_labeled["target"] = target_ternary
        return df_labeled, target_ternary
