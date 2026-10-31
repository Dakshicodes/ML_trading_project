"""
tests/test_features.py – Unit tests for feature computation and look-ahead bias verification.
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import pytest

from src.features import FEATURE_COLUMNS, compute_features, create_labels


@pytest.fixture
def sample_ohlcv_data() -> pd.DataFrame:
    """Generate a realistic synthetic 200-candle OHLCV DataFrame."""
    np.random.seed(42)
    n = 200
    base_time = 1700000000000
    interval_ms = 5 * 60 * 1000

    timestamps = [base_time + i * interval_ms for i in range(n)]
    # Random walk close prices
    returns = np.random.normal(0, 0.002, n)
    close = 2000.0 * np.exp(np.cumsum(returns))
    high = close * (1.0 + np.abs(np.random.normal(0, 0.001, n)))
    low = close * (1.0 - np.abs(np.random.normal(0, 0.001, n)))
    open_price = (high + low) / 2.0
    volume = np.random.uniform(10.0, 100.0, n)

    df = pd.DataFrame(
        {
            "open_time": timestamps,
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )
    return df


def test_feature_columns_present(sample_ohlcv_data):
    """Verify that compute_features appends every declared feature column."""
    feat_df = compute_features(sample_ohlcv_data)

    for col in FEATURE_COLUMNS:
        assert col in feat_df.columns, f"Expected feature column '{col}' is missing"
        # Check that post warm-up features are valid finite numbers
        valid_subset = feat_df[col].iloc[60:]
        assert not valid_subset.isnull().any(), f"Column '{col}' has unexpected NaNs after warm-up"
        assert not np.isinf(valid_subset).any(), f"Column '{col}' has infinite values"


def test_strict_zero_lookahead_bias(sample_ohlcv_data):
    """
    CRITICAL TEST: Ensure modifying future rows (e.g., row 150)
    has ZERO effect on features calculated at earlier rows (e.g., row 70).
    """
    # 1. Compute features on unmodified data
    feat_original = compute_features(sample_ohlcv_data)

    # 2. Perturb future candles (from row 100 onward) drastically
    mutated_df = sample_ohlcv_data.copy()
    mutated_df.loc[100:, "close"] = mutated_df.loc[100:, "close"] * 5.0
    mutated_df.loc[100:, "high"] = mutated_df.loc[100:, "high"] * 5.5
    mutated_df.loc[100:, "low"] = mutated_df.loc[100:, "low"] * 4.5
    mutated_df.loc[100:, "volume"] = mutated_df.loc[100:, "volume"] * 100.0

    feat_mutated = compute_features(mutated_df)

    # 3. Assert rows 0 through 99 are identical across all features
    past_original = feat_original.loc[:99, FEATURE_COLUMNS]
    past_mutated = feat_mutated.loc[:99, FEATURE_COLUMNS]

    # Warm-up rows before index 50 may have NaNs, so check both NaN consistency and values
    pd.testing.assert_frame_equal(
        past_original,
        past_mutated,
        check_exact=True,
        obj="Past features altered by future candles (LOOK-AHEAD BIAS DETECTED!)",
    )


def test_target_labeling_logic():
    """Verify target labeling thresholds and forward-looking shifts."""
    df = pd.DataFrame(
        {
            "open_time": [1000, 2000, 3000, 4000, 5000],
            "open": [100.0, 100.0, 100.0, 100.0, 100.0],
            "high": [101.0, 101.0, 101.0, 101.0, 101.0],
            "low": [99.0, 99.0, 99.0, 99.0, 99.0],
            "close": [100.0, 100.0, 100.0, 105.0, 95.0],
            "volume": [10.0, 10.0, 10.0, 10.0, 10.0],
        }
    )

    # Horizon = 2 bars, Hurdle = 2% (0.02)
    labeled_df, target = create_labels(df, horizon_bars=2, fee_per_side=0.005, slippage_per_side=0.005)

    # Row 0: close=100.0, future close at t+2 (Row 2)=100.0 -> ret=0.0 -> target=0 (Neutral)
    assert target.iloc[0] == 0

    # Row 1: close=100.0, future close at t+2 (Row 3)=105.0 -> ret=+5% > +2% hurdle -> target=1 (Up)
    assert target.iloc[1] == 1

    # Row 2: close=100.0, future close at t+2 (Row 4)=95.0 -> ret=-5% < -2% hurdle -> target=-1 (Down)
    assert target.iloc[2] == -1

    # The last 2 rows must have NaN for forward_return as future is unknown
    assert pd.isna(labeled_df["forward_return"].iloc[-1])
    assert pd.isna(labeled_df["forward_return"].iloc[-2])


def test_features_on_real_dataset_sample():
    """Ensure feature calculation runs without error on the actual collected clean dataset."""
    from config import CLEAN_CSV

    if CLEAN_CSV.exists():
        real_df = pd.read_csv(CLEAN_CSV, nrows=500)
        feat_df = compute_features(real_df)
        assert len(feat_df) == 500
        assert all(c in feat_df.columns for c in FEATURE_COLUMNS)
