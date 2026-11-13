"""
tests/test_store.py – Unit tests for SignalPredictor and CandleStore with no-repainting verification.
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

from src.predictor import SignalPredictor
from src.store import CandleStore


@pytest.fixture
def sample_history_df() -> pd.DataFrame:
    """Generate 100 realistic 5m candles for testing store and predictor."""
    np.random.seed(42)
    n = 100
    base_time = 1700000000000
    interval_ms = 5 * 60 * 1000

    timestamps = [base_time + i * interval_ms for i in range(n)]
    returns = np.random.normal(0, 0.002, n)
    close = 2500.0 * np.exp(np.cumsum(returns))
    high = close * 1.002
    low = close * 0.998
    open_price = (high + low) / 2.0
    volume = np.random.uniform(50.0, 200.0, n)

    return pd.DataFrame(
        {
            "open_time": timestamps,
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


def test_predictor_output_contract(sample_history_df):
    """Verify SignalPredictor outputs required keys with valid bounds."""
    predictor = SignalPredictor()
    result = predictor.predict(sample_history_df)

    assert "direction" in result
    assert result["direction"] in ["Up", "Down", "Neutral"]
    assert "confidence" in result
    assert 0.0 <= result["confidence"] <= 1.0
    assert "prob_up" in result
    assert 0.0 <= result["prob_up"] <= 1.0
    assert "prob_down" in result
    assert 0.0 <= result["prob_down"] <= 1.0
    assert abs(result["prob_up"] + result["prob_down"] - 1.0) < 1e-3


def test_predictor_insufficient_warmup_error(sample_history_df):
    """Verify ValueError is raised if candle history is too short for indicators."""
    predictor = SignalPredictor()
    short_df = sample_history_df.head(30)
    with pytest.raises(ValueError, match="requires at least 60 candles"):
        predictor.predict(short_df)


def test_candle_store_initialization(sample_history_df):
    """Verify initial seeding and pre-computed signals in CandleStore."""
    store = CandleStore(max_candles=200)
    loaded = store.load_initial_history(sample_history_df, compute_historical_signals=True)

    assert loaded == 100
    assert store.count_candles() == 100
    chart_data = store.get_chart_data(limit=50)
    assert len(chart_data) == 50

    # Verify chart format fits TradingView Lightweight Charts requirements
    first_item = chart_data[0]
    assert "time" in first_item
    assert "open" in first_item
    assert "high" in first_item
    assert "low" in first_item
    assert "close" in first_item
    assert "signal" in first_item
    assert "confidence" in first_item
    # time must be in seconds (not milliseconds)
    assert first_item["time"] < 1e11


def test_strictly_no_repainting(sample_history_df):
    """
    CRITICAL TEST (Brief Section 4.1):
    Signals locked on closed candles must NEVER change when new ticks arrive
    or subsequent candles close.
    """
    store = CandleStore(max_candles=200)
    store.load_initial_history(sample_history_df.iloc[:80], compute_historical_signals=True)

    # 1. Close a new candle (candle 81)
    row_81 = sample_history_df.iloc[81]
    candle_81 = {
        "open_time": int(row_81["open_time"]),
        "open": float(row_81["open"]),
        "high": float(row_81["high"]),
        "low": float(row_81["low"]),
        "close": float(row_81["close"]),
        "volume": float(row_81["volume"]),
        "is_closed": True,
    }
    sig_81 = store.on_candle_update(candle_81)
    assert sig_81 is not None, "Closed candle 81 must generate a signal"
    orig_direction_81 = sig_81["direction"]
    orig_confidence_81 = sig_81["confidence"]

    # 2. Simulate live ticks arriving for an unclosed candle 82
    time_82 = int(row_81["open_time"]) + 300000
    for price_tick in [2400.0, 2600.0, 2300.0, 2700.0]:
        store.on_candle_update(
            {
                "open_time": time_82,
                "open": 2500.0,
                "high": max(2500.0, price_tick),
                "low": min(2500.0, price_tick),
                "close": price_tick,
                "volume": 10.0,
                "is_closed": False,
            }
        )
        # Verify candle 81's signal has NOT changed during forming ticks
        chart = store.get_chart_data(limit=10)
        c81_data = next(item for item in chart if item["time"] == candle_81["open_time"] // 1000)
        assert c81_data["signal"] == orig_direction_81
        assert c81_data["confidence"] == orig_confidence_81

    # 3. Close candle 82 with huge volatility and check candle 81 again
    store.on_candle_update(
        {
            "open_time": time_82,
            "open": 2500.0,
            "high": 3500.0,
            "low": 2000.0,
            "close": 3400.0,
            "volume": 500.0,
            "is_closed": True,
        }
    )

    # Re-verify candle 81
    chart_after_c82 = store.get_chart_data(limit=10)
    c81_final = next(item for item in chart_after_c82 if item["time"] == candle_81["open_time"] // 1000)
    assert c81_final["signal"] == orig_direction_81, "LOCKED SIGNAL REPAINTED! VIOLATION DETECTED."
    assert c81_final["confidence"] == orig_confidence_81, "LOCKED CONFIDENCE REPAINTED! VIOLATION DETECTED."
