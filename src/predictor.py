"""
src/predictor.py – Real-time inference engine for the AI Trading Indicator.

Loads the trained model artifact from models/model.joblib, executes feature
engineering on the latest rolling candle window, and outputs:
- Direction: 'Up', 'Down', or 'Neutral'
- Confidence: Calibrated probability percentage (e.g. 0.64 for 64%)
- Continuous class probabilities: prob_up, prob_down
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.features import FEATURE_COLUMNS, compute_features

logger = logging.getLogger("src.predictor")


class SignalPredictor:
    """
    Inference class that wraps the trained model and feature pipeline.
    """

    def __init__(self, model_path: Path = config.MODEL_PATH, confidence_threshold: Optional[float] = None):
        self.model_path = model_path
        self.model = None
        self.feature_names = FEATURE_COLUMNS
        self.confidence_threshold = confidence_threshold
        self.round_trip_cost = 0.0024
        self.symbol = config.SYMBOL
        self.interval = config.INTERVAL
        self.horizon_bars = config.HORIZON_BARS
        self.min_warmup_candles = 60  # Minimum candles required for rolling indicators (EMA-50, etc.)
        self.load_model()

    def load_model(self) -> None:
        """Load the serialized model bundle."""
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model bundle not found at {self.model_path}. Train the model first via src/train.py."
            )

        logger.info("Loading model bundle from %s...", self.model_path)
        bundle = joblib.load(self.model_path)
        self.model = bundle["model"]
        self.feature_names = bundle.get("feature_names", FEATURE_COLUMNS)

        # Allow override; otherwise use calibrated threshold from training
        if self.confidence_threshold is None:
            self.confidence_threshold = 0.52

        self.round_trip_cost = bundle.get("round_trip_cost", 0.0024)
        logger.info(
            "Predictor initialized with confidence_threshold=%.2f (horizon=%d bars).",
            self.confidence_threshold,
            self.horizon_bars,
        )

    def predict(self, candle_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Compute features on the sliding window and predict signal for the latest closed candle.

        Parameters
        ----------
        candle_df : pd.DataFrame
            Must contain at least 60 candles with columns: open_time, open, high, low, close, volume.
            Sorted chronologically in ascending order.

        Returns
        -------
        Dict with keys:
            - direction: "Up", "Down", or "Neutral"
            - confidence: float (0.0 to 1.0)
            - prob_up: float (0.0 to 1.0)
            - prob_down: float (0.0 to 1.0)
            - threshold: float
            - open_time: int (timestamp of the evaluated candle)
            - close_price: float
        """
        if len(candle_df) < self.min_warmup_candles:
            raise ValueError(
                f"Predictor requires at least {self.min_warmup_candles} candles for feature warm-up, but received {len(candle_df)}."
            )

        # Ensure correct column types
        df_copy = candle_df[["open_time", "open", "high", "low", "close", "volume"]].copy()
        for col in ["open", "high", "low", "close", "volume"]:
            df_copy[col] = df_copy[col].astype(np.float64)
        df_copy["open_time"] = df_copy["open_time"].astype(np.int64)

        # Compute technical features (Single Source of Truth)
        feat_df = compute_features(df_copy)

        # Extract latest row (the candle being evaluated)
        latest_row = feat_df.iloc[-1]
        latest_features = latest_row[self.feature_names].values.reshape(1, -1)

        # Check for any unexpected NaNs
        if np.isnan(latest_features).any():
            nan_cols = [self.feature_names[i] for i, val in enumerate(latest_features[0]) if np.isnan(val)]
            raise ValueError(f"Feature calculation produced NaNs in latest candle for columns: {nan_cols}")

        # Model predict probability: class 1 is Up, class 0 is Down
        probs = self.model.predict_proba(latest_features)[0]
        # In binary classifier: classes_ are [0, 1]
        prob_up = float(probs[1])
        prob_down = float(probs[0])

        tau = float(self.confidence_threshold)

        if prob_up >= tau:
            direction = "Up"
            confidence = prob_up
        elif prob_down >= tau:
            direction = "Down"
            confidence = prob_down
        else:
            direction = "Neutral"
            confidence = max(prob_up, prob_down)

        return {
            "direction": direction,
            "confidence": round(confidence, 4),
            "prob_up": round(prob_up, 4),
            "prob_down": round(prob_down, 4),
            "threshold": round(tau, 4),
            "open_time": int(latest_row["open_time"]),
            "close_price": float(latest_row["close"]),
        }


# Global lazy singleton instance
_predictor_instance: Optional[SignalPredictor] = None


def get_predictor() -> SignalPredictor:
    """Get or initialize the global SignalPredictor singleton."""
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = SignalPredictor()
    return _predictor_instance


def predict_signal(candle_df: pd.DataFrame) -> Dict[str, Any]:
    """Convenience function calling the singleton predictor."""
    return get_predictor().predict(candle_df)
