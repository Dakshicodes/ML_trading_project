"""
src/store.py – In-memory candle buffer and signal store with strict no-repainting guarantees.

Maintains a sliding window of historical candles, tracks the active forming candle,
and stores immutable prediction signals for each closed candle.
"""

import logging
import sys
import threading
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.predictor import get_predictor

logger = logging.getLogger("src.store")


class CandleStore:
    """
    Thread-safe in-memory store for OHLCV candles and immutable prediction signals.
    """

    def __init__(self, max_candles: int = 500):
        self.max_candles = max_candles
        self._lock = threading.Lock()
        self._candles: List[Dict[str, Any]] = []  # Closed candles
        self._forming_candle: Optional[Dict[str, Any]] = None  # Active unclosed candle
        self._locked_signals: Dict[int, Dict[str, Any]] = {}  # open_time -> signal dict

    def load_initial_history(self, df: pd.DataFrame, compute_historical_signals: bool = True) -> int:
        """
        Seed the store with historical candles (e.g. from data/processed).
        Pre-computes signals for recent closed candles so the live dashboard opens ready.
        """
        with self._lock:
            self._candles.clear()
            self._locked_signals.clear()
            self._forming_candle = None

            # Keep the last max_candles rows
            tail_df = df.tail(self.max_candles).copy().reset_index(drop=True)

            for _, row in tail_df.iterrows():
                candle = {
                    "open_time": int(row["open_time"]),
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "volume": float(row["volume"]),
                    "is_closed": True,
                }
                self._candles.append(candle)

            logger.info("Loaded %d historical candles into CandleStore.", len(self._candles))

        if compute_historical_signals and len(tail_df) >= 60:
            predictor = get_predictor()
            logger.info("Pre-computing signals for the latest 100 historical closed candles...")
            # Generate signals for the last min(100, len - 60) candles
            start_idx = max(60, len(tail_df) - 100)
            with self._lock:
                for i in range(start_idx, len(tail_df)):
                    sub_df = tail_df.iloc[: i + 1]
                    c_time = int(sub_df.iloc[-1]["open_time"])
                    if c_time not in self._locked_signals:
                        try:
                            sig = predictor.predict(sub_df)
                            self._locked_signals[c_time] = sig
                        except Exception as e:
                            logger.debug("Could not compute signal for candle %d: %s", c_time, e)

        logger.info("Historical initialization complete. Locked signals: %d", len(self._locked_signals))
        return len(self._candles)

    def on_candle_update(self, candle: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Handle an incoming candle update from a live WebSocket or polling feed.

        Parameters
        ----------
        candle : Dict with keys:
            open_time, open, high, low, close, volume, is_closed

        Returns
        -------
        Optional[Dict]: Generated signal if candle just closed, else None.
        """
        is_closed = bool(candle.get("is_closed", False))
        open_time = int(candle["open_time"])

        new_signal = None

        with self._lock:
            if not is_closed:
                # Active candle is forming in real-time
                self._forming_candle = {
                    "open_time": open_time,
                    "open": float(candle["open"]),
                    "high": float(candle["high"]),
                    "low": float(candle["low"]),
                    "close": float(candle["close"]),
                    "volume": float(candle["volume"]),
                    "is_closed": False,
                }
                return None

            # Candle just closed: lock it in
            closed_candle = {
                "open_time": open_time,
                "open": float(candle["open"]),
                "high": float(candle["high"]),
                "low": float(candle["low"]),
                "close": float(candle["close"]),
                "volume": float(candle["volume"]),
                "is_closed": True,
            }

            # Check if this candle was already recorded as closed
            existing_idx = None
            for idx, c in enumerate(self._candles):
                if c["open_time"] == open_time:
                    existing_idx = idx
                    break

            if existing_idx is not None:
                self._candles[existing_idx] = closed_candle
            else:
                self._candles.append(closed_candle)
                if len(self._candles) > self.max_candles:
                    self._candles.pop(0)

            # Clear forming candle if matching
            if self._forming_candle and self._forming_candle["open_time"] == open_time:
                self._forming_candle = None

            # Enforce NO-REPAINTING: If a signal was already computed for this candle, NEVER change it!
            if open_time in self._locked_signals:
                logger.debug("Signal for candle %d is already locked (preventing repainting).", open_time)
                return self._locked_signals[open_time]

            # Otherwise, prepare DataFrame snapshot and compute new signal
            snapshot_df = pd.DataFrame(self._candles)

        # Compute signal outside lock to avoid blocking readers during inference
        if len(snapshot_df) >= 60:
            try:
                predictor = get_predictor()
                new_signal = predictor.predict(snapshot_df)
                with self._lock:
                    # Lock signal permanently for this closed candle
                    self._locked_signals[open_time] = new_signal
                logger.info(
                    "Locked signal for closed candle %d: %s (%.1f%% confidence)",
                    open_time,
                    new_signal["direction"],
                    new_signal["confidence"] * 100,
                )
            except Exception as e:
                logger.error("Failed to generate signal for closed candle %d: %s", open_time, e)

        return new_signal

    def get_chart_data(self, limit: int = 200) -> List[Dict[str, Any]]:
        """
        Format candle series for TradingView Lightweight Charts.
        Converts timestamps to seconds (open_time // 1000) and attaches locked signals.
        """
        with self._lock:
            candles_subset = self._candles[-limit:] if limit < len(self._candles) else list(self._candles)

            result = []
            for c in candles_subset:
                t_sec = c["open_time"] // 1000
                sig = self._locked_signals.get(c["open_time"], None)

                item = {
                    "time": t_sec,
                    "open": c["open"],
                    "high": c["high"],
                    "low": c["low"],
                    "close": c["close"],
                    "volume": c["volume"],
                    "is_closed": True,
                    "signal": sig["direction"] if sig else "Neutral",
                    "confidence": sig["confidence"] if sig else 0.0,
                    "prob_up": sig["prob_up"] if sig else 0.5,
                    "prob_down": sig["prob_down"] if sig else 0.5,
                }
                result.append(item)

            # If an active unclosed candle exists, append it as the final tick (without locked signal)
            if self._forming_candle:
                fc = self._forming_candle
                result.append(
                    {
                        "time": fc["open_time"] // 1000,
                        "open": fc["open"],
                        "high": fc["high"],
                        "low": fc["low"],
                        "close": fc["close"],
                        "volume": fc["volume"],
                        "is_closed": False,
                        "signal": "Neutral",
                        "confidence": 0.0,
                        "prob_up": 0.5,
                        "prob_down": 0.5,
                    }
                )

            return result

    def get_latest_signal(self) -> Optional[Dict[str, Any]]:
        """Return the most recently locked prediction signal."""
        with self._lock:
            if not self._locked_signals:
                return None
            latest_time = max(self._locked_signals.keys())
            return self._locked_signals[latest_time]

    def count_candles(self) -> int:
        """Return total closed candles currently stored."""
        with self._lock:
            return len(self._candles)


# Global singleton instance
_store_instance: Optional[CandleStore] = None


def get_store() -> CandleStore:
    """Get or initialize the global CandleStore singleton."""
    global _store_instance
    if _store_instance is None:
        _store_instance = CandleStore()
    return _store_instance
