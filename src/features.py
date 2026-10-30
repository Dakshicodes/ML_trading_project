"""
src/features.py – Feature engineering for the AI-Powered Trading Indicator.

Design rule
-----------
ONE public function, `compute_features(df)`, is the single source of truth for
all derived columns.  It is called identically by training (labels_splits.py /
train.py) and live inference (predictor.py).

Only past-to-current data is used; no look-ahead bias is permitted.
All features are computed from the closed candle at each row.

TODO (Step N): implement the feature calculations.
"""

import pandas as pd


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute technical features from raw OHLCV candles.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: open_time, open, high, low, close, volume.
        Rows must be sorted in ascending open_time order.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with additional feature columns appended.
        Rows with NaN features (warm-up period) are dropped.
    """
    raise NotImplementedError("Feature computation not implemented yet.")
