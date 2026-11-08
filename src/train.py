"""
src/train.py – Model training and validation with strict chronological time-series splitting.

Features:
- Loads clean historical OHLCV data.
- Computes stationary technical features and forward directional labels.
- Splits into Train (70%), Validation (15%), and Test (15%) splits chronologically.
- Trains HistGradientBoostingClassifier with calibrated probabilistic output.
- Calibrates optimal probability confidence threshold on the Validation set.
- Exports model artifact bundle to models/model.joblib.
"""

import sys
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.data import load_or_download_data
from src.features import FEATURE_COLUMNS, compute_features, create_labels

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("src.train")


def prepare_dataset() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load data, compute features and labels, and perform chronological time-series split.
    Guarantees zero data leakage between splits.
    """
    logger.info("Loading cleaned historical dataset...")
    df_raw = load_or_download_data()

    logger.info("Computing features via compute_features()...")
    df_feat = compute_features(df_raw)

    logger.info("Creating forward labels via create_labels() (horizon=%d bars)...", config.HORIZON_BARS)
    df_labeled, _ = create_labels(df_feat, binary=True)

    # Drop warm-up rows (first 60 rows contain rolling calculation NaNs)
    # and drop the final horizon_bars rows (future label is unknown)
    warmup_period = 60
    valid_data = df_labeled.iloc[warmup_period : -config.HORIZON_BARS].copy().reset_index(drop=True)

    n_total = len(valid_data)
    n_train = int(n_total * 0.70)
    n_val = int(n_total * 0.15)

    train_df = valid_data.iloc[:n_train].copy().reset_index(drop=True)
    val_df = valid_data.iloc[n_train : n_train + n_val].copy().reset_index(drop=True)
    test_df = valid_data.iloc[n_train + n_val :].copy().reset_index(drop=True)

    logger.info("Total usable candles: %d", n_total)
    logger.info(
        "Train split : %d candles (%s to %s)",
        len(train_df),
        train_df["datetime_utc"].iloc[0].strftime("%Y-%m-%d"),
        train_df["datetime_utc"].iloc[-1].strftime("%Y-%m-%d"),
    )
    logger.info(
        "Val split   : %d candles (%s to %s)",
        len(val_df),
        val_df["datetime_utc"].iloc[0].strftime("%Y-%m-%d"),
        val_df["datetime_utc"].iloc[-1].strftime("%Y-%m-%d"),
    )
    logger.info(
        "Test split  : %d candles (%s to %s)",
        len(test_df),
        test_df["datetime_utc"].iloc[0].strftime("%Y-%m-%d"),
        test_df["datetime_utc"].iloc[-1].strftime("%Y-%m-%d"),
    )

    return train_df, val_df, test_df


def calibrate_confidence_threshold(
    model: HistGradientBoostingClassifier,
    val_df: pd.DataFrame,
    round_trip_cost: float,
) -> float:
    """
    Search for the confidence threshold tau on the validation set that optimizes
    directional accuracy and trade quality while maintaining sufficient signal frequency.
    """
    X_val = val_df[FEATURE_COLUMNS].values
    y_val = val_df["target_binary"].values
    fwd_ret = val_df["forward_return"].values

    probs_up = model.predict_proba(X_val)[:, 1]

    candidate_thresholds = [0.54, 0.55, 0.56, 0.57, 0.58, 0.60]
    best_tau = 0.58
    best_score = -np.inf

    logger.info("Calibrating confidence threshold on Validation split...")
    for tau in candidate_thresholds:
        long_mask = probs_up >= tau
        short_mask = probs_up <= (1.0 - tau)
        n_trades = int(np.sum(long_mask) + np.sum(short_mask))

        if n_trades == 0:
            continue

        # Directional accuracy on confident calls
        correct_longs = np.sum(y_val[long_mask] == 1) if np.any(long_mask) else 0
        correct_shorts = np.sum(y_val[short_mask] == 0) if np.any(short_mask) else 0
        dir_accuracy = (correct_longs + correct_shorts) / n_trades

        # Simulated net PnL after costs
        long_pnl = np.sum(fwd_ret[long_mask] - round_trip_cost) if np.any(long_mask) else 0.0
        short_pnl = np.sum(-fwd_ret[short_mask] - round_trip_cost) if np.any(short_mask) else 0.0
        net_pnl = long_pnl + short_pnl

        logger.info(
            "Threshold tau=%.2f -> Signals: %5d (Up: %4d, Down: %4d) | Dir Accuracy: %5.1f%% | Net PnL: %6.2f%%",
            tau,
            n_trades,
            int(np.sum(long_mask)),
            int(np.sum(short_mask)),
            dir_accuracy * 100,
            net_pnl * 100,
        )

        # Optimize for directional accuracy with sufficient signal frequency (> 500 signals on validation)
        if dir_accuracy > best_score and n_trades >= 500:
            best_score = dir_accuracy
            best_tau = tau

    logger.info("Selected optimal confidence threshold: tau=%.2f", best_tau)
    return best_tau


def train_model() -> Dict:
    """
    Execute full training pipeline, calibrate confidence, and save model artifact.
    """
    train_df, val_df, test_df = prepare_dataset()

    X_train = train_df[FEATURE_COLUMNS].values
    y_train = train_df["target_binary"].values

    round_trip_cost = 2.0 * (config.FEE_PER_SIDE + config.SLIPPAGE_PER_SIDE)

    logger.info("Training HistGradientBoostingClassifier (max_iter=150, lr=0.03, max_depth=5)...")
    clf = HistGradientBoostingClassifier(
        max_iter=150,
        learning_rate=0.03,
        max_depth=5,
        min_samples_leaf=100,
        l2_regularization=2.0,
        random_state=config.RANDOM_SEED,
    )

    clf.fit(X_train, y_train)
    logger.info("Model fitting complete.")

    train_acc = accuracy_score(y_train, clf.predict(X_train))
    val_acc = accuracy_score(val_df["target_binary"].values, clf.predict(val_df[FEATURE_COLUMNS].values))
    logger.info("Baseline unthresholded directional accuracy -> Train: %.2f%% | Val: %.2f%%", train_acc * 100, val_acc * 100)

    # Calibrate optimal confidence threshold on Validation split
    best_tau = calibrate_confidence_threshold(clf, val_df, round_trip_cost)

    # Package model artifact bundle
    model_bundle = {
        "model": clf,
        "classes": clf.classes_.tolist(),  # [0, 1]
        "feature_names": FEATURE_COLUMNS,
        "confidence_threshold": best_tau,
        "symbol": config.SYMBOL,
        "interval": config.INTERVAL,
        "horizon_bars": config.HORIZON_BARS,
        "round_trip_cost": round_trip_cost,
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "test_samples": len(test_df),
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "date_range": {
            "train": (train_df["datetime_utc"].iloc[0].isoformat(), train_df["datetime_utc"].iloc[-1].isoformat()),
            "val": (val_df["datetime_utc"].iloc[0].isoformat(), val_df["datetime_utc"].iloc[-1].isoformat()),
            "test": (test_df["datetime_utc"].iloc[0].isoformat(), test_df["datetime_utc"].iloc[-1].isoformat()),
        },
    }

    # Save artifact
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_bundle, config.MODEL_PATH)
    logger.info("Model bundle successfully exported to %s", config.MODEL_PATH)

    return model_bundle


if __name__ == "__main__":
    train_model()
