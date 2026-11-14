"""
src/evaluate.py – Comprehensive evaluation and baseline comparison on unseen test data.

Compares our AI Model against mandatory baselines required by the brief:
1. Always Up Baseline (Naive long bias)
2. Traditional Technical Indicator Rule (RSI 14 oversold < 30 / overbought > 70)
3. Logistic Regression Baseline (Linear L2-regularized model)
4. AI Indicator Model (HistGradientBoosting with calibrated probability filter)

Calculates directional accuracy, win rates, and simulated PnL accounting for
round-trip fees (0.10% * 2) and slippage (0.02% * 2).
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.features import FEATURE_COLUMNS
from src.train import prepare_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("src.evaluate")


def simulate_strategy_performance(
    signals: np.ndarray,
    forward_returns: np.ndarray,
    actual_binary_targets: np.ndarray,
    round_trip_cost: float,
) -> Dict:
    """
    Simulate trading metrics given an array of signals (-1: Short, 0: Neutral, 1: Long).
    Only non-zero signals generate trades and incur transaction costs.
    """
    long_mask = signals == 1
    short_mask = signals == -1
    active_mask = long_mask | short_mask
    total_trades = int(np.sum(active_mask))

    if total_trades == 0:
        return {
            "total_trades": 0,
            "win_rate": 0.0,
            "directional_accuracy": 0.0,
            "net_cumulative_return_pct": 0.0,
            "gross_cumulative_return_pct": 0.0,
            "avg_trade_net_return_pct": 0.0,
            "profit_factor": 0.0,
            "max_drawdown_pct": 0.0,
        }

    # Gross returns per trade
    trade_gross_returns = np.zeros(len(signals), dtype=np.float64)
    trade_gross_returns[long_mask] = forward_returns[long_mask]
    trade_gross_returns[short_mask] = -forward_returns[short_mask]

    # Net returns deducting round-trip fees & slippage
    trade_net_returns = np.zeros(len(signals), dtype=np.float64)
    trade_net_returns[active_mask] = trade_gross_returns[active_mask] - round_trip_cost

    active_net_returns = trade_net_returns[active_mask]
    winning_trades = np.sum(active_net_returns > 0)
    win_rate = winning_trades / total_trades

    # Directional accuracy: did the price move in the predicted direction?
    correct_longs = np.sum(actual_binary_targets[long_mask] == 1)
    correct_shorts = np.sum(actual_binary_targets[short_mask] == 0)
    directional_accuracy = (correct_longs + correct_shorts) / total_trades

    # Cumulative equity curve
    cumulative_equity = np.cumsum(active_net_returns)
    net_cumulative_return = np.sum(active_net_returns)
    gross_cumulative_return = np.sum(trade_gross_returns[active_mask])

    # Profit factor (gross profits / gross losses)
    gains = active_net_returns[active_net_returns > 0].sum()
    losses = np.abs(active_net_returns[active_net_returns < 0].sum())
    profit_factor = gains / (losses + 1e-10)

    # Max Drawdown calculation on trade-by-trade equity
    running_max = np.maximum.accumulate(cumulative_equity)
    drawdowns = running_max - cumulative_equity
    max_drawdown = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0

    return {
        "total_trades": total_trades,
        "long_trades": int(np.sum(long_mask)),
        "short_trades": int(np.sum(short_mask)),
        "directional_accuracy": float(directional_accuracy * 100),
        "win_rate": float(win_rate * 100),
        "gross_cumulative_return_pct": float(gross_cumulative_return * 100),
        "net_cumulative_return_pct": float(net_cumulative_return * 100),
        "avg_trade_net_return_pct": float((net_cumulative_return / total_trades) * 100),
        "profit_factor": float(profit_factor),
        "max_drawdown_pct": float(max_drawdown * 100),
    }


def evaluate_all() -> Dict:
    """
    Run full benchmark comparison on the unseen Test partition.
    """
    if not config.MODEL_PATH.exists():
        raise FileNotFoundError(f"Trained model not found at {config.MODEL_PATH}. Run src/train.py first.")

    logger.info("Loading trained model artifact from %s...", config.MODEL_PATH)
    bundle = joblib.load(config.MODEL_PATH)
    clf = bundle["model"]
    tau = bundle["confidence_threshold"]
    round_trip_cost = bundle["round_trip_cost"]

    train_df, val_df, test_df = prepare_dataset()
    logger.info("Evaluating on unseen Test set (%d candles)...", len(test_df))

    X_train = train_df[FEATURE_COLUMNS].values
    y_train = train_df["target_binary"].values
    X_test = test_df[FEATURE_COLUMNS].values
    y_test = test_df["target_binary"].values
    fwd_returns_test = test_df["forward_return"].values

    results = {}

    # -------------------------------------------------------------
    # 1. Baseline: Always Up (Buy & Hold forward horizon)
    # -------------------------------------------------------------
    signals_always_up = np.ones(len(test_df), dtype=np.int64)
    results["1_Always_Up_Baseline"] = simulate_strategy_performance(
        signals_always_up, fwd_returns_test, y_test, round_trip_cost
    )

    # -------------------------------------------------------------
    # 2. Baseline: Traditional RSI-14 Rule (Oversold < 30 -> Long, Overbought > 70 -> Short)
    # -------------------------------------------------------------
    rsi_test = test_df["rsi_14"].values
    signals_rsi = np.zeros(len(test_df), dtype=np.int64)
    signals_rsi[rsi_test < 30.0] = 1
    signals_rsi[rsi_test > 70.0] = -1
    results["2_RSI_14_Rule_Baseline"] = simulate_strategy_performance(
        signals_rsi, fwd_returns_test, y_test, round_trip_cost
    )

    # -------------------------------------------------------------
    # 3. Baseline: Logistic Regression (L2 Regularized with Scaler)
    # -------------------------------------------------------------
    logger.info("Fitting Logistic Regression linear baseline...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    lr = LogisticRegression(max_iter=500, random_state=config.RANDOM_SEED)
    lr.fit(X_train_scaled, y_train)

    lr_probs_up = lr.predict_proba(X_test_scaled)[:, 1]
    signals_lr = np.zeros(len(test_df), dtype=np.int64)
    signals_lr[lr_probs_up >= tau] = 1
    signals_lr[lr_probs_up <= (1.0 - tau)] = -1

    results["3_Logistic_Regression_Baseline"] = simulate_strategy_performance(
        signals_lr, fwd_returns_test, y_test, round_trip_cost
    )

    # -------------------------------------------------------------
    # 4. Proposed AI Model: HistGradientBoosting with Calibrated Filter
    # -------------------------------------------------------------
    logger.info("Evaluating AI Model (HistGradientBoosting, tau=%.2f)...", tau)
    probs_up = clf.predict_proba(X_test)[:, 1]

    signals_ai = np.zeros(len(test_df), dtype=np.int64)
    signals_ai[probs_up >= tau] = 1
    signals_ai[probs_up <= (1.0 - tau)] = -1

    results["4_AI_Indicator_Model"] = simulate_strategy_performance(
        signals_ai, fwd_returns_test, y_test, round_trip_cost
    )

    # -------------------------------------------------------------
    # Print formatted comparison table
    # -------------------------------------------------------------
    summary_rows = []
    for model_name, m in results.items():
        clean_name = model_name[2:].replace("_", " ")
        summary_rows.append(
            {
                "Strategy": clean_name,
                "Trades": m["total_trades"],
                "Directional Accuracy": f"{m['directional_accuracy']:.1f}%",
                "Win Rate (after fees)": f"{m['win_rate']:.1f}%",
                "Gross Return": f"{m['gross_cumulative_return_pct']:+.2f}%",
                "Net Return (after fees)": f"{m['net_cumulative_return_pct']:+.2f}%",
                "Profit Factor": f"{m['profit_factor']:.2f}",
                "Max Drawdown": f"{m['max_drawdown_pct']:.2f}%",
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    print("\n" + "=" * 105)
    print(f"OUT-OF-SAMPLE TEST RESULTS (ETHUSDT 5m, {len(test_df):,} candles, Horizon: 15 min)")
    print(f"Confidence Threshold: {tau*100:.0f}% | Round-trip transaction costs: {round_trip_cost*100:.2f}%")
    print("=" * 105)
    print(summary_df.to_string(index=False))
    print("=" * 105 + "\n")

    # -------------------------------------------------------------
    # Granular Breakdown across Confidence Thresholds (τ)
    # -------------------------------------------------------------
    breakdown_rows = []
    for t_cand in [0.52, 0.55, 0.58, 0.60, 0.62, 0.65]:
        longs = probs_up >= t_cand
        shorts = probs_up <= (1.0 - t_cand)
        n_up = int(np.sum(longs))
        n_down = int(np.sum(shorts))
        n_tot = n_up + n_down
        if n_tot > 0:
            up_acc = (np.sum(y_test[longs] == 1) / n_up * 100) if n_up > 0 else 0.0
            down_acc = (np.sum(y_test[shorts] == 0) / n_down * 100) if n_down > 0 else 0.0
            comb_acc = ((np.sum(y_test[longs] == 1) + np.sum(y_test[shorts] == 0)) / n_tot * 100)
            gross_pnl = (np.sum(fwd_returns_test[longs]) - np.sum(fwd_returns_test[shorts])) * 100
            breakdown_rows.append(
                {
                    "Confidence (τ)": f"≥ {t_cand*100:.0f}%",
                    "Total Signals": n_tot,
                    "Long Signals": n_up,
                    "Short Signals": n_down,
                    "Long Precision": f"{up_acc:.1f}%",
                    "Short Precision": f"{down_acc:.1f}%",
                    "Overall Precision": f"{comb_acc:.1f}%",
                    "Gross Return": f"{gross_pnl:+.2f}%",
                }
            )

    breakdown_df = pd.DataFrame(breakdown_rows)
    print("=" * 105)
    print("PRECISION & ACCURACY BY CONFIDENCE LEVEL (OUT-OF-SAMPLE TEST SET)")
    print("=" * 105)
    print(breakdown_df.to_string(index=False))
    print("=" * 105 + "\n")

    # Save to reports/
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = config.REPORTS_DIR / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(
            {
                "evaluated_at": bundle["trained_at_utc"],
                "symbol": config.SYMBOL,
                "interval": config.INTERVAL,
                "horizon_bars": config.HORIZON_BARS,
                "round_trip_cost_pct": round_trip_cost * 100,
                "test_candles": len(test_df),
                "confidence_threshold": tau,
                "results": results,
            },
            f,
            indent=2,
        )
    logger.info("Saved metrics to %s", metrics_path)

    # Save text report table
    report_txt_path = config.REPORTS_DIR / "evaluation_summary.txt"
    with open(report_txt_path, "w") as f:
        f.write(summary_df.to_string(index=False))

    return results


if __name__ == "__main__":
    evaluate_all()
