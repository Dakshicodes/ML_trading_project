"""
src/data.py – Historical market data ingestion and validation for Binance.

Fetches historical klines (candlesticks) from Binance public REST API,
validates and cleans OHLCV data, checks for missing gaps, and saves
clean datasets to the data/ directory.
"""

import sys
import time
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

# Add project root to sys.path if not present
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("src.data")

BINANCE_KLINE_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "trades",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
    "ignore",
]

CORE_COLUMNS = ["open_time", "open", "high", "low", "close", "volume", "trades"]


def fetch_klines_batch(
    session: requests.Session,
    symbol: str,
    interval: str,
    start_time: int,
    end_time: Optional[int] = None,
    limit: int = 1000,
    base_url: str = config.REST_BASE_URL,
    max_retries: int = 4,
    backoff_factor: float = 1.0,
) -> list:
    """
    Fetch a single batch of up to `limit` klines from Binance REST API with retry logic.
    """
    url = f"{base_url.rstrip('/')}/api/v3/klines"
    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": start_time,
        "limit": min(limit, 1000),
    }
    if end_time is not None:
        params["endTime"] = end_time

    for attempt in range(1, max_retries + 1):
        try:
            resp = session.get(url, params=params, timeout=15)
            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code == 429:
                wait_time = backoff_factor * (2 ** attempt)
                logger.warning("Rate limit hit (429). Backing off for %.1fs...", wait_time)
                time.sleep(wait_time)
            else:
                logger.warning(
                    "Unexpected HTTP %d: %s. Attempt %d/%d",
                    resp.status_code,
                    resp.text[:200],
                    attempt,
                    max_retries,
                )
                time.sleep(backoff_factor * attempt)
        except (requests.RequestException, ConnectionError) as exc:
            logger.warning("Network error (%s). Attempt %d/%d", exc, attempt, max_retries)
            time.sleep(backoff_factor * attempt)

    raise ConnectionError(f"Failed to fetch klines from Binance after {max_retries} attempts.")


def download_historical_klines(
    symbol: str = config.SYMBOL,
    interval: str = config.INTERVAL,
    history_months: int = config.HISTORY_MONTHS,
    base_url: str = config.REST_BASE_URL,
    interval_ms: int = config.INTERVAL_MS,
) -> pd.DataFrame:
    """
    Download complete historical klines spanning `history_months` up to the present.
    Paginates 1,000 candles at a time.
    """
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    # Estimate start time: history_months * 30 days in milliseconds
    start_time = now_ms - (history_months * 30 * 24 * 60 * 60 * 1000)

    logger.info(
        "Starting download for %s %s from %s to %s (~%d months)",
        symbol,
        interval,
        datetime.fromtimestamp(start_time / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        datetime.fromtimestamp(now_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        history_months,
    )

    all_rows = []
    current_start = start_time
    session = requests.Session()

    batch_count = 0
    while current_start < now_ms:
        batch = fetch_klines_batch(
            session=session,
            symbol=symbol,
            interval=interval,
            start_time=current_start,
            end_time=now_ms,
            limit=1000,
            base_url=base_url,
        )

        if not batch:
            logger.info("No further candles returned by API.")
            break

        all_rows.extend(batch)
        batch_count += 1

        last_open_time = batch[-1][0]
        # If the last candle is newer than or equal to current_start, advance to next candle
        if last_open_time < current_start:
            break
        current_start = last_open_time + interval_ms

        if batch_count % 20 == 0 or len(batch) < 1000:
            last_date_str = datetime.fromtimestamp(
                last_open_time / 1000, tz=timezone.utc
            ).strftime("%Y-%m-%d %H:%M")
            logger.info(
                "Batch %d: total %d candles fetched so far (up to %s)",
                batch_count,
                len(all_rows),
                last_date_str,
            )

        # Subtle polite pause to avoid exhausting rate limit
        time.sleep(0.05)

    if not all_rows:
        raise ValueError(f"No klines data retrieved for {symbol} ({interval}).")

    df = pd.DataFrame(all_rows, columns=BINANCE_KLINE_COLUMNS)
    logger.info("Raw download complete: %d rows fetched.", len(df))
    return df


def validate_and_clean_data(
    df: pd.DataFrame,
    interval_ms: int = config.INTERVAL_MS,
) -> pd.DataFrame:
    """
    Validate, sanitize, and cast raw kline data.
    - Standardizes column types.
    - Drops duplicates and ensures ascending order.
    - Checks high/low bounding sanity.
    - Identifies and logs missing timestamp gaps.
    """
    logger.info("Validating and cleaning raw data...")
    cleaned = df.copy()

    # Cast types
    cleaned["open_time"] = cleaned["open_time"].astype(np.int64)
    cleaned["close_time"] = cleaned["close_time"].astype(np.int64)
    cleaned["trades"] = cleaned["trades"].astype(np.int64)

    for col in ["open", "high", "low", "close", "volume"]:
        cleaned[col] = cleaned[col].astype(np.float64)

    # Keep only core columns
    cleaned = cleaned[CORE_COLUMNS]

    # Deduplicate by open_time
    duplicates = cleaned.duplicated(subset=["open_time"]).sum()
    if duplicates > 0:
        logger.warning("Found %d duplicate candle timestamps. Dropping duplicates.", duplicates)
        cleaned = cleaned.drop_duplicates(subset=["open_time"], keep="last")

    # Ensure strictly sorted by open_time
    cleaned = cleaned.sort_values("open_time").reset_index(drop=True)

    # Sanity checks on OHLC constraints
    invalid_hl = cleaned["high"] < cleaned["low"]
    invalid_ho = cleaned["high"] < cleaned["open"]
    invalid_hc = cleaned["high"] < cleaned["close"]
    invalid_lo = cleaned["low"] > cleaned["open"]
    invalid_lc = cleaned["low"] > cleaned["close"]

    invalid_mask = invalid_hl | invalid_ho | invalid_hc | invalid_lo | invalid_lc
    invalid_count = invalid_mask.sum()
    if invalid_count > 0:
        logger.warning(
            "Found %d candles violating basic OHLC bounds (high >= open,close,low). Dropping invalid rows.",
            invalid_count,
        )
        cleaned = cleaned[~invalid_mask].reset_index(drop=True)

    # Check for timestamp gaps
    time_diffs = cleaned["open_time"].diff()
    gaps = time_diffs[time_diffs > interval_ms]
    if not gaps.empty:
        total_missing_candles = int((gaps - interval_ms).sum() // interval_ms)
        logger.info(
            "Found %d gap intervals (totaling ~%d missing candles) across the timeline.",
            len(gaps),
            total_missing_candles,
        )
    else:
        logger.info("Timeline is perfectly continuous (zero gaps detected).")

    # Add human-readable UTC timestamp column for inspection convenience
    cleaned["datetime_utc"] = pd.to_datetime(cleaned["open_time"], unit="ms", utc=True)

    logger.info(
        "Cleaned data: %d rows from %s to %s.",
        len(cleaned),
        cleaned["datetime_utc"].iloc[0].strftime("%Y-%m-%d %H:%M:%S"),
        cleaned["datetime_utc"].iloc[-1].strftime("%Y-%m-%d %H:%M:%S"),
    )
    return cleaned


def save_data(
    raw_df: pd.DataFrame,
    clean_df: pd.DataFrame,
    raw_path: Path = config.RAW_CSV,
    clean_path: Path = config.CLEAN_CSV,
) -> None:
    """
    Save raw and cleaned datasets to disk, creating destination folders if needed.
    Also saves a compressed Parquet copy for fast load times.
    """
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    clean_path.parent.mkdir(parents=True, exist_ok=True)

    # Save raw CSV
    logger.info("Saving raw data to %s...", raw_path)
    raw_df.to_csv(raw_path, index=False)

    # Save clean CSV
    logger.info("Saving clean data to %s...", clean_path)
    clean_df.to_csv(clean_path, index=False)

    # Save clean Parquet for faster loading during training
    parquet_path = clean_path.with_suffix(".parquet")
    logger.info("Saving clean Parquet to %s...", parquet_path)
    clean_df.to_parquet(parquet_path, index=False)

    logger.info("All data successfully saved to disk.")


def load_or_download_data(
    force_download: bool = False,
    symbol: str = config.SYMBOL,
    interval: str = config.INTERVAL,
    history_months: int = config.HISTORY_MONTHS,
) -> pd.DataFrame:
    """
    Convenience loader: loads clean Parquet/CSV if already present, or downloads afresh.
    """
    parquet_path = config.CLEAN_CSV.with_suffix(".parquet")
    if not force_download:
        if parquet_path.exists():
            logger.info("Loading existing clean dataset from %s", parquet_path)
            return pd.read_parquet(parquet_path)
        elif config.CLEAN_CSV.exists():
            logger.info("Loading existing clean dataset from %s", config.CLEAN_CSV)
            return pd.read_csv(config.CLEAN_CSV)

    logger.info("Dataset not found locally or force_download=True. Downloading from Binance...")
    raw_df = download_historical_klines(
        symbol=symbol,
        interval=interval,
        history_months=history_months,
    )
    clean_df = validate_and_clean_data(raw_df)
    save_data(raw_df, clean_df)
    return clean_df


if __name__ == "__main__":
    logger.info("Starting historical data collection pipeline...")
    raw = download_historical_klines()
    clean = validate_and_clean_data(raw)
    save_data(raw, clean)
    logger.info("Historical data pipeline finished successfully.")
