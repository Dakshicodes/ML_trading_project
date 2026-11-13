"""
src/live.py – Live market data streaming from Binance WebSocket.

Connects to the public Binance WebSocket kline stream (wss://stream.binance.com:9443),
parses real-time 5-minute candle updates, pushes them into CandleStore, and triggers
real-time prediction when a candle closes.
"""

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Callable, Optional

import websockets

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.store import get_store

logger = logging.getLogger("src.live")

# Public Binance WebSocket base URL
BINANCE_WS_URL = "wss://stream.binance.com:9443/ws"


def get_kline_stream_url(symbol: str = config.SYMBOL, interval: str = config.INTERVAL) -> str:
    """Generate the Binance stream URL for the symbol and interval."""
    stream_name = f"{symbol.lower()}@kline_{interval}"
    return f"{BINANCE_WS_URL}/{stream_name}"


async def stream_binance_klines(
    on_candle_callback: Optional[Callable] = None,
    symbol: str = config.SYMBOL,
    interval: str = config.INTERVAL,
    max_reconnect_attempts: int = 100,
) -> None:
    """
    Asynchronously listen to the Binance kline stream.
    Re-establishes connection automatically if network drops.
    """
    url = get_kline_stream_url(symbol, interval)
    store = get_store()
    backoff = 1.0

    logger.info("Connecting to Binance WebSocket stream: %s", url)

    attempt = 0
    while attempt < max_reconnect_attempts:
        try:
            async with websockets.connect(url, ping_interval=20, ping_timeout=20) as ws:
                logger.info("Connected to Binance live kline stream for %s %s.", symbol, interval)
                backoff = 1.0  # Reset backoff on successful connection
                attempt = 0

                async for message in ws:
                    data = json.loads(message)
                    k = data.get("k", {})
                    if not k:
                        continue

                    # Parse Binance kline payload
                    is_closed = bool(k.get("x", False))
                    candle = {
                        "open_time": int(k["t"]),
                        "open": float(k["o"]),
                        "high": float(k["h"]),
                        "low": float(k["l"]),
                        "close": float(k["c"]),
                        "volume": float(k["v"]),
                        "is_closed": is_closed,
                    }

                    # Push candle to in-memory store
                    signal = store.on_candle_update(candle)

                    # Dispatch to external callback (e.g. FastAPI WebSocket broadcaster)
                    if on_candle_callback:
                        try:
                            if asyncio.iscoroutinefunction(on_candle_callback):
                                await on_candle_callback(candle, signal)
                            else:
                                on_candle_callback(candle, signal)
                        except Exception as cb_err:
                            logger.error("Error in on_candle_callback: %s", cb_err)

        except (websockets.ConnectionClosed, ConnectionError, OSError) as e:
            attempt += 1
            logger.warning(
                "WebSocket connection interrupted (%s). Reconnecting in %.1fs (attempt %d)...",
                e,
                backoff,
                attempt,
            )
            await asyncio.sleep(backoff)
            backoff = min(backoff * 1.5, 30.0)
        except Exception as e:
            attempt += 1
            logger.error("Unexpected error in live stream: %s. Retrying in %.1fs...", e, backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 1.5, 30.0)


def start_live_feed_thread(on_candle_callback: Optional[Callable] = None):
    """Run the live feed in a separate background event loop thread."""
    import threading

    def runner():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(stream_binance_klines(on_candle_callback=on_candle_callback))

    t = threading.Thread(target=runner, daemon=True, name="BinanceLiveFeedThread")
    t.start()
    return t


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger.info("Starting standalone live feed test...")

    def print_callback(candle, signal):
        status = "CLOSED" if candle["is_closed"] else "TICK"
        sig_str = f" | Signal: {signal['direction']} ({signal['confidence']*100:.1f}%)" if signal else ""
        print(f"[{status}] ETH: ${candle['close']:.2f}{sig_str}")

    asyncio.run(stream_binance_klines(on_candle_callback=print_callback))
