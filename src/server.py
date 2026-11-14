"""
src/server.py – FastAPI web server and WebSocket broadcaster for the AI Trading Indicator.

Serves:
1. REST API endpoints (/api/candles, /api/status)
2. Live WebSocket (/ws) broadcasting real-time candle updates and model signals
3. Interactive frontend dashboard (web/index.html)
"""

import asyncio
import json
import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List, Optional, Set

import pandas as pd
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.live import stream_binance_klines
from src.predictor import get_predictor
from src.store import get_store

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("src.server")


class ConnectionManager:
    """Manages active browser WebSocket connections."""

    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        async with self._lock:
            self.active_connections.add(websocket)
        logger.info("Client connected. Total active connections: %d", len(self.active_connections))

    async def disconnect(self, websocket: WebSocket):
        async with self._lock:
            self.active_connections.discard(websocket)
        logger.info("Client disconnected. Remaining connections: %d", len(self.active_connections))

    async def broadcast(self, message: dict):
        if not self.active_connections:
            return
        payload = json.dumps(message)
        dead_connections = []
        async with self._lock:
            for connection in self.active_connections:
                try:
                    await connection.send_text(payload)
                except Exception:
                    dead_connections.append(connection)

            for dead in dead_connections:
                self.active_connections.discard(dead)


manager = ConnectionManager()


async def on_live_candle_received(candle: dict, signal: Optional[dict]):
    """
    Callback fired by live.py whenever a candle tick arrives or a candle closes.
    Broadcasts the update to all connected web clients.
    """
    open_time_sec = candle["open_time"] // 1000

    broadcast_payload = {
        "type": "candle_update",
        "candle": {
            "time": open_time_sec,
            "open": candle["open"],
            "high": candle["high"],
            "low": candle["low"],
            "close": candle["close"],
            "volume": candle["volume"],
            "is_closed": candle["is_closed"],
        },
        "signal": signal,
    }

    await manager.broadcast(broadcast_payload)


def fetch_recent_live_history(limit: int = 250) -> Optional[pd.DataFrame]:
    """
    Fetch the most recent candles directly from Binance REST API on startup.
    This ensures that historical chart candles seamlessly connect to the live WebSocket
    without any date or price gap.
    """
    try:
        import requests
        url = f"{config.REST_BASE_URL.rstrip('/')}/api/v3/klines"
        params = {"symbol": config.SYMBOL, "interval": config.INTERVAL, "limit": limit}
        resp = requests.get(url, params=params, timeout=5)
        if resp.status_code == 200:
            raw_klines = resp.json()
            if raw_klines:
                rows = []
                for k in raw_klines:
                    rows.append({
                        "open_time": int(k[0]),
                        "open": float(k[1]),
                        "high": float(k[2]),
                        "low": float(k[3]),
                        "close": float(k[4]),
                        "volume": float(k[5]),
                    })
                df = pd.DataFrame(rows)
                logger.info("Fetched %d fresh recent candles directly from Binance REST API.", len(df))
                return df
    except Exception as e:
        logger.warning("Could not fetch fresh live history via REST (%s). Falling back to local file.", e)
    return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup & shutdown lifespan event handler.
    Seeds the store with recent history and launches the live WebSocket client.
    """
    logger.info("Initializing CandleStore with historical data...")
    store = get_store()

    # 1. Try seeding with fresh live history first to prevent date/price gaps
    recent_df = fetch_recent_live_history(limit=250)
    if recent_df is not None and len(recent_df) >= 60:
        store.load_initial_history(recent_df, compute_historical_signals=True)
    else:
        parquet_file = config.CLEAN_CSV.with_suffix(".parquet")
        if parquet_file.exists():
            df_hist = pd.read_parquet(parquet_file)
            store.load_initial_history(df_hist, compute_historical_signals=True)
        elif config.CLEAN_CSV.exists():
            df_hist = pd.read_csv(config.CLEAN_CSV)
            store.load_initial_history(df_hist, compute_historical_signals=True)
        else:
            logger.warning("No pre-existing historical dataset found to seed store.")

    # Initialize model early
    try:
        get_predictor()
    except Exception as e:
        logger.warning("Predictor initialization warning: %s", e)

    # Launch Binance live streaming task in the background
    logger.info("Starting Binance live stream background task...")
    live_task = asyncio.create_task(stream_binance_klines(on_candle_callback=on_live_candle_received))

    yield  # Application runs here

    logger.info("Shutting down live stream...")
    live_task.cancel()
    try:
        await live_task
    except asyncio.CancelledError:
        pass


app = FastAPI(
    title="AI-Powered Trading Indicator Server",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for local web development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/candles")
async def get_candles(limit: int = 250):
    """Return historical candles with pre-computed signals for the chart."""
    store = get_store()
    data = store.get_chart_data(limit=limit)
    return {"status": "ok", "count": len(data), "candles": data}


@app.get("/api/status")
async def get_status():
    """Return live system status, model parameters, and latest signal."""
    store = get_store()
    latest_signal = store.get_latest_signal()
    chart_data = store.get_chart_data(limit=1)
    current_price = chart_data[-1]["close"] if chart_data else None

    predictor = None
    threshold = 0.60
    try:
        predictor = get_predictor()
        threshold = predictor.confidence_threshold
    except Exception:
        pass

    return {
        "status": "online",
        "symbol": config.SYMBOL,
        "interval": config.INTERVAL,
        "horizon_bars": config.HORIZON_BARS,
        "horizon_minutes": config.HORIZON_BARS * 5,
        "current_price": current_price,
        "confidence_threshold": threshold,
        "latest_signal": latest_signal,
        "total_candles_in_memory": store.count_candles(),
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket channel for live streaming chart updates to browser."""
    await manager.connect(websocket)
    try:
        # Send current status upon connection
        store = get_store()
        latest = store.get_latest_signal()
        await websocket.send_text(
            json.dumps(
                {
                    "type": "initial_handshake",
                    "symbol": config.SYMBOL,
                    "interval": config.INTERVAL,
                    "horizon_bars": config.HORIZON_BARS,
                    "latest_signal": latest,
                }
            )
        )

        while True:
            # Keep connection open; receive ping or client messages
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))
    except WebSocketDisconnect:
        await manager.disconnect(websocket)
    except Exception as e:
        logger.debug("WebSocket client error: %s", e)
        await manager.disconnect(websocket)


# Mount frontend static files
web_dir = PROJECT_ROOT / "web"
if web_dir.exists():
    app.mount("/", StaticFiles(directory=str(web_dir), html=True), name="web")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.server:app", host="127.0.0.1", port=8000, reload=False)
