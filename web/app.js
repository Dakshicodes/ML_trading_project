/**
 * app.js – Frontend live chart, WebSocket client & Live Log Sidebar for the AI Trading Indicator.
 * Integrates TradingView's Lightweight Charts library.
 */

// State
let chart = null;
let candleSeries = null;
let volumeSeries = null;
let resistanceTopLine = null;
let resistanceBottomLine = null;
let supportTopLine = null;
let supportBottomLine = null;
let markers = [];
let jsonHistoricalCandles = [];
let ws = null;
let reconnectTimer = null;
let logEntriesCount = 0;
let currentFilter = "all";
let lastLogTickTime = 0;

// DOM Elements
const currentPriceEl = document.getElementById("currentPrice");
const signalDirectionEl = document.getElementById("signalDirection");
const signalConfidenceEl = document.getElementById("signalConfidence");
const signalCardEl = document.getElementById("signalCard");
const connectionStatusEl = document.getElementById("connectionStatus");
const statusTextEl = document.getElementById("statusText");
const chartContainer = document.getElementById("tvChartContainer");
const logStreamEl = document.getElementById("logStream");
const logCountEl = document.getElementById("logCount");
const clearLogBtn = document.getElementById("clearLogBtn");
const filterTabs = document.querySelectorAll(".filter-tab");

/**
 * Format Date to HH:MM:SS string
 */
function formatTime(dateObj = new Date()) {
  return dateObj.toTimeString().split(" ")[0];
}

/**
 * Append entry to the Live Log Sidebar
 */
function addLogEntry(typeClass, tagText, tagClass, messageHtml, category = "all") {
  logEntriesCount++;
  logCountEl.textContent = `${logEntriesCount} events`;

  const entry = document.createElement("div");
  entry.className = `log-entry ${typeClass}`;
  entry.dataset.category = category;

  // Apply active filter visibility
  if (currentFilter !== "all" && category !== currentFilter && category !== "system") {
    entry.style.display = "none";
  }

  const timeStr = formatTime();
  entry.innerHTML = `
    <span class="log-time">${timeStr}</span>
    <span class="log-tag ${tagClass}">${tagText}</span>
    <span class="log-msg">${messageHtml}</span>
  `;

  logStreamEl.appendChild(entry);

  // Auto-scroll to bottom
  logStreamEl.scrollTop = logStreamEl.scrollHeight;

  // Keep memory light: retain max 150 entries
  while (logStreamEl.children.length > 150) {
    logStreamEl.removeChild(logStreamEl.firstChild);
  }
}

/**
 * Initialize TradingView Lightweight Charts instance
 */
function initChart() {
  chart = LightweightCharts.createChart(chartContainer, {
    layout: {
      background: { color: "#0f1318" },
      textColor: "#8b949e",
      fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
    },
    grid: {
      vertLines: { color: "#1e242d" },
      horzLines: { color: "#1e242d" },
    },
    crosshair: {
      mode: LightweightCharts.CrosshairMode.Normal,
    },
    rightPriceScale: {
      borderColor: "#27303c",
      autoScale: true,
      scaleMargins: {
        top: 0.1,
        bottom: 0.25,
      },
    },
    timeScale: {
      borderColor: "#27303c",
      timeVisible: true,
      secondsVisible: false,
      barSpacing: 12,                   // Healthy, readable initial candle width
      minBarSpacing: 6,                 // Prevents candles from collapsing into tiny needles!
      rightOffset: 10,                  // Comfortable breathing room on the right edge
      fixLeftEdge: false,
      shiftVisibleRangeOnNewBar: true,  // Automatically scrolls forward as new live candles form
    },
    handleScale: {
      mouseWheel: true,
      pinch: true,
      axisPressedMouseMove: true,
    },
    handleScroll: {
      mouseWheel: false,
      pressedMouseMove: true,
      horzTouchDrag: true,
    },
  });

  // Candlestick Series
  candleSeries = chart.addCandlestickSeries({
    upColor: "#0ecb81",
    downColor: "#f6465d",
    borderVisible: false,
    wickUpColor: "#0ecb81",
    wickDownColor: "#f6465d",
  });

  // Volume Histogram Series at the bottom
  volumeSeries = chart.addHistogramSeries({
    color: "#26a69a",
    priceFormat: {
      type: "volume",
    },
    priceScaleId: "",
    scaleMargins: {
      top: 0.8,
      bottom: 0,
    },
  });

  // Responsive resize
  window.addEventListener("resize", () => {
    if (chart && chartContainer) {
      chart.applyOptions({
        width: chartContainer.clientWidth,
        height: chartContainer.clientHeight,
      });
    }
  });
}

function updateSupportResistance(candles) {
  try {
    if (!candles || candles.length < 20 || !candleSeries) return;

    // Use the last 60 candles to find realistic swing highs and swing lows
    const recent = candles.slice(-60);
    let highestHigh = -Infinity;
    let highestClose = -Infinity;
    let lowestLow = Infinity;
    let lowestClose = Infinity;

    for (const c of recent) {
      if (c.high > highestHigh) highestHigh = c.high;
      if (c.close > highestClose) highestClose = c.close;
      if (c.low < lowestLow) lowestLow = c.low;
      if (c.close < lowestClose) lowestClose = c.close;
    }

    if (highestHigh === -Infinity || lowestLow === Infinity) return;

    const resTop = highestHigh;
    const resBottom = Math.max(highestClose, highestHigh * 0.995);
    const supBottom = lowestLow;
    const supTop = Math.min(lowestClose, lowestLow * 1.005);

    // If lines don't exist, create them once
    if (!resistanceTopLine) {
      resistanceTopLine = candleSeries.createPriceLine({
        price: resTop,
        color: "#f6465d",
        lineWidth: 2,
        lineStyle: LightweightCharts.LineStyle.Solid,
        axisLabelVisible: true,
        title: "RESISTANCE ZONE (TOP)",
      });
    } else {
      resistanceTopLine.applyOptions({ price: resTop });
    }

    if (!resistanceBottomLine) {
      resistanceBottomLine = candleSeries.createPriceLine({
        price: resBottom,
        color: "rgba(246, 70, 93, 0.6)",
        lineWidth: 1,
        lineStyle: LightweightCharts.LineStyle.Dashed,
        axisLabelVisible: false,
        title: "RESISTANCE (BASE)",
      });
    } else {
      resistanceBottomLine.applyOptions({ price: resBottom });
    }

    if (!supportTopLine) {
      supportTopLine = candleSeries.createPriceLine({
        price: supTop,
        color: "rgba(14, 203, 129, 0.6)",
        lineWidth: 1,
        lineStyle: LightweightCharts.LineStyle.Dashed,
        axisLabelVisible: false,
        title: "SUPPORT (CEILING)",
      });
    } else {
      supportTopLine.applyOptions({ price: supTop });
    }

    if (!supportBottomLine) {
      supportBottomLine = candleSeries.createPriceLine({
        price: supBottom,
        color: "#0ecb81",
        lineWidth: 2,
        lineStyle: LightweightCharts.LineStyle.Solid,
        axisLabelVisible: true,
        title: "SUPPORT ZONE (BOTTOM)",
      });
    } else {
      supportBottomLine.applyOptions({ price: supBottom });
    }
  } catch (err) {
    console.error("Error in updateSupportResistance:", err);
  }
}

// Indian Standard Time offset (UTC + 5 hours 30 minutes in seconds)
const TIMEZONE_OFFSET_SEC = 5.5 * 3600;

function toLocalTs(utcSec) {
  return utcSec + TIMEZONE_OFFSET_SEC;
}

/**
 * Update the Top Header with the latest signal and price
 */
function updateSignalUI(direction, confidence, price) {
  if (price !== undefined && price !== null) {
    currentPriceEl.textContent = `$${parseFloat(price).toFixed(2)}`;
  }

  signalCardEl.classList.remove("signal-up", "signal-down");

  if (direction === "Up") {
    signalDirectionEl.textContent = "▲ UP";
    signalConfidenceEl.textContent = `Confidence: ${(confidence * 100).toFixed(1)}%`;
    signalCardEl.classList.add("signal-up");
  } else if (direction === "Down") {
    signalDirectionEl.textContent = "▼ DOWN";
    signalConfidenceEl.textContent = `Confidence: ${(confidence * 100).toFixed(1)}%`;
    signalCardEl.classList.add("signal-down");
  } else {
    signalDirectionEl.textContent = "— NEUTRAL";
    signalConfidenceEl.textContent = "Confidence: Sideways Noise";
  }
}

/**
 * Build marker object for TradingView chart overlay
 */
function createMarker(time, direction, confidence) {
  if (direction === "Up") {
    return {
      time: time,
      position: "belowBar",
      color: "#0ecb81",
      shape: "arrowUp",
      text: `UP ${(confidence * 100).toFixed(0)}%`,
    };
  } else if (direction === "Down") {
    return {
      time: time,
      position: "aboveBar",
      color: "#f6465d",
      shape: "arrowDown",
      text: `DOWN ${(confidence * 100).toFixed(0)}%`,
    };
  }
  return null;
}

/**
 * Load initial historical candles via REST API
 */
async function loadHistoricalData() {
  try {
    addLogEntry("system", "INFO", "tag-system", "Loading initial historical data...", "system");
    const res = await fetch("/api/candles?limit=250");
    const json = await res.json();
    if (!json.candles || json.candles.length === 0) return;
    jsonHistoricalCandles = json.candles;

    const candleData = [];
    const volData = [];
    markers = [];

    let latestSig = null;
    let signalCount = 0;

    for (const c of json.candles) {
      const localTime = toLocalTs(c.time);
      candleData.push({
        time: localTime,
        open: c.open,
        high: c.high,
        low: c.low,
        close: c.close,
      });

      volData.push({
        time: localTime,
        value: c.volume,
        color: c.close >= c.open ? "rgba(14, 203, 129, 0.4)" : "rgba(246, 70, 93, 0.4)",
      });

      if (c.signal === "Up" || c.signal === "Down") {
        const m = createMarker(localTime, c.signal, c.confidence);
        if (m) markers.push(m);
        latestSig = { direction: c.signal, confidence: c.confidence };
        signalCount++;
      }
    }

    candleSeries.setData(candleData);
    volumeSeries.setData(volData);
    candleSeries.setMarkers(markers);
    updateSupportResistance(json.candles);

    const lastCandle = json.candles[json.candles.length - 1];
    if (latestSig) {
      updateSignalUI(latestSig.direction, latestSig.confidence, lastCandle.close);
    } else {
      updateSignalUI("Neutral", 0.0, lastCandle.close);
    }

    // Keep comfortable candle size and scroll to current live candle
    chart.timeScale().applyOptions({
      barSpacing: 12,
      rightOffset: 10,
    });
    chart.timeScale().scrollToRealTime();

    addLogEntry(
      "system",
      "LOADED",
      "tag-system",
      `Loaded <span class="highlight-val">${json.candles.length}</span> candles (<span class="highlight-val">${signalCount}</span> historical signals)`,
      "system"
    );
  } catch (err) {
    console.error("Failed to load historical candles:", err);
    addLogEntry("down-signal", "ERROR", "tag-down", `Failed to load history: ${err.message}`, "system");
  }
}

/**
 * Connect to live WebSocket for real-time streaming
 */
function connectWebSocket() {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const wsUrl = `${protocol}//${window.location.host}/ws`;

  statusTextEl.textContent = "CONNECTING";
  connectionStatusEl.classList.remove("connected");

  addLogEntry("system", "WS", "tag-system", `Connecting to WebSocket: ${wsUrl}...`, "system");

  ws = new WebSocket(wsUrl);

  ws.onopen = () => {
    statusTextEl.textContent = "LIVE";
    connectionStatusEl.classList.add("connected");
    addLogEntry("up-signal", "CONNECTED", "tag-up", "Live Binance kline stream active.", "system");
    if (reconnectTimer) clearTimeout(reconnectTimer);
  };

  ws.onmessage = (event) => {
    try {
      const msg = JSON.parse(event.data);

      if (msg.type === "candle_update") {
        const c = msg.candle;
        const localTime = toLocalTs(c.time);

        // Update candlestick and volume in real-time
        candleSeries.update({
          time: localTime,
          open: c.open,
          high: c.high,
          low: c.low,
          close: c.close,
        });

        volumeSeries.update({
          time: localTime,
          value: c.volume,
          color: c.close >= c.open ? "rgba(14, 203, 129, 0.4)" : "rgba(246, 70, 93, 0.4)",
        });

        currentPriceEl.textContent = `$${parseFloat(c.close).toFixed(2)}`;

        // Throttle tick logs to once every 3 seconds to avoid spamming the log box
        const now = Date.now();
        if (!c.is_closed && now - lastLogTickTime > 3000) {
          lastLogTickTime = now;
          addLogEntry(
            "tick-update",
            "TICK",
            "tag-tick",
            `Live ETH: <span class="highlight-val">$${c.close.toFixed(2)}</span> (Vol: ${c.volume.toFixed(1)})`,
            "candles"
          );
        }

        // When a candle CLOSES and a prediction signal is generated
        if (c.is_closed) {
          try {
            if (jsonHistoricalCandles && jsonHistoricalCandles.length > 0) {
              jsonHistoricalCandles.push(c);
              updateSupportResistance(jsonHistoricalCandles);
            }
          } catch (e) {
            console.error("Support/Resistance update error:", e);
          }
          addLogEntry(
            "candle-close",
            "CLOSED",
            "tag-candle",
            `5m Candle closed @ <span class="highlight-val">$${c.close.toFixed(2)}</span>. Evaluating model...`,
            "candles"
          );

          if (msg.signal) {
            const sig = msg.signal;
            updateSignalUI(sig.direction, sig.confidence, c.close);

            if (sig.direction === "Up") {
              const newMarker = createMarker(localTime, sig.direction, sig.confidence);
              if (newMarker) {
                markers.push(newMarker);
                candleSeries.setMarkers(markers);
              }
              addLogEntry(
                "up-signal",
                "SIGNAL UP",
                "tag-up",
                `▲ <span class="highlight-val">UP SIGNAL</span> locked: ${(sig.confidence * 100).toFixed(1)}% confidence (Horizon: +15m)`,
                "signals"
              );
            } else if (sig.direction === "Down") {
              const newMarker = createMarker(localTime, sig.direction, sig.confidence);
              if (newMarker) {
                markers.push(newMarker);
                candleSeries.setMarkers(markers);
              }
              addLogEntry(
                "down-signal",
                "SIGNAL DOWN",
                "tag-down",
                `▼ <span class="highlight-val">DOWN SIGNAL</span> locked: ${(sig.confidence * 100).toFixed(1)}% confidence (Horizon: +15m)`,
                "signals"
              );
            } else {
              addLogEntry(
                "neutral-signal",
                "NEUTRAL",
                "tag-neutral",
                `No directional edge (Conf: ${(sig.confidence * 100).toFixed(1)}% < 60%). Choppy chop avoided.`,
                "signals"
              );
            }
          }
        }
      }
    } catch (e) {
      console.error("Error processing WebSocket message:", e);
    }
  };

  ws.onclose = () => {
    statusTextEl.textContent = "RECONNECTING";
    connectionStatusEl.classList.remove("connected");
    addLogEntry("down-signal", "DISCONNECTED", "tag-down", "WebSocket dropped. Retrying in 3s...", "system");
    reconnectTimer = setTimeout(connectWebSocket, 3000);
  };

  ws.onerror = (err) => {
    console.error("WebSocket error:", err);
    ws.close();
  };
}

/**
 * Filter Tabs Logic
 */
filterTabs.forEach((tab) => {
  tab.addEventListener("click", () => {
    filterTabs.forEach((t) => t.classList.remove("active"));
    tab.classList.add("active");
    currentFilter = tab.dataset.filter;

    const entries = logStreamEl.querySelectorAll(".log-entry");
    entries.forEach((entry) => {
      const cat = entry.dataset.category;
      if (currentFilter === "all" || cat === currentFilter || cat === "system") {
        entry.style.display = "flex";
      } else {
        entry.style.display = "none";
      }
    });
  });
});

/**
 * Clear Log Button Logic
 */
clearLogBtn.addEventListener("click", () => {
  logStreamEl.innerHTML = "";
  logEntriesCount = 0;
  logCountEl.textContent = "0 events";
  addLogEntry("system", "CLEARED", "tag-system", "Log stream cleared.", "system");
});

// Initialize on page load
document.addEventListener("DOMContentLoaded", () => {
  initChart();
  loadHistoricalData();
  connectWebSocket();
});
