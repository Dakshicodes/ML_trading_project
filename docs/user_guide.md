# AI Trading Indicator: Trader User Guide

**Target Asset**: Ethereum (`ETHUSDT`)  
**Chart Timeframe**: 5-Minute (`5m`)  
**Prediction Horizon**: 3 Candles (15 Minutes)  

---

## 1. Indicator Overview

This indicator uses a machine learning gradient-boosted decision tree ensemble trained on 24 months of Binance market data. On every closed 5-minute candle, it evaluates 22 technical indicators (momentum, volatility, moving average spreads, volume dynamics, and candlestick anatomy) to answer one question:

$$\textbf{Will ETHUSDT close higher or lower 15 minutes (3 candles) from now?}$$

---

## 2. How to Read the Chart Signals

The indicator displays three visual states on the TradingView chart:

| Visual on Chart | Meaning | Recommended Action |
| :--- | :--- | :--- |
| **▲ UP (Green Arrow)** | High probability of upward expansion over the next 15 minutes ($P(\text{Up}) \ge 60\%$). | Look for long / buy entries on the next open candle. |
| **▼ DOWN (Red Arrow)** | High probability of downward expansion over the next 15 minutes ($P(\text{Down}) \ge 60\%$). | Look for short / sell entries on the next open candle. |
| **No Marker (Neutral)** | Model confidence is below the $60\%$ hurdle; market is consolidating or choppy. | **DO NOT TRADE**. Stay in cash to avoid fee churn. |

### Understanding the Confidence Percentage
Every directional marker includes an exact confidence rating:
* **`60% – 63%` (Standard Edge)**: Valid signal. Take normal position size ($1\%$ account risk).
* **`64% – 68%+` (High Conviction)**: Statistically strongest setups. Historical win rate exceeds $62\% - 66\%$.
* **Below `60%`**: Suppressed as `Neutral` to protect your account from chop.

---

## 3. Recommended Trade Execution Strategy

1. **Entry Rule**:
   - **Wait for the candle to CLOSE**. Never enter on a forming / unclosed candle.
   - When the 5m candle closes and a green **`▲ UP`** or red **`▼ DOWN`** marker appears, enter at the open of the next candle.
2. **Horizon & Exit Rule**:
   - The indicator is trained specifically on a **3-bar (15-minute) horizon**.
   - Review or close the position at the end of 3 bars (15 minutes), or trail a stop once in profit.
3. **Stop-Loss Placement**:
   - Set an initial stop-loss just beyond the opposite wick of the signal candle (or at $1.0\times$ ATR).
   - Never trade without a hard stop-loss.

---

## 4. When NOT to Trust the Indicator

Machine learning indicators identify statistical patterns in technical price action. **They do not read the news.** You should pause or ignore indicator signals under the following market conditions:

### 4.1 Scheduled Macroeconomic News Releases
Technical patterns break down completely during major macroeconomic announcements. Do not trade 15 minutes before or after:
* **US Consumer Price Index (CPI)**
* **Federal Reserve FOMC Rate Decisions & Press Conferences**
* **Non-Farm Payrolls (NFP)**

### 4.2 Crypto Black Swan / Regulatory Events
During sudden external market shocks (e.g., exchange insolvencies, SEC regulatory announcements, smart contract exploits), technical indicators fail because price movement is driven entirely by panic liquidity liquidation.

### 4.3 Extremely Thin Holiday Liquidity
During major global holidays (e.g., December 24–26, New Year's Day), trading volume dries up and market makers widen their spreads. Low liquidity makes price prone to random wicks that trigger false breakouts.

---

## 5. Risk Management Rules

1. **Maximum Risk per Trade**: Never risk more than **$1.0\%$ to $1.5\%$** of your total trading capital on any single 15-minute setup.
2. **Beware of Exchange Fees**: In 5-minute crypto trading, taker fees ($0.10\%$ each side) and slippage accumulate rapidly. Avoid trading during `Neutral` signals.
3. **Zero Repainting**: Signals locked on closed candles are permanent. If a signal does not work out, accept the stop-loss and wait for the next high-conviction setup.
