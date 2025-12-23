# Kalshi Prediction Markets Trading Tools

## Trading Thesis (Based on Domer - World's #1 Prediction Markets Trader)

### Core Philosophy

> **"I'm more inclined to trade short-term swings, while casual players only care about the final outcome"**  
> — Domer, Odds On Open Podcast (Dec 18, 2025)

This toolkit is designed for **swing trading prediction markets**, not holding positions to maturity.

### Key Insights from Domer:

1. **Trade Price Movement, Not Final Outcomes**
   - Don't focus on pricing the market correctly
   - Focus on predicting day-to-day movements
   - Enter and exit positions based on short-term swings
   - You don't need to be right about the final outcome

2. **Information-Driven Repricing**
   - Prediction markets move on discrete news events, not continuous noise
   - Identify information-driven repricing before markets fully adjust
   - React faster than retail players who are focused on final outcomes

3. **Edge Lives in Micro Markets**
   - High-volume headline markets are efficient and crowded
   - Micro events and sub-events have lower liquidity and attention
   - Alternative markets and smaller categories prone to pricing errors
   - Less competition = more edge

4. **Retail vs Smart Money**
   - **Retail behavior**: Overreact to news, chase momentum, poor position sizing
   - **Smart money**: Patient capital, fade overreactions, scale with edge
   - Large traders can temporarily distort prices in thin markets
   - Sentiment shifts create tradeable swings

5. **Position Management**
   - Don't hold to resolution unless you have strong conviction
   - Scale position sizing with edge and uncertainty
   - Losses often come from crowded trades and poor sizing, not direction
   - Multiple small bets > one large bet

---

## Trading Strategy Framework

### What We're Looking For:

```
1. SENTIMENT SHIFTS
   - Sudden liquidity changes (walls appearing/disappearing)
   - Fill velocity spikes (aggressive buying/selling)
   - Price moves >5-10% in <1 hour

2. RETAIL OVERREACTIONS
   - Price spikes on news → FADE (mean reversion play)
   - Wide spreads + thin book → chaos → opportunity
   - Aggressive fills eating through book → emotion

3. SMART MONEY SIGNALS
   - Large passive orders appearing → patient capital
   - Tight spreads + deep book → efficient pricing
   - Slow accumulation while price stable → follow

4. ENTRY/EXIT TRIGGERS
   - ENTRY: Price dislocated from fair value (sentiment overshoot)
   - EXIT: Target hit (2-5 cent profit) OR liquidity drying up
   - STOP: Price moves against thesis (cut losses quickly)
```

### Example Trade Flow:

```
1. Market at 0.07 (7% probability)
2. News breaks → retail panic buys → price spikes to 0.12 in 10 min
3. SIGNAL: Price move too fast, no fundamental change
4. ACTION: Short YES at 0.12 (fade the overreaction)
5. TARGET: 0.08-0.09 (mean reversion)
6. EXIT: Price drops to 0.09 → +3 cent profit in 2 hours
7. NEVER CARE: What happens at final resolution
```

---

## Tools in This Directory

### 1. `monitor_kalshi_markets.py`
**Purpose**: Collect order book snapshots at regular intervals

**Usage**:
```bash
python scripts/kalshi/monitor_kalshi_markets.py
```

**What it does**:
- Fetches order book data from Kalshi API
- Saves timestamped snapshots to `data/04_output/prediction_markets/order_books/`
- Run every 1-5 minutes to build historical data
- Required for fill detection and momentum analysis

**Configuration**:
- Markets to monitor (add tickers manually)
- Polling frequency
- API credentials

---

### 2. `view_kalshi_order_book.py`
**Purpose**: Analyze order book structure and detect fills/momentum

**Usage**:
```bash
python scripts/kalshi/view_kalshi_order_book.py MARKET_TICKER
```

**Example**:
```bash
python scripts/kalshi/view_kalshi_order_book.py KXELONMARS-99
```

**What it shows**:

#### Order Book Distribution
- YES side (bids) - who's buying the event
- NO side (asks) - who's selling the event
- Market spread and best prices
- Liquidity gaps (where orders are missing)
- Complete 0.01-0.99 range with zero-filling

#### Fill Detection
- **Legitimate Liquidity**: Filters out "fake" orders too far from market
  - Dynamic threshold: `price ± min(5 cents, 20% of price)`
  - Example: At 0.07, only count orders in [0.05-0.09] range
  - Ignores wishful-thinking orders (e.g., 0.01 when market is 0.07)

- **Significant Fills**: Orders >3% of legit volume OR >250 contracts
  - Tracks what actually got filled vs just added
  - Shows momentum (who's hitting bids/asks aggressively)

- **Summary Stats**:
  - Volume changes (liquidity entering/exiting)
  - Price movements (best bid/ask shifts)
  - Market direction (bullish/bearish/neutral)

#### Distribution Stats
- Total unfilled limit orders per side
- Weighted average prices
- Largest orders (potential walls)
- Price concentration (tight vs spread out)

#### Order Book Evolution
- Compare last 3 snapshots
- Price changes over time
- Depth changes (liquidity growth/shrinkage)
- Imbalance shifts (which side gaining relative strength)

**Key Metrics**:
```
Legitimate Liquidity: YES [0.05-0.10] | NO [0.86-0.96]
  → Only these ranges matter for trading

Significant Fills: 500 contracts at 0.07 (7.7% of volume)
  → Real activity, not noise

Market Direction: ↗️ BULLISH
  → Price discovery higher, momentum signal
```

---

### 3. `find_kalshi_markets.py`
**Purpose**: Discover tradeable markets on Kalshi

**Usage**:
```bash
python scripts/kalshi/find_kalshi_markets.py
```

**What it does**:
- Lists available markets from Kalshi API
- Filters by category, volume, expiration
- Helps identify micro markets with edge potential
- Shows basic market info (volume, status, close time)

**Use cases**:
- Find new markets to monitor
- Identify low-attention events (edge opportunities)
- Check market status before trading

---

## Config Values (Tunable)

Located in `view_kalshi_order_book.py`:

```python
# Legitimate liquidity definition
MAX_SPREAD_CENTS = 0.05  # Maximum 5 cents from best bid
SPREAD_PCT = 0.20        # OR 20% of price (whichever is smaller)

# Fill significance thresholds  
FILL_THRESHOLD_PCT = 0.03  # 3% of legitimate volume
MIN_FILL_SIZE = 250        # Absolute minimum contracts to flag

# Display options
ALERT_ON_SIGNIFICANT = True  # Show 🔥 section for significant fills
LOG_ALL_FILLS = True         # Show 📊 section for complete record
```

**Why these matter**:
- `MAX_SPREAD_CENTS` / `SPREAD_PCT`: Defines what orders are "real" vs noise
  - Too tight → miss legitimate depth
  - Too wide → count fake orders
- `FILL_THRESHOLD_PCT`: What constitutes a "significant" move
  - Lower = more signals (noisy)
  - Higher = fewer signals (miss opportunities)
- `MIN_FILL_SIZE`: Absolute floor for alerting
  - Prevents noise in thin markets

---

## Data Flow

```
1. monitor_kalshi_markets.py (collect data)
   ↓
   Saves: data/04_output/prediction_markets/order_books/
          MARKET_TICKER_TIMESTAMP.json
   ↓
2. view_kalshi_order_book.py (analyze)
   ↓
   Reads: Last 2-3 snapshots
   ↓
   Outputs:
   - Order book structure
   - Fill detection
   - Momentum signals
   - Evolution over time
```

---

## Quick Start

### Step 1: Set up API credentials
```bash
# Add to your environment or config
KALSHI_API_KEY=your_key_here
KALSHI_API_SECRET=your_secret_here
```

### Step 2: Find markets to monitor
```bash
python scripts/kalshi/find_kalshi_markets.py
```

### Step 3: Start collecting data
```bash
# Run every 1-5 minutes (use cron or while loop)
python scripts/kalshi/monitor_kalshi_markets.py
```

### Step 4: Analyze order books
```bash
python scripts/kalshi/view_kalshi_order_book.py KXELONMARS-99
```

### Step 5: Look for signals
- Large fills in legitimate range → momentum
- Price spikes >10% → fade opportunity
- Liquidity drying up → exit signal
- Patient capital entering → follow signal

---

## Next Steps (TODO)

### Immediate:
- [ ] Add signal detection module (`detect_signals.py`)
- [ ] Implement momentum scoring (0-10 scale)
- [ ] Add real-time alerting (Slack/email notifications)

### Phase 2:
- [ ] Backtest framework (track hypothetical trades)
- [ ] Win rate tracking by signal type
- [ ] Round-trip P&L calculator
- [ ] Hold time analysis (optimal exit timing)

### Phase 3:
- [ ] Multi-market scanner (find best opportunities)
- [ ] Correlation analysis (hedge strategies)
- [ ] Automated position sizing recommendations
- [ ] Risk management dashboard

---

## Key Lessons from Domer

1. **"Don't try to price the market correctly - predict the next move"**
   - Edge is in forecasting short-term direction, not final outcomes

2. **"Casual players only care about final outcome"**
   - This is your advantage - you can exit anytime

3. **"Micro markets have more edge"**
   - Less attention = more pricing errors = more opportunity

4. **"Information-based trading, not noise"**
   - Markets move on discrete news, not random walks
   - React to repricing events before market fully adjusts

5. **"Position sizing > direction"**
   - Many losses from poor sizing, not being wrong
   - Scale with edge and uncertainty

---

## Reference

- **Podcast**: [Odds On Open - Domer on Trading Global Political Events](https://www.youtube.com/watch?v=TJpXnvFuvZg)
- **Transcript**: `transcripts/youtube/20251218 odds on podcast Domer on Trading Global Political Events.txt`
- **Platform**: [Kalshi](https://kalshi.com)
- **Documentation**: [ORDER_BOOK_QUANTIFICATION.md](../../docs/ORDER_BOOK_QUANTIFICATION.md)

---

## Notes

- These tools focus on **order book microstructure** (liquidity, fills, spreads)
- Not trying to model fundamentals or final probabilities
- Goal: Detect sentiment shifts and momentum for short-term trades
- Always remember: **You don't need to hold to maturity**

