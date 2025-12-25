# Where Market Making Meets Market Microstructure

**Speaker:** Sasha Stoikov  
**Event:** IAQF seminar, February 14, 2023  
**Video:** https://www.youtube.com/watch?v=S7eig5VXFpY&t=2287s

---

## TL;DR

Market making operates at two time scales: daily (macro) and millisecond (micro). The talk explores how optimal market making strategies change when you zoom into market microstructure details.

---

## Part 1: Market Making Fundamentals

### What is Market Making?

Algorithm that continuously updates bid/ask quotes to provide liquidity

### How It Works

- Start with fair price (mid price)
- Adjust for inventory: `Reservation Price = Mid - β×Q` (if long, lower price to incentivize selling)
- Post quotes: `Bid = R - δ` and `Ask = R + δ`
- **Key tradeoff:** Closer to mid = more trades but smaller profits per trade; farther = bigger spreads but fewer fills

### Mathematical Framework

- Hamilton-Jacobi-Bellman equations with Poisson arrival rates
- Spread has two components:
  - **Risk component:** Volatility-driven
  - **Microstructure component:** Competition-driven
- Inventory control is critical - aim to stay near zero position

---

## Part 2: Real-World Crypto Trading

### Hummingbot Experiment

Used open-source Avellaneda-Stoikov model on Bitcoin

**Results:**
- ✅ **Good:** Inventory stayed controlled
- ❌ **Bad:** Lost money overall (Bitcoin dropped + mid price is poor signal)

**Key insight:** Model works better for crypto than stocks due to tiny tick sizes creating "wiggle room" for quote positioning

---

## Part 3: Market Microstructure Insights

### Two Types of Markets

**Large tick:**
- Futures, liquid ETFs, low-price stocks
- Example: $10 stock with $0.01 spread = 10 basis points

**Small tick:**
- Crypto, high-price tech stocks
- Price grid is tiny relative to price

### Order Book Imbalance - The "Least Well-Kept Secret"

**Formula:** `Imbalance = BidSize / (BidSize + AskSize)`

- When imbalance → 1 (big bid, small ask), price likely moves **UP**
- **Paradox:** Everyone rushes to trade on the side with LESS liquidity
- This signal is robust and persistent across markets

### Micro Price Concept

- Better than mid price for "fair value"
- Adjusts mid based on order book imbalance
- For large spreads, top-of-book is less informative
- For Bank of America (1-tick spread):
  - Low imbalance: subtract ~0.5¢ from mid
  - High imbalance: add ~0.5¢ to mid

### VAMP (Volume Adjusted Mid Price)

Developed with Cornell Financial Engineering students

**Concept:**
- For crypto: weight prices by depth (e.g., $100k of liquidity)
- Creates wider "true spread" than displayed bid-ask
- Look deeper into order book, not just top level

**Results:**
- Showed promise initially vs mid-price-based MM
- Still vulnerable to volatility spikes and sudden price drops

---

## Key Takeaways

1. **Mid price is inadequate for market making** - Need microstructure signals (especially order book depth/imbalance) to estimate true fair value

2. **Tick size matters** - Small tick markets (crypto) allow for more flexible quote positioning; large tick markets (futures) require different strategies

3. **Order book imbalance is king** - In large tick markets, the ratio of bid size to total size is the most predictive single factor

4. **Market making remains risky** - "Picking up pennies in front of a tractor" - profitable until catastrophic price moves

5. **Volatility adjustment helps but isn't enough** - Real-time volatility estimates can widen spreads, but may not react fast enough to extreme moves

---

## Academic Context

### Key Papers Referenced

- **Ho & Stoll (1980s):** Original inventory problem framework for dealers/specialists
- **Avellaneda & Stoikov (2008):** Electronic market making with competing agents
- **Guéant et al. (2012):** Bounded inventory constraints
- **Cartea et al. (2014):** Price impact and adverse selection
- **Guilbaud & Pham (2017):** Multi-asset case (recommended starting point)
- Recent work: Stochastic volatility for crypto markets

### Open Source Implementation

**Hummingbot:** Open-source market making software widely used in crypto
- Implements Avellaneda-Stoikov and other strategies
- Adjustable parameters: risk aversion, spread, order size, update frequency
- Many crypto projects use it for initial liquidity provision

---

## Deep Dive: Most Interesting Insights

### 1. The Order Book Imbalance Paradox

**The Counterintuitive Discovery:**
When `Imbalance = BidSize / (BidSize + AskSize)` → 1 (e.g., bid size = 100, ask size = 1), everyone rushes to BUY on the thin ask side. The side with LESS liquidity is where the action happens.

**Why this matters:**
- Robust across ALL markets (stocks, futures, crypto)
- "Least well-kept secret in HFT" but doesn't disappear
- Structural feature, not an arbitrage opportunity

**Trading Strategy Implication:**
Instead of processing news, polls, or fundamental data yourself → **read the order book to see where smart money is positioning**. Let informed traders do the work, then follow their footprint via imbalance signals.

**Critical Questions for Implementation:**

1. **Signal Strength & Thresholds:**
   - What imbalance ratio is significant? (0.7? 0.8? 0.9?)
   - Does the absolute size matter? (bid=10/ask=1 vs bid=1000/ask=100 - both have same ratio but different implications)
   - Should we look at $ value of imbalance rather than contract count?

2. **Timing & Decay:**
   - How quickly does this signal decay in Kalshi markets?
   - If you see imbalance at 10am, is it still predictive at 10:05am? 11am? 
   - Do you need to act immediately or can you wait for better pricing?
   - What's the half-life of the information?

3. **Smart Money vs Dumb Money:**
   - How do you distinguish informed positioning from:
     - Retail herding (everyone piling in after seeing Twitter/news)?
     - Single whale with bad information?
     - Bot/algo that's just broken?
   - Can you filter by order characteristics (size, timing, fill behavior)?

4. **Market Lifecycle:**
   - Does imbalance matter more when:
     - Market just opened (establishing consensus)?
     - Near resolution time (last-minute informed traders)?
     - After major news (smart money reacting faster)?
   - Or is it constant signal strength?

5. **Cross-Market Confirmation:**
   - If you see imbalance on "Trump wins Iowa" → do you also check:
     - "Trump wins GOP nomination"?
     - Other correlated markets?
   - Is confirmation across correlated markets stronger signal?

6. **Position Sizing Based on Conviction:**
   - Stronger imbalance = larger position?
   - Or binary: either trade or don't?
   - How do you scale bet size with signal strength?

7. **Spoofing & Manipulation:**
   - In Kalshi's relatively thin markets, how easy is it to fake imbalance?
   - Can you detect when orders are pulled quickly (spoof indicators)?
   - Do you wait for actual fills to confirm real positioning vs just quotes?

8. **Historical Backtesting:**
   - Can you pull historical order book data from Kalshi?
   - What's the hit rate: when imbalance > X, market moves that direction Y% of time?
   - What's the expected value per signal?

---

### 2. Mid Price is a Lie (Especially in Crypto)

**The Problem:**
The displayed bid-ask spread in crypto is "unnaturally tight" - it doesn't reflect the true cost of trading meaningful size. Market makers following the mid price lose money.

**VAMP Solution (Volume Adjusted Mid Price):**
- Weight prices by depth to execute $100k (or meaningful size)
- Creates a much wider "true spread" than top-of-book
- Better predictor of future price movement (1-60 second timeframes)

**Trading Strategy Implication:**
Don't trust the top-of-book spread. Look deeper to see where the REAL money is positioned. This reveals true support/resistance levels where smart money is willing to deploy size.

**Critical Questions for Implementation:**

1. **Depth as Smart Money Signal:**
   - If you see tight spread (70-71) but then massive bids at 65, is that:
     - Smart money's true valuation (willing to buy a lot at 65)?
     - Support level they're defending?
     - Or just stale orders they forgot to cancel?
   - Does depth profile reveal conviction better than top-of-book?

2. **Kalshi-Specific Depth Analysis:**
   - What does a typical Kalshi order book look like?
     - How many price levels have liquidity?
     - Average order size at each level?
     - How quickly does liquidity fall off as you go deeper?
   - Are there "clusters" of orders at psychologically important prices (50¢, 75¢, etc.)?

3. **Cumulative Depth Imbalance:**
   - Instead of just top-of-book ratio, calculate:
     - `Total bid liquidity (all levels) / Total ask liquidity`
   - Does this deeper imbalance predict better than top-of-book?
   - How many levels deep should you look? (5 levels? 10? Until liquidity drops to negligible?)

4. **Dynamic Depth Changes:**
   - When large orders suddenly appear deep in the book:
     - Is someone setting a trap (stop-loss zone)?
     - Is someone showing conviction (willing to deploy size)?
   - When orders get pulled from depth:
     - Is support/resistance weakening?
     - Is smart money repositioning?

5. **True Cost of Trading Size:**
   - If you want to buy $10k worth, what's your TRUE average price?
   - Calculate VAMP for YOUR typical position size
   - Does this change your entry/exit decisions?

6. **Window Dressing Detection:**
   - In Kalshi, can you detect tiny "show orders" at best bid/ask?
   - If top-of-book is 100 contracts but next level is 10,000 contracts:
     - Is someone trying to make spread look tight?
     - Or is that 10k the real market?

7. **Order Book Shape as Signal:**
   - **Bullish shape:** Deep bids stacked below, thin asks above
   - **Bearish shape:** Thin bids below, deep asks stacked above
   - **Uncertain shape:** Thin on both sides (no conviction)
   - Can you classify book shape and trade accordingly?

8. **Integration with Imbalance:**
   - Combine BOTH signals:
     - Top-of-book imbalance (immediate directional signal)
     - Depth profile (conviction/support levels)
   - Strongest trades: when both align (imbalanced top + deep support in same direction)

9. **Historical Pattern Recognition:**
   - Before major market moves in Kalshi, what did order book look like?
   - Can you identify characteristic patterns that precede:
     - Big price jumps?
     - Market resolution becoming clearer?
     - Smart money accumulation?

