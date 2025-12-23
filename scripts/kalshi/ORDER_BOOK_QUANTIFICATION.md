# Order Book Quantification Methodology

**Goal**: Create metrics that capture "tradeable" vs "thin/distorted" markets for short-term prediction market trading.

## Key Insights from Analysis

### What Matters Most for Trading

Based on analysis of Kalshi order books, the most important factors are:

1. **Weighted Depth** (25% of health score)
   - Liquidity near current price matters most
   - Use exponential decay: liquidity 5% away is worth less
   - Better predictor than raw depth

2. **Spread** (20% of health score)
   - Tight spread = efficient market
   - Wide spread = transaction costs eat profits
   - Critical for short-term trading

3. **Execution Quality** (15% of health score)
   - Slippage for typical order sizes (1000 shares)
   - Shows if you can actually execute without moving market
   - Low slippage = tradeable, high slippage = watch out

4. **Total Depth** (15% of health score)
   - Absolute liquidity available
   - <1K = very thin, >100K = deep

5. **Balance** (15% of health score)
   - One-sided books indicate manipulation/whales
   - 50/50 is ideal, 70/30 is warning sign

6. **Concentration** (10% of health score)
   - Large walls can be pulled quickly
   - One order >50% of depth = red flag

---

## Recommended Metrics to Track

### Primary Metrics (for alerts)

```python
# 1. Weighted Depth (decay_rate = 0.2)
weighted_depth = sum(size * (1 / (1 + 0.2 * distance)) for each level)

# 2. Spread
spread = abs(1.0 - best_yes_bid - best_no_bid)

# 3. Health Score (composite)
health_score = weighted_average([
    depth_score,
    weighted_depth_score,
    balance_score,
    spread_score,
    concentration_score,
    execution_score
])
```

### Alert Thresholds

**Price Movement**:
- >3% move in 10 minutes = HIGH alert
- >1% move = MEDIUM alert

**Liquidity**:
- Health score drops below 40 = HIGH alert (market degrading)
- Weighted depth drops >50% = HIGH alert
- Spread widens >2x = MEDIUM alert

**Order Book Structure**:
- Large wall appears (>10K shares single order) = INFO alert
- Imbalance shifts >0.15 = MEDIUM alert (one-sided pressure)
- Concentration >50% = HIGH alert (potential manipulation)

---

## Comparison: Two Markets

### KXELONMARS-99 (Health: 87.4)
```
✅ Excellent market quality
- Total depth: 75K shares
- Weighted depth: 35K (decay 0.2)
- Spread: 3%
- Imbalance: 0.59 (slight YES bias)
- Slippage for 1K shares: 0%
- Main issue: One wall at 21K shares (28% concentration)
```

### KXPERSONPRESFUENTES-45 (Health: 72.4)
```
✅ Good market quality
- Total depth: 73K shares (similar to above)
- Weighted depth: 14K (much less near-price liquidity!)
- Spread: 2% (tighter)
- Imbalance: 0.49 (perfectly balanced)
- Slippage for 1K shares: 24% ⚠️ (big problem)
- Main issue: Liquidity spread far from mid, hard to execute
```

**Key Difference**: Same total depth, but KXPERSONPRESFUENTES has liquidity far from mid price → worse execution.

This validates **weighted depth > raw depth** for trading decisions.

---

## Implementation Recommendations

### For Monitoring Script

Update `scripts/kalshi/monitor_kalshi_markets.py` to calculate:

1. **Health Score** - single number for quick assessment
2. **Weighted Depth (decay=0.2)** - track changes
3. **Slippage for 1K shares** - can we actually trade?

### Alert Priority

**HIGH (send email immediately)**:
- Price moves >3% in 10 min
- Health score <40
- Weighted depth drops >60%

**MEDIUM (log + daily summary email)**:
- Price moves 1-3%
- Health score 40-60
- Imbalance shifts >0.15
- Spread widens >2x

**INFO (log only)**:
- Large walls appear/disappear
- Normal volatility within healthy market

---

## Next Steps

### A. Integrate into Monitor ✅
Add health score calculation to `monitor_kalshi_markets.py`

### B. Refine Weights
After collecting more data:
- Are health score weights optimal?
- Should execution quality matter more/less?
- Adjust based on trading experience

### C. Market-Specific Baselines
Different markets have different "normal":
- Long-dated events (2099) may always be thin
- Near-term events should have tight books
- Track typical ranges per market type

### D. Predictive Metrics
Does health score predict:
- Price stability vs volatility?
- Probability of manipulation?
- Good entry/exit opportunities?

---

## Methodology Summary

**Question**: How to quantify order book quality?

**Answer**: Composite health score (0-100) combining:
- Liquidity amount (weighted by proximity to mid)
- Transaction costs (spread)
- Execution feasibility (slippage)
- Market structure (balance, concentration)

**Key Insight**: Raw depth misleading. Liquidity **location** matters more than amount.

**Validation**: Two markets with same total depth scored 87 vs 72 because liquidity distribution very different.

---

## Files Created

1. `scripts/kalshi/monitor_kalshi_markets.py` - Real-time monitoring
2. `scripts/kalshi/view_kalshi_order_book.py` - Visualize distribution
3. `scripts/kalshi/find_kalshi_markets.py` - Market discovery
4. `docs/PREDICTION_MARKETS_TRADING_TOOLS.md` - Overall strategy

**Data Outputs**:
- `data/04_output/prediction_markets/snapshots_summary.csv` - Time series metrics
- `data/04_output/prediction_markets/order_books/*.json` - Full order book snapshots

