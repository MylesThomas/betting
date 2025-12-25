# Overreaction Score Scaling Strategy

## Problem Statement

The current overreaction score uses **absolute thresholds** that work for mid-sized markets but won't scale to the full spectrum of prediction markets:

- **Presidential election**: 10M+ volume, 10,000+ contracts/min is normal
- **Mid-sized market**: 1-10M volume, 2,000 contracts/min is high activity
- **Micro event**: 50k volume, 100 contracts/min is huge

Hard-coded thresholds like `FILL_VELOCITY_HIGH = 2000` will:
- **False positives** on large markets (flag normal activity as overreaction)
- **False negatives** on small markets (miss actual panic)

---

## Solution: Build It Right Once

**Decision**: Skip Phase 2 (volume-normalized), go straight to Phase 3 (baselines)

**Rationale**:
- Phase 2 is a compromise - not accurate, just "better than nothing"
- Why build something temporary that gets thrown away?
- Phase 3 is the right architecture - build it once, build it right
- 48h wait before alerts is GOOD (no false positives from uncalibrated thresholds)

**Strategy**: Phase 3 with Phase 1 fallback during calibration

---

## Implementation Approach

### Phase 1: Absolute Thresholds (Calibration Mode) ✅

**What**: Hardcoded config values calibrated for mid-sized markets

**Implementation**:
```python
FILL_VELOCITY_HIGH = 2000  # contracts/min
AGGRESSION_PANIC = 0.75     # 75% aggressive
SPREAD_CHAOS = 3.0          # 3x widening
DEPTH_DRAIN = 0.6           # 60% remaining
```

**Pros**:
- ✅ Simple to implement
- ✅ Works immediately (no historical data needed)
- ✅ Good for POC testing

**Cons**:
- ❌ Only accurate for mid-sized markets (~1-10M volume)
- ❌ Will fail on presidential markets or micro events
- ❌ Not self-calibrating

**Status**: Complete  
**Use Case**: **Fallback mode during calibration** (when <48h of data)

**New Role**:
- Used while collecting baseline data
- Scores displayed but marked as "CALIBRATING"
- **NO ALERTS** sent during this phase
- Shows progress: "12h / 48h collected"

---

### Phase 2: Volume-Normalized Metrics ❌ (SKIPPED)

**What**: Normalize all metrics by market volume (no bucketing needed!)

**Key Insight**: Express everything as **% of total market volume** instead of absolute numbers

**Implementation**:

```python
def calculate_overreaction_score(fill_data, market_volume):
    """
    All thresholds normalized by market_volume.
    Automatically scales to ANY market size.
    """
    
    # 1. Normalize fill velocity (contracts/min → % of market/min)
    velocity_absolute = contracts_filled / time_minutes
    velocity_pct = (velocity_absolute / market_volume) * 100
    
    if velocity_pct > 0.5:   # >0.5% of market per minute = panic
        velocity_score = 3
    elif velocity_pct > 0.2:  # >0.2% = high activity
        velocity_score = 2
    elif velocity_pct < 0.05: # <0.05% = very orderly (underreaction)
        velocity_score = -1
    else:
        velocity_score = 1
    
    # 2. Normalize depth (contracts → % of market in book)
    depth_pct = (legitimate_depth / market_volume) * 100
    
    if depth_pct < 1.0:     # <1% of market in book = thin
        depth_score = 2      # Easier to manipulate
    elif depth_pct > 5.0:   # >5% of market in book = deep
        depth_score = 0      # Stable
    else:
        depth_score = 1
    
    # 3. Dynamic fill threshold (% of market)
    min_fill_size = market_volume * 0.0025  # 0.25% of market
    # 100k market → 25 contracts
    # 1M market → 250 contracts
    # 10M market → 2,500 contracts
    
    # 4. Keep relative metrics unchanged (already 0-1)
    aggression_ratio = filled / (filled + added)  # Already normalized
    spread_widening = curr_spread / prev_spread   # Already relative
    depth_change = curr_depth / prev_depth        # Already relative
```

**Example: Why This Works**

```python
# Market A: Presidential election (10M volume)
- 5,000 contracts/min filled
- Velocity: 5000 / 10_000_000 = 0.05% per min → NORMAL (score: 1)
- Depth: 50k in book = 0.5% of market → THIN (score: 2)

# Market B: Micro event (100k volume)  
- 500 contracts/min filled (10x less absolute!)
- Velocity: 500 / 100_000 = 0.50% per min → PANIC! (score: 3)
- Depth: 5k in book = 5.0% of market → DEEP (score: 0)
```

**Threshold Table** (volume-normalized):

| Metric | Very Low | Low | Normal | High | Very High |
|--------|----------|-----|--------|------|-----------|
| Fill velocity (%/min) | <0.01 | 0.01-0.05 | 0.05-0.2 | 0.2-0.5 | >0.5 |
| Depth (% of market) | <0.5 | 0.5-1.0 | 1.0-3.0 | 3.0-5.0 | >5.0 |
| Fill size (% of market) | <0.1 | 0.1-0.25 | 0.25-1.0 | 1.0-2.0 | >2.0 |

**Edge Cases**:

```python
# 1. Very new market (volume < 10k)
if market_volume < 10_000:
    # Fallback to Phase 1 absolute thresholds
    return calculate_score_phase1(fill_data)

# 2. Prefer recent volume over lifetime
volume = market_data.get('volume_24h', market_data['volume'])

# 3. Zero volume protection
if market_volume == 0:
    market_volume = 10_000  # Assume minimum
```

**Data Required**:
- Market volume from Kalshi API: `GET /markets/{ticker}`
- Field: `market['volume']` (lifetime) or `market['volume_24h']` if available
- Already in market metadata, no extra API calls

**Pros**:
- ✅ **Truly automatic** - works for ANY market size
- ✅ No arbitrary bucketing needed
- ✅ More intuitive ("0.5%/min" vs "2000 contracts")
- ✅ Scales as market grows/shrinks
- ✅ Simpler code (~100 lines vs 150 for bucketing)

**Cons**:
- ⚠️ Lifetime volume includes old/stale trades
  - Solution: Use 24h volume if available
- ⚠️ Very new markets (<10k volume) need special handling
  - Solution: Fallback to Phase 1
- ⚠️ Some metrics already relative (aggression, spread)
  - Solution: Keep those as-is

**Status**: SKIPPED  
**Reason**: Temporary compromise, not worth building. Go straight to Phase 3.

**Kept for reference**: Shows alternative approach (volume-normalized thresholds)

---

### Phase 3: Market-Specific Baselines 🔲 (PRIMARY IMPLEMENTATION)

**What**: Use each market's own historical data as baseline

**Implementation**:
```python
def calculate_overreaction_score(fill_data, market_baseline):
    """Score relative to THIS market's normal behavior."""
    
    # Load market's rolling baseline (last 24-48h)
    baseline = {
        'avg_fill_velocity': 1500,      # This market's average
        'avg_aggression': 0.52,
        'avg_spread': 0.025,
        'avg_depth': 18000,
    }
    
    # Score based on RELATIVE deviation
    velocity_multiple = current_velocity / baseline['avg_fill_velocity']
    
    if velocity_multiple > 4.0:     # 4x above THIS market's normal
        velocity_score = 3
    elif velocity_multiple > 2.0:   # 2x above normal
        velocity_score = 2
    elif velocity_multiple < 0.5:   # 50% below normal
        velocity_score = -1  # Underreaction
    else:
        velocity_score = 1
```

**Data Storage**:
```json
// data/04_output/prediction_markets/market_baselines/KXELONMARS-99_baseline.json
{
    "market_ticker": "KXELONMARS-99",
    "last_updated": "2025-12-22T19:00:00Z",
    "window": "24h",
    "metrics": {
        "fill_velocity": {
            "mean": 850,
            "std": 420,
            "p50": 600,
            "p90": 1500,
            "p95": 2200
        },
        "aggression_ratio": {
            "mean": 0.52,
            "std": 0.18,
            "p50": 0.51,
            "p90": 0.72,
            "p95": 0.81
        },
        "spread": {
            "mean": 0.025,
            "std": 0.012,
            "p50": 0.022,
            "p90": 0.045,
            "p95": 0.068
        },
        "depth": {
            "mean": 18500,
            "std": 8200,
            "p50": 16000,
            "p90": 32000,
            "p95": 42000
        }
    },
    "sample_size": 288  // 48h at 10min intervals
}
```

**Baseline Update Process**:
1. Run every 10 minutes via `monitor_kalshi_markets.py`
2. Calculate metrics, append to rolling window
3. Update baseline file if >1h since last update
4. Keep last 48h of data (288 snapshots at 10min intervals)
5. Recalculate mean/std/percentiles

**Scoring Logic**:
```python
# Instead of absolute thresholds, use percentiles
def score_component(current_value, baseline_distribution):
    """Score based on where current value sits in distribution."""
    
    percentile = get_percentile(current_value, baseline_distribution)
    
    if percentile > 95:       # Top 5% = extreme
        return 3
    elif percentile > 90:     # Top 10% = high
        return 2
    elif percentile > 75:     # Top 25% = elevated
        return 1
    elif percentile < 5:      # Bottom 5% = unusually low
        return -1
    else:
        return 0  # Normal range (5th-75th percentile)
```

**Pros**:
- ✅ Perfectly calibrated to each market's unique behavior
- ✅ Self-adjusting (adapts as market matures)
- ✅ Accounts for time-of-day patterns, market lifecycle
- ✅ Can detect anomalies specific to that market

**Cons**:
- ❌ Requires 24-48h of data before useful
- ❌ More complex implementation (~200 lines)
- ❌ Need storage/database for baselines
- ❌ Computational overhead (percentile calculations)

**Calibration Logic**:
```python
def calculate_overreaction_score(fill_data, market_ticker):
    """
    Primary: Use Phase 3 (market baseline)
    Fallback: Use Phase 1 during calibration (<48h data)
    """
    
    baseline = load_baseline(market_ticker)
    
    if baseline and baseline['hours_of_data'] >= 48:
        # ✅ READY: Use market-specific baseline
        score = calculate_score_phase3(fill_data, baseline)
        score['alert_ready'] = True
        score['status'] = 'BASELINE READY'
        score['method'] = f'Phase 3 ({baseline["hours_of_data"]:.0f}h baseline)'
        return score
    
    else:
        # ⚠️ CALIBRATING: Use absolute thresholds, no alerts
        score = calculate_score_phase1(fill_data)
        score['alert_ready'] = False
        hours = baseline['hours_of_data'] if baseline else 0
        score['status'] = f'CALIBRATING ({hours:.0f}h / 48h)'
        score['method'] = 'Phase 1 (fallback)'
        return score


def should_send_alert(score_data):
    """Only send alerts when baseline ready."""
    if not score_data['alert_ready']:
        return False  # Still calibrating
    
    if score_data['score'] >= 8:
        return True, "FADE"
    elif score_data['score'] <= 2:
        return True, "FOLLOW"
    
    return False, None
```

**Timeline**:
- **Day 1-2**: Collect data (show scores but no alerts)
- **Day 3+**: Baseline ready (48h data), alerts active
- **Ongoing**: Baseline improves with more data

**Status**: Ready to implement  
**ETA**: Code today, alerts in 48h  
**Priority**: HIGH (this is the right architecture)

---

## NEW Implementation Timeline

### Days 1-2: Data Collection + Calibration
- [x] Implement Phase 1 scoring (done)
- [ ] Implement baseline storage infrastructure
- [ ] Start monitoring 5-10 markets every 5 minutes
- [ ] Display: "⚠️ CALIBRATING (12h / 48h collected)"
- **NO ALERTS** during this period

### Day 3+: Baseline Ready + Alerts Active
- [ ] After 48h: Switch to Phase 3 scoring
- [ ] Display: "✅ BASELINE READY (54h data)"
- [ ] **START SENDING ALERTS** (email/Slack)
- [ ] New markets: Auto-calibrate (48h wait before alerts)

### Week 2+: Scale + Refine
- [ ] Add 20-50 more markets
- [ ] Each calibrates independently (48h each)
- [ ] Tune percentile thresholds based on observed signals
- [ ] Track alert accuracy (win rate)

### Ongoing: Self-Improvement
- [ ] Baselines update continuously (rolling 48h window)
- [ ] Adapt to market evolution
- [ ] Capture time-of-day patterns
- [ ] Improve with more data

---

## OLD Timeline (For Reference)

### Week 1: Phase 1 Testing ✅
- [x] Implement absolute thresholds
- [ ] Test on 5 POC markets
- [ ] Validate scoring makes sense
- [ ] Adjust thresholds based on observed data

### Week 2: Phase 2 Implementation
- [ ] Add market volume fetching
- [ ] Implement classification function
- [ ] Test across market size spectrum
- [ ] Scale to 20-50 markets

### Week 3-4: Data Collection
- [ ] Monitor 50+ markets continuously
- [ ] Build 48h of history per market
- [ ] Analyze patterns, validate Phase 2

### Month 2: Phase 3 Implementation
- [ ] Design baseline storage schema
- [ ] Implement rolling window calculations
- [ ] Build baseline update process
- [ ] Migrate to percentile-based scoring
- [ ] Validate improvement over Phase 2

---

## Success Metrics

### Phase 1:
- ✅ Works for 5 POC markets
- ✅ Scores feel intuitive (7+ = obvious overreaction)
- ✅ No false positives on normal activity

### Phase 2:
- Scales to 100+ markets without manual tuning
- Accuracy: >80% of 7+ scores are actual overreactions
- False positive rate: <20%
- Works across market sizes (mega to micro)

### Phase 3:
- Accuracy: >90% of 7+ scores are actual overreactions
- False positive rate: <10%
- Adapts to market evolution (adjusts as market matures)
- Captures market-specific patterns (time-of-day, etc.)

---

## Alternative Considered: ML-Based Scoring

**Approach**: Train model to predict overreaction from features

**Why Not**:
1. Need labeled data (manual annotation of overreactions)
2. Black box (hard to explain why score = 8)
3. Overfitting risk with small datasets
4. Domer's framework is already well-defined rules

**Maybe Later**: Phase 4 could use ML to refine scoring, but start rule-based

---

## Key Design Principles

1. **Start Simple**: Phase 1 absolute thresholds → good enough for POC
2. **Progressive Enhancement**: Each phase additive, not replacement
3. **Transparency**: Always explainable (no black boxes)
4. **Fail-Safe**: If baseline missing, fall back to Phase 2 → Phase 1
5. **Testable**: Each phase independently validateable

---

## Notes for Implementation

- Store market volume in snapshot JSON (already fetched)
- Cache market config for session (avoid repeated lookups)
- Log which phase is being used per market (for debugging)
- Add `--phase` flag to force specific phase for testing
- Document threshold rationale (why these numbers?)

---

## Questions to Answer During Testing

**Phase 1**:
- Are thresholds too sensitive? Too loose?
- Which components matter most? (weight adjustment)
- Do scores correlate with actual trading opportunities?

**Phase 2**:
- Are volume tiers appropriate? Should there be 5 tiers instead of 4?
- Should we use different features for classification? (market age, event type)
- Do mega markets need different component weights?

**Phase 3**:
- What window size optimal? (24h vs 48h vs 7d)
- Should we weight recent data more? (exponential decay)
- How to handle market lifecycle? (thin early, thick later)
- Should baselines differ by time-of-day? (US hours vs overnight)

