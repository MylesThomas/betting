# Operational Workflow - Daily Monitoring Loop

## Daily Schedule

### Morning Routine (8-9 AM)

**1. Discover New Markets** (10 min)
```bash
# Run market discovery
python scripts/kalshi/find_kalshi_markets.py

# Review output, identify interesting markets:
# - High volume markets (>100k)
# - Upcoming events (next 7 days)
# - Micro events with edge potential
# - Avoid: Long-dated (2099), influenceable events
```

**2. Update Monitoring List** (5 min)
```python
# Edit: scripts/kalshi/monitor_kalshi_markets.py
# Add new tickers to MARKETS_TO_MONITOR list

MARKETS_TO_MONITOR = [
    'KXELONMARS-99',
    'KXTRUMP2028-45',
    'NEW_MARKET_TODAY',  # Add here
    # ... existing markets
]
```

**3. Review Overnight Activity** (10 min)
```bash
# Check each monitored market for signals
for ticker in $(cat monitored_markets.txt); do
    python scripts/kalshi/view_kalshi_order_book.py $ticker
done

# Look for:
# - Overreaction scores >7 (fade opportunities)
# - Underreaction scores <3 (follow opportunities)
# - Large price moves overnight
```

---

## Continuous Monitoring Loop

### Every 5 Minutes (Automated via cron/script)

```bash
# Run monitoring script (saves snapshots)
python scripts/kalshi/monitor_kalshi_markets.py

# This:
# 1. Fetches current order books for all tracked markets
# 2. Saves timestamped snapshots to disk
# 3. Logs activity to console
# 4. (Future) Sends alerts if thresholds hit
```

**Cron Setup**:
```bash
# Edit crontab
crontab -e

# Add line (runs every 5 minutes):
*/5 * * * * cd /Users/thomasmyles/dev/betting && python scripts/kalshi/monitor_kalshi_markets.py >> logs/kalshi_monitor.log 2>&1
```

**Or Background Loop**:
```bash
# Run continuously in background
nohup bash -c 'while true; do python scripts/kalshi/monitor_kalshi_markets.py; sleep 300; done' &
```

---

### Every 30 Minutes (Quick Check)

```bash
# Review markets with recent activity
python scripts/kalshi/quick_scan.py  # (TODO: build this)

# Shows summary:
# Market                Score  Change   Signal
# KXELONMARS-99         8/10   +15%     FADE
# KXTRUMP2028-45        2/10   +2%      FOLLOW
# ...
```

---

### Every 2 Hours (Deep Analysis)

```bash
# Analyze each market in detail
for ticker in $(cat monitored_markets.txt); do
    echo "=== $ticker ==="
    python scripts/kalshi/view_kalshi_order_book.py $ticker
    echo ""
done > market_analysis_$(date +%Y%m%d_%H%M).txt
```

---

## Iterative Improvement Loop

### Week 1: Baseline & Calibration

**Day 1-2**: Set up monitoring
- [ ] Add 5-10 markets to tracking list
- [ ] Run monitor every 5 minutes
- [ ] Build 48h of data

**Day 3-5**: Calibrate thresholds
- [ ] Review overreaction scores
- [ ] Do they match intuition?
- [ ] Adjust config if needed:
  ```python
  # If too many false positives, increase thresholds
  FILL_VELOCITY_HIGH = 2500  # was 2000
  
  # If missing obvious overreactions, decrease
  FILL_VELOCITY_HIGH = 1500  # was 2000
  ```

**Day 6-7**: Validate signals
- [ ] Track when score >7 (fade signals)
- [ ] Did market mean revert? How long?
- [ ] Track when score <3 (follow signals)
- [ ] Did market continue moving?

---

### Week 2: Scale & Enhance

**Monday**: Expand coverage
- [ ] Add 10-20 more markets
- [ ] Focus on variety (mega, large, small)
- [ ] Test Phase 2 (market classification)

**Tuesday-Thursday**: Add metrics
- [ ] Track new component: Order book depth decay rate
- [ ] Track new component: Bid-ask imbalance momentum
- [ ] Track new component: Wall appearance/disappearance

**Friday**: Refine scoring
- [ ] Adjust component weights based on data
- [ ] Which components most predictive?
- [ ] Update README with findings

---

### Week 3-4: Automate Alerts

**Build alert system**:
```python
# scripts/kalshi/alert_on_signals.py

def check_for_signals(market_ticker):
    """Run analysis, send alert if threshold hit."""
    score = calculate_overreaction_score(...)
    
    if score >= 8:  # Strong fade signal
        send_email(
            subject=f"FADE SIGNAL: {market_ticker}",
            body=f"Score: {score}/10\n{details}"
        )
    
    elif score <= 2:  # Strong follow signal
        send_email(
            subject=f"FOLLOW SIGNAL: {market_ticker}",
            body=f"Score: {score}/10\n{details}"
        )
```

**Add to monitoring loop**:
```bash
# Every 5 minutes, after saving snapshot
python scripts/kalshi/monitor_kalshi_markets.py
python scripts/kalshi/alert_on_signals.py  # New step
```

---

### Month 2: Phase 3 Baselines

**Week 5**: Build baseline infrastructure
- [ ] Create `market_baselines/` directory
- [ ] Implement rolling window calculations
- [ ] Store per-market 48h baselines

**Week 6-7**: Migrate to relative scoring
- [ ] Use percentile-based scoring
- [ ] Compare Phase 2 vs Phase 3 accuracy
- [ ] A/B test both approaches

**Week 8**: Production deployment
- [ ] Switch to Phase 3 for all markets with 48h+ history
- [ ] Fallback to Phase 2 for new markets
- [ ] Document findings

---

## Market Discovery Schedule

### Daily (Morning)
```bash
# Quick scan for new markets
python scripts/kalshi/find_kalshi_markets.py --new-today

# Add interesting ones to tracking
```

### Weekly (Monday)
```bash
# Deep scan across all categories
python scripts/kalshi/find_kalshi_markets.py --category politics
python scripts/kalshi/find_kalshi_markets.py --category economics
python scripts/kalshi/find_kalshi_markets.py --category sports

# Review volume/activity for current tracked markets
# Remove dead markets, add new high-volume ones
```

### Monthly
```bash
# Full portfolio review
# - Which markets generated best signals?
# - Which were false positives?
# - Update tracking list based on performance
```

---

## Data Collection Targets

### Snapshots Per Day
```
24 hours × 12 snapshots/hour (5 min intervals) = 288 snapshots/day
```

### Storage Per Market
```
~5KB per snapshot × 288 snapshots = ~1.4 MB/day per market
50 markets × 1.4 MB = 70 MB/day total
```

### Retention Policy
```
Keep last 7 days of snapshots: ~500 MB
Archive to compressed format after 7 days
Delete after 30 days (unless notable events)
```

---

## Quick Commands Reference

```bash
# Start monitoring (background)
nohup python scripts/kalshi/monitor_kalshi_markets.py &

# Check specific market
python scripts/kalshi/view_kalshi_order_book.py TICKER

# Discover new markets
python scripts/kalshi/find_kalshi_markets.py

# View monitoring log
tail -f logs/kalshi_monitor.log

# Count snapshots collected
ls data/04_output/prediction_markets/order_books/ | wc -l

# Check latest signal for all markets
python scripts/kalshi/scan_all_markets.py  # (TODO)
```

---

## Metrics to Track (Manual Log)

### Daily (in spreadsheet/notion)
- Number of markets monitored: __
- Number of signals generated: __
  - FADE (7-10): __
  - FOLLOW (0-3): __
- Signals acted on: __
- Outcome (correct/incorrect): __

### Weekly Review
- Signal accuracy: __ / __ correct
- False positive rate: __%
- Best performing market: __
- Worst performing market: __
- Adjustments made: __

### Monthly KPIs
- Total signals: __
- Win rate: __%
- Average hold time: __ hours
- Best signal type: FADE / FOLLOW / NEUTRAL

---

## Iteration Checklist

### After First Week
- [ ] Do thresholds feel right? Adjust if needed
- [ ] Are scores explainable? Do they match intuition?
- [ ] Should we add/remove component metrics?
- [ ] Is 5-min polling frequency right? (vs 1-min or 10-min)

### After First Month
- [ ] Implement Phase 2 (market classification)
- [ ] Build alert system
- [ ] Scale to 50+ markets
- [ ] Document which markets best for trading

### After 2 Months
- [ ] Implement Phase 3 (baselines)
- [ ] Backtest signals (hypothetical P&L)
- [ ] Optimize component weights
- [ ] Consider ML enhancements

---

## Emergency Procedures

### If monitoring stops
```bash
# Check if process running
ps aux | grep monitor_kalshi

# Restart if needed
python scripts/kalshi/monitor_kalshi_markets.py &

# Check logs for errors
tail -50 logs/kalshi_monitor.log
```

### If API rate limits hit
```bash
# Slow down polling
# Change from 5 min → 10 min intervals

# Or reduce number of markets
# Remove low-priority tickers
```

### If disk fills up
```bash
# Archive old snapshots
cd data/04_output/prediction_markets/order_books/
tar -czf archive_$(date +%Y%m).tar.gz *.json
rm *.json

# Or adjust retention policy
```

---

## Future Enhancements (Backlog)

- [ ] Build dashboard (Streamlit/Grafana)
- [ ] Real-time websocket monitoring (vs polling)
- [ ] Slack/Discord bot for signals
- [ ] Mobile app notifications
- [ ] Correlation analysis across markets
- [ ] Automated position sizing calculator
- [ ] Integration with Kalshi trading API
- [ ] Backtesting framework with hypothetical P&L

