# Strategy Selection Issue - Circular Dependency

## Problem

**You found a critical flaw:** If an OVER strategy has 39.9% hit rate, the inverse UNDER should have ~60% hit rate and be very profitable. But it's not showing up in our results.

## Root Cause

The backtest has a **circular dependency**:

1. **Backtest loads strategies from JSON** (line 967 in backtest script)
   ```python
   strategies = load_strategies_from_s3(strategy_type, season='2025-26')
   ```

2. **JSON only contains historically profitable strategies**
   - Created by analysis scripts with `--min-roi 5.0`
   - If UNDER was never profitable before, it's not in the JSON

3. **Backtest only tests strategies in the JSON**
   - If "25-30 (High Star) | 6-10 Dog | UNDER" was never in the JSON, it never gets tested
   - Even if the inverse (OVER) is now terrible (39.9% hit rate)

4. **New profitable strategies never discovered**
   - Market conditions change
   - Strategies that were bad become good
   - Inverses of bad strategies never tested

## Example from Your Logs

```
❌ #20. 25-30 (High Star) | 6-10 Dog | OVER | Perimeter (<40.0%)
        AGGREGATE: 67W-101L-0T | Hit Rate: 39.9% | ROI: -26.2%
        2023-24: 22W-48L-0T | Hit Rate: 31.4% | ROI: -44.0%
        2024-25: 34W-35L-0T | Hit Rate: 49.3% | ROI: -6.5%
        2025-26: 11W-18L-0T | Hit Rate: 37.9% | ROI: -30.3%
```

**Logic dictates:** If OVER is 39.9% hit rate, UNDER should be ~60% (inverses should sum to ~100%)

**But:** "25-30 (High Star) | 6-10 Dog | UNDER" doesn't exist in the top 20 strategies. Why? Because it was never in the original strategy JSON file, so it was never tested.

## Solution: Test ALL Strategy Combinations

Instead of loading strategies from JSON, we should **generate all possible combinations**:

### 2D Strategy Combinations
- **Line Tiers:** 5-10, 10-15, 15-20, 20-25, 25-30, 30-35, 35-40
- **Spread Bins:** Pick'em (-2 to +2), 2-6 Fav, 2-6 Dog, 6-10 Fav, 6-10 Dog, 10-15 Fav, 10-15 Dog
- **Bet Side:** OVER, UNDER

**Total:** 7 × 7 × 2 = **98 combinations**

### 3D Strategy Combinations
- Same as 2D, plus:
- **Scorer Type:** Rim Attacker (≥40%), Perimeter (<40%)

**Total:** 7 × 7 × 2 × 2 = **196 combinations**

## Required Changes

### 1. Modify Backtest Script

**File:** `backtesting/20260108_nba_points_props_strategy_backtest.py`

**Current (lines 356-398):**
```python
def apply_strategies_to_data(df, strategies, granularity, strategy_type, min_roi):
    # Loads strategies from JSON
    strat_list = strategies['strategies']
    # Filters by min_roi
    strat_list = [s for s in strat_list if s.get('roi', 0) >= min_roi]
    # Only tests filtered strategies
```

**Needed:**
```python
def generate_all_strategy_combinations(strategy_type, granularity):
    """Generate all possible strategy combinations to test."""
    
    # Define all possible values
    line_tiers = bin_all_line_tiers(granularity)
    spread_bins = bin_all_spread_bins(granularity)
    bet_sides = ['OVER', 'UNDER']
    
    combinations = []
    
    if strategy_type == '2d':
        for line_tier in line_tiers:
            for spread_bin in spread_bins:
                for bet_side in bet_sides:
                    combinations.append({
                        'line_tier': line_tier,
                        'spread_bin': spread_bin,
                        'bet_side': bet_side
                    })
    
    elif strategy_type == '3d':
        scorer_types = ['Rim Attacker (≥40.0%)', 'Perimeter (<40.0%)']
        for line_tier in line_tiers:
            for spread_bin in spread_bins:
                for bet_side in bet_sides:
                    for scorer_type in scorer_types:
                        combinations.append({
                            'line_tier': line_tier,
                            'spread_bin': spread_bin,
                            'bet_side': bet_side,
                            'scorer_type': scorer_type
                        })
    
    return combinations

def apply_strategies_to_data(df, granularity, strategy_type):
    # Remove strategies parameter - generate all combinations instead
    all_strategies = generate_all_strategy_combinations(strategy_type, granularity)
    # Test ALL combinations, filter by min plays later
```

### 2. Update Lambda Function Call

**File:** `scripts/lambda_function_refresh_strategy_statistics.py`

**No changes needed** - already passes `--min-roi -1000.0` which should capture all strategies

### 3. Benefits

✅ **Test all 98 (2D) or 196 (3D) combinations** every backtest
✅ **Discover new profitable strategies** as market conditions change
✅ **Capture inverse strategies** automatically
✅ **Track strategy performance over time** for all combinations
✅ **Filter by aggregate performance** (50+ plays) not historical ROI

## Impact Analysis

### Pros
- **Complete coverage** - no profitable strategy missed
- **Adapts to market changes** - strategies improve/degrade naturally
- **Simpler logic** - no circular dependency
- **Better decisions** - full picture of all strategies

### Cons
- **Larger plays.csv files** - from ~200 plays to ~1,000-2,000 plays per season
  - S3 storage: negligible (text compresses well)
  - Estimate: 5-10MB uncompressed, <1MB compressed
- **Longer backtest time** - testing 98/196 combinations vs 8/20
  - Estimate: +5-10 seconds per season
  - Still well within Lambda limits
- **More strategies in final JSON** - but filtered by 50+ plays aggregate

## Recommendation

**Implement the fix ASAP.** The circular dependency is causing us to miss profitable strategies. The cost is minimal (slightly longer backtest, larger files) but the benefit is huge (complete strategy coverage).

## Next Steps

1. ✅ **Email notifications** - Already added to Lambda function
2. **Modify backtest script** - Add `generate_all_strategy_combinations()` function
3. **Test locally** - Verify all combinations tested
4. **Deploy to Lambda** - Update function code
5. **Verify results** - Should see "25-30 (High Star) | 6-10 Dog | UNDER" in next run

## Evidence This Is The Right Fix

Your observation about the inverse is mathematically sound:
- If OVER hits 39.9% → UNDER should hit ~60%
- If OVER has -26.2% ROI → UNDER should have ~+30% ROI
- This would be the #1 or #2 strategy by hit rate!

The fact it's missing proves we're not testing all combinations.
