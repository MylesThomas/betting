# Bench Pickem Under Overlap Analysis
**Date:** 2026-01-18 5:20pm CT  
**Issue:** Why `bench_pickem_perimeter_under + bench_pickem_rim_under ≠ bench_pickem_under`

## Summary
✅ **Hypothesis CONFIRMED**: The 2D strategy `bench_pickem_under` includes players without `scorer_type` classification, while 3D strategies only include players with defined scorer types.

## Data Analysis (2026-01-18)

### Play Counts
- **2D bench_pickem_under**: 7 plays
- **3D bench_pickem_rim_under**: 3 plays  
- **3D bench_pickem_perimeter_under**: 3 plays
- **3D total**: 6 plays

### The Gap
- **Missing in 3D**: 1 player (Moussa Diabate)
- **Reason**: Moussa Diabate has **NO scorer_type classification**

### Player Details
```
Player: Moussa Diabate
Team: CHA
Line Tier: 5-10 (Bench)
Spread Bin: Pick'em (-2 to +2)
Bet Side: UNDER
Expected ROI: +13.6%
Strategy (2D): bench_pickem_under ✅
Strategy (3D): NOT INCLUDED ❌ (no scorer_type)
```

## Set Logic Reality

```
bench_pickem_under (2D) = {
    players in "5-10 (Bench)" + "Pick'em (-2 to +2)" + UNDER
    ├─ With scorer_type = "Rim (≥40.0%)" → bench_pickem_rim_under
    ├─ With scorer_type = "Perimeter (<40.0%)" → bench_pickem_perimeter_under  
    └─ With scorer_type = NULL/NaN → ONLY in 2D strategy
}
```

**Formula:**
```
bench_pickem_under = bench_pickem_rim_under 
                   + bench_pickem_perimeter_under 
                   + unclassified_players
```

## Root Cause
Players without sufficient game data to calculate their rim attack % do not get assigned a `scorer_type`. These players:
- ✅ Appear in 2D strategies (no scorer_type required)
- ❌ Do NOT appear in 3D strategies (scorer_type required)

## Impact on Strategy Config
The `bench_pickem_under` strategy (2D) will **always** have ≥ plays than the sum of its 3D components because it captures the "unclassified" players.

This is **working as designed** and not a bug.

## Recommendation
- Keep both the 2D aggregate (`bench_pickem_under`) and 3D granular strategies
- The 2D strategy acts as a "catch-all" that includes players without scorer_type
- Monitor if unclassified players have different performance characteristics
- Consider creating a separate strategy for "unclassified scorer types" if the sample size grows

## Files Used
- `/tmp/2026-01-18_2d_top3.csv` - Filtered 2D plays
- `/tmp/2026-01-18_3d_top3.csv` - Filtered 3D plays
- `/tmp/analyze_bench_pickem_overlap.py` - Analysis script

