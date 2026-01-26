# Strategy Performance Analysis - Quick Reference

## 📊 What We Built

A comprehensive weekly analysis system to track which NBA points props strategies have the best edge/ROI over the last 3 seasons.

## 🚀 Quick Start

```bash
# Run weekly analysis
python scripts/analyze_all_strategies_performance.py

# Compare to last week
python scripts/compare_strategy_performance.py --auto

# Deep dive on specific strategy
python scripts/backtest_strategy_analyzer.py \
    --strategy-type 3d \
    --line-tier "5-10 (Bench)" \
    --spread-bin "Pick'em (-2 to +2)" \
    --bet-side UNDER \
    --scorer-type "Rim Attacker (≥40.0%)"
```

## 📁 Files Created

### Scripts (in `scripts/`)
1. **analyze_all_strategies_performance.py** - Main analysis script
2. **compare_strategy_performance.py** - Week-over-week comparison
3. **backtest_strategy_analyzer.py** - Individual strategy deep dive (already existed)

### Documentation (in `scripts/` and `docs/`)
1. **README_STRATEGY_PERFORMANCE_ANALYSIS.md** - Full documentation
2. **strategy_performance_summary_2026-01-25.md** - Latest findings summary

### Output Files (in `data/04_output/`)
1. **all_strategies_ranked_latest.json** - Current analysis (JSON)
2. **all_strategies_ranked_latest.csv** - Current analysis (CSV)
3. **all_strategies_ranked_YYYYMMDD_HHMMSS.*** - Timestamped versions

## 🎯 Top 3 Strategies (2025-26 Season)

| Rank | Strategy | Type | ROI | Profit | Plays | Consistency |
|------|----------|------|-----|--------|-------|-------------|
| 1 | 20_25_2_6_dog_rim_under | 3D | 15.9% | $4,480 | 281 | 2/3 ✅ |
| 2 | 5_10_pickem_rim_under | 3D | 13.4% | $3,750 | 279 | 2/3 ✅ |
| 3 | 20_25_2_6_fav_under | 2D | 7.4% | $6,146 | 832 | **3/3** ⭐ |

## 💡 Key Insights

- **UNDER strategies dominate**: 9/9 profitable strategies are UNDERs
- **OVER strategies fail**: 0/11 OVER strategies profitable
- **Only 1 strategy profitable all 3 seasons**: `20_25_2_6_fav_under`
- **2D more reliable than 3D**: 50% vs 25% profitable rate

## 📈 Weekly Workflow

1. **Saturday/Sunday** (after week's games complete):
   - Run `analyze_all_strategies_performance.py`
   - Review top performers
   
2. **Compare to last week**:
   - Run `compare_strategy_performance.py --auto`
   - Identify biggest movers
   
3. **Update configs**:
   - Adjust `docs/strategies/top3_unders_strategies_nba_points_props_v*.json`
   - Update play finder scripts if needed

4. **Monday** (start of new week):
   - Use updated strategies for daily plays

## 🔗 Related Files

- **Strategy configs**: `docs/strategies/`
- **Play finders**: 
  - `scripts/find_role_spread_points_model_plays.py` (2D)
  - `scripts/find_role_spread_scorer_points_model_plays.py` (3D)
- **Daily tracking**: `scripts/track_daily_plays_performance.py`
- **Backtest data**: S3 `s3://nba-betting-mt/data/04_output/backtests/`

## ⚠️ Important Notes

- Requires AWS credentials for S3 access
- Assumes $100 unit size for ROI calculations
- Minimum 50 plays filter to avoid low-sample noise
- Data includes seasons: 2023-24, 2024-25, 2025-26

## 📚 Full Documentation

See `scripts/README_STRATEGY_PERFORMANCE_ANALYSIS.md` for complete details.
