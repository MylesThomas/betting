# Strategy Performance Plots Implementation

**Date**: 2026-02-07  
**Feature**: Daily automated strategy performance visualization  
**Status**: ✅ Complete

---

## Overview

Added automated daily generation of 4-panel performance plots for all 15 strategies in `enhanced_unders_v5.json`. These plots show win rate trajectory over time across 3 seasons plus overall performance.

---

## What Was Built

### 1. Plot Generation (4-Panel Layout)

Each strategy gets a PNG with 4 subplots:

```
┌─────────────────────┬─────────────────────┐
│   2023-24 Season    │   2024-25 Season    │
│   Win Rate vs Date  │   Win Rate vs Date  │
└─────────────────────┴─────────────────────┘
┌─────────────────────┬─────────────────────┐
│   2025-26 Season    │   Overall (All)     │
│   Win Rate vs Date  │   Win Rate vs Date  │
└─────────────────────┴─────────────────────┘
```

**Features:**
- Y-axis: Always 0-100% (win rate)
- X-axis: Date (chronological order)
- Baseline: 50% line (gray dashed)
- Stats overlay: W-L record and final win rate
- Color-coded by season:
  - 2023-24: Blue (#1f77b4)
  - 2024-25: Orange (#ff7f0e)
  - 2025-26: Green (#2ca02c)

**Example filename:**
```
20-25_Star_2-6_Fav_UNDER.png
```

### 2. Automated Daily Generation

**Workflow:**
1. **6:00 AM ET**: `lambda_function_refresh_strategy_statistics.py` runs
2. Generates fresh strategy stats for all combinations
3. **NEW:** Generates 4-panel plots for all 15 strategies
4. Uploads plots to S3: `data/04_output/strategy_plots/2025-26/`
5. Sends SNS email with plot summary + S3 paths
6. **7:00 AM ET**: Daily plays email references plots

**Lambda Function:** `nba-strategy-stats-refresher`  
**Schedule:** Daily at 6:00 AM ET (cron: `0 11 * * ? *`)

### 3. S3 Storage Structure

```
s3://nba-betting-mt/
└── data/04_output/strategy_plots/
    └── 2025-26/                           # Season folder
        ├── 35-40_Elite_Pickem_UNDER.png
        ├── 30-35_Superstar_2-6_Dog_UNDER.png
        ├── 30-35_Superstar_10-15_Fav_UNDER.png
        ├── ...                            # (15 total plots)
        └── bench_pickem_rim_under.png     # 3D strategies
```

**Naming Convention:**
- 2D: `{line_tier}_{spread_bin}_{bet_side}.png`
- 3D: `{line_tier}_{spread_bin}_{bet_side}_{scorer_type}.png`
- Spaces replaced with `_`, special chars removed

### 4. Email Integration

**Strategy Refresh Email (6am):**
```
✅ Strategy Statistics Refresh Complete - 2025-26

Results:
2D: ✅ 98 strategies, 45,231 plays
3D: ✅ 196 strategies, 38,127 plays

📈 STRATEGY PERFORMANCE PLOTS
════════════════════════════════════════════════════════════════════════════════
Generated 15 performance plots (4-panel: 2023-24, 2024-25, 2025-26, Overall)
Location: s3://nba-betting-mt/data/04_output/strategy_plots/2025-26/
════════════════════════════════════════════════════════════════════════════════

TOP 20 2D STRATEGIES BY HIT RATE
════════════════════════════════════════════════════════════════════════════════
...
```

---

## Files Modified

### 1. `scripts/generate_role_spread_points_model_daily_email.py`

**Added Functions:**
- `load_strategy_config_from_s3()` - Load enhanced_unders_v5.json from S3
- `load_backtest_plays_for_strategy()` - Load historical plays for specific strategy
- `generate_strategy_performance_plot()` - Create 4-panel matplotlib PNG
- `format_strategy_config_analysis()` - Generate text analysis + plots

**Updated Functions:**
- `generate_email_text()` - Added `include_strategy_analysis` parameter
- `generate_email_html()` - Added `include_strategy_analysis` parameter

**New CLI Args:**
- `--include-strategy-plots` - Flag to generate plots (default: False)

### 2. `scripts/lambda_function_refresh_strategy_statistics.py`

**Added Functions:**
- `generate_all_strategy_plots()` - Generate plots for all strategies in ranking
- `load_backtest_plays_for_strategy_simple()` - Load plays data for plotting

**Updated:**
- SNS notification message now includes plot summary
- Plots generated automatically after stats calculation
- Uses matplotlib with 'Agg' backend (non-interactive for Lambda)

---

## Technical Details

### Dependencies

**Required:**
- `matplotlib` ✅ (in Lambda layer: betting-dashboard-dependencies)
- `pandas` ✅ (in Lambda layer)
- `boto3` ✅ (AWS SDK)

**matplotlib Configuration:**
```python
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for Lambda
import matplotlib.pyplot as plt
```

### Performance Metrics

**Per Strategy:**
- Load 3 CSV files from S3 (one per season)
- Filter to specific strategy (line_tier + spread_bin + bet_side)
- Calculate cumulative win rates
- Generate 4-panel plot
- Upload PNG to S3

**Total Time (15 strategies):**
- Estimated: **2-5 minutes**
- Runs at 6am ET (before 7am plays email)
- Does not block daily workflow

### Lambda Configuration

**Function:** `nba-strategy-stats-refresher`
- Runtime: Python 3.12
- Memory: 1024 MB
- Timeout: 15 minutes (sufficient for plots + stats)
- Layer: betting-dashboard-dependencies (includes matplotlib)

**Schedule:**
- EventBridge rule: `cron(0 11 * * ? *)` (6:00 AM ET daily)
- Runs before main workflow (7:00 AM ET)

---

## Usage Examples

### Local Testing

```bash
# Test plot generation locally
python scripts/lambda_function_refresh_strategy_statistics.py \
  --season 2025-26 \
  --strategy both

# Plots will be generated in /tmp/ and uploaded to S3
# Check S3: s3://nba-betting-mt/data/04_output/strategy_plots/2025-26/
```

### View Plots

```bash
# List all plots in S3
aws s3 ls s3://nba-betting-mt/data/04_output/strategy_plots/2025-26/

# Download specific plot
aws s3 cp s3://nba-betting-mt/data/04_output/strategy_plots/2025-26/30-35_Superstar_2-6_Dog_UNDER.png ./
```

### Manual Trigger (Lambda Console)

```json
{
  "season": "2025-26",
  "strategy": "both",
  "skip_backtest": false
}
```

---

## Strategy Config Integration

**Config File:** `s3://nba-betting-mt/strategies/enhanced_unders_v5.json`

**Strategies Plotted:** All 15 strategies
- 1x Elite tier (79% ROI)
- 2x Superstar tier (22-25% ROI)
- 3x High Star tier (10-18% ROI)
- 4x Star tier (8-10% ROI)
- 3x Supplemental (5-6% ROI)
- 2x 3D legacy (10-26% ROI)

Each strategy plot shows:
- Historical win rate trajectory
- Performance across 3 seasons
- Visual validation of consistency
- Quick spot-check for degradation

---

## Next Steps

### Potential Enhancements (Future)

1. **Interactive Plots** (Plotly)
   - Hover tooltips with game details
   - Zoom/pan capabilities
   - HTML embedding in email

2. **Additional Metrics**
   - ROI trajectory (not just win rate)
   - Rolling 20-game average
   - Confidence intervals

3. **Comparative Analysis**
   - Overlay multiple strategies
   - Best vs worst performance
   - Seasonal comparisons

4. **Automated Alerts**
   - Detect win rate drops > 5%
   - Flag strategies falling below 50%
   - Send warnings before removing

5. **Historical Archive**
   - Keep dated versions of plots
   - Track performance changes over time
   - Automated cleanup (>30 days old)

---

## Summary

✅ **Complete** - Strategy performance plots now generated daily  
✅ **Automated** - Runs at 6am ET as part of strategy refresh  
✅ **Scalable** - Works for all 15 strategies (2D + 3D)  
✅ **Integrated** - Plots uploaded to S3, paths in email  
✅ **Validated** - Uses production backtest data (3 seasons)

**Impact:**
- Visual validation of strategy performance
- Quick identification of degrading strategies
- Historical trend analysis across seasons
- Data-driven decision making for strategy rotation

---

**Author:** Myles Thomas  
**Date:** 2026-02-07  
**Version:** v1.0
