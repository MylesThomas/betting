# NBA Player Points Props - Backtest Workflow

This document explains how to backtest your 2025-26 season strategies against historical data from past NBA seasons.

## Overview

The backtest validates whether strategies trained on 2025-26 data would have been profitable in previous seasons (2024-25, 2023-24, 2022-23, 2021-22).

**Strategy Types:**
- **2D Strategy**: Player tier × Game spread (63 combinations in detailed mode)
- **3D Strategy**: Player tier × Game spread × Scorer type (126 combinations in detailed mode)

## Quick Start

### Option 1: Automated Data Fetching + Backtest

```bash
# 1. Fetch all historical data (will prompt for confirmation)
# Note: Default includes 2025-26, but you already have it
python3 backtesting/fetch_historical_data_for_backtest.py --seasons 2024-25 2023-24 2022-23 2021-22

# Or fetch all seasons including 2025-26 (default)
python3 backtesting/fetch_historical_data_for_backtest.py

# Or fetch specific seasons only
python3 backtesting/fetch_historical_data_for_backtest.py --seasons 2024-25 2023-24

# Skip confirmation prompt (careful!)
python3 backtesting/fetch_historical_data_for_backtest.py --yes

# 2. Run backtest on all seasons
python3 backtesting/20260108_nba_points_props_strategy_backtest.py
```

### Option 2: Manual Step-by-Step

See [Full Workflow](#full-workflow) below.

## Prerequisites

1. **The Odds API Key**: Set in environment or AWS Secrets Manager
2. **AWS S3 Access**: Read/write permissions for these buckets:
   - `the-odds-api-mt` (props and game lines)
   - `nba-api-mt` (game results and shot charts)
   - `nba-betting-mt` (merged analysis datasets)

3. **2025-26 Strategies**: Must already exist in S3:
   - `s3://nba-betting-mt/data/03_intermediate/points_by_role_gamespread_strategies_2025-26.json` (2D)
   - `s3://nba-betting-mt/data/03_intermediate/points_by_role_gamespread_6feet_strategies_2025-26_rim40.json` (3D)

## Full Workflow

### Step 1: Fetch Historical Data

For **each season** you want to backtest (2021-22, 2022-23, 2023-24, 2024-25):

#### 1a. Fetch Player Props + Game Results

```bash
python3 scripts/fetch_nba_player_props.py --mode 2 --fetch-games --s3 --season 2024-25
```

**What this does:**
- Fetches player props from The Odds API for every game date in the season
- Fetches actual game results (box scores) from NBA API
- Uploads to S3:
  - `s3://the-odds-api-mt/nba/historical_player_props/2024-25/*.csv`
  - `s3://nba-api-mt/player_game_logs/2024-25/*.csv`

**Cost:** ~15,000-25,000 API credits per season (The Odds API)

#### 1b. Fetch Game Lines (Spreads)

```bash
python3 scripts/fetch_historical_nba_season_lines.py --season 2024-25 --prod-run
```

**What this does:**
- Fetches closing game lines (spreads, moneylines) for every game
- Uploads to S3:
  - `s3://the-odds-api-mt/nba/historical_game_lines/2024-25/*.csv`

**Cost:** ~500-1,000 API credits per season

#### 1c. Fetch Shot Charts (for 3D Strategy only)

```bash
python3 scripts/fetch_all_nba_shot_charts.py --auto --seasons 2024-25
```

**What this does:**
- Fetches shot distance data for all players (to calculate rim scorer %)
- Uploads to S3:
  - `s3://nba-api-mt/player_shot_charts/2024-25/*.csv`

**Cost:** Free (NBA Stats API)

#### 1d. Join All Data Sources

**For 2D Strategy (tier × spread):**
```bash
python3 scripts/join_nba_points_props_actuals_charts_gamelines.py --season 2024-25 --s3
```

**For 3D Strategy (tier × spread × scorer_type):**
```bash
python3 scripts/join_nba_points_props_actuals_charts_gamelines.py --season 2024-25 --s3 --rim-scorer-pct 40
```

**What this does:**
- Joins props, game results, game lines, and shot charts into unified dataset
- Uploads to S3:
  - `s3://nba-betting-mt/data/03_intermediate/player_props_with_actuals_2024-25.csv` (2D)
  - `s3://nba-betting-mt/data/03_intermediate/player_props_with_actuals_2024-25_rim40.csv` (3D)

**Repeat Steps 1a-1d for each season you want to backtest.**

### Step 2: Run Backtest

Once historical data is ready, run the backtest:

```bash
# Backtest both strategies on all available seasons
python3 backtesting/20260108_nba_points_props_strategy_backtest.py

# Backtest specific seasons
python3 backtesting/20260108_nba_points_props_strategy_backtest.py --seasons 2024-25 2023-24

# Backtest only 2D strategy
python3 backtesting/20260108_nba_points_props_strategy_backtest.py --strategy 2d

# Backtest only 3D strategy
python3 backtesting/20260108_nba_points_props_strategy_backtest.py --strategy 3d
```

### Step 3: Review Results

Backtest outputs are saved to:
```
data/04_output/backtests/points_props_YYYYMMDD_HHMMSS/
├── 2d_strategy_summary.csv          # Season-by-season performance (2D)
├── 2d_strategy_all_plays.csv        # Detailed play-by-play results (2D)
├── 2d_aggregate.json                # Aggregate statistics (2D)
├── 3d_strategy_summary.csv          # Season-by-season performance (3D)
├── 3d_strategy_all_plays.csv        # Detailed play-by-play results (3D)
└── 3d_aggregate.json                # Aggregate statistics (3D)
```

**Key metrics to review:**
- **Win Rate**: Percentage of bets that won (target: >52.4% to beat -110 odds)
- **ROI**: Return on investment (target: >5% for confident edge)
- **Total Profit**: Dollar profit across all plays
- **Sample Size**: Number of plays (more is better for statistical confidence)

## Example Output

```
════════════════════════════════════════════════════════════════════════════════
📊 FINAL AGGREGATE RESULTS
════════════════════════════════════════════════════════════════════════════════

2D Strategy:
   Seasons: 2024-25, 2023-24, 2022-23
   Total Plays: 2,847
   Win Rate: 54.2%
   Total Profit: $4,567.89
   Total Staked: $284,700.00
   ROI: +1.6%

3D Strategy:
   Seasons: 2024-25, 2023-24, 2022-23
   Total Plays: 1,923
   Win Rate: 56.8%
   Total Profit: $8,234.56
   Total Staked: $192,300.00
   ROI: +4.3%

✅ Backtest complete!
✅ Results saved to: data/04_output/backtests/points_props_20260108_143052
```

## Important Notes

### Data Availability

- **The Odds API**: Historical data goes back to **2021-22 season**
- **NBA API**: Game results available for all past seasons
- You must fetch data for each season individually (no bulk download)

### API Costs

Fetching a full season costs approximately:
- **Player Props**: 15,000-25,000 credits (~$3-5 per season on 500k plan)
- **Game Lines**: 500-1,000 credits (~$0.10-0.20 per season)
- **Total**: ~20,000-26,000 credits per season (~$4-5.20)

For 4 seasons (2021-22 to 2024-25): ~100,000 credits (~$20)

### Interpretation Guidelines

**Good Backtest Results:**
- Win rate > 52.4% (breakeven at -110 odds)
- ROI > 5% (meaningful edge after variance)
- Consistent performance across multiple seasons
- Large sample size (>1000 total plays)

**Warning Signs:**
- Win rate < 52% (losing strategy)
- High ROI but small sample size (likely noise)
- Performance degrades in recent seasons (market adaptation)
- Wildly different performance across seasons (overfitting)

**What to do if backtest fails:**
- Consider retraining strategies using multi-season data instead of just 2025-26
- Adjust minimum ROI threshold (currently 5%)
- Filter strategies by sample size (currently 50 games minimum)
- Investigate which specific strategy combinations underperformed

## Troubleshooting

### "Data file not found" error

The backtest script will tell you exactly which data is missing and how to fetch it:

```
❌ Data file not found: s3://nba-betting-mt/data/03_intermediate/player_props_with_actuals_2023-24.csv

⚠️  To generate this file, run:
   python3 scripts/fetch_nba_player_props.py --mode 2 --fetch-games --s3 --season 2023-24
   python3 scripts/fetch_historical_nba_season_lines.py --season 2023-24 --prod-run
   python3 scripts/join_nba_points_props_actuals_charts_gamelines.py --season 2023-24 --s3
```

### "NoSuchBucket" error

Check AWS credentials and S3 bucket permissions:
```bash
aws s3 ls s3://nba-betting-mt/
aws s3 ls s3://the-odds-api-mt/
aws s3 ls s3://nba-api-mt/
```

### "Strategy file not found" error

Make sure you've generated the 2025-26 strategies first:
```bash
# Generate 2D strategies
python3 analysis/analyze_points_props_role_spread_model.py --season 2025-26 --granularity detailed --min-roi 5.0

# Generate 3D strategies
python3 analysis/analyze_points_props_role_spread_6feet_scorer_model.py --granularity detailed --min-roi 5.0
```

## Next Steps

After reviewing backtest results:

1. **If backtest is positive**: Deploy strategies with confidence
2. **If backtest is mixed**: Consider filtering to only the best-performing strategy combinations
3. **If backtest is negative**: Retrain using multi-season data or adjust model parameters

---

**Author**: Myles Thomas  
**Date**: 2026-01-08  
**Related Scripts**:
- `backtesting/20260108_nba_points_props_strategy_backtest.py` (main backtest)
- `backtesting/fetch_historical_data_for_backtest.py` (automated data fetching)

