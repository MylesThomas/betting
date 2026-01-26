# NBA Points Props Strategy Performance Analysis

## Overview

This script analyzes ALL NBA points props strategies across multiple seasons to identify which have the best edge/ROI. It's designed to be run weekly to track performance as new data comes in.

**Script:** `analyze_all_strategies_performance.py`

## Quick Start

```bash
# Analyze all strategies with default settings (recommended for weekly runs)
python scripts/analyze_all_strategies_performance.py

# Generate visualizations (saved to ~/Downloads/tmp and displayed)
python scripts/analyze_all_strategies_performance.py --viz

# Generate visualizations with Kelly criterion bet sizing
python scripts/analyze_all_strategies_performance.py --viz --kelly --bankroll 10000 --kelly-fraction 0.5

# Compare different Kelly fractions (quarter, half, full)
python scripts/analyze_all_strategies_performance.py --viz --kelly --bankroll 10000 --kelly-fraction 0.25
python scripts/analyze_all_strategies_performance.py --viz --kelly --bankroll 10000 --kelly-fraction 1.0

# Generate visualizations without displaying (just save files)
python scripts/analyze_all_strategies_performance.py --viz --no-viz-display

# Analyze with custom parameters
python scripts/analyze_all_strategies_performance.py --top-n 30 --min-plays 100 --viz --viz-top-n 15
```

## What It Does

1. **Loads backtest data** from S3 for both 2D and 3D strategies across all seasons
2. **Groups by strategy parameters**: line_tier, spread_bin, bet_side, scorer_type
3. **Calculates comprehensive metrics**:
   - ROI (Return on Investment)
   - Total profit
   - Win rate
   - Consistency (profitable seasons / total seasons)
   - Season-by-season breakdowns
4. **Ranks strategies** by overall performance
5. **Outputs results** in JSON and CSV formats
6. **Generates visualizations** (optional with `--viz` flag)

## Output Files

The script generates files in `~/Downloads/tmp/` (changed from `data/04_output/`):

1. **all_strategies_ranked_YYYYMMDD_HHMMSS.json** - Timestamped detailed JSON
2. **all_strategies_ranked_YYYYMMDD_HHMMSS.csv** - Timestamped CSV with flattened season stats
3. **all_strategies_ranked_latest.json** - Latest run (overwrites)
4. **all_strategies_ranked_latest.csv** - Latest run (overwrites)

## CLI Options

```bash
--strategy-types 2d 3d      # Strategy types to analyze (default: 2d 3d)
--seasons 2023-24 2024-25   # Seasons to analyze (default: last 3)
--min-plays 50              # Minimum plays to include strategy (default: 50)
--top-n 20                  # Number of top strategies to display (default: 20)
--output-dir path/to/dir    # Output directory (default: ~/Downloads/tmp)
--no-save                   # Skip saving results to files
--viz                       # Generate visualizations
--viz-output-dir path       # Directory for visualizations (default: ~/Downloads/tmp)
--viz-top-n 10              # Number of strategies to visualize (default: 10)
--no-viz-display            # Save visualizations but don't display interactively
--kelly                     # Use Kelly criterion for bet sizing
--bankroll 10000            # Starting bankroll for Kelly (default: 10000)
--kelly-fraction 0.5        # Kelly fraction: 0.25=quarter, 0.5=half, 1.0=full (default: 0.5)
```

## Visualizations

When using the `--viz` flag, the script generates 4 visualizations:

### 1. Cumulative Profit Over Time
- **Shows:** How profit accumulates play-by-play for top 5 strategies
- **X-axis:** Play number (chronological)
- **Y-axis:** Cumulative profit ($)
- **Insight:** Reveals strategy consistency, volatility, and momentum

### 2. ROI by Season Comparison (Grouped Bar Chart)
- **Shows:** ROI for each season side-by-side for top 10 strategies
- **X-axis:** Strategy names
- **Y-axis:** ROI (%)
- **Grouped bars:** One per season (2023-24, 2024-25, 2025-26)
- **Insight:** Identifies which strategies are consistent vs seasonal

### 3. Win Rate Over Time (Rolling Average)
- **Shows:** Win rate trends with 50-game rolling window
- **X-axis:** Play number
- **Y-axis:** Win rate (%)
- **Lines:** Top 5 strategies
- **Breakeven line:** 52.38% (needed to overcome -110 juice)
- **Insight:** Shows if edge is growing, stable, or degrading

### 4. Profit Distribution (Box Plot)
- **Shows:** Profit variance per play for top 10 strategies
- **X-axis:** Strategy names
- **Y-axis:** Profit per play ($)
- **Insight:** Reveals volatility and risk profile of each strategy

All visualizations are saved to `~/Downloads/tmp/` by default (customizable with `--viz-output-dir`).

## Key Findings (as of 2026-01-25)

### Top 3 Strategies by ROI

1. **20_25_2_6_dog_rim_under** (3D)
   - ROI: **15.9%**
   - Profit: $4,480
   - Plays: 281
   - Win Rate: 60.0%
   - Profitable: 2/3 seasons
   - Star players (20-25 pts) on slight underdog teams (2-6 spread), rim attackers, UNDER

2. **5_10_pickem_rim_under** (3D)
   - ROI: **13.4%**
   - Profit: $3,750
   - Plays: 279
   - Win Rate: 58.8%
   - Profitable: 2/3 seasons
   - Bench players (5-10 pts) in pick'em games, rim attackers, UNDER

3. **20_25_2_6_fav_under** (2D) - **MOST CONSISTENT**
   - ROI: **7.4%**
   - Profit: $6,146
   - Plays: 832
   - Win Rate: 56.2%
   - Profitable: **3/3 seasons** (ONLY strategy profitable all 3)
   - Star players on favored teams (2-6 spread), UNDER

### Key Insights

#### By Bet Side
- **UNDER strategies** dominate: 9 of 9 profitable strategies are UNDERs
- **OVER strategies** underperform: 0 of 11 profitable
- **Average ROI**: UNDER +0.6% vs OVER -11.6%

#### By Strategy Type
- **2D strategies** more reliable: 50% profitable (4/8)
- **3D strategies** less reliable: 25% profitable (5/20)
- **Best 2D average ROI**: -3.2%
- **Best 3D average ROI**: -4.6%

#### Consistency
- Only **1 strategy profitable all 3 seasons** (20_25_2_6_fav_under)
- **11 strategies** profitable 2/3 seasons
- **10 strategies** never profitable (0/3 seasons)

#### Player Roles
- **Star players (20-25)**: Strong UNDER edge in favored games
- **Bench players (5-10)**: Strong UNDER edge in pick'em games (rim attackers)
- **Role players (10-15)**: Mixed results, less reliable

#### Scorer Types (3D only)
- **Rim attackers**: Better UNDER performance overall
- **Perimeter shooters**: More volatile, less reliable

## Weekly Workflow

1. **Run the analysis**:
   ```bash
   python scripts/analyze_all_strategies_performance.py --viz
   ```

2. **Review output** in terminal for quick insights

3. **Check visualizations** in `~/Downloads/tmp/`

4. **Check detailed files** for deep dives:
   - JSON: Full strategy details with season breakdowns
   - CSV: Easy to load into Excel/Google Sheets for visualization

5. **Compare to previous weeks** using timestamped files

6. **Update strategy configs** based on findings

## Integration with Existing Scripts

This script complements:
- `backtest_strategy_analyzer.py` - Deep dive into individual strategies
- `docs/strategies/top3_unders_strategies_nba_points_props_v*.json` - Strategy configs
- `find_role_spread_points_model_plays.py` - Daily play finder
- `find_role_spread_scorer_points_model_plays.py` - 3D play finder

## Notes

- Assumes $100 unit size for ROI calculations
- Requires AWS credentials configured for S3 access
- Data pulled from `s3://nba-betting-mt/data/04_output/backtests/`
- Training ROI comes from 2025-26 season analysis
- Minimum plays filter prevents low-sample strategies from skewing results
- Visualizations require matplotlib and seaborn (`pip install seaborn`)

## Future Enhancements

- [ ] Add confidence intervals for ROI estimates
- [ ] Include Sharpe ratio / risk-adjusted metrics
- [ ] Track strategy performance degradation over time
- [ ] Add alerts when top strategies change significantly
- [ ] Integrate with daily play tracking for live performance updates
