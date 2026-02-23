# NFL Regression Plays Finder

Identifies profitable NFL betting opportunities based on luck regression analysis.

## Strategies (from historical analysis thru Week 12)

### PRIMARY: Back Unlucky Favorites
- **Criteria**: Team had ≤-7 luck last week, is a favorite this week with spread ≤7
- **Performance**: 67-73% ATS, +27-39% ROI
- **Sample**: 26 games

### SECONDARY: Back Lucky Big Underdogs  
- **Criteria**: Team had ≥+7 luck last week, is underdog this week with spread ≥7
- **Performance**: 71% ATS, +36% ROI
- **Sample**: 7 games

## Usage

### Historical Backtesting
```bash
python implementation/find_nfl_regression_plays.py --week 6
python implementation/find_nfl_regression_plays.py --week 10 --verbose-mode
```

### Current Week (Live Betting)
```bash
# Set API key if lines need to be fetched
export ODDS_API_KEY='your_key_here'

# Find plays for current week
python implementation/find_nfl_regression_plays.py --current-week
python implementation/find_nfl_regression_plays.py --current-week --verbose-mode
```

## Output Files

### 1. `nfl_week_N_all_teams.csv` (MAIN OUTPUT)
- **1 row per team** (not per game)
- Includes ALL teams, even those with False strategy flags
- Can filter for plays: `df[df['strat_primary_back_unlucky_fav_small_spread'] == True]`

**Key Columns:**
- `team`, `opponent`, `location` (home/away)
- `last_week_luck`: Difference between actual and expected performance
- `this_week_spread`: Team's spread (negative = favorite)
- `is_favorite`: Boolean
- `spread_category`: "0-3", "3-7", or "7+"
- `strat_primary_back_unlucky_fav_small_spread`: Boolean
- `strat_secondary_back_lucky_dog_large_spread`: Boolean

### 2. `nfl_week_N_plays_only.csv`
- Filtered version with only strategy matches
- Same columns as above

## Verbose Mode

Add `--verbose-mode` to see detailed reasoning for each team:

```
Team: LAC (home) vs PIT
📊 Last Week (Week 9) Performance:
   Luck: -8.7
   Status: 💔 UNLUCKY (-7.0 or less)

📈 This Week (Week 10) Betting Line:
   Spread: -2.5
   Role: ⭐ FAVORITE
   Spread Category: 0-3

🎯 Strategy Evaluation:
   PRIMARY (Back Unlucky Fav ≤7):
      ✓ Unlucky last week (-8.7)
      ✓ Is a favorite (-2.5)
      ✓ Spread is ≤7 (2.5)
      ✅ PRIMARY MATCH - Expected: 67-73% ATS, +27-39% ROI

💰 BETTING RECOMMENDATION: BET LAC -2.5
```

## Current Week Mode Details

When using `--current-week`:
1. Checks historical CSVs for games in next 7 days
2. If no games found, fetches from Odds API automatically
3. Saves fetched data to `game_lines/historical/nfl_game_lines_YYYY-MM-DD.csv`
4. Uses most recent week's luck data
5. Proceeds with normal analysis

## Prerequisites

1. Run analysis to generate tracking data:
```bash
python backtesting/20251126_nfl_spread_covering_vs_score_differential.py --observe-trends --team all
```

2. For current week, set API key:
```bash
export ODDS_API_KEY='your_key_here'
```

## Example Workflow

```bash
# Week 13 is coming up...

# 1. Fetch latest lines (optional, auto-fetches if needed)
python scripts/fetch_nfl_season_lines.py

# 2. Find plays
python implementation/find_nfl_regression_plays.py --current-week --verbose-mode

# 3. Review output
open data/04_output/todays_plays/nfl_week_13_all_teams.csv
```
