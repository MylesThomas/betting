# Betting Analytics Dashboard

Streamlit dashboard for viewing arbitrage opportunities and data-driven betting strategies for NBA and NFL.

## Quick Start

```bash
# Install dependencies
pip install -r streamlit_app/requirements.txt

# Run the dashboard
streamlit run streamlit_app/app.py
```

The dashboard will open in your browser at `http://localhost:8501`

## Features

- 📊 View today's arbitrage opportunities
- 🔄 Manual "Run Now" button to find new arbs
- 📈 Key metrics (total arbs, avg profit, best opportunity)
- 📅 Historical performance tracking
- 💾 Download opportunities as CSV
- 🎚️ Filter by minimum profit percentage
- 🏀 NBA: 3PT Under strategies, player props
- 🏈 NFL: Regression-to-mean betting strategies

---

## Cache Management

### Update Team/Player Cache

If you see missing players or teams in the dashboard:

```bash
cd betting
python scripts/build_full_roster_cache.py
```

**When to rebuild:**
- Missing teams (should have all 30 NBA teams)
- After major trades
- At the start of a new season
- Error message: "X player(s) not in cache"

### Fix Streamlit Cache Issues

If the app shows "aggressively caching, not reading new player_team_cache.csv":

1. Go to [Streamlit Cloud](https://share.streamlit.io/)
2. Find your app → **...** (menu) → **Reboot app**

---

## NBA Workflows

### 1. Daily 3PT Under Strategy (Underdog Unders)

**Goal:** Backtest 3PT under strategy on 2025-26 season, then find today's plays.

#### Step 1: Get This Season's Game Data

```bash
cd /Users/thomasmyles/dev/betting && python scripts/build_season_game_logs.py

# Or explicitly specify season
python scripts/build_season_game_logs.py --season 2025-26
```

**Output:** `data/01_input/nba_api/season_game_logs/2025_26/{player}.csv`

#### Step 2: Get Props Data (Historical + Today)

```bash
# Build calendar
python scripts/nba_calendar_builder.py --season 2025-26

# Fetch player_threes props for entire season
python scripts/fetch_and_build_season_props.py --season 2025-26 --market player_threes
```

**Output:**
- `data/03_intermediate/combined_props_2025_26_player_threes.csv`
- `data/03_intermediate/consensus_props_2025_26_player_threes.csv`

#### Step 3: Find Today's Opportunities

```bash
# Find 3PT under opportunities
python implementation/find_3pt_underdog_unders_today.py

# Sanity check a specific player
python implementation/find_3pt_underdog_unders_today.py --sanity-check "Jose Alvarado" --line 0.5
```

**Output:** `data/04_output/todays_plays/3pt_underdog_unders_YYYYMMDD.csv`

**Note:** Strategy verified on 24-25 season via:
```bash
python backtesting/20251121_nba_3pt_prop_miss_streaks_24_25.py --underdog-unders
```

---

### 2. Modeling 3PT Lines (0.5 Over/Under)

**Goal:** Model probability of 0.5 lines going over/under.

#### Step 1: Fill Missing Game Data

Backfill player game data for games without props:

```bash
# Test the script
cd /Users/thomasmyles/dev/betting && python scripts/fill_player_missing_games.py --test 2>&1 | grep -E "(Test [0-9]+:|OVERALL: SUCCESS)"

# Run for all players
python scripts/fill_player_missing_games.py --all
```

**Data Sources:**
- Props: `01_input/the-odds-api/historical_props/props_2024...*.csv`
- Game data: `01_input/nba_api/season_game_logs/2024_25/{player}.csv`

#### Step 2: Model O/U Probabilities

```bash
# Test specific player
python backtesting/20251124_nba_3pt_prop_modeling_ou_half_point_lines.py \
  --test-market-edges \
  --detailed-log \
  --players "Giannis Antetokounmpo"

# Run for all players
cd /Users/thomasmyles/dev/betting && python backtesting/20251124_nba_3pt_prop_modeling_ou_half_point_lines.py --all-players 2>&1 | tee /tmp/all_players_output.txt
```

---

## NFL Workflows

> **⚠️ Note:** NFL workflows are currently **manual** (not automated via Lambda yet). 
> 
> **Future automation:** Lambda function created ([`docs/lambda_function_nfl.py`](docs/lambda_function_nfl.py)) to send weekly email alerts with play recommendations. Setup guide: [`docs/AWS_NFL_LAMBDA_SETUP.md`](docs/AWS_NFL_LAMBDA_SETUP.md)
> 
> **Planned schedule:** Every Monday at 12:00 PM ET (may need re-run after Monday Night Football)

### 1. NFL Regression-to-Mean Betting Strategy

**Goal:** Find "lucky" and "unlucky" teams based on score differential vs. expected points, then identify regression opportunities.

#### Background

Using Unexpected Points data (actual score vs. adjusted expected score), we identify:
- **Lucky teams:** Overperformed by 7+ points last week
- **Unlucky teams:** Underperformed by 7+ points last week

**Key Findings (Through Week 12 2025):**
- ✅ **Primary Strategy:** Back unlucky favorites (spread ≤7)
  - 0-3 spread: 67% ATS, +27% ROI
  - 3-7 spread: 73% ATS, +39% ROI
- ✅ **Secondary Strategy:** Back lucky big underdogs (spread ≥7)
  - 7+ spread: 71% ATS, +36% ROI

#### Step 1: Get Unexpected Points Data

1. Download data from [Unexpected Points Spreadsheet](https://docs.google.com/spreadsheets/d/1ktlf_ekms7aI6r0tF_HeX0zaxps-bHWYsgglUReC558/edit?usp=sharing)
2. This contains box score + "adjusted score" for each game

#### Step 2: Fetch NFL Season Line Data

```bash
# Test it works (single game)
python scripts/fetch_nfl_season_lines.py
# Select option 1, then enter a game date

# Get all data (from 2025-09-04 until today)
python scripts/fetch_nfl_season_lines.py
# Select option 1 (runs full season fetch)

# Optional: Get London games that might have been missed
python scripts/fetch_nfl_season_lines.py --london
```

**Output:** `data/01_input/the-odds-api/nfl/game_lines/historical/nfl_game_lines_YYYY-MM-DD.csv`

#### Step 3: Analyze Lines & Join with Unexpected Points

```bash
# Analyze spread data through current week
python scripts/20251125_analyze_nfl_lines.py

# Optional: Debug mode (prints bye weeks, etc.)
python scripts/20251125_analyze_nfl_lines.py --debug
```

#### Step 4: Backtest Regression Strategy

```bash
# Basic backtest
python backtesting/20251126_nfl_spread_covering_vs_score_differential.py

# Debug mode
python backtesting/20251126_nfl_spread_covering_vs_score_differential.py --debug

# Observe trends for specific team
python backtesting/20251126_nfl_spread_covering_vs_score_differential.py --observe-trends --team "GB" --threshold 7

# Analyze all teams at different thresholds
python backtesting/20251126_nfl_spread_covering_vs_score_differential.py --observe-trends --team "all" --threshold 7

# Group by spread categories (0-3, 3-7, 7+)
python backtesting/20251126_nfl_spread_covering_vs_score_differential.py --observe-trends --team all --threshold 7 --group-by-spread

# Include favorite/underdog breakdown
python backtesting/20251126_nfl_spread_covering_vs_score_differential.py --observe-trends --team all --threshold 7 --group-by-spread --include-fav-dog
```

**Example Output:**
```
THRESHOLD COMPARISON (Lucky → Unlucky Spectrum)
========================================================
Threshold           Sample      ATS Record   ATS %    ROI       Avg Spread      Signal
-----------------------------------------------------------------------------------------------
After +7            60 games    28–32        46.7%    -10.9%    -0.98 (fav)     ❌ FADE
After –7            60 games    33–27        55.0%    +6.5%     +1.41 (dog)     🔄 REVERSAL
After –10           34 games    18–16        52.9%    +1.1%     +2.68 (dog)     🟢 Small edge
After –20           5 games     4–1          80.0%    +52.7%    +1.00 (dog)     🔥🔥 MEGA BOUNCE
```

#### Step 5: Find This Week's Plays

```bash
# Find current week's opportunities
python implementation/find_nfl_regression_plays.py --current-week

# Verbose mode (see detailed analysis)
python implementation/find_nfl_regression_plays.py --current-week --verbose-mode

# Safe mode (asks y/n before API calls)
python implementation/find_nfl_regression_plays.py --current-week --verbose-mode --safe-mode

# Production mode (no prompts, goes to API)
python implementation/find_nfl_regression_plays.py --current-week --verbose-mode --no-safe-mode
```

**Example Output:**
```
====================================================================================================
BETTING OPPORTUNITIES FOR WEEK 13
====================================================================================================

✅ Found 1 plays (1 primary, 0 secondary)

🔥 PRIMARY STRATEGY: Unlucky Favorites (spread ≤7)
   Expected: 67-73% ATS, +27-39% ROI
----------------------------------------------------------------------------------------------------

  ✅ BET: TB -3.0
     Game: TB vs ARI
     Last week (W12) luck: -8.1
     This week spread: -3.0 (0-3 favorite)
```

**Output Files:**
- `data/04_output/nfl/todays_plays/nfl_luck_regression_all_teams_week_X_YYYY-MM-DD_HHMMSS.csv` (all teams)
- `data/04_output/nfl/todays_plays/nfl_luck_regression_plays_week_X_YYYY-MM-DD_HHMMSS.csv` (filtered plays)

---

## AWS Automation

### Current Status
- ✅ **NBA Dashboard:** Fully automated (Lambda + EventBridge)
- ⏳ **NFL Plays:** Manual (future: Lambda email alerts with weekly play recommendations)

### NBA Architecture
```
AWS EventBridge (Scheduler)
    ↓ triggers at 7 AM ET daily
AWS Lambda (Python function)
    ↓ clones repo, runs update script
    ↓ pushes to GitHub
Streamlit Cloud
    ↓ auto-detects changes
    ↓ redeploys dashboard
```

### Planned NFL Architecture
```
AWS EventBridge (Scheduler)
    ↓ triggers every Monday at 12 PM ET
AWS Lambda (Python function)
    ↓ runs find_nfl_regression_plays.py
    ↓ analyzes all 32 teams
    ↓ identifies betting opportunities
AWS SNS (Simple Notification Service)
    ↓ sends formatted email
    ✉️ TL;DR: "Week X: 1 Play - TB -3.0 vs ARI"
    ✉️ Warning: Re-run after MNF if needed
    ✉️ Full logs + reasoning
    ✉️ Summary repeated at end
```

**Setup Guide:** See [`docs/AWS_NFL_LAMBDA_SETUP.md`](docs/AWS_NFL_LAMBDA_SETUP.md)

### Rebuilding Lambda Layer

If you get a `ModuleNotFoundError` (e.g., `No module named 'yaml'`):

```bash
echo '🏗️  Rebuilding Lambda Layer...'
cd /Users/thomasmyles/dev/betting

# Clean up old layer
rm -rf lambda_layer
mkdir -p lambda_layer/python

# Install packages for Linux x86_64 (Lambda architecture)
pip install --platform manylinux2014_x86_64 \
  --target=lambda_layer/python \
  --implementation cp \
  --python-version 3.12 \
  --only-binary=:all: \
  --upgrade \
  -r docs/lambda_requirements.txt

# Zip it up
cd lambda_layer
zip -r layer.zip python -q

# Check size (must be under 50 MB)
SIZE=$(ls -lh layer.zip | awk '{print $5}')
printf '   Size: %s (must be under 50 MB)\n' "$SIZE"

echo '📍 Layer location: /Users/thomasmyles/dev/betting/lambda_layer/layer.zip'
```

**Next Steps:**
1. Go to AWS Lambda console → **Layers** → `betting-dashboard-dependencies`
2. Click **Create version**
3. Upload `lambda_layer/layer.zip`
4. Set compatible runtime: **Python 3.12**
5. Click **Create**
6. Update function: Lambda → Functions → `betting-dashboard-daily-update` → Layers → Edit → Increment version

**Full Setup Guide:** See [`docs/AWS_AUTOMATION_CHECKLIST.md`](docs/AWS_AUTOMATION_CHECKLIST.md)

---

## Development

### Running Locally

The dashboard reads from `data/04_output/arbs/` directory. Make sure you have arb data files:
```
data/04_output/arbs/
├── arb_threes_20251121.csv
├── arb_points_20251121.csv
└── ...
```

Generate test data:
```bash
python scripts/find_arb_opportunities.py --markets player_threes
```

### Data Pipeline Overview

```
01_input/         → Raw data (The Odds API, NBA API, Unexpected Points)
02_cache/         → Cached rosters, player-team mappings
03_intermediate/  → Processed props, consensus lines, game logs
04_output/        → Final betting opportunities, backtesting results
```

---

## Screenshots

![Arb Dashboard](./screenshots/20251124_arbitrage_dashboard.png)

---

## Troubleshooting

### Missing Player/Team Data

**Error:** `X player(s) not in cache`

**Solution:**
```bash
python scripts/build_full_roster_cache.py
```

### Streamlit Cache Not Updating

**Error:** "Aggressively caching, not reading new player_team_cache.csv"

**Solution:**
1. Go to [Streamlit Cloud](https://share.streamlit.io/)
2. Find app → **...** → **Reboot app**

### Lambda Function Errors

**Error:** `ModuleNotFoundError: No module named 'yaml'`

**Solution:** Rebuild Lambda layer (see AWS Automation section above)

### API Rate Limits

**Issue:** The Odds API rate limit hit

**Solution:**
- Check usage: https://the-odds-api.com/account/
- Reduce markets in fetch scripts
- Upgrade API plan if needed

---

## Cost Breakdown

### AWS Costs (~$0.40/month)
- Lambda: $0.00 (free tier: 1M requests/month)
- EventBridge: $0.00 (free tier: 14M events/month)
- Secrets Manager: $0.40/month
- CloudWatch Logs: $0.00 (free tier: 5 GB/month)

### API Costs
- The Odds API: Varies by plan (free tier available)
- NBA API: Free

---

## Project Structure

```
betting/
├── api_setup/                    # API configuration
├── backtesting/                  # Historical strategy analysis
├── data/
│   ├── 01_input/                 # Raw data sources
│   ├── 02_cache/                 # Cached rosters/mappings
│   ├── 03_intermediate/          # Processed data
│   └── 04_output/                # Final outputs
│       ├── arbs/                 # Arbitrage opportunities
│       ├── nfl/todays_plays/     # NFL betting plays
│       └── todays_plays/         # NBA betting plays
├── docs/                         # Documentation
│   ├── AWS_AUTOMATION_CHECKLIST.md
│   ├── lambda_function.py
│   └── lambda_requirements.txt
├── implementation/               # Production betting finders
│   ├── find_3pt_underdog_unders_today.py
│   └── find_nfl_regression_plays.py
├── scripts/                      # Data fetching & processing
│   ├── build_season_game_logs.py
│   ├── fetch_nfl_season_lines.py
│   ├── find_arb_opportunities.py
│   └── ...
├── src/                          # Utility modules
│   ├── nfl_team_utils.py
│   ├── odds_utils.py
│   └── config_loader.py
└── streamlit_app/                # Dashboard UI
    ├── app.py
    └── requirements.txt
```

---

## Resources

- [The Odds API Documentation](https://the-odds-api.com/liveapi/guides/v4/)
- [NBA API (nba_api)](https://github.com/swar/nba_api)
- [AWS Lambda Documentation](https://docs.aws.amazon.com/lambda/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Unexpected Points NFL Data](https://docs.google.com/spreadsheets/d/1ktlf_ekms7aI6r0tF_HeX0zaxps-bHWYsgglUReC558/edit?usp=sharing)

---

## Contributing

This is a personal betting analytics project. The strategies are based on historical analysis and are not guaranteed to be profitable. Always gamble responsibly.

**Strategy Development Workflow:**
1. **Backtest** strategy on historical data (`backtesting/`)
2. **Validate** results with sanity checks
3. **Implement** finder script (`implementation/`)
4. **Monitor** performance in production
5. **Iterate** based on results

