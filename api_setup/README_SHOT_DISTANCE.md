# NBA Shot Distance Data - Setup & Usage Guide

## Overview

We now have complete shot-by-shot data for NBA players including **exact shot distance** (in feet), location coordinates, shot type, and game context.

## What's Available

### Shot Data Fields
Each shot includes:
- `SHOT_DISTANCE` - Distance in feet (0-83 feet)
- `LOC_X`, `LOC_Y` - Court coordinates
- `SHOT_MADE_FLAG` - 1 = made, 0 = missed
- `ACTION_TYPE` - Specific shot type (Dunk, Layup, Jump Shot, etc.)
- `SHOT_TYPE` - 2PT or 3PT Field Goal
- `SHOT_ZONE_BASIC` - Restricted Area, In The Paint, Above the Break 3, etc.
- `GAME_DATE` - When the shot occurred
- `PERIOD`, `MINUTES_REMAINING`, `SECONDS_REMAINING` - Game context
- `HTM`, `VTM` - Home and visiting teams

### Distance Ranges
- **0-3 feet** - At the rim (dunks, layups)
- **3-6 feet** - Short range (floaters, short hooks)
- **6-10 feet** - Floater range
- **10-16 feet** - Mid-range
- **16+ feet** - Long two / three-pointers

## Quick Start

### 1. Setup Script (Already Run)
```bash
# Fetch shot charts for sample players (5 players, ~8 seconds)
python scripts/test_fetch_shot_charts.py
```

✅ **We've already run this!** Data saved for:
- LeBron James
- Stephen Curry
- Giannis Antetokounmpo
- Luka Doncic
- Nikola Jokic

**Output**: `/Users/thomasmyles/dev/betting/data/01_input/nba_api/shot_charts/2024_25/*.csv`

### 2. Fetch All Players (~500 players, ~10 minutes)
```bash
# Fetch shot charts for ALL active NBA players
python scripts/fetch_all_nba_shot_charts.py --auto
```

**Features**:
- Automatically resumes if interrupted (Ctrl+C)
- Saves progress after each player
- Creates summary CSV with all player stats
- Rate-limited to respect NBA API

### 3. Analyze Close-Range Shooting
```bash
# Analyze shots within 6 feet for all downloaded players
python analysis/analyze_nba_close_range_shooting.py
```

## Example Results (From Test Run)

### Close-Range Shooting (0-6 feet)
| Player | Close Range Attempts | FG% (0-6 ft) |
|--------|---------------------|--------------|
| Giannis Antetokounmpo | 869 | 70.3% |
| Nikola Jokic | 697 | 67.7% |
| Luka Doncic | 428 | 67.5% |
| LeBron James | 533 | 66.0% |
| Stephen Curry | 271 | 57.6% |

### Distance Breakdown (Giannis)
- **Rim (0-3 ft)**: 749 attempts, 74.9% FG%
- **Short (3-6 ft)**: 120 attempts, 41.7% FG%
- **Floater (6-10 ft)**: 74 attempts, 33.8% FG%

## Usage Examples

### Basic Analysis
```python
import pandas as pd

# Load any player's shot data
df = pd.read_csv('data/01_input/nba_api/shot_charts/2024_25/LeBron_James_2544.csv')

# Filter shots within 6 feet
close_shots = df[df['SHOT_DISTANCE'] <= 6]

# Calculate FG% within 6 feet
makes = close_shots['SHOT_MADE_FLAG'].sum()
attempts = len(close_shots)
fg_pct = (makes / attempts) * 100
print(f"FG% within 6 feet: {fg_pct:.1f}%")
```

### Advanced Filtering
```python
# At-rim shots only (0-3 feet)
rim_shots = df[df['SHOT_DISTANCE'] <= 3]

# Driving layups only
driving_layups = df[df['ACTION_TYPE'] == 'Driving Layup Shot']

# Close-range shots vs specific opponent
vs_warriors = df[(df['SHOT_DISTANCE'] <= 6) & 
                 ((df['HTM'] == 'GSW') | (df['VTM'] == 'GSW'))]

# Shots in 4th quarter
fourth_quarter = df[df['PERIOD'] == 4]

# Shots in last 2 minutes of close games
clutch_shots = df[(df['PERIOD'] == 4) & 
                  (df['MINUTES_REMAINING'] <= 2)]
```

### Compare Players
```python
from glob import glob

# Load all players
shot_files = glob('data/01_input/nba_api/shot_charts/2024_25/*.csv')

results = []
for file in shot_files:
    df = pd.read_csv(file)
    player_name = df['PLAYER_NAME'].iloc[0]
    
    # Analyze 0-6 foot shots
    close = df[df['SHOT_DISTANCE'] <= 6]
    fg_pct = (close['SHOT_MADE_FLAG'].sum() / len(close) * 100)
    
    results.append({
        'player': player_name,
        'attempts': len(close),
        'fg_pct': fg_pct
    })

# Convert to DataFrame and sort
comparison = pd.DataFrame(results).sort_values('fg_pct', ascending=False)
print(comparison)
```

## File Structure

```
data/01_input/nba_api/shot_charts/
├── 2024_25/
│   ├── LeBron_James_2544.csv          # Individual player files
│   ├── Stephen_Curry_201939.csv
│   ├── ... (one per player)
├── 2024_25_progress.json              # Resume tracking
└── 2024_25_summary.csv                # Summary stats for all players
```

## Files Created

### Setup & Data Collection
- `api_setup/nba_shot_chart_setup.py` - Core functions for fetching shot data
- `scripts/fetch_all_nba_shot_charts.py` - Fetch all ~500 players (with resume)
- `scripts/test_fetch_shot_charts.py` - Quick test with 5 players

### Analysis
- `analysis/analyze_nba_close_range_shooting.py` - Analyze 0-6 foot shooting

## Betting Applications

### 1. Close-Range Efficiency Analysis
- Identify players shooting way above/below their typical close-range %
- Look for matchup-based close-range opportunities
- Track trends over time (hot/cold streaks at the rim)

### 2. Shot Selection Analysis
- Which players are taking more/fewer close-range shots than usual?
- Has injury or role change affected shot profile?
- Compare shot distance vs. defensive matchup

### 3. Game Script Analysis
- Do players shoot better/worse at rim in close games?
- Effect of home vs away on close-range efficiency
- Back-to-back game impact on rim attempts/efficiency

### 4. Prop Betting
- Points props: Is player getting to rim more/less than usual?
- Scoring efficiency: Close-range % predicts scoring outbursts
- Usage correlation: Rim attempts correlate with minutes/usage

## API Rate Limits

- **NBA Stats API**: Free, no API key required
- **Rate Limit**: ~0.6 seconds between requests (built into scripts)
- **Data Freshness**: Updates daily with latest games
- **Historical**: Can fetch any season back to ~2014-15

## Next Steps

1. ✅ Test with 5 players (DONE)
2. ⏳ Fetch all ~500 players: `python scripts/fetch_all_nba_shot_charts.py --auto`
3. 📊 Build custom analysis for your betting strategy
4. 🔄 Schedule daily updates to get latest games

## Questions?

- Shot data includes every field goal attempt with exact distance
- Can filter by any distance range (0-6 feet is just an example)
- Can analyze by shot type, opponent, game situation, etc.
- All data saved as CSV files for easy analysis

