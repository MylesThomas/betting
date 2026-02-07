# ESPN Play-by-Play Data Solution

## Summary

✅ **Successfully extracted game logs with timestamps from ESPN API**

Successfully retrieved and parsed play-by-play data for Bucks vs Pelicans game on Feb 4, 2026.

## Data Source

**ESPN API** (http://site.api.espn.com)

- ✅ Free, no authentication required
- ✅ Includes play-by-play with timestamps
- ✅ Includes boxscore data
- ✅ Player IDs and names included
- ✅ Quarter/time information for each play

## Files Created

Located in `/tmp/`:

### 1. Raw Data
- `bucks_pelicans_pbp_20260204.json` - Full game JSON from ESPN API
- `espn_playbyplay.json` - Sample play-by-play data

### 2. Parsed Data
- `bucks_pelicans_plays_20260204.csv` - All 472 plays (scoring + non-scoring)
- `bucks_pelicans_scoring_20260204.csv` - 132 scoring plays only

### 3. Monte Carlo Ready Format
- **`bucks_pelicans_minute_by_minute.csv`** ⭐ THIS IS THE KEY FILE
  - Every player's cumulative points at every minute of the game
  - 1,007 rows (19 players × 53 minutes including OT)
  - Format: `player_id, player_name, minute, cumulative_points`
  
- `bucks_pelicans_timeline.csv` - All scoring events with exact timestamps
- `bucks_pelicans_quarter_splits.csv` - Quarter-by-quarter breakdown

## What the Minute-by-Minute Data Looks Like

```csv
player_id,player_name,minute,cumulative_points
4397136,Saddiq Bey,0,0
4397136,Saddiq Bey,1,0
4397136,Saddiq Bey,2,3      # Scored 3-pointer at minute 2
4397136,Saddiq Bey,3,3
4397136,Saddiq Bey,4,3
4397136,Saddiq Bey,5,5      # Scored 2 points at minute 5
...
4397136,Saddiq Bey,48,20    # End of regulation: 20 points
4397136,Saddiq Bey,49,20
4397136,Saddiq Bey,50,22    # OT: scored 2 more
4397136,Saddiq Bey,51,22
4397136,Saddiq Bey,52,22    # Final: 22 points
```

## How to Use This for Monte Carlo Simulation

### For Backtesting

```python
# Load minute-by-minute data
df = pd.read_csv('bucks_pelicans_minute_by_minute.csv')

# Filter for a specific player
player_data = df[df['player_name'] == 'Trey Murphy III']

# Simulate from minute 24 (halftime)
current_minute = 24
current_points = player_data[player_data['minute'] == current_minute]['cumulative_points'].iloc[0]
final_points = player_data[player_data['minute'] == player_data['minute'].max()]['cumulative_points'].iloc[0]

print(f"At minute {current_minute}: {current_points} points")
print(f"Final: {final_points} points")
print(f"Scored {final_points - current_points} points after minute {current_minute}")

# For Monte Carlo: compare to prop line
prop_line = 30.5
actual_outcome = 'over' if final_points > prop_line else 'under'
```

## How to Get Data for Any Game

### Step 1: Find games on a specific date

```python
import requests

date_str = "20260204"  # YYYYMMDD format
url = f"http://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={date_str}"
response = requests.get(url)
games = response.json()['events']

for game in games:
    competitors = game['competitions'][0]['competitors']
    away_team = competitors[1]['team']['displayName']
    home_team = competitors[0]['team']['displayName']
    game_id = game['id']
    print(f"{game_id}: {away_team} @ {home_team}")
```

### Step 2: Get play-by-play for a game

```python
game_id = "401810584"  # From step 1
url = f"http://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary?event={game_id}"
response = requests.get(url)
game_data = response.json()

plays = game_data['plays']  # List of all plays
boxscore = game_data['boxscore']  # Player stats
```

### Step 3: Parse to minute-by-minute format

Use the script: `tmp/parse_to_monte_carlo_format.py`

## Scripts Available

### 1. `explore_nba_data_sources.py`
Tests multiple data sources (NBA API, ESPN, Basketball Reference).
Result: ESPN works best.

### 2. `get_bucks_pelicans_game.py`
Gets play-by-play for a specific game.
- Searches for game by team names and date
- Downloads full play-by-play data
- Parses scoring plays
- Extracts boxscore stats

### 3. `parse_to_monte_carlo_format.py`
Converts raw play-by-play into minute-by-minute format.
- Creates player timeline with timestamps
- Fills in gaps (every minute, not just scoring plays)
- Calculates cumulative stats
- Extracts quarter splits

## What We Have Now

✅ **Game logs with timestamps** - ESPN play-by-play data
✅ **Minute-by-minute scoring** - Cumulative points every minute
✅ **Quarter splits** - Points by quarter for each player
✅ **Boxscore data** - Final stats (points, minutes, FG%, etc.)

## What We Still Need (for Full Monte Carlo)

### Historical Data Collection
- Need to collect this data for ~20 games per player
- Can iterate through dates and download each game
- Store in database or CSV files

### Player Minutes Played by Quarter
ESPN boxscore doesn't include minutes by quarter in the API response.

**Options:**
1. **Infer from play-by-play**: Track when player enters/exits (substitution plays)
2. **Use NBA API** (requires fixing SSL cert issue)
3. **Scrape Basketball Reference** (has quarter minutes in boxscore tables)
4. **Start with season averages**: Use avg MPG and assume ~equal split by half

**Recommendation for V1:** 
Start with season averages for minutes. Once model is working, add more granular minute tracking.

## Next Steps

1. **Build data collection pipeline**
   - Script to download games for date range
   - Parse all games to minute-by-minute format
   - Store in structured format (CSV or SQLite)

2. **Create player historical profiles**
   - Aggregate last 20 games for each player
   - Calculate distributions (PPM, minutes, etc.)
   - Store for quick lookup during simulation

3. **Implement Monte Carlo simulation**
   - Use the variable structure from the planning doc
   - Start with V1 (simple model)
   - Backtest on historical games

4. **Build backtesting framework**
   - Run simulation at every minute of every game
   - Calculate calibration metrics (Brier score, ROC AUC)
   - Compare to baselines

## Example: Trey Murphy III from this game

**Final stats:** 38 points in 40 minutes (went to OT)

**Timeline:**
- Minute 0 (start): 0 points
- Minute 12 (end Q1): 15 points
- Minute 24 (halftime): 24 points
- Minute 36 (end Q3): 24 points (cold 3Q!)
- Minute 48 (end regulation): 32 points
- Minute 53 (end OT): 38 points

**Monte Carlo scenario:**
If prop line was "Over 30.5 points", at halftime:
- He has 24 points
- Needs 6.5 more
- Still 24+ minutes to play
- Probability would depend on his historical 2H scoring rate

This is exactly the scenario our Monte Carlo model will handle!
