# Historical Props Fetching Algorithm

## Overview
Fetch historical player prop lines for every NBA game in 2024-25 season to analyze trends.

## The Algorithm

```
1. Get dates for NBA games
   - Use nba_api to fetch all games from 2024-25 season
   - Extract unique game dates
   - Result: 163 game days (Oct 22, 2024 - Apr 13, 2025)

2. For each date:
   - Get event IDs for each game played on that date
   - API: /v4/historical/sports/basketball_nba/events
   - Timestamp: YYYY-MM-DDT12:00:00Z (noon UTC)
   - Cost: 1 credit per call
   
3. For each event ID:
   - Get odds for game, focusing on 3pt props
   - API: /v4/historical/sports/basketball_nba/events/{event_id}/odds
   - Timestamp: YYYY-MM-DDT20:00:00Z (8 PM UTC / 3 PM ET)
   - Markets: 'player_threes' (or multiple: 'player_points,player_rebounds,player_assists,player_threes')
   - Cost: 10 credits per event per market
```

## Timestamp Strategy

**Why 8 PM UTC (3 PM ET)?**
- NBA games typically start 7-10 PM ET (00:00-03:00 UTC next day)
- Fetching at 8 PM UTC (3 PM ET) captures:
  - ✅ Pre-game lines (before tip-off)
  - ✅ All bookmakers have posted odds
  - ✅ Lines are settled (less movement)
  - ❌ Too late = some games already started
  - ❌ Too early = some odds not posted yet

## Cost Estimation

### Full Season - Player Threes Only
```
163 game days
~1,230 total games
Cost: 1,230 games × 10 credits = 12,300 credits
Budget: 20,000 credits
✅ Feasible with 7,700 credits remaining
```

### Full Season - 4 Markets
```
Markets: player_points, player_rebounds, player_assists, player_threes
Cost: 1,230 games × 40 credits = 49,200 credits
❌ Over budget (need ~$120/month plan)
```

### Recommended Approach
```
1. Fetch all games with player_threes (12,300 credits)
2. Use remaining 7,700 credits for:
   - Re-fetching specific dates if needed
   - Additional markets for interesting games
   - Testing and validation
```

## Implementation Steps

### ✅ Completed
1. **nba_calendar_builder.py**
   - Fetches all 2024-25 games
   - Creates calendar of 163 game dates
   - Saves to `nba_calendar/game_dates_2024_25.json`

2. **fetch_historical_props.py**
   - Fetches historical event IDs for a date
   - Fetches historical odds for each event
   - Parses player props into clean format
   - Saves to CSV

### 🧪 Testing
```bash
cd api_setup
python fetch_historical_props.py
```
- Tests with opening day (Oct 22, 2024)
- 2 games × 10 credits = ~20 credits
- Validates API key and data format

### 📦 Next: Full Season Fetch
```python
# Load game dates
with open('nba_calendar/game_dates_2024_25.json') as f:
    calendar = json.load(f)
    dates = calendar['dates']

# Fetch each date
for date in dates:
    df = fetch_date_props(date, markets='player_threes', save=True)
    # Takes ~2 minutes per date
    # Total time: ~5-6 hours for full season
```

## Output Format

### Files Created
```
historical_props/
├── props_2024-10-22_player_threes.csv
├── props_2024-10-23_player_threes.csv
├── props_2024-10-24_player_threes.csv
└── ...
```

### CSV Columns
```
player          | String  | "LeBron James"
game            | String  | "LAL @ MIN"
game_time       | ISO8601 | "2024-10-22T23:30:00Z"
market          | String  | "player_threes"
line            | Float   | 2.5
over_odds       | Int     | -110
under_odds      | Int     | -110
bookmaker       | String  | "draftkings"
```

## Next Analysis Steps

After collecting historical props:

1. **Merge with actual results**
   ```python
   # NBA API: actual 3PT made
   # Historical props: what the line was
   # Join on: player, date, game
   # Result: Did they hit over/under?
   ```

2. **Find trend opportunities**
   ```python
   # Find: 8+ game under streaks
   # Check: What was the prop line?
   # Analyze: Regression opportunities
   ```

3. **Backtest strategy**
   ```python
   # Strategy: Bet over when player is on long under streak
   # Calculate: Win rate, ROI, Kelly criterion
   ```

## API Reference

Based on official example: `the-odds-api/samples-python/historical_event_odds.py`

**Key differences from current/live odds API:**
- Parameter: `api_key` (underscore) not `apiKey` (camelCase)
- Response wrapped in `{'data': ..., 'timestamp': ..., 'previous_timestamp': ..., 'next_timestamp': ...}`
- Cost: 10x higher than live odds (10 credits vs 1 credit per market)
- Available: Only on paid plans ($30/month minimum)

## Notes

- Historical odds available from June 6, 2020
- Snapshots at 5-minute intervals (since Sept 2022)
- Player props available after May 3, 2023
- Data might have occasional errors (rare)
- Rate limit: Space requests 0.5 seconds apart

