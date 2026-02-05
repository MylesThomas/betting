# Player Team History

Track player team assignments over time using career game logs.

## Problem

After trades, static roster caches show only current teams, making it impossible to correctly join team data for historical games.

**Example:**
- Anthony Davis traded from LAL → DAL on Feb 6, 2026
- Old cache: Only shows `DAL`
- Problem: Can't get his team for games before the trade

## Solution

Date-range based team history:

```
player_normalized  team  valid_from  valid_to
Anthony Davis      LAL   2019-10-22  2026-02-05
Anthony Davis      DAL   2026-02-07  NULL
```

Join props by game date to get correct team at that time.

## Quick Start

### Build the History

```bash
# Full career (takes 10-15 mins)
python -m src.player_team_history.builder

# Current season only (faster for testing)
python -m src.player_team_history.builder --current-season-only
```

**Output:**
- S3: `s3://nba-betting-mt/data/02_cache/player_team_history.parquet`
- Local backup: `data/02_cache/player_team_history.parquet`

**Update frequency:**
- Daily during trade deadline week
- Weekly during season
- After major trades

### Use in Your Scripts

```python
from src.player_team_history import add_team_from_history

# Your props data with game dates
props_df = pd.DataFrame({
    'player': ['Anthony Davis', 'Anthony Davis'],
    'game_date': ['2026-01-15', '2026-02-10'],
    'points': [28, 25]
})

# Add team column using game dates
props_df = add_team_from_history(props_df, player_col='player', date_col='game_date')

# Result:
#   player          game_date  points  team
#   Anthony Davis   2026-01-15  28     LAL   ✅ Before trade
#   Anthony Davis   2026-02-10  25     DAL   ✅ After trade
```

## API Reference

### `add_team_from_history(df, player_col='player', date_col='game_date')`

Add team column to DataFrame based on game dates. **Main function.**

**Args:**
- `df`: DataFrame with player names and game dates
- `player_col`: Name of player column (default: 'player')
- `date_col`: Name of game date column (default: 'game_date')

**Returns:**
- DataFrame with new 'team' column

**Raises:**
- `ValueError`: If date column missing or player/date not found

**Example:**
```python
props_df = add_team_from_history(props_df)
```

### `get_player_team_at_date(player_name, game_date, history_df=None)`

Get a player's team on a specific date.

**Args:**
- `player_name`: Player name (will be normalized)
- `game_date`: Game date (string, datetime, or date object)
- `history_df`: Optional pre-loaded history (loads from S3 if None)

**Returns:**
- Team abbreviation (e.g., 'LAL')

**Raises:**
- `ValueError`: If player or date not found

**Example:**
```python
from src.player_team_history import get_player_team_at_date

team = get_player_team_at_date('Anthony Davis', '2026-01-15')
# Returns: 'LAL'
```

### `get_team_history_for_player(player_name, history_df=None)`

Get full team history for one player.

**Args:**
- `player_name`: Player name (will be normalized)
- `history_df`: Optional pre-loaded history

**Returns:**
- DataFrame with player's full team history

**Raises:**
- `ValueError`: If player not found

**Example:**
```python
from src.player_team_history import get_team_history_for_player

history = get_team_history_for_player('Anthony Davis')
# Returns:
#   player_normalized  team  valid_from  valid_to
#   Anthony Davis      LAL   2019-10-22  2026-02-05
#   Anthony Davis      DAL   2026-02-07  NULL
```

### `load_team_history()`

Load full team history from S3.

**Returns:**
- DataFrame with all player team history

**Example:**
```python
from src.player_team_history import load_team_history

history_df = load_team_history()
# Use for multiple lookups to avoid reloading
```

## Data Schema

### player_team_history.parquet

Stored in: `s3://nba-betting-mt/data/02_cache/player_team_history.parquet`

**Columns:**

| Column | Type | Description |
|--------|------|-------------|
| `player_normalized` | string | Normalized player name (for matching) |
| `team` | string | Team abbreviation (LAL, GSW, etc.) |
| `valid_from` | date | First game date with team |
| `valid_to` | date | Last game date with team (NULL = current) |

**Join Logic:**
```sql
WHERE game_date >= valid_from 
  AND (valid_to IS NULL OR game_date <= valid_to)
```

**Example rows:**
```
player_normalized  team  valid_from  valid_to
Anthony Davis      LAL   2019-10-22  2026-02-05
Anthony Davis      DAL   2026-02-07  NULL
Lebron James       LAL   2018-07-09  NULL
Kawhi Leonard      LAC   2019-07-10  NULL
```

## Data Source

**NBA API PlayerGameLogs:**
- Career game logs for all active players
- Regular season + Playoffs only (no preseason)
- Only includes games where player logged minutes
- Grouped by consecutive games with same team
- Date ranges created from first/last game with each team

**Why game logs vs transactions?**
- More reliable and complete
- Ground truth of actual games played
- Handles injuries, suspensions naturally
- No need for transaction API parsing

## Migration Guide

### Files That Need Updating

Scripts that use `player_team_cache` need migration:

- [ ] `scripts/find_nba_role_spread_plays.py`
- [ ] `analysis/detect_nba_dnp_scenarios.py`
- [ ] Any script joining props with team data

### Before (Old Way)

```python
from src.team_utils_simple import add_team_column_simple

# Only worked for current teams
props_df = add_team_column_simple(props_df, player_col='player')
# ❌ Wrong team for historical games after trades
```

### After (New Way)

```python
from src.player_team_history import add_team_from_history

# Correctly handles historical team assignments
props_df = add_team_from_history(props_df, player_col='player', date_col='game_date')
# ✅ Correct team for every game date
```

### Migration Steps

For each file:

1. **Update import:**
   ```python
   # Old
   from src.team_utils_simple import add_team_column_simple
   
   # New
   from src.player_team_history import add_team_from_history
   ```

2. **Update function call:**
   ```python
   # Old
   df = add_team_column_simple(df, player_col='player')
   
   # New
   df = add_team_from_history(df, player_col='player', date_col='game_date')
   ```

3. **Ensure date column exists:**
   - Your DataFrame must have a date column (e.g., 'game_date')
   - Must be in format: 'YYYY-MM-DD' or datetime object

4. **Test with trade scenarios:**
   - Test with data spanning before/after trade deadline
   - Verify correct teams for both periods

## Keep Both Caches?

**Yes** - different use cases:

### player_team_cache.csv (Old)
- **Use:** Current team lookups (today's games, future props)
- **Update:** Weekly via `build_full_roster_cache.py`
- **Pro:** Simple, fast for current teams

### player_team_history.parquet (New)
- **Use:** Historical analysis, past game props
- **Update:** After trades via `build_player_team_history.py`
- **Pro:** Correct team for any date, handles trades

## Testing

```bash
# Test the utils
python -m src.player_team_history.utils

# Expected output:
# - Loads history from S3
# - Tests get_player_team_at_date()
# - Tests add_team_from_history()
# - Shows example player history
```

## FAQ

**Q: What if a player hasn't played any games yet?**

A: History only includes players with game logs. Use current roster cache for inactive/new players.

**Q: Does this include preseason?**

A: No, regular season + playoffs only.

**Q: What if player was on roster but didn't play?**

A: Only games where they logged minutes are tracked. This is intentional - you're joining by game dates where they had props.

**Q: How far back does history go?**

A: Full career via NBA API (typically 2010+).

**Q: What happens if player/date not found?**

A: **Raises `ValueError`** - fails fast with clear error message showing what's missing.

**Q: Why parquet instead of CSV?**

A: Better compression, faster reads, proper date types, no parsing needed.

## Architecture

```
src/player_team_history/
├── __init__.py       # Package exports
├── builder.py        # Build history from game logs
├── utils.py          # Query and join functions
└── README.md         # This file

data/02_cache/
└── player_team_history.parquet  # Local backup

s3://nba-betting-mt/data/02_cache/
└── player_team_history.parquet  # Source of truth
```

**Build flow:**
1. Fetch roster cache → get active players
2. For each player: fetch career game logs (NBA API)
3. Group consecutive games by team → create date ranges
4. Save to parquet → upload to S3

**Query flow:**
1. Load from S3 (once)
2. Join DataFrame using game dates
3. Return correct team for each row

## Development

### Run Builder

```bash
# Full career
python -m src.player_team_history.builder

# Current season only
python -m src.player_team_history.builder --current-season-only
```

### Run Tests

```bash
python -m src.player_team_history.utils
```

### Import in Scripts

```python
from src.player_team_history import (
    add_team_from_history,        # Main function
    get_player_team_at_date,      # Single lookup
    get_team_history_for_player,  # Full player history
    load_team_history             # Load full dataset
)
```
