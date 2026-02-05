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

```bash
# Step 1: Build team history from cached game logs
python src/player_team_history/01_build.py

# Step 2: Analyze any failures
python src/player_team_history/02_analyze_failures.py

# Step 3: Inspect cache (optional)
python src/player_team_history/03_cache.py --stats

# Step 4: Validate output
python src/player_team_history/04_validate.py

# Step 5: Export to S3
aws s3 cp ~/Downloads/tmp/player_team_history/history.parquet \
  s3://nba-betting-mt/nba/player_team_history/history.parquet
```

### Output Files

All files in `~/Downloads/tmp/player_team_history/`:

- `history.parquet` - Final team history (player, team, valid_from, valid_to)
- `checkpoint.parquet` - For resuming interrupted builds
- `failures.txt` - Detailed failure report
- `cache/players/*.parquet` - Complete player game logs
- `cache/seasons/*.parquet` - Individual season caches

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

## Name Normalization

The pipeline handles name variations across APIs:

**File:** `src/player_team_history/name_normalization.py`

```python
# Odds API → NBA API mappings
get_odds_api_to_nba_mappings() = {
    'Alfred Joel Horford Reynoso': 'Al Horford',      # Full legal name
    'Cameron Johnson': 'Cam Johnson',                  # Shortened name
    'Christian James Mccollum': 'Cj Mccollum',        # Full legal name
    'Scottie Pippen Jr': 'Scotty Pippen Jr',          # Spelling variation
    # ... 50+ more mappings
}
```

**Common patterns:**
- Full legal names → Common names (`Christian James Mccollum` → `Cj Mccollum`)
- Shortened names (`Cam` ↔ `Cameron`)
- Initials (`P.J.` → `Pj`, `O.G.` → `Og`)
- Spelling variations (`Scotty` vs `Scottie`)
- Hyphenation (`Dorian Finney Smith` → `Dorian Finney-Smith`)

## Cache Strategy

**Two-tier caching for 100x speedup:**

1. **Season cache** (`cache/seasons/Player_Name_2023-24.parquet`) - Individual seasons
2. **Player cache** (`cache/players/Player_Name.parquet`) - Complete player history (only saved when all seasons succeed)

The build checks player cache first, falls back to season cache, then fetches from NBA API only if needed.

## Build Process

```bash
# Test with sample (fast iteration)
python src/player_team_history/01_build.py --sample 100 --verbose

# Resume from checkpoint
python src/player_team_history/01_build.py --resume

# Force fresh fetch (bypass cache)
python src/player_team_history/01_build.py --no-cache

# Full build (uses cache, typically 5-10 minutes)
python src/player_team_history/01_build.py
```

## Validation Checks

The validation script (`04_validate.py`) runs comprehensive checks:

**Critical checks:**
- No duplicate stints
- Valid date ranges (valid_from ≤ valid_to)
- Valid NBA team codes
- No overlapping stints
- NULL valid_to only on final stint per player
- Chronological order within each player
- Date sanity (1946 to today + 1 year)

**Warning checks:**
- Very short stints (< 7 days, may be 10-day contracts)
- Consecutive same-team stints (may need consolidation)

**Statistics:**
- Stint distribution breakdown (1, 2-3, 4-6, 7+ stints)
- Players with most team changes
- Active players count

## Iterating on Failures

When the build completes with failures:

1. Check `failures.txt` for details
2. Run `python src/player_team_history/02_analyze_failures.py`
3. The analyzer suggests exact code to add to `get_odds_api_to_nba_mappings()`
4. Add mappings to `name_normalization.py`
5. Re-run `01_build.py` (uses cache, only re-processes failed players)

## Querying Output

```bash
# Query from S3 with DuckDB
duckdb -c "
SELECT player_normalized, team, valid_from, valid_to
FROM 's3://nba-betting-mt/nba/player_team_history/history.parquet'
WHERE player_normalized = 'Anthony Davis'
ORDER BY valid_from;
"
```

Example output:
```
player_normalized | team | valid_from  | valid_to
------------------+------+------------+------------
Anthony Davis     | NOH  | 2012-10-31 | 2013-04-10
Anthony Davis     | NOP  | 2013-10-30 | 2019-03-24
Anthony Davis     | LAL  | 2019-10-22 | NULL
```

## Known Gaps

- **ESPN API mappings**: Not yet comprehensive (most ESPN data matches after basic normalization)
- **Retired players**: May fail if not in current NBA API data
- **Rookies**: May have incomplete data if recently drafted

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
