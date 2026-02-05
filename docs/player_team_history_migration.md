# Player Team History - Migration Guide

## Problem

The old `player_team_cache.csv` had a single timestamp for all players, making it impossible to correctly join team data for historical games after trades happen.

**Example of the issue:**
- Anthony Davis was traded from LAL → DAL on Feb 6, 2026
- Old cache: `Anthony Davis,DAL,2026-02-06` (only shows current team)
- Problem: Can't get his team for games before the trade (should be LAL)

## Solution

New **date-range based team history** that tracks every team stint:

```csv
player_normalized,team,valid_from,valid_to
Anthony Davis,LAL,2019-10-22,2026-02-05
Anthony Davis,DAL,2026-02-07,NULL
```

Now you can join props data by game date to get the correct team at that time.

## How to Build the History

### Initial Build (Full Career)

```bash
# Fetch career game logs for all active players (takes 10-15 mins)
python scripts/build_player_team_history_from_gamelogs.py
```

This creates:
- `data/02_cache/player_team_history.csv` (local)
- `s3://nba-betting-mt/data/02_cache/player_team_history.csv` (S3)

### Quick Test (Current Season Only)

```bash
# Faster for testing - only current season
python scripts/build_player_team_history_from_gamelogs.py --current-season-only
```

### Update After Trades

```bash
# Re-run the build script to update with latest game logs
python scripts/build_player_team_history_from_gamelogs.py
```

Run this:
- Daily during trade deadline week
- Weekly during season
- After major trades

## How to Use in Your Scripts

### Before (Old Way)

```python
from src.team_utils_simple import add_team_column_simple

# This only worked for current teams
props_df = add_team_column_simple(props_df, player_col='player')
# Problem: Wrong team for historical games after trades!
```

### After (New Way)

```python
from src.team_history_utils import add_team_from_history

# This correctly handles historical team assignments
props_df = add_team_from_history(props_df, player_col='player', date_col='game_date')
# ✅ Correct team for every game date, even after trades
```

### Example

```python
import pandas as pd
from src.team_history_utils import add_team_from_history

# Your props data with game dates
props_df = pd.DataFrame({
    'player': ['Anthony Davis', 'Anthony Davis'],
    'game_date': ['2026-01-15', '2026-02-10'],  # Before and after trade
    'points': [28, 25]
})

# Add team column using game date
props_df = add_team_from_history(props_df, player_col='player', date_col='game_date')

print(props_df)
# Output:
#          player  game_date  points team
#   Anthony Davis 2026-01-15      28  LAL  ✅ Correct (before trade)
#   Anthony Davis 2026-02-10      25  DAL  ✅ Correct (after trade)
```

## Other Functions

### Get Team for Specific Player/Date

```python
from src.team_history_utils import get_player_team_at_date

team = get_player_team_at_date('Anthony Davis', '2026-01-15')
print(team)  # 'LAL'
```

### Get Full Player History

```python
from src.team_history_utils import get_team_history_for_player

history = get_team_history_for_player('Anthony Davis')
print(history)
#   player_normalized team valid_from   valid_to
#     Anthony Davis  LAL 2019-10-22 2026-02-05
#     Anthony Davis  DAL 2026-02-07       None
```

## Migration Checklist

Files that need updating:

- [ ] `scripts/find_nba_role_spread_plays.py` - uses `player_team_cache`
- [ ] `analysis/detect_nba_dnp_scenarios.py` - uses `player_team_cache`
- [ ] Any other scripts that join props with team data

For each file:
1. Change import from `team_utils_simple` → `team_history_utils`
2. Change function from `add_team_column_simple()` → `add_team_from_history()`
3. Add `date_col='game_date'` parameter (adjust column name to match your data)
4. Test with data that spans trade deadline

## Data Schema

### player_team_history.csv

```
player_normalized  team  valid_from  valid_to
------------------+------+-----------+----------
Anthony Davis      LAL   2019-10-22  2026-02-05
Anthony Davis      DAL   2026-02-07  NULL
Lebron James       LAL   2018-07-09  NULL
...
```

**Columns:**
- `player_normalized`: Normalized player name (for matching)
- `team`: Team abbreviation (LAL, GSW, etc.)
- `valid_from`: First game date with team (date)
- `valid_to`: Last game date with team (date), NULL = current team

**Join Logic:**
```sql
WHERE game_date >= valid_from 
  AND (valid_to IS NULL OR game_date <= valid_to)
```

## Keep Both Caches?

**Yes** - keep both for different use cases:

### player_team_cache.csv (Old)
- **Use for:** Current team lookups (today's games, future props)
- **Update:** Run `build_full_roster_cache.py` weekly
- **Pro:** Simple, fast lookups when you don't need history

### player_team_history.csv (New)
- **Use for:** Historical analysis, joining past game props
- **Update:** Run `build_player_team_history_from_gamelogs.py` after trades
- **Pro:** Correct team for any game date, handles trades

## Testing

```bash
# Test the history utils
python src/team_history_utils.py

# Expected output:
# - Loads history
# - Tests get_player_team_at_date()
# - Tests add_team_from_history()
# - Shows example player history
```

## FAQ

**Q: What if a player is in the roster cache but not in history?**

A: History only includes players who have played games. New players (injuries, recent trades) won't have game logs yet. Use the current roster cache for those.

**Q: Does this include preseason games?**

A: No, only regular season + playoffs (per your requirements).

**Q: What if a player didn't log minutes?**

A: We only track games they actually played in. If they were on the roster but inactive, they won't have a history entry for that date. This is intentional - you're joining by game date where they had props.

**Q: How far back does history go?**

A: Full career as far as NBA API has data (typically ~2010+).

**Q: Can I run this for just one player?**

A: Currently no, but you could modify the script to accept a player list. For now, it fetches all active players from the roster cache.
