# Player Name Matching Tools

This document explains the tools built to handle player name matching between different data sources (props, game results, etc.)

## Problem

Player names vary between data sources:
- **Accents**: Props have "Luka Doncic", game results have "Luka Dončić"
- **Suffixes**: Props have "Derrick Jones", game results have "Derrick Jones Jr."
- **Nicknames**: Props have "Herb Jones", game results have "Herbert Jones"
- **Initials**: Props have "P.J. Washington", game results have "Pj Washington"

## Solution

### 1. Core Utility: `src/player_name_utils.py`

Reusable functions for player name normalization:

```python
from player_name_utils import normalize_player_name

# Normalize a single name
name = normalize_player_name("Luka Dončić")  # -> "Luka Doncic"

# Normalize a DataFrame column
from player_name_utils import normalize_player_names_df
df = normalize_player_names_df(df, 'player')
```

**What it does:**
- Removes accents/diacritics (ć → c, ö → o)
- Standardizes suffixes (Jr. → Jr, removes III/II)
- Applies known name mappings (Herb Jones → Herbert Jones)
- Handles initial variations (P.J. → Pj)

**Key Functions:**
- `normalize_player_name(name)` - Normalize a single player name
- `normalize_player_names_df(df, col)` - Normalize names in a DataFrame
- `get_name_mappings()` - Get dictionary of known name variations
- `check_player_name_match(props_df, results_df)` - Check which players match
- `print_name_mismatch_report()` - Print detailed mismatch analysis

### 2. Debugging Script: `scripts/debug_player_name_mismatches.py`

Standalone script to diagnose name matching issues:

```bash
python scripts/debug_player_name_mismatches.py
```

**Output:**
1. Shows which names changed during normalization
2. Lists players in props but not in game results
3. Suggests possible matches using fuzzy matching
4. Shows which specific player-date combinations are missing
5. Identifies top players with most missing dates

**Example Output:**
```
Players in props but NOT in game results:
  Carlton Carrington
    Possible matches: ['Bub Carrington']
    
  Derrick Jones
    Possible matches: ['Derrick Jones Jr']
```

### 3. Verification Script: `scripts/20251120_verify_overlap_w_2425_3pt_prop_data.py`

Production script that joins props with game results:

```bash
python scripts/20251120_verify_overlap_w_2425_3pt_prop_data.py
```

**What it does:**
1. Loads prop data and game results
2. Normalizes player names using the utility
3. Performs left join (props on left)
4. Checks for NULL game results
5. Outputs detailed diagnostics
6. Saves merged data to `data/props_with_game_results_2024_25.csv`

**Current Match Rate: 93.96%**

The remaining ~6% are typically:
- Players who were injured/inactive on those dates
- Postponed/cancelled games
- Special tournament games (NBA Cup, etc.)

## How to Fix New Name Mismatches

When you encounter a new name mismatch:

### Option 1: Add to Name Mappings (Preferred)

Edit `src/player_name_utils.py`:

```python
def get_name_mappings():
    return {
        # ... existing mappings ...
        
        # Add your new mapping
        'New Player Name': 'Actual Name In Results',
    }
```

### Option 2: Update Normalization Logic

For pattern-based fixes, edit `normalize_player_name()` in `src/player_name_utils.py`:

```python
def normalize_player_name(name, keep_case=False):
    # ... existing logic ...
    
    # Add your pattern-based fix
    name = name.replace('Some Pattern', 'Fixed Pattern')
    
    return name
```

## Workflow

### When matching new datasets:

1. **Run the verification script:**
   ```bash
   python scripts/20251120_verify_overlap_w_2425_3pt_prop_data.py
   ```

2. **If match rate < 100%, run the debugging script:**
   ```bash
   python scripts/debug_player_name_mismatches.py
   ```

3. **Review suggested matches and add to `get_name_mappings()`**

4. **Re-run verification script to confirm fixes**

## Example: Adding a New Mapping

You found "Rob Dillingham" in props doesn't match "Robert Dillingham" in results:

```python
# 1. Add to src/player_name_utils.py
def get_name_mappings():
    return {
        # ... existing ...
        'Rob Dillingham': 'Robert Dillingham',
    }

# 2. Re-run verification
python scripts/20251120_verify_overlap_w_2425_3pt_prop_data.py

# Match rate should improve!
```

## Files Created

```
src/
  player_name_utils.py              # Core utility functions

scripts/
  debug_player_name_mismatches.py   # Debugging tool
  20251120_verify_overlap_w_2425_3pt_prop_data.py  # Verification script

data/
  props_with_game_results_2024_25.csv  # Output: merged data
```

## Common Name Issues

| Issue | Example | Solution |
|-------|---------|----------|
| Accents | Dončić → Doncic | Automatic (accent removal) |
| Suffix | Jr. → Jr | Automatic (suffix standardization) |
| Generational | Jimmy Butler III → Jimmy Butler | Automatic (III/II removal) |
| Nickname | Herb → Herbert | Manual mapping |
| Missing suffix | Derrick Jones → Derrick Jones Jr | Manual mapping |
| Initials | P.J. → Pj | Manual mapping |

## Notes

- **93.96% match rate is normal** - remaining mismatches are usually DNP (did not play) situations
- **Name mappings are centralized** in `src/player_name_utils.py` for reuse across all scripts
- **Debugging output is verbose** to help identify patterns and fix issues quickly
- **The utility is importable** - use it in any script that needs player name normalization

## Future Enhancements

Possible improvements:
1. Fuzzy matching with confidence scores
2. Automatic learning of name mappings from successful matches
3. Database of historical name variations
4. API to validate player names against official NBA roster

