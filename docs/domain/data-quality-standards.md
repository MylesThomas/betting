# Data Quality Standards

**Status:** 🚧 DRAFT - Needs your review  
**Last updated:** 2026-02-13

**⚠️ REVIEW INSTRUCTIONS:**
- ➕ Add YOUR actual validation rules from code
- 📝 Add YOUR thresholds and standards
- ✅ Mark what's correct
- ❌ Flag what's wrong

---

## Props Data Quality: What Makes Data "Good"?

### Required Fields

Every player prop must have:

```python
{
  "player_id": str,        # NOT NULL (fail if missing)
  "player_name": str,      # NOT EMPTY
  "team": str,             # Normalized team name
  "opponent": str,         # Normalized team name
  "game_date": str,        # ISO 8601 format
  "market": str,           # One of known market types
  "line": float,           # The over/under threshold
  "over_odds": int,        # American odds
  "under_odds": int,       # American odds
  "timestamp": str,        # ISO 8601 format
  "book": str              # Sportsbook name
}
```

**⚠️ YOUR INPUT:** Check actual data structure in your code - is this accurate?

### Timestamp Freshness

**For pre-game props (before game starts):**
- **< 5 minutes old:** ✅ Good (actionable)
- **5-15 minutes old:** ⚠️ Getting stale (use with caution)
- **> 15 minutes old:** ❌ Stale (lines may have moved)

**For live/in-game betting (during the game):**
- **< 1 minute old:** ✅ Good (must be very fresh)
- **1-5 minutes old:** ❌ Too stale (game situation changed)
- Reason: Need to log into app and place bet quickly
- By the time you execute, situation may have changed

**For historical data (analysis/backtesting):**
- Any timestamp is OK (just for analysis)
- But must be consistent (all from same time window)

**⚠️ YOUR INPUT:** 
- Do you do live betting, or only pre-game?
- What freshness requirements do you actually use in your code?

### Odds Range Validation

**American odds valid ranges:**
- **Negative (favorites):** -100, -101, -102... down to -10000
- **Positive (underdogs):** +100, +101, +110... up to +10000
- **CRITICAL:** There is NO valid range between -100 and +100
  - If you see -50, -10, +10, +50 → **DATA ERROR**
  - Valid odds jump from -100 to +100 (no in-between)

**Validation rules:**
```python
# Invalid odds (data error)
if -100 < odds < 100:
    raise ValueError(f"Invalid odds: {odds}")

# Valid ranges (no real upper limit, but sanity check)
if odds <= -100:  # Favorite
    # Theoretically no limit, but extremely unlikely to see < -50000
    if odds < -50000:
        warn(f"Extremely heavy favorite: {odds} - verify correctness")
elif odds >= 100:  # Underdog  
    # No real upper limit (can be +1000000 for extreme long shots)
    # But unlikely to see these on NBA props (not lottery bets)
    if odds > 50000:
        warn(f"Extremely unlikely underdog: {odds} - verify correctness")
else:
    # Between -100 and +100 = ERROR
    raise ValueError(f"Odds must be <= -100 or >= +100, got {odds}")
```

**Typical range for NBA props:**
- Most props: **-300 to +300**
- Heavy favorites: -400 to -500
- Big underdogs: +400 to +500
- Outside this range: Possible but less common

**Red flags:**
- Odds between -100 and +100 (impossible)
- Odds > +10000 or < -10000 (data error)
- Only one side available (book may have pulled market)
- Stuck at same odds across multiple fetches (stale)

**⚠️ YOUR INPUT:** 
- Do you have automated validation that rejects odds in -100 to +100 range?
- What's the most extreme odds you've seen on NBA props?

### Player-Team Consistency

**Validation checks:**
- Player must be on listed team (check against roster cache)
- If player recently traded → flag for review
- No mismatched player-team pairs (Curry on Lakers = error)

**Roster cache requirements:**
- Updated within **7 days** (YOUR threshold?)
- All active players present
- Inactive/injured players flagged
- Trade updates within **24 hours** (YOUR threshold?)

**⚠️ YOUR INPUT:** How do you handle mid-season trades in validation?

### Market Type Validation

**Known NBA markets** (from `config/the-odds-api_config.yaml`):

```
✅ Valid markets:
- player_points
- player_rebounds  
- player_assists
- player_threes
- player_blocks
- player_steals
- player_points_rebounds_assists (PRA)
- player_points_assists (PA)
- player_points_rebounds (PR)
- player_double_double
- player_triple_double
- [... see config for full list]
```

**Handling unknown markets:**
- Flag for review (might be new market type)
- Don't automatically reject (books add new markets)
- Log and alert human

**⚠️ YOUR INPUT:** Do you auto-reject unknown markets or just flag them?

---

## Line Reasonableness Checks

### Sanity Bounds by Market

**Player Points:**
- Typical range: **10.5 - 40.5**
- Red flag if: < 5 or > 50
- Outliers: Bench players (< 10), superstars in select games (> 35)

**Player Rebounds:**
- Typical range: **3.5 - 15.5**
- Red flag if: < 1 or > 20

**Player Assists:**
- Typical range: **2.5 - 12.5**  
- Red flag if: < 1 or > 15

**Player Threes:**
- Typical range: **0.5 - 6.5**
- Red flag if: > 10

**⚠️ YOUR INPUT:** 
- What are YOUR actual sanity bounds?
- Do you have per-player adjustments (Jokic can have 25 rebound line)?
- How do you handle outliers?

### Odds Consistency Checks

**Both sides should be reasonably close:**
- Over -110 / Under -110 = ✅ Standard
- Over -120 / Under +100 = ✅ Reasonable
- Over -200 / Under +150 = ⚠️ Heavy juice (possible)
- Over -500 / Under +400 = ❌ Extremely lopsided (check for error)

**⚠️ YOUR INPUT:** What odds spreads trigger warnings in your code?

---

## Duplicate Detection

### Same Player, Same Market, Same Book

**Should only appear once:**
- Same player_id
- Same market type
- Same book
- Same game_date

**If duplicates found:**
- Take most recent timestamp
- Flag as data quality issue
- Log source of duplicates

**⚠️ YOUR INPUT:** How do you handle duplicates in your ingestion pipeline?

---

## Line Movement Data Quality

### Significant vs Insignificant Movement

**Significant movement (worth tracking):**
- **Points props:** > 0.5 points
- **Odds changes:** > 10 cents (e.g., -110 to -120)
- **Time window:** < 10 minutes (YOUR threshold?)

**Insignificant (noise):**
- < 0.25 points
- < 5 cents odds change
- Gradual drift over hours

**⚠️ YOUR INPUT:**
- What are YOUR actual thresholds from `config/line_steam_config.yaml`?
- Do thresholds differ by market type?

### Steam Detection Quality

**High-quality steam signal:**
- **3+ major books** move together (YOUR threshold?)
- Move in **< 5 minutes** (YOUR threshold?)
- All move **same direction** (not splitting)
- Magnitude > 0.5 points or 10 cents

**Low-quality signal (ignore):**
- Only 1 book moves
- Gradual movement over 30+ minutes
- Books moving opposite directions

**⚠️ YOUR INPUT:** Pull actual thresholds from your steam detection code!

---

## Cache & Historical Data Quality

### Roster Cache

**Quality requirements:**
- **Freshness:** < 7 days old (YOUR threshold?)
- **Completeness:** All teams present
- **Accuracy:** Matches official NBA rosters
- **Trade updates:** < 24 hours after trade (YOUR threshold?)

**Red flags:**
- Missing teams
- Outdated player-team mappings
- Inactive players not flagged

### Player-Team Mapping Cache

**Requirements:**
- Current season only (or specify historical?)
- Trade updates reflected immediately
- Historical mappings preserved (for backtesting)

**YOUR ACTUAL IMPLEMENTATION:**
- Built in `src/player_team_history/`
- See that module for structure and logic
- Historical mappings preserved (for backtesting)
- Trade updates handled (see module for details)

### S3 Historical Data

**File structure quality:**
- Consistent naming: `YYYY-MM-DD_props.json`
- No gaps in daily data
- No duplicate files for same date
- Reasonable file sizes (not 0 bytes, not absurdly large)

**Data integrity:**
- All required fields present
- No corrupted JSON
- Timestamp matches filename date
- Consistent schema across files

**⚠️ YOUR INPUT:** What are your S3 data quality checks?

---

## Validation Rules Summary (For `scripts/validate_props_data.py`)

### Must-Have Validations

```python
✅ Required fields present
✅ Timestamp freshness check (< 5 min for live)
✅ Odds in valid range (-10000 to +10000)
✅ Player-team consistency (roster check)
✅ Market type is known
✅ No duplicate props (player + market + book + date)
✅ Line within sanity bounds for market type
✅ Both over and under odds present
```

### Warning-Level Validations

### Warning-Level Validations

```python
⚠️ Odds outside typical NBA prop range (-300 to +300)
⚠️ Pre-game timestamp 5-15 min old (getting stale)
⚠️ Line at extreme end of sanity bounds (but possible)
⚠️ Unknown market type (flag for review)
⚠️ Heavy juice discrepancy between sides
```

### Data Quality Metrics to Track

```python
📊 % of props with complete fields
📊 Average timestamp freshness
📊 % with valid player-team mappings (via src/player_team_history/)
📊 Duplicate rate
📊 Outlier rate (lines outside sanity bounds)
📊 Unknown market rate
```

---

## Real-World Examples (From Your Data)

**Future TODO:** Create example fixtures in `tests/fixtures/`:

1. **Good prop data** (all fields valid, fresh timestamp)
2. **Bad data: Stale** (timestamp > 1 hour old for pre-game)
3. **Bad data: Missing fields** (no player_id or odds)
4. **Bad data: Outlier** (impossible line like 100.5 points)
5. **Edge case: Recent trade** (player on new team, cache not updated)

---

## Implementation Checklist

**Files with actual validation logic:**
- `lambda/nba_player_props_ingest/lambda_function.py` - Ingestion validation
- `scripts/build_consensus_props.py` - Aggregation validation
- `config/line_steam_config.yaml` - Movement thresholds
- `src/player_team_history/` - Player-team mapping and history
- `src/player_name_utils.py` - Player name normalization

---

## Related Documents

- `docs/domain/betting-fundamentals.md` - What the data means
- `docs/domain/market-mechanics.md` - Line movement significance
- `docs/validation/data-validation-rules.md` - Technical validation spec
- `tests/fixtures/` - Example good/bad data (to be created)

---

## For Agents

**When validating props data, check:**
1. All required fields present? (fail fast if not)
2. Timestamp fresh enough? 
   - Pre-game: < 5 min ideal
   - Live betting: < 1 min required
3. Odds in valid range? 
   - Must be <= -100 OR >= +100 (no in-between)
   - Warn if > 50000 or < -50000 (verify correctness)
4. Player on correct team? (check via `src/player_team_history/`)
5. Market type recognized? (flag if unknown)
6. Line reasonable for market? (10-40 for points, 3-15 for rebounds, etc.)
7. No duplicates? (same player + market + book + date)

**Remember:** Fail fast on required fields and invalid odds range. Warn on suspicious but possible values.
