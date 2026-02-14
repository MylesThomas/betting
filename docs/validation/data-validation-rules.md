# Data Validation Rules

**Status:** 📝 Planned  
**Last updated:** 2026-02-13

This document will specify rules for validating props data quality.

## Planned Validation Rules

### Required Fields Check

Every prop must have:
- `player_id` (string, not null)
- `player_name` (string, not empty)
- `team` (string, normalized)
- `opponent` (string, normalized)
- `market` (string, one of known types)
- `line` (float)
- `over_odds` (int, American format)
- `under_odds` (int, American format)
- `timestamp` (ISO 8601 format)
- `book` (string)

### Odds Range Check

American odds must be:
- Between -10000 and +10000
- Not exactly 0 (meaningless)
- Both over/under present (not just one side)

### Timestamp Freshness

For live lines:
- Timestamp must be < 5 minutes old
- For historical data: any timestamp OK

### Player-Team Consistency

- Player must be on listed team (check roster cache)
- If player recently traded, flag for review
- No mismatched player-team pairs

### Market Type Validation

Known markets:
- `player_points` / `player_points_over_under`
- `player_rebounds`
- `player_assists`
- `player_threes`
- `player_pts_rebs_asts`

Unknown markets → flag for review (might be new)

### Line Reasonableness

Sanity checks:
- Points line typically 10-40 (not 100)
- Rebounds line typically 5-15 (not 50)
- Odds usually between -200 and +200

### Duplicate Detection

- Same player, same market, same book → should only appear once
- If duplicates, take most recent timestamp

---

**To be written in:** Phase 4.2 (Validation Harnesses)  
**Will be enforced by:** `scripts/validate_props_data.py`
