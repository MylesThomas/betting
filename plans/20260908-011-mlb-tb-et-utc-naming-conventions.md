# MLB TB Snapshot — ET vs UTC Naming Conventions

**TL;DR**: All `game_date` fields are now ET calendar dates. `snapshot_ts_utc` and `commence_time` stay in UTC — they're timestamps, not dates. The only open question is whether timestamp columns should carry an `_et` / `_utc` suffix. Decision: suffix timestamp columns that have a timezone-dependent interpretation; leave date columns bare since "ET" is now the implicit convention.

---

## Decisions Made (2026-09-08)

### Why we reset all snapshot data

The original `snapshot_props.py` derived `game_date` as `commence_time[:10]` — a naive UTC date slice. For west coast evening games (e.g., Athletics 7 PM PT = 02:06 UTC next day), this produced a `game_date` one day ahead of the ET calendar date. That broke:

1. **CLV join** — `recommendations.csv` uses ET `gameday` as `game_date`; snapshots used UTC date. West coast games would never join.
2. **Plot titles** — title shows "game date 2026-09-08" while first pitch renders as "10:06 PM ET Sep 7" — confusing.
3. **S3 folder naming** — a "Sep 7" Athletics game landed in the `2026-09-08/` folder, so `compute_clv(game_date="2026-09-07")` couldn't find it.

**Fix**: `snapshot_props.py` now converts `commence_time` (UTC) → ET before taking the date. All 16 existing parquet files were deleted; overnight collection starts fresh.

### Naming convention chosen: ET is implicit for dates

All `game_date` columns across the pipeline (snapshots, recommendations, CLV output, settled bets) represent the **ET calendar date** of the game. No `_et` suffix on the column name — "ET" is the house convention for dates, just as "American" is implied for odds columns.

---

## Open Question: Should timestamp columns carry `_utc` / `_et` suffixes?

### Current column names
| Column | Type | TZ | Suffix? |
|---|---|---|---|
| `snapshot_ts_utc` | timestamp | UTC | ✅ already has `_utc` |
| `commence_time` | timestamp string | UTC | ❓ no suffix |
| `game_date` | date string | ET | ❌ no suffix (by convention) |
| `snapshot_et` (computed in plot script) | datetime | ET | ✅ has `_et` |
| `nine_am_et_under_odds` | odds | ET anchor | ✅ has `_et` |
| `first_seen_under_odds` | odds | n/a | ✅ self-descriptive |

### Decision options

**Option A — status quo, add `_utc` to `commence_time`**
Rename `commence_time` → `commence_time_utc` in parquet schema. All other columns stay as-is. Low churn, fixes the one ambiguous field.

**Option B — full audit, suffix everything that has timezone meaning**
Any column storing an absolute point in time gets `_utc`. Any derived ET representation gets `_et`. Dates stay bare (ET implicit). This is more explicit but requires a schema migration across all downstream code.

**Option C — leave it, document the convention**
`commence_time` is unambiguous in context (it's always the raw API value = UTC). Add a comment to the schema docstring in `snapshot_props.py`. Zero migration cost.

### Recommendation
**Option A** — rename `commence_time` → `commence_time_utc`. It's the one field that looks like it could be local time but isn't. The rename is a one-line change in `snapshot_props.py` and `compute_clv.py`; since we just reset all snapshot data there's nothing to migrate.

---

## Files affected by any naming change
- `src/mlb_total_bases_modeling/scripts/snapshot_props.py` — writes the column
- `src/mlb_total_bases_modeling/scripts/compute_clv.py` — reads `commence_time`
- `src/mlb_total_bases_modeling/analysis/plot_clv_snapshot.py` — reads `commence_time`
- `src/mlb_total_bases_modeling/lambda/lambda_snapshot.py` — docstring only
- DuckDB queries in `tmp_query.sql` — would need updating
