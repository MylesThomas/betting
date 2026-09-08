# MLB TB Snapshot — ET vs UTC Naming Conventions

**Status: DONE** (2026-09-07)

**TL;DR**: Keep both UTC and ET versions of every timestamp column. Dates always have `_et` suffix. UTC timestamps always have `_utc` suffix. ET string companions stored alongside for human readability in SQL/DuckDB.

---

## Why We Reset All Snapshot Data

The original `snapshot_props.py` derived `game_date` as `commence_time[:10]` — a naive UTC date slice. For west coast evening games (e.g., Athletics 7 PM PT = 02:06 UTC next day), this produced a date one day ahead of the ET calendar date. That broke:

1. **CLV join** — `recommendations.csv` uses ET `gameday`; snapshots used UTC date. West coast games would never join.
2. **Plot titles** — title showed "game date 2026-09-08" while first pitch rendered as "10:06 PM ET Sep 7".
3. **S3 folder naming** — a "Sep 7" game landed in the `2026-09-08/` folder, so `compute_clv(game_date="2026-09-07")` couldn't find it.

**Fix**: `snapshot_props.py` now converts `commence_time` (UTC) → ET before taking the date. All existing parquet files deleted; collection restarts fresh after redeploy.

---

## Final Column Naming Convention

| Column | Type | Example | Notes |
|---|---|---|---|
| `snapshot_ts_utc` | str (ISO) | `2026-09-07T14:06:00Z` | for epoch math / sorting |
| `snapshot_ts_et` | str | `2026-09-07 10:06 AM ET` | human-readable companion |
| `game_date_et` | str | `2026-09-07` | **primary join key** — always ET |
| `game_date_utc` | str | `2026-09-08` | raw UTC date, debugging only |
| `commence_time_utc` | str (ISO) | `2026-09-08T02:06:00Z` | for epoch math |
| `commence_time_et` | str | `2026-09-07 10:06 PM ET` | human-readable companion |

**Rules:**
- All dates carry `_et` suffix (`game_date_et`)
- All UTC timestamp strings carry `_utc` suffix
- Every UTC timestamp has an ET string companion in the parquet
- `game_date_et` is the folder name in S3 and the join key with `recommendations.csv`

---

## Files Changed

- `src/mlb_total_bases_modeling/scripts/snapshot_props.py` — writes all 6 new/renamed columns
- `src/mlb_total_bases_modeling/scripts/compute_clv.py` — `commence_time` → `commence_time_utc`
- `src/mlb_total_bases_modeling/scripts/smoke_test_snapshots.py` — REQUIRED_COLUMNS updated
- `src/mlb_total_bases_modeling/scripts/verify_snapshot.py` — REQUIRED_COLUMNS updated
- `src/mlb_total_bases_modeling/analysis/plot_clv_snapshot.py` — reads ET strings from parquet for display; UTC strings for epoch math only

## What Still Needs To Happen

- Delete S3 snapshot data and redeploy Lambda (Myles does this)
- Run `bash tmp_script.sh` for the commit series
