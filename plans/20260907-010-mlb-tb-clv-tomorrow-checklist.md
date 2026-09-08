# MLB TB CLV — Tomorrow's Check Checklist

**Created:** 2026-09-07  
**For:** Morning + evening of 2026-09-08

---

## Context

The hourly prop snapshot Lambda (`mlb-tb-prop-snapshot`, us-east-2) has been running since 2026-09-07, writing parquets to:
- `s3://the-odds-api-mt/mlb/total_bases_model/prop_snapshots/{season}/{game_date}/snapshot_{ts_utc}.parquet`

Sep 7 was the first day of snapshots but only has partial coverage (first snap at 7:31 AM ET, games started ~1:30pm).  
**Sep 8 is the first full day** — lines were captured from Sep 7 evening through Sep 8 first pitches.

CLV = `first_seen_under_odds - closing_under_odds` (positive = beat the close, favorable for UNDER bettor).  
Key question to answer: **how much does the market move between first-seen and 9am ET** (our current betting window)?

Local data lives at:
- `~/Downloads/tmp/total_bases/prop_snapshots/` — snapshot parquets
- `~/Downloads/tmp/total_bases/daily_runs/` — recommendations CSVs

---

## Morning checklist (before Sep 8 first pitches)

```bash
# 1. Sync latest data
aws s3 sync s3://the-odds-api-mt/mlb/total_bases_model/prop_snapshots/ ~/Downloads/tmp/total_bases/prop_snapshots/
aws s3 sync s3://the-odds-api-mt/mlb/total_bases_model/daily_runs/ ~/Downloads/tmp/total_bases/daily_runs/

# 2. Check hourly snapshots are landing for Sep 8
ls ~/Downloads/tmp/total_bases/prop_snapshots/2026/2026-09-08/

# 3. Run CLV for Sep 7 (games closed — should have closing prices)
uv run python src/mlb_total_bases_modeling/scripts/compute_clv.py --gameday 2026-09-07

# 4. Plot a Sep 7 player we bet on (check Query 1 SQL for n_play_bets > 0)
uv run python src/mlb_total_bases_modeling/analysis/plot_clv_snapshot.py --game-date 2026-09-07 --player "Blaze Alexander"
# Swap player name as needed — use DuckDB query to find players with n_play_bets > 0

# 5. Aggregate CLV summary for Sep 7 (early read — partial day, no overnight data)
uv run python src/mlb_total_bases_modeling/analysis/agg_clv_summary.py --gameday 2026-09-07
# Saves: ~/Downloads/tmp/total_bases/clv_summary_2026-09-07.csv
# Appends: ~/Downloads/tmp/total_bases/clv_summary_log.csv

# 6. Verify snapshot pipeline health
uv run python src/mlb_total_bases_modeling/scripts/verify_snapshot.py
```

**What to look for in the CLV output:**
- `clv_first_seen_cents` vs `clv_9am_cents` — if first_seen is consistently 10-20¢ better than 9am, we're leaving edge on the table by waiting
- `clv_tier` — most plays should be `ok` or `mild+`; if you see `severe-` consistently it means we're betting into a moving market
- Consensus CLV across books should be directionally consistent

---

## Evening checklist (after Sep 8 first pitches ~7pm ET)

```bash
# 1. Sync again
aws s3 sync s3://the-odds-api-mt/mlb/total_bases_model/prop_snapshots/ ~/Downloads/tmp/total_bases/prop_snapshots/
aws s3 sync s3://the-odds-api-mt/mlb/total_bases_model/daily_runs/ ~/Downloads/tmp/total_bases/daily_runs/

# 2. Run CLV for Sep 8 — first full day with overnight snapshots
uv run python src/mlb_total_bases_modeling/scripts/compute_clv.py --gameday 2026-09-08

# 3. Plot Sep 8 players (find who we bet via SQL Query 1)
uv run python src/mlb_total_bases_modeling/analysis/plot_clv_snapshot.py --game-date 2026-09-08 --player "<player name>"

# 4. Aggregate CLV summary — how much better/worse was open vs 9am ET vs close?
uv run python src/mlb_total_bases_modeling/analysis/agg_clv_summary.py --gameday 2026-09-08
# Key output: "Edge left on table" = avg clv_first_seen - avg clv_9am
# Verdict "SIGNIFICANT" if >5¢ — means we should bet at first-seen instead of 9am

# 5. Units +/- comparison: first snapshot vs 9am ET entry
# agg_clv_summary.py will have a section for this — build it out in the AM using Sep 7 data.
# Logic: expected units gained per bet = novig_under × (decimal_fs - decimal_9am)
# Sum across all play bets = total expected units gained from first-seen timing vs 9am.
# novig_under comes from recommendations.csv; decimal_fs/dec_9am from first_seen_under_odds
# and nine_am_et_under_odds in the CLV data. Implement in _print_units_comparison().

# 6. Run settlement email for Sep 8
uv run python src/mlb_total_bases_modeling/scripts/settle_total_bases.py --gameday 2026-09-08 --output /tmp/sep8_settled.json
```

---

## Key files

| File | Purpose |
|------|---------|
| `src/mlb_total_bases_modeling/scripts/compute_clv.py` | CLV computation — loads all snapshots for a game_date, outputs per-book CLV |
| `src/mlb_total_bases_modeling/analysis/plot_clv_snapshot.py` | Odds movement plot — `--game-date` + `--player` |
| `src/mlb_total_bases_modeling/scripts/verify_snapshot.py` | 7-check pipeline health assertion |
| `src/mlb_total_bases_modeling/scripts/settle_total_bases.py` | Settlement + HTML email with CLV section |
| `~/Downloads/tmp/total_bases/temp_sql_query.sql` | DuckDB queries — Q1 finds players with n_play_bets, Q2 shows per-player snapshots |

---

## Key insight from Sep 7

Blaze Alexander plot showed lines moved ~30¢ adverse from 7:31 AM ET (first-seen) to first pitch (~1:36 PM ET). CLV was uniformly positive (+26¢ to +50¢) across all books — meaning first-seen price was significantly better than close.

**If Sep 8 shows the same pattern**, the case is strong to trigger recommendations at first-seen rather than waiting for the 9am ET model run.

---

## Lambda / infra status

- `mlb-tb-prop-snapshot` Lambda: DEPLOYED, running fixed code (dtype + async invoke fixes)
- EventBridge rule `mlb-tb-prop-snapshot-hourly`: ENABLED, `cron(0 * * * ? *)`
- Snapshots write to S3 at top of every hour, all day
