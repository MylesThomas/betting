# NCAAB Conference Rematch Analysis (2nd/3rd Game SU & ATS)

**Created:** 2026-02-17  
**Status:** Plan – pending review  
**Scope (v1):** One conference, one season via CLI args. Start: Big Ten 2025-26.

## Goal

- **2nd game:** Team lost 1st meeting → that team’s SU and ATS record in the 2nd game.
- **3rd game:** Team lost both 1st and 2nd → that team’s SU and ATS record in the 3rd game.

Args = one `--conference` and one `--season` (e.g. Big Ten 2025-26).

---

## Cache (ET day)

- **Key:** Today’s date in **Eastern Time** (e.g. `2026-02-17`).
- **Location:** e.g. `~/Downloads/tmp/ncaab_cache/rematch_joined_{season}_{et_date}.parquet` (one file per season, keyed by ET date).
- **Logic:** If cache file exists for `(season, today_et)` → load from cache. Else → load outcomes + lines from S3 for the season(s) provided, join, write cache for today (ET), then proceed.
- **Effect:** Same ET day reuses cache; next day (ET) triggers fresh S3 load.

**S3 sources:**

| Data | S3 path |
|------|--------|
| Outcomes | `s3://ncaab-betting-mt/data/01_input/historical_game_results/*.csv` |
| Lines | `s3://ncaab-betting-mt/data/01_input/the-odds-api/ncaab/game_lines/*.csv` |

Refresh lines before run: `python scripts/fetch_historical_ncaab_season_lines.py --s3 --skip-existing`

**All conferences for one season (2024-25):** Use `--conferences all` to run every conference in the mapping; output is one CSV with a `conference` column. Example:
```bash
python scripts/fetch_historical_ncaab_season_lines.py --season 2024-25 --s3 --skip-existing
python scripts/fetch_historical_game_results_espn_api.py --sport ncaab --season 2024-25 --start-date 2024-11-03 --end-date 2025-04-20 --s3 --skip-existing
python analysis/analyze_ncaab_conference_rematch_su_ats.py --conferences all --season "2024-25" --csv
```
Output: `data/04_output/ncaab_rematch_all_conferences_2024-25_{et_date}.csv`. In DuckDB, filter with `WHERE conference = 'Big Ten'` (or any conference).

---

## Plan (concise)

1. **Setup** – Script: `analysis/analyze_ncaab_conference_rematch_su_ats.py`. Args: `--conference` (default Big Ten), `--season` (default 2025-26). Resolve project root via `.gitignore`; add `src` and `tmp` for imports.

2. **Cache (ET day)** – Today in ET → string `YYYY-MM-DD`. Cache path: e.g. `{cache_dir}/rematch_joined_{season}_{et_date}.parquet`. If path exists → read joined df from cache. Else → load from S3 (step 3), join (step 4), write this parquet, then continue.

3. **Load from S3** – For `args.season` get `(start_date, end_date)` from `SEASON_DATES` (reuse from `tmp/join_ncaab_outcomes_and_lines`). Load outcomes and lines from S3 (same buckets/paths as join script). Validate season; fail if empty.

4. **Join** – `join_outcomes_and_lines(outcomes_df, lines_df, min_games=5)`. Columns: `GAME_DATE`, `HOME_TEAM`, `AWAY_TEAM`, `HOME_SCORE`, `AWAY_SCORE`, `consensus_spread`.

5. **Conference filter** – Both teams in `args.conference` (use `src/ncaab_conference_data.py` or CSV). Fail if no teams in conference.

6. **Meeting number** – Per unordered pair `(A, B)`, sort games by date; assign `meeting_number` 1, 2, 3….

7. **Focal team** – 2nd game: focal = loser of 1st. 3rd game: focal = loser of both 1st and 2nd.

8. **SU/ATS** – SU: focal wins if more points in that game. ATS: use `consensus_spread` (home perspective; negative = home favored). Focal covers if (focal margin vs spread) > 0; push = 0. Only games with non-null spread for ATS.

9. **Aggregate & report** – 2nd: N, SU W–L (%), ATS W–L–P (%). 3rd: same. Print summary; optional CSV to `data/04_output/`.

10. **Examples** – Print a few rematch games (date, pair, focal, SU, ATS) for sanity check.

---

## File summary

| Item | Choice |
|------|--------|
| Script | `analysis/analyze_ncaab_conference_rematch_su_ats.py` |
| Outcomes | S3 `ncaab-betting-mt` / `data/01_input/historical_game_results/` |
| Lines | S3 `ncaab-betting-mt` / `data/01_input/the-odds-api/ncaab/game_lines/` |
| Join / SEASON_DATES | Reuse `tmp/join_ncaab_outcomes_and_lines.py` |
| Conference | `src/ncaab_conference_data.py` or `tmp/ncaab_conference_mapping.csv` |
| Cache | `~/Downloads/tmp/ncaab_cache/rematch_joined_{season}_{et_date}.parquet` |
| Output | Stdout + optional CSV under `data/04_output/` |

---

## DuckDB queries

The script runs DuckDB sense-check queries at the end on the **subset that was selected** (single conference or all conferences). See the `_run_duckdb_queries()` function at the end of `analysis/analyze_ncaab_conference_rematch_su_ats.py`. It prints:

- Counts by `rematch_type`
- SU/ATS for rematch 2nd (focal = loser of game 1)
- Avg spread change 1→2 (home perspective)
- Avg line move toward focal (positive = in focal's favor g1→g2), and by focal W/L and cover/no-cover game 2 (requires `home_1st_meeting` in the data; re-run with `--csv` if missing)

For ad-hoc CSV analysis, load the output CSV in DuckDB and filter by `conference` / `rematch_type` as needed. Output path: `data/04_output/ncaab_rematch_{Conference}_{season}_{et_date}.csv` or `ncaab_rematch_all_conferences_{season}_{et_date}.csv`.




---

## Out of scope (v1)

- Multiple conferences/seasons in one run; conference-tournament split.
