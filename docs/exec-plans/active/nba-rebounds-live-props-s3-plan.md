# Plan: Live `player_rebounds` props → S3 → scoring input pipeline

**Status:** Plan (not implemented in this doc)  
**Layer:** Ingestion only (`scripts/` or `lambda/`) — no scoring logic here.  
**Downstream:** `prod_score_rebounds_slate.py` expects a **scoring-input parquet** with the props columns required for scoring joins and no-vig edge calculation (`rebounds_feature_spec.V3_PROPS_SCORE_COLS` + merge keys).

---

## 1. Goal

1. Fetch **only** The Odds API market **`player_rebounds`** for the **current NBA slate** (today / next tip window).
2. Land **raw** responses in **S3** with a stable layout + run metadata.
3. Produce a **normalized artifact** that either:
   - matches **historical CSV columns** so existing DuckDB paths keep working, and/or
  - writes **`rebounds_props_scoring_input`-compatible** parquet for **`--props`** on the prod scorer (pre-game: `REB` null).

---

## 2. What exists today (reuse, don’t reinvent)

| Asset | Role |
|-------|------|
| `scripts/fetch_nba_player_props.py` | Historical props: events + per-event odds, `parse_player_props()`, S3 `s3://the-odds-api-mt/nba/historical_player_props/{season}/{date}.csv`, column set documented in script header. |
| `build_rebounds_full_universe.load_props()` | Reads those CSVs via DuckDB, **`market = 'player_rebounds'`**, normalizes `player_normalized`, `date` (ET), `line`, odds. |
| `build_rebounds_full_universe.build_market_panel` / `build_v3_props_raw` | Turns per-book lines into **`book_line`** + scorer-ready rows; **requires** `logs` with **`game_id`** per `(season, date, player_normalized)`. |
| `prod_score_rebounds_slate.py` | Merges feat slice + props scoring input on **`GROUP_KEYS`**; does not need `REB` for scoring (outcome column unused there). |

**Credit win:** request **`markets=player_rebounds` only** (not the full `DEFAULT_MARKETS` bundle in the big fetcher).

---

## 3. API approach (live vs historical)

**Historical path (already in repo):**  
`historical/sports/basketball_nba/events` + `historical/sports/basketball_nba/events/{id}/odds` with a fixed snapshot time — good for **past** dates, not “right now”.

**Live / upcoming slate (new work):** use **non-historical** v4 endpoints, same pattern as other repo scripts (e.g. `sports/basketball_nba/events/{event_id}/odds` with `markets=player_rebounds`, `regions=us`, `oddsFormat=american`).

**Recommended flow:**

1. **Discover events** for the target calendar day (ET or UTC — document choice; align `date` string with `v2`’s ET `game_time` conversion).
2. For each `event_id`, **GET odds** with **`markets=player_rebounds` only**.
3. Parse with the **same outcome rules** as `parse_player_props()` in `fetch_nba_player_props.py` (Over/Under per `point` line); **filter** rows to `market_key == 'player_rebounds'` if the API returns multiple markets.
4. Attach metadata: `fetch_timestamp_utc`, `season` (from `season_utils.get_current_nba_season()` or config), `odds_api_event_id`, `commence_time`.

**Rate limiting:** reuse `RATE_LIMIT_DELAY`-style spacing; log `x-requests-*` headers.

---

## 4. S3 layout (two-stage)

### Stage A — raw (debug / reparse)

Prefix (example):

`s3://the-odds-api-mt/nba/live_player_props/player_rebounds/{season}/YYYY-MM-DD/{fetch_ts_utc}.jsonl`

- One line per event: `{ "event_id", "commence_time", "response": { ... full JSON ... } }`
- **Idempotent runs:** new key per fetch; never overwrite silently.

### Stage B — curated CSV (column-compatible with historical)

Either:

- **Option 1 (simplest for DuckDB):** write the **same logical columns** as historical props CSV (`player`, `away_team`, `home_team`, `game_time`, `market`, `prop_line`, `over_odds`, `under_odds`, `bookmaker`, …, `fetch_date`, `season`) to:

  `s3://the-odds-api-mt/nba/live_player_props/player_rebounds/{season}/YYYY-MM-DD/latest.csv`  
  plus versioned: `.../runs/{fetch_ts_utc}.csv`

- **Option 2:** keep live separate and add a small **Glue/DuckDB** view — only if you explicitly don’t want live mixed with backfill paths.

**Recommendation:** Option 1 versioned keys + optional `latest.csv` pointer (copy) for schedulers.

---

## 5. Hard part: `game_id` + scoring-input parquet (pre-game)

`v2` attaches **`game_id`** by inner-joining props to **player game logs** for that date. **Before games finish, logs are often empty** — so live scoring needs an alternate join:

1. **NBA schedule / scoreboard** for the slate date (e.g. `nba_api.live` scoreboard or team schedule endpoint) → map `(date, team, opponent)` → **`GAME_ID`**.
2. **Player → team** for that day (roster or injury report snapshot — pick one supported source; document failure if unknown).
3. Build a **minimal `logs_stub`**: columns at least `season`, `date`, `player_normalized`, `game_id`, `REB` (NaN pre-game).
4. Run **shared normalization** (extract from `v2` into a module if needed):
   - `load_props`-style rename → `build_market_panel(props, logs_stub)` → `build_v3_props_raw(...)`.
5. Write **`rebounds_props_scoring_input_YYYY-MM-DD.parquet`** to S3 (e.g. `s3://.../nba/rebounds/prod_inputs/scoring_props/{season}/`).

**Post-game same day:** optionally refresh once logs exist so `REB` is filled (backtest / settlement); scorer does not require it.

---

## 6. Implementation phases (ordered)

| Phase | Deliverable | Notes |
|-------|-------------|--------|
| **P0** | Extract **`parse_player_props`** (or odds JSON → rows) into **`src/` utility** callable from scripts | Keeps ingestion thin; **no** `sys.path` hacks in new code — run with `python -m` from repo root or setuptools. |
| **P1** | **`scripts/fetch_nba_player_rebounds_live.py`** | CLI: `--date`, `--dry-run` (print row counts, no S3), `--s3`, `--season`. Env: `ODDS_API_KEY` / `THE_ODDS_API_KEY`. |
| **P2** | **S3 writes** stage A + B | boto3 `put_object`; manifest sidecar JSON (`credits_used`, row counts, event ids). |
| **P3** | **`game_id` resolver** module + tests against one known slate | Use real API in manual test only; unit tests with **recorded fixture JSON** (check in `tests/fixtures/` if allowed) — **no fabricated odds**. |
| **P4** | **`scripts/build_rebounds_scoring_input.py`** | Reads live CSV from S3 or local, outputs scoring-input parquet local + optional S3. |
| **P5** | **Schedule** | EventBridge/cron **morning ET**; alarm on zero rows when scoreboard shows games. |
| **P6** | **Docs** | Update `docs/design-docs/nba-rebounds-daily-pipeline.md` “Command sequence” with live path; link this plan. |

---

## 7. End-to-end command sequence (target state)

```bash
# 1) Ingest live rebounds-only props → S3
python scripts/fetch_nba_player_rebounds_live.py --date YYYY-MM-DD --s3

# 2) Build scoring-input parquet (join game_id, consensus, no-vig)
python scripts/build_rebounds_scoring_input.py --live-csv s3://.../runs/{ts}.csv --output rebounds_props_scoring_input.parquet

# 3) Features slice (after v2 universe includes that date or equivalent feat row source)
python .../prod_slice_rebounds_features.py --feat ... --as-of-date YYYY-MM-DD --output ...

# 4) Score
python .../prod_score_rebounds_slate.py --models-dir ... --feat-slice ... --props rebounds_props_scoring_input.parquet --slate-date YYYY-MM-DD --output ...

# 5) Notify
python .../prod_notify_rebounds_sns.py --scored ...
```

---

## 8. Config / secrets

- **API key:** existing `.env` pattern from `fetch_nba_player_props.py`.
- **Buckets/prefixes:** add block to `config/nba_rebounds_prod.example.yaml` (`live_props_prefix`, `scoring_props_prefix`) when implementing.
- **Architecture:** `dependency-boundaries.md` — ingestion may call external APIs and S3; **analysis scripts must not** call Odds API directly (orchestrator in `scripts/` is fine).

---

## 9. Risks & mitigations

| Risk | Mitigation |
|------|------------|
| No props until books post lines | Retry window; alert if empty near lock. |
| Player name mismatch vs NBA logs | Single normalization: `normalize_from_odds_api`; log unmatched %; fail run if above threshold. |
| `game_id` ambiguous (trades / two games) | Prefer official game list for date; tie-break by `commence_time` vs Odds API event. |
| API cost drift | Log credits per run; rebounds-only market keeps cost bounded. |

---

## 10. Definition of done

- [ ] One command fetches **rebounds-only** live props and writes **versioned S3** artifacts.
- [ ] Second command produces **scoring-input** parquet with **`game_id`** for pre-game slate.
- [ ] `prod_score_rebounds_slate.py` runs end-to-end with `--props` set to that parquet (plus valid feat slice + models).
- [ ] Design doc updated; no scoring logic duplicated in fetch script.
