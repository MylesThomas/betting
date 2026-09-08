# MLB Total Bases — Hourly Prop Snapshots + CLV + Tightening Watch

**Goal**: Capture `batter_total_bases` odds every hour for all available MLB games, compute
Closing Line Value (CLV) per bet, and surface a "tightening watch" when a player's rolling
line/odds drift adversely vs. their season-to-date baseline.

---

## Overview

Two new components:

1. **Snapshot Lambda** (`mlb-tb-prop-snapshot`) — runs hourly all day, writes one parquet per
   run to S3 containing all player+book rows for every MLB game the Odds API returns.
2. **CLV + Tightening Report** — added to the existing settlement Lambda; runs at 9am ET
   morning-after (plus manual `--gameday` flag).

---

## New Files

| File | Purpose |
|------|---------|
| `src/mlb_total_bases_modeling/scripts/snapshot_props.py` | Core fetch + first-seen detection logic |
| `src/mlb_total_bases_modeling/lambda/lambda_snapshot.py` | Lambda handler for hourly snapshot runs |
| `src/mlb_total_bases_modeling/scripts/compute_clv.py` | CLV computation (first-seen, 9am ET, closing) |
| `src/mlb_total_bases_modeling/scripts/compute_tightening.py` | Cross-day rolling tightening watch |
| `src/mlb_total_bases_modeling/scripts/verify_snapshot.py` | Schema + first-seen + credits assertions |
| `src/mlb_total_bases_modeling/scripts/verify_clv.py` | CLV correctness using market_raw as proxy |
| `src/mlb_total_bases_modeling/scripts/verify_snapshot_actionnetwork.py` | Playwright → ActionNetwork spot-check (manual) |
| `src/mlb_total_bases_modeling/scripts/smoke_test_snapshots.py` | Full 5-check chain smoke test |
| `lambda/mlb_tb_snapshot/Dockerfile` | Container image for snapshot Lambda |
| `lambda/mlb_tb_snapshot/deploy_mlb_tb_snapshot.sh` | ECR deploy script (mirrors rebounds pattern) |
| `lambda/mlb_tb_verify_daily/lambda_function.py` | Daily noon-ET health check Lambda |

## Modified Files

| File | Change |
|------|--------|
| `src/mlb_total_bases_modeling/scripts/settle_total_bases.py` | Add CLV + tightening sections to HTML email |
| `CONTEXT.md` | Add new domain terms (already done) |

---

## S3 Paths

```
Snapshots (new):
  s3://the-odds-api-mt/mlb/total_bases_model/prop_snapshots/{season}/{game_date}/snapshot_{snapshot_ts_utc}.parquet
  # One file per hourly run. All player+book rows for all available MLB games.
  # snapshot_ts_utc format: YYYYMMDD_HHMMSS (UTC)

Existing (unchanged):
  s3://the-odds-api-mt/mlb/total_bases_model/market_raw/mlb_total_bases_market_raw.parquet
  s3://the-odds-api-mt/mlb/total_bases_model/daily_runs/{gameday}/recommendations.csv
  s3://the-odds-api-mt/mlb/total_bases_model/settled/mlb_tb_settled_bets.parquet
```

---

## Snapshot Parquet Schema

One row per player × bookmaker × market_key × snapshot run.

| Column | Type | Notes |
|--------|------|-------|
| `snapshot_ts_utc` | str | ISO timestamp of this Lambda run |
| `season` | int | Calendar year |
| `game_date` | str | YYYY-MM-DD from `commence_time` |
| `event_id` | str | Odds API event ID |
| `home_team` | str | |
| `away_team` | str | |
| `commence_time` | str | ISO game start timestamp (UTC) |
| `bookmaker` | str | e.g. `fanduel`, `draftkings` |
| `market_key` | str | `batter_total_bases` or `batter_total_bases_alternate` |
| `player_name` | str | As returned by Odds API |
| `over_line` | float | e.g. `1.5` |
| `over_american_odds` | int | e.g. `-115` |
| `under_line` | float | e.g. `1.5` |
| `under_american_odds` | int | e.g. `-105` |
| `binary_player_game_first_seen` | bool | True if first snapshot ever for this player+event_id pair |
| `last_odds_player_game` | str (JSON) | Prior snapshot's odds for this player+event_id+book: `{"over": {"line_value": X, "american_odds": Y}, "under": {...}}`. Null on first-seen rows. Stored as JSON string for parquet compatibility. |
| `credits_before` | int | `x-requests-remaining` before the run's first API call |
| `credits_after` | int | `x-requests-remaining` after the run's last API call |

---

## CLV Output (per player+game+book)

Computed in `compute_clv.py`, joined into settlement output.

| Column | Notes |
|--------|-------|
| `first_seen_under_odds` | American odds at first-seen snapshot |
| `nine_am_et_under_odds` | American odds at snapshot closest to 09:00 ET on game_date |
| `closing_under_odds` | American odds at last snapshot before `commence_time` |
| `closing_line` | Line at closing snapshot |
| `clv_first_seen_cents` | `first_seen_under_odds - closing_under_odds` (positive = beat the close) |
| `clv_9am_cents` | `nine_am_et_under_odds - closing_under_odds` |
| `clv_line_shift` | `closing_line - first_seen_line` (positive = line moved up = adverse for UNDER) |
| `clv_tier` | `ok` / `mild` / `moderate` / `strong` / `severe` based on worst CLV of the two references |

**Price CLV tiers** (applied to `max(clv_first_seen_cents, clv_9am_cents)`, higher = worse):
- `ok`: < 5 cents
- `mild`: 5–10 cents
- `moderate`: 10–20 cents
- `strong`: 20–30 cents
- `severe`: ≥ 30 cents

**Consensus CLV**: median across all available books for that player+game.

---

## Tightening Watch Output (per player)

Computed in `compute_tightening.py`, shown as a separate email section.

Rolling averages computed across all historical snapshot first-seen rows for each player.

| Column | Notes |
|--------|-------|
| `player_name` | |
| `n_games_30d` | Games with snapshot data in last 30 days |
| `n_games_season` | Games with snapshot data this season |
| `avg_line_30d` | Mean of modal line per game, last 30 days |
| `avg_line_season` | Mean of modal line per game, season-to-date |
| `avg_under_odds_30d` | Mean consensus under odds per game, last 30 days |
| `avg_under_odds_season` | Mean consensus under odds per game, season-to-date |
| `today_modal_line` | Most common line offered today across books |
| `today_avg_under_odds` | Median under odds across books today |
| `line_drift` | `today_modal_line - avg_line_season` |
| `odds_drift_cents` | `today_avg_under_odds - avg_under_odds_season` |
| `tightening_flag` | `ok` / `mild` / `moderate` / `strong` / `severe` (same tier thresholds as CLV) |

Only players who had a play or track recommendation on `gameday` appear in the report.

---

## Verification

### Summary

| Check | Script / Mechanism | Cadence | Alert |
|-------|--------------------|---------|-------|
| Schema + row count + first-seen + credits | `verify_snapshot.py` → `smoke_test_snapshots.py` | Auto at end of deploy `.sh` + daily Lambda | SES email on daily failure |
| CLV correctness | `verify_clv.py` (market_raw proxy) | Manual after Step B built | Print pass/fail |
| Email rendering | Daily settlement email (CLV sections baked in) | Daily 9am ET (existing settlement run) | Visual — inbox |
| End-to-end chain | `smoke_test_snapshots.py` (5 checks) | Auto at end of deploy `.sh` | Exit non-zero blocks deploy |
| Second-source market check | `verify_snapshot_actionnetwork.py` (Playwright) | Manual on-demand when something looks off | Print pass/fail |
| Daily system health | `mlb-tb-snapshot-verify` Lambda | Daily noon ET | SES email on failure only |

---

### `verify_snapshot.py`

Callable standalone: `uv run python src/mlb_total_bases_modeling/scripts/verify_snapshot.py`

Checks (print ✓/✗ per check, exit non-zero if any fail):
1. At least 1 snapshot parquet exists in S3 for today's date
2. All required schema columns present with correct dtypes
3. At least 1 `binary_player_game_first_seen=True` row exists
4. `last_odds_player_game` is null on all first-seen rows
5. `last_odds_player_game` is non-null on at least some non-first-seen rows (confirms prior-snapshot join works)
6. `credits_before > credits_after` (confirms credit logging consumed real credits)
7. Spot-print last 3 snapshots for 3 randomly sampled players (human eyeball check)

---

### `verify_clv.py`

Callable standalone: `uv run python src/mlb_total_bases_modeling/scripts/verify_clv.py`

Steps:
1. Load `mlb_total_bases_market_raw.parquet` from S3 (already exists, 2024–2026 data)
2. Synthesise a mini snapshot dataset for a fixed past game_date (hard-code e.g. `2026-07-10`)
   by reshaping market_raw rows into snapshot schema — treating each row as a single snapshot
3. Run `compute_clv()` against the synthetic data
4. Assert no nulls in `clv_first_seen_cents`, `clv_9am_cents`, `clv_line_shift`
5. Assert CLV values are within plausible range (e.g. -200 to +200 cents)
6. Print CLV for 3 named players — human confirms values look reasonable vs. the raw market_raw data

---

### `smoke_test_snapshots.py`

Called automatically at end of `deploy_mlb_tb_snapshot.sh` after a real Lambda invocation.
Also callable standalone: `uv run python src/mlb_total_bases_modeling/scripts/smoke_test_snapshots.py`

5 checks in order (print ✓/✗, exit non-zero if any fail):
1. Load latest snapshot parquet from S3 — assert non-empty and schema valid
2. Assert ≥ 1 `binary_player_game_first_seen=True` row in latest snapshot
3. Re-call Odds API live for 1 event, compare `under_american_odds` to snapshot (±5 cents tolerance)
4. Run `compute_clv()` on the most recent game_date that has ≥ 2 snapshots — assert no nulls in CLV columns
5. Run `compute_tightening()` for the same game_date — assert output DataFrame is non-empty

Prints final summary: `N/5 checks passed`.

---

### `verify_snapshot_actionnetwork.py` (manual, on-demand)

Run when something looks suspicious in the settlement email.

Steps:
1. Load latest snapshot from S3, pick 3–5 players with the most books
2. Use Playwright to navigate to ActionNetwork's MLB player props page
3. Scrape total bases lines for those players
4. Compare `under_american_odds` from snapshot to ActionNetwork (±10 cents tolerance)
5. Print match/mismatch per player with exact values

No schedule — run manually: `uv run python src/mlb_total_bases_modeling/scripts/verify_snapshot_actionnetwork.py`

---

### Daily verification Lambda (`mlb-tb-snapshot-verify`)

Runs noon ET daily: `cron(0 17 * * ? *)` (17:00 UTC).

Checks:
1. At least 1 snapshot parquet exists for today in `prop_snapshots/{season}/{today}/`
2. Schema valid on the latest snapshot file
3. At least 1 `binary_player_game_first_seen=True` row in today's data
4. Yesterday's settled output (if exists) has non-null `clv_first_seen_cents` column

On **any failure**: send SES email to `SES_TO` with subject:
`MLB TB Snapshot — daily health check FAILED — {date}` and a brief summary of which check failed.

On **all pass**: silent. No email.

Env vars: `ODDS_API_KEY`, `SES_SOURCE=tqstrats@gmail.com`, `SES_TO=mylescgthomas@gmail.com`,
`S3_BUCKET=the-odds-api-mt`

---

## Implementation Steps + Agent Orchestration

Steps are ordered by dependency. Steps marked **‖** can run in parallel.

### Step A — `snapshot_props.py` + `lambda_snapshot.py` + verification scripts ‖ Step E

**Agent**: single maker agent.

**`snapshot_props.py`**:
- Fetch all MLB events via live endpoint: `GET /v4/sports/baseball_mlb/events`
- For each event, fetch odds: `GET /v4/sports/baseball_mlb/events/{event_id}/odds`
  - `markets=batter_total_bases,batter_total_bases_alternate`
  - `regions=us,us2`
  - `oddsFormat=american`
- Record `credits_before` from `x-requests-remaining` before first call
- Record `credits_after` after last call
- **First-seen detection**: load all existing parquet files for each game's `game_date`
  from `prop_snapshots/{season}/{game_date}/`. Build set of `(player_name, event_id)` already seen.
  Any pair not in the set → `binary_player_game_first_seen=True`. Populate `last_odds_player_game`
  from the most recent prior snapshot row for that player+event_id+bookmaker.
- Write DataFrame to S3.

**`lambda_snapshot.py`**:
- Minimal handler: calls `snapshot_props.main()`.
- Returns `{"status": "ok", "rows_written": N, "credits_used": X}`.

**Also in this step**: write `verify_snapshot.py` and `smoke_test_snapshots.py` (checks 1–3 only;
checks 4–5 require Steps B and C to exist).

---

### Step E — CONTEXT.md update ‖ Step A

Already done. No agent needed.

---

### Step B — `compute_clv.py` (after Step A) ‖ Step C

**Agent**: single maker agent.

`compute_clv(game_date: str, season: int) -> pd.DataFrame`:
1. Load all parquet files under `prop_snapshots/{season}/{game_date}/` into one DataFrame.
2. For each `(player_name, event_id, bookmaker, market_key)` group:
   a. **First-seen row**: row where `binary_player_game_first_seen=True`
   b. **9am ET snapshot**: snapshot_ts_utc closest to `{game_date}T14:00:00Z` (09:00 ET = 14:00 UTC)
   c. **Closing snapshot**: row with max `snapshot_ts_utc` where `snapshot_ts_utc < commence_time`
3. Compute `clv_first_seen_cents`, `clv_9am_cents`, `clv_line_shift`.
4. Assign `clv_tier` from worst of the two price CLV values.
5. Compute consensus (median across books) per `(player_name, event_id, market_key)`.
6. Return merged DataFrame.

Supports `--gameday` CLI arg for manual runs.

**Also in this step**: write `verify_clv.py`.

---

### Step C — `compute_tightening.py` (after Step A) ‖ Step B

**Agent**: single maker agent.

`compute_tightening(game_date: str, season: int, players: list[str]) -> pd.DataFrame`:
1. Load all `prop_snapshots/{season}/*/snapshot_*.parquet` for the current season
   (only first-seen rows: `binary_player_game_first_seen=True`).
2. Per player, compute per-game-date aggregates: `modal_line`, `avg_under_odds`.
3. Compute 30-day and season-to-date rolling windows.
4. For today's `game_date`, compute `today_modal_line` and `today_avg_under_odds` from all rows.
5. Compute `line_drift`, `odds_drift_cents`, assign `tightening_flag`.
6. Filter to `players` list.

Supports `--gameday` and `--players` CLI args for manual runs.

---

### Step D — Settlement integration (after Steps B and C)

**Agent**: single maker agent.

Modify `settle_total_bases.py`:
1. After `settle_bets()`, call `compute_clv(gameday, season)` and
   `compute_tightening(gameday, season, players=settled_rows["player_name"].tolist())`.
2. Join CLV on `(player_name, bookmaker, market_key)`; fall back to consensus CLV on miss.
3. Add to `build_html_email()`:
   - CLV columns in bet table: `clv_first_seen_cents` and `clv_9am_cents` side-by-side, color-coded by tier
   - CLV summary in footer: avg CLV across plays, % that beat the close
   - Tightening Watch section at bottom (only rendered if ≥ 1 player flagged at mild or above)
4. Both sections degrade silently if no snapshot data exists — settlement still sends.
5. Keep `--gameday` and `--output` args fully functional.

**Also in this step**: complete `smoke_test_snapshots.py` checks 4–5 (now that B and C exist).

---

### Step F — Daily verification Lambda (after Step D)

**Agent**: single maker agent.

Write `lambda/mlb_tb_verify_daily/lambda_function.py` per the spec in the Verification section above.
Deploy via its own `deploy_mlb_tb_verify_daily.sh` following the ECR container pattern.
EventBridge: `cron(0 17 * * ? *)` (noon ET), ENABLED immediately (unlike snapshot Lambda which starts DISABLED).

---

## Dependency Graph

```
Step A (snapshot + verify_snapshot + smoke 1-3)
  ├──> Step B (CLV + verify_clv)      ──┐
  └──> Step C (tightening)            ──┼──> Step D (settlement + smoke 4-5)
                                         │         │
Step E (CONTEXT.md, done)                │         └──> Step F (daily verify Lambda)
                                         │
                              [parallel B ‖ C]
```

**Parallel execution plan:**
- Launch Step A agent (Step E already done).
- After Step A: launch Step B agent + Step C agent simultaneously.
- After B and C: launch Step D agent.
- After D: launch Step F agent.

---

## Deploy Script — `deploy_mlb_tb_snapshot.sh`

Follows the ECR container pattern from `deploy_nba_rebounds_daily.sh` exactly.

```
Lambda name:   mlb-tb-prop-snapshot
ECR repo:      mlb-tb-prop-snapshot
Memory:        512 MB
Timeout:       300 s (5 min)
Region:        us-east-2
IAM role:      betting-dashboard-daily-update-role-ille2llh
```

**Env vars baked into Lambda:**
```
ODDS_API_KEY=<from env>
SES_SOURCE=tqstrats@gmail.com
SES_TO=mylescgthomas@gmail.com
SNS_TOPIC_ARN=arn:aws:sns:us-east-2:232692785472:betting-arb-alerts
S3_BUCKET=the-odds-api-mt
```

**EventBridge rule** (created DISABLED — enable manually after smoke test passes):
```
Name:     mlb-tb-prop-snapshot-hourly
Schedule: cron(0 * * * ? *)   # top of every hour, all day
```

**Deploy script steps:**
1. Prerequisites check (AWS CLI, Docker, credentials, IAM role)
2. Build + push Docker image to ECR
3. Create/update Lambda with env vars
4. EventBridge rule (DISABLED)
5. Lambda DryRun invoke
6. **Real Lambda invoke** (wait for completion) → **run `smoke_test_snapshots.py`** → exit non-zero on failure

Step 6 ensures every deploy is self-validating. If smoke test fails, the EventBridge rule stays DISABLED and you have a clear failure signal before any real traffic runs.

---

## Post-Deploy Enable Checklist

After smoke test passes and you're satisfied with 1–2 days of manual data:

- [ ] Enable EventBridge rule `mlb-tb-prop-snapshot-hourly`
- [ ] Confirm first few hourly runs land in S3 with correct `game_date` partitions
- [ ] Run `verify_snapshot.py` manually — all 7 checks pass
- [ ] Wait for a settled gameday, run `settle_total_bases.py --gameday <date> --output /tmp/r.json`,
      open HTML in browser, confirm CLV and tightening sections appear
- [ ] Enable EventBridge rule for daily verification Lambda `mlb-tb-snapshot-verify`
