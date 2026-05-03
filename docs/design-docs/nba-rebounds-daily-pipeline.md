# NBA rebounds — daily production pipeline (design)

## Production decision — probability / edge (Option A)

**Prod will use Option A (parametric Normal)** for per-prop `P(over)`, `P(under)`, and edge vs book **no-vig** implied probabilities. This matches the notebook plan and the current reference backtest stack.

- **Not in scope for initial prod:** Option B (empirical / shifted game-history bootstrap for `P(over/under)`). Revisit only if research shows material lift over Option A.
- **Naming:** Option A is **closed-form** (Normal CDF), not Monte Carlo sampling. Avoid calling daily scoring “Monte Carlo” unless we add explicit simulation or a separate bootstrap layer (e.g. reporting-only uncertainty on portfolio metrics).

### Production scoring spec (Option A) — align with reference backtest implementation

Inputs per prop row (after join to model + features):

- `consensus` — `consensus_reb_line` (or prod equivalent).
- `yhat` — model point forecast (OLS and/or XGB; compute edges per model).
- `line` — book line for that row.
- `σ` — **`roll_reb_std_5`** in prod (σ window **N = 5**). Use `σ = max(roll_reb_std_5, 0.25)` (same floor as reference backtest).
- `shrink` — **0.00** in prod; `mean_adj = consensus + (1 - shrink) * (yhat - consensus)` → full move toward `yhat` from consensus.

Then:

- `z = (line - mean_adj) / σ`
- `P(over line) = 1 - Φ(z)`
- `P(under line) = Φ(z)`

Edges vs no-vig from the row’s `p_over_novig`, `p_under_novig`:

- `edge_over = P(over) - p_over_novig`
- `edge_under = P(under) - p_under_novig`

Play rule: **under_only** — bet under when **`edge_under > min_edge`** (strict, matches shared `pick_side` behavior); never bet over. **`min_edge = 0.05`** (vs no-vig). *Doc previously said `>=`; code + tests use `>`.*

### Metric semantics lock (do not reinterpret)

- **No-vig probabilities (`p_over_novig`, `p_under_novig`) are used only for model edge estimation and play decisioning** (`edge_over`, `edge_under`, thresholding).
- **Backtest and production performance metrics** (`ROI`, `PnL`, realized hit rate, bankroll curves) must use the **actual traded market odds** on the placed side.
- Do not report no-vig-based returns as realized returns. If no-vig outputs are shown, label them as **expected edge diagnostics** only.

### Locked prod policy (matches current reference sweep)

| Parameter | Prod value | Backtest row (your table) |
|-----------|------------|---------------------------|
| `side_policy` | `under_only` | Top block + heatmap row |
| `sigma_window` | **5** → column `roll_reb_std_5` | `under_only`, σ=5 |
| `shrinkage` | **0.00** | ROI **0.023664**, **n_bets 7093**, hit_rate **0.600** |
| `min_edge` | **0.05** | Same row |

**Research note (optional alternative):** same grid with **`shrinkage = 0.25`**, `min_edge = 0.05`, `sigma_window = 5` shows **slightly higher ROI** (0.023800) but **fewer bets** (6457). Prod choice **shrink 0** is valid if you prefer maximum `yhat` weight and more volume.

---

## Research context — Option A vs B vs PnL bootstrap

Two different things appear under “Monte Carlo” in research:

### Edge and P(under) / P(over) for a prop (live scoring)

| Label in notebook | Meaning | Implemented in scripts? |
|-------------------|---------|-------------------------|
| **Option A — parametric** | Normal around `mean_adj` with `σ = roll_reb_std_N` (floored); CDF for `P(over/under)` | **Yes** — implemented in the shared scorer + current backtest scripts. **→ Prod.** |
| **Option B — empirical** | Bootstrap/shift player’s past actual REB games to approximate `P(over/under)` | **No** — notebook only; not in the current production path. |

### “MC” / bootstrap on backtest PnL (`v7_spec_sweep_rebounds_oos.py` / notebook)

- **Bootstrap** over **placed bet** unit outcomes (resample with replacement → distribution of total PnL / mean ROI).
- **Optional for prod reporting** (not required for placing a bet): e.g. internal dashboards or post-hoc run summaries. **Not** a substitute for Option A at score time.

---

## Goal (daily automation)

End state: a **scheduled job** (e.g. nightly ET before slate) that:

1. Refreshes **player box score / game log** source of truth so rolling features are current.
2. **Builds features** for rebounds modeling (aligned with research parquet schema or a slim prod schema).
3. **Trains** two predictors on a defined training window:
   - **OLS** + **XGBoost** on the **B_min_max six features** (locked for prod v1 — `src/nba_rebounds_modeling/rebounds_feature_spec.py`).
4. Ingests **tonight’s** (or **next slate’s**) **player rebounds props** at **bookmaker × line** grain.
5. For **each prop row** (player / game / date / bookmaker / line):
   - Compute **yhat_ols**, **yhat_xgb**.
   - Compute **play signal** using **Option A** (Normal CDF) per model: `mean_adj`, `σ`, `P(over/under)`, edges vs no-vig; apply locked **side policy** (`under_only` or both) and **`min_edge`**.
   - Classify: **play OLS only / XGB only / both / neither** (per agreed rules).
   - *(Optional later)* Attach **bootstrap CIs** on aggregate metrics for email/dashboards — separate from per-row scoring.
6. **Write** a structured artifact to **S3** (audit trail: inputs, odds, consensus, both yhats, edges, flags, run id, timestamp).
7. **Email** (or SNS → email) summary of **today’s plays** with: consensus line, book line/odds, **projected reb** for OLS and XGB, edge summary, and play classification.

---

## Running the pipeline

### Daily command

```bash
python scripts/run_rebounds_daily_pipeline.py \
  --config config/nba_rebounds_prod.yaml \
  --slate-date YYYY-MM-DD
```

Omit `--slate-date` to default to ET today. Add `--input-universe-mode replace` on the first run of the day (or to force a full S3 rebuild of the input universe); use `append` (default) after that.

Config can also be an S3 URI — useful for Lambda or any environment without a local checkout of config:

```bash
python scripts/run_rebounds_daily_pipeline.py \
  --config s3://nba-betting-mt/rebounds/config/nba_rebounds_prod.yaml \
  --slate-date YYYY-MM-DD
```

### What each step does

| Step | Script | Input | Output |
|------|--------|-------|--------|
| 1. Feature universe | `build_rebounds_full_universe.py` | Player logs + spreads from S3 | `/tmp/rebounds_features_{run_id}.parquet` → `s3://nba-betting-mt/rebounds/features/rebounds_feature_universe.parquet` |
| 2. Live props fetch | `fetch_nba_player_rebounds_live.py` | Odds API | `{s3_run_prefix}/live_rebounds_props_raw_{date}.csv` |
| 3. Scoring input | `build_rebounds_scoring_input.py` | Live CSV from S3 | `{s3_run_prefix}/rebounds_props_scoring_input_{date}.parquet` |
| 4. Feature slice | `prod_slice_rebounds_features.py` | Feature universe from S3 | `{s3_run_prefix}/rebounds_features_slice_{date}.parquet` (0 rows pregame → triggers step 4b) |
| 4b. Pregame backfill | `build_rebounds_pregame_feature_slice.py` | Feature universe + scoring input | Same slice path, filled from latest historical form |
| 5. Train | `prod_train_rebounds_models.py` | Feature universe from S3 | `/tmp/rebounds_models_{run_id}/` → `s3://nba-betting-mt/rebounds/models/{model_run_id}/` |
| 6. Score | `prod_score_rebounds_slate.py` | Slice + props + models | `{s3_run_prefix}/rebounds_scored_{date}.parquet` |
| 7. Notify | `prod_notify_rebounds_sns.py` | Scored parquet from S3 | SNS → email |

### What lands in S3

Every run writes to a unique prefix:

```
s3://nba-betting-mt/rebounds/daily_runs/{date}/{run_id}/
  live_rebounds_props_raw_{date}.csv
  rebounds_props_scoring_input_{date}.parquet
  rebounds_features_slice_{date}.parquet
  rebounds_scored_{date}.parquet
  rebounds_props_history_{date}.parquet   ← historical props snapshot
  run_manifest.json                        ← full audit trail for the run
```

Models are at: `s3://nba-betting-mt/rebounds/models/{model_run_id}/`

The canonical feature universe (updated each run) is at:
`s3://nba-betting-mt/rebounds/features/rebounds_feature_universe.parquet`

### Required env / secrets

```bash
export ODDS_API_KEY=...          # or THE_ODDS_API_KEY
export AWS_DEFAULT_REGION=us-east-2
# credentials via aws sso login, IAM role, or long-lived keys
```

### Key config knobs (`config/nba_rebounds_prod.yaml`)

| Key | What it controls |
|-----|-----------------|
| `retrain_daily` | `true` = retrain models every run; `false` = pass `--models-dir` instead |
| `build_props_scoring_input_from_live` | `true` = fetch live props from Odds API; `false` = use `props_input_uri` |
| `enable_pregame_feature_backfill` | `true` = auto-build feature slice from latest form when same-day slice is empty (normal for pregame runs) |
| `max_feature_lag_days` | Pipeline fails if feature universe is older than this many days |
| `prod_min_edge` | Minimum edge vs no-vig to flag a play (default 0.05) |
| `feature_build_cache_dir` | Local cache for DuckDB S3 reads; defaults to `/tmp/rebounds_cache` |

### No games / off-season

If the Odds API returns no events for the slate date, the pipeline exits cleanly:

```
no_games_for_slate
  slate=YYYY-MM-DD
  pipeline_complete
```

---

## Implemented modules & CLI (v1)

| Piece | Path |
|-------|------|
| Option A vector batch + `pick_side` | `src/nba_rebounds_modeling/option_a_scoring.py` |
| Champion feature column list | `src/nba_rebounds_modeling/rebounds_feature_spec.py` |
| Unit tests (CDF + strict min_edge) | `tests/unit/test_rebounds_option_a_scoring.py` |
| Reference backtest (refactored to shared scoring) | `00_research/scripts/v3_run_rebounds_edge_backtest.py` |
| Feature slice for one slate date | `00_research/scripts/prod_slice_rebounds_features.py` |
| Train OLS + XGB, `manifest.json`, optional S3 model upload | `00_research/scripts/prod_train_rebounds_models.py` |
| Score slate (join + Option A + play flags + optional S3 parquet) | `00_research/scripts/prod_score_rebounds_slate.py` |
| SNS or stdout plays table | `00_research/scripts/prod_notify_rebounds_sns.py` |
| Example prod YAML | `config/nba_rebounds_prod.example.yaml` |

### Individual scripts (for debugging / manual runs)

All scripts accept `s3://` URIs for `--feat`, `--props`, `--output`, etc. See `--help` on each.

The orchestrated daily run (`scripts/run_rebounds_daily_pipeline.py`) is the recommended path — see **Running the pipeline** above.

---

## Design principles (repo)

- **Fail fast** on missing required columns / keys; no silent defaults for critical config.
- **No fake data** in prod paths; integration tests use fixtures only if explicitly scoped.
- **Single source of truth** for: feature definitions, train window, champion spec, edge policy (version in config or manifest alongside each run).

---

## Test + production runbook (operator steps)

### 0) One-time setup

1. Copy config template: `config/nba_rebounds_prod.example.yaml` → `config/nba_rebounds_prod.yaml`.
2. Confirm S3 keys are set:
   - `feature_universe_s3_uri`, `input_universe_s3_uri`, `s3_bucket`, `s3_runs_prefix`, `s3_models_prefix`
   - `retrain_daily: true`
   - `build_props_scoring_input_from_live: true`
   - `live_fetch_to_s3: true`
3. Export secrets/env:
   - `ODDS_API_KEY` (or `THE_ODDS_API_KEY`)
   - AWS credentials for S3/SNS
4. Set `sns_topic_arn` in config for email notifications.

### 1) Preflight checks (before first prod run)

Run once from repo root:

```bash
python scripts/fetch_nba_player_rebounds_live.py --help
python scripts/build_rebounds_scoring_input.py --help
python scripts/run_nba_rebounds_daily_pipeline.py --help
python scripts/agent_precommit_check.py
```

Pass criteria:
- All CLI help commands run.
- Precommit check passes.

### 2) Single-day dry run (manual)

Pick an upcoming slate date and run:

```bash
python scripts/run_rebounds_daily_pipeline.py \
  --config config/nba_rebounds_prod.yaml \
  --slate-date YYYY-MM-DD \
  --input-universe-mode replace
```

Pass criteria:
- Command exits 0 or `no_games_for_slate` (no props available).
- All outputs present in S3 under `s3://nba-betting-mt/rebounds/daily_runs/{date}/{run_id}/`.
- `run_manifest.json` written to that prefix.
- Notify step publishes to SNS.

Notes:
- Requires `ODDS_API_KEY` set in env.
- Pregame runs (before game logs land) automatically use `enable_pregame_feature_backfill`.

### 3) Burn-in gate (recommended before “real money” prod)

Run once daily for 7 consecutive slates (manual or scheduled) and log each run id.

Pass criteria:
- 7/7 successful runs (no step failures).
- Non-zero scored rows on slates with posted props.
- No stale-feature/model guardrail failures.

### 4) Production cutover

1. Schedule the same command via EventBridge/cron (nightly ET before slate lock).
2. Keep fail-hard enabled (current behavior).
3. Add alerting on non-zero exit.
4. Keep artifacts in S3/local run folder for audit.

Prod command:

```bash
python scripts/run_rebounds_daily_pipeline.py \
  --config config/nba_rebounds_prod.yaml
```

### 5) Daily operator checklist (quick)

- Confirm `ODDS_API_KEY` and AWS credentials are valid.
- Run command (ET today is the default slate date).
- Verify scored parquet row count + play counts in the SNS email.
- Check `run_manifest.json` in S3 if debugging a run.

---

## Phased implementation plan (refreshed)

### Phase 0 — Lock semantics and naming

- [x] **Probability model for prod:** Option A (parametric Normal + CDF) — **decided**.
- [x] **Champion edge policy:** `under_only`, **`shrink = 0`**, **`sigma_window = 5`** (`roll_reb_std_5`), **`min_edge = 0.05`** — see **Locked prod policy** above.
- [x] **Champion feature spec (prod v1):** B_min_max six columns in `rebounds_feature_spec.py`. P1/A3 research alternatives out of scope until explicitly swapped in code + manifest.
- [ ] **Email / reporting copy:** describe outputs as **model-implied probs** (Normal); if bootstrap summaries are added, label them as **PnL uncertainty** (historical bets), not per-prop MC.
- [ ] **Production naming cleanup:** use descriptive artifact names in configs/docs/run outputs (for example `rebounds_props_scoring_input.parquet`) instead of version labels.

### Phase 1 — Data refresh

- [ ] Identify existing **NBA box score / game log** ingestion (lambda, script, or external API job).
- [ ] Add or extend a **idempotent daily task** that lands raw logs to the path the **feature build** expects (S3 layout + manifest date).

### Phase 2 — Features

- [x] **Slate slice CLI:** `prod_slice_rebounds_features.py` filters the full feature parquet to `--as-of-date` and sets `rebounds_feature_schema_version`.
- [ ] **Incremental / as-of build** inside DuckDB (optional): avoid rebuilding all seasons daily.
- [ ] Add a deterministic **feature freshness check** in the daily run (latest date present, row-count sanity, required columns).

### Phase 3 — Train + serialize

- [x] **`prod_train_rebounds_models.py`:** OLS pickle + XGB JSON + `manifest.json`; optional S3 upload when `s3_bucket` + `s3_models_prefix` set in YAML.
- [ ] Decide and document retrain cadence: nightly full refit vs weekly refit + daily “score only”.
- [ ] Add model staleness guardrail (max model age before score job warns/fails).

### Phase 4 — Score slate + policy

- [ ] **Live props fetch** wired to this scorer — follow **`docs/exec-plans/active/nba-rebounds-live-props-s3-plan.md`** (ingestion only; output scoring-input schema).
- [x] **Join + Option A + play flags:** `prod_score_rebounds_slate.py` (drops duplicate non-key columns from feat before merge so `consensus_reb_line` stays single-sourced from props).
- [x] **Shared scoring module** + **unit tests** (`option_a_scoring.py`, `test_rebounds_option_a_scoring.py`); reference backtest refactored to call it.
- [ ] **Future:** Option B as separate module.
- [ ] Add end-to-end schema validation between props input parquet and scorer required columns before betting logic runs.

### Phase 5 — Persistence + notify

- [x] **Scored parquet** + optional `s3://` via `--s3-uri` on score script; **SNS** stub `prod_notify_rebounds_sns.py`.
- [ ] Rich email template (HTML) if desired beyond SNS text body.
- [ ] Settle and append next-day outcomes (`result`, `pnl_units`) into a closed-loop scored artifact.

### Phase 6 — Ops

- [ ] Schedule (EventBridge / cron), alarms on failure, DLQ, secrets for email and S3.
- [ ] Backfill / dry-run mode for local validation.
- [ ] Burn-in gate: 7 consecutive successful daily dry-runs before production cutover.

---

## Open questions

1. **Retrain daily** vs **score with frozen model** between retrain cadences?
2. **Multi-book**: email one row per book line or collapse to “best line” per player?
3. **Settlement**: do we append results next day for closed-loop S3 rows?
4. **Alignment with architecture**: confirm this lives under **ingestion → storage → analysis** boundaries per `docs/design-docs/dependency-boundaries.md`.

---

## References

- Live rebounds props + S3 + scoring-input bridge plan: `docs/exec-plans/active/nba-rebounds-live-props-s3-plan.md`
- Research notebook: `src/nba_rebounds_modeling/00_research/notebooks/v1_rebounds_game_to_game_foundations.ipynb` (Option A/B markdown; Option A implemented in scripts).
- Backtests: `v3_run_rebounds_edge_backtest.py`, `v4_run_rebounds_under_only_season_robustness.py`, `v5_compare_rebounds_models_oos.py`.
- Spec sweep + PnL + bet bootstrap: `v7_spec_sweep_rebounds_oos.py`.
- Prod scoring: `src/nba_rebounds_modeling/option_a_scoring.py`.
