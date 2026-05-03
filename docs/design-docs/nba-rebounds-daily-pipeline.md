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

### Command sequence (local; paths are examples)

1. **Build / refresh full feature universe** (when logs + props are current):  
   `python src/nba_rebounds_modeling/00_research/scripts/build_rebounds_full_universe.py --season '*' --output .../rebounds_model_features.parquet --output-v3 .../rebounds_props_scoring_input.parquet`

2. **Slice features to slate date:**  
   `python .../prod_slice_rebounds_features.py --feat .../rebounds_model_features.parquet --as-of-date YYYY-MM-DD --output .../rebounds_features_slice_YYYY-MM-DD.parquet`

3. **Train models** (copy `config/nba_rebounds_prod.example.yaml` → `config/nba_rebounds_prod.yaml` and set `train_end_date` / S3 as needed):  
   `python .../prod_train_rebounds_models.py --config config/nba_rebounds_prod.yaml --feat .../rebounds_model_features.parquet --output-dir .../rebounds_prod_models/RUN_ID`

4. **Score slate** (props for that date must be in the scorer input schema — historical slice or live fetch output):  
   `python .../prod_score_rebounds_slate.py --models-dir .../RUN_ID --feat-slice .../rebounds_features_slice_YYYY-MM-DD.parquet --props .../rebounds_props_scoring_input.parquet --slate-date YYYY-MM-DD --output .../rebounds_scored_YYYY-MM-DD.parquet`

5. **Notify (optional):**  
   `python .../prod_notify_rebounds_sns.py --scored .../rebounds_scored_YYYY-MM-DD.parquet --which both`  
   Set `SNS_TOPIC_ARN` or `--topic-arn` to publish.

6. **Orchestrated daily run (new):**  
   `python scripts/run_nba_rebounds_daily_pipeline.py --config config/nba_rebounds_prod.yaml`  
   (Defaults to **ET today**; optional override: `--slate-date YYYY-MM-DD`.)  
   Retrain cadence is controlled in config (`retrain_daily: true` for prod).  
   For score-only runs with an existing model directory, set `retrain_daily: false` and pass `--models-dir .../RUN_ID`.

**Live props path:** rebounds-only Odds API fetch → S3/local CSV → **props scoring-input builder** (`game_id` join, canonical no-vig fields) → **scoring-input parquet** → `--props` here. Historical bulk path remains `scripts/fetch_nba_player_props.py` + `v2` DuckDB (`market = 'player_rebounds'`).

---

## Design principles (repo)

- **Fail fast** on missing required columns / keys; no silent defaults for critical config.
- **No fake data** in prod paths; integration tests use fixtures only if explicitly scoped.
- **Single source of truth** for: feature definitions, train window, champion spec, edge policy (version in config or manifest alongside each run).

---

## Test + production runbook (operator steps)

### 0) One-time setup

1. Copy config template: `config/nba_rebounds_prod.example.yaml` -> `config/nba_rebounds_prod.yaml`.
2. Set required paths and policy keys:
   - `full_feature_parquet`
   - `daily_runs_root`
   - `retrain_daily: true`
   - `build_props_scoring_input_from_live: true`
   - `live_fetch_to_s3: true` (optional but recommended)
3. Export secrets/env:
   - `ODDS_API_KEY` (or `THE_ODDS_API_KEY`)
   - AWS credentials for S3/SNS (if using S3 upload + notify)
4. If using SNS notifications, set `sns_topic_arn` in config.

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

Pick a known slate date and run:

```bash
python scripts/run_nba_rebounds_daily_pipeline.py \
  --config config/nba_rebounds_prod.yaml
```

Pass criteria:
- Command exits 0.
- Run directory exists under `daily_runs_root/YYYY-MM-DD/<run_id>/`.
- Outputs present:
  - `live_rebounds_props_raw_YYYY-MM-DD.csv`
  - `rebounds_props_scoring_input_YYYY-MM-DD.parquet`
  - `rebounds_features_slice_YYYY-MM-DD.parquet`
  - `rebounds_scored_YYYY-MM-DD.parquet`
  - `models/manifest.json` (daily retrain policy)
- Notify step succeeds (fail-hard policy).

Notes:
- ET-today runs require same-day props input. If `build_props_scoring_input_from_live: true`,
  ensure `ODDS_API_KEY` (or `THE_ODDS_API_KEY`) is set.
- If same-day feature slice is empty pregame, runner automatically builds a fallback
  feature slice from latest historical player form + same-day props context
  (`enable_pregame_feature_backfill: true`).

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
python scripts/run_nba_rebounds_daily_pipeline.py \
  --config config/nba_rebounds_prod.yaml
```

### 5) Daily operator checklist (quick)

- Confirm API key + AWS credentials are valid.
- Confirm `full_feature_parquet` includes latest date needed for slate.
- Run command.
- Verify scored parquet row count + play counts are sane.
- Confirm SNS/email delivery.

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
