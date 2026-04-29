# Plan v4: audit lists in **prod pipeline** + tests + prod bash verifier

## Intent (what you asked for)

**Pytest is not the primary gate.** The same contracts as `rebounds_audit_list_verify.py` should run **while the real universe / feature build runs** (the path that ultimately feeds plays: `build_rebounds_full_universe` → feature parquet → slice → score). If audit lists drift from scalars, **the pipeline must fail** (non-zero exit) before bad rows are uploaded to S3 or used downstream.

Pytest stays useful for **fast regression** and **refactors**; prod + bash cover “this actually ran against today’s artifact.”

---

## Problem statement (short)

- `tests/unit/test_rebounds_audit_lists.py` uses **synthetic** data and opaque names (`v2bru`, `v6`). Good for algebra, not sufficient as the only safety net.
- DuckDB `run_quality_checks` today checks nulls/dups/sample rows — it does **not** assert list↔scalar consistency.

---

## Contract (single source of truth)

`src/nba_rebounds_modeling/rebounds_audit_list_verify.py`:

- `verify_audit_lists_row` / `verify_audit_lists_dataframe` vs `B_MIN_MAX_FEATS` + `B_MIN_MAX_AUDIT_LIST_COLS` in `rebounds_feature_spec.py`.

Rules unchanged: lines ↔ min/max; tails ↔ rolling means/std; spread list + team side ↔ `spread_signed`.

---

## A. In-pipeline verification (primary) — **do this first**

### Where

`src/nba_rebounds_modeling/00_research/scripts/build_rebounds_full_universe.py`, after **`feature_universe`** is assembled in memory (same object that is written to parquet), **before** or **immediately after** `to_parquet` — prefer **before write** so a bad build never ships.

Today flow (reference):

- `feature_universe = panel[...].merge(logs_target).merge(rolling)` then `require_columns` / dup check / `to_parquet` / `run_quality_checks`.

### What to call

- **`verify_audit_lists_dataframe(feature_universe, team_frame=..., max_rows=N)`**  
  - **Lines + tails:** use rows from `feature_universe` only (no extra join).
  - **Spread:** `verify_audit_lists_row` only gets spread checks when `team_normalized`, `home_team_norm`, `away_team_norm` are supplied. `feature_universe` does not currently carry home/away on each row; build a **`team_frame`** keyed by `["season", "date", "player_normalized", "game_id"]`:
    - Left: `logs[["season","date","player_normalized","game_id","team_normalized"]].drop_duplicates(...)`
    - Right: game-level attach used inside `attach_spread` — the frame that has `home_team_norm`, `away_team_norm` per `(season, date, game_id)` (today merged into `gattach` / `gspread` path; **refactor** to retain one small DataFrame `game_spread_sides_df` with keys + `home_team_norm`, `away_team_norm` for reuse in audit, or recompute with a tiny helper next to `attach_spread` to avoid duplication drift).

### Sampling vs full scan

| Mode | When |
|------|------|
| **Sampled** (default) | `max_rows=500` (or 1000) random seed fixed — cheap every Lambda run. |
| **Full** (optional) | Env `REBOUNDS_AUDIT_LIST_FULL_SCAN=1` or CLI `--audit-list-full-scan` — nightly / manual; O(rows) Python loop may be slow on 30k+ rows — acceptable for explicit runs; for prod daily default keep **sampled** unless measured fast enough. |

### Fail-hard + kill switch

- **Default (prod):** audit **on**; any `AssertionError` / `ValueError` from verifier → **exit non-zero** → Lambda step fails → no S3 upload of bad universe (if placed before upload; if after write, delete partial object or rely on “next run overwrites” — **prefer before write**).
- **Kill switch:** `REBOUNDS_AUDIT_LIST_STRICT=0` skips audit (document for emergency backfill only). Omit or `1` in normal prod.

### Logging

- Print one line: `audit_list_verify | mode=sample|full | n_checked=... | ok`  
- On failure: print `(season, date, player_normalized, game_id)` for the failing row before re-raising.

### Optional extension

- **`run_quality_checks`**: add optional DuckDB **SQL spot checks** for invariants that are easy in SQL (e.g. `min_line <= max_line`, nonempty lines when min_line non-null) — **secondary** to Python verifier; keeps one mental model in `rebounds_audit_list_verify.py`.

---

## B. Pytest (secondary) — keep, tighten

- **Tier 0:** Rename `v2bru` → `universe_builder`, `v6` → `shot_profile_df` in tests; optional rename `v6_shots` in `build_rebounds_full_universe.py`.
- **Tier 1:** Small committed parquet fixtures under `tests/fixtures/rebounds_audit/` + `scripts/extract_rebounds_audit_fixture.py` (provenance README).
- **Tier 2:** `@pytest.mark.integration` + env `REBOUNDS_AUDIT_FEAT_PATH` / S3 for large sampled runs.

Pytest proves **code**; pipeline + bash prove **deployment and data path**.

---

## C. Bash script — **verify prod (and local) end-to-end**

Add **`scripts/verify_rebounds_audit_lists_prod.sh`** (or `bash scripts/verify_rebounds_audit_lists.sh --prod`).

### Responsibilities

1. **Prereqs:** `aws` CLI, `python3`, repo root; optional `duckdb` CLI not required if Python does I/O.
2. **Download** (or accept local path):
   - Default prod URIs aligned with `config/nba_rebounds_prod.lambda.yaml`:
     - `s3://nba-betting-mt/rebounds/features/rebounds_feature_universe.parquet` (or override `REBOUNDS_FEAT_S3_URI`)
     - For spread checks: `s3://nba-betting-mt/rebounds/input/rebounds_input_universe.parquet` (or override) — columns needed to build `team_frame` (player’s team + game; if input universe lacks home/away, script merges same way as verifier docstring or skips spread subset with a loud warning).
3. **Run** a small Python entrypoint (preferred over inline `python -c`):

   **`scripts/verify_rebounds_audit_lists_parquet.py`** (new):

   - Args: `--parquet PATH`, `--input-universe PATH` (optional), `--max-rows N`, `--full-scan`, `--strict-spread` (fail if spread columns missing when input universe provided).
   - Loads parquet with pandas/pyarrow, builds `team_frame` if possible, calls `verify_audit_lists_dataframe`.
   - Exit code **0** ok, **1** failure, **2** skip/misconfig (e.g. missing columns with doc message).

4. **Exit** non-zero on verification failure so CI / cron / human `&&` chains work.

### Example usage (document in script header)

```bash
# Prod artifacts (read-only)
export AWS_REGION="${AWS_REGION:-us-east-2}"
REBOUNDS_AUTO_BUILD_IF_MISSING_TEAM=0 bash scripts/verify_rebounds_audit_lists_prod.sh

# Local file
python scripts/verify_rebounds_audit_lists_parquet.py \
  --parquet "$HOME/Downloads/tmp/rebounds_full_universe.parquet" \
  --max-rows 2000
```

Auto-build of a missing feature file uses `REBOUNDS_AUTO_BUILD_IF_MISSING_TEAM=1` (default in the prod shell) and may run `build_rebounds_feature_universe.py`; that path needs the same S3 access as a normal full build (see **Local full feature build** in `plan_rebs_model_inputs_v3.md`).

### Scheduling (optional)

- **EventBridge** weekly rule invoking a **minimal** Lambda or ECS one-off that only downloads + runs verifier (cheaper than full pipeline) — only if you want monitoring independent of daily build; otherwise **relying on in-pipeline checks** is enough.
- Alternatively: run the bash script **manually after deploy** and document in `plan_rebs_results_formatting_v2.md` deploy checklist.

---

## Implementation order (revised)

1. **Refactor** `attach_spread` (or `main`) to expose **`game_spread_sides_df`** (or equivalent) for building `team_frame` without re-querying S3.
2. **Call** `verify_audit_lists_dataframe` from `build_rebounds_full_universe.main()` with env/CLI for sample vs full + strict flag.
3. **`scripts/verify_rebounds_audit_lists_parquet.py`** + **`scripts/verify_rebounds_audit_lists_prod.sh`**.
4. **Pytest** renames + fixtures + integration marker (unchanged priority from prior plan).

---

## Acceptance criteria

- A deliberate bug in tail vs scalar construction causes **`build_rebounds_full_universe`** to exit **non-zero** in a normal run (with strict on).
- **`verify_rebounds_audit_lists_prod.sh`** succeeds against current prod S3 artifacts when AWS creds are valid.
- Default daily prod path remains **fast enough** (sampled audit); full scan is opt-in.
- Pytest still passes in CI without S3 for the lightweight cases.

---

## Implementation status (landed in repo)

- **`rebounds_feature_spec.py`**: **`TEAM_CONTEXT_COLS`** (`team_normalized`, `home_team_norm`, `away_team_norm`) on feature-universe output (not model features; for spread audit + S3 verify).
- **`build_rebounds_full_universe.py`**: Merges team + home/away into **`feature_universe`** before write; `attach_spread` / `build_audit_team_frame` unchanged for rows missing inline context. `verify_audit_lists_dataframe` after dup check, etc.
- **`rebounds_audit_list_verify.py`**: `team_audit_kwargs_from_row` first, then `team_frame` join; ...; **`numpy.ndarray`** length-2 spread pairs accepted (parquet/merge) so spread audit does not spuriously fail.
- **`src/nba_rebounds_modeling/duckdb_s3_creds.py`**: DuckDB S3 config uses **boto3** credentials (SSO/session) for `build_rebounds_input_universe` / `build_rebounds_full_universe` to avoid 403s when only the CLI has valid temporary creds.
- **`scripts/verify_rebounds_audit_lists_parquet.py`**: ... optional auto-build when team columns missing (and `s3://`); stderr note if feature file **lacks** `TEAM_CONTEXT_COLS` when auto-build is off.
- **`scripts/verify_rebounds_audit_lists_prod.sh`**: ... **`REBOUNDS_TEAM_FRAME_URI`** only for **legacy** parquets; after one **full universe build + upload**, only **`REBOUNDS_FEAT_PARQUET_URI`** is required.
- **`prod_slice_rebounds_features.py`**: `SCHEMA_VERSION` bumped for new columns carried through date slice.
- **Pipeline** (`run_rebounds_daily_pipeline.py` → `build_rebounds_feature_universe.py` → S3) matches the above; deploy Lambda to refresh production artifacts.
- **`tests/unit/test_rebounds_audit_lists.py`**: synthetic + **ndarray spread** case.
