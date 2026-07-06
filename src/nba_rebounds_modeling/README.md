# NBA Rebounds Modeling

Predicts total rebounds (`REB`) vs market line for NBA player props.
Strategy: **Option A under-only** plays filtered by edge threshold.

This model is proven in production and has been running live.

## Model

Two models trained jointly on the feature universe:

| Model | Artifacts |
|---|---|
| OLS (statsmodels) | `ols_model.pkl` |
| XGBoost | `xgb_model.json` |

Features (`rebounds_feature_spec.py` is the canonical source of truth):

```
min_line, max_line, spread_signed,
roll_reb_mean_60, roll_fg3a_mean_20, roll_reb_std_5
```

**If you change `rebounds_feature_spec.py`, update `docs/design-docs/nba-rebounds-daily-pipeline.md` and re-run tests.**

## S3 paths

| Resource | Path |
|---|---|
| Input universe | `s3://nba-betting-mt/rebounds/input/rebounds_input_universe.parquet` |
| Feature universe | `s3://nba-betting-mt/rebounds/features/` |
| Model artifacts | `s3://nba-betting-mt/rebounds/models/{run_id}/` |
| Scored slates | `s3://nba-betting-mt/rebounds/scored/` |

## Daily pipeline

On game days the flow is:

1. **Slice features** for today's slate
2. **Score** with trained models → Option A plays
3. **Notify** via SNS

```sh
python src/nba_rebounds_modeling/00_research/scripts/run_rebounds_daily_pipeline.py
```

Or step by step:

```sh
# Slice features for today
python src/nba_rebounds_modeling/00_research/scripts/slice_rebounds_features_for_slate.py \
    --slate-date 2025-03-15

# Score
python src/nba_rebounds_modeling/00_research/scripts/prod_score_rebounds_slate.py \
    --models-dir ~/Downloads/tmp/rebounds_prod_models/run_001 \
    --feat-slice ~/Downloads/tmp/rebounds_features_slice_2025-03-15.parquet \
    --props ~/Downloads/tmp/rebounds_props_scoring_input.parquet \
    --slate-date 2025-03-15 \
    --output ~/Downloads/tmp/rebounds_scored_2025-03-15.parquet

# Notify
python src/nba_rebounds_modeling/00_research/scripts/notify_rebounds_plays.py
```

## Retraining

Rebuild the input universe (append mode for incremental updates):

```sh
python src/nba_rebounds_modeling/00_research/scripts/build_rebounds_input_universe.py \
    --season "*" \
    --output /tmp/rebounds_prod/cache/rebounds_input_universe.parquet \
    --s3-uri s3://nba-betting-mt/rebounds/input/rebounds_input_universe.parquet \
    --mode append
```

Train and serialize artifacts:

```sh
python src/nba_rebounds_modeling/00_research/scripts/prod_train_rebounds_models.py \
    --feat ~/Downloads/tmp/rebounds_model_features_v2.parquet \
    --output-dir ~/Downloads/tmp/rebounds_prod_models/run_001
```

## Settle

```sh
python src/nba_rebounds_modeling/00_research/scripts/settle_rebounds_runs.py
```

## Scripts reference

### Prod scripts (`00_research/scripts/prod_*` and entry points)

| Script | Purpose |
|---|---|
| `run_rebounds_daily_pipeline.py` | Gameday orchestrator — delegates to `scripts/run_rebounds_daily_pipeline.py` |
| `build_rebounds_input_universe.py` | Build canonical player-date-game input parquet (FGA, FG3A, FTA, spread) |
| `prod_train_rebounds_models.py` | Train OLS + XGB, serialize artifacts, optional S3 upload |
| `slice_rebounds_features_for_slate.py` | Slice feature universe to a single slate date |
| `prod_slice_rebounds_features.py` | Implementation behind the above |
| `prod_score_rebounds_slate.py` | Score a slate: load models + feature slice + props → plays parquet |
| `prod_notify_rebounds_sns.py` | Send SNS notification with today's plays |
| `notify_rebounds_plays.py` | Thin wrapper around `prod_notify_rebounds_sns.py` |
| `settle_rebounds_runs.py` | Settle run artifacts with realized outcomes |

### Root-level modules

| File | Purpose |
|---|---|
| `rebounds_feature_spec.py` | **Canonical** feature list + group keys — single source of truth for training and scoring |
| `option_a_scoring.py` | Option A under-only play selection logic |
| `rebounds_audit_list_verify.py` | Verify audit columns are present on feature rows |
| `duckdb_s3_creds.py` | DuckDB S3 credential helper for local parquet queries |

### Research scripts (`00_research/scripts/v*_`)

Versioned exploration — do not modify. Reference only.

| Script | What it explored |
|---|---|
| `20260323_build_rebounds_universe.py` | Initial universe build |
| `20260323_run_rebounds_edge_backtest.py` | Edge backtest framework |
| `20260323_run_rebounds_under_only_season_robustness.py` | Under-only strategy robustness across seasons |
| `20260323_compare_rebounds_models_oos.py` | OLS vs XGBoost out-of-sample comparison |
| `20260323_diagnose_xgb_vs_linear_rebounds_oos.py` | XGBoost vs linear diagnosis |
| `20260323_spec_sweep_rebounds_oos.py` | Feature spec sweep |
| `compare_dedupe_backtest.py` | Backtest with deduplication at player-game-line level |
