# NFL Sacks Modeling

Predicts P(player sacks ≥ 1) on the 0.5 line for NFL defensive players.
Strategy: bet **Under 0.5 sacks** when model P(over) < threshold.

**OOS performance (2024 + 2025):** +71.8u, +4.0% ROI on 1,779 bets.

## Model

Logistic Regression (sklearn Pipeline: StandardScaler + LR) trained on the 0.5 line only.
Pushes (sacks == 0.5) excluded from training per `config.yaml`.

Target: `is_over` — 1 if `sacks >= 1.0`, 0 if `sacks == 0.0`.

Features: rolling sack rate, QB hit rate, snap%, game total, team spread,
games played YTD, position group/side, implied over/under probabilities (multi-book).

Rolling windows from `config.yaml`: 1, 3, 5, 8, 16, career.
Player identity keyed on `pfr_player_id` (not name) to handle same-name collisions.

## S3 paths

| Resource | Path |
|---|---|
| Historical spine | `s3://the-odds-api-mt/nfl/sacks_model/spine/nfl_sacks_historical_spine.parquet` |
| Trained model | `s3://the-odds-api-mt/nfl/sacks_model/model/lr_model.pkl` |
| Daily bet sheets | `s3://the-odds-api-mt/nfl/sacks_model/daily_runs/{gameday}/bet_sheet.{csv,html}` |
| Settled sheets | `s3://the-odds-api-mt/nfl/sacks_model/daily_runs/{gameday}/bet_sheet_settled.csv` |

## Pipeline (local / manual)

All scripts run from repo root.

### Build spine

```sh
python src/nfl_sacks_modeling/scripts/build_historical_spine.py
# or for a specific season:
python src/nfl_sacks_modeling/scripts/build_historical_spine.py --seasons 2025
# force re-fetch all from nfl_data_py:
python src/nfl_sacks_modeling/scripts/build_historical_spine.py --force-refetch
```

Output: `~/Downloads/tmp/nfl_sacks_historical_spine.parquet`
Cache: `~/Downloads/tmp/nfl_sacks_spine_cache/{season}/{pfr_player_id}.parquet`

### Build features + train

```sh
python src/nfl_sacks_modeling/scripts/build_sacks_features.py
python src/nfl_sacks_modeling/scripts/train_model.py
```

### Run pipeline (gameday)

```sh
python src/nfl_sacks_modeling/scripts/run_pipeline.py --gameday 2026-09-11
python src/nfl_sacks_modeling/scripts/run_pipeline.py   # defaults to today ET
```

Fetches live props from Odds API, joins spine, scores, uploads bet sheet to S3, sends SES/SNS notification.

### Settle bets

```sh
python src/nfl_sacks_modeling/scripts/settle_sacks.py --gameday 2026-09-11
python src/nfl_sacks_modeling/scripts/settle_sacks.py   # defaults to yesterday ET
```

Settlement: win if sacks == 0, push if sacks == 0.5, loss if sacks >= 1.

## Lambda deployment

Container image deployed to ECR; Lambda reads `mode` from the EventBridge payload.

```sh
bash src/nfl_sacks_modeling/lambda/deploy_nfl_sacks_lambda.sh
```

### Lambda modes

| Mode | Schedule | Purpose |
|---|---|---|
| `spine_update` | Tue 9am ET | Re-fetch current season from nfl_data_py, upload spine to S3 |
| `spine_verify` | Wed 9am ET | Rebuild spine locally, compare vs S3 (does NOT upload) |
| `pipeline` | Thu/Sun/Mon 9am ET | Fetch live props, score, upload bet sheet, notify |
| `settle` | Daily 10am ET | Settle yesterday's bets, send HTML summary email |

### EventBridge rules

**5 rules, all DISABLED. Enable before 2026-09-09 (week 1).**

Env vars required in Lambda: `ODDS_API_KEY`, `SNS_TOPIC_ARN`.
Optional: `SES_SOURCE`, `SES_TO` (comma-separated) for HTML emails, `NFL_SEASON`.

## Scripts reference

| Script | Purpose |
|---|---|
| `build_historical_spine.py` | Multi-season spine from nfl_data_py (PBP + snap counts, 2013–present) |
| `build_sacks_dataset.py` / `_v2` / `_v3` | Versioned dataset builds (v3 is current) |
| `build_sacks_features.py` | Join spine + Odds API props → feature parquet |
| `build_features.py` | Feature engineering helpers |
| `train_model.py` | Fit LR, calibrate via 5-fold CV, serialize to pkl |
| `run_pipeline.py` | Gameday entrypoint: fetch → score → upload → notify |
| `update_spine.py` | Incremental spine update (current season only) |
| `settle_sacks.py` | Settle bets + send SES summary |
| `oos_eval.py` | OOS evaluation across seasons |
| `calibration.py` | Calibration curve analysis |
| `compare_models.py` | Compare LR vs other model specs |
| `strategy_grid.py` | Threshold/strategy grid search |
| `eval_vs_market.py` | Model edge vs market implied probs |
| `shap_analysis.py` | SHAP feature importance |
| `eda_sacks_2025.py` | Exploratory analysis on 2025 data |
| `garrett_validation_table.py` | Myles Garrett case study validation |
| `fetch_game_lines_cle_2025.py` | One-off: CLE 2025 game lines fetch |
| `fetch_garrett_sacks_coverage.py` | One-off: Garrett prop coverage check |

Research scripts (`research/scripts/`): early ideation only, not used in prod.
