# NBA player_threes model v2 plan

**Created:** 2026-03-05  
**Owner:** Thomas + Agents  
**Status:** In progress

## Goal

Stand up a v2 run that upgrades both mean modeling and uncertainty while preserving the modular `01 -> 02 -> 03 -> 04` workflow and reproducible run artifacts.

## Locked v2 changes

1. `01_signal_discovery` upgrades to a 3-input regression:
   - input 1: `mean_3pm`
   - input 2: `predicted_3pa` (new)
   - input 3: `predicted_minutes` (new)
   - output: `predicted_3pm` (`y_hat`)
2. `02_probability_engine` replaces global-only uncertainty with player-history sampling that emphasizes recent games:
   - sample minutes from player game history
   - sample 3PA from player game history
   - support either:
     - weighted finite window (for example latest `n=100` or `n=200` games with weights summing to 1), or
     - exponential decay weighting

## Scope and constraints (v2)

- Target market remains `player_threes`
- Use DuckDB for data loading/querying
- No fake data
- Fail fast for required fields; do not add defensive checks for required keys/columns
- Keep ladder contract handling from v1 (each line is its own contract)
- Keep raw-edge execution rule from v1 (no-vig values are diagnostics only)
- Preserve run artifact format under `src/nba_three_point_modeling/03_backtesting/runs/{run_id}/`
- `run_id` remains `{timestamp}_{mean_model}_{unc_model}_{strategy}` with optional informative suffix

## v2 architecture updates

### `01_signal_discovery` (mean model)

Implementation target:
- Add a v2 model file under `src/nba_three_point_modeling/01_signal_discovery/models/`
- Model fit/predict interface stays compatible with current runner

Required feature contract for v2 model input frame:
- `mean_3pm` (or the existing v1 equivalent mapped clearly)
- `predicted_3pa`
- `predicted_minutes`
- target: `actual_fg3m`

Tasks:
- Define deterministic feature construction for `predicted_3pa` and `predicted_minutes`
- Fit 3-feature regression
- Emit `y_hat` using the v2 model id
- Ensure existing backtest runner can select this model via config

### `02_probability_engine` (uncertainty model)

Implementation target:
- Add v2 uncertainty model under `src/nba_three_point_modeling/02_probability_engine/uncertainty_models/`
- Keep pricing output contract unchanged (`p_over`, `p_under`, `edge_*`, price views)

Sampling policy requirements:
- Draw from player history for:
  - minutes played
  - 3PA
- Weight recent games higher than older games
- Make weighting strategy configurable:
  - `windowed_weighted` (with `history_n`)
  - `exp_decay` (with `decay_alpha`)
- Keep `n_sims` configurable (v2 defaults can start at 1000, with ability to test 100/200/etc.)

Tasks:
- Build weighted sampler helper(s) in `99_utils` or uncertainty module
- Integrate sampler into probability simulation path
- Keep raw and no-vig implied/edge outputs intact
- Document chosen default weighting rule in config comments

## Backtesting and validation updates

### `03_backtesting`

Tasks:
- Extend `current_config.yaml` schema to support v2 knobs, e.g.:
  - `mean_model_id: v2_three_input_regression`
  - `uncertainty_model_id: v2_weighted_history_sampler`
  - `history_n`
  - `weighting_mode`
  - `decay_alpha` (if decay mode used)
- Ensure config snapshot includes new params in run artifacts

### `04_validation`

Tasks:
- Compare v2 vs latest stable v1 run on:
  - RMSE
  - ROI
  - win rate
  - signal rate
  - calibration diagnostics (if already in pipeline)
- Produce concise delta summary (what improved, what regressed)

## Implementation task checklist

- [x] Add v2 plan-aware docs entry in active README
- [x] Implement 3-input mean model in `01_signal_discovery`
- [x] Implement recency-weighted player-history uncertainty model in `02_probability_engine`
- [x] Wire model/uncertainty selection through backtest config
- [x] Add one-player full-history fetch utility for `s3://nba-api-mt/full_player_history/{full name}.csv`
- [x] Add or update tests for:
  - model interface contracts
  - weighted sampling behavior (weights sum to 1, recent games have higher mass)
  - reproducibility assumptions for backtest outputs
- [x] Run targeted test suite for `tests/analysis/player_threes_v1/` plus any v2 tests

## Deliverables

- Updated docs:
  - `docs/exec-plans/active/nba-player-threes-model-v2-plan.md`
  - `docs/exec-plans/active/README.md`
- Code changes across:
  - `src/nba_three_point_modeling/01_signal_discovery/`
  - `src/nba_three_point_modeling/02_probability_engine/`
  - `src/nba_three_point_modeling/03_backtesting/` (config wiring)
  - `tests/analysis/` (v2 coverage)
- One successful v2 run directory with full artifacts and summary metrics

## Completion criteria

Plan is complete when:
- A v2 `run_id` is produced end-to-end with the new 3-input mean model and recency-weighted uncertainty model
- Run artifacts are complete and readable with v2 config recorded
- Validation output clearly compares v2 vs v1 and states net impact

## Related

- `docs/exec-plans/active/nba-player-threes-model-v1-plan.md`
- `src/nba_three_point_modeling/01_signal_discovery/`
- `src/nba_three_point_modeling/02_probability_engine/`
- `src/nba_three_point_modeling/03_backtesting/current_config.yaml`

---

## 2026-03-05 add-on: player_team_history dual-artifact pipeline

This active plan now also tracks a supporting data-pipeline extension under `src/player_team_history/` to keep player-team joins and box-score coverage in sync for downstream model features.

### Goal

Extend `src/player_team_history/01_build.py` from one artifact to two:
- `history.parquet` (team stints)
- `box_scores.parquet` (one row per `player_normalized + Game_ID`, with `PLAYER_INFO_*` metadata)

### Locked decisions

- Keep current `src/player_team_history/` workflow (`01_build.py`, `02_analyze_failures.py`, `03_cache.py`, `04_validate.py`)
- Reuse normalized player universe and existing season/player cache logic
- Keep build resumable with checkpoints and cache reuse
- Fail fast for required fields (`Player_ID`, `Game_ID`, `GAME_DATE`, `SEASON_ID`, `TEAM`)
- Keep failures actionable (mapping issues vs expected no-games vs processing/schema errors)

### Implementation tasks

- [x] Extend `01_build.py` to build both team stints and player/game box rows
- [x] Persist `box_scores.parquet` and `box_scores_checkpoint.parquet` under `~/Downloads/tmp/player_team_history/`
- [x] Add `cache/player_info/*.parquet` and include `PLAYER_INFO_*` metadata in box rows
- [x] Extend failure report categories for box-score-specific failures
- [x] Extend `02_analyze_failures.py` parsing/reporting for new failure sections
- [x] Extend `03_cache.py --stats` to include player_info cache + box artifact summary
- [x] Extend `04_validate.py` to validate both history and box-score outputs

### Runbook

Sample run:
```bash
python src/player_team_history/01_build.py --sample 100 --verbose
```

Larger sample:
```bash
python src/player_team_history/01_build.py --sample 1000
```

Validate outputs:
```bash
python src/player_team_history/04_validate.py
```

Inspect caches and box artifact:
```bash
python src/player_team_history/03_cache.py --stats
```

Analyze failures for mapping iteration:
```bash
python src/player_team_history/02_analyze_failures.py
```
