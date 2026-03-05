# NBA player_threes model v1 plan

**Created:** 2026-03-04  
**Owner:** Thomas + Agents  
**Status:** In progress

## Goal

Build a modular, reproducible workflow for NBA `player_threes` modeling where:
- `01_signal_discovery` produces mean prediction (`y_hat`)
- `02_probability_engine` converts mean to uncertainty-aware probabilities/edges
- `03_backtesting` applies strategy rules and simulates results
- `04_validation` compares runs and benchmarks improvements vs baselines

## Scope and constraints (v1)

- Target market: `player_threes`
- Seasons: `2023-24`, `2024-25`, `2025-26` for build context; v1 backtest scope can be one player/one season
- DuckDB-first data loading from caches/S3 paths
- Box scores remain canonical left-side table (null market/context joins allowed)
- Different ladder lines (0.5, 1.5, 2.5, ...) are separate bet contracts
- Monte Carlo simulation required (`n=1000` default); do not use point-mean alone for betting decisions

## Modular workflow structure

Root module path:
- `src/nba_three_point_modeling/`

### `00_research`

Responsibility:
- Sandbox for ad-hoc exploration and early idea testing before promotion to formal modules.

v1 scope:
- Add exploratory notebook under `src/nba_three_point_modeling/00_research/notebooks/`
- Include quick sanity check such as average 3PM by player sorted descending (Steph should appear near top).

### `01_signal_discovery`

Responsibility:
- Core statistical modeling and feature evaluation:
  - regression/model fitting
  - regularization/feature selection
  - cross-validation
  - stability + predictive power assessment
- Build derived metrics (for example, 3PT z-score relative to league distribution).

Output:
- Validated mean-model specification + engineered feature set.

v1 scope:
- Baseline OLS model predicting game `FG3M` using `season_avg_3pm` as single feature.

Model file convention:
- `src/nba_three_point_modeling/01_signal_discovery/models/{baseline,v1,...,v99}.py`
- Example files: `baseline.py`, `v1.py`
- Each model exposes a stable interface like `predict(features) -> y_hat`
- Coefficients/rules can be stored directly in code for v1.

### `02_probability_engine`

Responsibility:
- Convert each game/player mean prediction to distribution-aware pricing outputs at each line.
- Compute `P(over)` / `P(under)`, fair odds, and edge vs market implied probability.
- Assume market lines/odds are provided at player-game-line level.

v1 scope:
- Use `y_hat` + fixed global historical variance (same for all players).
- Run Monte Carlo (`n=1000` default) to estimate probabilities and edges.
- No player-specific variance model yet.

Uncertainty model convention:
- `src/nba_three_point_modeling/02_probability_engine/uncertainty_models/`
  - `global_variance.py` (v1)
  - `player_specific_variance.py` (v2 candidate)
  - `baseline.py` allowed for naive benchmarks

### `03_backtesting`

Responsibility:
- Own strategy logic: thresholds, sizing, decision rules.
- Simulate historical decisions and realized PnL.
- Write reproducible run artifacts.

v1 scope:
- Single model/strategy, one player, one season (for example Curry 2025-26).
- Configurable params, but constrained evaluation domain.

Output convention:
- `src/nba_three_point_modeling/03_backtesting/runs/{run_id}/`
  - `config.yaml`
  - `manifest.json`
  - `predictions.parquet`
  - `bets.parquet`
  - `summary.json`

### `04_validation`

Responsibility:
- Evaluate and benchmark results from `03_backtesting/runs/*`.
- Compare current run against prior runs, alternative specs, and naive baseline.
- Distinguish real improvement vs variance/overfit.

v1 scope:
- Compare one run against a naive baseline (for example rolling 5-game average predictor).
- Produce standardized metric summary and comparison table.

Output convention:
- `validation_summary.json` per run
- `comparison_table.parquet` and/or `comparison_table.csv`

### `99_utils`

Responsibility:
- Shared helpers across modules:
  - math/stat helpers
  - normalization and scaling
  - z-score and rolling calculations
  - config loading and shared data transforms

## Data scope and source tables

Primary sources:
1. Player game logs (canonical left):
   - `s3://nba-api-mt/player_game_logs/{season}/*.csv`
2. Historical player props (`market = 'player_threes'`):
   - `s3://the-odds-api-mt/nba/historical_player_props/{season}/*.csv`
3. Historical game lines (`moneyline`, `spread`):
   - `s3://the-odds-api-mt/nba/historical_game_lines/{season}/nba_game_lines_*.csv`

Join policy:
1. Build canonical player-game rows from box scores.
2. Parse `home_team`/`away_team` from matchup.
3. Normalize dates to ET day.
4. Join props by player + home/away + ET date.
5. Join game context by home/away + ET date.
6. Preserve left rows even if market/context missing.

## Interface contracts

### `01_signal_discovery` output

`predictions_df` columns:
- `run_id`
- `game_id`
- `player_id`
- `date`
- `y_hat`
- `model_id`
- `model_version`
- `feature_version`

### `02_probability_engine` input

- `predictions_df`
- `lines_df` columns:
  - `game_id`
  - `player_id`
  - `date`
  - `sportsbook`
  - `market`
  - `line`
  - `odds_over`
  - `odds_under`
  - `snapshot_ts`
  - `is_consensus` (binary flag for consensus line rows)

### `02_probability_engine` output

`priced_lines_df` columns:
- `run_id`
- `game_id`
- `player_id`
- `date`
- `line`
- `p_over`
- `p_under`
- `fair_odds_over`
- `fair_odds_under`
- `edge_over`
- `edge_under`
- `uncertainty_model_id`
- `n_sims`

### `03_backtesting` inputs and outputs

Inputs:
- `priced_lines_df`
- `strategy_config`

Outputs under `03_backtesting/runs/{run_id}/`:
- `predictions.parquet` (game-level `y_hat` + merged market snapshot)
- `bets.parquet` columns:
  - `run_id`
  - `game_id`
  - `player_id`
  - `date`
  - `line`
  - `side`
  - `odds`
  - `stake`
  - `p_model`
  - `edge`
  - `result`
  - `pnl`
- `summary.json` (ROI, win rate, hit rate, calibration stats, percent signals, etc.)
- `manifest.json` (config snapshot + git SHA + data_version)

### `04_validation` inputs and outputs

Inputs:
- one or more `03_backtesting/runs/{run_id}/` folders

Outputs:
- `validation_summary.json` per run
- `comparison_table.parquet` and/or `.csv` with run vs baseline deltas

## Market line policy

### Modeling (`01_signal_discovery`)

- Use consensus line only (closest to 50/50 after vig removal).
- Treat this as market-implied median outcome for residual comparisons.
- Vig-removal default for v1: normalize implied probs from listed odds (no longshot-bias correction in scope).
- Add binary marker `is_consensus` to prop-line rows so downstream filtering/debugging can use `is_consensus == 1`.

### Validation metrics vs market (`04_validation`)

- For residual-vs-market metrics, use consensus line only for consistency.

### Backtesting (`03_backtesting`)

- Evaluate all available lines (including alt lines) when enabled in config.
- Ladder lines are independent contracts.
- A single game can generate multiple valid ladder bets if edge thresholds are met.
- For each player/game/line/side combo, retain:
  - median price view
  - best price view (and record source book)
- Best price for a given contract should be equal to or better than median price for that same contract.

### Median-price computation rule (locked)

At fixed `player/game/line/side`:
- Do not compute median directly on American odds.
- Convert each book odds to implied probability first.
- Take median in implied-probability space.
- Convert that median implied probability back to American odds for the median-price view.

Example:
- American odds: `-100`, `-150`, `-200`
- Implied probs: `50.0%`, `60.0%`, `66.7%`
- Median implied prob: `60.0%`
- Median-price view odds: `-150`

## Probability and simulation policy (required)

- Mean prediction alone is insufficient for bet decisions.
- Use Monte Carlo simulation with `n=1000` default.
- v1 uncertainty model uses global variance.
- Compute per contract:
  - `p_over`
  - `p_under`
  - market implied probability from odds
  - probability edge (`model_prob - implied_prob`)

## Run flow automation

### Step 0: ideation
- Prototype in `00_research` on small slice.
- Promote stable ideas into `01_signal_discovery/models/` or `02_probability_engine/uncertainty_models/`.

### Step 1: configure
- Edit `src/nba_three_point_modeling/03_backtesting/current_config.yaml`:
  - `mean_model_id`
  - `uncertainty_model_id`
  - `strategy_id`
  - key params (`threshold`, `n_sims`, `season`, `player`, etc.)

### Step 2: run backtest
- Script reads `current_config.yaml`.
- Auto-generates `run_id = {timestamp}_{mean_model}_{unc_model}_{strategy}`.
- Optional suffixes can be appended for key filters/thresholds when needed for readability.
- Creates `runs/{run_id}/`, snapshots config, writes all run artifacts.

### Step 3: run validation
- `04_validation` reads `runs/{run_id}/` and compares to prior runs.
- Writes standardized comparison outputs.

## v1 metrics

Primary:
- RMSE (model mean prediction vs actual box score result)
- Win rate
- ROI
- Percent signals generated

Stake/ROI convention (v1):
- Assume unlimited bankroll.
- Stake per bet is sized to target `$100` profit:
  - negative American odds (for example `-110`): stake `110` to win `100`
  - positive American odds (for example `+200`): stake `50` to win `100`
- Track units won/lost and ROI on total capital risked.

Market alignment diagnostics:
- Mean/std distribution of residuals:
  - model prediction vs consensus line (market-implied median)

Probability diagnostics:
- Brier score or equivalent probability scoring metric

## Project phases

### Phase 1 - module scaffolding + data contracts
- [ ] Create module folder tree under `src/nba_three_point_modeling/`.
- [ ] Implement shared schema contracts for module interfaces.
- [ ] Define consensus-line and ladder-contract extraction rules.

### Phase 2 - v1 signal discovery
- [ ] Implement baseline OLS (`season_avg_3pm` only).
- [ ] Output `predictions_df` with contract fields.
- [ ] Add naive baseline model file for future comparisons.

### Phase 3 - v1 probability engine
- [ ] Implement `global_variance.py`.
- [ ] Simulate `n=1000` outcomes per contract.
- [ ] Output `priced_lines_df` with edges.

### Phase 4 - v1 backtesting
- [ ] Implement config-driven strategy runner.
- [ ] Write deterministic run artifacts and manifest.
- [ ] Support optional all-lines evaluation toggle.

### Phase 5 - v1 validation
- [ ] Compare one run vs naive baseline.
- [ ] Emit standardized validation summary + comparison table.

## Test suite plan

Core tests:
- `tests/analysis/player_threes_v1/test_interface_contracts.py`
- `tests/analysis/player_threes_v1/test_consensus_line_policy.py`
- `tests/analysis/player_threes_v1/test_ladder_contracts.py`
- `tests/analysis/player_threes_v1/test_monte_carlo_probabilities.py`
- `tests/analysis/player_threes_v1/test_backtest_outputs.py`
- `tests/analysis/player_threes_v1/test_validation_comparison.py`

## Completion criteria

Plan is complete when:
- Modular folders/interfaces exist and are used end-to-end.
- `01 -> 02 -> 03 -> 04` handoff works with reproducible run artifacts.
- v1 OLS + global variance Monte Carlo (`n=1000`) runs for one player/season.
- Backtest reports RMSE, residual diagnostics vs consensus, win rate, ROI, and percent signals.
- Validation compares run against naive baseline and writes standardized outputs.

## Locked v1 defaults

- Vig handling for consensus line:
  - no longshot-bias adjustment in v1
  - use standard no-vig normalization from listed odds
- Equally close 50/50 situations:
  - retain both median-price and best-price views at player/game/line/side contract level
  - best-price view must include source sportsbook
  - median-price is computed in implied-probability space, then converted back to American odds
- Stake sizing:
  - unlimited-bankroll assumption
  - per-bet stake targets `$100` win amount
- `run_id` format:
  - `run_id = {timestamp}_{mean_model}_{unc_model}_{strategy}`
  - optional suffix for thresholds/filters is allowed

## Related

- `scripts/build_nba_multimarket_strategy_dataset.py`
- `docs/exec-plans/active/nba-multimarket-strategy-analysis.md`
- `docs/domain/betting-fundamentals.md`
- `docs/domain/market-mechanics.md`
