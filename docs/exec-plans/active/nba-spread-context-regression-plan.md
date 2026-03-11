# NBA spread context regression plan

**Created:** 2026-03-10  
**Owner:** Thomas + Agents  
**Status:** Proposed

## Goal

Measure how much pregame game spread context helps predict player box score outcomes, and identify which targets are most sensitive to spread signal.

Specifically compare:
- Continuous spread (`x = spread_signed`) where underdogs are positive and favorites are negative.
- Bucketed spread (`x = spread_bin`) for nonlinear effects.

Run this across multiple `y` targets (minutes and box score stats) to rank where spread context matters most.

## Scope

- Domain: NBA player-game rows in the existing v5 market-eligible universe.
- Inputs: reuse phase-0 universe and add game-spread context join.
- Method: univariate OLS (spread only), plus optional two-feature model with `market_consensus_line`.
- Primary audience: research agent executing fast, reproducible sweeps.

## Key definitions

- `spread_signed`: Pregame spread from player perspective.
  - If player team is favored by 7.5, `spread_signed = -7.5`.
  - If player team is underdog by 7.5, `spread_signed = +7.5`.
- `spread_abs`: `abs(spread_signed)`.
- `spread_bin`: Ordered categorical bins from large favorite to large dog.
  - Proposed edges: `(-inf,-12], (-12,-8], (-8,-4], (-4,-1], (-1,1], (1,4], (4,8], (8,12], (12,inf)`.

## Targets to evaluate (`y`)

Required:
- `MIN`
- `FG3M`
- `FG3A`
- `FG3A_per_min`
- `FG3_PCT`

Recommended add-ons (if present in universe):
- `PTS`, `REB`, `AST`, `FGA`, `FGM`, `TOV`, `FTA`, `FTM`

## Experiment matrix

For each target `y`, run:

1. **Baseline**
   - `y ~ 1` (global mean or existing baseline from `baseline_player_season_mean` for context table).

2. **Continuous spread**
   - `y ~ spread_signed`
   - Record coefficient, p-value, R2, RMSE, MAE, and gain vs baseline.

3. **Binned spread**
   - `y ~ C(spread_bin)` (dummy-coded bins; neutral bin as reference).
   - Record overall model metrics and per-bin deltas vs neutral.

4. **Optional incremental check**
   - `y ~ market_consensus_line + spread_signed`
   - Quantify added value of spread beyond existing market context.

## Implementation plan (agent handoff)

1. **Create spread-enriched universe script**
   - File: `src/nba_three_point_modeling/00_research/scripts/v6_build_spread_universe.py`
   - Responsibilities:
     - Load v5 universe parquet.
     - Load historical game lines and derive one pregame spread per game/team/date.
     - Map player row to team-side spread and compute `spread_signed`, `spread_abs`, `spread_bin`.
     - Write:
       - `~/Downloads/tmp/v6_spread_universe.parquet`
       - `~/Downloads/tmp/v6_spread_universe_qc.csv`
   - QC checks:
     - `% rows with non-null spread_signed`
     - duplicate key count (`player_normalized`, `date`, `game_id`)
     - bin coverage counts

2. **Create spread regression sweep script**
   - File: `src/nba_three_point_modeling/00_research/scripts/v6_spread_target_sweep.py`
   - Responsibilities:
     - Input: `v6_spread_universe.parquet`
     - Loop all target `y` columns.
     - Fit baseline, continuous-spread, and binned-spread specs.
     - Save outputs:
       - `~/Downloads/tmp/v6_spread_model_summary.csv`
       - `~/Downloads/tmp/v6_spread_bin_effects.csv`
       - `~/Downloads/tmp/v6_spread_ranked_targets.csv`
   - Ranking rule:
     - Primary: `rmse_gain_vs_baseline` (descending)
     - Secondary: `r2_gain_vs_baseline` (descending)

3. **Create orchestration wrapper**
   - File: `src/nba_three_point_modeling/00_research/scripts/v6_run_spread_workflow.py`
   - Phase flow:
     - `phase0`: build spread universe
     - `phase1`: run target sweeps
     - `phase2`: print compact review table

4. **Create review script (DuckDB-first)**
   - File: `src/nba_three_point_modeling/00_research/scripts/v6_review_spread_outputs_duckdb.py`
   - Responsibilities:
     - Show top targets by spread utility.
     - Show strongest positive/negative spread bins per target.
     - Flag weak-signal targets where spread adds no practical lift.

## Output contract

`v6_spread_model_summary.csv` columns:
- `target`
- `model` (`baseline`, `spread_signed`, `spread_bin`)
- `n_rows`
- `rmse`
- `mae`
- `r2`
- `rmse_gain_vs_baseline`
- `mae_gain_vs_baseline`
- `r2_gain_vs_baseline`
- `coef_spread_signed` (null for non-continuous model)
- `coef_spread_signed_pvalue` (if computed)

`v6_spread_bin_effects.csv` columns:
- `target`
- `spread_bin`
- `bin_n`
- `bin_mean_y`
- `delta_vs_neutral`
- `model_implied_delta` (if available)

`v6_spread_ranked_targets.csv` columns:
- `target`
- `best_spread_model`
- `rmse_gain_vs_baseline`
- `r2_gain_vs_baseline`
- `signal_tier` (`high`, `medium`, `low`)

## CLI examples for the executing agent

```bash
python src/nba_three_point_modeling/00_research/scripts/v6_build_spread_universe.py \
  --input-universe ~/Downloads/tmp/v5_eval_universe.parquet \
  --season "*" \
  --output-universe ~/Downloads/tmp/v6_spread_universe.parquet \
  --output-qc ~/Downloads/tmp/v6_spread_universe_qc.csv
```

```bash
python src/nba_three_point_modeling/00_research/scripts/v6_spread_target_sweep.py \
  --input-universe ~/Downloads/tmp/v6_spread_universe.parquet \
  --targets "MIN,FG3M,FG3A,FG3A_per_min,FG3_PCT,PTS,REB,AST,FGA,FGM,TOV,FTA,FTM" \
  --output-summary ~/Downloads/tmp/v6_spread_model_summary.csv \
  --output-bin-effects ~/Downloads/tmp/v6_spread_bin_effects.csv \
  --output-ranked ~/Downloads/tmp/v6_spread_ranked_targets.csv
```

```bash
python src/nba_three_point_modeling/00_research/scripts/v6_review_spread_outputs_duckdb.py \
  --tmp-dir ~/Downloads/tmp --top-n 10
```

## Decision criteria

Treat spread as meaningful for a target when all are true:
- `rmse_gain_vs_baseline > 0.01` (or stronger threshold after first pass),
- `r2_gain_vs_baseline > 0`,
- effect is directionally stable across neighboring spread bins,
- no severe sample sparsity in extreme bins.

If spread signal is weak for all targets, deprioritize spread-only models and test interaction terms with rest/context features in a follow-up.

## Risks and guardrails

- Ensure spread is truly pregame; do not leak in-game updates.
- Keep one deterministic spread snapshot per game/team/date.
- Preserve fail-fast behavior: required columns must exist; do not silently fallback.
- Do not fabricate missing spread data.

## Acceptance criteria

Plan is complete when:
- A new agent can run the three CLI commands without guessing inputs.
- Output files are generated with the exact contracts above.
- Ranked table clearly answers: "which `y` targets benefit most from spread context?"
- Review script prints a concise recommendation on whether spread should be promoted into main modeling features.
