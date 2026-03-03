# NBA multimarket strategy analysis

**Created:** 2026-02-25  
**Status:** In progress

## Goal

Use the unified NBA strategy parquet (game results + props + game lines + actuals, one row per player-game) to:

1. **Top-down:** See which variables matter most for each of the 9 prop markets.
2. **Derive backtestable signals** (e.g. “unders on high lines when favorite”) and validate on historical data.

## Context

- **Dataset:** Built by `scripts/build_nba_multimarket_strategy_dataset.py`. Output: single parquet (e.g. `~/Downloads/tmp/nba_prop_strategies.parquet`) with matched player-games only (null GAME_ID rows dropped).
- **Existing backtest:** `backtesting/20260108_nba_points_props_strategy_backtest.py` is **points-only** and uses tier × spread (and 3D with scorer type) from a different pipeline. The new parquet is **multi-market** and has game lines but no tier/scorer_type yet.
- **No analysis script currently reads this parquet.** Next step is to add analysis that consumes it.

## Dataset schema (parquet)

- **Identity:** `player`, `player_normalized`, `game_date`, `season`, `team_full`, `GAME_ID`, `home_team`, `away_team`
- **Game context:** `HOME_SCORE`, `AWAY_SCORE`, `away_spread`, `home_spread`, `away_spread_odds`, `home_spread_odds`, `away_moneyline`, `home_moneyline`
- **Per market (9):** `market_median_value_{market}` (median line across books), `actual_*` (from game logs, e.g. `actual_points`, `actual_rebounds`). Markets: player_assists, player_blocks, player_double_double, player_points, player_points_rebounds_assists, player_rebounds, player_steals, player_threes, player_triple_double.
- **Note:** Over/under odds are not in the wide output; backtest can assume -110 or add median over/under in a later build step.

## Iteration: by market only (n=9)

For the XGBoost + regression-tree variable-importance step we iterate over **one dimension only:**

- **By market:** 9 markets. For each market we train one XGBoost and one regression tree to predict that market’s actual; same feature set across all markets; output feature importance (and optionally SHAP) per market.

We do **not** iterate over season, player tier, or other groups for this step—one model pair per market, trained on all available rows (all seasons in the parquet) for that market. Segment discovery and backtest can later slice by season or other dimensions.

Markets (order from build script): `player_assists`, `player_blocks`, `player_double_double`, `player_points`, `player_points_rebounds_assists`, `player_rebounds`, `player_steals`, `player_threes`, `player_triple_double`.

## Approach

1. **Describe & validate** – Load parquet, confirm row counts and coverage per market/season. Simple hit-rate vs line (over/under) by market.
2. **Variable importance per market (ML)** – For each market, train **XGBoost** and **regression trees** (e.g. sklearn `DecisionTreeRegressor` or `ExtraTreesRegressor`) to predict the **actual** stat (e.g. `actual_pts` for player_points). Use feature importance (and optionally SHAP) to see what correlates with a higher actual.
   - **Target:** The actual outcome for that market (e.g. `actual_pts`, `actual_reb`, or the computed `actual_*` for PRA/double-double/triple-double).
   - **Features:** Median line for that market (`market_median_value_{market}`), game spread from player’s team perspective (favorite/underdog or raw spread), home/away (is `team_full == home_team`), other game context (e.g. total implied from moneylines, or keep it minimal: line + spread + home).
   - **Output:** Feature importance rankings per market; compare across markets to see which variables matter where. Regression trees also give interpretable splits (e.g. when line > 25 and spread < -3, mean actual = X).
3. **Segment discovery** – Use ML importance + simple segmentation (spread bins, line tiers, home/away) to identify (market, segment) combos with better hit rate or error profile; document as candidate strategies.
4. **Signal definition & backtest** – Define concrete rules from the above, compute historical plays and P&amp;L (e.g. at -110), report ROI and sample size. Optionally add over/under odds from build later for realistic backtest.

## Tasks

- [ ] Add analysis script that loads the strategy parquet (path as arg or default `~/Downloads/tmp/nba_prop_strategies.parquet`).
- [ ] Per-market summary: hit rate (over/under vs median line), median error, count; optionally by season.
- [ ] **Per-market XGBoost + regression trees:** For each market with a numeric actual, train models to predict that actual; derive and report feature importance (and optionally SHAP). Use consistent feature set (line, team spread, home/away, etc.).
- [ ] Document which variables matter most per market (from ML importance + segmentation).
- [ ] Segment by spread and line tier; report hit rate and count per segment per market (complement to ML).
- [ ] Define 1–2 concrete signals from findings and implement backtest (plays + P&amp;L at -110).
- [ ] (Optional) Add median over/under odds to build script and use in backtest for realistic ROI.

## Where to put code

- **Analysis:** `analysis/` (e.g. `analyze_nba_multimarket_strategy.py` or `nba_multimarket_strategy_topdown.py`) for load, describe, variable importance, segment discovery.
- **Backtest:** Either extend `backtesting/` with a script that reads this parquet and runs signal rules, or a single script in `analysis/` that does both exploration and a simple backtest.

## Related

- Build script: `scripts/build_nba_multimarket_strategy_dataset.py`
- Existing points backtest: `backtesting/20260108_nba_points_props_strategy_backtest.py`
- Domain: `docs/domain/betting-fundamentals.md`, `docs/domain/market-mechanics.md`
