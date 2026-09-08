UMP GRID SEARCH PLAN — MLB Pitcher Strikeouts
=============================================
Script: 20260821_grid_ump.py
Goal: use existing model predictions as-is; test home plate ump identity
      as a FILTER on betting decisions (not a model feature)

WHY HOME PLATE UMP ONLY
-----------------------
Home plate ump calls balls and strikes — directly controls strikeout rate.
1st/2nd/3rd base umps have zero effect on Ks. All data we fetched via
MLB Stats API (hydrate=officials, officialType="Home Plate") is home plate
only. This is correct and intentional.

WHAT WE ALREADY HAVE
---------------------
- ump_features.parquet on S3: game_pk, ump_id, ump_name, ump_k_delta (+ variants)
  100% coverage across 6,364 games (2024–2026)
- step4_edge.parquet: existing model predictions + market odds per bet
- step3_oof_residuals.npy: for bootstrap P(over) computation
- ~100 unique home plate umps in dataset

APPROACH: 3 LAYERS OF COMBINATORICS

Layer 1 — Per-ump sweep (individual umps as filters)
------------------------------------------------------
For each of the ~100 umps with ≥20 qualifying bets:
  - Filter grid search to only games where that ump worked
  - Use EXISTING grid dims: shrinkage × min_edge × direction
    (skip odds/line bucket to keep n_bets manageable per ump)
  - Record: n_bets, win_rate, units_won, ROI, max_drawdown per ump
Output: step5_ump_leaderboard.csv sorted by units_won

This answers: which individual umps does the model perform best/worst in?

Layer 2 — Ump tier as grid dimension
--------------------------------------
Add ump_k_delta quantile buckets as a new grid axis alongside existing dims:
  UMP_TIERS = [
    "all",                          # baseline (no filter)
    "hp_k_friendly_top25",          # ump_k_delta >= 75th pct
    "hp_k_friendly_top10",          # ump_k_delta >= 90th pct
    "hp_batter_friendly_bot25",     # ump_k_delta <= 25th pct
    "hp_batter_friendly_bot10",     # ump_k_delta <= 10th pct
  ]
Full grid: shrinkage × min_edge × direction × odds_bucket × line_bucket × ump_tier
Output: step5_grid_ump.csv (same format as step5_grid_oos.csv + ump_tier column)

Key hypothesis to check:
  - K-friendly ump + OVER edge → better hit rate?
  - Batter-friendly ump + UNDER edge → better hit rate?
  These would be directional interactions, not just a blanket ump filter.

Layer 3 — True combinatorics on candidate umps
------------------------------------------------
After Layer 1, take umps that individually show ROI > 5% with ≥20 bets
(call this set C, expect |C| ≤ 15).
Enumerate all 2^|C| subsets of C (≤ 32,768 combos at |C|=15).
For each subset, run the best-performing grid config from Layer 2 and record:
  n_bets, units_won, ROI, max_drawdown
Filter to subsets with n_bets ≥ 30.
Output: step5_ump_combos.csv sorted by units_won

This answers: is there a specific club of umps where the model is sharper?

DATA JOIN PLAN
--------------
step4_edge.parquet has player_key + game_date but NOT game_pk or ump info.
Bridge:
  1. Load spine from S3 → get (player_key, game_date, game_pk) mapping
  2. Join game_pk onto edge df
  3. Join ump_features on game_pk → gets ump_name, ump_k_delta

Percentile thresholds for Layer 2 computed from ump_k_delta distribution
across ALL games in the edge dataset (not training data) — no lookahead
needed here since ump_k_delta itself is already lag-1 (prior games only).

OUTPUT FILES
------------
~/Downloads/tmp/mlb_strikeouts/
  step5_ump_leaderboard.csv   — per-ump ROI table (Layer 1)
  step5_grid_ump.csv          — full grid with ump_tier dim (Layer 2)
  step5_ump_combos.csv        — subset combos of candidate umps (Layer 3)

CAVEATS
-------
- Sample size: ~3 seasons, ~100 umps, ~6K games. Per-ump n is thin.
  Treat any individual ump finding as exploratory, not deployable.
- Layer 3 risks massive multiple comparison overfitting — treat as
  hypothesis generation only. Need 2027 season data to validate.
- The model was NOT retrained with ump features (that showed no lift).
  We are purely testing ump as a post-hoc filter on existing predictions.
