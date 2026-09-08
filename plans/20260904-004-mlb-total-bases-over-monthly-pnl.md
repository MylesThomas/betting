OVER MIRROR — MLB Total Bases OVER 1.5 (2026)
==============================================

PURPOSE
-------
Sanity check. Take every row that qualified for the UNDER strategy and compute
P&L as if the OVER side were taken instead. Expect negative ROI; positive ROI
would indicate a problem with the model's signal direction.

FILTER (identical to live UNDER strategy)
-----------------------------------------
- market_key == "batter_total_bases"
- line == 1.5
- season == 2026
- edge_under >= 5%   (same qualifying rows as live strategy, bet flipped)
- over_price > 1, under_price > 1

WIN CONDITION / PAYOUT
----------------------
Win:  actual_tb >= 2  →  payout = over_price - 1
Loss: actual_tb < 2   →  payout = -1
Missing actuals (no Statcast data) → excluded from P&L

COLUMN DEFINITIONS
------------------
under%   = actual UNDER hit rate (fraction where actual_tb < 2)
over%    = actual OVER hit rate  (fraction where actual_tb >= 2)
*_net    = net units
*_roi    = ROI %

S3 INPUTS (us-east-2, bucket: the-odds-api-mt)
-----------------------------------------------
  mlb/total_bases_model/spine/mlb_total_bases_spine.parquet
  mlb/total_bases_model/market_raw/mlb_total_bases_market_raw.parquet
  mlb/total_bases_model/model/mlb_tb_regression_v2.joblib

SCORING STEPS
-------------
1. Load spine + model from S3
2. Score spine with model to get p_model per player-game
3. Join per-bookmaker market_raw on (name_norm, game_date, line)
4. Apply filter above
5. Fetch actuals via pybaseball.statcast() per unique game_date in qualifying rows
6. Compute OVER P&L per row; exclude no_data rows
7. Print output tables

OUTPUT TABLES (stdout)
----------------------
All tables share columns: n | under% | under_net | under_roi | over% | over_net | over_roi

1. MONTHLY — grouped by month, + TOTAL row
2. EDGE BUCKET — [5-10%), [10-15%), [15%+]
3. BY BOOKMAKER — one row per bookmaker, totals only
4. BY TIER — "play" (edge >= min_bet_edge) vs "track" (below play threshold), totals only

Name normalization required for spine join — see normalize_name() and
MANUAL_MAP in src/mlb_total_bases_modeling/scripts/settle_total_bases.py

OUTPUT SCRIPT
-------------
src/mlb_total_bases_modeling/scripts/20260904_over_mirror_pnl.py
(kept for reference; not hardened for automated re-runs)
