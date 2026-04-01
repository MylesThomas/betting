"""
Champion rebound regression feature list (B_min_max / v3–v5 stack).

Context: aligns OLS + XGB training, v3 edge backtest, and prod score/train scripts.
If you change columns, update docs/design-docs/nba-rebounds-daily-pipeline.md and re-run v3 tests.
"""

from __future__ import annotations

TARGET = "REB"

# Same 6 columns as v3_run_rebounds_edge_backtest.py / v5_compare_rebounds_models_oos.py
B_MIN_MAX_FEATS = [
    "min_line",
    "max_line",
    "spread_signed",
    "roll_reb_mean_60",
    "roll_fg3a_mean_20",
    "roll_reb_std_5",
]

GROUP_KEYS = ["season", "date", "player_normalized", "game_id"]

# v3_rebounds_props_raw.parquet columns needed for scoring joins + odds
V3_PROPS_SCORE_COLS = [
    "season",
    "date",
    "player_normalized",
    "game_id",
    "bookmaker",
    "line",
    "over_odds",
    "under_odds",
    "consensus_reb_line",
]
