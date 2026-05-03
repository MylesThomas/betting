"""
Canonical rebounds feature list and column groups for prod training and scoring.

Context: aligns OLS + XGB training, edge backtest, and prod score/train scripts.
If you change columns, update docs/design-docs/nba-rebounds-daily-pipeline.md and re-run tests.
"""

from __future__ import annotations

TARGET = "REB"

B_MIN_MAX_FEATS = [
    "min_line",
    "max_line",
    "spread_signed",
    "roll_reb_mean_60",
    "roll_fg3a_mean_20",
    "roll_reb_std_5",
]

# Audit trail: raw lists behind B_MIN_MAX_FEATS (min/max share one lines list).
B_MIN_MAX_AUDIT_LIST_COLS = [
    "input_reb_prop_lines",
    "input_spread_by_side",
    "input_reb_tail_60",
    "input_fg3a_tail_20",
    "input_reb_tail_5",
]

GROUP_KEYS = ["season", "date", "player_normalized", "game_id"]

# Carried on each feature-universe row (not in B_MIN_MAX; for spread audit + S3 verify without a second file).
TEAM_CONTEXT_COLS = ["team_normalized", "home_team_norm", "away_team_norm"]
PROPS_SCORE_COLS = [
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
