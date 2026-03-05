"""DataFrame schema contracts for modular handoffs."""

PREDICTIONS_COLUMNS = [
    "run_id",
    "game_id",
    "player_id",
    "date",
    "y_hat",
    "model_id",
    "model_version",
    "feature_version",
]

LINES_COLUMNS = [
    "game_id",
    "player_id",
    "date",
    "sportsbook",
    "market",
    "line",
    "odds_over",
    "odds_under",
    "snapshot_ts",
    "is_consensus",
]

PRICED_LINES_COLUMNS = [
    "run_id",
    "game_id",
    "player_id",
    "date",
    "line",
    "p_over",
    "p_under",
    "fair_odds_over",
    "fair_odds_under",
    "edge_over",
    "edge_under",
    "uncertainty_model_id",
    "n_sims",
]

BETS_COLUMNS = [
    "run_id",
    "game_id",
    "player_id",
    "date",
    "line",
    "side",
    "odds",
    "stake",
    "p_model",
    "edge",
    "result",
    "pnl",
]

