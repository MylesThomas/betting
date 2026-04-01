"""Pricing pipeline: y_hat -> Monte Carlo probabilities and edges."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

MODULE_DIR = Path(__file__).resolve().parent
UTILS_DIR = MODULE_DIR.parent / "99_utils"
UNCERTAINTY_DIR = MODULE_DIR / "uncertainty_models"
for extra_path in [str(UTILS_DIR), str(UNCERTAINTY_DIR)]:
    if extra_path not in sys.path:
        sys.path.insert(0, extra_path)

from odds import american_to_implied_prob
from odds import implied_prob_to_american
from odds import remove_vig_two_way


def price_lines_with_monte_carlo(
    predictions_df: pd.DataFrame,
    lines_df: pd.DataFrame,
    uncertainty_model: Any,
    n_sims: int = 1000,
    random_seed: int = 7,
) -> pd.DataFrame:
    """
    Create priced line contracts with model probabilities and edge metrics.

    Edge policy:
    - `edge_*` and `edge_*_raw` are computed against raw implied probabilities
      from market odds (this is what betting decisions should use).
    - `edge_*_novig` is kept as a diagnostic view only.
    """
    merged = predictions_df.merge(lines_df, on=["date"], how="inner")
    if merged.empty:
        raise ValueError("No join between predictions and lines for pricing")
    required_prediction_context_cols = [
        "team_point_spread",
        "player_consensus_prop_line",
        "team_point_spread_abs",
        "team_point_spread_bucket",
        "spread_context_active_fg3m",
    ]
    for col in required_prediction_context_cols:
        if col not in merged.columns:
            raise ValueError(f"Missing required prediction context column for pricing: {col}")

    rng = np.random.default_rng(random_seed)
    rows = []
    for row in merged.itertuples(index=False):
        y_hat = float(row.y_hat)
        sims = uncertainty_model.simulate_fg3m(y_hat=y_hat, n_sims=n_sims, rng=rng)
        line = float(row.line)
        p_over = float(np.mean(sims > line))
        p_under = float(np.mean(sims < line))

        p_implied_over_raw = american_to_implied_prob(float(row.best_over_odds))
        p_implied_under_raw = american_to_implied_prob(float(row.best_under_odds))
        p_implied_over_novig, p_implied_under_novig = remove_vig_two_way(
            p_implied_over_raw, p_implied_under_raw
        )
        edge_over_raw = p_over - p_implied_over_raw
        edge_under_raw = p_under - p_implied_under_raw
        edge_over_novig = p_over - p_implied_over_novig
        edge_under_novig = p_under - p_implied_under_novig

        rows.append(
            {
                "run_id": row.run_id,
                "game_id": row.game_id,
                "player_id": row.player_id,
                "date": row.date,
                "line": line,
                "p_over": p_over,
                "p_under": p_under,
                "fair_odds_over": float(implied_prob_to_american(max(min(p_over, 0.999), 0.001))),
                "fair_odds_under": float(implied_prob_to_american(max(min(p_under, 0.999), 0.001))),
                "p_implied_over_raw": p_implied_over_raw,
                "p_implied_under_raw": p_implied_under_raw,
                "p_implied_over_novig": p_implied_over_novig,
                "p_implied_under_novig": p_implied_under_novig,
                "edge_over": edge_over_raw,
                "edge_under": edge_under_raw,
                "edge_over_raw": edge_over_raw,
                "edge_under_raw": edge_under_raw,
                "edge_over_novig": edge_over_novig,
                "edge_under_novig": edge_under_novig,
                "uncertainty_model_id": uncertainty_model.model_id,
                "n_sims": int(n_sims),
                "best_over_odds": float(row.best_over_odds),
                "best_under_odds": float(row.best_under_odds),
                "median_over_odds": float(row.median_over_odds),
                "median_under_odds": float(row.median_under_odds),
                "best_over_book": row.best_over_book,
                "best_under_book": row.best_under_book,
                "is_consensus": int(row.is_consensus),
                "actual_fg3m": float(row.actual_fg3m),
                "y_hat": y_hat,
                "team_point_spread": float(row.team_point_spread),
                "player_consensus_prop_line": float(row.player_consensus_prop_line),
                "team_point_spread_abs": float(row.team_point_spread_abs),
                "team_point_spread_bucket": row.team_point_spread_bucket,
                "spread_context_active_fg3m": int(row.spread_context_active_fg3m),
            }
        )
    return pd.DataFrame(rows)

