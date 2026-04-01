"""
Option A (Normal CDF) scoring for rebounds props — single source for v3 backtest and prod.

Context (owner brief):
- mean_adj = consensus + (1 - shrink) * (yhat - consensus)
- sigma = max(roll_reb_std_N, sigma_floor); prod uses roll_reb_std_5 and floor 0.25
- z = (line - mean_adj) / sigma; P_under = Phi(z); P_over = 1 - P_under
- edge_under = P_under - p_under_novig; under_only play when edge_under > min_edge (strict, matches v3 pick_side)

Not Monte Carlo: closed-form Normal. Option B (empirical player bootstrap) lives elsewhere if added later.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

# Locked prod defaults (see docs/design-docs/nba-rebounds-daily-pipeline.md)
SIGMA_FLOOR_DEFAULT = 0.25
PROD_SHRINK = 0.0
PROD_MIN_EDGE = 0.05
PROD_SIGMA_COL = "roll_reb_std_5"


def option_a_vector_batch(
    consensus: np.ndarray,
    yhat: np.ndarray,
    line: np.ndarray,
    sigma_raw: np.ndarray,
    shrink: float,
    p_nov_o: np.ndarray,
    p_nov_u: np.ndarray,
    *,
    sigma_floor: float = SIGMA_FLOOR_DEFAULT,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Vectorized Option A for many prop rows (same shrink for all rows).

    Returns:
        mean_adj, z, p_over, p_under, edge_over, edge_under
    """
    c = consensus.astype(np.float64, copy=False)
    y = yhat.astype(np.float64, copy=False)
    ell = line.astype(np.float64, copy=False)
    sig = np.maximum(sigma_raw.astype(np.float64, copy=False), float(sigma_floor))
    mean_adj = c + (1.0 - float(shrink)) * (y - c)
    z = (ell - mean_adj) / sig
    p_under = norm.cdf(z)
    p_over = 1.0 - p_under
    po = p_nov_o.astype(np.float64, copy=False)
    pu = p_nov_u.astype(np.float64, copy=False)
    edge_over = p_over - po
    edge_under = p_under - pu
    return mean_adj, z, p_over, p_under, edge_over, edge_under


def pick_side(
    i: int,
    edge_o: np.ndarray,
    edge_u: np.ndarray,
    min_edge: float,
    side_policy: str,
) -> str | None:
    """Same semantics as historical v3 loop (strict edge > min_edge)."""
    eo = edge_o[i]
    eu = edge_u[i]
    if side_policy == "over_only":
        return "over" if eo > min_edge else None
    if side_policy == "under_only":
        return "under" if eu > min_edge else None
    if eo > min_edge and eu > min_edge:
        return "over" if eo >= eu else "under"
    if eo > min_edge:
        return "over"
    if eu > min_edge:
        return "under"
    return None


def play_under_only_mask(edge_under: np.ndarray, min_edge: float) -> np.ndarray:
    """Boolean mask: would place under bet (strict > min_edge, same as v3 pick_side under_only)."""
    return edge_under > float(min_edge)
