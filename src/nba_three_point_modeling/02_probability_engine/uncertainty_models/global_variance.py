"""
Global variance uncertainty model for v1.

Context:
- V1 uses one fixed residual variance for all players/lines.
- This module estimates sigma from model residuals and samples game outcomes.
"""

from __future__ import annotations

import numpy as np


class GlobalVarianceModel:
    """Normal residual model with one global sigma."""

    def __init__(self, sigma: float) -> None:
        self.model_id = "global_variance"
        self.sigma = float(sigma)

    def simulate_fg3m(self, y_hat: float, n_sims: int, rng: np.random.Generator) -> np.ndarray:
        """Draw simulated FG3M outcomes around y_hat and clamp to non-negative."""
        sims = rng.normal(loc=y_hat, scale=self.sigma, size=n_sims)
        return np.clip(sims, 0.0, None)


def fit_global_variance(residuals: np.ndarray) -> GlobalVarianceModel:
    """Fit global sigma from residual distribution."""
    sigma = float(np.std(residuals, ddof=1))
    if sigma == 0.0:
        sigma = 0.25
    return GlobalVarianceModel(sigma=sigma)

