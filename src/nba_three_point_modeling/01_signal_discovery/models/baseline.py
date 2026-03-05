"""
Baseline mean model for v1.

Context:
- This v1 model implements the agreed first signal discovery baseline:
  OLS with a single feature, `season_avg_3pm`, to predict game-level `FG3M`.
- Coefficients are fit from historical player-game rows in memory.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class BaselineSeasonAvgOLSModel:
    """Simple OLS model y = intercept + slope * season_avg_3pm."""

    intercept: float
    slope: float
    model_id: str = "baseline_ols_season_avg_3pm"
    model_version: str = "v1"
    feature_version: str = "season_avg_3pm_only"

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Predict expected FG3M using `season_avg_3pm`."""
        x = features["season_avg_3pm"].to_numpy(dtype=float)
        return self.intercept + self.slope * x


def fit_baseline_model(train_df: pd.DataFrame) -> BaselineSeasonAvgOLSModel:
    """Fit OLS with one feature (`season_avg_3pm`)."""
    x = train_df["season_avg_3pm"].to_numpy(dtype=float)
    y = train_df["actual_fg3m"].to_numpy(dtype=float)
    X = np.column_stack([np.ones(len(x)), x])
    intercept, slope = np.linalg.lstsq(X, y, rcond=None)[0]
    return BaselineSeasonAvgOLSModel(intercept=float(intercept), slope=float(slope))

