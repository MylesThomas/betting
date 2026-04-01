"""
V3 mean model: v2 feature set + market/spread context for FG3M prediction.

Context:
- Promotes spread context found in v6 research into production-style signal
  discovery using canonical feature names:
  - `player_consensus_prop_line`
  - `team_point_spread`
- This model remains deterministic OLS and emits a single expected value `y_hat`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from v2_three_input_regression import build_v2_feature_frame


V3_FEATURE_COLUMNS = [
    "mean_3pm",
    "predicted_3pa",
    "predicted_minutes",
    "player_consensus_prop_line",
    "team_point_spread",
]


def build_v3_feature_frame(games_with_context_df: pd.DataFrame) -> pd.DataFrame:
    """Build deterministic v3 feature frame from games + context columns."""
    features_df = build_v2_feature_frame(games_with_context_df)
    required_cols = ["player_consensus_prop_line", "team_point_spread"]
    for col in required_cols:
        if col not in features_df.columns:
            raise ValueError(f"Missing required v3 context feature: {col}")
    return features_df


@dataclass
class V3MarketSpreadRegressionModel:
    """OLS model with v2 base features plus market/spread context terms."""

    intercept: float
    beta_mean_3pm: float
    beta_predicted_3pa: float
    beta_predicted_minutes: float
    beta_player_consensus_prop_line: float
    beta_team_point_spread: float
    model_id: str = "v3_market_spread_regression"
    model_version: str = "v3"
    feature_version: str = (
        "mean_3pm_predicted_3pa_predicted_minutes_"
        "player_consensus_prop_line_team_point_spread"
    )

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Predict expected FG3M from v3 feature set."""
        x = features[V3_FEATURE_COLUMNS].to_numpy(dtype=float)
        beta = np.array(
            [
                self.beta_mean_3pm,
                self.beta_predicted_3pa,
                self.beta_predicted_minutes,
                self.beta_player_consensus_prop_line,
                self.beta_team_point_spread,
            ],
            dtype=float,
        )
        y_hat = self.intercept + x @ beta
        return np.clip(y_hat, 0.0, None)


def fit_v3_market_spread_model(train_df: pd.DataFrame) -> V3MarketSpreadRegressionModel:
    """Fit OLS coefficients for v3 market+spread FG3M model."""
    x = train_df[V3_FEATURE_COLUMNS].to_numpy(dtype=float)
    y = train_df["actual_fg3m"].to_numpy(dtype=float)
    X = np.column_stack([np.ones(len(train_df)), x])
    coefs = np.linalg.lstsq(X, y, rcond=None)[0]
    return V3MarketSpreadRegressionModel(
        intercept=float(coefs[0]),
        beta_mean_3pm=float(coefs[1]),
        beta_predicted_3pa=float(coefs[2]),
        beta_predicted_minutes=float(coefs[3]),
        beta_player_consensus_prop_line=float(coefs[4]),
        beta_team_point_spread=float(coefs[5]),
    )
