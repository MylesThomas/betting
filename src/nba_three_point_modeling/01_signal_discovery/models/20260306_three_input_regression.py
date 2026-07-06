"""
V2 mean model with three inputs for player_threes prediction.

Context:
- This model extends v1 by using three deterministic features to predict game
  `FG3M`:
  1) mean_3pm (season-to-date lagged mean),
  2) predicted_3pa (season-to-date lagged mean attempts),
  3) predicted_minutes (season-to-date lagged mean minutes).
- Output remains a single expected value `y_hat`.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


V2_FEATURE_COLUMNS = ["mean_3pm", "predicted_3pa", "predicted_minutes"]


def build_v2_feature_frame(games_df: pd.DataFrame) -> pd.DataFrame:
    """Build lagged deterministic features required by v2 model."""
    features_df = games_df.copy().sort_values("date").reset_index(drop=True)
    mean_3pm = (
        features_df["actual_fg3m"]
        .astype(float)
        .expanding(min_periods=1)
        .mean()
        .shift(1)
    )
    predicted_3pa = (
        features_df["actual_fg3a"]
        .astype(float)
        .expanding(min_periods=1)
        .mean()
        .shift(1)
    )
    predicted_minutes = (
        features_df["actual_min"]
        .astype(float)
        .expanding(min_periods=1)
        .mean()
        .shift(1)
    )

    features_df["mean_3pm"] = mean_3pm.fillna(
        float(features_df["actual_fg3m"].astype(float).mean())
    )
    features_df["predicted_3pa"] = predicted_3pa.fillna(
        float(features_df["actual_fg3a"].astype(float).mean())
    )
    features_df["predicted_minutes"] = predicted_minutes.fillna(
        float(features_df["actual_min"].astype(float).mean())
    )
    return features_df


@dataclass
class V2ThreeInputRegressionModel:
    """OLS model for y = b0 + b1*mean_3pm + b2*predicted_3pa + b3*predicted_minutes."""

    intercept: float
    beta_mean_3pm: float
    beta_predicted_3pa: float
    beta_predicted_minutes: float
    model_id: str = "v2_three_input_regression"
    model_version: str = "v2"
    feature_version: str = "mean_3pm_predicted_3pa_predicted_minutes"

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Predict expected FG3M from the v2 feature set."""
        x = features[V2_FEATURE_COLUMNS].to_numpy(dtype=float)
        beta = np.array(
            [
                self.beta_mean_3pm,
                self.beta_predicted_3pa,
                self.beta_predicted_minutes,
            ],
            dtype=float,
        )
        y_hat = self.intercept + x @ beta
        return np.clip(y_hat, 0.0, None)


def fit_v2_three_input_model(train_df: pd.DataFrame) -> V2ThreeInputRegressionModel:
    """Fit OLS coefficients for the v2 three-feature model."""
    x = train_df[V2_FEATURE_COLUMNS].to_numpy(dtype=float)
    y = train_df["actual_fg3m"].to_numpy(dtype=float)
    X = np.column_stack([np.ones(len(train_df)), x])
    coefs = np.linalg.lstsq(X, y, rcond=None)[0]
    return V2ThreeInputRegressionModel(
        intercept=float(coefs[0]),
        beta_mean_3pm=float(coefs[1]),
        beta_predicted_3pa=float(coefs[2]),
        beta_predicted_minutes=float(coefs[3]),
    )
