"""
V2 recency-weighted uncertainty model based on player history sampling.

Context:
- We sample minutes and 3PA tendencies from historical games, weighting recent
  games more heavily.
- The weighted sampling distribution is used to simulate FG3M around a provided
  mean prediction (`y_hat`) while preserving recency-driven variance shape.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


def build_recency_weights(
    n_rows: int,
    weighting_mode: str,
    decay_alpha: float,
) -> np.ndarray:
    """Build a normalized recency weight vector (oldest->newest order)."""
    if weighting_mode == "windowed_weighted":
        base = np.arange(1, n_rows + 1, dtype=float)
    elif weighting_mode == "exp_decay":
        age = np.arange(n_rows - 1, -1, -1, dtype=float)
        base = np.exp(-decay_alpha * age)
    else:
        raise ValueError(
            f"Unsupported weighting_mode '{weighting_mode}'. "
            "Use 'windowed_weighted' or 'exp_decay'."
        )
    return base / base.sum()


@dataclass
class V2WeightedHistorySamplerModel:
    """Player-history uncertainty model with configurable recency weighting."""

    history_df: pd.DataFrame
    history_weights: np.ndarray
    weighting_mode: str
    history_n: int
    decay_alpha: float
    model_id: str = "v2_weighted_history_sampler"

    def simulate_fg3m(self, y_hat: float, n_sims: int, rng: np.random.Generator) -> np.ndarray:
        """Simulate FG3M using weighted samples of minutes and 3PA history."""
        idx = rng.choice(
            np.arange(len(self.history_df), dtype=int),
            size=n_sims,
            replace=True,
            p=self.history_weights,
        )
        minutes_samples = self.history_df["actual_min"].to_numpy(dtype=float)[idx]
        attempts_per_min_samples = self.history_df["attempts_per_min"].to_numpy(dtype=float)[idx]
        fg3_pct_samples = self.history_df["fg3_pct"].to_numpy(dtype=float)[idx]

        sim_attempts = np.clip(minutes_samples * attempts_per_min_samples, 0.0, None)
        raw_fg3m = np.clip(sim_attempts * fg3_pct_samples, 0.0, None)
        raw_mean = float(raw_fg3m.mean())
        if raw_mean == 0.0:
            return np.full(n_sims, fill_value=max(y_hat, 0.0), dtype=float)

        scale = max(y_hat, 0.0) / raw_mean
        sims = np.clip(raw_fg3m * scale, 0.0, None)
        return sims


def fit_v2_weighted_history_sampler(
    history_df: pd.DataFrame,
    history_n: int,
    weighting_mode: str,
    decay_alpha: float,
) -> V2WeightedHistorySamplerModel:
    """Fit v2 sampler inputs from player history table."""
    history = history_df.sort_values("date").copy()
    history = history.tail(int(history_n)).reset_index(drop=True)
    history["attempts_per_min"] = (
        history["actual_fg3a"].astype(float) / history["actual_min"].astype(float)
    ).replace([np.inf, -np.inf], 0.0)
    history["attempts_per_min"] = history["attempts_per_min"].fillna(0.0)
    history["fg3_pct"] = (
        history["actual_fg3m"].astype(float) / history["actual_fg3a"].astype(float)
    ).replace([np.inf, -np.inf], 0.0)
    history["fg3_pct"] = history["fg3_pct"].fillna(0.0).clip(0.0, 1.0)

    weights = build_recency_weights(
        n_rows=len(history),
        weighting_mode=weighting_mode,
        decay_alpha=decay_alpha,
    )
    return V2WeightedHistorySamplerModel(
        history_df=history,
        history_weights=weights,
        weighting_mode=weighting_mode,
        history_n=int(history_n),
        decay_alpha=float(decay_alpha),
    )
