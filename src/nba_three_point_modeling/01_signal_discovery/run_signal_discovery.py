"""Run v1 signal discovery and emit predictions_df contract."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

MODULE_DIR = Path(__file__).resolve().parent
ROOT_DIR = MODULE_DIR.parent
UTILS_DIR = ROOT_DIR / "99_utils"
MODELS_DIR = MODULE_DIR / "models"
for extra_path in [str(UTILS_DIR), str(MODELS_DIR)]:
    if extra_path not in sys.path:
        sys.path.insert(0, extra_path)

from data_loading import build_v1_data_bundle
from baseline import fit_baseline_model
from v2_three_input_regression import build_v2_feature_frame
from v2_three_input_regression import fit_v2_three_input_model


def build_predictions_df(
    run_id: str,
    season: str,
    player_name: str,
    mean_model_id: str = "baseline_ols_season_avg_3pm",
) -> pd.DataFrame:
    """Build predictions_df with required 01->02 interface columns."""
    bundle = build_v1_data_bundle(season=season, player_name=player_name)
    games = bundle.player_games_df.copy().sort_values("date")

    split_idx = max(5, int(len(games) * 0.7))
    train_df = games.iloc[:split_idx].copy()
    test_df = games.iloc[split_idx:].copy()
    if test_df.empty:
        test_df = games.copy()

    if mean_model_id == "baseline_ols_season_avg_3pm":
        model = fit_baseline_model(train_df)
        test_features = test_df
    elif mean_model_id == "v2_three_input_regression":
        train_features = build_v2_feature_frame(train_df)
        model = fit_v2_three_input_model(train_features)
        test_features = build_v2_feature_frame(test_df)
    else:
        raise ValueError(f"Unsupported mean_model_id: {mean_model_id}")
    test_df["y_hat"] = model.predict(test_features)
    return test_df.assign(
        run_id=run_id,
        model_id=model.model_id,
        model_version=model.model_version,
        feature_version=model.feature_version,
    )[
        [
            "run_id",
            "game_id",
            "player_id",
            "date",
            "y_hat",
            "model_id",
            "model_version",
            "feature_version",
            "actual_fg3m",
        ]
    ]

