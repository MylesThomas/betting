"""Run v1 signal discovery and emit predictions_df contract."""

from __future__ import annotations

import sys
import json
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
from data_loading import build_player_game_context_features
from baseline import fit_baseline_model
from v2_three_input_regression import build_v2_feature_frame
from v2_three_input_regression import fit_v2_three_input_model
from v3_market_spread_regression import build_v3_feature_frame
from v3_market_spread_regression import fit_v3_market_spread_model


PROMOTED_TARGETS_DEFAULT = {"FG3A", "FG3M", "MIN", "FGA", "PTS"}
NON_EXTREME_BINS = {"(-12,-8]", "(-8,-4]", "(-4,-1]", "(-1,1]", "(1,4]", "(4,8]", "(8,12]"}


def _is_directionally_stable(bin_target_df: pd.DataFrame) -> bool:
    """Assess directional stability in non-extreme bins by limiting sign flips."""
    if bin_target_df.empty:
        return False
    subset = bin_target_df[bin_target_df["spread_bin"].isin(NON_EXTREME_BINS)].copy()
    subset = subset.sort_values("spread_bin")
    if len(subset) < 4:
        return False
    diffs = subset["delta_vs_neutral"].diff().dropna().to_numpy(dtype=float)
    signs = [1 if x > 0 else (-1 if x < 0 else 0) for x in diffs]
    non_zero_signs = [x for x in signs if x != 0]
    if len(non_zero_signs) <= 1:
        return True
    sign_flips = 0
    for i in range(1, len(non_zero_signs)):
        if non_zero_signs[i] != non_zero_signs[i - 1]:
            sign_flips += 1
    return sign_flips <= 1


def build_target_feature_promotion(
    summary_df: pd.DataFrame,
    bin_effects_df: pd.DataFrame,
    min_non_extreme_bin_n: int = 300,
) -> tuple[pd.DataFrame, dict]:
    """Build target-level spread promotion decisions and active-feature manifest."""
    rows: list[dict] = []
    for target in sorted(summary_df["target"].unique().tolist()):
        target_rows = summary_df[summary_df["target"] == target].copy()
        best = (
            target_rows[target_rows["model"] != "baseline"]
            .sort_values(["rmse_gain_vs_baseline", "r2_gain_vs_baseline", "model"], ascending=[False, False, True])
            .iloc[0]
        )
        gate_lift = float(best["rmse_gain_vs_baseline"]) > 0.0 and float(best["r2_gain_vs_baseline"]) > 0.0
        target_bins = bin_effects_df[
            (bin_effects_df["target"] == target) & (bin_effects_df["model"] == "spread_binned")
        ].copy()
        non_extreme = target_bins[target_bins["spread_bin"].isin(NON_EXTREME_BINS)].copy()
        gate_sample = (not non_extreme.empty) and int(non_extreme["n_rows"].min()) >= int(min_non_extreme_bin_n)
        gate_stable = _is_directionally_stable(target_bins)
        promote = gate_lift and gate_sample and gate_stable and (target in PROMOTED_TARGETS_DEFAULT)
        rows.append(
            {
                "target": target,
                "selected_model": str(best["model"]),
                "rmse_gain_vs_baseline": float(best["rmse_gain_vs_baseline"]),
                "r2_gain_vs_baseline": float(best["r2_gain_vs_baseline"]),
                "gate_positive_lift": int(gate_lift),
                "gate_non_extreme_bin_sample": int(gate_sample),
                "gate_directional_stability": int(gate_stable),
                "promote_spread_context": int(promote),
            }
        )
    promotion_df = pd.DataFrame(rows).sort_values("target").reset_index(drop=True)
    active_features = {
        row["target"]: bool(row["promote_spread_context"])
        for _, row in promotion_df.iterrows()
    }
    manifest = {
        "feature_names": {
            "team_point_spread": "team_point_spread",
            "player_consensus_prop_line": "player_consensus_prop_line",
        },
        "active_targets": active_features,
    }
    return promotion_df, manifest


def build_predictions_df(
    run_id: str,
    season: str,
    player_name: str,
    mean_model_id: str = "baseline_ols_season_avg_3pm",
    v6_summary_csv: str = "",
    v6_bin_effects_csv: str = "",
    output_promotion_csv: str = "",
    output_manifest_json: str = "",
) -> pd.DataFrame:
    """Build predictions_df with required 01->02 interface columns."""
    bundle = build_v1_data_bundle(season=season, player_name=player_name)
    games = bundle.player_games_df.copy().sort_values("date")
    games_with_context = build_player_game_context_features(
        player_games_df=games,
        lines_df=bundle.lines_df,
    ).copy()

    split_idx = max(5, int(len(games) * 0.7))
    train_df = games.iloc[:split_idx].copy()
    test_df = games.iloc[split_idx:].copy()
    if test_df.empty:
        test_df = games.copy()

    resolved_model_id = mean_model_id
    if mean_model_id == "v3_three_input_regression":
        resolved_model_id = "v2_three_input_regression"
    if mean_model_id == "v4_market_spread_regression":
        resolved_model_id = "v3_market_spread_regression"

    if resolved_model_id == "baseline_ols_season_avg_3pm":
        model = fit_baseline_model(train_df)
        test_features = test_df
    elif resolved_model_id == "v2_three_input_regression":
        train_features = build_v2_feature_frame(train_df)
        model = fit_v2_three_input_model(train_features)
        test_features = build_v2_feature_frame(test_df)
    elif resolved_model_id == "v3_market_spread_regression":
        train_features = build_v3_feature_frame(games_with_context.iloc[:split_idx].copy())
        model = fit_v3_market_spread_model(train_features)
        score_context = games_with_context.iloc[split_idx:].copy()
        if score_context.empty:
            score_context = games_with_context.copy()
        test_features = build_v3_feature_frame(score_context)
        test_df = score_context
    else:
        raise ValueError(f"Unsupported mean_model_id: {mean_model_id}")
    test_df["y_hat"] = model.predict(test_features)
    if "team_point_spread" not in test_df.columns or "player_consensus_prop_line" not in test_df.columns:
        context_cols = games_with_context[
            ["game_id", "date", "team_point_spread", "player_consensus_prop_line"]
        ].drop_duplicates(subset=["game_id", "date"])
        test_df = test_df.merge(context_cols, on=["game_id", "date"], how="left")

    if v6_summary_csv.strip() != "" and v6_bin_effects_csv.strip() != "":
        summary_df = pd.read_csv(Path(v6_summary_csv).expanduser())
        bin_effects_df = pd.read_csv(Path(v6_bin_effects_csv).expanduser())
        promotion_df, manifest = build_target_feature_promotion(
            summary_df=summary_df,
            bin_effects_df=bin_effects_df,
            min_non_extreme_bin_n=300,
        )
        if output_promotion_csv.strip() != "":
            promotion_df.to_csv(Path(output_promotion_csv).expanduser(), index=False)
        if output_manifest_json.strip() != "":
            Path(output_manifest_json).expanduser().write_text(json.dumps(manifest, indent=2))

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
            "team_point_spread",
            "player_consensus_prop_line",
        ]
    ]

