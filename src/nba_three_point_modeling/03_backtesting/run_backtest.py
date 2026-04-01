"""
Run one v1 player_threes backtest and write reproducible run artifacts.

Context:
- Implements the locked v1 flow:
  mean model (01) -> probability engine (02) -> strategy/backtest (03).
- Reads `current_config.yaml`, auto-generates run_id, snapshots config,
  and writes predictions, bets, manifest, and summary.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime
from datetime import timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

MODULE_DIR = Path(__file__).resolve().parent
ROOT_DIR = MODULE_DIR.parent
UTILS_DIR = ROOT_DIR / "99_utils"
MODELS_DIR = ROOT_DIR / "01_signal_discovery" / "models"
PROB_DIR = ROOT_DIR / "02_probability_engine"
UNCERTAINTY_DIR = PROB_DIR / "uncertainty_models"
for extra_path in [
    str(UTILS_DIR),
    str(MODELS_DIR),
    str(PROB_DIR),
    str(UNCERTAINTY_DIR),
]:
    if extra_path not in sys.path:
        sys.path.insert(0, extra_path)

from data_loading import build_v1_data_bundle
from data_loading import build_player_game_context_features
from data_loading import load_player_history_from_season_logs
from odds import target_profit_stake
from baseline import fit_baseline_model
from v2_three_input_regression import build_v2_feature_frame
from v2_three_input_regression import fit_v2_three_input_model
from v3_market_spread_regression import build_v3_feature_frame
from v3_market_spread_regression import fit_v3_market_spread_model
from global_variance import fit_global_variance
from v2_weighted_history_sampler import fit_v2_weighted_history_sampler
from pricing import price_lines_with_monte_carlo


def _git_sha(repo_root: Path) -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root)
            .decode("utf-8")
            .strip()
        )
    except Exception:
        return "unknown"


def _strategy_slug(config: dict) -> str:
    return f"{config['strategy_id']}_thr{config['edge_threshold']}"


def _run_id(config: dict) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base = (
        f"{timestamp}_{config['mean_model_id']}_"
        f"{config['uncertainty_model_id']}_{_strategy_slug(config)}"
    )
    if config["run_suffix"] != "":
        return f"{base}_{config['run_suffix']}"
    return base


def _select_side_and_odds(row: pd.Series, price_view: str) -> tuple[str, float, float]:
    if row["actual_bet"] == "over":
        side = "over"
        p_model = float(row["p_over"])
        odds = float(row[f"{price_view}_over_odds"])
    elif row["actual_bet"] == "under":
        side = "under"
        p_model = float(row["p_under"])
        odds = float(row[f"{price_view}_under_odds"])
    else:
        raise ValueError("actual_bet must be 'over' or 'under' for executed bets")
    return side, odds, p_model


def _compute_bet_result(actual_fg3m: float, line: float, side: str) -> str:
    if side == "over":
        if actual_fg3m > line:
            return "win"
        if actual_fg3m < line:
            return "loss"
        return "push"
    if actual_fg3m < line:
        return "win"
    if actual_fg3m > line:
        return "loss"
    return "push"


def _compute_pnl(result: str, odds: float, stake: float) -> float:
    if result == "push":
        return 0.0
    if result == "loss":
        return -stake
    if odds > 0:
        return stake * (odds / 100.0)
    return stake * (100.0 / (-odds))


def _fit_mean_model(
    mean_model_id: str,
    train_df: pd.DataFrame,
    score_df: pd.DataFrame,
    spread_context_active: bool,
):
    resolved_model_id = mean_model_id
    if mean_model_id == "v3_three_input_regression":
        resolved_model_id = "v2_three_input_regression"
    if mean_model_id == "v4_market_spread_regression":
        resolved_model_id = "v3_market_spread_regression"

    if resolved_model_id == "baseline_ols_season_avg_3pm":
        model = fit_baseline_model(train_df)
        score_features = score_df
    elif resolved_model_id == "v2_three_input_regression":
        train_features = build_v2_feature_frame(train_df)
        model = fit_v2_three_input_model(train_features)
        score_features = build_v2_feature_frame(score_df)
    elif resolved_model_id == "v3_market_spread_regression":
        if not spread_context_active:
            train_features = build_v2_feature_frame(train_df)
            model = fit_v2_three_input_model(train_features)
            score_features = build_v2_feature_frame(score_df)
        else:
            train_features = build_v3_feature_frame(train_df)
            model = fit_v3_market_spread_model(train_features)
            score_features = build_v3_feature_frame(score_df)
    else:
        raise ValueError(f"Unsupported mean_model_id: {mean_model_id}")
    return model, score_features


def _is_directionally_stable(target_bins: pd.DataFrame) -> bool:
    """Assess directional stability in non-extreme bins by limiting sign flips."""
    non_extreme = {"(-12,-8]", "(-8,-4]", "(-4,-1]", "(-1,1]", "(1,4]", "(4,8]", "(8,12]"}
    subset = target_bins[target_bins["spread_bin"].isin(non_extreme)].copy().sort_values("spread_bin")
    if len(subset) < 4:
        return False
    diffs = subset["delta_vs_neutral"].diff().dropna().to_numpy(dtype=float)
    signs = [1 if x > 0 else (-1 if x < 0 else 0) for x in diffs]
    non_zero = [x for x in signs if x != 0]
    if len(non_zero) <= 1:
        return True
    sign_flips = 0
    for i in range(1, len(non_zero)):
        if non_zero[i] != non_zero[i - 1]:
            sign_flips += 1
    return sign_flips <= 1


def _build_target_feature_promotion_manifest(config: dict) -> tuple[pd.DataFrame, dict]:
    """Build target spread-promotion decisions from v6 artifacts."""
    summary_df = pd.read_csv(Path(config["spread_summary_csv"]).expanduser())
    bin_df = pd.read_csv(Path(config["spread_bin_effects_csv"]).expanduser())
    promoted_defaults = {"FG3A", "FG3M", "MIN", "FGA", "PTS"}
    gate_mode = config["spread_gate_mode"]
    if gate_mode not in {"strict", "relaxed", "off"}:
        raise ValueError(f"Unsupported spread_gate_mode: {gate_mode}")
    strict_bin_n = int(config["spread_min_non_extreme_bin_n_strict"])
    relaxed_bin_n = int(config["spread_min_non_extreme_bin_n_relaxed"])
    relaxed_require_stability = bool(config["spread_require_directional_stability_relaxed"])
    rows = []
    for target in sorted(summary_df["target"].unique().tolist()):
        candidates = summary_df[(summary_df["target"] == target) & (summary_df["model"] != "baseline")].copy()
        best = candidates.sort_values(
            ["rmse_gain_vs_baseline", "r2_gain_vs_baseline", "model"],
            ascending=[False, False, True],
        ).iloc[0]
        gate_lift = float(best["rmse_gain_vs_baseline"]) > 0.0 and float(best["r2_gain_vs_baseline"]) > 0.0
        bins = bin_df[(bin_df["target"] == target) & (bin_df["model"] == "spread_binned")].copy()
        central_bins = bins[
            bins["spread_bin"].isin(
                ["(-12,-8]", "(-8,-4]", "(-4,-1]", "(-1,1]", "(1,4]", "(4,8]", "(8,12]"]
            )
        ]
        min_required_n = strict_bin_n if gate_mode == "strict" else relaxed_bin_n
        gate_sample = (not central_bins.empty) and int(central_bins["n_rows"].min()) >= min_required_n
        gate_stability_raw = _is_directionally_stable(bins)
        gate_stability = gate_stability_raw if gate_mode == "strict" else (
            gate_stability_raw if relaxed_require_stability else True
        )
        if gate_mode == "off":
            promote = target in promoted_defaults
            gate_sample = True
            gate_stability = True
        else:
            promote = gate_lift and gate_sample and gate_stability and (target in promoted_defaults)
        rows.append(
            {
                "target": target,
                "selected_model": best["model"],
                "rmse_gain_vs_baseline": float(best["rmse_gain_vs_baseline"]),
                "r2_gain_vs_baseline": float(best["r2_gain_vs_baseline"]),
                "spread_gate_mode": gate_mode,
                "spread_min_non_extreme_bin_n_applied": int(min_required_n),
                "gate_positive_lift": int(gate_lift),
                "gate_non_extreme_bin_sample": int(gate_sample),
                "gate_directional_stability_raw": int(gate_stability_raw),
                "gate_directional_stability_applied": int(gate_stability),
                "promote_spread_context": int(promote),
                "final_promotion_decision": "promote" if bool(promote) else "holdout",
            }
        )
    promotion_df = pd.DataFrame(rows).sort_values("target").reset_index(drop=True)
    target_active = {
        row["target"]: bool(row["promote_spread_context"])
        for _, row in promotion_df.iterrows()
    }
    manifest = {
        "feature_names": {
            "spread": "team_point_spread",
            "market_line": "player_consensus_prop_line",
        },
        "spread_gate_mode": gate_mode,
        "active_targets": target_active,
    }
    return promotion_df, manifest


def _fit_uncertainty_model(
    config: dict,
    train_residuals: np.ndarray,
    player_name: str,
):
    uncertainty_model_id = config["uncertainty_model_id"]
    if uncertainty_model_id == "global_variance":
        return fit_global_variance(train_residuals)
    if uncertainty_model_id == "v2_weighted_history_sampler":
        history_df = load_player_history_from_season_logs(
            player_name=player_name,
            history_seasons=config["history_seasons"],
        )
        return fit_v2_weighted_history_sampler(
            history_df=history_df,
            history_n=int(config["history_n"]),
            weighting_mode=config["weighting_mode"],
            decay_alpha=float(config["decay_alpha"]),
        )
    raise ValueError(f"Unsupported uncertainty_model_id: {uncertainty_model_id}")


def main() -> None:
    config_path = MODULE_DIR / "current_config.yaml"
    config = yaml.safe_load(config_path.read_text())
    edge_mode = config["edge_mode"]
    if edge_mode != "raw":
        raise ValueError(f"Unsupported edge_mode for v1: {edge_mode}")

    bundle = build_v1_data_bundle(season=config["season"], player_name=config["player_name"])
    games = bundle.player_games_df.copy().sort_values("date")
    games_with_context = build_player_game_context_features(
        player_games_df=games,
        lines_df=bundle.lines_df,
    ).copy()
    lines = bundle.lines_df.copy()

    if config["enable_spread_context"]:
        promotion_df, feature_manifest = _build_target_feature_promotion_manifest(config=config)
    else:
        promotion_df = pd.DataFrame(
            [
                {
                    "target": "FG3M",
                    "selected_model": "v2_three_input_regression",
                    "rmse_gain_vs_baseline": 0.0,
                    "r2_gain_vs_baseline": 0.0,
                    "spread_gate_mode": "off",
                    "spread_min_non_extreme_bin_n_applied": 0,
                    "gate_positive_lift": 0,
                    "gate_non_extreme_bin_sample": 0,
                    "gate_directional_stability_raw": 0,
                    "gate_directional_stability_applied": 0,
                    "promote_spread_context": 0,
                    "final_promotion_decision": "holdout",
                }
            ]
        )
        feature_manifest = {
            "feature_names": {
                "spread": "team_point_spread",
                "market_line": "player_consensus_prop_line",
            },
            "spread_gate_mode": "off",
            "active_targets": {"FG3M": False},
        }

    spread_context_active_for_fg3m = (
        config["enable_spread_context"]
        and config["enable_spread_context_by_target"]["FG3M"]
        and feature_manifest["active_targets"]["FG3M"]
    )
    fg3m_gate_row = promotion_df[promotion_df["target"] == "FG3M"].iloc[0]

    # v1 visibility requirement: score all played games so section 5 can show
    # the full season coverage (subject to available line contracts).
    train_df = games_with_context.copy()
    score_df = games_with_context.copy()

    model, score_features = _fit_mean_model(
        mean_model_id=config["mean_model_id"],
        train_df=train_df,
        score_df=score_df,
        spread_context_active=spread_context_active_for_fg3m,
    )
    score_df["y_hat"] = model.predict(score_features)
    score_df["run_id"] = "pending"
    score_df["model_id"] = model.model_id
    score_df["model_version"] = model.model_version
    score_df["feature_version"] = model.feature_version

    predictions_df = score_df[
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
            "team_point_spread_abs",
            "team_point_spread_bucket",
        ]
    ].copy()
    predictions_df["spread_context_active_fg3m"] = int(spread_context_active_for_fg3m)

    if model.model_id == "baseline_ols_season_avg_3pm":
        train_predictions = model.predict(train_df)
    elif model.model_id == "v2_three_input_regression":
        train_predictions = model.predict(build_v2_feature_frame(train_df))
    else:
        train_predictions = model.predict(build_v3_feature_frame(train_df))
    train_residuals = train_df["actual_fg3m"].to_numpy(dtype=float) - train_predictions
    uncertainty_model = _fit_uncertainty_model(
        config=config,
        train_residuals=train_residuals,
        player_name=config["player_name"],
    )
    priced_df = price_lines_with_monte_carlo(
        predictions_df=predictions_df,
        lines_df=lines if config["use_all_lines"] else lines[lines["is_consensus"] == 1].copy(),
        uncertainty_model=uncertainty_model,
        n_sims=int(config["n_sims"]),
    )

    run_id = _run_id(config)
    priced_df["run_id"] = run_id
    predictions_df["run_id"] = run_id

    edge_threshold = float(config["edge_threshold"])
    priced_df["max_edge"] = priced_df[["edge_over_raw", "edge_under_raw"]].max(axis=1)
    priced_df["edge_over"] = priced_df["edge_over_raw"]
    priced_df["edge_under"] = priced_df["edge_under_raw"]
    priced_df["best_bet"] = np.where(
        (priced_df["edge_over_raw"] <= 0.0) & (priced_df["edge_under_raw"] <= 0.0),
        "na",
        np.where(priced_df["edge_over_raw"] >= priced_df["edge_under_raw"], "over", "under"),
    )
    priced_df["actual_bet"] = np.where(
        (priced_df["best_bet"] != "na") & (priced_df["max_edge"] >= edge_threshold),
        priced_df["best_bet"],
        "na",
    )
    signals = priced_df[
        priced_df["actual_bet"] != "na"
    ].copy()

    bets_rows = []
    price_view = config["evaluation_price_view"]
    for _, row in signals.iterrows():
        side, odds, p_model = _select_side_and_odds(row=row, price_view=price_view)
        stake = target_profit_stake(odds, target_profit=100.0)
        result = _compute_bet_result(actual_fg3m=float(row["actual_fg3m"]), line=float(row["line"]), side=side)
        pnl = _compute_pnl(result=result, odds=odds, stake=stake)
        bets_rows.append(
            {
                "run_id": run_id,
                "game_id": row["game_id"],
                "player_id": row["player_id"],
                "date": row["date"],
                "line": row["line"],
                "side": side,
                "odds": odds,
                "stake": stake,
                "p_model": p_model,
                "edge": float(max(row["edge_over_raw"], row["edge_under_raw"])),
                "result": result,
                "pnl": pnl,
            }
        )
    bets_df = pd.DataFrame(bets_rows)

    runs_dir = MODULE_DIR / "runs"
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    config_out = run_dir / "config.yaml"
    config_out.write_text(yaml.safe_dump(config, sort_keys=False))

    predictions_out = run_dir / "predictions.parquet"
    priced_df.to_parquet(predictions_out, index=False)

    bets_out = run_dir / "bets.parquet"
    bets_df.to_parquet(bets_out, index=False)

    total_risked = float(bets_df["stake"].sum()) if not bets_df.empty else 0.0
    total_pnl = float(bets_df["pnl"].sum()) if not bets_df.empty else 0.0
    roi = (total_pnl / total_risked) if total_risked > 0 else 0.0
    win_rate = (
        float((bets_df["result"] == "win").mean())
        if not bets_df.empty
        else 0.0
    )
    settled_bets = bets_df[bets_df["result"] != "push"].copy() if not bets_df.empty else bets_df.copy()
    if not settled_bets.empty:
        observed = (settled_bets["result"] == "win").astype(float).to_numpy(dtype=float)
        predicted_prob = settled_bets["p_model"].to_numpy(dtype=float)
        brier_score = float(np.mean((predicted_prob - observed) ** 2))
    else:
        brier_score = 0.0
    rmse = float(np.sqrt(np.mean((predictions_df["y_hat"] - predictions_df["actual_fg3m"]) ** 2)))
    signal_rate = float(len(bets_df) / len(priced_df)) if len(priced_df) > 0 else 0.0

    consensus_rows = priced_df[priced_df["is_consensus"] == 1].copy()
    consensus_residual = float((consensus_rows["y_hat"] - consensus_rows["line"]).mean()) if not consensus_rows.empty else 0.0
    consensus_residual_std = float((consensus_rows["y_hat"] - consensus_rows["line"]).std()) if not consensus_rows.empty else 0.0

    summary = {
        "run_id": run_id,
        "player_name": config["player_name"],
        "season": config["season"],
        "n_games_played": int(games["date"].nunique()),
        "n_games_with_priced_lines": int(priced_df["date"].nunique()),
        "n_predictions": int(len(priced_df)),
        "n_bets": int(len(bets_df)),
        "rmse": rmse,
        "win_rate": win_rate,
        "roi": roi,
        "total_pnl": total_pnl,
        "total_risked": total_risked,
        "signal_rate": signal_rate,
        "brier_score": brier_score,
        "consensus_residual_mean": consensus_residual,
        "consensus_residual_std": consensus_residual_std,
        "uncertainty_model_id": uncertainty_model.model_id,
        "edge_mode": edge_mode,
        "spread_context_enabled": int(config["enable_spread_context"]),
        "spread_gate_mode": feature_manifest["spread_gate_mode"],
        "spread_context_active_fg3m": int(spread_context_active_for_fg3m),
        "fg3m_gate_positive_lift": int(fg3m_gate_row["gate_positive_lift"]),
        "fg3m_gate_non_extreme_bin_sample": int(fg3m_gate_row["gate_non_extreme_bin_sample"]),
        "fg3m_gate_directional_stability_raw": int(fg3m_gate_row["gate_directional_stability_raw"]),
        "fg3m_gate_directional_stability_applied": int(
            fg3m_gate_row["gate_directional_stability_applied"]
        ),
        "fg3m_promote_spread_context": int(fg3m_gate_row["promote_spread_context"]),
    }
    if hasattr(uncertainty_model, "sigma"):
        summary["uncertainty_sigma"] = float(uncertainty_model.sigma)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    repo_root = ROOT_DIR.parent.parent
    manifest = {
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(repo_root),
        "data_version": f"s3-season-{config['season']}",
        "files": {
            "predictions.parquet_sha256": hashlib.sha256(predictions_out.read_bytes()).hexdigest(),
            "bets.parquet_sha256": hashlib.sha256(bets_out.read_bytes()).hexdigest(),
        },
        "env": {"AWS_ACCESS_KEY_ID_present": ("AWS_ACCESS_KEY_ID" in os.environ)},
        "feature_manifest": feature_manifest,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    promotion_df.to_csv(run_dir / "target_feature_promotion.csv", index=False)

    monitoring_rows = (
        priced_df.groupby("team_point_spread_bucket", as_index=False)
        .agg(
            n_rows=("line", "count"),
            mean_edge_over=("edge_over_raw", "mean"),
            mean_edge_under=("edge_under_raw", "mean"),
            mean_actual_fg3m=("actual_fg3m", "mean"),
            mean_y_hat=("y_hat", "mean"),
        )
        .sort_values("team_point_spread_bucket")
    )
    monitoring_rows["calibration_delta_fg3m"] = (
        monitoring_rows["mean_y_hat"] - monitoring_rows["mean_actual_fg3m"]
    )
    monitoring_rows.to_csv(run_dir / "spread_context_monitoring.csv", index=False)
    print(f"Created run: {run_id}")
    print(f"Artifacts: {run_dir}")


if __name__ == "__main__":
    main()

