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
from odds import american_to_implied_prob
from odds import target_profit_stake
from baseline import fit_baseline_model
from global_variance import fit_global_variance
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


def main() -> None:
    config_path = MODULE_DIR / "current_config.yaml"
    config = yaml.safe_load(config_path.read_text())
    edge_mode = config.get("edge_mode", "raw")
    if edge_mode != "raw":
        raise ValueError(f"Unsupported edge_mode for v1: {edge_mode}")

    bundle = build_v1_data_bundle(season=config["season"], player_name=config["player_name"])
    games = bundle.player_games_df.copy().sort_values("date")
    lines = bundle.lines_df.copy()

    # v1 visibility requirement: score all played games so section 5 can show
    # the full season coverage (subject to available line contracts).
    train_df = games.copy()
    score_df = games.copy()

    model = fit_baseline_model(train_df)
    score_df["y_hat"] = model.predict(score_df)
    score_df["run_id"] = "pending"
    score_df["model_id"] = model.model_id
    score_df["model_version"] = model.model_version
    score_df["feature_version"] = model.feature_version

    predictions_df = score_df[
        ["run_id", "game_id", "player_id", "date", "y_hat", "model_id", "model_version", "feature_version", "actual_fg3m"]
    ].copy()

    train_residuals = train_df["actual_fg3m"].to_numpy(dtype=float) - model.predict(train_df)
    uncertainty_model = fit_global_variance(train_residuals)
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
        "consensus_residual_mean": consensus_residual,
        "consensus_residual_std": consensus_residual_std,
        "uncertainty_sigma": float(uncertainty_model.sigma),
        "edge_mode": edge_mode,
    }
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
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Created run: {run_id}")
    print(f"Artifacts: {run_dir}")


if __name__ == "__main__":
    main()

