"""
Run reproducible v3 vs v4 multi-player backtest A/B and write decision artifacts.

Context:
- This script promotes universe-level decisioning over single-player anecdotes.
- For each requested player, it runs two backtests with identical config except
  `mean_model_id`:
  - `v3_three_input_regression`
  - `v4_market_spread_regression`
- It writes per-player and aggregate CSV outputs plus a machine-readable
  promotion decision JSON.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from datetime import timezone
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any

import numpy as np
import pandas as pd
import yaml


MODULE_DIR = Path(__file__).resolve().parent
RUNS_DIR = MODULE_DIR / "runs"
CONFIG_PATH = MODULE_DIR / "current_config.yaml"
PER_PLAYER_OUT = Path("~/Downloads/tmp/v3_v4_ab_multiplayer_per_player.csv").expanduser()
AGG_OUT = Path("~/Downloads/tmp/v3_v4_ab_multiplayer_aggregate.csv").expanduser()
DECISION_OUT = Path("~/Downloads/tmp/v3_v4_promotion_decision.json").expanduser()


def parse_args() -> argparse.Namespace:
    """Parse CLI args for multi-player v3/v4 A-B runs."""
    parser = argparse.ArgumentParser(
        description=(
            "Run v3/v4 multiplayer A-B using one season and a player cohort, "
            "then write per-player, aggregate, and promotion-decision artifacts."
        )
    )
    parser.add_argument("--season", type=str, required=True, help="Season string, e.g. 2025-26.")
    parser.add_argument(
        "--players-csv",
        type=str,
        default="",
        help="CSV with required column `player_name`.",
    )
    parser.add_argument(
        "--players-list",
        type=str,
        default="",
        help="Comma-separated player names (mutually exclusive with --players-csv).",
    )
    parser.add_argument("--max-players", type=int, default=0, help="Optional cap after loading player cohort.")
    parser.add_argument(
        "--spread-gate-mode",
        type=str,
        default="",
        choices=["", "strict", "relaxed", "off"],
        help="Optional override for spread gate mode in both v3 and v4 runs.",
    )
    parser.add_argument("--resume", action="store_true", help="Skip players already present as completed rows.")
    parser.add_argument(
        "--config-path",
        type=str,
        default=str(CONFIG_PATH),
        help="Base backtest config path used as template.",
    )
    parser.add_argument(
        "--output-per-player-csv",
        type=str,
        default=str(PER_PLAYER_OUT),
        help="Per-player output CSV path.",
    )
    parser.add_argument(
        "--output-aggregate-csv",
        type=str,
        default=str(AGG_OUT),
        help="Aggregate output CSV path.",
    )
    parser.add_argument(
        "--decision-json",
        type=str,
        default=str(DECISION_OUT),
        help="Promotion decision JSON output path.",
    )
    parser.add_argument(
        "--apply-config-update-if-promoted",
        action="store_true",
        help="If decision promotes v4, update config mean_model_id to v4.",
    )
    return parser.parse_args()


def _slugify(name: str) -> str:
    chars = []
    for ch in name.lower():
        if ch.isalnum():
            chars.append(ch)
        else:
            chars.append("_")
    return "".join(chars).strip("_")


def _load_players(args: argparse.Namespace) -> list[str]:
    """Load player cohort from CSV or inline list."""
    has_csv = args.players_csv.strip() != ""
    has_list = args.players_list.strip() != ""
    if has_csv == has_list:
        raise ValueError("Provide exactly one of --players-csv or --players-list")
    if has_csv:
        players_path = Path(args.players_csv).expanduser()
        if not players_path.exists():
            raise FileNotFoundError(f"Missing players CSV: {players_path}")
        df = pd.read_csv(players_path)
        if "player_name" not in df.columns:
            raise ValueError(f"Missing required column in players CSV: player_name ({players_path})")
        players = [str(x).strip() for x in df["player_name"].tolist() if str(x).strip() != ""]
    else:
        players = [p.strip() for p in args.players_list.split(",") if p.strip() != ""]
    if not players:
        raise ValueError("Loaded zero players from provided input")
    unique = list(dict.fromkeys(players))
    if args.max_players > 0:
        unique = unique[: int(args.max_players)]
    return unique


def _run_with_model(
    base_config: dict[str, Any],
    config_path: Path,
    player_name: str,
    season: str,
    model_id: str,
    group_id: str,
    spread_gate_mode: str,
) -> Path:
    """Execute one backtest run for one player/model and return created run dir."""
    config = dict(base_config)
    config["player_name"] = player_name
    config["season"] = season
    config["mean_model_id"] = model_id
    if spread_gate_mode != "":
        config["spread_gate_mode"] = spread_gate_mode
    model_tag = "v3" if model_id == "v3_three_input_regression" else "v4"
    config["run_suffix"] = (
        f"{base_config['run_suffix']}_abmp_{group_id}_{_slugify(player_name)}_{model_tag}"
    )
    before = {p.name for p in RUNS_DIR.iterdir() if p.is_dir()}
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    subprocess.run(["python", str(MODULE_DIR / "run_backtest.py")], check=True)
    after = {p.name for p in RUNS_DIR.iterdir() if p.is_dir()}
    created = sorted(list(after - before))
    if len(created) != 1:
        raise ValueError(f"Expected exactly one created run dir, found: {created}")
    return RUNS_DIR / created[0]


def _summary(run_dir: Path) -> dict[str, Any]:
    """Load run summary json."""
    return json.loads((run_dir / "summary.json").read_text())


def _pair_row(
    group_id: str,
    pair_id: str,
    player_name: str,
    season: str,
    spread_gate_mode: str,
    summary_v3: dict[str, Any],
    summary_v4: dict[str, Any],
) -> dict[str, Any]:
    """Build one per-player pair row with v3/v4 metrics and deltas."""
    row = {
        "group_id": group_id,
        "pair_id": pair_id,
        "player_name": player_name,
        "season": season,
        "spread_gate_mode": spread_gate_mode,
        "run_id_v3": summary_v3["run_id"],
        "run_id_v4": summary_v4["run_id"],
        "rmse_v3": float(summary_v3["rmse"]),
        "rmse_v4": float(summary_v4["rmse"]),
        "win_rate_v3": float(summary_v3["win_rate"]),
        "win_rate_v4": float(summary_v4["win_rate"]),
        "roi_v3": float(summary_v3["roi"]),
        "roi_v4": float(summary_v4["roi"]),
        "n_bets_v3": int(summary_v3["n_bets"]),
        "n_bets_v4": int(summary_v4["n_bets"]),
        "signal_rate_v3": float(summary_v3["signal_rate"]),
        "signal_rate_v4": float(summary_v4["signal_rate"]),
        "spread_context_active_fg3m_v3": int(summary_v3["spread_context_active_fg3m"]),
        "spread_context_active_fg3m_v4": int(summary_v4["spread_context_active_fg3m"]),
    }
    row["delta_rmse_v4_minus_v3"] = row["rmse_v4"] - row["rmse_v3"]
    row["delta_win_rate_v4_minus_v3"] = row["win_rate_v4"] - row["win_rate_v3"]
    row["delta_roi_v4_minus_v3"] = row["roi_v4"] - row["roi_v3"]
    row["delta_n_bets_v4_minus_v3"] = row["n_bets_v4"] - row["n_bets_v3"]
    row["delta_signal_rate_v4_minus_v3"] = row["signal_rate_v4"] - row["signal_rate_v3"]
    row["status"] = "completed"
    row["error_message"] = ""
    return row


def _error_row(
    group_id: str,
    pair_id: str,
    player_name: str,
    season: str,
    spread_gate_mode: str,
    error_message: str,
) -> dict[str, Any]:
    """Build one per-player error row when either run fails."""
    return {
        "group_id": group_id,
        "pair_id": pair_id,
        "player_name": player_name,
        "season": season,
        "spread_gate_mode": spread_gate_mode,
        "run_id_v3": "",
        "run_id_v4": "",
        "rmse_v3": np.nan,
        "rmse_v4": np.nan,
        "win_rate_v3": np.nan,
        "win_rate_v4": np.nan,
        "roi_v3": np.nan,
        "roi_v4": np.nan,
        "n_bets_v3": np.nan,
        "n_bets_v4": np.nan,
        "signal_rate_v3": np.nan,
        "signal_rate_v4": np.nan,
        "spread_context_active_fg3m_v3": np.nan,
        "spread_context_active_fg3m_v4": np.nan,
        "delta_rmse_v4_minus_v3": np.nan,
        "delta_win_rate_v4_minus_v3": np.nan,
        "delta_roi_v4_minus_v3": np.nan,
        "delta_n_bets_v4_minus_v3": np.nan,
        "delta_signal_rate_v4_minus_v3": np.nan,
        "status": "error",
        "error_message": error_message,
    }


def _aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Build one-row aggregate metrics frame from completed per-player rows."""
    done = df[df["status"] == "completed"].copy()
    if done.empty:
        return pd.DataFrame(
            [
                {
                    "n_players_total": int(len(df)),
                    "n_players_completed": 0,
                    "n_players_error": int((df["status"] == "error").sum()),
                    "delta_rmse_v4_minus_v3": np.nan,
                    "delta_roi_v4_minus_v3": np.nan,
                    "delta_win_rate_v4_minus_v3": np.nan,
                    "delta_signal_rate_v4_minus_v3": np.nan,
                    "players_non_negative_rmse_or_roi": 0,
                    "player_non_negative_share": np.nan,
                }
            ]
        )
    non_negative = (
        (done["delta_rmse_v4_minus_v3"] <= 0.0) | (done["delta_roi_v4_minus_v3"] >= 0.0)
    )
    return pd.DataFrame(
        [
            {
                "n_players_total": int(len(df)),
                "n_players_completed": int(len(done)),
                "n_players_error": int((df["status"] == "error").sum()),
                "delta_rmse_v4_minus_v3": float(done["delta_rmse_v4_minus_v3"].mean()),
                "delta_roi_v4_minus_v3": float(done["delta_roi_v4_minus_v3"].mean()),
                "delta_win_rate_v4_minus_v3": float(done["delta_win_rate_v4_minus_v3"].mean()),
                "delta_signal_rate_v4_minus_v3": float(done["delta_signal_rate_v4_minus_v3"].mean()),
                "players_non_negative_rmse_or_roi": int(non_negative.sum()),
                "player_non_negative_share": float(non_negative.mean()),
            }
        ]
    )


def _decision(aggregate_row: pd.Series, per_player_df: pd.DataFrame) -> dict[str, Any]:
    """Apply promotion policy and return machine-readable decision payload."""
    completed = per_player_df[per_player_df["status"] == "completed"].copy()
    if completed.empty:
        return {
            "decision": "keep_v3_default",
            "evidence_summary": "no completed players",
            "caveats": ["all players failed or were skipped"],
            "rollback_model_id": "v3_three_input_regression",
        }
    delta_rmse = float(aggregate_row["delta_rmse_v4_minus_v3"])
    delta_roi = float(aggregate_row["delta_roi_v4_minus_v3"])
    delta_win = float(aggregate_row["delta_win_rate_v4_minus_v3"])
    delta_signal = float(aggregate_row["delta_signal_rate_v4_minus_v3"])
    non_negative_share = float(aggregate_row["player_non_negative_share"])
    no_major_drop = (delta_win >= -0.03) and (delta_signal >= -0.03)
    decision = (
        "promote_v4_default"
        if (delta_rmse <= 0.0 and delta_roi >= 0.0 and no_major_drop and non_negative_share > 0.5)
        else "keep_v3_default"
    )
    caveats: list[str] = []
    seasons = sorted(completed["season"].unique().tolist())
    if len(seasons) <= 1:
        caveats.append("single-season coverage")
    if int((per_player_df["status"] == "error").sum()) > 0:
        caveats.append("some players failed during A/B run")
    return {
        "decision": decision,
        "evidence_summary": {
            "delta_rmse_v4_minus_v3": delta_rmse,
            "delta_roi_v4_minus_v3": delta_roi,
            "delta_win_rate_v4_minus_v3": delta_win,
            "delta_signal_rate_v4_minus_v3": delta_signal,
            "player_non_negative_share": non_negative_share,
            "n_players_completed": int(len(completed)),
        },
        "caveats": caveats,
        "rollback_model_id": "v3_three_input_regression",
    }


def main() -> None:
    """Run full multiplayer v3/v4 A-B and write per-player/aggregate/decision outputs."""
    args = parse_args()
    players = _load_players(args)
    config_path = Path(args.config_path).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config file: {config_path}")
    base_text = config_path.read_text()
    base_config = yaml.safe_load(base_text)
    spread_gate_mode = args.spread_gate_mode if args.spread_gate_mode != "" else base_config["spread_gate_mode"]
    group_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    per_player_out = Path(args.output_per_player_csv).expanduser()
    aggregate_out = Path(args.output_aggregate_csv).expanduser()
    decision_out = Path(args.decision_json).expanduser()
    per_player_out.parent.mkdir(parents=True, exist_ok=True)
    aggregate_out.parent.mkdir(parents=True, exist_ok=True)
    decision_out.parent.mkdir(parents=True, exist_ok=True)

    existing = pd.DataFrame()
    if args.resume and per_player_out.exists():
        existing = pd.read_csv(per_player_out)
    completed_pairs: set[tuple[str, str]] = set()
    if not existing.empty and "status" in existing.columns:
        done = existing[existing["status"] == "completed"].copy()
        completed_pairs = {(str(r["player_name"]), str(r["season"])) for _, r in done.iterrows()}

    backup_path = config_path.with_suffix(".yaml.ab_multiplayer_backup")
    shutil.copy2(config_path, backup_path)
    new_rows: list[dict[str, Any]] = []
    try:
        for player_name in players:
            pair_key = (player_name, args.season)
            pair_id = f"{group_id}_{_slugify(player_name)}_{args.season}"
            if pair_key in completed_pairs:
                print(f"skip_resume player={player_name} season={args.season}")
                continue
            try:
                run_dir_v3 = _run_with_model(
                    base_config=base_config,
                    config_path=config_path,
                    player_name=player_name,
                    season=args.season,
                    model_id="v3_three_input_regression",
                    group_id=group_id,
                    spread_gate_mode=spread_gate_mode,
                )
                run_dir_v4 = _run_with_model(
                    base_config=base_config,
                    config_path=config_path,
                    player_name=player_name,
                    season=args.season,
                    model_id="v4_market_spread_regression",
                    group_id=group_id,
                    spread_gate_mode=spread_gate_mode,
                )
                row = _pair_row(
                    group_id=group_id,
                    pair_id=pair_id,
                    player_name=player_name,
                    season=args.season,
                    spread_gate_mode=spread_gate_mode,
                    summary_v3=_summary(run_dir_v3),
                    summary_v4=_summary(run_dir_v4),
                )
            except Exception as exc:
                row = _error_row(
                    group_id=group_id,
                    pair_id=pair_id,
                    player_name=player_name,
                    season=args.season,
                    spread_gate_mode=spread_gate_mode,
                    error_message=str(exc),
                )
            new_rows.append(row)
            incremental = pd.concat([existing, pd.DataFrame(new_rows)], ignore_index=True)
            incremental.to_csv(per_player_out, index=False)
    finally:
        config_path.write_text(base_text)
        if backup_path.exists():
            backup_path.unlink()

    full = pd.concat([existing, pd.DataFrame(new_rows)], ignore_index=True)
    full.to_csv(per_player_out, index=False)
    aggregate = _aggregate(full)
    aggregate.to_csv(aggregate_out, index=False)
    decision_payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "group_id": group_id,
        "spread_gate_mode": spread_gate_mode,
        "season": args.season,
    }
    decision_payload.update(_decision(aggregate.iloc[0], full))
    decision_out.write_text(json.dumps(decision_payload, indent=2))

    if args.apply_config_update_if_promoted and decision_payload["decision"] == "promote_v4_default":
        config = yaml.safe_load(base_text)
        config["mean_model_id"] = "v4_market_spread_regression"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False))

    print(f"group_id={group_id}")
    print(f"players_total={len(players)}")
    print(f"per_player_csv={per_player_out}")
    print(f"aggregate_csv={aggregate_out}")
    print(f"decision_json={decision_out}")
    print(aggregate.to_string(index=False))
    print(json.dumps(decision_payload, indent=2))


if __name__ == "__main__":
    main()
